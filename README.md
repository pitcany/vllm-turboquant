# TurboQuant: KV Cache Compression for LLM Inference

Implementation of TurboQuant KV cache compression (ICLR 2026, arXiv:2504.19874) with vLLM integration.

> ## ⚠️ Integration is currently non-functional on default vLLM
>
> End-to-end verification on real hardware (vLLM 0.17.1 + RTX 5090 + Qwen2.5-0.5B-Instruct, 2026-04-29) revealed that **the vLLM integration does not actually do anything on a default vLLM configuration**. Three independent issues:
>
> 1. **`torch.compile` bypasses our hooks.** vLLM 0.17.x compiles the attention graph by default (`compilation_config.mode = VLLM_COMPILE`). KV writes go through a compiled `vllm::unified_kv_cache_update` op that does not call back into the Python `Impl.do_kv_cache_update` method we monkey-patch. Result: `mode="capture_only"` captures roughly 2 tokens out of 300+ during a real generation. Setting `enforce_eager=True` restores per-token capture but disables CUDA-graph speedups.
>
> 2. **Prefix caching evades capture.** Even in eager mode, vLLM's prefix cache (also default-on) reuses uncompressed KV blocks across requests, so TQ never sees the prefill on the second pass. The integration has no path that captures from already-cached blocks.
>
> 3. **`mode="hybrid"` ignores the paged KV cache.** When the hybrid branch fires, it computes attention from `state.store.get_flat_cache()` and `state.engine.ring.peek()` — i.e., only what TQ captured — and ignores the paged `kv_cache` tensor that holds the actual prompt. Combined with (1)+(2), hybrid decode produces degenerate output (e.g. `"......\n\n.."` repetition) because the model is attending only to the last few generated tokens.
>
> **What this means for the benchmark numbers below**: they were measured against configurations where the hooks did not fire as intended, so they reflect baseline vLLM throughput, not TurboQuant throughput. They should be treated as not-yet-reproduced.
>
> **What works today**: the numerical kernels (codebooks, rotation, quantizer, store) are sound and unit-tested on CPU and CUDA. The public API surface (`turboquant.vllm.enable_turboquant`, `free_kv_cache`, `get_stats`) installs cleanly and emits useful errors. `quickstart.py NO_VLLM=1` runs the quantizer math standalone. None of the integration's correctness or performance claims for end-to-end generation are currently verified.
>
> **What it would take to fix**: register the capture as a torch op so it survives `torch.compile`, rewrite hybrid attention to combine paged `kv_cache` + TQ store rather than only TQ store, and either capture on prefix-cache hit or require prefix caching off. Realistically 1–2 weeks of focused work; tracked as the next major piece of engineering.
>
> **Plan**: see [`docs/plan-path-b.md`](docs/plan-path-b.md). That document is the contract for the next phase of work — working principles, sprint plan, acceptance criteria per sprint, and definitions-of-done for each finding above.
>
> ### Status update — Sprints 1–3 complete (2026-04-30)
>
> - **F1 closed** by [`304ba1f`](https://github.com/pitcany/vllm-turboquant/commit/304ba1f) (Sprint 1 / S1.3 / 1C — post-`execute_model` paged-cache reader survives FULL CUDAGraph). Default-compile + `enable_prefix_caching=False` now captures all prefill + all decode K/Vs (`docs/traces/s1_compiled.log`).
> - **F2 closed** by [`4c902f1`](https://github.com/pitcany/vllm-turboquant/commit/4c902f1) (Sprint 2 / S2.1). `enable_turboquant` now raises a typed error when prefix caching is on with `mode != "off"`. Capture-on-cache-hit is filed as a separate ticket.
> - **F3 half-closed.** Combiner rewrite landed in [`0bf9510`](https://github.com/pitcany/vllm-turboquant/commit/0bf9510) (Sprint 3 / S3.2): hybrid attention now folds three KV segments (`kv_paged ∪ kv_tq ∪ kv_ring`) via streaming online softmax, the degenerate `quant, quant, quant` output from issue (3) above is gone, and 6 unit tests prove the math is identical to a single softmax over the concatenation. **But:** the empirical top-1-agreement bar on `Llama-3.2-1B-Instruct` is *not* met at any bit budget tried (3-bit/2-bit → 7.69%, 3-bit/4-bit → 0.39%, 4-bit/4-bit → 2.73% — see [`docs/integration-state.md`](docs/integration-state.md) § "S3.3 follow-up"). The plan's §5 second-bullet stop-loss is engaged.
> - **Pivot for Sprint 4.** Capture-only + `free_kv_cache` for memory-only wins (plan §5 third bullet). The "Verified configuration" section that replaces this ⚠️ notice will advertise `mode="capture_only"`, with `mode="hybrid"` retained as a research-mode setting and a fused-Triton-kernel target for Sprint 5.
>
> The rest of this README is preserved for reference. Adjust your expectations accordingly.

## Benchmark Results (NOT CURRENTLY REPRODUCIBLE — see notice above)

### RTX 5090 (32GB) -- Qwen3.5-27B-AWQ (dense, 4-bit weights, TP=1)

**Setup**: Single RTX 5090, vLLM 0.18.0, `gpu_memory_utilization=0.90`, 16 full-attention layers out of 64 total (rest are linear-attention).

| Metric | Baseline (bf16 KV) | TurboQuant (3b key / 2b val) |
|--------|-------------------|------------------------------|
| Prefill tok/s (30k ctx) | 1,804 | 1,907 (+5.7%) |
| Decode tok/s (30k ctx) | 1.264 | 1.303 (+3.1%) |
| KV cache freed | -- | **30.0 GB** (across 4 GPUs) |
| Max token capacity | 457,072 | **914,144** (2.0x) |
| Peak activation memory | 644.6 MB | 599.2 MB (-7.0%) |

### 8x RTX 3090 (24GB each) -- Qwen3.5-35B-A3B MoE (pruned, 205 experts, TP=8)

**Setup**: 8x RTX 3090, vLLM 0.18.0, `gpu_memory_utilization=0.92`, AMD EPYC 7443P 24-Core, 504GB RAM. Model has 10 full-attention layers + 30 linear-attention layers (40 total). TQ compresses only the 10 full-attention layers.

#### Throughput & Latency (Baseline, bf16 KV)

| Context | Prefill tok/s | Decode tok/s | TTFT (s) | Needles Found |
|--------:|--------------:|-------------:|---------:|--------------:|
| 1,000 | 7,127 | 129.7 | 0.14 | 4/5 |
| 4,000 | 8,887 | 131.5 | 0.45 | 4/5 |
| 8,000 | 9,684 | 131.1 | 0.83 | 4/5 |
| 16,000 | 9,933 | 133.0 | 1.61 | 4/5 |
| 32,000 | 9,761 | 116.7 | 3.28 | 4/5 |
| 64,000 | 8,843 | 122.6 | 7.24 | 4/5 |
| 100,000 | 8,479 | 106.8 | 11.79 | 4/5 |
| 131,000 | 8,238 | 98.3 | 15.90 | 4/5 |

- **Prefill** saturates around 10k tok/s, degrades gently to 8.2k at 131k context.
- **Decode** drops from 133 to 98 tok/s at 131k (KV readback cost from full-attention layers).
- **TTFT** scales linearly with context length (purely compute-bound).
- **Needles** 4/5 found consistently at ALL context lengths -- the model reformats one answer.

#### VRAM Breakdown (per GPU at 131k context)

| Component | Size |
|-----------|-----:|
| Total VRAM | 24,576 MB |
| Reserved (0.92 util) | 22,610 MB |
| Model weights | ~6,750 MB |
| KV cache pool | **9,035 MB** |
| -- full_attention (10 layers) | 3,614 MB |
| -- linear_attention (30 layers) | 5,421 MB |
| CUDA overhead + graphs | ~6,825 MB |

#### Baseline vs TurboQuant KV Cache

| Context | Baseline KV/GPU | TQ KV/GPU | Savings/GPU | Savings % |
|--------:|----------------:|----------:|------------:|----------:|
| 8,000 | 55.7 MB | 38.5 MB | **17.2 MB** | 30.9% |
| 32,000 | 191.5 MB | 132.3 MB | **59.3 MB** | 30.9% |
| 64,000 | 374.3 MB | 258.5 MB | **115.8 MB** | 30.9% |
| 100,000 | 578.1 MB | 399.2 MB | **178.8 MB** | 30.9% |
| 131,000 | 755.7 MB | 521.9 MB | **233.8 MB** | 30.9% |

- Savings are **30.9% of total KV** because TQ only compresses the 10 full-attention layers (40% of KV).
- The 30 linear-attention layers (60% of KV) are **not compressible** by TQ.
- On a **pure dense transformer**, savings would be **77%** (4.4x compression).

#### Context Extension

| | Tokens | Multiplier |
|---|-------:|:----------:|
| Baseline capacity | 1,411,680 | 1.0x |
| With TQ | 2,043,808 | **1.45x** |

Alternatively, freed VRAM supports **3 additional concurrent 131k-context requests**.

#### Coherence & Quality

| Test | Result |
|------|--------|
| Single needle (512-131k tokens) | **PASS** at all lengths |
| 5-needle at near-max context | **5/5** retrieved |
| 3-needle multi-fact coherence | **3/3** retrieved |
| Golden ratio completion (all lengths) | **PASS**, perplexity 1.05-1.35 |
| Math reasoning at max context | Coherent (model math error from pruning, not context) |

#### TQ Quantization Quality (head_dim=256, measured on GPU)

| Component | cos_sim | Notes |
|-----------|--------:|-------|
| TQ key compression (3-bit) | **1.000000** | Near-lossless |
| TQ key compression (4-bit) | **1.000000** | Near-lossless |
| Value quantization (2-bit) | 0.940 | Bottleneck for quality |
| Value quantization (4-bit) | 0.997 | Recommended for quality-sensitive use |
| Combined (3b key + 2b val) | 0.940 | Value quant dominates degradation |

#### GPU Utilization During Inference

| Context | Peak VRAM/GPU | GPU Util | CPU % | Power |
|--------:|--------------:|---------:|------:|------:|
| 1,000 | 22,284 MB | 0% idle | 0.2% | 132W |
| 32,000 | 22,286 MB | 57% peak | 0.4% | 142W |
| 131,000 | 22,306 MB | 0% idle | 0.4% | 130W |

- VRAM is **essentially flat** -- KV cache at 131k is only 190 MB/GPU (0.8% of VRAM).
- No CPU offloading. No KV offloading. Everything fits in VRAM.
- GPU interconnect is **PCIe** (no NVLink) -- NODE topology between all GPUs.

### Paper Validation (Theorems 1-3)

9 tests validating the paper's theoretical claims:

| Claim | Verdict | Details |
|-------|---------|---------|
| MSE distortion bounds (Thm 1) | **PASS** | Within bounds for unit-norm vectors |
| Codebook MSE matches Table 1 | **PASS** | Lloyd-Max codebook is faithful |
| Unbiasedness (Thm 2) | **PASS** | Relative bias < 0.1% |
| Distortion 1/4^b scaling (Thm 3) | **PASS** | 2-bit=0.70x, 3-bit=0.82x, 4-bit=0.97x of bound |
| Recall@8 (3-bit, N=4096) | **0.55** | Paper threshold met (>=0.40) |
| Rank correlation (N=2048) | **PASS** | Spearman rho > 0.85 |
| Needle retrieval | **PASS** | Works at all SNR levels |
| Compression ratio | **4.41x** | At head_dim=256 on full-attention layers |

### Adversarial Audit

Honest assessment of claims (audit script not yet included in this repo):

| Claim | Verdict |
|-------|---------|
| "5.1x compression" | **Misleading** -- doesn't count Pi/S matrices or ring buffer. Honest: ~4.6x at 4k tokens, ~5x at 32k+ |
| "Needle-in-haystack passes" | **True but trivial** -- query=key test is too easy. Real LLM queries are not copies of keys |
| "Recall@8 >= 0.40" | **Low bar** -- 3-bit recall@1 is only 38%. BUT dominant attention tokens are always preserved |
| "Hybrid decode saves memory" | **Storage yes, compute no** -- dequantizes all history to float32 per decode step |
| "Distortion follows 1/4^b" | **True** -- initial audit was wrong (unnormalized vectors). Unit-norm: within bound |
| "30k TQ is faster" | **Within noise** -- N=1 run, total wall time TQ is actually slower |
| "200k context works" | **Unverified** -- didn't crash, but output quality never checked |
| "2x context on dense model" | **True** -- measured 30 GB freed on Qwen3.5-27B with 4x RTX 3090 |

## How It Works

TurboQuant compresses KV cache entries using:
1. **Random orthogonal rotation** to spread information across dimensions
2. **Lloyd-Max optimal scalar quantization** (b-1 bits) on Beta-distributed rotated values
3. **QJL projection** for residual sign bits (1 bit per dimension)
4. **Group quantization** for values (2-bit or 4-bit, per-group scales and zeros)
5. **Bit-packing**: 4 values per byte (2-bit) or 2 per byte (4-bit)

The combined estimator is **unbiased**: E[estimated inner product] = true inner product.

## Architecture

```
turboquant/
  codebook.py          # Lloyd-Max optimal scalar quantizer for Beta distribution
  codebooks/           # Pre-generated codebook files (d=64/128/576, bits 1-4)
  rotation.py          # Random orthogonal rotation + QJL projection matrices
  quantizer.py         # TurboQuantMSE + TurboQuantProd (Algorithms 1 & 2)
  kv_cache.py          # KV cache manager with value bit-packing
  capture.py           # Modular KV capture hooks for attention layers
  store.py             # Compressed KV store (quantize + append + flat cache)
  score.py             # Attention scoring from compressed keys
  integration/vllm.py  # vLLM adapter (monkey-patch, free_kv_cache, hybrid decode)
  vllm_attn_backend.py # Thin shim delegating to integration/vllm.py (deprecated)

proof.py               # 2-process A/B benchmark (baseline vs TQ) -- needs local model
benchmark.py           # Multi-model throughput/quality harness -- needs local model
```

## Usage

```bash
pip install -e .[vllm]

# Sanity check the package imports
python -c "import turboquant; print(turboquant.__version__)"

# CPU-only smoke test (just the quantizer math, no vLLM):
NO_VLLM=1 python quickstart.py

# Single-GPU end-to-end smoke test on a small public model:
python quickstart.py
# -> downloads Qwen/Qwen2.5-0.5B-Instruct, runs baseline vs TQ in one process

# Multi-GPU:
CUDA_VISIBLE_DEVICES=0,1 TP=2 python quickstart.py

# Pick another model:
MODEL=meta-llama/Llama-3.2-1B-Instruct python quickstart.py
```

### Programmatic use

```python
# Import turboquant.vllm BEFORE vllm.LLM. Importing it sets
# VLLM_ALLOW_INSECURE_SERIALIZATION=1 (if not already set), which vLLM 0.17.x
# requires so collective_rpc can transport our installer closure to worker
# subprocesses. The env var has to be present when the engine subprocess is
# forked, i.e. *before* LLM(...).
import turboquant.vllm
from vllm import LLM, SamplingParams
from turboquant.vllm import enable_turboquant, free_kv_cache, get_stats

llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", max_model_len=4096)

info = enable_turboquant(
    llm,
    key_bits=3,
    value_bits=2,        # bump to 4 for quality-sensitive workloads
    buffer_size=128,
    mode="capture_only", # or "hybrid"
)
# info -> {'workers': N, 'hooks_per_worker': [..], 'mode': 'capture_only', ...}

out = llm.generate(["Hello, "], SamplingParams(max_tokens=32))
print(out[0].outputs[0].text)

# Optional: drop paged KV-cache tensors for TQ-hooked layers after prefill.
freed_bytes = free_kv_cache(llm)

# Inspect TQ state:
print(get_stats(llm))
```

`turboquant.vllm.enable_turboquant` walks the vLLM v1 engine (which is a moving
target) and runs hook installation on every worker via `collective_rpc`.
If a future vLLM release relocates `engine_core.engine_core.model_executor`,
this function raises `TurboQuantVLLMError` with a clear message instead of
silently doing nothing.

`proof.py` and `benchmark.py` are kept as multi-process harnesses for
side-by-side baseline/TQ comparison on a model on disk; they are *not* the
recommended user entry point.

## Reproducing benchmarks

The benchmark numbers in this README were collected against internal model checkpoints
(referred to as `Qwen3.5-27B` and `Qwen3.5-35B-A3B`) that are **not** public Hugging
Face IDs. To rerun on your own model:

1. Edit `benchmark.py` `MODELS` dict to point `path` at a local HF model directory.
2. Adjust `tp`, `gpu_mem`, `max_model_len`, and `block_size` for your hardware.
3. `CUDA_VISIBLE_DEVICES=... MODEL=<key> python benchmark.py`

## Test Results

`pytest tests/` runs 94 unit tests covering the quantizer, codebooks, bit-packing, the compressed store, and the public vLLM-shim API surface. CI runs them on Python 3.10/3.11/3.12 (CPU only — CUDA-marked variants are skipped on the runner). The unit tests do **not** cover end-to-end behavior with vLLM; see the integration notice at the top of this README for what's currently broken there.

## Limitations

- **Integration is broken on default vLLM** (see top-of-README notice). The next four bullets describe the *intended* behavior; treat them as design goals, not guarantees, until the integration issues are fixed.
- **Prefill still uses paged cache**: KV cache is allocated at engine init and used during prefill. TQ is intended to free it after. True zero-allocation requires deeper vLLM integration.
- **Only full-attention layers**: Linear-attention/Mamba layers are not compressed.
- **Value quantization is the bottleneck**: 2-bit values give cos_sim≈0.94 in isolation; 4-bit values give cos_sim≈0.997. Numbers measured on the standalone quantizer, not end-to-end with vLLM.
- **Hybrid decode dequantizes all history**: When the hybrid branch fires, all compressed tokens are expanded to float32 and attention runs through `torch.einsum`. Empirically ~0.5× baseline tok/s on Qwen2.5-0.5B in eager mode. There is no fused Triton kernel on the live path; an earlier sketch was removed because nothing in the package called it. Wiring a fused decode kernel into `score._attend_compressed_only` is a known perf win, not yet done.
- **MoE models benefit less**: Models with linear-attention layers (Qwen3.5 MoE, Mamba hybrids) have incompressible state that limits TQ's overall impact.

## Environment

Last validated against:
- vLLM 0.17.x / 0.18.x, PyTorch 2.10, CUDA 12.8, Python 3.12
- RTX 5090 (32GB), single GPU, dense AWQ checkpoint
- 8x RTX 3090 (24GB), TP=8, MoE checkpoint with hybrid full-/linear-attention

The integration installs hooks via `vllm.v1.executor.abstract.Executor` and
`vllm.v1.worker.gpu_worker`. Other vLLM versions are likely to break it; pin
your env or expect to patch `turboquant/vllm_attn_backend.py`.
