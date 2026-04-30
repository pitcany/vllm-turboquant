# TurboQuant: KV Cache Compression for LLM Inference

Implementation of TurboQuant KV cache compression (ICLR 2026, arXiv:2504.19874) with vLLM integration.

> **Status: alpha.** The numerical kernels (codebooks, rotation, quantizer, store) are stable; the vLLM integration monkey-patches private vLLM v1 internals and is pinned to a narrow vLLM range. Benchmarks below were collected on internal model checkpoints and are **not currently reproducible from this repo as-is** — see "Limitations" and "Reproducing benchmarks".

## Benchmark Results

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

A formal test suite has not yet been ported into this repo. Treat `proof.py` /
`benchmark.py` output as the only currently runnable validation path.

## Limitations

- **Prefill still uses paged cache**: KV cache is allocated at engine init and used during prefill. TQ frees it after. True zero-allocation requires deeper vLLM integration.
- **Only full-attention layers**: Linear-attention/Mamba layers are not compressed.
- **Value quantization is the bottleneck**: 2-bit values cause cos_sim=0.94 degradation. Use 4-bit values (cos_sim=0.997) for quality-sensitive workloads.
- **Hybrid decode dequantizes all history**: During compute, all compressed tokens are expanded to float32 and attention runs through `torch.einsum`. There is no fused Triton kernel on the live path; an earlier sketch was removed because nothing in the package called it. Wiring a fused decode kernel into `score._attend_compressed_only` is a known perf win, not yet done.
- **MoE models benefit less**: Models with linear-attention layers (Qwen3.5 MoE, Mamba hybrids) have incompressible state that limits TQ's overall impact.

## Environment

Last validated against:
- vLLM 0.17.x / 0.18.x, PyTorch 2.10, CUDA 12.8, Python 3.12
- RTX 5090 (32GB), single GPU, dense AWQ checkpoint
- 8x RTX 3090 (24GB), TP=8, MoE checkpoint with hybrid full-/linear-attention

The integration installs hooks via `vllm.v1.executor.abstract.Executor` and
`vllm.v1.worker.gpu_worker`. Other vLLM versions are likely to break it; pin
your env or expect to patch `turboquant/vllm_attn_backend.py`.
