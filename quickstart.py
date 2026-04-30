#!/usr/bin/env python3
"""
TurboQuant quickstart: baseline vs TQ on a small public Hugging Face model.

This runs entirely in one process — no subprocess fan-out — and serves as the
"does TQ install cleanly on my box?" smoke test.

Usage
-----

    # Default: Qwen2.5-0.5B-Instruct on a single GPU
    python quickstart.py

    # Pick another HF model:
    MODEL=meta-llama/Llama-3.2-1B-Instruct python quickstart.py

    # Multi-GPU:
    CUDA_VISIBLE_DEVICES=0,1 TP=2 python quickstart.py

    # Skip the baseline pass (faster):
    SKIP_BASELINE=1 python quickstart.py

    # Drive the hybrid decode path through the compressed-history branch.
    # Uses a long prompt (>buffer_size+16 tokens) so the prefill split
    # forces tokens into the compressed store, then decodes 128 more.
    MODE=hybrid LONG_PROMPT=1 BUFFER_SIZE=16 python quickstart.py

    # CPU-only sanity (just exercise the quantizer math, no vLLM):
    NO_VLLM=1 python quickstart.py

Exit codes
----------
0  success
1  configuration / env error (no GPU, vLLM missing, etc.)
2  TurboQuant integration failed (likely vLLM version mismatch)
"""

from __future__ import annotations

import os
import sys
import time

# Importing turboquant.vllm has a side effect of setting
# VLLM_ALLOW_INSECURE_SERIALIZATION=1 if it isn't already set, which is
# required for vLLM v1 collective_rpc to transport TurboQuant's installer
# closure to worker processes. This must happen BEFORE `from vllm import LLM`
# so the engine subprocess inherits the env var.
import turboquant.vllm  # noqa: F401  (side-effect import, see comment above)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SHORT_PROMPT = "Briefly explain how KV cache compression speeds up LLM inference."

# Long prompt — ~200+ tokens. Used by LONG_PROMPT=1 so prefill exceeds the
# ring buffer and at least some tokens land in the compressed store, which
# is the only way the hybrid-decode branch is exercised.
LONG_PROMPT = (
    "You are a careful technical writer. Write a thorough, step-by-step "
    "explanation of the following request, citing concrete numbers where "
    "possible and clearly distinguishing what is observed from what is "
    "speculated. The audience is an experienced systems engineer who is "
    "skeptical of marketing claims. Avoid bullet lists; use paragraphs. "
    "Do not repeat the request back. Begin with a single-sentence thesis.\n\n"
    "Request: explain how KV cache compression in transformer inference "
    "(per-token quantization, low-rank projections, or a combination) "
    "trades off memory savings against decoding throughput and output "
    "quality. Cover (a) what is in the cache and why, (b) why naive "
    "uniform quantization underperforms compared to rotation-then-quantize "
    "schemes such as Lloyd-Max on the Beta distribution arising from "
    "random orthogonal rotation of unit-norm vectors, (c) how unbiased "
    "inner-product estimators using residual sign bits (QJL-style) keep "
    "attention scores faithful, and (d) what concrete throughput and "
    "memory numbers a practitioner should expect on modern hardware "
    "running a 7B-parameter model at long context.\n\n"
    "Now write the explanation. Be precise. Be honest. Be brief."
)


def _no_vllm_path() -> int:
    """Exercise just the quantizer math; useful on machines without GPU/vLLM."""
    import torch

    from turboquant import TurboQuantMSE, TurboQuantProd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[no-vllm] device: {device}")

    x = torch.randn(64, 128, device=device)

    mse = TurboQuantMSE(dim=128, bits=3, device=device, seed=0)
    xhat = mse.dequantize(mse.quantize(x))
    cos_mse = torch.nn.functional.cosine_similarity(x, xhat, dim=-1).mean().item()
    print(f"[no-vllm] MSE 3-bit cos_sim: {cos_mse:.4f}")

    prod = TurboQuantProd(dim=128, bits=3, device=device, seed=0)
    qk = prod.quantize(x)
    y = torch.randn(8, 128, device=device)
    est = prod.attention_score(y, qk)
    true = y @ x.T
    err = (est - true).abs().mean().item()
    rel = err / true.abs().mean().clamp_min(1e-9).item()
    print(f"[no-vllm] Prod 3-bit attention-score mean abs err: {err:.4f} (rel {rel:.2%})")
    print("[no-vllm] OK")
    return 0


def _run_pass(llm, label: str, prompt: str, max_tokens: int) -> dict:
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    t0 = time.perf_counter()
    out = llm.generate([prompt], sp)
    dt = time.perf_counter() - t0
    text = out[0].outputs[0].text
    n_tokens = len(out[0].outputs[0].token_ids)
    tps = n_tokens / dt if dt > 0 else float("inf")
    print(f"[{label}] {n_tokens} tok in {dt:.2f}s  ({tps:.1f} tok/s)")
    print(f"[{label}] >>> {text.strip()[:160]}{'…' if len(text) > 160 else ''}")
    return {"tokens": n_tokens, "elapsed": dt, "tps": tps, "text": text}


def main() -> int:
    if os.environ.get("NO_VLLM"):
        return _no_vllm_path()

    try:
        from vllm import LLM
    except Exception as exc:
        print(
            f"ERROR: failed to import vllm ({exc}). Install with `pip install -e .[vllm]` or rerun with NO_VLLM=1.",
            file=sys.stderr,
        )
        return 1

    try:
        from turboquant.vllm import enable_turboquant, get_stats
    except Exception as exc:
        print(f"ERROR: failed to import turboquant.vllm: {exc}", file=sys.stderr)
        return 1

    model = os.environ.get("MODEL", DEFAULT_MODEL)
    tp = int(os.environ.get("TP", "1"))
    max_len = int(os.environ.get("MAX_MODEL_LEN", "4096"))
    gpu_mem = float(os.environ.get("GPU_MEM", "0.85"))
    skip_baseline = bool(os.environ.get("SKIP_BASELINE"))
    mode = os.environ.get("MODE", "capture_only")
    buffer_size = int(os.environ.get("BUFFER_SIZE", "128"))
    long_prompt = bool(os.environ.get("LONG_PROMPT"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "128"))
    prompt = LONG_PROMPT if long_prompt else SHORT_PROMPT

    print(f"Model: {model}")
    print(
        f"TP={tp}  max_model_len={max_len}  gpu_mem={gpu_mem}  "
        f"mode={mode}  buffer_size={buffer_size}  "
        f"prompt={'long' if long_prompt else 'short'}  max_tokens={max_tokens}"
    )
    print()

    enforce_eager = bool(os.environ.get("ENFORCE_EAGER"))
    # ENABLE_PREFIX_CACHING=0 disables vLLM's default prefix cache. Used by
    # docs/plan-path-b.md S0.2 to capture a trace under
    # `enforce_eager=True, enable_prefix_caching=False`.
    prefix_env = os.environ.get("ENABLE_PREFIX_CACHING")
    enable_prefix_caching = True if prefix_env is None else prefix_env not in ("0", "false", "False", "")
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        max_model_len=max_len,
        gpu_memory_utilization=gpu_mem,
        max_num_seqs=1,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        enable_prefix_caching=enable_prefix_caching,
    )

    baseline = None
    if not skip_baseline:
        print(">>> Pass 1/2: baseline vLLM")
        baseline = _run_pass(llm, "baseline", prompt, max_tokens)
        print()

    print(f">>> Installing TurboQuant hooks (mode={mode}, buffer_size={buffer_size})")
    try:
        info = enable_turboquant(
            llm,
            key_bits=3,
            value_bits=2,
            buffer_size=buffer_size,
            mode=mode,
        )
    except Exception as exc:
        print(f"ERROR: enable_turboquant failed: {exc}", file=sys.stderr)
        return 2
    print(f"  installed: {info}")
    print()

    print(f">>> Pass 2/2: TurboQuant {mode}")
    tq = _run_pass(llm, "tq", prompt, max_tokens)
    print()

    compressed_tokens = 0
    try:
        stats = get_stats(llm)
        print(f">>> TQ stats per worker: {stats}")
        # If hybrid mode was supposed to fire, surface whether it actually did.
        compressed_tokens = max((s.get("total_compressed_tokens", 0) for s in stats), default=0)
        if mode == "hybrid":
            if compressed_tokens >= 16:
                print(f">>> hybrid path was exercisable: {compressed_tokens} compressed tokens >= 16 threshold")
            else:
                print(
                    f">>> WARNING: only {compressed_tokens} compressed tokens accumulated "
                    f"(<16 threshold) — hybrid decode never used the compressed branch. "
                    f"Try LONG_PROMPT=1 BUFFER_SIZE=16 to force it."
                )
    except Exception as exc:
        print(f"  (get_stats failed, non-fatal: {exc})")

    if baseline is not None:
        ratio = tq["tps"] / max(baseline["tps"], 0.1)
        same_text = baseline["text"].strip() == tq["text"].strip()
        print()
        print("=" * 60)
        print(f"  baseline:           {baseline['tps']:.1f} tok/s")
        print(f"  tq ({mode}):        {tq['tps']:.1f} tok/s  ({ratio:.2f}x)")
        print(f"  same output text:   {same_text}")
        if mode == "hybrid" and not same_text:
            print(f"  baseline tail: ...{baseline['text'][-120:]!r}")
            print(f"  tq tail:       ...{tq['text'][-120:]!r}")
        print(f"  compressed tokens: {compressed_tokens}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
