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

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Briefly explain how KV cache compression speeds up LLM inference."


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
    print(f"[no-vllm] Prod 3-bit attention-score mean abs err: "
          f"{err:.4f} (rel {rel:.2%})")
    print("[no-vllm] OK")
    return 0


def _run_pass(llm, label: str) -> dict:
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0, max_tokens=64)
    t0 = time.perf_counter()
    out = llm.generate([PROMPT], sp)
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
        print(f"ERROR: failed to import vllm ({exc}). "
              "Install with `pip install -e .[vllm]` or rerun with NO_VLLM=1.",
              file=sys.stderr)
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

    print(f"Model: {model}")
    print(f"TP={tp}  max_model_len={max_len}  gpu_mem={gpu_mem}")
    print()

    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        max_model_len=max_len,
        gpu_memory_utilization=gpu_mem,
        max_num_seqs=1,
        trust_remote_code=True,
    )

    baseline = None
    if not skip_baseline:
        print(">>> Pass 1/2: baseline vLLM")
        baseline = _run_pass(llm, "baseline")
        print()

    print(">>> Installing TurboQuant hooks (mode=capture_only)")
    try:
        info = enable_turboquant(
            llm,
            key_bits=3,
            value_bits=2,
            buffer_size=128,
            mode="capture_only",
        )
    except Exception as exc:
        print(f"ERROR: enable_turboquant failed: {exc}", file=sys.stderr)
        return 2
    print(f"  installed: {info}")
    print()

    print(">>> Pass 2/2: TurboQuant capture_only")
    tq = _run_pass(llm, "tq")
    print()

    try:
        stats = get_stats(llm)
        print(f">>> TQ stats per worker: {stats}")
    except Exception as exc:
        print(f"  (get_stats failed, non-fatal: {exc})")

    if baseline is not None:
        ratio = tq["tps"] / max(baseline["tps"], 0.1)
        print()
        print("=" * 60)
        print(f"  baseline: {baseline['tps']:.1f} tok/s")
        print(f"  tq:       {tq['tps']:.1f} tok/s  ({ratio:.2f}x)")
        same_text = baseline["text"].strip() == tq["text"].strip()
        print(f"  same output text: {same_text}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
