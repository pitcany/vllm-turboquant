#!/usr/bin/env python3
"""TurboQuant Path B benchmark — capture-only memory-only wins (Sprint 4 / S4.2).

The reproducer the README's "Verified configuration" section advertises.
Per ``docs/plan-path-b.md`` §3 / Sprint 4 / S4.2:

> A runnable script (replaces or supplements ``benchmark.py``) that takes
> ``--model HF_ID --context-len N --decode-tokens M`` and reports:
>   - VRAM used by KV cache (baseline vs. TQ-after-``free_kv_cache``)
>   - Decode tok/s (baseline vs. TQ)
>   - Top-1 agreement % (the correctness number)

Why ``mode="capture_only"`` and not ``mode="hybrid"``: per Sprint 3 / S3.3
(``docs/integration-state.md`` § "S3.3 follow-up"), hybrid attention does
not clear the 95% top-1 agreement bar on Llama-3.2-1B-Instruct at any
bit budget tested (3/2 → 7.69%, 3/4 → 0.39%, 4/4 → 2.73%). The §5
second-bullet stop-loss is engaged; Sprint 4 takes the §5 third-bullet
pivot to capture-only memory-only wins. In capture_only TQ never
intercepts the attention compute, so output is **baseline-by-construction**
(top-1 agreement = 1.0). The story is purely VRAM saved.

Invocation::

    CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \\
      LD_LIBRARY_PATH="$CONDA_PREFIX/lib" \\
      python scripts/bench_path_b.py \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --context-len 2048 \\
        --decode-tokens 256

What it measures, in order:

1. Build LLM, time prefill+decode on the prompt → baseline decode tok/s.
   Capture token IDs.
2. Snapshot VRAM (``torch.cuda.memory_allocated`` per visible device,
   plus ``nvidia-smi`` for the same numbers — they sometimes diverge
   under vLLM's CUDAGraph captured allocations).
3. ``enable_turboquant(mode="capture_only")``. Re-run prefill+decode →
   TQ decode tok/s. Capture token IDs.
4. Compute top-1 agreement (= 1.0 by construction in capture_only).
5. ``free_kv_cache(llm)`` → returns bytes-freed.
6. Snapshot VRAM again.
7. Print a table the user can verify.

The "VRAM saved by TQ store vs. paged cache" number is computed two
ways:
  - **Allocator delta**: ``free_kv_cache`` return value (bytes the
    sentinel swap reclaimed from ``attn_module.kv_cache[0]``).
  - **Process-level delta**: nvidia-smi pre-free vs. post-free, after
    ``torch.cuda.empty_cache()``.
Both are reported because vLLM's CUDAGraph captures hold references
that prevent the allocator-level delta from translating 1:1 into
process-level reclamation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Force spawn so the worker subprocess gets a clean Python interpreter.
# Pytest's parent process preloads torch which inits CUDA in the parent
# before vLLM forks workers. vLLM's default fork then crashes.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


# Same long prompt as quickstart.py / tests/test_correctness_e2e.py /
# tests/test_vllm_smoke.py. ~226 tokens on Qwen2.5-0.5B's tokenizer;
# slightly different on Llama. Sized to exceed buffer_size=16 + reach
# MIN_HISTORY_FOR_TQ=16 in kv_tq even after prefix caching is off.
DEFAULT_PROMPT = (
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


def _nvidia_smi_used_mib() -> list[int]:
    """Per-visible-GPU memory.used in MiB via nvidia-smi.

    Returns one entry per device in ``CUDA_VISIBLE_DEVICES`` order. Empty
    list if nvidia-smi is unavailable.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    out = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(int(line))
        except ValueError:
            continue
    return out


def _torch_alloc_mib(device_idx: int = 0) -> float:
    """torch.cuda.memory_allocated for the given visible device, in MiB."""
    import torch

    return torch.cuda.memory_allocated(device_idx) / (1024 * 1024)


def _greedy(llm, prompt: str, max_tokens: int) -> tuple[list[int], float]:
    """One greedy generation. Returns (token_ids, elapsed_seconds)."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    t0 = time.time()
    out = llm.generate([prompt], sp)
    elapsed = time.time() - t0
    return list(out[0].outputs[0].token_ids), elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model id (plan §2 verification model is the default).",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=4096,
        help="vLLM max_model_len. Must accommodate prompt + decode horizon.",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=256,
        help="max_tokens for the greedy decode (top-1 agreement is computed over the first min(baseline, tq) tokens).",
    )
    parser.add_argument(
        "--key-bits",
        type=int,
        default=3,
        help="TurboQuantProd key bits (plan §2 default = 3).",
    )
    parser.add_argument(
        "--value-bits",
        type=int,
        default=2,
        help="V symmetric-group bits (plan §2 default = 2).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=16,
        help="ring buffer capacity. Must be small enough that prefill "
        "overflows it into kv_tq for free_kv_cache to find anything to "
        "release.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM gpu_memory_utilization knob.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="prompt text (default: the canonical Sprint 1+ workload).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="if given, also writes the structured result as JSON here.",
    )
    args = parser.parse_args(argv)

    import torch

    # NB: ``torch.cuda.is_available()`` emits a false-negative under some
    # driver/runtime combinations (e.g. driver 575.57 reporting CUDA 12.9
    # vs. a torch wheel built for an older runtime); ``device_count() > 0``
    # is the more permissive check vLLM itself uses.
    if torch.cuda.device_count() == 0:
        print("CUDA unavailable — skipping benchmark.", file=sys.stderr)
        return 2

    # Side-effect import: sets VLLM_ALLOW_INSECURE_SERIALIZATION=1 so the
    # worker subprocess accepts our installer closure. Must happen before
    # vllm.LLM is imported.
    from vllm import LLM

    import turboquant.vllm  # noqa: F401
    from turboquant.vllm import enable_turboquant, free_kv_cache, get_stats

    print(f"[bench] model={args.model}", flush=True)
    print(
        f"[bench] context_len={args.context_len} decode_tokens={args.decode_tokens} "
        f"key_bits={args.key_bits} value_bits={args.value_bits} buffer_size={args.buffer_size}",
        flush=True,
    )

    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=1,
            max_model_len=args.context_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=1,
            enforce_eager=True,
            enable_prefix_caching=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[bench] could not construct LLM({args.model!r}): {exc}", file=sys.stderr)
        return 3

    # ---- Pass 1: baseline ----
    baseline_tokens, baseline_elapsed = _greedy(llm, args.prompt, args.decode_tokens)
    baseline_tok_per_s = len(baseline_tokens) / baseline_elapsed if baseline_elapsed else 0.0
    vram_baseline_smi = _nvidia_smi_used_mib()
    vram_baseline_torch = _torch_alloc_mib()

    # ---- Pass 2: TurboQuant capture-only ----
    enable_turboquant(
        llm,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        buffer_size=args.buffer_size,
        mode="capture_only",
    )
    tq_tokens, tq_elapsed = _greedy(llm, args.prompt, args.decode_tokens)
    tq_tok_per_s = len(tq_tokens) / tq_elapsed if tq_elapsed else 0.0
    vram_tq_pre_free_smi = _nvidia_smi_used_mib()
    vram_tq_pre_free_torch = _torch_alloc_mib()

    n = min(len(baseline_tokens), len(tq_tokens))
    matches = sum(1 for a, b in zip(baseline_tokens[:n], tq_tokens[:n]) if a == b)
    agreement = matches / n if n else 0.0
    first_div = next(
        (i for i, (a, b) in enumerate(zip(baseline_tokens[:n], tq_tokens[:n])) if a != b),
        n,
    )

    stats = get_stats(llm)
    captured = sum(s.get("total_compressed_tokens", 0) + s.get("total_buffered_tokens", 0) for s in stats)
    tq_store_bytes = sum(s.get("total_memory_bytes", 0) for s in stats)

    # ---- Pass 3: free + measure ----
    freed_bytes = 0
    free_error: str | None = None
    try:
        freed_bytes = free_kv_cache(llm)
    except Exception as exc:  # noqa: BLE001
        free_error = f"{type(exc).__name__}: {exc}"
        print(f"[bench] free_kv_cache failed: {free_error}", file=sys.stderr)

    torch.cuda.empty_cache()
    vram_post_free_smi = _nvidia_smi_used_mib()
    vram_post_free_torch = _torch_alloc_mib()

    # ---- Report ----
    smi_delta = (
        (vram_tq_pre_free_smi[0] - vram_post_free_smi[0]) if vram_tq_pre_free_smi and vram_post_free_smi else None
    )
    result: dict[str, Any] = {
        "model": args.model,
        "context_len": args.context_len,
        "decode_tokens": args.decode_tokens,
        "key_bits": args.key_bits,
        "value_bits": args.value_bits,
        "buffer_size": args.buffer_size,
        "baseline_decode_tok_per_s": round(baseline_tok_per_s, 1),
        "tq_decode_tok_per_s": round(tq_tok_per_s, 1),
        "tq_speed_ratio": round(tq_tok_per_s / baseline_tok_per_s, 3) if baseline_tok_per_s else None,
        "baseline_token_count": len(baseline_tokens),
        "tq_token_count": len(tq_tokens),
        "compared_tokens": n,
        "top1_agreement": round(agreement, 4),
        "first_divergence": first_div,
        "captured_tokens_per_layer_avg": captured // max(len(stats), 1) if stats else 0,
        "tq_store_bytes": tq_store_bytes,
        "freed_bytes_allocator": freed_bytes,
        "free_error": free_error,
        "vram_baseline_mib_smi": vram_baseline_smi,
        "vram_baseline_mib_torch": round(vram_baseline_torch, 1),
        "vram_tq_pre_free_mib_smi": vram_tq_pre_free_smi,
        "vram_tq_pre_free_mib_torch": round(vram_tq_pre_free_torch, 1),
        "vram_post_free_mib_smi": vram_post_free_smi,
        "vram_post_free_mib_torch": round(vram_post_free_torch, 1),
        "vram_smi_delta_mib_after_free": smi_delta,
    }

    print()
    print("=" * 72)
    print("TurboQuant Path B benchmark — capture-only memory-only wins (S4.2)")
    print("=" * 72)
    print(f"Model:                  {args.model}")
    print(f"Context len / decode:   {args.context_len} / {args.decode_tokens}")
    print(f"Key/value bits:         {args.key_bits} / {args.value_bits}")
    print()
    print("Throughput (decode tok/s)")
    print(f"  baseline:             {baseline_tok_per_s:>8.1f}")
    print(
        f"  TQ (capture_only):    {tq_tok_per_s:>8.1f}  ({result['tq_speed_ratio']:.3f}× baseline)"
        if result["tq_speed_ratio"] is not None
        else f"  TQ (capture_only):    {tq_tok_per_s:>8.1f}"
    )
    print()
    print("Correctness (capture_only is baseline-by-construction)")
    print(f"  top-1 agreement:      {agreement:.4f}  ({matches}/{n})")
    print(f"  first divergence:     {first_div}")
    print()
    print("Memory")
    print(f"  TQ store size:        {tq_store_bytes / (1024 * 1024):>8.1f} MiB")
    print(f"  free_kv_cache freed:  {freed_bytes / (1024 * 1024):>8.1f} MiB (allocator)")
    if smi_delta is not None:
        print(f"  nvidia-smi delta:     {smi_delta:>8.0f} MiB (process-level after empty_cache)")
    if free_error:
        print(f"  free_kv_cache error:  {free_error}")
    print()
    print(f"VRAM via nvidia-smi (MiB):    baseline={vram_baseline_smi}")
    print(f"                              tq_pre_free={vram_tq_pre_free_smi}")
    print(f"                              tq_post_free={vram_post_free_smi}")
    print()
    print("JSON result:")
    print(json.dumps(result, indent=2))

    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"\n[bench] wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
