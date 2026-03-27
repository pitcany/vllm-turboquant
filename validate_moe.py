#!/usr/bin/env python3
"""
TurboQuant MoE Validation on Qwen3.5-35B-A3B (8x RTX 3090)

Phase 1: Baseline measurements via vLLM OpenAI API
  - Max context length the server can handle (configured at 8192)
  - KV cache usage at different context lengths
  - TTFT and generation speed
  - Output coherence: needle-in-haystack with REAL model output

Phase 2: TurboQuant integration
  - Offline: capture KV states from the model, compress with TQ
  - Measure compression ratio on REAL activations (not synthetic)
  - Measure attention output quality degradation

Architecture: Qwen3.5-35B-A3B (pruned MoE)
  - 40 layers: 30 linear_attention + 10 full_attention (every 4th)
  - full_attention: head_dim=256, num_attention_heads=16, num_kv_heads=2
  - linear_attention: linear_key_head_dim=128, linear_num_key_heads=16
  - 205 experts, 8 active per token
  - TQ only applies to the 10 full_attention layers
"""

import json
import time
import sys
import os
import subprocess

BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3.5-35B-A3B"


def curl_post(endpoint, data, timeout=120):
    """Make a POST request via curl (available on the remote machine)."""
    cmd = [
        "curl", "-s", "-X", "POST",
        f"{BASE_URL}/{endpoint}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(data),
        "--max-time", str(timeout),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
    if result.returncode != 0:
        return {"error": result.stderr}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON: {result.stdout[:500]}"}


def curl_get(endpoint):
    cmd = ["curl", "-s", f"http://localhost:8000/{endpoint}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    return result.stdout


def get_kv_cache_usage():
    metrics = curl_get("metrics")
    for line in metrics.splitlines():
        if line.startswith("vllm:kv_cache_usage_perc{"):
            return float(line.split()[-1])
    return None


def get_gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    gpus = []
    for line in result.stdout.strip().splitlines():
        used, free = [int(x.strip()) for x in line.split(",")]
        gpus.append({"used_mb": used, "free_mb": free})
    return gpus


# ============================================================
# PHASE 1A: Baseline — generation at different prompt lengths
# ============================================================

def build_needle_prompt(context_len, needle_pos_frac=0.5):
    """Build a needle-in-haystack prompt at the given context length.

    Places a specific fact in a sea of filler text, then asks about it.
    Uses token-efficient filler (repeated simple sentences).
    """
    needle = "The secret code for project Zephyr is AURORA-7742."
    question = "What is the secret code for project Zephyr? Answer with just the code."

    # Rough estimate: 1 token ~ 4 chars for English
    target_chars = context_len * 4

    filler_block = (
        "The quick brown fox jumps over the lazy dog. "
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "In a distant galaxy, stars are born and die in cycles of cosmic renewal. "
        "The weather today is partly cloudy with a chance of rain in the afternoon. "
        "Scientists recently discovered a new species of deep-sea fish near hydrothermal vents. "
    )

    needle_char_pos = int(target_chars * needle_pos_frac)

    # Build filler before needle
    filler_before = ""
    while len(filler_before) < needle_char_pos:
        filler_before += filler_block
    filler_before = filler_before[:needle_char_pos]

    # Build filler after needle
    remaining = target_chars - len(filler_before) - len(needle) - len(question) - 100
    filler_after = ""
    while len(filler_after) < remaining:
        filler_after += filler_block
    filler_after = filler_after[:max(0, remaining)]

    prompt = f"{filler_before}\n\n{needle}\n\n{filler_after}\n\n{question}"
    return prompt, "AURORA-7742"


def run_baseline_tests():
    print("=" * 60)
    print("  PHASE 1: Baseline Measurements (vLLM API)")
    print("=" * 60)
    print()

    # Check current state
    kv_usage = get_kv_cache_usage()
    gpu_mem = get_gpu_memory()
    print(f"Initial KV cache usage: {kv_usage:.4%}")
    print(f"GPU memory (avg): {sum(g['used_mb'] for g in gpu_mem)/len(gpu_mem):.0f} MB used, "
          f"{sum(g['free_mb'] for g in gpu_mem)/len(gpu_mem):.0f} MB free")
    print()

    # Test at increasing context lengths
    results = []
    for ctx_len in [512, 1024, 2048, 4096, 6144, 7680]:
        print(f"--- Context length ~{ctx_len} tokens ---")

        prompt, expected = build_needle_prompt(ctx_len, needle_pos_frac=0.5)

        kv_before = get_kv_cache_usage()
        t0 = time.time()

        resp = curl_post("chat/completions", {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=180)

        elapsed = time.time() - t0
        kv_after = get_kv_cache_usage()

        if "error" in resp:
            print(f"  ERROR: {resp['error'][:200]}")
            results.append({"ctx_len": ctx_len, "error": True})
            continue

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        output = ""
        if resp.get("choices"):
            output = resp["choices"][0].get("message", {}).get("content", "")

        needle_found = expected in output
        ttft = elapsed  # approximate (includes network + generation)
        tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

        result = {
            "ctx_len": ctx_len,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_s": round(elapsed, 2),
            "tok_per_sec": round(tok_per_sec, 1),
            "kv_delta": round(kv_after - kv_before, 6) if kv_before and kv_after else None,
            "needle_found": needle_found,
            "output_preview": output[:100],
        }
        results.append(result)

        print(f"  Prompt tokens: {prompt_tokens}")
        print(f"  Completion tokens: {completion_tokens}")
        print(f"  Elapsed: {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")
        print(f"  KV cache delta: {result['kv_delta']}")
        print(f"  Needle found: {needle_found}")
        print(f"  Output: {output[:80]}...")
        print()

    return results


# ============================================================
# PHASE 1B: KV cache memory analysis
# ============================================================

def analyze_kv_memory():
    print("=" * 60)
    print("  KV Cache Memory Analysis")
    print("=" * 60)
    print()

    # From config:
    # full_attention layers: 10 (every 4th of 40)
    # head_dim=256, num_kv_heads=2
    # linear_attention layers: 30
    # linear_key_head_dim=128, linear_num_key_heads=16, linear_num_value_heads=32
    #
    # vLLM block_size=272
    # num_gpu_blocks=4553

    n_full_attn_layers = 10
    full_kv_heads = 2
    full_head_dim = 256

    n_linear_layers = 30
    # Linear attention uses a recurrent state, not a standard KV cache.
    # For Qwen3.5 MoE, linear attention layers use a "linear attention"
    # mechanism where the state size is fixed (not proportional to seq_len).

    # KV cache per token per full_attention layer:
    # K: kv_heads * head_dim * dtype_size = 2 * 256 * 2 = 1024 bytes (bf16)
    # V: same = 1024 bytes
    # Total per token per layer: 2048 bytes
    kv_per_token_per_layer = full_kv_heads * full_head_dim * 2 * 2  # 2 for K+V, 2 for bf16
    kv_per_token_total = kv_per_token_per_layer * n_full_attn_layers

    print(f"Full attention layers: {n_full_attn_layers} (of 40 total)")
    print(f"KV heads: {full_kv_heads}, head_dim: {full_head_dim}")
    print(f"KV cache per token per full_attn layer: {kv_per_token_per_layer} bytes")
    print(f"KV cache per token (all full_attn): {kv_per_token_total} bytes ({kv_per_token_total/1024:.1f} KB)")
    print()

    # vLLM block info
    block_size = 272  # tokens per block
    num_blocks = 4553
    total_kv_bytes = num_blocks * block_size * kv_per_token_total
    print(f"vLLM blocks: {num_blocks} x {block_size} tokens")
    print(f"Total KV capacity: {num_blocks * block_size:,} tokens")
    print(f"Total KV memory: {total_kv_bytes / 1e9:.2f} GB")
    print()

    # With TurboQuant at 3-bit keys + 2-bit values on the 10 full_attn layers:
    # Per token per layer:
    #   Key: 3 bits * head_dim = 3 * 256 = 768 bits = 96 bytes (per head)
    #        * 2 kv_heads = 192 bytes
    #   Plus norms: 2 * 4 bytes * 2 kv_heads = 16 bytes
    #   Value: 2 bits * head_dim = 2 * 256 = 512 bits = 64 bytes (per head)
    #          * 2 kv_heads = 128 bytes
    #   Plus scales/zeros: 2 * 2 * (256/32) * 2 kv_heads = 64 bytes
    #   Total TQ: ~400 bytes per token per layer
    #
    # vs FP16: 2048 bytes per token per layer
    # Ratio: 2048 / 400 = ~5.1x on the full_attn layers only

    tq_key_bytes = full_kv_heads * (full_head_dim * 3 // 8 + 4 + full_head_dim // 8 + 4)  # mse_indices + norms + qjl_signs + residual_norms
    tq_val_bytes = full_kv_heads * (full_head_dim * 2 // 8 + (full_head_dim // 32) * 4)  # packed values + scales/zeros
    tq_per_token_per_layer = tq_key_bytes + tq_val_bytes
    tq_per_token_total = tq_per_token_per_layer * n_full_attn_layers

    ratio = kv_per_token_total / tq_per_token_total

    print(f"TQ KV per token per full_attn layer: {tq_per_token_per_layer} bytes")
    print(f"TQ KV per token (all full_attn): {tq_per_token_total} bytes ({tq_per_token_total/1024:.1f} KB)")
    print(f"Compression ratio (full_attn only): {ratio:.2f}x")
    print()

    # But linear attention layers also use KV cache in vLLM (even if it's recurrent)
    # The actual savings depend on what fraction of total KV memory is full_attention
    # Let's estimate: vLLM treats linear attention layers as having a fixed-size state
    # So the KV cache memory is dominated by full_attention layers for long sequences.

    for seq_len in [1024, 2048, 4096, 8192]:
        fp16_full_attn = seq_len * kv_per_token_total
        tq_full_attn = seq_len * tq_per_token_total
        savings_mb = (fp16_full_attn - tq_full_attn) / 1e6
        # Per GPU (TP=8): each GPU holds 1/8 of KV heads...
        # Actually with 2 KV heads and TP=8, some GPUs may have 0 KV heads for full_attn
        # Let's assume KV is distributed evenly
        savings_per_gpu = savings_mb / 8
        print(f"  seq_len={seq_len:>5}: FP16={fp16_full_attn/1e6:.1f}MB, TQ={tq_full_attn/1e6:.1f}MB, "
              f"savings={savings_mb:.1f}MB total ({savings_per_gpu:.1f}MB/GPU)")

    print()
    print("NOTE: With only 2 KV heads and TP=8, each GPU handles 0.25 KV heads on average.")
    print("The KV cache is already tiny relative to model weights.")
    print("TQ savings are real but small in absolute terms for this model.")


# ============================================================
# PHASE 1C: Coherence test — multi-needle at different depths
# ============================================================

def run_coherence_test():
    print()
    print("=" * 60)
    print("  PHASE 1C: Multi-Needle Coherence Test")
    print("=" * 60)
    print()

    # Place 3 facts at different positions, then ask about all of them
    facts = [
        ("The capital of the fictional country Zephyria is Windholm.", "Windholm"),
        ("Agent Cooper's badge number is 4491.", "4491"),
        ("The password to the vault is CRIMSON-DELTA-9.", "CRIMSON-DELTA-9"),
    ]

    ctx_len = 6144  # Use most of the 8192 context
    filler_block = (
        "The quick brown fox jumps over the lazy dog. "
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "In a distant galaxy, stars are born and die in cycles. "
        "The weather is partly cloudy with rain expected later. "
        "Scientists discovered a new species near deep-sea vents. "
    )

    target_chars = ctx_len * 4
    segment_len = target_chars // 4  # 4 segments: filler, fact1, filler, fact2, filler, fact3, filler

    parts = []
    for i, (fact, _) in enumerate(facts):
        filler = ""
        while len(filler) < segment_len:
            filler += filler_block
        filler = filler[:segment_len]
        parts.append(filler)
        parts.append(f"\n{fact}\n")

    # Final filler
    filler = ""
    while len(filler) < segment_len:
        filler += filler_block
    parts.append(filler[:segment_len])

    question = (
        "\n\nAnswer these three questions:\n"
        "1. What is the capital of Zephyria?\n"
        "2. What is Agent Cooper's badge number?\n"
        "3. What is the password to the vault?\n"
        "Give just the answers, one per line."
    )
    parts.append(question)

    prompt = "".join(parts)

    print(f"Prompt length: ~{len(prompt)//4} tokens (estimated)")
    t0 = time.time()

    resp = curl_post("chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=180)

    elapsed = time.time() - t0

    if "error" in resp:
        print(f"ERROR: {resp['error'][:200]}")
        return

    output = ""
    if resp.get("choices"):
        output = resp["choices"][0].get("message", {}).get("content", "")

    usage = resp.get("usage", {})
    print(f"Prompt tokens: {usage.get('prompt_tokens', '?')}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Output:\n{output}")
    print()

    found = 0
    for fact_text, expected in facts:
        if expected.lower() in output.lower():
            found += 1
            print(f"  FOUND: {expected}")
        else:
            print(f"  MISSED: {expected}")

    print(f"\nCoherence score: {found}/{len(facts)} facts retrieved")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(f"TurboQuant MoE Validation")
    print(f"Model: {MODEL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Phase 1A: Baseline generation at different context lengths
    results = run_baseline_tests()

    # KV memory analysis
    analyze_kv_memory()

    # Phase 1C: Coherence test
    run_coherence_test()

    # Save results
    with open("/tmp/tq_moe_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /tmp/tq_moe_baseline.json")
