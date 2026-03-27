#!/usr/bin/env python3
"""
Measure KV cache usage during CONCURRENT requests to see actual block allocation.
Also compute the definitive baseline vs TQ comparison.
"""

import json
import time
import subprocess
import tempfile
import os
import threading

BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3.5-35B-A3B"

NUM_GPU_BLOCKS = 5190
BLOCK_SIZE = 272
TP_SIZE = 8

# Per block costs
FULL_ATTN_PER_BLOCK = BLOCK_SIZE * 10 * 2 * 256 * 2 * 2  # 5.57 MB
LINEAR_PER_BLOCK = 278528 * 30                              # 8.36 MB
TOTAL_PER_BLOCK = FULL_ATTN_PER_BLOCK + LINEAR_PER_BLOCK    # 13.93 MB

# TQ per block
TQ_FULL_ATTN_PER_BLOCK = BLOCK_SIZE * 10 * 464             # 1.26 MB
TQ_TOTAL_PER_BLOCK = TQ_FULL_ATTN_PER_BLOCK + LINEAR_PER_BLOCK  # 9.62 MB


def get_kv_usage():
    result = subprocess.run(["curl", "-s", "http://localhost:8000/metrics"],
                          capture_output=True, text=True, timeout=10)
    for line in result.stdout.splitlines():
        if line.startswith("vllm:kv_cache_usage_perc{"):
            return float(line.split()[-1])
    return 0


def send_long_request(target_tokens, max_gen=500):
    """Send a request that generates many tokens to keep KV cache allocated."""
    filler = "The quick brown fox jumps over the lazy dog. " * (target_tokens // 10)
    filler = filler[:target_tokens * 4]
    prompt = filler + "\n\nWrite a very long and detailed essay about the history of computing, at least 500 words."

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir='/tmp') as f:
        json.dump({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_gen,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }, f)
        tmppath = f.name

    cmd = ["curl", "-s", "-X", "POST", f"{BASE_URL}/chat/completions",
           "-H", "Content-Type: application/json", "-d", f"@{tmppath}",
           "--max-time", "600"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=610)
    os.unlink(tmppath)
    try:
        return json.loads(result.stdout)
    except:
        return {"error": "failed"}


def monitor_kv_usage(duration, interval=0.2):
    """Monitor KV cache usage for a given duration."""
    samples = []
    t0 = time.time()
    while time.time() - t0 < duration:
        kv = get_kv_usage()
        samples.append((time.time() - t0, kv))
        time.sleep(interval)
    return samples


print("=" * 80)
print("  ACTUAL KV CACHE MEASUREMENT DURING INFERENCE")
print("=" * 80)
print()

# Send a long-generation request and monitor KV usage during it
for target_ctx in [8000, 32000, 64000, 100000, 131000]:
    print(f"--- Context ~{target_ctx:,} tokens + 500 generation tokens ---")

    # Start monitoring in background
    kv_samples = []
    stop_monitor = threading.Event()

    def monitor():
        t0 = time.time()
        while not stop_monitor.is_set():
            kv = get_kv_usage()
            kv_samples.append((time.time() - t0, kv))
            time.sleep(0.1)

    mon_thread = threading.Thread(target=monitor, daemon=True)
    mon_thread.start()

    # Send request
    t0 = time.time()
    resp = send_long_request(target_ctx, max_gen=500)
    elapsed = time.time() - t0

    # Let it settle
    time.sleep(0.5)
    stop_monitor.set()
    mon_thread.join(timeout=2)

    # Find peak KV usage
    if kv_samples:
        peak_kv = max(s[1] for s in kv_samples)
        peak_time = [s[0] for s in kv_samples if s[1] == peak_kv][0]
    else:
        peak_kv = 0
        peak_time = 0

    # Compute actual bytes
    blocks_used = peak_kv * NUM_GPU_BLOCKS
    baseline_bytes = blocks_used * TOTAL_PER_BLOCK
    baseline_full_attn = blocks_used * FULL_ATTN_PER_BLOCK
    baseline_linear = blocks_used * LINEAR_PER_BLOCK

    # TQ equivalent
    tokens_in_cache = blocks_used * BLOCK_SIZE
    tq_full_attn = tokens_in_cache * 10 * 464  # TQ compressed
    tq_total = tq_full_attn + baseline_linear
    savings = baseline_bytes - tq_total

    usage = resp.get("usage", {}) if "error" not in resp else {}
    prompt_tokens = usage.get("prompt_tokens", "?")
    completion_tokens = usage.get("completion_tokens", "?")

    print(f"  Prompt: {prompt_tokens} tok, Generated: {completion_tokens} tok, Time: {elapsed:.1f}s")
    print(f"  Peak KV usage: {peak_kv:.4%} at t={peak_time:.1f}s")
    print(f"  Blocks used: {blocks_used:.0f} ({tokens_in_cache:.0f} tokens)")
    print()
    print(f"  BASELINE (bf16 KV):")
    print(f"    full_attn KV:  {baseline_full_attn/1e6:>10.1f} MB total  ({baseline_full_attn/TP_SIZE/1e6:>8.1f} MB/GPU)")
    print(f"    linear state:  {baseline_linear/1e6:>10.1f} MB total  ({baseline_linear/TP_SIZE/1e6:>8.1f} MB/GPU)")
    print(f"    TOTAL:         {baseline_bytes/1e6:>10.1f} MB total  ({baseline_bytes/TP_SIZE/1e6:>8.1f} MB/GPU)")
    print()
    print(f"  WITH TQ (3b key / 2b val):")
    print(f"    tq full_attn:  {tq_full_attn/1e6:>10.1f} MB total  ({tq_full_attn/TP_SIZE/1e6:>8.1f} MB/GPU)")
    print(f"    linear state:  {baseline_linear/1e6:>10.1f} MB total  ({baseline_linear/TP_SIZE/1e6:>8.1f} MB/GPU)")
    print(f"    TOTAL:         {tq_total/1e6:>10.1f} MB total  ({tq_total/TP_SIZE/1e6:>8.1f} MB/GPU)")
    print()
    print(f"  SAVINGS:         {savings/1e6:>10.1f} MB total  ({savings/TP_SIZE/1e6:>8.1f} MB/GPU)  ({savings/baseline_bytes*100:.1f}% reduction)" if baseline_bytes > 0 else "  SAVINGS: N/A")
    print()


# Final summary
print("=" * 80)
print("  SUMMARY: What TQ buys you on this model")
print("=" * 80)
print()
print("KV cache pool: 9,035 MB/GPU (72.3 GB total)")
print(f"  40% is full_attention (3,614 MB/GPU) -- compressible by TQ at 4.4x")
print(f"  60% is linear_attention (5,421 MB/GPU) -- NOT compressible")
print()
print("With TQ applied to full_attention layers:")
print(f"  full_attn compressed from 3,614 to ~820 MB/GPU (4.4x)")
print(f"  Total KV pool: 820 + 5,421 = 6,241 MB/GPU (vs 9,035 baseline)")
print(f"  VRAM savings: 2,794 MB/GPU = 22.4 GB total")
print()
print("Context extension:")
print(f"  Baseline: {NUM_GPU_BLOCKS * BLOCK_SIZE:,} tokens capacity")
new_blocks = int(NUM_GPU_BLOCKS * TOTAL_PER_BLOCK / TQ_TOTAL_PER_BLOCK)
print(f"  With TQ:  {new_blocks * BLOCK_SIZE:,} tokens capacity")
print(f"  Extension: {new_blocks * BLOCK_SIZE / (NUM_GPU_BLOCKS * BLOCK_SIZE):.2f}x")
print()
print("OR: use freed VRAM for larger batch size / more concurrent requests")
print(f"  Extra VRAM: 2,794 MB/GPU could serve ~{int(2794 / (TOTAL_PER_BLOCK/BLOCK_SIZE*131072/1e6/TP_SIZE))} additional 131k-context requests concurrently")
