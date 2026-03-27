#!/usr/bin/env python3
"""
Baseline vs TQ: Actual VRAM measurement.

vLLM pre-allocates KV cache blocks at startup, so nvidia-smi won't show per-request
deltas. Instead we measure:

1. BASELINE (current server, bf16 KV):
   - vLLM reports num_gpu_blocks=5190, block_size=272
   - KV cache usage % during inference at different context lengths
   - We can compute exact bytes from: blocks_used * block_size * per_token_kv_bytes
   
2. Calculate what TQ would save based on measured block usage

3. Also: restart server with LOWER gpu_memory_utilization to see where it OOMs,
   which tells us the REAL memory pressure.
"""

import json
import time
import subprocess
import tempfile
import os

BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3.5-35B-A3B"

# From model config and vLLM metrics:
NUM_FULL_ATTN_LAYERS = 10
NUM_LINEAR_LAYERS = 30
HEAD_DIM = 256
NUM_KV_HEADS = 2
BLOCK_SIZE = 272       # tokens per block
NUM_GPU_BLOCKS = 5190  # total blocks allocated
TP_SIZE = 8

# Per-token KV size for full_attention layers (bf16)
# K + V per layer = 2 * kv_heads * head_dim * 2 bytes (bf16) = 2 * 2 * 256 * 2 = 2048 bytes
KV_PER_TOKEN_PER_FULL_LAYER = 2 * NUM_KV_HEADS * HEAD_DIM * 2  # 2048 bytes
KV_PER_TOKEN_ALL_FULL = KV_PER_TOKEN_PER_FULL_LAYER * NUM_FULL_ATTN_LAYERS  # 20480 bytes

# Linear attention state per block (mamba_page_size_padded=278528 per layer)
# This is a FIXED cost per block, not per token
MAMBA_PAGE_SIZE = 278528  # bytes per layer per block (from vLLM config)

# Per block total:
# full_attn: BLOCK_SIZE * KV_PER_TOKEN_ALL_FULL = 272 * 20480 = 5,570,560 bytes
# linear_attn: MAMBA_PAGE_SIZE * NUM_LINEAR_LAYERS = 278528 * 30 = 8,355,840 bytes
# Total per block: ~13.9 MB
FULL_ATTN_PER_BLOCK = BLOCK_SIZE * KV_PER_TOKEN_ALL_FULL
LINEAR_PER_BLOCK = MAMBA_PAGE_SIZE * NUM_LINEAR_LAYERS
TOTAL_PER_BLOCK = FULL_ATTN_PER_BLOCK + LINEAR_PER_BLOCK

# TQ per-token (3-bit keys, 2-bit values) for full_attention layers:
# Key: mse_indices (packed) + qjl_signs (packed) + norms (float32) + residual_norms (float32)
# For head_dim=256, 3-bit TQ:
#   mse_indices: 256 * 2 / 8 = 64 bytes (2-bit MSE, packed)
#   qjl_signs: 256 / 8 = 32 bytes
#   norms: 4 bytes, residual_norms: 4 bytes
#   Total key per head: 104 bytes
# Value (2-bit group quant, group_size=32):
#   data: 256 * 2 / 8 = 64 bytes (packed)
#   scales + zeros: 2 * (256/32) * 4 = 64 bytes
#   Total value per head: 128 bytes
# Per token per layer: (104 + 128) * 2 kv_heads = 464 bytes
TQ_PER_TOKEN_PER_FULL_LAYER = 464
TQ_PER_TOKEN_ALL_FULL = TQ_PER_TOKEN_PER_FULL_LAYER * NUM_FULL_ATTN_LAYERS  # 4640 bytes

# TQ overhead per layer (Pi + S matrices): 256*256*4 * 2 = 524288 bytes = 0.5 MB
TQ_OVERHEAD_PER_LAYER = 2 * HEAD_DIM * HEAD_DIM * 4
TQ_OVERHEAD_TOTAL = TQ_OVERHEAD_PER_LAYER * NUM_FULL_ATTN_LAYERS  # 5.24 MB


def curl_post_file(endpoint, data, timeout=1200):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir='/tmp') as f:
        json.dump(data, f)
        tmppath = f.name
    try:
        cmd = ["curl", "-s", "-X", "POST", f"{BASE_URL}/{endpoint}",
               "-H", "Content-Type: application/json", "-d", f"@{tmppath}",
               "--max-time", str(timeout)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
        return json.loads(result.stdout) if result.returncode == 0 else {"error": result.stderr[:500]}
    except json.JSONDecodeError:
        return {"error": "bad json"}
    finally:
        os.unlink(tmppath)


def get_kv_usage():
    result = subprocess.run(["curl", "-s", "http://localhost:8000/metrics"],
                          capture_output=True, text=True, timeout=10)
    for line in result.stdout.splitlines():
        if line.startswith("vllm:kv_cache_usage_perc{"):
            return float(line.split()[-1])
    return 0


def build_filler_prompt(target_tokens):
    filler = (
        "In the vast digital landscape, information flows through networks. "
        "Each node processes data according to its designated protocols. "
        "Modern cloud infrastructure supports concurrent operations. "
        "Load balancers distribute traffic among server clusters. "
    )
    target_chars = target_tokens * 4
    text = ""
    while len(text) < target_chars:
        text += filler
    return text[:target_chars] + "\n\nSay 'OK' and nothing else."


def measure_kv_during_request(target_tokens):
    """Send a request and measure KV cache usage DURING processing."""
    prompt = build_filler_prompt(target_tokens)

    # Start the request (non-blocking via streaming)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir='/tmp') as f:
        json.dump({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": True,
        }, f)
        tmppath = f.name

    proc = subprocess.Popen(
        ["curl", "-s", "-N", "-X", "POST", f"{BASE_URL}/chat/completions",
         "-H", "Content-Type: application/json", "-d", f"@{tmppath}",
         "--max-time", "600"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    # Poll KV usage while request is in flight
    max_kv = 0
    samples = 0
    ttft = None
    t0 = time.time()

    for line in proc.stdout:
        line = line.strip()
        # Sample KV usage
        kv = get_kv_usage()
        max_kv = max(max_kv, kv)
        samples += 1

        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if content and ttft is None:
                    ttft = time.time() - t0
            except json.JSONDecodeError:
                pass

    proc.wait(timeout=10)
    elapsed = time.time() - t0

    # One more sample after completion
    kv_after = get_kv_usage()

    os.unlink(tmppath)
    return max_kv, kv_after, ttft, elapsed, samples


print("=" * 80)
print("  BASELINE vs TURBOQUANT: VRAM ANALYSIS")
print("=" * 80)
print()

# Total pre-allocated KV cache memory
total_kv_cache_bytes = NUM_GPU_BLOCKS * TOTAL_PER_BLOCK
print(f"vLLM KV cache pool:")
print(f"  Blocks: {NUM_GPU_BLOCKS} x {BLOCK_SIZE} tokens = {NUM_GPU_BLOCKS * BLOCK_SIZE:,} token capacity")
print(f"  Per block: full_attn={FULL_ATTN_PER_BLOCK/1e6:.2f}MB + linear={LINEAR_PER_BLOCK/1e6:.2f}MB = {TOTAL_PER_BLOCK/1e6:.2f}MB")
print(f"  Total pre-allocated: {total_kv_cache_bytes/1e9:.2f} GB across {TP_SIZE} GPUs ({total_kv_cache_bytes/TP_SIZE/1e9:.2f} GB/GPU)")
print()

# How much of this is full_attention (compressible by TQ)?
full_attn_total = NUM_GPU_BLOCKS * FULL_ATTN_PER_BLOCK
linear_total = NUM_GPU_BLOCKS * LINEAR_PER_BLOCK
print(f"  Full attention (TQ-compressible): {full_attn_total/1e9:.2f} GB ({full_attn_total/total_kv_cache_bytes*100:.1f}%)")
print(f"  Linear attention (incompressible): {linear_total/1e9:.2f} GB ({linear_total/total_kv_cache_bytes*100:.1f}%)")
print()

print("=" * 80)
print("  BASELINE: KV cache usage at different context lengths")
print("=" * 80)
print()

print(f"{'Context':>10} {'Prompt':>8} {'KV peak':>10} {'KV bytes':>14} {'full_attn':>12} {'linear':>12} {'TTFT':>8}")
print(f"{'target':>10} {'tokens':>8} {'(%)':>10} {'used':>14} {'KV':>12} {'state':>12} {'(s)':>8}")
print("-" * 85)

results = []
for target in [1000, 4000, 8000, 16000, 32000, 64000, 100000, 131000]:
    max_kv, kv_after, ttft, elapsed, samples = measure_kv_during_request(target)

    # Estimate actual tokens from KV usage
    blocks_used = max_kv * NUM_GPU_BLOCKS
    tokens_used = blocks_used * BLOCK_SIZE

    # Bytes used
    full_attn_bytes = blocks_used * FULL_ATTN_PER_BLOCK
    linear_bytes = blocks_used * LINEAR_PER_BLOCK
    total_bytes = full_attn_bytes + linear_bytes

    # With TQ: full_attn portion compressed
    tq_full_attn_bytes = (tokens_used * TQ_PER_TOKEN_ALL_FULL) + TQ_OVERHEAD_TOTAL if tokens_used > 0 else 0
    tq_total = tq_full_attn_bytes + linear_bytes
    savings = total_bytes - tq_total

    r = {
        "target": target,
        "kv_peak_pct": max_kv,
        "blocks_used": blocks_used,
        "tokens_est": tokens_used,
        "total_bytes": total_bytes,
        "full_attn_bytes": full_attn_bytes,
        "linear_bytes": linear_bytes,
        "tq_total": tq_total,
        "savings_bytes": savings,
        "ttft": ttft,
    }
    results.append(r)

    print(f"{target:>10,} {'?':>8} {max_kv:>10.4%} {total_bytes/1e6:>12.1f}MB {full_attn_bytes/1e6:>10.1f}MB {linear_bytes/1e6:>10.1f}MB {ttft if ttft else '?':>8}")


print()
print("=" * 80)
print("  BASELINE vs TURBOQUANT COMPARISON")
print("=" * 80)
print()

print(f"{'Context':>10} {'Baseline':>14} {'With TQ':>14} {'Savings':>14} {'Savings':>10} {'Savings':>12}")
print(f"{'target':>10} {'KV total':>14} {'KV total':>14} {'total':>14} {'per GPU':>10} {'(%)':>12}")
print("-" * 80)

for r in results:
    baseline = r["total_bytes"]
    tq = r["tq_total"]
    sav = r["savings_bytes"]
    if baseline > 0:
        pct = sav / baseline * 100
        print(f"{r['target']:>10,} {baseline/1e6:>12.1f}MB {tq/1e6:>12.1f}MB {sav/1e6:>12.1f}MB {sav/TP_SIZE/1e6:>8.1f}MB {pct:>10.1f}%")
    else:
        print(f"{r['target']:>10,} {'N/A':>14}")


# Now do the key comparison: what if we gave TQ's freed memory BACK to context?
print()
print("=" * 80)
print("  CONTEXT EXTENSION: How much more context with TQ?")
print("=" * 80)
print()

# Current capacity: 5190 blocks * 272 = 1,411,680 tokens
# But max_model_len=131072

# If we compress full_attn KV with TQ, we free up memory that could be used for MORE blocks
# The freed memory per block = (FULL_ATTN_PER_BLOCK - TQ equivalent per block)
tq_full_attn_per_block = BLOCK_SIZE * TQ_PER_TOKEN_ALL_FULL
saved_per_block = FULL_ATTN_PER_BLOCK - tq_full_attn_per_block
# But we can't just add more blocks -- the linear_attn state per block stays the same
# New blocks would need: LINEAR_PER_BLOCK + tq_full_attn_per_block bytes each
new_block_cost = LINEAR_PER_BLOCK + tq_full_attn_per_block

# Total memory currently used for KV: NUM_GPU_BLOCKS * TOTAL_PER_BLOCK
# With TQ, each block costs: LINEAR_PER_BLOCK + tq_full_attn_per_block
# So we can fit: total_memory / new_block_cost blocks
total_kv_memory = NUM_GPU_BLOCKS * TOTAL_PER_BLOCK
new_num_blocks = total_kv_memory // new_block_cost
new_capacity = new_num_blocks * BLOCK_SIZE

print(f"Current KV cache budget: {total_kv_memory/1e9:.2f} GB")
print(f"Current block cost: {TOTAL_PER_BLOCK/1e6:.2f} MB (full_attn: {FULL_ATTN_PER_BLOCK/1e6:.2f} + linear: {LINEAR_PER_BLOCK/1e6:.2f})")
print(f"TQ block cost:      {new_block_cost/1e6:.2f} MB (tq_full_attn: {tq_full_attn_per_block/1e6:.2f} + linear: {LINEAR_PER_BLOCK/1e6:.2f})")
print()
print(f"Current: {NUM_GPU_BLOCKS} blocks = {NUM_GPU_BLOCKS * BLOCK_SIZE:,} tokens")
print(f"With TQ: {new_num_blocks} blocks = {new_capacity:,} tokens")
print(f"Context extension: {new_capacity / (NUM_GPU_BLOCKS * BLOCK_SIZE):.2f}x")
print()

# What about a model with NO linear attention (pure transformer)?
print("--- For comparison: if this were a PURE transformer (no linear attention) ---")
pure_block_cost_baseline = FULL_ATTN_PER_BLOCK  # no linear state
pure_block_cost_tq = tq_full_attn_per_block
pure_blocks_baseline = total_kv_memory // pure_block_cost_baseline
pure_blocks_tq = total_kv_memory // pure_block_cost_tq
print(f"Pure transformer baseline: {pure_blocks_baseline} blocks = {pure_blocks_baseline * BLOCK_SIZE:,} tokens")
print(f"Pure transformer with TQ:  {pure_blocks_tq} blocks = {pure_blocks_tq * BLOCK_SIZE:,} tokens")
print(f"Context extension:         {pure_blocks_tq / pure_blocks_baseline:.2f}x")

print()
print("=" * 80)
print("  ACTUAL VRAM BREAKDOWN (per GPU)")
print("=" * 80)
print()

# nvidia-smi shows ~22.3 GB used per GPU
# gpu_memory_utilization=0.92 means 24576 * 0.92 = 22610 MB reserved
# Model weights: 54 GB / 8 = 6.75 GB per GPU
# KV cache pool: total_kv_memory / 8
# CUDA overhead + graphs: remainder

model_weights_per_gpu = 54000 / TP_SIZE  # approximate from disk size
kv_cache_per_gpu = total_kv_memory / TP_SIZE
reserved_per_gpu = 24576 * 0.92
overhead = reserved_per_gpu - model_weights_per_gpu - kv_cache_per_gpu / 1e6

print(f"Total VRAM per GPU: 24,576 MB")
print(f"Reserved (0.92):    {reserved_per_gpu:.0f} MB")
print(f"Model weights:      ~{model_weights_per_gpu:.0f} MB")
print(f"KV cache pool:      {kv_cache_per_gpu/1e6:.0f} MB")
print(f"  - full_attn:      {full_attn_total/TP_SIZE/1e6:.0f} MB")
print(f"  - linear state:   {linear_total/TP_SIZE/1e6:.0f} MB")
print(f"CUDA overhead:      ~{overhead:.0f} MB")
print()

# With TQ:
tq_kv_per_gpu = (new_num_blocks * new_block_cost) / TP_SIZE
tq_overhead = TQ_OVERHEAD_TOTAL / TP_SIZE
print(f"WITH TQ:")
print(f"KV cache pool:      {tq_kv_per_gpu/1e6:.0f} MB (same budget, more tokens)")
print(f"  - tq_full_attn:   {new_num_blocks * tq_full_attn_per_block / TP_SIZE / 1e6:.0f} MB")
print(f"  - linear state:   {new_num_blocks * LINEAR_PER_BLOCK / TP_SIZE / 1e6:.0f} MB")
print(f"TQ overhead (Pi+S): {tq_overhead/1e6:.1f} MB")
print(f"VRAM freed:         {(kv_cache_per_gpu - tq_kv_per_gpu)/1e6:.0f} MB/GPU (if not reused for more tokens)")
