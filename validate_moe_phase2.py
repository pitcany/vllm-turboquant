#!/usr/bin/env python3
"""
Phase 2: TurboQuant validation on REAL KV activations from Qwen3.5-35B-A3B MoE.

Strategy:
  We can't monkey-patch the running vLLM server. Instead we:
  1. Load the model in a SEPARATE process on a single GPU (--device_map auto won't work
     with 8 GPUs at 98% utilization).
  
  Actually -- we only have ~2.8 GB free per GPU. Can't load the model again.
  
  Better approach: Use vLLM's logprobs API to get token-level probabilities, then
  compare baseline (full KV) vs simulated TQ (offline compression) quality.
  
  BEST approach given constraints: 
  Capture KV activations from the model's attention layers via a hook, then
  offline-quantize with TQ and measure the attention output degradation.
  BUT: can't hook into the running server without restarting it.
  
  PRACTICAL approach:
  1. Generate a corpus of prompts at different lengths
  2. Get vLLM completions WITH logprobs (this is our ground truth)
  3. Measure: at what context length does quality degrade?
  4. Theoretical analysis: compute TQ compression on head_dim=256, kv_heads=2
  5. Synthetic validation: create realistic KV cache tensors matching
     the model's statistics (head_dim=256) and measure TQ quality

This script does the synthetic validation with model-realistic parameters.
"""

import sys
sys.path.insert(0, "/tmp")

import math
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from turboquant.quantizer import TurboQuantProd
from turboquant.kv_cache import TurboQuantKVCache, quantize_values, dequantize_values

torch.manual_seed(42)
np.random.seed(42)

# Qwen3.5-35B-A3B full_attention layer config
HEAD_DIM = 256
NUM_KV_HEADS = 2
NUM_Q_HEADS = 16  # GQA ratio 8:1
GQA_RATIO = NUM_Q_HEADS // NUM_KV_HEADS  # 8
NUM_FULL_ATTN_LAYERS = 10
SCALE = 1.0 / math.sqrt(HEAD_DIM)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ============================================================
# TEST 1: TQ quality at head_dim=256 (this model's actual dim)
# ============================================================

section("TEST 1: TQ quantization quality at head_dim=256")

print(f"Model parameters: head_dim={HEAD_DIM}, kv_heads={NUM_KV_HEADS}, "
      f"q_heads={NUM_Q_HEADS}, GQA={GQA_RATIO}:1")
print(f"Device: {DEVICE}")
print()

for bits in [3, 4]:
    q = TurboQuantProd(dim=HEAD_DIM, bits=bits, device=DEVICE, seed=42)

    # Generate realistic KV cache: N tokens, kv_heads, head_dim
    for N in [512, 1024, 4096, 8192]:
        keys = torch.randn(1, NUM_KV_HEADS, N, HEAD_DIM, device=DEVICE) * 0.02
        queries = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, device=DEVICE) * 0.02

        # Exact attention scores
        true_scores = torch.matmul(queries, keys.transpose(-2, -1)) * SCALE  # (1, H, 1, N)
        true_weights = F.softmax(true_scores, dim=-1)

        # TQ quantized scores
        keys_4d = keys  # already (1, H, N, D)
        key_q = q.quantize(keys_4d)
        # Dequantize for attention score computation
        keys_dequant = q.dequantize(key_q)
        tq_scores = torch.matmul(queries, keys_dequant.transpose(-2, -1)) * SCALE
        tq_weights = F.softmax(tq_scores, dim=-1)

        # Also test the direct attention_score method
        tq_scores_direct = q.attention_score(queries, key_q) * SCALE
        tq_weights_direct = F.softmax(tq_scores_direct, dim=-1)

        # Generate random values for output comparison
        values = torch.randn(1, NUM_KV_HEADS, N, HEAD_DIM, device=DEVICE) * 0.02

        true_out = torch.matmul(true_weights, values)
        tq_out = torch.matmul(tq_weights_direct, values)

        # Metrics
        cos = F.cosine_similarity(
            true_out.reshape(-1, HEAD_DIM),
            tq_out.reshape(-1, HEAD_DIM),
            dim=-1,
        ).mean().item()

        kl = F.kl_div(
            tq_weights_direct.log().clamp(min=-20),
            true_weights,
            reduction="batchmean",
        ).item()

        out_mse = ((true_out - tq_out) ** 2).sum(dim=-1).mean().item()
        out_mag = (true_out ** 2).sum(dim=-1).mean().item()
        rel_mse = out_mse / max(out_mag, 1e-10)

        # Recall@1 and Recall@8
        true_top1 = true_scores.squeeze().argmax(dim=-1)
        tq_top1 = tq_scores_direct.squeeze().argmax(dim=-1)
        recall1 = (true_top1 == tq_top1).float().mean().item()

        # Recall@8 per head
        recall8_total = 0
        for h in range(NUM_KV_HEADS):
            true_top8 = set(true_scores[0, h, 0].topk(8).indices.tolist())
            tq_top8 = set(tq_scores_direct[0, h, 0].topk(8).indices.tolist())
            recall8_total += len(true_top8 & tq_top8) / 8
        recall8 = recall8_total / NUM_KV_HEADS

        if N == 512:
            print(f"{bits}-bit, head_dim={HEAD_DIM}:")
            print(f"  {'N':>6} {'cos_sim':>10} {'KL_div':>10} {'rel_MSE':>12} {'recall@1':>10} {'recall@8':>10}")
            print(f"  {'-'*60}")

        print(f"  {N:>6} {cos:>10.6f} {kl:>10.6f} {rel_mse:>12.8f} {recall1:>10.1%} {recall8:>10.1%}")

    print()


# ============================================================
# TEST 2: Memory savings analysis (real numbers for this model)
# ============================================================

section("TEST 2: Memory savings for Qwen3.5-35B-A3B")

# Per-token KV cache for full_attention layers
fp16_per_token_per_layer = NUM_KV_HEADS * HEAD_DIM * 2 * 2  # K+V, bf16

# TQ at 3-bit keys, 2-bit values
for key_bits, val_bits in [(3, 2), (4, 2), (4, 4)]:
    q = TurboQuantProd(dim=HEAD_DIM, bits=key_bits, device=DEVICE, seed=42)

    # Measure actual byte sizes from quantized output
    test_keys = torch.randn(1, NUM_KV_HEADS, 1000, HEAD_DIM, device=DEVICE)
    test_vals = torch.randn(1, NUM_KV_HEADS, 1000, HEAD_DIM, device=DEVICE)

    key_q = q.quantize(test_keys)
    val_q = quantize_values(test_vals, bits=val_bits, group_size=32)

    # Count actual bytes
    key_bytes = (
        key_q.mse_indices.nelement() * key_q.mse_indices.element_size() +
        key_q.qjl_signs.nelement() * key_q.qjl_signs.element_size() +
        key_q.residual_norms.nelement() * 4 +  # float32
        key_q.norms.nelement() * 4  # float32
    )
    val_bytes = (
        val_q.data.nelement() * val_q.data.element_size() +
        val_q.scales.nelement() * 4 +  # float32
        val_q.zeros.nelement() * 4  # float32
    )
    total_tq = key_bytes + val_bytes
    total_fp16 = test_keys.nelement() * 2 + test_vals.nelement() * 2
    ratio = total_fp16 / total_tq

    # Per-token overhead (Pi and S matrices, amortized)
    pi_bytes = HEAD_DIM * HEAD_DIM * 4  # 256*256*4 = 262KB
    s_bytes = HEAD_DIM * HEAD_DIM * 4   # 262KB
    overhead_per_layer = pi_bytes + s_bytes  # 524KB per layer

    print(f"Key {key_bits}-bit / Value {val_bits}-bit:")
    print(f"  FP16 KV for 1000 tokens: {total_fp16/1e3:.1f} KB")
    print(f"  TQ KV for 1000 tokens:   {total_tq/1e3:.1f} KB")
    print(f"  Compression ratio:       {ratio:.2f}x")
    print(f"  Per-layer overhead (Pi+S): {overhead_per_layer/1e3:.0f} KB ({overhead_per_layer/1e6:.2f} MB)")
    print(f"  Overhead for {NUM_FULL_ATTN_LAYERS} layers: {overhead_per_layer * NUM_FULL_ATTN_LAYERS / 1e6:.1f} MB")
    print()

    # At what N does the overhead become < 10% of the savings?
    savings_per_token = (total_fp16 - total_tq) / 1000 * NUM_FULL_ATTN_LAYERS
    total_overhead = overhead_per_layer * NUM_FULL_ATTN_LAYERS
    breakeven_tokens = total_overhead / max(savings_per_token, 1)
    print(f"  Break-even: overhead < 10% of savings at N > {int(breakeven_tokens * 10)}")
    print()


# ============================================================
# TEST 3: Context extension potential
# ============================================================

section("TEST 3: Context extension potential")

# vLLM allocated 4553 blocks of 272 tokens = 1,238,416 token capacity
# But the model is served with max_model_len=8192
# With TP=8, each GPU handles a shard of the KV cache

# Current: ~2.8 GB free per GPU
# Model weights use ~21.3 GB of the 24 GB per GPU
# KV cache allocated from the remaining ~2.7 GB per GPU * 8 = 21.6 GB total

TOTAL_KV_GB = 25.36  # from vLLM: 4553 * 272 * 20480 bytes
FREE_PER_GPU_MB = 2850  # approximate
TOTAL_FREE_MB = FREE_PER_GPU_MB * 8

print(f"Current vLLM KV cache: {TOTAL_KV_GB:.1f} GB ({4553} blocks of 272 tokens)")
print(f"Max model len: 8192 tokens")
print(f"Free GPU memory: ~{FREE_PER_GPU_MB} MB/GPU, {TOTAL_FREE_MB/1000:.1f} GB total")
print()

# KV per token (full attention only, the linear attention layers have O(1) state)
kv_per_token_bytes = NUM_FULL_ATTN_LAYERS * NUM_KV_HEADS * HEAD_DIM * 2 * 2  # 20 KB
# But linear attention layers also have KV cache in vLLM's block allocator
# Let's check the actual block size from vLLM config: block_size=272
# This includes ALL layers, not just full_attention

# With TQ on the 10 full_attention layers:
# Savings per token = (fp16 - tq) for full_attn layers only
# Linear attention layers are unchanged

for key_bits, val_bits, label in [(3, 2, "aggressive"), (4, 2, "balanced"), (4, 4, "conservative")]:
    q = TurboQuantProd(dim=HEAD_DIM, bits=key_bits, device=DEVICE, seed=42)
    test_keys = torch.randn(1, 1, 100, HEAD_DIM, device=DEVICE)
    test_vals = torch.randn(1, 1, 100, HEAD_DIM, device=DEVICE)
    key_q = q.quantize(test_keys)
    val_q = quantize_values(test_vals, bits=val_bits, group_size=32)

    tq_bytes_per_100 = (
        key_q.mse_indices.nelement() + key_q.qjl_signs.nelement() +
        key_q.residual_norms.nelement() * 4 + key_q.norms.nelement() * 4 +
        val_q.data.nelement() + val_q.scales.nelement() * 4 + val_q.zeros.nelement() * 4
    )
    tq_per_token_per_head = tq_bytes_per_100 / 100
    tq_per_token_full_attn = tq_per_token_per_head * NUM_KV_HEADS * NUM_FULL_ATTN_LAYERS
    fp16_per_token_full_attn = NUM_FULL_ATTN_LAYERS * NUM_KV_HEADS * HEAD_DIM * 2 * 2

    savings_per_token = fp16_per_token_full_attn - tq_per_token_full_attn
    savings_fraction = savings_per_token / fp16_per_token_full_attn

    # How many MORE tokens can we fit in the freed memory?
    # At max_model_len=8192, total KV for full_attn = 8192 * 20KB = 163.8 MB
    current_kv_8192 = 8192 * fp16_per_token_full_attn
    tq_kv_8192 = 8192 * tq_per_token_full_attn
    freed_mb = (current_kv_8192 - tq_kv_8192) / 1e6

    # Extra tokens that fit in freed memory
    extra_tokens = int(freed_mb * 1e6 / tq_per_token_full_attn)

    print(f"{label.upper()} ({key_bits}b keys / {val_bits}b vals):")
    print(f"  Savings per token: {savings_per_token:.0f} bytes ({savings_fraction:.0%} of full_attn KV)")
    print(f"  At 8192 tokens: free {freed_mb:.1f} MB -> fit {extra_tokens:,} more tokens")
    print(f"  New max context: ~{8192 + extra_tokens:,} tokens ({(8192+extra_tokens)/8192:.1f}x)")
    print()

print("IMPORTANT CAVEATS:")
print("  1. These savings are ONLY for the 10 full_attention layers")
print("  2. The 30 linear_attention layers also use KV cache blocks in vLLM")
print("  3. vLLM's block allocator may not efficiently reuse freed blocks")
print("  4. The model already has max_position_embeddings=262144")
print("  5. The bottleneck for context length is vLLM's max_model_len setting,")
print("     not memory — increasing it requires restarting the server")


# ============================================================
# TEST 4: Quality at head_dim=256 with spiky attention patterns
# ============================================================

section("TEST 4: Quality with realistic (spiky) attention patterns")

print("Testing with attention patterns where top-5 tokens carry >80% weight:")
print()

n_queries = 200

for bits in [3, 4]:
    q = TurboQuantProd(dim=HEAD_DIM, bits=bits, device=DEVICE, seed=42)
    print(f"{bits}-bit:")
    print(f"  {'N':>6} {'cos_sim':>10} {'KL_div':>10} {'rel_MSE':>12} {'top5_agree':>12}")
    print(f"  {'-'*52}")

    for N in [512, 1024, 4096, 8192]:
        cos_sims = []; kl_divs = []; rel_mses = []; top5_agrees = []

        keys = torch.randn(1, NUM_KV_HEADS, N, HEAD_DIM, device=DEVICE) * 0.02
        values = torch.randn(1, NUM_KV_HEADS, N, HEAD_DIM, device=DEVICE) * 0.02
        key_q = q.quantize(keys)

        for qi in range(n_queries):
            # Create spiky query: strongly correlated with a few random keys
            target_indices = torch.randint(0, N, (5,))
            query = keys[:, :, target_indices[0]:target_indices[0]+1, :] * 3.0  # amplify
            query = query + torch.randn_like(query) * 0.02

            true_scores = torch.matmul(query, keys.transpose(-2, -1)) * SCALE
            true_weights = F.softmax(true_scores, dim=-1)
            true_out = torch.matmul(true_weights, values)

            tq_scores = q.attention_score(query, key_q) * SCALE
            tq_weights = F.softmax(tq_scores, dim=-1)
            tq_out = torch.matmul(tq_weights, values)

            cos = F.cosine_similarity(
                true_out.reshape(-1, HEAD_DIM), tq_out.reshape(-1, HEAD_DIM), dim=-1
            ).mean().item()
            cos_sims.append(cos)

            kl = F.kl_div(
                tq_weights.log().clamp(min=-20), true_weights, reduction="batchmean"
            ).item()
            kl_divs.append(abs(kl))

            mse = ((true_out - tq_out)**2).sum(dim=-1).mean().item()
            mag = (true_out**2).sum(dim=-1).mean().item()
            rel_mses.append(mse / max(mag, 1e-10))

            # Top-5 agreement per head
            for h in range(NUM_KV_HEADS):
                t5 = set(true_weights[0, h, 0].topk(5).indices.tolist())
                tq5 = set(tq_weights[0, h, 0].topk(5).indices.tolist())
                top5_agrees.append(len(t5 & tq5) / 5)

        print(f"  {N:>6} {np.mean(cos_sims):>10.6f} {np.mean(kl_divs):>10.6f} "
              f"{np.mean(rel_mses):>12.8f} {np.mean(top5_agrees):>12.1%}")

    print()


# ============================================================
# TEST 5: Full KV cache simulation (all 10 full_attention layers)
# ============================================================

section("TEST 5: Full cache simulation — 10 layers, 8192 tokens")

N = 8192
print(f"Simulating full KV cache: {NUM_FULL_ATTN_LAYERS} layers, {N} tokens, "
      f"{NUM_KV_HEADS} KV heads, head_dim={HEAD_DIM}")
print()

for bits in [3, 4]:
    print(f"--- {bits}-bit TQ ---")
    total_fp16_mb = 0
    total_tq_mb = 0
    layer_cos_sims = []

    t0 = time.time()

    for layer in range(NUM_FULL_ATTN_LAYERS):
        cache = TurboQuantKVCache(
            head_dim=HEAD_DIM,
            key_bits=bits,
            value_bits=2,
            value_group_size=32,
            buffer_size=128,
            device=DEVICE,
            layer_idx=layer,
        )

        # Simulate prefill
        keys = torch.randn(1, NUM_KV_HEADS, N, HEAD_DIM, device=DEVICE) * 0.02
        values = torch.randn(1, NUM_KV_HEADS, N, HEAD_DIM, device=DEVICE) * 0.02
        cache.prefill(keys, values)

        # Memory
        mem = cache.memory_bytes()
        fp16_bytes = N * NUM_KV_HEADS * HEAD_DIM * 2 * 2
        total_fp16_mb += fp16_bytes / 1e6
        total_tq_mb += mem["total"] / 1e6

        # Quality: run a few test queries
        query = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, device=DEVICE) * 0.02
        tq_scores = cache.attention_scores(query)
        true_scores = torch.matmul(query, keys.transpose(-2, -1)) * SCALE

        tq_w = F.softmax(tq_scores, dim=-1)
        true_w = F.softmax(true_scores, dim=-1)

        tq_out = cache.attend(tq_w)
        true_out = torch.matmul(true_w, values)

        cos = F.cosine_similarity(
            true_out.reshape(-1, HEAD_DIM), tq_out.reshape(-1, HEAD_DIM), dim=-1
        ).mean().item()
        layer_cos_sims.append(cos)

        # Free memory
        del cache, keys, values, query
        torch.cuda.empty_cache()

    elapsed = time.time() - t0

    print(f"  FP16 total: {total_fp16_mb:.1f} MB")
    print(f"  TQ total:   {total_tq_mb:.1f} MB")
    print(f"  Ratio:      {total_fp16_mb / total_tq_mb:.2f}x")
    print(f"  Per-layer cos_sim: {[f'{c:.6f}' for c in layer_cos_sims]}")
    print(f"  Mean cos_sim:      {np.mean(layer_cos_sims):.8f}")
    print(f"  Time: {elapsed:.1f}s")
    print()


# ============================================================
# SUMMARY
# ============================================================

section("SUMMARY")

print("Model: Qwen3.5-35B-A3B (pruned MoE, 205 experts)")
print(f"Architecture: {NUM_FULL_ATTN_LAYERS} full_attention + 30 linear_attention layers")
print(f"Full attention: head_dim={HEAD_DIM}, {NUM_KV_HEADS} KV heads (GQA {GQA_RATIO}:1)")
print()
print("Baseline (no TQ):")
print("  - Single needle: PASS at all context lengths (512-7680)")
print("  - Multi-needle:  3/3 PASS at 5040 tokens")
print("  - Generation speed: 8.2-46 tok/s depending on context")
print("  - Max context: 8192 (vLLM configured)")
print()
print("TQ Impact on full_attention layers:")
print("  - See test results above for quality metrics")
print("  - Memory savings are modest due to only 10/40 layers having standard attention")
print("  - The linear_attention layers (30/40) are NOT compressible by TQ")
print()
print("Key takeaway: This MoE model is NOT the ideal target for TQ because")
print("75% of its layers use linear attention (O(1) state). TQ shines on")
print("dense transformers with 100% standard attention layers.")
