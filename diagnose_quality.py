#!/usr/bin/env python3
"""Diagnose: is cos_sim=0.93 from key or value quantization?"""
import sys; sys.path.insert(0, "/tmp")
import torch, torch.nn.functional as F, math
from turboquant.quantizer import TurboQuantProd
from turboquant.kv_cache import quantize_values, dequantize_values

torch.manual_seed(42)
D=256; H=2; N=8192; SCALE=1.0/math.sqrt(D); dev="cuda:0"

keys = torch.randn(1, H, N, D, device=dev) * 0.02
values = torch.randn(1, H, N, D, device=dev) * 0.02
query = torch.randn(1, H, 1, D, device=dev) * 0.02

true_scores = torch.matmul(query, keys.transpose(-2, -1)) * SCALE
true_w = F.softmax(true_scores, dim=-1)
true_out = torch.matmul(true_w, values)

# Case 1: TQ keys, exact values
q = TurboQuantProd(dim=D, bits=3, device=dev, seed=42)
key_q = q.quantize(keys)
tq_scores = q.attention_score(query, key_q) * SCALE
tq_w = F.softmax(tq_scores, dim=-1)
tq_out_exact_v = torch.matmul(tq_w, values)
cos1 = F.cosine_similarity(true_out.reshape(-1, D), tq_out_exact_v.reshape(-1, D), dim=-1).mean().item()

# Case 2: Exact keys, 2-bit values
val_q = quantize_values(values, bits=2, group_size=32)
v_dequant = dequantize_values(val_q, group_size=32)
exact_out_tq_v = torch.matmul(true_w, v_dequant)
cos2 = F.cosine_similarity(true_out.reshape(-1, D), exact_out_tq_v.reshape(-1, D), dim=-1).mean().item()

# Case 3: TQ keys + 2-bit values
tq_out_both = torch.matmul(tq_w, v_dequant)
cos3 = F.cosine_similarity(true_out.reshape(-1, D), tq_out_both.reshape(-1, D), dim=-1).mean().item()

# Case 4: Exact keys, 4-bit values
val_q4 = quantize_values(values, bits=4, group_size=32)
v_dequant4 = dequantize_values(val_q4, group_size=32)
exact_out_4v = torch.matmul(true_w, v_dequant4)
cos4 = F.cosine_similarity(true_out.reshape(-1, D), exact_out_4v.reshape(-1, D), dim=-1).mean().item()

# Value reconstruction quality
v_cos = F.cosine_similarity(values.reshape(-1, D), v_dequant.reshape(-1, D), dim=-1).mean().item()
v4_cos = F.cosine_similarity(values.reshape(-1, D), v_dequant4.reshape(-1, D), dim=-1).mean().item()

print("DIAGNOSIS: What causes cos_sim drop to 0.93?")
print(f"  TQ keys + exact values:   cos={cos1:.6f}  -- key quant only")
print(f"  Exact keys + 2b values:   cos={cos2:.6f}  -- value quant only")
print(f"  TQ keys + 2b values:      cos={cos3:.6f}  -- both")
print(f"  Exact keys + 4b values:   cos={cos4:.6f}  -- value quant 4-bit")
print()
print("Value vector reconstruction:")
print(f"  2-bit cos_sim: {v_cos:.6f}")
print(f"  4-bit cos_sim: {v4_cos:.6f}")
print()
if cos2 < cos1:
    print("VERDICT: The quality drop is from 2-bit VALUE quantization, not TQ key compression.")
else:
    print("VERDICT: The quality drop is from TQ key compression.")
