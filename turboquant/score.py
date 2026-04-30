"""
TurboQuant score module — attention computation over compressed + exact segments.

Handles the read path:
  - Combine attention over up to three KV segments via streaming online softmax:
      * kv_paged — K/V for tokens not yet in kv_tq (e.g. the current decode
        token at hybrid forward time, whose K/V is in scope as the forward
        ``key``/``value`` arguments but won't reach kv_tq until after the
        post-execute paged-cache reader fires for *this* step).
      * kv_tq    — dequantised compressed historical store.
      * kv_ring  — exact recent tokens kept in the ring buffer.
  - Pure-PyTorch online softmax (Milakov & Gimelshein 2018) is used so the
    attention output is mathematically equivalent to one big softmax over
    the concatenation of all three segments. This is the F3 fix landed in
    Path B Sprint 3 / S3.2 (see docs/integration-state.md § "F3 …").

Design rule: compressed path is only invoked when history is large enough
to justify it (>= MIN_HISTORY_FOR_TQ tokens).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch

from turboquant.kv_cache import dequantize_values
from turboquant.store import CompressedKVStore

logger = logging.getLogger("turboquant.score")

MIN_HISTORY_FOR_TQ = 16


def compute_hybrid_attention(
    query: torch.Tensor,
    store: CompressedKVStore,
    recent_k: Optional[torch.Tensor],
    recent_v: Optional[torch.Tensor],
    num_query_heads: int,
    scale: Optional[float] = None,
    kv_paged_k: Optional[torch.Tensor] = None,
    kv_paged_v: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute attention output combining up to three KV segments.

    Args:
        query: (num_tokens, num_query_heads, head_dim) — typically num_tokens=1
            for decode.
        store: compressed KV store with historical tokens (kv_tq).
        recent_k: (recent_len, num_kv_heads, head_dim) or None (kv_ring).
        recent_v: (recent_len, num_kv_heads, head_dim) or None (kv_ring).
        num_query_heads: total query heads (for GQA expansion).
        scale: attention scale factor (default: 1/sqrt(head_dim)).
        kv_paged_k: (num_paged, num_kv_heads, head_dim) or None — K for
            tokens not yet captured into kv_tq. At hybrid forward time this
            is the current decode token's K, sourced from the forward
            ``key`` argument (see docs/integration-state.md § "S3.1 — Audit").
        kv_paged_v: (num_paged, num_kv_heads, head_dim) or None — V for the
            same positions as ``kv_paged_k``.

    Returns:
        output: (num_tokens, num_query_heads, head_dim) in query.dtype.
    """
    head_dim = store.head_dim
    num_kv_heads = store.num_kv_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    flat = store.get_flat_cache()
    has_tq = flat is not None and flat.num_tokens >= MIN_HISTORY_FOR_TQ
    has_ring = recent_k is not None and recent_k.shape[0] > 0
    has_paged = kv_paged_k is not None and kv_paged_k.shape[0] > 0

    if not (has_tq or has_ring or has_paged):
        return torch.zeros(
            query.shape[0],
            num_query_heads,
            head_dim,
            device=query.device,
            dtype=query.dtype,
        )

    gqa_ratio = num_query_heads // num_kv_heads

    # Materialise each non-empty segment as (num_kv_heads, N_seg, head_dim)
    # in float32 — matches the existing _matmul_attend accumulation
    # precision and keeps the online-softmax math numerically stable.
    segments: list[tuple[torch.Tensor, torch.Tensor]] = []
    if has_tq:
        k_tq = store.quantizer.dequantize(flat.prod_q).float()
        v_tq = dequantize_values(flat.value_q, 32).float()
        segments.append((k_tq, v_tq))
    if has_ring:
        segments.append((recent_k.transpose(0, 1).float(), recent_v.transpose(0, 1).float()))
    if has_paged:
        segments.append((kv_paged_k.transpose(0, 1).float(), kv_paged_v.transpose(0, 1).float()))

    return _attend_online_softmax(query, segments, gqa_ratio, num_kv_heads, scale)


def _attend_online_softmax(
    query: torch.Tensor,
    segments: list[tuple[torch.Tensor, torch.Tensor]],
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Streaming online softmax over an arbitrary list of non-empty KV segments.

    Reference: Milakov & Gimelshein, "Online normalizer calculation for
    softmax" (2018). Folds each segment into running (m, denom, acc) state so
    the result is mathematically identical to a single softmax over the
    concatenation of all segment K/V — without ever materialising that
    concatenation in memory.

    Args:
        query: (T, Q, D) where Q = num_kv_heads * gqa_ratio.
        segments: list of (keys, values), each (num_kv_heads, N_i, D), float32.
            Caller must filter out empty segments; this helper assumes
            len(segments) >= 1 and N_i >= 1 for every segment.
        gqa_ratio: query-heads-per-kv-head broadcast factor.
        num_kv_heads: number of KV heads.
        scale: attention scale (typically 1/sqrt(head_dim)).

    Returns:
        (T, Q, D) in query.dtype.
    """
    T, Q, D = query.shape
    H_kv = num_kv_heads
    if Q != H_kv * gqa_ratio:
        raise ValueError(f"Incompatible GQA shapes: Q={Q}, H_kv={H_kv}, gqa_ratio={gqa_ratio}")

    # (H_kv, G, T, D) — same layout the existing _matmul_attend uses, just
    # accumulated across segments.
    q = query.float().view(T, H_kv, gqa_ratio, D).permute(1, 2, 0, 3).contiguous()

    # Running stats: m (running max logit), denom (running normaliser), acc
    # (running unnormalised numerator). Shapes (H_kv, G, T, 1) and
    # (H_kv, G, T, D). All in fp32.
    m = torch.full((H_kv, gqa_ratio, T, 1), float("-inf"), device=q.device, dtype=torch.float32)
    denom = torch.zeros((H_kv, gqa_ratio, T, 1), device=q.device, dtype=torch.float32)
    acc = torch.zeros((H_kv, gqa_ratio, T, D), device=q.device, dtype=torch.float32)

    for k_seg, v_seg in segments:
        # Broadcast (H_kv, 1, N, D) over the GQA group dim.
        k = k_seg.unsqueeze(1)
        v = v_seg.unsqueeze(1)
        scores = torch.einsum("hgtd,hgnd->hgtn", q, k) * scale  # (H_kv, G, T, N)
        m_seg = scores.amax(dim=-1, keepdim=True)
        m_new = torch.maximum(m, m_seg)
        # exp(-inf - finite) = 0, which is exactly what initialises the
        # accumulator on the first segment without any branch on m == -inf.
        alpha = torch.exp(m - m_new)
        p = torch.exp(scores - m_new)
        denom = denom * alpha + p.sum(dim=-1, keepdim=True)
        acc = acc * alpha + torch.einsum("hgtn,hgnd->hgtd", p, v)
        m = m_new

    out = acc / denom
    return out.permute(2, 0, 1, 3).reshape(T, Q, D).to(query.dtype)
