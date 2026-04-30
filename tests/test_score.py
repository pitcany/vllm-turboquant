"""Unit tests for compute_hybrid_attention's online softmax (Sprint 3 / S3.2).

The F3 fix relies on a single property: streaming softmax over a list of
KV segments must be mathematically equivalent to a single softmax over the
concatenation of those segments. If that property holds, then attending
across kv_paged + kv_tq + kv_ring three-way is identical to attending over
their concatenation, which is identical to standard attention over full
history. These tests verify that property at the pure-PyTorch level on
CPU so the correctness bar doesn't depend on a CUDA box being available.
"""

from __future__ import annotations

import math

import pytest
import torch


@pytest.mark.unit
def test_online_softmax_two_segments_match_concat() -> None:
    """Folding two segments via online softmax == concatenated single softmax."""
    from turboquant.score import _attend_online_softmax

    torch.manual_seed(0)
    T, H_kv, G, D = 1, 4, 2, 16
    Q = H_kv * G
    scale = 1.0 / math.sqrt(D)

    query = torch.randn(T, Q, D)
    k1 = torch.randn(H_kv, 8, D)
    v1 = torch.randn(H_kv, 8, D)
    k2 = torch.randn(H_kv, 32, D)
    v2 = torch.randn(H_kv, 32, D)

    online = _attend_online_softmax(query, [(k1, v1), (k2, v2)], G, H_kv, scale)
    baseline = _attend_online_softmax(
        query,
        [(torch.cat([k1, k2], dim=1), torch.cat([v1, v2], dim=1))],
        G,
        H_kv,
        scale,
    )

    torch.testing.assert_close(online, baseline, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_online_softmax_three_segments_match_concat() -> None:
    """The kv_paged + kv_tq + kv_ring shape that S3.2 actually exercises."""
    from turboquant.score import _attend_online_softmax

    torch.manual_seed(1)
    T, H_kv, G, D = 1, 8, 3, 32
    Q = H_kv * G
    scale = 1.0 / math.sqrt(D)

    query = torch.randn(T, Q, D)
    # Sized to mimic decode step ordering: kv_tq (large historical), kv_ring
    # (small recent), kv_paged (single current decode token).
    keys = [torch.randn(H_kv, n, D) for n in (256, 16, 1)]
    vals = [torch.randn(H_kv, n, D) for n in (256, 16, 1)]
    segs = list(zip(keys, vals))

    online = _attend_online_softmax(query, segs, G, H_kv, scale)
    baseline = _attend_online_softmax(
        query,
        [(torch.cat(keys, dim=1), torch.cat(vals, dim=1))],
        G,
        H_kv,
        scale,
    )

    torch.testing.assert_close(online, baseline, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_online_softmax_single_segment_matches_standard_softmax() -> None:
    """Sanity: with one segment the helper reduces to plain softmax-attn."""
    from turboquant.score import _attend_online_softmax

    torch.manual_seed(2)
    T, H_kv, G, D = 1, 4, 1, 8  # no GQA, one head group
    Q = H_kv * G
    N = 40
    scale = 1.0 / math.sqrt(D)

    query = torch.randn(T, Q, D)
    k = torch.randn(H_kv, N, D)
    v = torch.randn(H_kv, N, D)

    online = _attend_online_softmax(query, [(k, v)], G, H_kv, scale)

    # Reference: write standard attention longhand with the same axis
    # conventions the helper uses internally.
    q = query.float().view(T, H_kv, G, D).permute(1, 2, 0, 3)  # (H, G, T, D)
    k_b = k.unsqueeze(1)
    v_b = v.unsqueeze(1)
    scores = torch.einsum("hgtd,hgnd->hgtn", q, k_b) * scale
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("hgtn,hgnd->hgtd", weights, v_b).permute(2, 0, 1, 3).reshape(T, Q, D)

    torch.testing.assert_close(online, out, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_online_softmax_order_invariance() -> None:
    """Result is permutation-invariant over the segment list (softmax is set-wise)."""
    from turboquant.score import _attend_online_softmax

    torch.manual_seed(3)
    T, H_kv, G, D = 1, 2, 4, 16
    Q = H_kv * G
    scale = 1.0 / math.sqrt(D)

    query = torch.randn(T, Q, D)
    segs = [(torch.randn(H_kv, n, D), torch.randn(H_kv, n, D)) for n in (3, 17, 1, 23)]

    forward = _attend_online_softmax(query, segs, G, H_kv, scale)
    reverse = _attend_online_softmax(query, list(reversed(segs)), G, H_kv, scale)

    torch.testing.assert_close(forward, reverse, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_compute_hybrid_attention_with_paged_segment_matches_concat() -> None:
    """End-to-end: kv_paged + kv_ring (no quantised history) routes through
    the new path and matches a flat concatenation reference. This exercises
    the dispatch in compute_hybrid_attention without standing up a real
    quantised store (which needs CUDA codebooks at the right shapes).
    """
    from turboquant.score import _attend_online_softmax, compute_hybrid_attention
    from turboquant.store import CompressedKVStore

    torch.manual_seed(4)
    head_dim, num_kv_heads, num_query_heads = 64, 2, 4
    gqa_ratio = num_query_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    # Empty store — has_tq is False, so only kv_ring + kv_paged contribute.
    store = CompressedKVStore(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        key_bits=3,
        value_bits=2,
        value_group_size=32,
        device=torch.device("cpu"),
        layer_idx=0,
    )

    T = 1
    query = torch.randn(T, num_query_heads, head_dim)
    recent_k = torch.randn(8, num_kv_heads, head_dim)
    recent_v = torch.randn(8, num_kv_heads, head_dim)
    paged_k = torch.randn(1, num_kv_heads, head_dim)
    paged_v = torch.randn(1, num_kv_heads, head_dim)

    out = compute_hybrid_attention(
        query=query,
        store=store,
        recent_k=recent_k,
        recent_v=recent_v,
        num_query_heads=num_query_heads,
        scale=scale,
        kv_paged_k=paged_k,
        kv_paged_v=paged_v,
    )

    # Reference: concat ring + paged into one segment, attend.
    ref_k = torch.cat([recent_k, paged_k], dim=0).transpose(0, 1).float()
    ref_v = torch.cat([recent_v, paged_v], dim=0).transpose(0, 1).float()
    ref = _attend_online_softmax(query, [(ref_k, ref_v)], gqa_ratio, num_kv_heads, scale)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_compute_hybrid_attention_no_segments_returns_zeros() -> None:
    """Empty store, empty ring, empty paged → zeros (degenerate path)."""
    from turboquant.score import compute_hybrid_attention
    from turboquant.store import CompressedKVStore

    head_dim, num_kv_heads, num_query_heads = 64, 2, 4
    store = CompressedKVStore(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        key_bits=3,
        value_bits=2,
        value_group_size=32,
        device=torch.device("cpu"),
        layer_idx=0,
    )

    query = torch.randn(1, num_query_heads, head_dim)
    out = compute_hybrid_attention(
        query=query,
        store=store,
        recent_k=None,
        recent_v=None,
        num_query_heads=num_query_heads,
    )

    assert torch.all(out == 0)
    assert out.shape == (1, num_query_heads, head_dim)
