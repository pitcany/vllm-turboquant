"""CompressedKVStore append + flat-cache invariants."""

from __future__ import annotations

import pytest
import torch

from turboquant.store import CompressedKVStore


def _make_store(device: torch.device, *, head_dim: int = 64, num_kv_heads: int = 4) -> CompressedKVStore:
    return CompressedKVStore(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        key_bits=3,
        value_bits=2,
        value_group_size=32,
        device=device,
        layer_idx=0,
    )


def _kv(num_tokens: int, *, head_dim: int, num_kv_heads: int, device: torch.device):
    k = torch.randn(num_tokens, num_kv_heads, head_dim, device=device)
    v = torch.randn(num_tokens, num_kv_heads, head_dim, device=device)
    return k, v


@pytest.mark.unit
def test_empty_store(device: torch.device) -> None:
    store = _make_store(device)
    assert store.num_tokens == 0
    assert store.num_chunks == 0
    assert store.get_flat_cache() is None
    assert store.memory_bytes() == 0


@pytest.mark.unit
def test_append_chunks_and_flat_cache(device: torch.device) -> None:
    store = _make_store(device)
    head_dim, num_kv_heads = store.head_dim, store.num_kv_heads

    chunk_sizes = [37, 5, 128]
    for n in chunk_sizes:
        k, v = _kv(n, head_dim=head_dim, num_kv_heads=num_kv_heads, device=device)
        store.append_chunk(k, v)

    assert store.num_tokens == sum(chunk_sizes)
    assert store.num_chunks == len(chunk_sizes)

    flat = store.get_flat_cache()
    assert flat is not None
    assert flat.num_tokens == sum(chunk_sizes)

    # Token dim is index -2 on the flattened key tensors.
    assert flat.prod_q.mse_indices.shape[-2] == sum(chunk_sizes)
    assert flat.value_q.data.shape[-2] == sum(chunk_sizes)


@pytest.mark.unit
def test_flat_cache_caches_until_next_write(device: torch.device) -> None:
    store = _make_store(device)
    head_dim, num_kv_heads = store.head_dim, store.num_kv_heads

    k, v = _kv(16, head_dim=head_dim, num_kv_heads=num_kv_heads, device=device)
    store.append_chunk(k, v)

    flat_a = store.get_flat_cache()
    flat_b = store.get_flat_cache()
    assert flat_a is flat_b, "flat cache should be memoized between reads"

    # appending invalidates
    k2, v2 = _kv(8, head_dim=head_dim, num_kv_heads=num_kv_heads, device=device)
    store.append_chunk(k2, v2)
    flat_c = store.get_flat_cache()
    assert flat_c is not flat_a
    assert flat_c.num_tokens == 24


@pytest.mark.unit
def test_reset_clears_state(device: torch.device) -> None:
    store = _make_store(device)
    head_dim, num_kv_heads = store.head_dim, store.num_kv_heads

    k, v = _kv(16, head_dim=head_dim, num_kv_heads=num_kv_heads, device=device)
    store.append_chunk(k, v)
    assert store.num_tokens == 16
    assert store.memory_bytes() > 0

    store.reset()
    assert store.num_tokens == 0
    assert store.num_chunks == 0
    assert store.get_flat_cache() is None
    assert store.memory_bytes() == 0


@pytest.mark.unit
def test_memory_bytes_grows_with_tokens(device: torch.device) -> None:
    store = _make_store(device)
    head_dim, num_kv_heads = store.head_dim, store.num_kv_heads

    k, v = _kv(8, head_dim=head_dim, num_kv_heads=num_kv_heads, device=device)
    store.append_chunk(k, v)
    bytes_after_one = store.memory_bytes()

    k2, v2 = _kv(8, head_dim=head_dim, num_kv_heads=num_kv_heads, device=device)
    store.append_chunk(k2, v2)
    bytes_after_two = store.memory_bytes()

    assert bytes_after_two > bytes_after_one
