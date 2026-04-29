"""Round-trip tests for the bit-packing helpers in turboquant.quantizer."""

from __future__ import annotations

import pytest
import torch

from turboquant.quantizer import _pack_indices, _unpack_indices


@pytest.mark.unit
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("d", [8, 16, 64, 128])
def test_pack_unpack_roundtrip_aligned(bits: int, d: int) -> None:
    n_clusters = 1 << bits
    torch.manual_seed(0)
    indices = torch.randint(0, n_clusters, (4, d), dtype=torch.long)
    packed = _pack_indices(indices, bits)
    unpacked = _unpack_indices(packed, bits, d)
    assert unpacked.shape == indices.shape
    assert torch.equal(unpacked, indices)


@pytest.mark.unit
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("d", [3, 7, 13, 31, 65])
def test_pack_unpack_roundtrip_unaligned(bits: int, d: int) -> None:
    """d that is not a multiple of vals_per_byte still round-trips."""
    n_clusters = 1 << bits
    torch.manual_seed(1)
    indices = torch.randint(0, n_clusters, (2, 3, d), dtype=torch.long)
    packed = _pack_indices(indices, bits)
    unpacked = _unpack_indices(packed, bits, d)
    assert torch.equal(unpacked, indices)


@pytest.mark.unit
def test_pack_2bit_byte_layout() -> None:
    """Spot-check that 2-bit packing follows the documented little-endian layout."""
    indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    packed = _pack_indices(indices, 2)
    # 0 | 1<<2 | 2<<4 | 3<<6 == 0xE4
    assert packed.dtype == torch.uint8
    assert packed.flatten().tolist() == [0xE4]
