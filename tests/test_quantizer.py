"""End-to-end tests for TurboQuantMSE / TurboQuantProd.

These exercise the public quantize/dequantize/attention_score paths on
both CPU and (when available) CUDA. The quality thresholds are loose
enough to be stable across PyTorch versions but tight enough that a
real regression (e.g. wrong rotation matrix sign, broken bit-packing)
will trip them.
"""

from __future__ import annotations

import pytest
import torch

from turboquant import TurboQuantMSE, TurboQuantProd


# Loose quality floors per bit budget. These are well below what a healthy
# Lloyd-Max/QJL pipeline produces but high enough that a real regression
# (e.g. broken centroid lookup, miswired rotation matrix, off-by-one in
# bit-packing) drops below them. Tighter audit-style numbers live in the
# benchmark scripts, not here.
_MSE_COS_FLOOR = {2: 0.55, 3: 0.80, 4: 0.88}


@pytest.mark.unit
@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("dim", [64, 128])
def test_mse_roundtrip(device: torch.device, bits: int, dim: int) -> None:
    torch.manual_seed(0)
    quant = TurboQuantMSE(dim=dim, bits=bits, device=device, seed=0)
    x = torch.randn(64, dim, device=device)

    q = quant.quantize(x)
    x_hat = quant.dequantize(q)

    assert x_hat.shape == x.shape
    assert x_hat.device.type == device.type
    assert torch.isfinite(x_hat).all()

    cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()
    assert cos >= _MSE_COS_FLOOR[bits], (
        f"MSE cos_sim {cos:.3f} below floor {_MSE_COS_FLOOR[bits]} for bits={bits}, dim={dim}"
    )

    # ||x_hat|| should be in the same ballpark as ||x||. At low bits the
    # reconstructed vector loses some energy (centroid lookup has norm < 1),
    # so we use a loose 2x band rather than ±25%.
    norm_ratio = x_hat.norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-9)
    assert (norm_ratio > 0.4).all() and (norm_ratio < 2.0).all()


@pytest.mark.unit
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_mse_indices_in_range(device: torch.device, bits: int) -> None:
    """Packed indices must round-trip through unpack into the cluster range."""
    from turboquant.quantizer import _unpack_indices

    quant = TurboQuantMSE(dim=64, bits=bits, device=device, seed=0)
    x = torch.randn(8, 64, device=device)
    q = quant.quantize(x)
    unpacked = _unpack_indices(q.indices, q.bits, 64)
    n_clusters = 1 << bits
    assert unpacked.min().item() >= 0
    assert unpacked.max().item() < n_clusters


@pytest.mark.unit
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_prod_roundtrip(device: torch.device, bits: int) -> None:
    quant = TurboQuantProd(dim=128, bits=bits, device=device, seed=0)
    x = torch.randn(32, 128, device=device)
    x_hat = quant.dequantize(quant.quantize(x))

    assert x_hat.shape == x.shape
    assert torch.isfinite(x_hat).all()
    cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()
    # Prod reconstruction is noisier than MSE since QJL adds variance.
    assert cos >= 0.5, f"Prod 4-bit cos_sim {cos:.3f} unexpectedly low"


@pytest.mark.unit
def test_prod_attention_score_unbiased(device: torch.device) -> None:
    """Average estimator over many keys should track the true inner product."""
    torch.manual_seed(0)
    dim = 128
    quant = TurboQuantProd(dim=dim, bits=4, device=device, seed=0)

    # Many keys, single query, large enough for the estimator's variance to wash out.
    n_keys = 1024
    keys = torch.randn(n_keys, dim, device=device)
    query = torch.randn(1, dim, device=device)

    qkeys = quant.quantize(keys)
    est = quant.attention_score(query, qkeys)  # (1, n_keys)
    true = query @ keys.T  # (1, n_keys)

    # The estimator is unbiased per-pair; average correlation should be high.
    cos = torch.nn.functional.cosine_similarity(
        est.flatten().float().unsqueeze(0), true.flatten().float().unsqueeze(0), dim=-1
    ).item()
    assert cos > 0.6, f"prod-vs-true cos_sim {cos:.3f} too low — estimator may be broken"


@pytest.mark.unit
def test_quantizer_rejects_bits_below_two() -> None:
    """Inner-product quantizer needs >=2 bits (1 MSE + 1 QJL)."""
    with pytest.raises(AssertionError):
        TurboQuantProd(dim=128, bits=1, device=torch.device("cpu"), seed=0)


@pytest.mark.unit
def test_cpu_built_module_handles_cuda_input() -> None:
    """Auto-migration: build on CPU, feed CUDA tensors, no device-mismatch crash."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    quant = TurboQuantMSE(dim=64, bits=3, device=torch.device("cpu"), seed=0)
    x = torch.randn(4, 64, device="cuda")
    x_hat = quant.dequantize(quant.quantize(x))
    assert x_hat.device.type == "cuda"
