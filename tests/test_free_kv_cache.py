"""Unit tests for free_kv_cache (Sprint 4 / S4.1).

Verifies the new migration precondition + per-layer skip-on-empty
behaviour. Cannot exercise the real paged-cache release without CUDA +
vLLM, so these tests fake the model_runner / static_forward_context /
LayerState surface tightly enough to drive both code paths.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch


def _make_fake_layer_state(
    layer_idx: int,
    num_compressed: int = 0,
    num_buffered: int = 0,
    supports_hybrid: bool = True,
) -> Any:
    """Minimal stand-in for turboquant.integration.vllm.LayerState."""

    store = SimpleNamespace(num_tokens=num_compressed)
    ring = SimpleNamespace(size=num_buffered)
    engine = SimpleNamespace(
        total_tokens=num_compressed + num_buffered,
        ring=ring,
    )
    config = SimpleNamespace(layer_idx=layer_idx)
    return SimpleNamespace(
        store=store,
        engine=engine,
        config=config,
        supports_hybrid=supports_hybrid,
        _kv_cache_freed=False,
    )


def _make_fake_model_runner(layer_states: dict, kv_caches: list | None = None):
    """Stand-in for vLLM's model_runner shape that free_kv_cache reads."""

    static_ctx: dict[str, Any] = {}
    for layer_name in layer_states:
        # Each TQ-hooked layer has an attn_module with kv_cache=[tensor].
        attn_module = SimpleNamespace(kv_cache=[torch.zeros(4, 2, 8, 64, dtype=torch.float16)])
        static_ctx[layer_name] = attn_module

    compilation_config = SimpleNamespace(static_forward_context=static_ctx)
    return SimpleNamespace(
        _tq_layer_states=layer_states,
        compilation_config=compilation_config,
        device=torch.device("cpu"),
        kv_caches=kv_caches if kv_caches is not None else [],
    )


@pytest.mark.unit
def test_refuses_when_no_layer_has_captured_tokens() -> None:
    """Pre-S4.1 silently destroyed the paged cache; post-S4.1 refuses."""
    from turboquant.integration.vllm import free_kv_cache

    layer_states = {f"layer_{i}": _make_fake_layer_state(i, num_compressed=0, num_buffered=0) for i in range(3)}
    runner = _make_fake_model_runner(layer_states)

    with pytest.raises(RuntimeError, match="no TQ-hooked layer has any captured tokens"):
        free_kv_cache(runner)


@pytest.mark.unit
def test_refuses_when_only_non_hybrid_layers_have_captured() -> None:
    """MLA / GDN layers are non-hybrid and don't count towards the migration check."""
    from turboquant.integration.vllm import free_kv_cache

    layer_states = {
        "mla_0": _make_fake_layer_state(0, num_compressed=128, num_buffered=0, supports_hybrid=False),
        "mla_1": _make_fake_layer_state(1, num_compressed=128, num_buffered=0, supports_hybrid=False),
    }
    runner = _make_fake_model_runner(layer_states)

    with pytest.raises(RuntimeError, match="no TQ-hooked layer has any captured tokens"):
        free_kv_cache(runner)


@pytest.mark.unit
def test_releases_layers_with_captured_tokens_and_returns_byte_count() -> None:
    """Happy path: every layer has captured tokens, all get released."""
    from turboquant.integration.vllm import free_kv_cache

    layer_states = {f"layer_{i}": _make_fake_layer_state(i, num_compressed=210, num_buffered=16) for i in range(4)}
    runner = _make_fake_model_runner(layer_states)
    expected_per_layer = 4 * 2 * 8 * 64 * 2  # (4*2*8*64) elements * 2 bytes (fp16)
    expected_total = expected_per_layer * 4

    freed = free_kv_cache(runner)

    assert freed == expected_total
    # All layers' kv_cache[0] should now be the 1-byte sentinel.
    for layer_name in layer_states:
        kv_list = runner.compilation_config.static_forward_context[layer_name].kv_cache
        assert kv_list[0].nelement() == 1
        assert kv_list[0].dtype == torch.int8
    # All layer states should be latched as freed.
    for state in layer_states.values():
        assert state._kv_cache_freed is True


@pytest.mark.unit
def test_skips_per_layer_when_some_layers_have_zero_captured() -> None:
    """The 'release only the migrated suffix' contract: skip empty layers."""
    from turboquant.integration.vllm import free_kv_cache

    # Layers 0 and 2 are populated; 1 and 3 are empty.
    layer_states = {
        "layer_0": _make_fake_layer_state(0, num_compressed=210, num_buffered=16),
        "layer_1": _make_fake_layer_state(1, num_compressed=0, num_buffered=0),
        "layer_2": _make_fake_layer_state(2, num_compressed=42, num_buffered=8),
        "layer_3": _make_fake_layer_state(3, num_compressed=0, num_buffered=0),
    }
    runner = _make_fake_model_runner(layer_states)

    freed = free_kv_cache(runner)

    # Two layers freed, two skipped.
    expected_per_layer = 4 * 2 * 8 * 64 * 2
    assert freed == 2 * expected_per_layer

    # Populated layers: kv_cache[0] now sentinel, _kv_cache_freed True.
    for name in ("layer_0", "layer_2"):
        kv = runner.compilation_config.static_forward_context[name].kv_cache[0]
        assert kv.nelement() == 1
        assert layer_states[name]._kv_cache_freed is True

    # Empty layers: kv_cache[0] untouched, _kv_cache_freed False.
    for name in ("layer_1", "layer_3"):
        kv = runner.compilation_config.static_forward_context[name].kv_cache[0]
        assert kv.nelement() == 4 * 2 * 8 * 64
        assert layer_states[name]._kv_cache_freed is False


@pytest.mark.unit
def test_no_layer_states_returns_zero_with_warning() -> None:
    """Backward-compat: model_runner without TQ hooks returns 0, doesn't raise."""
    from turboquant.integration.vllm import free_kv_cache

    runner = SimpleNamespace(
        _tq_layer_states=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
        device=torch.device("cpu"),
        kv_caches=[],
    )

    assert free_kv_cache(runner) == 0
