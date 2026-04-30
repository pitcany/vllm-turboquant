"""vLLM integration smoke tests.

We don't spin up a real model in CI; these are import / error-path tests
that verify ``turboquant.vllm`` exposes the expected surface and emits
useful errors when handed something that isn't a vLLM ``LLM`` instance.
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_public_surface_imports() -> None:
    from turboquant.vllm import (  # noqa: F401
        VALID_MODES,
        TurboQuantVLLMError,
        enable_turboquant,
        free_kv_cache,
        get_stats,
        reset,
    )


@pytest.mark.unit
def test_valid_modes_includes_canonical_set() -> None:
    from turboquant.vllm import VALID_MODES

    assert {"off", "capture_only", "hybrid", "full_tq"}.issubset(set(VALID_MODES))
    # Legacy aliases should also still work
    assert {"shadow", "accumulate", "active"}.issubset(set(VALID_MODES))


@pytest.mark.unit
def test_enable_turboquant_rejects_unknown_mode() -> None:
    from turboquant.vllm import enable_turboquant

    with pytest.raises(ValueError, match="Unknown TurboQuant mode"):
        enable_turboquant(object(), mode="bogus")


@pytest.mark.unit
def test_enable_turboquant_helpful_error_on_non_llm() -> None:
    """Passing a non-LLM should raise our typed error, not AttributeError."""
    from turboquant.vllm import TurboQuantVLLMError, enable_turboquant

    class Stub:  # missing llm_engine
        pass

    with pytest.raises(TurboQuantVLLMError, match="llm_engine"):
        enable_turboquant(Stub())


@pytest.mark.unit
def test_resolve_executor_prefers_top_level_collective_rpc() -> None:
    """Top-level engine.collective_rpc is the first path tried."""
    from turboquant.vllm import _resolve_executor

    class FakeEngine:
        def collective_rpc(self, fn):
            return [fn(self)]

        # Also has a deeper executor; we should pick the top-level shortcut.
        class _Inner:
            class _Inner2:
                class _Exec:
                    def collective_rpc(self, fn):
                        raise AssertionError("deep path should not be picked")
                model_executor = _Exec()
            engine_core = _Inner2()
        engine_core = _Inner()

    class FakeLLM:
        llm_engine = FakeEngine()

    resolved = _resolve_executor(FakeLLM())
    assert resolved is FakeLLM.llm_engine


@pytest.mark.unit
def test_resolve_executor_falls_back_to_deep_path() -> None:
    """When only the deep engine_core.engine_core.model_executor exists, use it."""
    from turboquant.vllm import _resolve_executor

    class _Exec:
        def collective_rpc(self, fn):
            return [fn(self)]

    class _Core2:
        model_executor = _Exec()

    class _Core1:
        engine_core = _Core2()

    class FakeEngine:
        # No top-level collective_rpc, no direct model_executor.
        engine_core = _Core1()

    class FakeLLM:
        llm_engine = FakeEngine()

    resolved = _resolve_executor(FakeLLM())
    assert resolved is _Core2.model_executor


@pytest.mark.unit
def test_resolve_executor_lists_paths_when_none_match() -> None:
    """Error message must mention every path we tried, so users can debug."""
    from turboquant.vllm import TurboQuantVLLMError, _resolve_executor

    class FakeEngine:
        pass  # no collective_rpc, no model_executor, no engine_core

    class FakeLLM:
        llm_engine = FakeEngine()

    with pytest.raises(TurboQuantVLLMError) as exc:
        _resolve_executor(FakeLLM())

    msg = str(exc.value)
    assert "collective_rpc" in msg
    assert "model_executor" in msg


@pytest.mark.integration
def test_smoke_with_vllm_if_available() -> None:
    """If vLLM is installed, importing the integration module shouldn't crash."""
    pytest.importorskip("vllm")
    import turboquant.integration.vllm  # noqa: F401
    import turboquant.vllm_attn_backend  # noqa: F401
