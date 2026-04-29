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


@pytest.mark.integration
def test_smoke_with_vllm_if_available() -> None:
    """If vLLM is installed, importing the integration module shouldn't crash."""
    pytest.importorskip("vllm")
    import turboquant.integration.vllm  # noqa: F401
    import turboquant.vllm_attn_backend  # noqa: F401
