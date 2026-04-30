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


@pytest.mark.unit
def test_rejects_prefix_caching() -> None:
    """S2.1: enable_turboquant must error loudly when prefix caching is on.

    F2 (see docs/integration-state.md § "F2 — Prefix caching evades capture
    even in eager mode") shows that vLLM's prefix cache reuses K/V blocks
    from prior requests, leaving TurboQuant's capture path with only the
    partial-block remainder of each prefill. Path B Conservative's chosen
    closure (per docs/plan-path-b.md §3 / Sprint 2 and §4) is to refuse the
    configuration at enable_turboquant time rather than ship silent
    misbehaviour. The capture-on-cache-hit alternative is out of scope.
    """
    from turboquant.vllm import TurboQuantVLLMError, enable_turboquant

    class _CacheCfg:
        enable_prefix_caching = True

    class _VllmCfg:
        cache_config = _CacheCfg()

    class FakeEngine:
        vllm_config = _VllmCfg()

        def collective_rpc(self, fn):  # pragma: no cover - guard fires first
            raise AssertionError("guard must fire before collective_rpc")

    class FakeLLM:
        llm_engine = FakeEngine()

    with pytest.raises(TurboQuantVLLMError, match="prefix"):
        enable_turboquant(FakeLLM(), mode="hybrid")


@pytest.mark.unit
def test_off_mode_allows_prefix_caching() -> None:
    """S2.1 corollary: mode="off" must not raise even with prefix caching on.

    Off-mode is a passthrough (TQ does nothing), so prefix caching can't
    evade what TQ isn't doing. Per the plan: "if True and mode != 'off'".
    """
    from turboquant.vllm import enable_turboquant

    class _CacheCfg:
        enable_prefix_caching = True

    class _VllmCfg:
        cache_config = _CacheCfg()

    class FakeEngine:
        vllm_config = _VllmCfg()

        def collective_rpc(self, fn):
            return [0]  # pretend zero hooks installed; off-mode is a no-op

    class FakeLLM:
        llm_engine = FakeEngine()

    info = enable_turboquant(FakeLLM(), mode="off")
    assert info["mode"] == "off"


@pytest.mark.integration
def test_smoke_with_vllm_if_available() -> None:
    """If vLLM is installed, importing the integration module shouldn't crash."""
    pytest.importorskip("vllm")
    import turboquant.integration.vllm  # noqa: F401
    import turboquant.vllm_attn_backend  # noqa: F401


# ---------------------------------------------------------------------------
# S1.4 — capture survives torch.compile (CUDA-marked, default-skipped)
# ---------------------------------------------------------------------------
#
# Asserts that under default vLLM compilation (no enforce_eager) and prefix
# caching disabled, the post-execute paged-cache reader (S1.3 / 1C) captures
# every prefill and decode K/V write. The expected store size after running
# MAX_TOKENS=65 against the LONG_PROMPT workload is exactly the prompt
# length (226) + decode-K/V writes (64) = 290; vLLM v1 emits N − 1 decode
# execute_model calls for max_tokens=N, so MAX_TOKENS=65 yields 64 decode
# writes. See docs/plan-path-b.md §3 / Sprint 1 acceptance and
# docs/integration-state.md § "S1.3 — 1C lands…".


def _cuda_smoke_skip_if_not_ready() -> None:
    """Skip when CUDA / vLLM aren't both available."""
    pytest.importorskip("vllm")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.mark.cuda
@pytest.mark.integration
def test_capture_under_compile() -> None:
    _cuda_smoke_skip_if_not_ready()

    import os

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams

    import turboquant.vllm  # noqa: F401  (side-effect: VLLM_ALLOW_INSECURE_SERIALIZATION)
    from turboquant.vllm import enable_turboquant, get_stats

    long_prompt = (
        "You are a careful technical writer. Write a thorough, step-by-step "
        "explanation of the following request, citing concrete numbers where "
        "possible and clearly distinguishing what is observed from what is "
        "speculated. The audience is an experienced systems engineer who is "
        "skeptical of marketing claims. Avoid bullet lists; use paragraphs. "
        "Do not repeat the request back. Begin with a single-sentence thesis.\n\n"
        "Request: explain how KV cache compression in transformer inference "
        "(per-token quantization, low-rank projections, or a combination) "
        "trades off memory savings against decoding throughput and output "
        "quality. Cover (a) what is in the cache and why, (b) why naive "
        "uniform quantization underperforms compared to rotation-then-quantize "
        "schemes such as Lloyd-Max on the Beta distribution arising from "
        "random orthogonal rotation of unit-norm vectors, (c) how unbiased "
        "inner-product estimators using residual sign bits (QJL-style) keep "
        "attention scores faithful, and (d) what concrete throughput and "
        "memory numbers a practitioner should expect on modern hardware "
        "running a 7B-parameter model at long context.\n\n"
        "Now write the explanation. Be precise. Be honest. Be brief."
    )

    try:
        llm = LLM(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            tensor_parallel_size=1,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            max_num_seqs=1,
            trust_remote_code=True,
            enforce_eager=False,
            enable_prefix_caching=False,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not construct LLM: {exc}")

    enable_turboquant(
        llm,
        key_bits=3,
        value_bits=2,
        buffer_size=16,
        mode="hybrid",
    )

    sp = SamplingParams(temperature=0, max_tokens=65)
    out = llm.generate([long_prompt], sp)
    n_generated = len(out[0].outputs[0].token_ids)
    assert n_generated == 65, f"expected 65 generated tokens, got {n_generated}"

    stats = get_stats(llm)
    assert stats, "expected at least one worker"
    worker = stats[0]
    captured = worker["total_compressed_tokens"] + worker["total_buffered_tokens"]

    # Workload spec: prompt tokenises to 226 tokens (verified in
    # docs/integration-state.md table). With max_tokens=65 the decode path
    # writes 64 K/Vs (vLLM v1 omits the K/V of the final sampled token).
    # Expected total = 226 prefill + 64 decode = 290.
    expected = 226 + 64
    assert captured >= expected, (
        f"capture under compile fell short: "
        f"compressed={worker['total_compressed_tokens']} "
        f"buffered={worker['total_buffered_tokens']} "
        f"sum={captured} expected≥{expected}"
    )
