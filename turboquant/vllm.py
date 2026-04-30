"""
Public, user-facing vLLM integration for TurboQuant.

This module is the supported entry point for end users. It encapsulates the
private vLLM v1 traversal that ``proof.py`` and ``benchmark.py`` previously
duplicated inline:

    llm.llm_engine.engine_core.engine_core.model_executor.collective_rpc(...)

The internals of vLLM v1 are a moving target; pin a tested vLLM range
(see setup.py) and treat any change in the symbols below as a breaking change
in this module too.

Typical usage
-------------

    from vllm import LLM, SamplingParams
    from turboquant.vllm import enable_turboquant, free_kv_cache, get_stats

    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", max_model_len=4096)
    info = enable_turboquant(llm, key_bits=3, value_bits=2, mode="capture_only")
    print(info)  # {'workers': 1, 'hooks_per_worker': [N], 'mode': 'capture_only'}

    out = llm.generate(["Hello, "], SamplingParams(max_tokens=32))
    print(out[0].outputs[0].text)

    stats = get_stats(llm)
    freed = free_kv_cache(llm)            # optional, after prefill is done
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger("turboquant.vllm")

# vLLM 0.17.x's default RPC serializer rejects closures with
# "Object of type <class 'function'> is not serializable" or, worse, lets
# the parent send a closure but the worker subprocess can't decode it
# ("Extension type code 2 is not supported"). The env var has to be set
# BEFORE vllm.LLM(...) constructs the engine subprocess, so we set it at
# import time on the assumption that the user imports turboquant.vllm
# before constructing the LLM. If they construct LLM first, set this var
# yourself in your environment before launching the process.
if not os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION"):
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    logger.info(
        "[TurboQuant] Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 at import time so "
        "vLLM v1 collective_rpc accepts our installer closure. Set this env var "
        "yourself before launching if you'd rather control it explicitly."
    )

# Public mode names. Map onto the canonical modes in
# turboquant.integration.vllm; legacy names from vllm_attn_backend
# remain accepted for backwards compatibility.
_USER_MODE_TO_INTERNAL = {
    "off": "off",
    "capture_only": "capture_only",
    "hybrid": "hybrid",
    "full_tq": "full_tq",
    # Legacy aliases (vllm_attn_backend constants):
    "shadow": "capture_only",
    "accumulate": "capture_only",
    "active": "hybrid",
}

VALID_MODES = tuple(sorted(set(_USER_MODE_TO_INTERNAL)))


class TurboQuantVLLMError(RuntimeError):
    """Raised when vLLM internals don't match what TurboQuant expects."""


def _resolve_executor(llm: Any) -> Any:
    """Find an object on ``llm`` that exposes ``collective_rpc`` for fan-out.

    vLLM v1 exposes the worker RPC at multiple call-sites depending on
    whether the engine runs in-process or in a subprocess:

    * top-level shortcut on the engine itself (works in both modes):
      ``llm.llm_engine.collective_rpc``
    * ``llm.llm_engine.model_executor`` (UniProcExecutor — in-process only)
    * deep path ``llm.llm_engine.engine_core.engine_core.model_executor``
      (only when ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` keeps the EngineCore
      in the same process).

    We try them in that order and return the first one that has
    ``collective_rpc``. Raises ``TurboQuantVLLMError`` with a helpful
    message if none match — that's the most likely thing to break on a
    vLLM version bump, so the message lists the paths we tried.
    """
    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        raise TurboQuantVLLMError(
            "llm.llm_engine not found; pass a vllm.LLM instance "
            "(synchronous engine). AsyncEngine is not yet supported."
        )

    candidates = []

    # 1. Top-level shortcut. The LLMEngine forwards collective_rpc to whichever
    #    process owns the workers, so this works for both in-proc and MP.
    if hasattr(engine, "collective_rpc"):
        candidates.append(("llm.llm_engine", engine))

    # 2. Direct model_executor attribute on the engine (in-proc shortcut).
    direct_executor = getattr(engine, "model_executor", None)
    if direct_executor is not None and hasattr(direct_executor, "collective_rpc"):
        candidates.append(("llm.llm_engine.model_executor", direct_executor))

    # 3. Legacy deep path (in-proc multiprocessing disabled).
    core = getattr(engine, "engine_core", None)
    if core is not None:
        inner = getattr(core, "engine_core", core)
        deep_executor = getattr(inner, "model_executor", None)
        if deep_executor is not None and hasattr(deep_executor, "collective_rpc"):
            candidates.append(
                ("llm.llm_engine.engine_core.engine_core.model_executor", deep_executor)
            )

    if not candidates:
        raise TurboQuantVLLMError(
            "Could not locate a collective_rpc-capable handle on the vLLM "
            "engine. Tried: llm.llm_engine.collective_rpc, "
            "llm.llm_engine.model_executor, "
            "llm.llm_engine.engine_core.engine_core.model_executor. "
            "Your vLLM version may be unsupported."
        )

    name, executor = candidates[0]
    logger.debug("[TurboQuant] resolved RPC handle via %s (%s)",
                 name, type(executor).__name__)

    if not hasattr(executor, "collective_rpc"):
        raise TurboQuantVLLMError(
            "executor.collective_rpc is missing; this build of vLLM does not "
            "expose the worker RPC TurboQuant needs."
        )
    return executor


def _check_vllm_version() -> Optional[str]:
    try:
        import vllm  # noqa: F401
    except Exception:
        return None
    version = getattr(vllm, "__version__", None) or "unknown"
    # Soft check; setup.py is the source of truth for hard pins.
    if not version.startswith(("0.17", "0.18")):
        logger.warning(
            "[TurboQuant] vLLM %s has not been validated; expect breakage. "
            "Tested range: 0.17.x / 0.18.x.",
            version,
        )
    return version


def enable_turboquant(
    llm: Any,
    *,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    buffer_size: int = 128,
    initial_layers_count: int = 4,
    initial_layers_key_bits: Optional[int] = None,
    mode: str = "capture_only",
    allow_insecure_serialization: bool = True,
) -> dict:
    """Install TurboQuant hooks on every worker behind ``llm``.

    Parameters
    ----------
    llm
        A ``vllm.LLM`` instance (synchronous engine).
    key_bits, value_bits
        TQ bit budget. ``key_bits=3`` and ``value_bits=2`` is the default
        config from the paper; use ``value_bits=4`` for quality-sensitive
        workloads (cos_sim ≈ 0.997 vs 0.94 at 2-bit).
    value_group_size
        Group size for value quantization (clipped to ``head_dim``).
    buffer_size
        Capacity of the per-layer ring buffer of recent exact tokens.
    initial_layers_count, initial_layers_key_bits
        First N layers get a higher key-bit budget (default ``key_bits + 1``,
        capped at 4) since early layers are more sensitive.
    mode
        One of ``"off"``, ``"capture_only"``, ``"hybrid"``, ``"full_tq"``.
        Legacy names (``shadow``, ``accumulate``, ``active``) are accepted.
    allow_insecure_serialization
        Controls whether ``import turboquant.vllm`` is allowed to set
        ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` (which it does by default,
        because vLLM 0.17.x's default RPC serializer can't transport our
        installer closure). Pass ``False`` and ``enable_turboquant`` will
        raise instead of proceeding when the env var isn't set, so you can
        control the flag explicitly.

    Returns
    -------
    dict with keys ``workers``, ``hooks_per_worker``, ``mode``,
    ``vllm_version``.
    """
    if mode not in _USER_MODE_TO_INTERNAL:
        raise ValueError(
            f"Unknown TurboQuant mode {mode!r}; valid: {VALID_MODES}"
        )
    internal_mode = _USER_MODE_TO_INTERNAL[mode]

    if not allow_insecure_serialization and \
            not os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION"):
        raise TurboQuantVLLMError(
            "VLLM_ALLOW_INSECURE_SERIALIZATION is not set, but "
            "allow_insecure_serialization=False was passed. vLLM v1 "
            "collective_rpc on 0.17.x rejects closures without that flag. "
            "Set the env var before constructing vllm.LLM(...) to enable "
            "TurboQuant, or import turboquant.vllm before vllm to let it "
            "set the flag automatically at import time."
        )

    vllm_version = _check_vllm_version()
    executor = _resolve_executor(llm)

    # Bind the config into a closure that runs on each worker process.
    cfg = dict(
        key_bits=key_bits,
        value_bits=value_bits,
        value_group_size=value_group_size,
        ring_capacity=buffer_size,
        initial_layers_count=initial_layers_count,
        initial_layers_key_bits=initial_layers_key_bits,
        mode=internal_mode,
    )

    def _install(worker, _cfg=cfg):  # runs in each worker process
        # Import inside the worker; vLLM serializes the function and
        # re-imports turboquant on the worker side.
        from turboquant.integration.vllm import install_hooks

        states = install_hooks(worker.model_runner, **_cfg)
        return len(states)

    try:
        hooks_per_worker = executor.collective_rpc(_install)
    except Exception as exc:  # surface a useful error to the caller
        raise TurboQuantVLLMError(
            f"collective_rpc(install_hooks) failed: {exc}"
        ) from exc

    info = {
        "workers": len(hooks_per_worker),
        "hooks_per_worker": list(hooks_per_worker),
        "mode": mode,
        "vllm_version": vllm_version,
    }
    logger.info("[TurboQuant] enabled: %s", info)
    return info


def free_kv_cache(llm: Any) -> int:
    """Drop the paged KV-cache tensors for TQ-hooked layers.

    Call this after prefill if you want to reclaim the VRAM that vLLM
    pre-allocated for the layers TQ now manages. Returns total bytes freed
    across workers.

    Note: this is the same operation the legacy ``free_kv_cache`` helper in
    ``vllm_attn_backend`` performs; it works by replacing the cache tensor
    entries with a tiny placeholder, which is fragile against vLLM allocator
    changes.
    """
    executor = _resolve_executor(llm)

    def _free(worker):
        from turboquant.integration.vllm import free_kv_cache as _impl
        return _impl(worker.model_runner)

    freed_per_worker = executor.collective_rpc(_free)
    total = sum(int(b) for b in freed_per_worker if b is not None)
    logger.info("[TurboQuant] freed %d bytes across %d workers",
                total, len(freed_per_worker))
    return total


def get_stats(llm: Any) -> list[dict]:
    """Return per-worker TurboQuant statistics."""
    executor = _resolve_executor(llm)

    def _stats(worker):
        from turboquant.integration.vllm import get_stats as _impl
        return _impl(worker.model_runner)

    return list(executor.collective_rpc(_stats))


def reset(llm: Any) -> None:
    """Reset the TQ ring buffers and stores on every worker.

    Use between unrelated generation requests if you don't want history from
    the previous request to leak into hybrid attention.
    """
    executor = _resolve_executor(llm)

    def _reset(worker):
        states = getattr(worker.model_runner, "_tq_layer_states", None) \
            or getattr(worker.model_runner, "_tq_states", None) \
            or {}
        for s in states.values():
            s.reset()
        return len(states)

    executor.collective_rpc(_reset)
