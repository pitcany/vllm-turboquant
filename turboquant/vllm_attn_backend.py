"""
TurboQuant attention backend shim — DEPRECATED.

This module exists only for backwards compatibility with scripts that import
``install_turboquant_hooks``, ``MODE_ACCUMULATE``/``MODE_ACTIVE``, etc.
New code should use the canonical, supported API:

    from turboquant.vllm import enable_turboquant, free_kv_cache, get_stats

The legacy mode names (``shadow``, ``accumulate``, ``active``) and the
``buffer_size`` parameter name are kept aliases of the new mode names
(``capture_only``, ``hybrid``) and ``ring_capacity``.

``enable_no_alloc`` is the only function here that has no canonical replacement
yet; it monkey-patches private vLLM v1 internals (``Executor.get_kv_cache_specs``,
``GPUModelRunner._update_hybrid_attention_mamba_layout``, ``GPUWorker.load_model``)
to install hooks during engine initialization. It remains experimental — expect
breakage on any vLLM version bump.
"""

from __future__ import annotations

import logging
import os
import warnings

import turboquant.integration.vllm as _new_backend

_DEBUG_LOG = os.environ.get("TURBOQUANT_DEBUG_LOG")  # set to a file path to enable

logger = logging.getLogger("turboquant.attn")

# ---- Legacy mode constants -------------------------------------------------
# These map onto the canonical modes in turboquant.integration.vllm.
MODE_SHADOW = "shadow"
MODE_ACCUMULATE = "accumulate"
MODE_ACTIVE = "active"
_VALID_MODES = (MODE_SHADOW, MODE_ACCUMULATE, MODE_ACTIVE)

_LEGACY_TO_NEW = {
    MODE_ACCUMULATE: _new_backend.MODE_CAPTURE_ONLY,
    MODE_SHADOW: _new_backend.MODE_CAPTURE_ONLY,
    MODE_ACTIVE: _new_backend.MODE_HYBRID,
}

_GLOBAL_MODE = MODE_ACCUMULATE


def _deprecation(what: str, replacement: str) -> None:
    warnings.warn(
        f"{what} is deprecated; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def set_mode(mode: str) -> None:
    """Deprecated. Use ``turboquant.integration.vllm.set_mode`` directly."""
    _deprecation(
        "turboquant.vllm_attn_backend.set_mode",
        "turboquant.integration.vllm.set_mode",
    )
    global _GLOBAL_MODE
    assert mode in _VALID_MODES
    _GLOBAL_MODE = mode
    _new_backend.set_mode(_LEGACY_TO_NEW.get(mode, _new_backend.MODE_CAPTURE_ONLY))


def get_mode() -> str:
    return _GLOBAL_MODE


def install_turboquant_hooks(
    model_runner,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    buffer_size: int = 128,
    initial_layers_count: int = 4,
    initial_layers_key_bits: int | None = None,
    mode: str = MODE_ACCUMULATE,
    no_alloc: bool = False,
):
    """Deprecated. Use ``turboquant.vllm.enable_turboquant(llm, ...)``.

    This function still works — it forwards to
    ``turboquant.integration.vllm.install_hooks`` — but ``enable_turboquant``
    handles the worker fan-out and version checks for you instead of
    requiring you to traverse vLLM's private engine-core path yourself.
    """
    _deprecation(
        "install_turboquant_hooks",
        "turboquant.vllm.enable_turboquant(llm, ...)",
    )
    global _GLOBAL_MODE
    new_mode = _LEGACY_TO_NEW.get(mode, _new_backend.MODE_CAPTURE_ONLY)

    layer_states = _new_backend.install_hooks(
        model_runner,
        key_bits=key_bits,
        value_bits=value_bits,
        value_group_size=value_group_size,
        ring_capacity=buffer_size,
        initial_layers_count=initial_layers_count,
        initial_layers_key_bits=initial_layers_key_bits,
        mode=new_mode,
        no_alloc=no_alloc,
    )

    _GLOBAL_MODE = mode
    # Mirror the canonical attribute under its old name for any caller still
    # reaching for `_tq_states`. New code should read `_tq_layer_states`.
    model_runner._tq_states = layer_states
    model_runner._tq_no_alloc = no_alloc
    return layer_states


def free_kv_cache(model_runner) -> int:
    """Free paged KV cache tensors for TQ-hooked layers.

    Thin delegator to ``turboquant.integration.vllm.free_kv_cache``. The
    canonical install path (``install_hooks``) sets
    ``model_runner._tq_layer_states``, which the canonical implementation
    consumes; the legacy ``_tq_states`` fallback that used to live here was
    dead code (``install_turboquant_hooks`` populates *both* attributes).
    """
    return _new_backend.free_kv_cache(model_runner)


# ---- Experimental no-alloc auto-installer ---------------------------------
# Kept here because it has no equivalent in integration/vllm.py and at least
# proof.py / benchmark.py rely on importing it from this module.

_TQ_NO_ALLOC_CONFIG = None


def enable_no_alloc(
    key_bits: int = 3,
    value_bits: int = 2,
    buffer_size: int = 128,
    initial_layers_count: int = 4,
):
    """Call BEFORE creating ``vllm.LLM(...)``.

    Patches ``vllm.v1.executor.abstract.Executor.get_kv_cache_specs``,
    ``GPUModelRunner._update_hybrid_attention_mamba_layout``, and
    ``GPUWorker.load_model`` so TQ hooks are installed automatically during
    engine initialization. Experimental — depends on private vLLM v1 internals
    and is likely to break on version bumps.
    """
    global _TQ_NO_ALLOC_CONFIG
    _TQ_NO_ALLOC_CONFIG = dict(
        key_bits=key_bits,
        value_bits=value_bits,
        buffer_size=buffer_size,
        initial_layers_count=initial_layers_count,
    )

    from vllm.v1.executor.abstract import Executor
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if hasattr(Executor, "_tq_patched"):
        return

    if not hasattr(GPUModelRunner, "_tq_layout_patch"):
        orig_layout_update = GPUModelRunner._update_hybrid_attention_mamba_layout

        def patched_layout_update(self, kv_caches):
            for layer_name, target_layer_name in getattr(
                self, "shared_kv_cache_layers", {}
            ).items():
                if layer_name not in kv_caches and target_layer_name in kv_caches:
                    kv_caches[layer_name] = kv_caches[target_layer_name]
            return orig_layout_update(self, kv_caches)

        GPUModelRunner._update_hybrid_attention_mamba_layout = patched_layout_update
        GPUModelRunner._tq_layout_patch = True

    orig_get_specs = Executor.get_kv_cache_specs

    def patched_get_kv_cache_specs(self):
        cfg = _TQ_NO_ALLOC_CONFIG
        if _DEBUG_LOG:
            with open(_DEBUG_LOG, "a") as f:
                f.write(f"patched_get_kv_cache_specs called pid={os.getpid()} cfg={cfg is not None}\n")
                f.flush()
        if cfg is None:
            return orig_get_specs(self)

        def _worker_install_tq(worker):
            # Note: we call the (deprecated) shim here because we want the
            # `_tq_states` attribute set on the model runner for the legacy
            # callers that still read it. New code should never hit this path.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                from turboquant.vllm_attn_backend import (
                    install_turboquant_hooks, MODE_ACTIVE,
                )
                tq_states = install_turboquant_hooks(
                    worker.model_runner,
                    key_bits=cfg["key_bits"],
                    value_bits=cfg["value_bits"],
                    buffer_size=cfg["buffer_size"],
                    initial_layers_count=cfg["initial_layers_count"],
                    mode=MODE_ACTIVE,
                    no_alloc=True,
                )
            static_ctx = worker.model_runner.compilation_config.static_forward_context
            flash_layers = [
                name
                for name, state in tq_states.items()
                if getattr(state, "supports_hybrid", False)
            ]
            shared_layers = 0
            if len(flash_layers) > 1:
                target = flash_layers[0]
                target_attn = static_ctx.get(target)
                if target_attn is not None and hasattr(target_attn, "kv_sharing_target_layer_name"):
                    target_attn.kv_sharing_target_layer_name = None
                for name in flash_layers[1:]:
                    attn = static_ctx.get(name)
                    if attn is None or not hasattr(attn, "kv_sharing_target_layer_name"):
                        continue
                    attn.kv_sharing_target_layer_name = target
                    shared_layers += 1

            return {
                "hooks": len(tq_states),
                "flash_layers": len(flash_layers),
                "shared_layers": shared_layers,
            }

        try:
            hooks = self.collective_rpc(_worker_install_tq)
            logger.info("[TurboQuant] Installed no_alloc hooks: %s", hooks)
        except Exception as exc:
            logger.exception("[TurboQuant] collective_rpc failed: %s", exc)
        return orig_get_specs(self)

    Executor.get_kv_cache_specs = patched_get_kv_cache_specs
    Executor._tq_patched = True

    # Patch the worker's load_model (NOT decorated, so our patch isn't bypassed)
    try:
        from vllm.v1.worker.gpu_worker import GPUWorker as WorkerCls
    except ImportError:
        try:
            from vllm.v1.worker.gpu_worker import Worker as WorkerCls
        except ImportError:
            WorkerCls = None

    if WorkerCls is not None:
        orig_worker_load = WorkerCls.load_model

        def patched_worker_load(self_worker):
            orig_worker_load(self_worker)
            cfg = _TQ_NO_ALLOC_CONFIG
            if cfg:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        from turboquant.vllm_attn_backend import (
                            install_turboquant_hooks, MODE_ACCUMULATE,
                        )
                        tq = install_turboquant_hooks(
                            self_worker.model_runner,
                            key_bits=cfg["key_bits"],
                            value_bits=cfg["value_bits"],
                            buffer_size=cfg["buffer_size"],
                            initial_layers_count=cfg["initial_layers_count"],
                            mode=MODE_ACCUMULATE,
                            no_alloc=False,
                        )
                    if _DEBUG_LOG:
                        with open(_DEBUG_LOG, "a") as f:
                            f.write(f"TQ hooks: {len(tq)} layers pid={os.getpid()}\n")
                            f.flush()
                except Exception as exc:
                    logger.exception("[TurboQuant] worker load_model TQ install failed: %s", exc)

        WorkerCls.load_model = patched_worker_load

    logger.info("[TurboQuant] Patched Executor for auto TQ hook installation")
