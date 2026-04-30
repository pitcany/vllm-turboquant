"""
TurboQuant vLLM integration — thin adapter layer.

Responsibilities:
  - Detect layer/backend type (flash vs MLA/GDN)
  - Install minimal monkey-patches that delegate to capture/store/score
  - Expose clean modes: off | capture_only | hybrid | full_tq
  - Keep patching surface tiny; all real logic lives in capture/store/score

Modes:
  - off:          no TQ activity, passthrough
  - capture_only: capture KV into compressed store, always use flash output
  - hybrid:       use compressed history + exact recent for decode
  - full_tq:      (future) TQ handles everything including prefill
"""

from __future__ import annotations

import logging
import math
import os
import types
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from turboquant.capture import KVCaptureEngine
from turboquant.score import compute_hybrid_attention
from turboquant.store import CompressedKVStore

logger = logging.getLogger("turboquant.integration.vllm")

MODE_OFF = "off"
MODE_CAPTURE_ONLY = "capture_only"
MODE_HYBRID = "hybrid"
MODE_FULL_TQ = "full_tq"
_VALID_MODES = (MODE_OFF, MODE_CAPTURE_ONLY, MODE_HYBRID, MODE_FULL_TQ)

_GLOBAL_MODE = MODE_CAPTURE_ONLY

# ---------------------------------------------------------------------------
# Diagnostic trace plumbing (S0.1 of docs/plan-path-b.md).
#
# When TURBOQUANT_TRACE=1 in the environment of *any* process running this
# module (the driver, or a vLLM worker subprocess that inherits env), every
# interception point below emits a single-line debug record on a dedicated
# logger ``turboquant.trace`` whose handler uses a fixed format so that
# subsequent grep over the captured log can settle questions like "did the
# compiled kv_update fire?". Default off — zero cost when the env var is
# unset (one os.environ.get + a bool check per call).
# ---------------------------------------------------------------------------

_TRACE_LOGGER = logging.getLogger("turboquant.trace")
_TRACE_FORMAT = "[TQ-TRACE] %(name)s pid=%(process)d layer=%(layer)d %(message)s"
_TRACE_CONFIGURED = False


def _trace_enabled() -> bool:
    return bool(os.environ.get("TURBOQUANT_TRACE"))


def _ensure_trace_configured() -> None:
    """Attach a stderr handler to ``turboquant.trace`` (idempotent, per-process)."""
    global _TRACE_CONFIGURED
    if _TRACE_CONFIGURED:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_TRACE_FORMAT))
    _TRACE_LOGGER.addHandler(handler)
    _TRACE_LOGGER.setLevel(logging.DEBUG)
    # Don't propagate to root so we don't double-print under vLLM's logger setup.
    _TRACE_LOGGER.propagate = False
    _TRACE_CONFIGURED = True


def _trace(layer_idx: int, msg: str) -> None:
    """Emit a single-line trace record. No-op unless TURBOQUANT_TRACE is set."""
    if not _trace_enabled():
        return
    _ensure_trace_configured()
    _TRACE_LOGGER.debug(msg, extra={"layer": layer_idx})


def set_mode(mode: str):
    global _GLOBAL_MODE
    assert mode in _VALID_MODES, f"Invalid mode: {mode}. Valid: {_VALID_MODES}"
    _GLOBAL_MODE = mode
    logger.info(f"[TurboQuant] Mode set to: {mode}")


def get_mode() -> str:
    return _GLOBAL_MODE


@dataclass
class LayerConfig:
    """Per-layer TQ configuration."""

    head_dim: int
    num_kv_heads: int
    num_query_heads: int
    key_bits: int = 3
    value_bits: int = 2
    value_group_size: int = 32
    ring_capacity: int = 128
    layer_idx: int = 0
    backend_kind: str = "flash"  # "flash" | "mla"
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))


@dataclass
class LayerState:
    """Per-layer runtime state. Owns the capture engine and store."""

    config: LayerConfig
    store: CompressedKVStore
    engine: KVCaptureEngine
    _log_count: int = 0

    @property
    def supports_hybrid(self) -> bool:
        return self.config.backend_kind == "flash"

    def reset(self):
        self.engine.reset()
        self._log_count = 0


def _create_layer_state(cfg: LayerConfig) -> LayerState:
    store = CompressedKVStore(
        head_dim=cfg.head_dim,
        num_kv_heads=cfg.num_kv_heads,
        key_bits=cfg.key_bits,
        value_bits=cfg.value_bits,
        value_group_size=cfg.value_group_size,
        device=cfg.device,
        layer_idx=cfg.layer_idx,
    )
    engine = KVCaptureEngine(
        store=store,
        ring_capacity=cfg.ring_capacity,
        device=cfg.device,
    )
    return LayerState(config=cfg, store=store, engine=engine)


def _infer_num_query_heads(attn_module, impl) -> int:
    for candidate in (
        getattr(attn_module, "num_heads", None),
        getattr(attn_module, "num_attention_heads", None),
        getattr(impl, "num_heads", None),
    ):
        if candidate:
            return int(candidate)
    return int(impl.num_kv_heads)


def _is_mla_impl(impl) -> bool:
    return hasattr(impl, "forward_mqa") and hasattr(impl, "do_kv_cache_update") and not hasattr(impl, "forward")


# ---------------------------------------------------------------------------
# Patched methods — kept as thin as possible
# ---------------------------------------------------------------------------


def _make_patched_kv_update(orig_fn, state: LayerState, no_alloc: bool = False):
    """Intercept KV cache writes to capture prefill K/V into TQ store.

    Decode-token capture (``num_tokens <= 1``) is *not* handled here anymore
    — see ``install_post_execute_callback`` (Path B Sprint 1 / S1.3 / 1C).
    Reasoning: under default vLLM compilation the decode path runs as a
    FULL CUDAGraph replay, which bypasses any Python hook (including this
    monkey-patch) by construction (``vllm/compilation/cuda_graph.py:208–323``,
    documented in ``docs/integration-state.md`` § "F1bis"). The per-step
    paged-cache reader installed by ``install_post_execute_callback`` is the
    only decode-capture path that survives. Routing both here would
    double-ingest under PIECEWISE / eager modes.
    """

    def patched(self_impl, layer, key, value, kv_cache, slot_mapping):
        if not no_alloc:
            orig_fn(self_impl, layer, key, value, kv_cache, slot_mapping)

        mode = _GLOBAL_MODE
        num_tokens = slot_mapping.shape[0]
        _trace(state.config.layer_idx, f"kv_update slot_mapping={num_tokens} mode={mode}")

        if mode == MODE_OFF:
            return

        if num_tokens <= 1:
            return
        state.engine.ingest_prefill(key, value, num_tokens)

    return patched


def _no_alloc_prefill_attention(
    state: LayerState,
    self_impl,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata,
):
    num_actual = attn_metadata.num_actual_tokens
    q = query[:num_actual]
    k = key[:num_actual]
    v = value[:num_actual]

    if q.dim() == 2:
        q = q.view(num_actual, state.config.num_query_heads, state.config.head_dim)
    if k.dim() == 2:
        k = k.view(num_actual, state.config.num_kv_heads, state.config.head_dim)
        v = v.view(num_actual, state.config.num_kv_heads, state.config.head_dim)

    if state.config.num_query_heads != state.config.num_kv_heads:
        repeats = state.config.num_query_heads // state.config.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    q_t = q.unsqueeze(0).transpose(1, 2)
    k_t = k.unsqueeze(0).transpose(1, 2)
    v_t = v.unsqueeze(0).transpose(1, 2)

    scale = getattr(self_impl, "scale", 1.0 / math.sqrt(state.config.head_dim))
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, scale=scale)
    return out.squeeze(0).transpose(0, 1)


def _make_patched_forward(orig_fn, state: LayerState, no_alloc: bool = False, capture_in_forward: bool = False):
    """Intercept forward to optionally use TQ decode.

    If capture_in_forward=True, also capture K/V from forward args
    (needed when the backend has no separate do_kv_cache_update method).
    """

    def _capture_kv(key, value, attn_metadata):
        """Capture K/V tensors into TQ store."""
        num_tokens = getattr(attn_metadata, "num_actual_tokens", key.shape[0])
        _trace(state.config.layer_idx, f"capture_kv num_tokens={num_tokens} via_forward=True")
        if num_tokens <= 1:
            state.engine.ingest_decode(key[:num_tokens], value[:num_tokens], num_tokens)
        else:
            state.engine.ingest_prefill(key[:num_tokens], value[:num_tokens], num_tokens)

    def _emit_forward_trace(branch: str, query, attn_metadata, mode: str) -> None:
        if not _trace_enabled():
            return
        max_q = getattr(attn_metadata, "max_query_len", -1) if attn_metadata is not None else -1
        _trace(
            state.config.layer_idx,
            f"forward q_shape={tuple(query.shape)} max_query_len={max_q} mode={mode} branch={branch}",
        )

    def patched(
        self_impl,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        mode = _GLOBAL_MODE

        # Capture K/V when no separate kv_update hook exists
        if capture_in_forward and mode not in (MODE_OFF,) and attn_metadata is not None:
            _capture_kv(key, value, attn_metadata)

        # Off or capture-only: always use flash
        if mode in (MODE_OFF, MODE_CAPTURE_ONLY):
            _emit_forward_trace(branch=mode, query=query, attn_metadata=attn_metadata, mode=mode)
            return orig_fn(
                self_impl,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        # Profiling pass or prefill: use flash
        if attn_metadata is None:
            _emit_forward_trace(branch="profile", query=query, attn_metadata=None, mode=mode)
            return orig_fn(
                self_impl,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        is_prefill = attn_metadata.max_query_len > 1
        if is_prefill:
            if no_alloc:
                _emit_forward_trace(branch="prefill_no_alloc", query=query, attn_metadata=attn_metadata, mode=mode)
                result = _no_alloc_prefill_attention(state, self_impl, query, key, value, attn_metadata)
                num_actual = attn_metadata.num_actual_tokens
                result_flat = result.reshape(num_actual, state.config.num_query_heads * state.config.head_dim).to(
                    query.dtype
                )
                if output is not None:
                    out_slice = output[:num_actual]
                    if out_slice.dim() == 3:
                        out_slice.copy_(result.to(out_slice.dtype))
                    else:
                        out_slice.copy_(result_flat.to(out_slice.dtype))
                    return output
                if query.dim() == 3:
                    return result.to(query.dtype)
                return result_flat
            _emit_forward_trace(branch="prefill_passthrough", query=query, attn_metadata=attn_metadata, mode=mode)
            return orig_fn(
                self_impl,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        # --- Hybrid decode ---
        if mode == MODE_HYBRID and state.supports_hybrid:
            flat = state.store.get_flat_cache()
            flat_n = flat.num_tokens if flat is not None else 0
            ring_n = state.engine.ring.size
            took_compressed = flat is not None and flat.num_tokens >= 16
            _trace(
                state.config.layer_idx,
                f"hybrid_decision flat_num_tokens={flat_n} ring_size={ring_n} took_compressed_path={took_compressed}",
            )
            if took_compressed:
                _emit_forward_trace(branch="hybrid_compressed", query=query, attn_metadata=attn_metadata, mode=mode)
                num_actual = attn_metadata.num_actual_tokens
                q = query[:num_actual]
                if q.dim() == 2:
                    q = q.view(num_actual, state.config.num_query_heads, state.config.head_dim)

                # S3.2 — kv_paged segment: K/V for tokens not yet in kv_tq.
                # At hybrid forward time the post-execute paged-cache reader
                # (S1.3 / 1C) hasn't fired yet for the current step, so kv_tq
                # covers tokens 0..N-1 and the current decode token's K/V is
                # missing. The K/V is already in scope as the forward `key`/
                # `value` arguments — equivalent to paged_cache[slot_N] at
                # this point in the pipeline since vLLM has just written it
                # there — so we feed it through directly without re-reading
                # the paged cache. See docs/integration-state.md § "S3.1 —
                # Audit".
                k_paged = key[:num_actual]
                v_paged = value[:num_actual]
                if k_paged.dim() == 2:
                    k_paged = k_paged.view(num_actual, state.config.num_kv_heads, state.config.head_dim)
                    v_paged = v_paged.view(num_actual, state.config.num_kv_heads, state.config.head_dim)

                recent = state.engine.ring.peek()
                recent_k = recent[0] if recent else None
                recent_v = recent[1] if recent else None

                _trace(
                    state.config.layer_idx,
                    f"hybrid_segments num_paged={num_actual} num_tq={flat_n} num_ring={ring_n}",
                )

                result = compute_hybrid_attention(
                    query=q,
                    store=state.store,
                    recent_k=recent_k,
                    recent_v=recent_v,
                    num_query_heads=state.config.num_query_heads,
                    scale=getattr(self_impl, "scale", None),
                    kv_paged_k=k_paged,
                    kv_paged_v=v_paged,
                )

                result_flat = result.reshape(num_actual, state.config.num_query_heads * state.config.head_dim).to(
                    query.dtype
                )

                if output is not None:
                    out_slice = output[:num_actual]
                    if out_slice.dim() == 3:
                        out_slice.copy_(result.to(out_slice.dtype))
                    else:
                        out_slice.copy_(result_flat.to(out_slice.dtype))
                    return output
                if query.dim() == 3:
                    return result.to(query.dtype)
                return result_flat

        # Fallback to flash
        if no_alloc:
            _emit_forward_trace(branch="no_alloc_zeros", query=query, attn_metadata=attn_metadata, mode=mode)
            num_actual = getattr(attn_metadata, "num_actual_tokens", query.shape[0])
            if query.dim() == 3:
                return torch.zeros_like(query[:num_actual])
            return torch.zeros(
                num_actual,
                state.config.num_query_heads * state.config.head_dim,
                dtype=query.dtype,
                device=query.device,
            )
        _emit_forward_trace(branch="hybrid_fallback", query=query, attn_metadata=attn_metadata, mode=mode)
        return orig_fn(
            self_impl,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    return patched


def _make_patched_mla_update(orig_fn, state: LayerState):
    """MLA KV update — log-only, no TQ capture yet."""

    def patched(self_impl, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale):
        orig_fn(self_impl, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale)
        if state._log_count < 1:
            logger.info(f"[TurboQuant] MLA update observed on layer {state.config.layer_idx}; TQ MLA path is deferred.")
            state._log_count += 1

    return patched


def _make_patched_mla_forward(orig_fn, state: LayerState):
    """MLA forward — passthrough (unsupported)."""

    def patched(self_impl, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        return orig_fn(self_impl, q, kv_c_and_k_pe_cache, attn_metadata, layer)

    return patched


# ---------------------------------------------------------------------------
# Post-execute paged-cache reader (S1.3 / 1C — survives FULL CUDAGraph)
# ---------------------------------------------------------------------------


def _make_post_execute_callback(model_runner):
    """Build the per-step paged-cache reader callback for decode-only steps.

    Reads the ``slot_mappings`` snapshot left on
    ``model_runner.execute_model_state`` (vllm 0.17.1
    ``v1/worker/gpu_model_runner.py:3706``), gathers the just-written K/V
    from each TQ-hooked layer's paged ``kv_cache`` tensor, and feeds it into
    ``state.engine.ingest_decode``. Runs after ``execute_model`` returns —
    i.e. after ``cudagraph.replay()`` has handed control back to Python — so
    it survives FULL CUDAGraph replay where every Python-level hook inside
    the captured region is bypassed by construction (see
    ``docs/integration-state.md`` § "F1bis").
    """

    layer_states = model_runner._tq_layer_states
    static_ctx = model_runner.compilation_config.static_forward_context

    def callback(scheduler_output) -> None:
        if not layer_states:
            return

        # Gate: only run on decode-only steps. Prefill / chunked-prefill
        # steps run under PIECEWISE mode and are already ingested by the
        # do_kv_cache_update monkey-patch. Letting both fire would
        # double-ingest the chunk.
        ns = getattr(scheduler_output, "num_scheduled_tokens", None)
        if not ns:
            return
        if max(ns.values()) > 1:
            return
        num_actual = scheduler_output.total_num_scheduled_tokens
        if num_actual <= 0:
            return

        emstate = getattr(model_runner, "execute_model_state", None)
        if emstate is None:
            return
        slot_mappings = getattr(emstate, "slot_mappings", None)
        if not slot_mappings:
            return
        if isinstance(slot_mappings, list):
            sm_layer = slot_mappings[0] if slot_mappings else None
        else:
            sm_layer = slot_mappings
        if not isinstance(sm_layer, dict):
            return

        for layer_name, state in layer_states.items():
            if not state.supports_hybrid:
                continue
            attn_module = static_ctx.get(layer_name)
            if attn_module is None:
                continue
            kv_list = getattr(attn_module, "kv_cache", None)
            if not kv_list:
                continue
            kv_cache_tensor = kv_list[0]
            # Flash backend layout: (2, num_blocks, block_size, num_kv_heads, head_dim).
            # The first dim splits K and V; see
            # vllm/v1/attention/backends/flash_attn.py:791 (`kv_cache.unbind(0)`).
            if kv_cache_tensor.dim() != 5 or kv_cache_tensor.shape[0] != 2:
                continue

            slot_mapping = sm_layer.get(layer_name)
            if slot_mapping is None:
                continue
            slot_mapping = slot_mapping[:num_actual]
            if slot_mapping.dtype != torch.int64:
                slot_mapping = slot_mapping.to(torch.int64)

            num_kv_heads = kv_cache_tensor.shape[-2]
            head_dim = kv_cache_tensor.shape[-1]
            key_cache = kv_cache_tensor[0]
            value_cache = kv_cache_tensor[1]
            key_flat = key_cache.reshape(-1, num_kv_heads, head_dim)
            value_flat = value_cache.reshape(-1, num_kv_heads, head_dim)
            k = key_flat.index_select(0, slot_mapping)
            v = value_flat.index_select(0, slot_mapping)
            _trace(
                state.config.layer_idx,
                f"paged_read num_tokens={num_actual}",
            )
            state.engine.ingest_decode(k, v, num_actual)

    return callback


def install_post_execute_callback(model_runner) -> None:
    """Wrap ``model_runner.execute_model`` so a per-step paged-cache reader
    fires right after each forward pass returns.

    This is the FULL-CUDAGraph-safe decode-capture path landed in Path B
    Sprint 1 / S1.3 (plan-path-b.md §3 / S1.3, anchored on
    ``vllm/compilation/cuda_graph.py:208–323``). Idempotent — repeated calls
    are no-ops once the wrapper is installed.
    """

    if getattr(model_runner, "_tq_post_execute_installed", False):
        return
    callback = _make_post_execute_callback(model_runner)
    orig_execute_model = model_runner.execute_model

    def wrapped_execute_model(scheduler_output, *args, **kwargs):
        result = orig_execute_model(scheduler_output, *args, **kwargs)
        try:
            with torch.inference_mode():
                callback(scheduler_output)
        except Exception:
            logger.exception("[TurboQuant] post-execute paged-cache reader failed")
        return result

    model_runner.execute_model = wrapped_execute_model
    model_runner._tq_post_execute_installed = True
    logger.info("[TurboQuant] post-execute paged-cache reader installed (S1.3 / 1C)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install_hooks(
    model_runner,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    ring_capacity: int = 128,
    initial_layers_count: int = 4,
    initial_layers_key_bits: int | None = None,
    mode: str = MODE_CAPTURE_ONLY,
    no_alloc: bool = False,
) -> dict[str, LayerState]:
    """Install TurboQuant hooks on all attention layers in a vLLM model runner.

    Returns: dict mapping layer_name -> LayerState
    """
    global _GLOBAL_MODE
    _GLOBAL_MODE = mode

    if initial_layers_key_bits is None:
        initial_layers_key_bits = min(key_bits + 1, 4)

    static_ctx = model_runner.compilation_config.static_forward_context
    device = model_runner.device

    layer_states: dict[str, LayerState] = {}
    layer_idx = 0

    for layer_name, attn_module in static_ctx.items():
        if not hasattr(attn_module, "impl"):
            continue

        impl = attn_module.impl
        num_kv_heads = getattr(impl, "num_kv_heads", None)
        if num_kv_heads is None:
            continue

        if hasattr(impl, "head_size"):
            head_dim = int(impl.head_size)
        elif hasattr(impl, "kv_lora_rank"):
            head_dim = int(impl.kv_lora_rank)
        else:
            continue

        bits = initial_layers_key_bits if layer_idx < initial_layers_count else key_bits
        backend_kind = "mla" if _is_mla_impl(impl) else "flash"
        num_query_heads = _infer_num_query_heads(attn_module, impl)

        cfg = LayerConfig(
            head_dim=head_dim,
            num_kv_heads=int(num_kv_heads),
            num_query_heads=num_query_heads,
            key_bits=bits,
            value_bits=value_bits,
            value_group_size=min(value_group_size, head_dim),
            ring_capacity=ring_capacity,
            layer_idx=layer_idx,
            backend_kind=backend_kind,
            device=device,
        )

        state = _create_layer_state(cfg)
        layer_states[layer_name] = state

        _trace(
            layer_idx,
            f"install name={layer_name!r} backend={backend_kind} head_dim={head_dim} num_kv_heads={int(num_kv_heads)}",
        )

        if backend_kind == "flash":
            has_separate_kv_update = hasattr(impl, "do_kv_cache_update")
            needs_forward_capture = not has_separate_kv_update

            if has_separate_kv_update:
                patched_update = _make_patched_kv_update(impl.do_kv_cache_update.__func__, state, no_alloc=no_alloc)
                impl.do_kv_cache_update = types.MethodType(
                    lambda self, *a, _p=patched_update, **kw: _p(self, *a, **kw), impl
                )

            patched_forward = _make_patched_forward(
                impl.forward.__func__,
                state,
                no_alloc=no_alloc,
                capture_in_forward=needs_forward_capture,
            )
            impl.forward = types.MethodType(lambda self, *a, _p=patched_forward, **kw: _p(self, *a, **kw), impl)

            if needs_forward_capture and layer_idx == 0:
                logger.info(
                    "[TurboQuant] No do_kv_cache_update found (vLLM 0.16 FlashInfer); capturing K/V in forward()"
                )
        else:
            if hasattr(impl, "do_kv_cache_update"):
                patched_update = _make_patched_mla_update(impl.do_kv_cache_update.__func__, state)
                impl.do_kv_cache_update = types.MethodType(
                    lambda self, *a, _p=patched_update, **kw: _p(self, *a, **kw), impl
                )
            if hasattr(impl, "forward_mqa"):
                patched_fwd = _make_patched_mla_forward(impl.forward_mqa.__func__, state)
                impl.forward_mqa = types.MethodType(lambda self, *a, _p=patched_fwd, **kw: _p(self, *a, **kw), impl)

        impl._tq_layer_state = state
        layer_idx += 1

    model_runner._tq_layer_states = layer_states
    model_runner._tq_no_alloc = no_alloc
    logger.info(f"[TurboQuant] Hooks on {len(layer_states)} layers (mode={mode}, no_alloc={no_alloc})")
    install_post_execute_callback(model_runner)
    return layer_states


def free_kv_cache(model_runner) -> int:
    """Free paged KV cache for TQ-hooked layers. Returns bytes freed.

    Only frees layers that have TQ state. Non-TQ layers (MLA/GDN) keep their cache.
    """
    layer_states = getattr(model_runner, "_tq_layer_states", None)
    if not layer_states:
        logger.warning("[TurboQuant] No layer states found, nothing to free")
        return 0

    static_ctx = model_runner.compilation_config.static_forward_context
    device = model_runner.device
    freed = 0
    tiny = torch.zeros(1, dtype=torch.int8, device=device)

    ptrs_to_free = set()
    for layer_name, state in layer_states.items():
        if not state.supports_hybrid:
            continue
        if layer_name not in static_ctx:
            continue
        attn_module = static_ctx[layer_name]
        kv_list = getattr(attn_module, "kv_cache", None)
        if kv_list and len(kv_list) > 0:
            ptrs_to_free.add(kv_list[0].data_ptr())

    for layer_name, state in layer_states.items():
        if not state.supports_hybrid:
            continue
        if layer_name not in static_ctx:
            continue
        attn_module = static_ctx[layer_name]
        kv_list = getattr(attn_module, "kv_cache", None)
        if kv_list and len(kv_list) > 0:
            old = kv_list[0]
            this_freed = old.nelement() * old.element_size()
            freed += this_freed
            kv_list[0] = tiny
            _trace(state.config.layer_idx, f"free_kv_cache freed_bytes={this_freed}")

    for i in range(len(model_runner.kv_caches)):
        entry = model_runner.kv_caches[i]
        if isinstance(entry, list):
            for j in range(len(entry)):
                if hasattr(entry[j], "data_ptr") and entry[j].data_ptr() in ptrs_to_free:
                    entry[j] = tiny
        elif hasattr(entry, "data_ptr") and entry.data_ptr() in ptrs_to_free:
            model_runner.kv_caches[i] = tiny

    torch.cuda.empty_cache()
    logger.info(f"[TurboQuant] Freed {freed / 1e6:.0f} MB KV cache ({len(layer_states)} layers)")
    return freed


def get_stats(model_runner) -> dict:
    """Return summary statistics for all TQ layer states."""
    layer_states = getattr(model_runner, "_tq_layer_states", None)
    if not layer_states:
        return {}

    stats = {}
    total_compressed = 0
    total_buffered = 0
    total_memory = 0

    for name, state in layer_states.items():
        compressed = state.store.num_tokens
        buffered = state.engine.ring.size
        mem = state.store.memory_bytes()
        total_compressed += compressed
        total_buffered += buffered
        total_memory += mem

    stats["num_layers"] = len(layer_states)
    stats["total_compressed_tokens"] = total_compressed // max(len(layer_states), 1)
    stats["total_buffered_tokens"] = total_buffered // max(len(layer_states), 1)
    stats["total_memory_bytes"] = total_memory
    stats["mode"] = _GLOBAL_MODE
    return stats
