# TurboQuant ↔ vLLM 0.17.x integration — observed state

This document is the canonical statement of what the vLLM integration does
and does not do today. Each finding cites specific line numbers from the
three `docs/traces/s0_*.log` evidence files committed in
[b95a580](https://github.com/pitcany/vllm-turboquant/commit/b95a580). It
replaces the inferred conclusions in the README's `⚠️` notice (which
themselves came from an in-session grep, not a captured artifact) per
[`docs/plan-path-b.md`](plan-path-b.md) §3 / S0.3.

The traces were captured on the same workload —
`MODE=hybrid LONG_PROMPT=1 BUFFER_SIZE=16 MAX_TOKENS=64` against
`Qwen/Qwen2.5-0.5B-Instruct` on an RTX 5090 — varying only
`enforce_eager` and `enable_prefix_caching` on `vllm.LLM(...)`. The
prompt tokenises to **226 tokens**; we then decode **64 more**. So the
intended capture-window is ~290 token-positions × 24 attention layers.

| Trace | `enforce_eager` | `enable_prefix_caching` | Compressed tokens after 64-decode | Notes |
|---|---|---|---|---|
| `s0_compiled.log` | False (default) | True (default) | **0** | F1 baseline |
| `s0_eager.log` | True | True (default) | **64** | F1 fixed → F2 visible |
| `s0_eager_no_prefix.log` | True | False | **274** | F1+F2 fixed → F3 visible |

The tokens-captured numbers are read directly from `>>> TQ stats per
worker` lines in each trace: `s0_compiled.log:134`, `s0_eager.log:4679`,
`s0_eager_no_prefix.log:4664`.

---

## F1 — `torch.compile` bypasses the Python `do_kv_cache_update` hook

**Hypothesis (from README ⚠️).** Under default vLLM, the attention graph
compiles via `compilation_config.mode = VLLM_COMPILE` and KV writes are
serviced by a compiled `vllm::unified_kv_cache_update` op that does not
call back into the Python `Impl.do_kv_cache_update` method TurboQuant
monkey-patches.

**Trace evidence.**

- Pass 2 (TQ-instrumented decode) begins at
  [`s0_compiled.log:80`](traces/s0_compiled.log) (`>>> Pass 2/2:
  TurboQuant hybrid`).
- Across the entire 64-token decode, the patched `do_kv_cache_update`
  fires **exactly 24 times** — one batched `slot_mapping=2` call per
  layer at lines 82–128 (`grep -n 'kv_update slot_mapping' s0_compiled.log`
  returns these 24 lines and only these). **Zero** records with
  `slot_mapping=1` (`grep -c 'kv_update slot_mapping=1' s0_compiled.log` →
  0).
- Likewise, `_emit_forward_trace` records exactly 24 `branch=*` lines
  for the entire run (`grep -c 'branch=' s0_compiled.log` → 24, all
  `branch=prefill_passthrough`). **No** `hybrid_decision`, **no**
  `branch=hybrid_compressed` — the hybrid decode path is never reached
  because `forward` is never even called per-token.
- Final stats line confirms the consequence:
  [`s0_compiled.log:134`](traces/s0_compiled.log) →
  `'total_compressed_tokens': 0, 'total_buffered_tokens': 2`.

**Comparison.** With `enforce_eager=True` the same hook is invoked
1,536 times for the same workload — `kv_update slot_mapping=1` × 1,512
(63 decode tokens × 24 layers) plus `slot_mapping=2` × 24
([`s0_eager.log:76`](traces/s0_eager.log) onward). The contrast is the
direct measurement of "compile kills the hook."

**Conclusion.** F1 is reproduced, the failure mode is the one the README
described, and it is gated on `enforce_eager=False` exactly. The
24 prefill calls in `s0_compiled.log` are the warmup batch that runs
once before the inductor graph is cached; after that the compiled op
takes over.

---

## F2 — Prefix caching evades capture even in eager mode

**Hypothesis (from README ⚠️).** Even with eager attention, vLLM's
default-on prefix cache reuses the prior request's KV blocks at
`vllm.LLM(enable_prefix_caching=True)`. Pass 2 of `quickstart.py` reuses
Pass 1's prefill blocks, so TQ's `do_kv_cache_update` is never called for
those positions and the prefill never lands in the compressed store.

**Trace evidence.**

- Workload prompt is 226 tokens (visible from
  [`s0_eager_no_prefix.log:4664`](traces/s0_eager_no_prefix.log) which
  reports `total_compressed_tokens: 274` ≈ 226 prefill + 63 decode − 15 in
  ring buffer; and from the `slot_mapping=226` prefill line below).
- With prefix caching **on**
  ([`s0_eager.log`](traces/s0_eager.log)) the only prefill `kv_update`
  records have `slot_mapping=2`, lines 76–122 (one per layer, 24 total).
  vLLM's default block size is 16, so 226 tokens form 14 cached blocks
  of 16 + a partial block of 2; only the partial block fires
  `kv_update`. Final stats:
  [`s0_eager.log:4679`](traces/s0_eager.log) →
  `'total_compressed_tokens': 64` — i.e. the 63 decode tokens plus the
  first one ingested via the warmup `slot_mapping=2`. The 224 cached
  prefill tokens **never enter the TQ store**.
- With prefix caching **off**
  ([`s0_eager_no_prefix.log:76`](traces/s0_eager_no_prefix.log) onward),
  the same 24 lines instead read `kv_update slot_mapping=226` — the
  full prefill is captured. Final stats:
  [`s0_eager_no_prefix.log:4664`](traces/s0_eager_no_prefix.log) →
  `'total_compressed_tokens': 274`.

**Conclusion.** F2 is reproduced. The exact-2-tokens-of-prefill artefact
is a *cleaner* signal than the README's "doesn't see the prefill"
phrasing: it shows the partial-block remainder that vLLM's prefix cache
can't share, which is the only reason any prefill lands in TQ at all in
the prefix-caching path.

---

## F3 — Hybrid decode ignores the paged `kv_cache` tensor

**Hypothesis (from README ⚠️).** When `mode="hybrid"` fires its compressed
branch, it computes attention against `state.store.get_flat_cache()` and
`state.engine.ring.peek()` only — never reading the paged `kv_cache`
tensor that holds the actual prompt's K/V. As a result, even when capture
*does* succeed, the hybrid output is degenerate because the model is
attending to a lossy compressed copy of history instead of the canonical
paged cache.

**Trace evidence.** This is the finding `s0_eager_no_prefix.log` exists
to settle, because that trace eliminates F1 and F2 as confounds:

- 274 tokens are in the compressed store at decode-time, covering
  the full 226-token prefill + most of the decode
  ([`s0_eager_no_prefix.log:4664`](traces/s0_eager_no_prefix.log)).
- Every single decode step takes the `hybrid_compressed` branch:
  `grep -c 'branch=hybrid_compressed' s0_eager_no_prefix.log` → **1,512**
  (= 63 decode steps × 24 layers; first decoded token's `forward`
  trace is suppressed by the warmup pass). Sample:
  [`s0_eager_no_prefix.log:3581`](traces/s0_eager_no_prefix.log)
  `hybrid_decision flat_num_tokens=274 ring_size=1
  took_compressed_path=True`.
- Despite that, the TQ output is degenerate:
  [`s0_eager_no_prefix.log:4672`](traces/s0_eager_no_prefix.log)
  `tq tail: ...'zation, quantization;q, quant, quant, quant, quant,
  quant, quant, quant, quant, quant, quant, quant, quant, quant, quant'`
  — vs. the baseline pass at line 4671
  `baseline tail: ...'se a consistent conclusion. ...'`.
  `same output text: False`
  ([`s0_eager_no_prefix.log:4670`](traces/s0_eager_no_prefix.log)).

**Code-level confirmation.** The `kv_cache` tensor is in scope at the
hybrid-compressed branch but never consumed:
[`turboquant/integration/vllm.py:257`](../turboquant/integration/vllm.py)
takes `kv_cache` as a formal of `patched`,
[`turboquant/integration/vllm.py:355`](../turboquant/integration/vllm.py)
calls

```python
result = compute_hybrid_attention(
    query=q,
    store=state.store,
    recent_k=recent_k,
    recent_v=recent_v,
    num_query_heads=state.config.num_query_heads,
    scale=getattr(self_impl, "scale", None),
)
```

— no `kv_cache` argument. And the receiving signature
[`turboquant/score.py:31`](../turboquant/score.py) `def
compute_hybrid_attention(query, store, recent_k, recent_v,
num_query_heads, scale=None)` confirms the function couldn't read the
paged cache even if we passed it.

**Conclusion.** F3 is reproduced and is genuinely a separate bug from F1
and F2. Even with a fully populated TQ store and zero capture losses, the
hybrid decode path produces degenerate output because the implementation
is structurally incapable of attending across `kv_paged ∪ kv_tq ∪
kv_ring`. This is the bug Sprint 3 of `plan-path-b.md` is scoped to fix.

---

## Throughput observations (incidental)

The plan does not gate Sprint 0 on perf, but the captured `tok/s` numbers
are worth recording so the "is TQ faster?" question never gets answered
from memory again:

| Config | Baseline tok/s | TQ tok/s | TQ ratio |
|---|---:|---:|---:|
| Compiled (default) | 449.5 | 638.0 | **1.42×** ([`s0_compiled.log:139`](traces/s0_compiled.log)) |
| Eager + prefix | 144.2 | 65.3 | 0.45× ([`s0_eager.log:4684`](traces/s0_eager.log)) |
| Eager, no prefix | 145.4 | 58.2 | 0.40× ([`s0_eager_no_prefix.log:4669`](traces/s0_eager_no_prefix.log)) |

The compiled "1.42× speedup" is misleading: as F1 above shows, **TQ does
nothing under that configuration** (compressed_tokens = 0). The TQ pass
runs faster than baseline because the compiled CUDA-graph path is faster
than vLLM's PIECEWISE init path on a second request, not because TQ is
contributing anything. In the two configurations where TQ actually
intercepts the decode, it is 0.40–0.45× of baseline — i.e. ~2× *slower*,
because `compute_hybrid_attention` runs an unfused `torch.einsum` over
dequantised float32 history per decode step. This is consistent with the
"hybrid decode dequantizes all history" caveat in the README's
*Limitations* section.

---

## Definitions of done — what would close each finding

Per [`docs/plan-path-b.md`](plan-path-b.md) §4, each finding closes when
a positive statement of the form *"as of commit SHA, this works under
config X, evidenced by trace `docs/traces/Y.log:Z`"* replaces the
hypothesis above.

| Finding | Closure criterion |
|---|---|
| F1 | A future `s1_compiled.log` shows ≥ 256 `kv_update slot_mapping=1` records (or equivalent) under default vLLM (`enforce_eager=False`), and the captured-token count matches input + output within 1%. |
| F2 | Either (a) a future `s2_*.log` shows `enable_turboquant` raising `TurboQuantVLLMError` when `cache_config.enable_prefix_caching` is `True`, **or** (b) a future `s2_*.log` shows the patched `do_kv_cache_update` *also* firing on prefix-cache hits with the cached K/V re-injected. (Sprint 2 takes path (a); path (b) is a follow-up.) |
| F3 | A future `s3_compiled.log` (or `s3_eager_no_prefix.log`) shows `top-1 token agreement ≥ 95%` between baseline vLLM and TQ-hybrid on a 256-token greedy decode against `Llama-3.2-1B-Instruct`, evidenced by `tests/test_correctness_e2e.py` passing. |

When all three rows are checked, the README's `⚠️` notice gets rewritten
into the "Verified configuration" section per Sprint 4.

---

## Reproducer

All three traces were produced by the same `quickstart.py`. From a clean
checkout in the `vllm-serve` conda env on a CUDA-capable host:

```bash
TURBOQUANT_TRACE=1 \
  [ENFORCE_EAGER=1] [ENABLE_PREFIX_CACHING=0] \
  MODE=hybrid LONG_PROMPT=1 BUFFER_SIZE=16 MAX_TOKENS=64 \
  CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib" \
  python quickstart.py > docs/traces/s0_<name>.log 2>&1
```

The `LD_LIBRARY_PATH` is needed on systems where the system `libstdc++`
predates the one that ships with the conda env (vLLM 0.17.x uses
`std::expected` from C++23). `VLLM_ALLOW_INSECURE_SERIALIZATION=1` is
set automatically at import time by `turboquant.vllm`.
