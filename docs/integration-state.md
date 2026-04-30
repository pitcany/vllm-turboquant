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

### S1.1 — `torch.library` override of `vllm::unified_kv_cache_update` is structurally blocked

Path B Sprint 1 / S1.1 attempted the plan's preferred fix (1B): re-register
`vllm::unified_kv_cache_update` via `torch.library` so the override calls
the original *and* writes a copy of `(key, value, slot_mapping)` into TQ's
store on every dispatch. This was abandoned at the recon step on
2026-04-29 — before any code change — because torch's dispatcher
refuses double-registration and exposes no public deregister API.

**Op signature (recon, not docstring).** Captured directly from the live
runtime:

```text
$ python -c "import torch, vllm.model_executor.layers.attention.attention; \
             print(torch._C._dispatch_dump('vllm::unified_kv_cache_update'))"
name: vllm::unified_kv_cache_update
schema: vllm::unified_kv_cache_update(Tensor key, Tensor value, str layer_name) -> Tensor
debug: registered at .../vllm/utils/torch_utils.py:789
alias analysis kind: FROM_SCHEMA
CUDA: registered at .../vllm/utils/torch_utils.py:789 :: (none) [ boxed ]
Meta: registered at .../vllm/utils/torch_utils.py:789 :: (none) [ boxed ]
```

The op is registered at module-load time of
`vllm.model_executor.layers.attention.attention` via
`direct_register_custom_op` (see `vllm/utils/torch_utils.py:792`),
which calls `vllm_lib.impl(op_name, op_func, dispatch_key="CUDA")` on a
single process-wide `vllm_lib = Library("vllm", "FRAGMENT")`
(`vllm/utils/torch_utils.py:789`). Both the `CUDA` backend impl and the
`Meta` fake impl are registered there.

**Re-registration error.** Re-registering on the same op + dispatch key
fails:

```text
$ python -c "
> import torch
> import vllm.model_executor.layers.attention.attention
> my_lib = torch.library.Library('vllm', 'IMPL')
> my_lib.impl('unified_kv_cache_update', lambda *a: None, 'CUDA')
> "
RuntimeError: This is not allowed since there's already a kernel
registered from python overriding unified_kv_cache_update's behavior
for CUDA dispatch key and vllm namespace.
```

This matches torch's documented one-impl-per-(op, dispatch-key) policy.

**Module-level patch is also stale.** Replacing
`vllm.model_executor.layers.attention.attention.unified_kv_cache_update`
on the module *after* registration has zero effect on calls dispatched
via `torch.ops.vllm.unified_kv_cache_update(...)`: the `Library.impl(...)`
call captured the original function reference, not a name lookup.

**No public deregister API.** Surveying `dir(torch._C)` for dispatch-related
APIs (run on torch 2.10.0 in this env), the only deregister-shaped name
is `_unset_dispatch_mode` (a TLS guard, unrelated). The only viable
"replace" path is `Library._destroy()`, which destroys *every*
registration on the library — and `vllm_lib` is the namespace owner for
all custom vLLM ops, so destroying it would unregister
`unified_attention`, `unified_attention_with_output`, `unified_mla_*`,
`mamba_mixer*`, `gdn_attention_core`, `kda_attention`, etc. That is the
plan's "bundling" anti-pattern (§1.8) at the worst possible scale.

**Conclusion (per plan §3 / Sprint 1, S1.2 stop-condition).** S1.1 is
structurally blocked by torch's dispatcher invariants, not by anything
specific to vLLM 0.17.1. Falling back to S1.2 (1A — forward pre-hooks
on the `Attention` `nn.Module` via `register_forward_pre_hook`).

### S1.2 — `register_forward_pre_hook` fires zero times after compilation

S1.2's 1A approach was probed empirically (commit pending) by
adding a temporary `attn_module.register_forward_pre_hook(_pre_hook)` in
[`turboquant/integration/vllm.py`](../turboquant/integration/vllm.py)'s
`install_hooks` (gated on `TURBOQUANT_PROBE_PRE_HOOK=1`), where
`attn_module` is the same vLLM `Attention` `nn.Module` whose `.impl` we
already monkey-patch. The probe was run on the cleanest S1 workload —
**default compile + `enable_prefix_caching=False`** — so that any
non-firing of the pre-hook isn't confounded by F2.

**Result.** The pre-hook fires **0 times** for the entire 226-prefill +
64-decode workload
([`docs/traces/s1_probe_pre_hook.log`](traces/s1_probe_pre_hook.log)
line counts):

```text
$ grep -c 'pre_hook fired' s1_probe_pre_hook.log
0
$ grep -c 'kv_update slot_mapping' s1_probe_pre_hook.log
24
```

The 24 `kv_update slot_mapping=226` lines (`s1_probe_pre_hook.log:80–126`)
prove the existing `impl.do_kv_cache_update` monkey-patch *does* still
fire during prefill, even under default compile — so the pre-hook's zero
count isn't because no attention is happening; it's because the
`nn.Module`-level hook iteration is being skipped entirely.

**Why pre-hooks don't fire.** vLLM's `compilation_config.mode =
VLLM_COMPILE` traces each `Attention.forward` via
`torch._dynamo.optimize` at model-load time. Dynamo inlines the `forward`
body into the captured graph; the compiled graph then calls `forward`
directly rather than going back through `nn.Module.__call__` (which is
where `_forward_pre_hooks` is iterated). Any pre-hook registered *after*
compilation — and `enable_turboquant` is necessarily after-compilation —
is held in the module's hook list but never read. This is a documented
behavior of the inductor frontend, not a vLLM-specific bug.

**Implication for the plan.** The S1.2 / 1A approach as written is not
viable on a model that vLLM has already compiled. Re-registering hooks
*before* compilation would require landing TQ as a vLLM plugin (executed
during `LLM.__init__`), which is a much larger architectural change than
S1.2 contemplates and crosses into the §5 stop-loss territory ("the
integration approach itself may be wrong").

---

## F1bis — F1 is overstated; default compile + no-prefix-cache *does* capture the prefill

The same probe trace incidentally settles a different question: **what
does default compile actually look like once F2 is removed as a confound?**

The original `s0_compiled.log` ran with `enable_prefix_caching=True` (the
vLLM default). That entangles two effects:
- F1 — torch.compile's CUDA-graph replay bypassing per-token decode
  hooks.
- F2 — prefix caching reusing pass-1 KV blocks, leaving only 2 unseen
  prefill tokens for `do_kv_cache_update` to even see.

`s1_probe_pre_hook.log` keeps everything else at default compile but sets
`enable_prefix_caching=False`. The result:

| Metric | `s0_compiled.log` (prefix=ON) | `s1_probe_pre_hook.log` (prefix=OFF) |
|---|---:|---:|
| `kv_update` trace lines | 24 (all `slot_mapping=2`) | 24 (all `slot_mapping=226`) |
| `forward` trace lines | 24 (all `prefill_passthrough`) | 24 (all `prefill_passthrough`) |
| `total_compressed_tokens` (final) | **0** ([`s0_compiled.log:134`](traces/s0_compiled.log)) | **210** ([`s1_probe_pre_hook.log:132`](traces/s1_probe_pre_hook.log)) |
| `total_buffered_tokens` (final) | 2 | 16 |
| Captured = prompt size? | No (0/226) | **Yes** (210 + 16 = 226) |
| `same output text` | False ([`s0_compiled.log:140`](traces/s0_compiled.log)) | **True** ([`s1_probe_pre_hook.log:138`](traces/s1_probe_pre_hook.log)) |
| Decode tok/s | 638.0 | 305.0 |

So under default compile **without prefix caching**:

- The full 226-token prefill *is* captured into the TQ store on a single
  PIECEWISE-mode call (`slot_mapping=226`) per layer. This invalidates
  F1's "after that the compiled op takes over" reading: prefill is
  always on the PIECEWISE path (it doesn't match the captured
  CUDA-graph batch sizes [1, 2]) and runs the patched
  `do_kv_cache_update` exactly as in eager mode.
- Decode bypass is real but *invisible to correctness*: the 64
  decode-step `forward` hook never fires (FULL CUDAGraph replay), so
  vLLM's compiled flash-attention path runs unmodified, so output
  matches the baseline byte-for-byte. The TQ store does not grow during
  decode, but that doesn't show up as a wrong answer because the
  hybrid-compressed branch is never *entered*.
- The "1.42× speedup" reported in the original
  `s0_compiled.log` is gone (305 vs. 638 tok/s = 0.48× ratio). The
  speedup was an artefact of the prefix cache serving 224/226 prefill
  tokens; once prefill is doing real work and TQ is ingesting it,
  capture overhead dominates the 64-decode workload. This is consistent
  with the 0.40–0.45× ratios already observed under eager mode for the
  same reason (`compute_hybrid_attention` overhead on the
  hybrid-compressed branch — except here the hybrid branch isn't even
  taken; the cost is purely the per-prefill `ingest_prefill` quantize
  pass).

**Refined statement of F1.** The decode-time hook bypass is a property
of vLLM's **FULL CUDAGraph mode** specifically, not of the Python op
`vllm::unified_kv_cache_update`:

```text
vllm/compilation/cuda_graph.py:208–323  (CUDAGraphWrapper.__call__)

  if entry.cudagraph is None:                  # capture path
      with torch.cuda.graph(cudagraph, ...):
          output = self.runnable(*args, **kwargs)   # Python runs ONCE
      entry.cudagraph = cudagraph
      ...
  ...
  entry.cudagraph.replay()                     # all later calls
  return entry.output
```

Once `entry.cudagraph` is populated for a (mode, batch_descriptor)
key, every subsequent matching call is a `cudagraph.replay()` — pure
CUDA-stream replay, no Python. The kernels recorded inside the graph
(`reshape_and_cache_flash`, `flash_attn_varlen`, etc.) execute, so the
*paged* `kv_cache` tensor still gets new K/V written every decode step,
but no Python instrumentation between those kernel launches survives.
This is why **only** decode (which matches batch=1, mode=FULL per
[`s0_compiled.log:41`](traces/s0_compiled.log)) is bypassed: prefill
runs PIECEWISE (per [`s0_compiled.log:40`](traces/s0_compiled.log)) and
the splitting-op boundaries between piecewise chunks are normal Python
calls, where our patches fire.

**Implication for the plan.** §3 / Sprint 1 was written as if the bypass
were op-specific — that overriding the Python implementation of
`vllm::unified_kv_cache_update` would route per-decode-step writes through
TQ. The actual mechanism (CUDA-graph replay) routes around any Python
hook by construction:

- 1B (torch.library override) — blocked structurally as documented above,
  *and* would not have helped: a Python-level override still wouldn't
  execute under graph replay.
- 1A (forward pre-hooks) — empirically blocked: zero firings, even
  during prefill. (Different cause: dynamo inlining `forward` into the
  compiled graph.)
- 1C (paged-cache reader) — the only path that survives FULL CUDAGraph,
  because it doesn't try to instrument anything inside the captured
  region. Read the paged `kv_cache` tensor each step from a Python
  entry point that *does* run every step (e.g., immediately after
  `model_runner.execute_model()` returns, on the worker), and migrate
  the per-step delta into TQ's store.

**Stop point.** Per `docs/plan-path-b.md` §3 / Sprint 1's stop-condition
("Anything in `s1_compiled.log` contradicts the entry condition (e.g.,
the bypass turns out *not* to be the `unified_kv_cache_update` op) →
stop, write what you saw into `docs/integration-state.md`, and ask
before changing the plan"), this section is the "what I saw"; the plan
needs an editing pass before code changes resume.

### S1.3 — 1C lands; capture survives FULL CUDAGraph (with one runtime caveat)

S1.3 / 1C — the post-`execute_model` paged-cache reader — landed on
2026-04-29 in commit
[`304ba1f`](https://github.com/pitcany/vllm-turboquant/commit/304ba1f)
and ran end-to-end on the same `MODE=hybrid LONG_PROMPT=1
BUFFER_SIZE=16` workload that produced `s1_probe_pre_hook.log`. Two
traces are now checked in:

| Metric | `s1_probe_pre_hook.log` (pre-S1.3) | `s1_compiled.log` (post-S1.3, `MAX_TOKENS=64`) | `s1_compiled_max_tokens_65.log` (post-S1.3, `MAX_TOKENS=65`) |
|---|---:|---:|---:|
| `kv_update slot_mapping=226` | 24 | 24 | 24 |
| `paged_read num_tokens=1` | 0 | **1512** (= 63 × 24) | **1536** (= 64 × 24) |
| `total_compressed_tokens` (final) | 210 | **274** ([`s1_compiled.log:1645`](traces/s1_compiled.log)) | **274** ([`s1_compiled_max_tokens_65.log:1669`](traces/s1_compiled_max_tokens_65.log)) |
| `total_buffered_tokens` (final) | 16 | 15 | 16 |
| Compressed + buffered | 226 | **289** | **290** |
| `same output text` | True | True | True |
| Decode tok/s | 305.0 | 242.7 | 239.2 |

The capture path now works. Specifically:

- The 24 prefill `kv_update slot_mapping=226` lines (per layer) confirm
  the existing `do_kv_cache_update` monkey-patch still fires during
  prefill on PIECEWISE mode. 1C is additive on prefill, not a
  replacement, and the monkey-patch's decode half is now retired so the
  two paths can't double-ingest.
- The 1512 / 1536 `paged_read` trace lines confirm the post-`execute_model`
  callback fires once per layer per decode step — it survives FULL
  CUDAGraph by reading the paged `kv_cache` tensor *after*
  `cudagraph.replay()` has handed control back to Python.
- "Same output text" stays True for the same reason it was True in
  `s1_probe_pre_hook.log`: under FULL CUDAGraph the `forward()` patch
  still never fires on decode, so the hybrid-compressed branch is never
  *entered*; vLLM's flash-attn runs unmodified inside the captured graph
  and emits the baseline output byte-for-byte. 1C captures into the TQ
  store on the side. The hybrid-attention rewrite that consumes the
  captured store is the Sprint 3 problem (F3), not an S1.3 regression.

**The off-by-one against the plan's 290 bar is a vLLM v1 runtime fact,
not a capture bug.** vLLM v1 emits 63 (not 64) decode `execute_model`
calls for `max_tokens=64`: prefill samples output token 0; each later
decode step writes the K/V of the *previous* sampled token and samples
the next; max_tokens halts generation immediately after the 64th sample
without a 65th decode call, so the K/V of the 64th sampled token is
never written. The `paged_read` count
(`grep -c paged_read s1_compiled.log` → 1512 = 63 × 24) is the direct
measurement. With `MAX_TOKENS=65` the same workload produces 1536
paged_reads and the store sums to exactly 290 — i.e., 1C scales
linearly with decode-K/V writes; the plan's bar IS reachable, just one
output token short of the canonical workload's budget.

**Decode tok/s** is now 242.7 vs. baseline 449.0 = **0.54×** at
~226-token context on Qwen2.5-0.5B. The plan's ≥ 0.7× tok/s bar
specifically applies to Llama-3.2-1B at 4k context (gated on HF Hub
access, not yet granted for `pitcany`). The 0.54× number on Qwen-0.5B
is dominated by the per-prefill `ingest_prefill` quantize pass (24
layers × 226 tokens) being amortised over only 64 decode tokens; that
ratio improves naturally with longer decode budgets and larger models.
Whether 1C's per-step paged-cache gather adds material overhead is best
measured on the Llama 4k workload in a follow-up commit; the trace
shows no per-layer CPU-GPU sync (gather is `tensor.index_select` on
GPU-resident `slot_mapping[:num_actual]`, where `num_actual` is a
host-side `int`).

**Open question for plan §3 / Sprint 1 acceptance.** The bar reads
"≥ 290 with `MAX_TOKENS=64`". Strict reading: 289 ≠ ≥ 290. Runtime
reading: 289 is the maximum achievable on `MAX_TOKENS=64` (vLLM v1
writes 63 decode K/Vs); the bar's mental model assumed 64 decode K/Vs.
The plan's working principle 1.6 ("trust the runtime, not the
docstring") points to relaxing the bar to "≥ 289 with `MAX_TOKENS=64`"
or moving the canonical workload to `MAX_TOKENS=65`. This file records
the observation; the plan edit is deferred until the user picks one.

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
