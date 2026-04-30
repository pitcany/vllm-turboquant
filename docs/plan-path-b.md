# Path B — Plan: Make the vLLM Integration Actually Work

> ## 🛑 SUPERSEDED — TurboQuant landed upstream in vLLM 0.20.0 (2026-04-30)
>
> This plan is **archived**. TurboQuant has been merged into vLLM mainline as
> a v1 attention backend (PRs
> [#38479](https://github.com/vllm-project/vllm/pull/38479),
> [#40092](https://github.com/vllm-project/vllm/pull/40092)) and ships in
> [vLLM v0.20.0](https://github.com/vllm-project/vllm/releases/tag/v0.20.0).
> The upstream integration takes the §5 first-bullet "re-architect as a
> vLLM plugin / attention backend" path this plan contemplated as a
> stop-loss option — and which we explicitly didn't take.
>
> Sprint 1, 2, 3 closed (with Sprint 3 hitting its own §5 second-bullet
> stop-loss on bit-budget quality at 1B scale). **Sprint 4 and Sprint 5
> are now N/A** — investing in this repo's monkey-patch surface is a
> dead end vs upstream's attention-backend implementation. See README's
> SUPERSEDED notice for the migration path and the design-decision diff
> against upstream.
>
> The plan body below is preserved verbatim as a research record of how
> Path B was scoped, what it found, and where it stopped.

This document is the contract for the next phase of work on
`pitcany/vllm-turboquant`. Read it before opening a Path B PR.

The README's top-of-file ⚠️ notice describes three independent issues that
together mean the vLLM integration does nothing useful on a default vLLM
configuration. Path B is the project to fix those, *truthfully*, with
end-to-end correctness verified on a real, public, downloadable model.

This plan supersedes the loose discussion in chat. Where they conflict,
this document wins. Where this document is silent, defer to the
working principles below.

---

## 1. Working principles

These are non-negotiable. They exist because Path A surfaced findings
the previous integration claims contradicted, and we don't want that to
happen again.

1. **No fix without a log-backed hypothesis.** Every change must be
   motivated by an observed log line, stack trace, captured tensor
   value, or counter — not by reasoning about what *should* happen.
2. **Reproduce before changing.** Every claim of "X is broken" needs a
   reproducer command + captured output checked into `docs/traces/`
   before we touch the code that handles X.
3. **Diagnostic logging at every interception point.** The integration
   layer must emit `logger.debug` at every monkey-patched method,
   every capture call, every hybrid branch, every `free_kv_cache`
   call. Gated behind `TURBOQUANT_TRACE=1` so it's free in normal use.
4. **Diff baselines, don't extrapolate.** When a change "fixes"
   something, the evidence must be a *before-vs-after diff* of the
   same captured trace under the same workload, both checked in or
   pasted in the PR. "Looks better" is not evidence.
5. **Bisect on real workloads.** When something regresses, bisect on
   commits using the actual quickstart workload that surfaced the
   bug, not on synthetic tests. Synthetic unit tests are necessary
   for not-regressing the kernels but they are demonstrably blind to
   the integration bugs we care about.
6. **Trust the runtime, not the docstring.** vLLM 0.17's docstrings,
   types, and module names lie about the runtime path under
   `compilation_config`. Treat live `inspect.getsource(...)`,
   `compilation_config.splitting_ops`, and stderr from the
   `EngineCore` subprocess as ground truth.
7. **Root-cause every silent failure.** If a hook fires zero times
   when it should fire, finish answering "why" before writing the
   fix. Multiple plausible causes must each be ruled in or out by a
   separate log, not by argument.
8. **One change per commit.** Each commit does exactly one thing
   whose effect is visible in a single trace diff. No bundling.

---

## 2. Decisions (defaults locked in)

These are the parameters Path B is executed against. Change them only
by editing this file in a separate commit, not silently in another PR.

| Decision | Value | Rationale |
|---|---|---|
| Verification model | `meta-llama/Llama-3.2-1B-Instruct` | Real public HF model, dense full-attention (no Mamba/MoE), fits a single 24GB GPU, 32k context, common reference point. |
| GPU CI runner | None (run-locally-before-PR) | Self-hosted GPU runners are operational overhead we're not ready for. Path B PRs must include a captured trace from a local CUDA run as evidence. |
| Correctness bar | Top-1 token agreement ≥ 95% on a 256-token greedy decode (`temperature=0`) from a fixed prompt, vs. the same model running baseline vLLM. | Identical text is too strict for a stochastic-noise quantizer. Perplexity bars are noisier than top-1 agreement on greedy decode and require LM-eval scaffolding we don't have. |
| Scope | **Conservative** — Sprints 0–4 only. | "TQ saves N% VRAM with ≤ 5% correctness loss and ≥ 0.8× baseline tok/s." No fused Triton kernel in scope; if memory savings are real, perf is a follow-up project. |
| Trace storage | `docs/traces/*.log` checked in. | Evidence-in-repo. Total budget 5 MB. If a trace exceeds that, summarize and link to a Gist. |

---

## 3. Sprint plan

### Sprint 0 — Instrumentation & reproduction harness (4–5 days)

**Goal.** End the sprint with a logging mode that, on a single
`quickstart.py` command, produces a per-step trace from which any
subsequent claim about "what fired" or "what didn't" can be settled by
`grep`. **No code fixes in Sprint 0.** Only logging, captured logs, and
a recipe.

**Tasks.**

- **S0.1** Add `TURBOQUANT_TRACE` env var. When set, enables
  `logger.debug` at every interception point in
  `turboquant/integration/vllm.py`:
  - `install_hooks` — one line per layer with `(layer_name,
    backend_kind, head_dim, num_kv_heads)`
  - `_make_patched_kv_update.patched` — one line per call with
    `(layer_idx, slot_mapping.shape[0], mode)`
  - `_make_patched_forward.patched` — one line per call with
    `(layer_idx, query.shape, attn_metadata.max_query_len, mode,
    branch_taken)`
  - `_capture_kv` — one line per call with `(layer_idx, num_tokens,
    via_forward)`
  - hybrid branch — one line per decision with `(flat.num_tokens,
    ring.size, took_compressed_path)`
  - `free_kv_cache` — one line per layer with `(layer_idx, freed_bytes)`
  Default off. Log format: `[TQ-TRACE] %(name)s pid=%(process)d
  layer=%d %(message)s`.

- **S0.2** Capture three reference traces under three configurations
  on the same workload (`MODE=hybrid LONG_PROMPT=1 BUFFER_SIZE=16
  MAX_TOKENS=64`). Check them into `docs/traces/`:
  - `s0_compiled.log` — default config
  - `s0_eager.log` — `enforce_eager=True`
  - `s0_eager_no_prefix.log` — `enforce_eager=True,
    enable_prefix_caching=False`

- **S0.3** Write `docs/integration-state.md` that *cites lines from
  those traces* as evidence for each finding. Replaces the inferred
  conclusions in the README's current ⚠️ notice with line-numbered
  references like `s0_compiled.log:142 — do_kv_cache_update never
  fires; s0_eager.log:89 — fires per-token`. This becomes the
  canonical statement of what's broken.

- **S0.4** e2e correctness probe: `tests/test_correctness_e2e.py`
  (CUDA-marked, skipped in default CI). Loads
  `Llama-3.2-1B-Instruct`, runs baseline vs. TQ generation on a fixed
  256-token greedy decode, compares token-id sequences position by
  position. Reports top-1 agreement %. Asserts `agreement >= 0.95`.
  Probe also dumps its own trace to `/tmp/tq_probe_*.log`.

**Acceptance criteria.**

- `docs/traces/s0_*.log` exist, are non-trivial (>50 lines each).
- `docs/integration-state.md` cites at least one specific line in
  each of the three traces for each of the README's three findings.
- `tests/test_correctness_e2e.py` exists, runs on a CUDA box, prints a
  number, currently fails (expected — that's the target Sprint 1
  shrinks).
- `pytest tests/` (CPU-only) still 94 passed.
- ruff check + format clean.

---

### Sprint 1 — Capture survives `torch.compile` (3–5 days)

**History.** This sprint was rescoped on 2026-04-29 after S1.1 / S1.2
recon (commits [`8338f18`](https://github.com/pitcany/vllm-turboquant/commit/8338f18)
and [`93304a0`](https://github.com/pitcany/vllm-turboquant/commit/93304a0))
showed the original framing of F1 was wrong. See
`docs/integration-state.md` § "S1.1 — `torch.library` override … is
structurally blocked", § "S1.2 — `register_forward_pre_hook` fires zero
times after compilation", and § "F1bis — F1 is overstated; default
compile + no-prefix-cache *does* capture the prefill". The reframed
findings drive everything below; do not read this section against the
pre-2026-04-29 version of `s0_compiled.log` alone.

**Entry condition.** `docs/integration-state.md` names a specific
line in `s0_compiled.log` (and now `s1_probe_pre_hook.log`) as the
immediate cause of capture failure. As of 2026-04-29 the cause is
`vllm/compilation/cuda_graph.py:208–323` — `CUDAGraphWrapper.__call__`
captures `runnable` once and replays via `cudagraph.replay()`, which
bypasses **all** Python-level hooks (custom op overrides, monkey-patches,
`register_forward_pre_hook`) inside the captured region. This is why both
1A and 1B fail by construction in FULL CUDAGraph mode.

**Tasks.**

- **S1.1** *Closed (failed).* Tried **1B (`torch.library` override of
  `vllm::unified_kv_cache_update`)**: blocked by torch's "kernel
  already registered" invariant on `(op, CUDA)`, no public deregister
  API. Even if it had registered, the override would be Python-level
  and would not survive FULL CUDAGraph replay. See
  `docs/integration-state.md` § "S1.1 …".

- **S1.2** *Closed (failed).* Tried **1A (forward pre-hooks)** via
  `attn_module.register_forward_pre_hook(...)` on the `Attention`
  `nn.Module`. Empirical result on default-compile + `prefix=OFF`
  workload: pre-hook fires **0 times** (`docs/traces/s1_probe_pre_hook.log`,
  `grep -c 'pre_hook fired'` → 0). Cause: `torch._dynamo.optimize`
  inlines `Attention.forward` past `nn.Module.__call__`'s hook
  iteration; hooks registered after the model has been compiled are
  never read. See `docs/integration-state.md` § "S1.2 …".

- **S1.3** *Active.* Land **1C (paged-cache reader)** as the only
  approach that survives FULL CUDAGraph by construction (it doesn't
  hook anything inside the captured region — it reads the paged
  `kv_cache` tensor *after* `model_runner.execute_model()` has
  returned, when Python is running again every step). Concretely:
  - install a per-`execute_model`-step Python callback (worker-side,
    via `collective_rpc` like the existing installer) that, for each
    layer in `model_runner._tq_layer_states`, reads the slice of the
    paged `kv_cache` tensor written this step (using `slot_mapping`
    from the just-completed `ForwardContext`) and feeds it into
    `state.engine.ingest_decode` / `ingest_prefill`;
  - retire the now-redundant per-token decode part of the existing
    `do_kv_cache_update` monkey-patch (keep the prefill path, since
    prefill runs PIECEWISE per `s0_compiled.log:40` and the patch fires
    cleanly there);
  - keep the decode-bypass detection: if the post-step delta is empty
    when it shouldn't be (e.g., decode token count not zero but no
    slots written), raise rather than silently no-op.

- **S1.4** Update `tests/test_vllm_smoke.py` with a CUDA-only test
  asserting captured token count == prefill+decode tokens for a
  fixed workload.

**Acceptance criteria.**

- A new `s1_compiled.log` (default compile, `prefix=OFF`,
  `MODE=hybrid LONG_PROMPT=1 BUFFER_SIZE=16 MAX_TOKENS=65`) shows
  **TQ store grows by ≥ 64 tokens during decode** — measured by the
  delta between `total_compressed_tokens + total_buffered_tokens` at
  end-of-prefill vs. end-of-run. Today's
  `docs/traces/s1_probe_pre_hook.log:132` measures the prefill-only
  baseline at 226 (210 + 16); the post-S1.3 trace must read ≥ 290
  (= 226 prefill + 64 decode-K/V writes). *Why `MAX_TOKENS=65` and not
  64:* vLLM v1 emits `N − 1` decode `execute_model` calls for
  `max_tokens=N` (the K/V for the final sampled token is never written,
  since no later decode uses it; verified empirically — see
  `docs/integration-state.md` § "S1.3 — 1C lands…"). To realise 64
  decode K/V writes the workload budget needs to be 65 generated tokens.
  *(Old criterion: "capture firing per-token under default vLLM config
  (no `enforce_eager`)" — retired because under FULL CUDAGraph no
  hook-based capture can fire per-token by construction; the post-step
  paged-cache reader doesn't show up as a per-token Python trace line
  either.)*
- A new TQ-vs-baseline correctness measurement on the same
  Qwen2.5-0.5B workload: top-1 token agreement ≥ 95% on the 65-token
  greedy decode output, evidenced by the same `s1_compiled.log`
  capturing both passes' tokenised output. *(Pre-S1.3 it's trivially 100% because the
  hybrid-compressed branch is never entered; post-S1.3 the branch can
  be entered, and we need to confirm it doesn't degrade output
  byte-for-byte before Sprint 3 takes over the > 95% bar on Llama.)*
- `tests/test_vllm_smoke.py` CUDA test passes (asserting captured
  token count ≈ prefill+decode tokens).
- Compile-mode tok/s on `Llama-3.2-1B-Instruct` at 4k context
  ≥ 0.7× raw baseline (measured, logged in PR). 1C reads from the
  already-existing paged `kv_cache` tensor — no extra kernel launches
  inside the CUDA graph — so the tok/s hit should be small.
- `docs/integration-state.md` updated with S1's before/after trace
  lines, and the existing F1 section flagged as superseded by F1bis
  rather than left contradicting the new evidence.

**Stop-loss for this sprint (refined).** If 1C also fails to grow the
TQ store during decode (e.g., the post-`execute_model` callback can't
get a usable `slot_mapping` for the last step, or the paged
`kv_cache[i]` already moved on by the time the callback runs), the
integration approach itself is wrong; escalate to §5 ("re-architect TQ
as a separate inference engine, or as a vLLM plugin via the upstream
plugin surface, not a monkey-patch").

---

### Sprint 2 — Prefix caching evasion (1 day)

**Entry condition.** Sprint 1 acceptance criteria met.

**Tasks.**

- **S2.1** In `enable_turboquant`, read
  `llm.llm_engine.vllm_config.cache_config.enable_prefix_caching`. If
  True and `mode != "off"`, raise `TurboQuantVLLMError` with the
  remediation: "Set `enable_prefix_caching=False` on `LLM(...)` until
  capture-on-cache-hit is implemented."
- **S2.2** File a follow-up issue / TODO in
  `docs/integration-state.md` for capture-on-cache-hit (option 2A).
  Out of scope for Path B Conservative.

**Acceptance criteria.**

- Workload that previously evaded capture due to prefix caching now
  errors loudly at `enable_turboquant` time.
- `tests/test_vllm_smoke.py` adds a unit test (no CUDA needed) that
  fakes a `cache_config.enable_prefix_caching=True` and asserts
  `TurboQuantVLLMError` is raised.

---

### Sprint 3 — Hybrid attention combines paged KV + TQ store (4–7 days)

**Entry condition.** Sprint 1 + 2 acceptance criteria met.

**Tasks.**

- **S3.1** Audit `score.compute_hybrid_attention` and
  `_make_patched_forward`'s hybrid branch. The current code reads
  only `state.store.get_flat_cache()` and `ring.peek()`, ignoring
  the paged `kv_cache` tensor. Document the gap with a trace line.
- **S3.2** Rewrite the hybrid path to consume **three KV segments**:
  - `kv_paged` — un-captured prompt tokens still in the paged cache
  - `kv_tq` — captured tokens in TQ's compressed store
  - `kv_ring` — recent exact tokens in the ring buffer
  Implement in pure PyTorch (option 3A). Online softmax over all
  three segments. ~80 LOC.
- **S3.3** e2e correctness probe (`test_correctness_e2e.py`) passes:
  top-1 agreement ≥ 95% on 256 decode tokens.
- **S3.4** Capture `s3_compiled.log` showing hybrid branch firing on
  decode tokens past the buffer window, attending across all three
  segments.

**Acceptance criteria.**

- `tests/test_correctness_e2e.py` passes (≥ 95% top-1 agreement on
  `Llama-3.2-1B-Instruct`).
- Hybrid mode no longer produces degenerate output (e.g.
  `"...\n\n.."`) on the long-prompt workload.
- `docs/integration-state.md` updated with the new trace evidence.

**Outcome (2026-04-30).** S3.1 + S3.2 + S3.3 landed as four commits on
`main` (`6dd98a0`, `0bf9510`, `de19e65`, `01d8189`) plus a follow-up
bit-budget commit. Acceptance criteria split:

- ✅ `s3_eager_no_prefix_qwen.log` shows the new
  `hybrid_segments num_paged=N num_tq=M num_ring=K` trace line firing
  per layer × per decode step with `num_paged=1` (was structurally 0).
- ✅ Hybrid no longer produces the degenerate `quant, quant, quant`
  tail from `s0_eager_no_prefix.log:4672`. Replaced by a *different*
  degenerate loop on Qwen-0.5B (`Use a 7B, 7B, 7B…`) and on Llama-1B
  (`I've got a lot of time to spend on this. …`).
- ✅ `docs/integration-state.md` updated with §S3.1, §S3.2, §S3.3
  sections including before/after diff tables.
- ❌ `tests/test_correctness_e2e.py` **does not pass the 95% bar.**
  Measured on `Llama-3.2-1B-Instruct` (256-token greedy decode,
  `LONG_PROMPT`, `buffer_size=16`, eager + no prefix-cache):

  | key_bits | value_bits | Top-1 agreement | First divergence |
  |---:|---:|---:|---:|
  | 3 | 2 (plan default) | **7.69%** | 1 |
  | 3 | 4 (§5 escalation) | 0.39% | 1 |
  | 4 | 4 (max budget) | 2.73% | 6 |

  All three runs confirm the combiner code is correct (208 / 4080 /
  4080 `hybrid_segments` lines = `decode_steps × 16 layers`,
  `num_paged=1 num_tq=211 num_ring=16` per call) and 6 unit tests
  (`tests/test_score.py`) prove the streaming-softmax math is
  identical to single-softmax-over-concat. **The bug is the bit
  budget at 1B scale on Llama-3.2, not the integration.** This is the
  §5 second-bullet stop-loss exactly as written; the §5 first-line
  exit ramp (`value_bits=4`) is exhausted.

**F3 closure status.** Open. Plan §4 row F3 still requires the 95%
agreement number on Llama; that number is not reachable with the
hybrid path's current accuracy. Sprint 4 takes the §5 third-bullet
pivot (capture-only + `free_kv_cache` for memory-only wins) — the
README's "Verified configuration" section will advertise
`mode="capture_only"`, with hybrid kept in the codebase as a
research-mode setting and a Sprint 5 / fused-Triton-kernel target.

---

### Sprint 4 — Real memory savings, the actual point (2–3 days) — N/A as of 2026-04-30

**Entry condition.** ~~Sprint 3 acceptance criteria met.~~ ~~Per Sprint
3's §5-second-bullet outcome, Sprint 4 enters with hybrid quality not
viable at 1B/3-bit/2-bit and pivots to the §5 third bullet (capture-only
+ `free_kv_cache` for memory-only wins).~~

**Sprint 4 is N/A as of 2026-04-30.** TurboQuant landed upstream in
[vLLM v0.20.0](https://github.com/vllm-project/vllm/releases/tag/v0.20.0)
as a v1 attention backend ([#38479](https://github.com/vllm-project/vllm/pull/38479),
[#40092](https://github.com/vllm-project/vllm/pull/40092)) — the §5
first-bullet "re-architect" path. Investing in this repo's
monkey-patch surface for the capture-only memory-only-wins story
would optimise a dead-end integration; the upstream backend is
strictly more capable (full hybrid path with FA3/FA4 prefill + 4
quality-tested presets, none of which use the 2-bit-value budget our
§5 finding showed is not viable at 1B scale). What actually landed
during the Sprint 4 attempt:

- ✅ **S4.1** ([commit `ec30102`](https://github.com/pitcany/vllm-turboquant/commit/ec30102))
  — `free_kv_cache` reworked: per-layer migration precondition,
  release-only-migrated-suffix semantics, post-free patched-forward
  guard against the 1-byte-sentinel-then-flash-attn-crash failure mode,
  5 CPU unit tests (`tests/test_free_kv_cache.py`). Honest improvement
  to a real bug; useful to anyone forking this repo even though the
  monkey-patch surface itself is now obsolete.
- ✅ **S4.2** ([commit `e28082c`](https://github.com/pitcany/vllm-turboquant/commit/e28082c))
  — `scripts/bench_path_b.py` written, lint-clean. Per-pass VRAM
  (allocator + nvidia-smi), decode tok/s, top-1 agreement, JSON output.
  Never run end-to-end — host CUDA driver state went bad mid-session
  and upstream landing made the run moot.
- ❌ **S4.3** — not started. README rewrite to "Verified configuration"
  was the next step; instead the README gained a SUPERSEDED notice
  pointing at upstream.

The original Sprint 4 task list is preserved verbatim below as a
research record. The S4.1 free_kv_cache fixes and S4.2 benchmark
script remain useful as a delta against any future fork that targets
a different vLLM version or a different bit budget.

**Tasks.**

- **S4.1** Rework `free_kv_cache` to release only the suffix that's
  been migrated to TQ's store, leaving any un-captured prefix
  in-place. Document the new semantics in the docstring.
- **S4.2** Real benchmark: a runnable script (replaces or
  supplements `benchmark.py`) that takes
  `--model HF_ID --context-len N --decode-tokens M` and reports:
  - VRAM used by KV cache (baseline vs. TQ-after-`free_kv_cache`)
  - Decode tok/s (baseline vs. TQ)
  - Top-1 agreement % (the correctness number)
- **S4.3** Update README:
  - Remove the ⚠️ "non-functional" notice.
  - Add a single "Verified configuration" table with the numbers
    measured on `Llama-3.2-1B-Instruct` from S4.2, with the exact
    command that reproduces them.
  - Demote the legacy benchmark tables to a "Historical claims —
    not yet reproduced on a public model" appendix.

**Acceptance criteria.**

- One reproducible benchmark command, documented in README, runs on
  `Llama-3.2-1B-Instruct` and produces a table the user can verify.
- VRAM saved per token > 0.
- Top-1 agreement ≥ 95%.
- Decode tok/s ≥ 0.8× baseline.

---

### Sprint 5 (out of scope for Conservative) — Fused Triton hybrid kernel — N/A as of 2026-04-30

Tracked as a separate project. ~~Will land as `docs/plan-path-b-perf.md`
when Sprint 4 is done.~~

**Sprint 5 is N/A as of 2026-04-30.** The fused-kernel target Sprint 5
contemplated has been delivered upstream as **FA3/FA4 prefill + Triton
decode kernels** in vLLM 0.20.0
([#38479](https://github.com/vllm-project/vllm/pull/38479),
[#40092](https://github.com/vllm-project/vllm/pull/40092)). Industrial-
grade kernels with CUDAGraph capture, stream overlap, in-kernel FP8
casts, and SM-aware Hopper / Ampere paths — substantially beyond what
"Path B Perf" was scoped to deliver.

---

## 4. Definitions of done — for each ⚠️ finding

The current README ⚠️ notice asserts three issues. Path B is **done**
when each is replaced by a positive statement of the form *"as of
commit SHA, this works under config X, evidenced by trace
docs/traces/Y.log:Z."*

| Finding | Definition of "done" | Status (2026-04-30) |
|---|---|---|
| **F1: torch.compile bypasses our hooks.** | `s1_compiled.log` shows the patched function firing per-token under `compilation_config` defaults; `tests/test_vllm_smoke.py::test_capture_under_compile` passes on CUDA. | ✅ **Closed** by `304ba1f` (S1.3 / 1C — post-execute paged-cache reader). Evidence: `docs/traces/s1_compiled.log` (`paged_read num_tokens=1` × 1512). |
| **F2: Prefix caching evades capture.** | Conservative scope: `enable_turboquant` raises a typed error if prefix caching is on, and `tests/test_vllm_smoke.py::test_rejects_prefix_caching` asserts that. The "really fix it" version (capture-on-cache-hit) is a separate ticket. | ✅ **Closed** by `4c902f1` (S2.1). Capture-on-cache-hit follow-up filed at `docs/integration-state.md` § "F2 closure path (b) — out of scope" (S2.2 / `2a4bbea`). |
| **F3: Hybrid mode ignores the paged KV cache.** | `tests/test_correctness_e2e.py` passes (≥ 95% top-1 agreement on `Llama-3.2-1B-Instruct`); `s3_compiled.log` shows the hybrid branch attending across all three segments. | 🛑 **Closed by upstream supersession.** Structural half closed by `0bf9510` (S3.2): combiner now folds `kv_paged ∪ kv_tq ∪ kv_ring` via online softmax; degenerate `quant, quant, quant` tail gone; `hybrid_segments num_paged=1 num_tq=… num_ring=…` trace fires per layer × per step. Empirical half (≥ 95% agreement on Llama-1B): **fails at every bit budget tried** in this repo (3/2 → 7.69%, 3/4 → 0.39%, 4/4 → 2.73%; see `docs/integration-state.md` § "S3.3 follow-up"). The upstream port in vLLM 0.20.0 ([#38479](https://github.com/vllm-project/vllm/pull/38479)) closes F3 in production via WHT rotation + norm correction + boundary-layer protection + dropping QJL — design choices this repo's hybrid path doesn't have. F3 is *closed* in the sense that anyone wanting hybrid TQ now uses upstream's `--kv-cache-dtype turboquant_*`. This repo's hybrid path remains a research artefact. |

When all three rows were ✅ Closed (or, per the 2026-04-30 archive
event, when the project's ⚠️-notice issues are resolved by upstream
supersession), the README's ⚠️ notice was supposed to be rewritten
into a "Verified configuration" section per Sprint 4. The actual
disposition is the SUPERSEDED notice at the top of the README — see
`README.md` for the migration path to vLLM 0.20.0.

---

## 5. Stop-loss criteria

When do we abandon Path B and pivot? Concrete bars:

- **End of Sprint 1**, if neither 1A, 1B, nor 1C yields working
  capture under default `compilation_config`. The integration approach
  itself may be wrong; TQ should be re-architected as a separate
  inference engine (or as a vLLM plugin via the upstream plugin
  surface, not a monkey-patch). *(2026-04-30: this is exactly the path
  vLLM 0.20.0's TurboQuant attention backend
  [#38479](https://github.com/vllm-project/vllm/pull/38479) takes.
  Vindicates the bullet as a bullet, even though we didn't have to
  invoke it for Sprint 1 itself.)*
- **End of Sprint 3**, if hybrid 3A produces > 30% top-1 disagreement
  even with full capture and prefix caching off. The underlying TQ
  approach (3-bit key + 2-bit value at 1B scale) may not be
  quality-viable at this size; escalate to `value_bits=4` or skip
  hybrid entirely (capture-only + free_kv_cache for memory-only wins).
  *(2026-04-30: engaged. Three bit budgets tested at 1B (3/2, 3/4,
  4/4); none cleared. Upstream's vLLM 0.20.0 port doesn't ship a
  2-bit-value preset either; minimum is `turboquant_t3nc` at 3/3 +
  norm correction.)*
- **Anytime**, if a sprint's "no code fixes in this sprint" rule is
  violated (i.e., we fix something speculatively, and a subsequent
  bug shows the fix was wrong). Stop, write the missed log, restart.
- **Upstream supersession** *(added 2026-04-30 in archive event)*: if
  the feature this plan is contracting against ships upstream during
  the project's lifetime, archive Path B and migrate to upstream.
  Triggered by vLLM 0.20.0 TurboQuant attention backend; see
  `README.md` SUPERSEDED notice.

---

## 6. Out of scope (for now)

- Async engine support. Path B targets the synchronous `vllm.LLM`
  only.
- TP > 1. Path B targets single-GPU. The integration was previously
  exercised on 4× and 8× GPU; TP correctness is its own project.
- Async / streamed generation.
- vLLM 0.18+. Path B targets 0.17.x explicitly until 0.17 works.

---

## 7. Document maintenance

This file is the contract. It is updated by:

- Editing decisions in §2 (record reason in commit message).
- Updating sprint acceptance criteria when a sprint completes
  (record evidence link).
- Adding a row to §4 when a finding is closed.

It is **not** updated by:

- Hand-waving in chat.
- Inferred conclusions without a captured trace.

---

*Last updated: 2026-04-30 (archive event — TurboQuant landed upstream
in vLLM 0.20.0 as a v1 attention backend. Path B archived. Sprint 4
S4.1 + S4.2 still landed before archive; S4.3 cancelled; Sprint 5
also N/A because upstream shipped FA3/FA4 prefill kernels which were
exactly Sprint 5's target. §4 F3 row updated to "closed by upstream
supersession". §5 gains a fourth "upstream supersession" bullet. Tag
v0.2-final marks the archive state.

2026-04-30 (Sprint 3 complete; §5 second-bullet stop-loss engaged
after Llama-3.2-1B agreement at 7.69% / 0.39% / 2.73% across three
bit budgets. F3 row in §4 marked half-closed: structural combiner
shipped, empirical 95% bar not reachable on Llama-1B. Sprint 4 entry
condition softened to "post-S3.3 outcome" — Sprint 4 takes the §5
third-bullet pivot and advertises `mode="capture_only"` in the
README's Verified Configuration section instead of `mode="hybrid"`.

2026-04-29 (initial draft + same-day Sprint 1 rescope after S1.1 /
S1.2 recon: 1B blocked structurally, 1A blocked empirically, S1.3 /
1C now active; F1 reframed as F1bis in `docs/integration-state.md`.
Same-day amendment: Sprint 1 canonical workload changed from
`MAX_TOKENS=64` to `MAX_TOKENS=65` so the ≥ 290 captured-tokens bar
is reachable — vLLM v1 emits N − 1 decode K/V writes for
`max_tokens=N`.)*
