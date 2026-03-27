# TurboQuant Research Handoff (RTX 5090 / TitPod)

## 1) Scope and current status

This handoff captures:
- how to access the remote RTX 5090 box,
- where code/data/scripts live (local + remote),
- what has been implemented and validated,
- exact run commands + known working configs,
- known blockers and next research steps.

Current state:
- **TurboQuant modular serving stack is implemented and working** (capture/store/score/integration adapter).
- **No-alloc + shared-KV path for long-context experiments is implemented**.
- **30k context A/B telemetry collected** (baseline vs TurboQuant, separately).
- **200k context with TurboQuant has completed successfully in prior runs** (single-shot completion done), but full dual-case telemetry at 200k is unstable due long-run process kills / baseline stall.

---

## 2) Access to device

## SSH details

*Redacted for public repo. Configure your own remote GPU host.*

```bash
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=12 user@your-gpu-host
```

---

## 3) Directory map

## Local

- Workspace root: `.`
- Main repo used in this work: `./turboquant_pkg`

## Remote

- Work root: `/5090-qwen3.5-27b`
- Venv: `/5090-qwen3.5-27b/.venv`
- Models: `/5090-qwen3.5-27b/models`
- Scripts: `/5090-qwen3.5-27b/scripts`
- TurboQuant repo mirror: `/5090-qwen3.5-27b/turboquant`
- Runtime package path: `/5090-qwen3.5-27b/turboquant/turboquant`
- Logs: `/5090-qwen3.5-27b/logs`

---

## 4) Environment and versions (remote)

- Python: `3.12.3`
- Torch: `2.10.0+cu128`
- vLLM: `0.18.0`
- transformers: `5.3.0.dev0`

GPU:
- RTX 5090 32GB (single GPU used in all runs shown here).

---

## 5) Models on remote

- `/5090-qwen3.5-27b/models/QuantTrio-Qwen3.5-27B-AWQ` (~21 GB) ← primary model used
- `/5090-qwen3.5-27b/models/Qwen-Qwen3.5-27B-FP8` (~29 GB) ← FP8 baseline experiments only

---

## 6) Code architecture implemented

Core modular files (in `turboquant/`):

- `capture.py`
  - `RingBuffer` + `KVCaptureEngine`
  - bulk prefill capture + decode buffering
- `store.py`
  - `CompressedKVStore`, chunked quantized storage, lazy flattening
- `score.py`
  - `compute_hybrid_attention()` for compressed-history + exact-recent merge
  - memory-safe GQA path (removed large head repeat blowups)
- `integration/vllm.py`
  - clean adapter (`off | capture_only | hybrid | full_tq`)
  - no-alloc support path added
- `vllm_attn_backend.py`
  - compatibility shim + no-alloc hook enablement + KV sharing patching

Important 200k-related implementation details:
- `enable_no_alloc(...)` used before `LLM(...)`.
- Hook install occurs during executor KV spec phase.
- Flash-attn layers can share KV target layer to reduce paged KV pressure.
- Added layout patch to avoid missing-layer KeyError in hybrid attn+mamba layout update.
- Added no-alloc prefill attention path in modular integration.

---

## 7) Remote scripts already present

Key scripts in `/5090-qwen3.5-27b/scripts`:
- `final_test3.py` (baseline + TQ + coherence + 24k needle + KV free)
- `long_ctx_runner.py`, `long_ctx_test.py` (long context harness)
- `full_bench.py`, `profile_overhead.py`, `verify_runner.py`, etc.

---

## 8) Validation completed

## Test suites

Local and remote both passed:
- `test_modular.py` → `19/19 pass`
- `test_turboquant.py` → pass

Remote `final_test3.py` passed historically with:
- coherence suite: pass
- 24K needle retrieval: pass
- KV freed: ~2.9 GB

---

## 9) Measured results (latest reliable set)

## A) 30k context (same prompt hash both runs)

Prompt:
- `max_model_len=32768`
- prompt tokens: `32720`
- prompt hash: `6f5f3d4d1971c67a2a0ffff7023626a282cdd5aea300c80b0748f1367f5bd510`

### Baseline 30k (separate TTFT and full runs)

- Init: `40.160s`
- TTFT: `18.138s`
- Prefill throughput est: `1803.92 tok/s`
- Full run (24 tok): `18.992s`
- End-to-end output rate: `1.264 tok/s`
- VRAM used:
  - after init: `27833 MB`
  - after TTFT/full: `~28393 MB`
- Power/temp (after full): `313.26W`, `55C`
- KV cache reserved (unique): `2.724 GB`
- Activation est (full pass): `644.61 MB`

### TurboQuant 30k (no-alloc stack)

- Init: `43.749s`
- TTFT: `17.162s`
- Prefill throughput est: `1906.52 tok/s`
- Full run (24 tok): `18.415s`
- End-to-end output rate: `1.303 tok/s`
- VRAM used:
  - after init: `27823 MB`
  - after TTFT: `28411 MB`
  - after full: `28419 MB`
- Power/temp (after full): `210.64W`, `53C`
- KV cache reserved (unique): `2.724 GB`
- Shared KV layers: `15`
- TQ stats in runtime: `num_layers=16`, mode `hybrid`
- Activation est (full pass): `599.20 MB`

### 30k delta (TQ vs baseline)

- TTFT improved by ~`0.98s`
- Prefill throughput ~`+5.7%`
- Full 24-token latency ~`0.58s` faster
- Full-pass activation estimate ~`45 MB` lower

## B) 200k status

Reliable observed facts:
- TurboQuant run completed at ~200k prompt:
  - prompt tokens: `199952`
  - output tokens: `24`
  - elapsed: `58.34s`
  - gpu mem near end: `~31.9 GB`
  - tq hooked layers: `16`
- Baseline 200k repeatedly stalls/times out in this setup.
- Full dual-case 200k telemetry scripts were repeatedly interrupted by remote process kills (`exit 137`) or cancellation side effects.

---

## 10) Known blockers / failure modes

1. **Interrupted runs leave GPU processes alive**
   - cancellation can leave a Python process attached to GPU using ~28 GB VRAM.
2. **Baseline 200k not practical in current config**
   - repeated timeout/stall behavior.
3. **Host/process kill (`exit 137`) during long scripted telemetry**
   - long single-process “collect everything” jobs can be killed before final JSON write.
4. **FP8 KV + TurboQuant stacking**
   - still incompatible in earlier testing path (FlashInfer metadata mismatch).

---

## 11) Operational playbook (researcher)

## A) Always clean GPU before runs

```bash
python3 - <<'PY'
import os, subprocess
try:
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True)
    for x in out.splitlines():
        x=x.strip()
        if x:
            try: os.kill(int(x),9)
            except: pass
except: pass
print(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'], text=True))
PY
```

## B) Use remote venv explicitly

```bash
/5090-qwen3.5-27b/.venv/bin/python ...
```

## C) Split experiments by phase

Do **separate invocations** for:
- init telemetry,
- TTFT/prefill telemetry,
- full generation telemetry,
instead of one giant all-in-one script.

## D) Suggested stable env flags

```bash
CUDA_VISIBLE_DEVICES=0
VLLM_ENABLE_V1_MULTIPROCESSING=0
TOKENIZERS_PARALLELISM=false
```

## E) TurboQuant enablement

Before `LLM(...)`:

```python
from turboquant.vllm_attn_backend import enable_no_alloc
enable_no_alloc(key_bits=3, value_bits=2, buffer_size=128, initial_layers_count=4)
```

---

## 12) Where outputs/logs should be written

Recommended:
- `/5090-qwen3.5-27b/logs/telemetry_30k_baseline.json`
- `/5090-qwen3.5-27b/logs/telemetry_30k_tq.json`
- `/5090-qwen3.5-27b/logs/telemetry_200k_tq.json`

Use one file per phase/case to avoid losing everything on kill.

---

## 13) What has been accomplished (research summary)

- Modular TurboQuant runtime refactor delivered (capture/store/score/integration).
- Compatibility shim maintained for older backend entrypoint.
- Runtime bug fixes done for real inference paths (shape handling, no-alloc plumbing, KV sharing wiring).
- Real remote validations completed on RTX 5090:
  - test suites pass,
  - coherence and 24k retrieval pass in established validation scripts,
  - ~2.9 GB KV freeing confirmed in previous runs,
  - 30k A/B telemetry shows small but consistent TQ advantage.
- 200k with TQ has been demonstrated as runnable; robust full baseline-vs-TQ 200k telemetry remains a follow-up task due baseline instability + long-run process kills.

---

## 14) Recommended next research steps

1. Build a **phase-isolated telemetry harness** (each phase separate process).
2. Add auto-timeout and partial JSON checkpointing after each phase.
3. Benchmark 30k/50k/80k/120k for baseline+TQ with identical prompt hashes.
4. For 200k:
   - prioritize TQ-only telemetry first,
   - then baseline with reduced scope and strict timeout budget.
5. Investigate FP8+TQ compatibility path only after stable 200k telemetry pipeline.
