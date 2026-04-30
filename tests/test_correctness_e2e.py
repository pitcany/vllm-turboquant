"""End-to-end correctness probe (CUDA-marked, skipped in default CI).

Loads ``meta-llama/Llama-3.2-1B-Instruct``, runs a fixed greedy decode
twice in the same vLLM engine — once baseline, once with TurboQuant
``mode="hybrid"`` hooks installed — and asserts top-1 token agreement
≥ 0.95.

This is the probe described in
[`docs/plan-path-b.md`](../docs/plan-path-b.md) §3 / S0.4. It is
**expected to fail today** — that's the point. The threshold (0.95) is
the target Path B Sprint 3 must shrink to. The test exists today as a
single number to drive integration work toward, and as a regression
guard once Sprint 3 lands. See ``docs/integration-state.md`` row F3 for
why it currently fails (hybrid attention ignores the paged ``kv_cache``
tensor).

Skipped in default CI because (a) it needs a CUDA GPU and (b) it
downloads a gated model. The ``cuda`` marker keeps it out of
``pytest -m unit`` runs (the CI default per
``.github/workflows/ci.yml``).

Invocation::

    pytest tests/test_correctness_e2e.py -m cuda -s

Environment overrides::

    TQ_E2E_MODEL          override the verification model (default
                          meta-llama/Llama-3.2-1B-Instruct, per
                          plan-path-b.md §2)
    TQ_E2E_MAX_TOKENS     decode horizon (default 256)
    TQ_E2E_THRESHOLD      agreement bar (default 0.95)
    HF_TOKEN              required for gated Llama download

The probe writes its own captured stderr (which contains the
``[TQ-TRACE]`` records emitted by ``TURBOQUANT_TRACE=1``) plus a header
of summary metrics to ``/tmp/tq_probe_<pid>.log`` for
post-hoc inspection.
"""

from __future__ import annotations

import os

# Pytest's parent process preloads torch (via conftest.py) which will init
# CUDA in the parent before vLLM forks workers. vLLM's default
# VLLM_WORKER_MULTIPROC_METHOD=fork then crashes with "Cannot re-initialize
# CUDA in forked subprocess". Force spawn so the worker gets a clean
# Python interpreter. Must be set before vllm is imported.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

# Skip the whole module if vLLM isn't installed — the unit suite must
# remain importable without optional deps. importorskip raises Skipped at
# collection time, which is what we want.
pytest.importorskip("vllm")

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

MODEL = os.environ.get("TQ_E2E_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
MAX_TOKENS = int(os.environ.get("TQ_E2E_MAX_TOKENS", "256"))
AGREEMENT_THRESHOLD = float(os.environ.get("TQ_E2E_THRESHOLD", "0.95"))
PROMPT = (
    "You are a careful technical writer. Explain in concrete steps how "
    "key-value cache compression in transformer inference trades off "
    "memory, throughput, and quality. Use precise numbers where possible. "
    "Do not use bullet lists. Begin with a single-sentence thesis."
)


def _greedy_token_ids(llm, prompt: str) -> list[int]:
    """Run a single greedy generation and return the output token ids."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    out = llm.generate([prompt], sp)
    return list(out[0].outputs[0].token_ids)


def test_top1_agreement_baseline_vs_tq_hybrid(capfd) -> None:
    # Side-effect import: sets VLLM_ALLOW_INSECURE_SERIALIZATION=1 if unset
    # so the worker subprocess (when one is spawned) accepts our installer
    # closure. Must happen BEFORE vllm.LLM(...).
    import turboquant.vllm  # noqa: F401

    # Turn on the per-interception trace so the captured stderr contains
    # grep-able evidence of which hooks fired in the TQ pass.
    os.environ.setdefault("TURBOQUANT_TRACE", "1")

    from vllm import LLM

    from turboquant.vllm import enable_turboquant

    try:
        llm = LLM(
            model=MODEL,
            tensor_parallel_size=1,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            max_num_seqs=1,
            enforce_eager=True,
            enable_prefix_caching=False,
        )
    except Exception as exc:  # noqa: BLE001
        # Most common cause is the gated Llama download — skip rather
        # than fail so the suite stays green on machines that can't
        # access the model.
        pytest.skip(f"could not construct LLM({MODEL!r}): {exc}")

    baseline_tokens = _greedy_token_ids(llm, PROMPT)

    enable_turboquant(
        llm,
        key_bits=3,
        value_bits=2,
        buffer_size=128,
        mode="hybrid",
    )
    tq_tokens = _greedy_token_ids(llm, PROMPT)

    n = min(len(baseline_tokens), len(tq_tokens))
    assert n > 0, f"empty token sequences: baseline={baseline_tokens!r} tq={tq_tokens!r}"
    matches = sum(1 for a, b in zip(baseline_tokens[:n], tq_tokens[:n]) if a == b)
    agreement = matches / n

    # Persist a probe artefact for after-the-fact grep. capfd has been
    # capturing fd-level stderr since this test started, so the captured
    # buffer contains both the parent's logger output *and* the worker
    # subprocess's stderr (which vLLM tees to the parent fd).
    probe_path = Path(f"/tmp/tq_probe_{os.getpid()}.log")
    captured = capfd.readouterr()
    header = (
        "# tq probe artefact (S0.4 of docs/plan-path-b.md)\n"
        f"# model={MODEL}\n"
        f"# max_tokens={MAX_TOKENS}\n"
        f"# threshold={AGREEMENT_THRESHOLD}\n"
        f"# baseline_len={len(baseline_tokens)}\n"
        f"# tq_len={len(tq_tokens)}\n"
        f"# compared={n}\n"
        f"# top1_agreement={agreement:.4f}\n"
        f"# baseline_tokens_head={baseline_tokens[:32]}\n"
        f"# tq_tokens_head={tq_tokens[:32]}\n"
        "\n=== captured stderr ===\n"
    )
    probe_path.write_text(header + captured.err + "\n=== captured stdout ===\n" + captured.out)

    # Re-emit a single summary line on stderr for human readers under -s.
    print(
        f"[tq-e2e] model={MODEL} compared={n} top1_agreement={agreement:.4f} "
        f"threshold={AGREEMENT_THRESHOLD} probe={probe_path}",
        file=sys.stderr,
    )

    assert agreement >= AGREEMENT_THRESHOLD, (
        f"Top-1 agreement {agreement:.4f} < threshold {AGREEMENT_THRESHOLD} "
        f"on {MODEL} ({n} tokens compared); see {probe_path} for trace."
    )
