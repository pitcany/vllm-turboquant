#!/usr/bin/env python3
"""Shared helpers for the TurboQuant telemetry harness."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONTEXTS = [30000, 50000, 80000, 120000, 200000]
DEFAULT_CASES = ["baseline", "tq"]
DEFAULT_PHASES = ["init", "ttft", "full", "quality"]
DEFAULT_QUALITY_PROMPT = (
    "Answer precisely with one numbered line per item: "
    "1) Capital of France? "
    "2) 17*23? "
    "3) Chemical formula for water? "
    "4) Author of Romeo and Juliet? "
    "5) What does KV cache store in transformer inference?"
)

_PROMPT_PARAGRAPHS = [
    "TurboQuant reduces KV memory by compressing historical keys and values while preserving exact recent tokens for decode stability.",
    "The research operator records prompt hashes, phase runtimes, GPU memory, temperature, and power so regressions can be audited later.",
    "Every benchmark phase is isolated because long chained jobs hide the difference between initialization cost, prefill cost, and decode cost.",
    "The remote 5090 lane is intentionally single-GPU and uses bounded output tokens to keep comparisons about context handling rather than long generations.",
    "Baseline and TurboQuant runs must reuse the exact same prompt text, tokenizer path, and output token budget or the telemetry is not comparable.",
    "Exit code 137 is treated as a first-class artifact because host kills are part of the observed system behavior, not an external footnote.",
    "A stable research lane writes prompt metadata, stdout tails, stderr tails, and partial manifests so interrupted runs still leave evidence.",
    "No-alloc mode shifts pressure away from paged KV allocation and toward the compressed TurboQuant store, so hook installation state matters.",
    "The campaign stops pretending that symmetric 200k telemetry is mandatory if baseline repeatedly stalls while TurboQuant remains healthy.",
    "A useful long-context report includes both successes and explicit failures, because missing artifacts destroy trust faster than bad numbers.",
]


@dataclass
class PromptBundle:
    text: str
    prompt_hash: str
    prompt_tokens: int
    token_ids: list[int]
    seed: int
    context_len: int


def ensure_repo_import_path() -> Path:
    """Make the TurboQuant repo importable from local or mirrored script layouts."""
    candidates = [
        REPO_ROOT,
        SCRIPT_DIR.parent / "turboquant",
    ]
    for candidate in candidates:
        if (candidate / "setup.py").exists() and (candidate / "turboquant").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    raise RuntimeError(
        f"Unable to locate TurboQuant repo root from {SCRIPT_DIR}. "
        "Expected setup.py + turboquant package in a parent or sibling checkout."
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_csv_ints(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return list(default)
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_strings(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def tail_text(text: str | None, limit: int = 4000) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def sha256_jsonable(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def command_available(name: str) -> bool:
    return shutil.which(name) is not None


def build_python_launcher() -> list[str]:
    if command_available("uv"):
        return ["uv", "run", "python"]
    return [sys.executable]


def _query_nvidia_csv(query: str) -> list[dict[str, str]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    columns = [col.strip() for col in query.split(",")]
    rows: list[dict[str, str]] = []
    for line in proc.stdout.strip().splitlines():
        values = next(csv.reader([line]))
        rows.append({columns[idx]: values[idx].strip() for idx in range(min(len(columns), len(values)))})
    return rows


def probe_gpu_metrics() -> dict[str, Any]:
    rows = _query_nvidia_csv(
        "name,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu"
    )
    if not rows:
        return {
            "gpu_name": None,
            "memory_used_mb": None,
            "memory_free_mb": None,
            "utilization_gpu_pct": None,
            "power_w": None,
            "temp_c": None,
            "gpus": [],
        }

    def _num(row: dict[str, str], key: str) -> float | int | None:
        raw = row.get(key)
        if raw in (None, "", "[Not Supported]"):
            return None
        if "." in raw:
            return float(raw)
        return int(raw)

    parsed = [
        {
            "gpu_name": row.get("name"),
            "memory_used_mb": _num(row, "memory.used"),
            "memory_free_mb": _num(row, "memory.free"),
            "utilization_gpu_pct": _num(row, "utilization.gpu"),
            "power_w": _num(row, "power.draw"),
            "temp_c": _num(row, "temperature.gpu"),
        }
        for row in rows
    ]
    primary = dict(parsed[0])
    primary["gpus"] = parsed
    return primary


def scrub_gpu_processes() -> dict[str, Any]:
    killed: list[int] = []
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        for raw in proc.stdout.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            pid = int(raw)
            try:
                os.kill(pid, 9)
                killed.append(pid)
            except ProcessLookupError:
                continue
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return {
        "killed_pids": killed,
        "post_scrub_gpu": probe_gpu_metrics(),
    }


def _build_prompt_units(seed: int) -> list[str]:
    rng = random.Random(seed)
    paragraphs = list(_PROMPT_PARAGRAPHS)
    rng.shuffle(paragraphs)
    units = [
        "System note: this prompt is synthetic long-context research traffic for deterministic telemetry collection.",
        "Operator requirements: preserve exact wording, record prompt hash, and keep outputs short.",
    ]
    for idx in range(4096):
        paragraph = paragraphs[idx % len(paragraphs)]
        units.append(
            f"Section {idx:04d}: {paragraph} "
            f"Evidence marker {rng.randint(1000, 9999)}. "
            f"Token budget focus: context stability, KV behavior, and staged retries."
        )
    return units


def build_prompt_bundle_from_tokenizer(tokenizer: Any, context_len: int, seed: int) -> PromptBundle:
    units = _build_prompt_units(seed)
    token_ids: list[int] = []
    for unit in units:
        rendered = unit + "\n"
        token_ids.extend(tokenizer.encode(rendered, add_special_tokens=False))
        if len(token_ids) >= max(context_len + 512, context_len):
            break

    if not token_ids:
        raise RuntimeError("Prompt builder produced no token ids.")

    token_ids = token_ids[:context_len]
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    actual_ids = tokenizer.encode(text, add_special_tokens=False)
    while len(actual_ids) > context_len and len(actual_ids) > 1:
        token_ids = token_ids[: context_len - (len(actual_ids) - context_len)]
        text = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        actual_ids = tokenizer.encode(text, add_special_tokens=False)

    prompt_hash = sha256_jsonable(actual_ids)
    return PromptBundle(
        text=text,
        prompt_hash=prompt_hash,
        prompt_tokens=len(actual_ids),
        token_ids=actual_ids,
        seed=seed,
        context_len=context_len,
    )


def generate_prompt_artifacts(
    model_path: str,
    context_len: int,
    seed: int,
    prompt_dir: Path,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    ensure_dir(prompt_dir)
    prompt_path = prompt_dir / "prompt.txt"
    meta_path = prompt_dir / "prompt_meta.json"
    if not force and prompt_path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    if dry_run:
        prompt_text = (
            f"DRY RUN prompt for context {context_len} seed {seed}. "
            "This stub bypasses tokenizer/model requirements and only exists for contract testing."
        )
        prompt_tokens = min(context_len, max(8, len(prompt_text.split())))
        prompt_hash = sha256_jsonable([context_len, seed, prompt_tokens, prompt_text])
        payload = {
            "context_len": context_len,
            "prompt_seed": seed,
            "prompt_hash": prompt_hash,
            "prompt_tokens": prompt_tokens,
            "model_path": model_path,
            "tokenizer_path": None,
            "generated_at": utc_now_iso(),
        }
        atomic_write_text(prompt_path, prompt_text)
        atomic_write_json(meta_path, payload)
        return payload

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    bundle = build_prompt_bundle_from_tokenizer(tokenizer, context_len, seed)
    payload = {
        "context_len": context_len,
        "prompt_seed": seed,
        "prompt_hash": bundle.prompt_hash,
        "prompt_tokens": bundle.prompt_tokens,
        "model_path": model_path,
        "tokenizer_path": model_path,
        "generated_at": utc_now_iso(),
    }
    atomic_write_text(prompt_path, bundle.text)
    atomic_write_json(meta_path, payload)
    return payload


def load_prompt_artifacts(prompt_dir: Path) -> tuple[str, dict[str, Any]]:
    prompt_path = prompt_dir / "prompt.txt"
    meta_path = prompt_dir / "prompt_meta.json"
    if not prompt_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing prompt artifacts in {prompt_dir}")
    return (
        prompt_path.read_text(encoding="utf-8"),
        json.loads(meta_path.read_text(encoding="utf-8")),
    )


def extract_executor(llm: Any) -> Any:
    engine = llm.llm_engine
    core = getattr(engine, "engine_core", engine)
    inner = getattr(core, "engine_core", core)
    return getattr(inner, "model_executor", None)


def inspect_runtime_stats(llm: Any) -> dict[str, Any]:
    executor = extract_executor(llm)
    if executor is None:
        return {
            "tq_hooked_layers": None,
            "shared_kv_layers": None,
            "kv_reserved_gb": None,
            "tq_total_memory_bytes": None,
            "tq_total_compressed_tokens": None,
            "tq_mode": None,
        }

    def _probe(worker):
        model_runner = worker.model_runner
        static_ctx = model_runner.compilation_config.static_forward_context
        tq_states = (
            getattr(model_runner, "_tq_layer_states", None)
            or getattr(model_runner, "_tq_states", None)
            or {}
        )
        shared_layers = 0
        for attn_module in static_ctx.values():
            if getattr(attn_module, "kv_sharing_target_layer_name", None):
                shared_layers += 1

        seen: dict[int, int] = {}

        def _capture_tensor_bytes(value):
            if hasattr(value, "data_ptr") and hasattr(value, "nelement") and hasattr(value, "element_size"):
                ptr = value.data_ptr()
                if ptr:
                    seen[ptr] = value.nelement() * value.element_size()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    _capture_tensor_bytes(item)

        for entry in getattr(model_runner, "kv_caches", []):
            _capture_tensor_bytes(entry)

        tq_total_memory_bytes = None
        tq_total_compressed_tokens = None
        tq_mode = None
        try:
            from turboquant.integration.vllm import get_stats as _get_stats

            tq_stats = _get_stats(model_runner)
            tq_total_memory_bytes = tq_stats.get("total_memory_bytes")
            tq_total_compressed_tokens = tq_stats.get("total_compressed_tokens")
            tq_mode = tq_stats.get("mode")
        except Exception:
            pass

        return {
            "tq_hooked_layers": len(tq_states),
            "shared_kv_layers": shared_layers,
            "kv_reserved_gb": round(sum(seen.values()) / 1e9, 6),
            "tq_total_memory_bytes": tq_total_memory_bytes,
            "tq_total_compressed_tokens": tq_total_compressed_tokens,
            "tq_mode": tq_mode,
        }

    results = executor.collective_rpc(_probe)
    return results[0] if results else {}


def maybe_free_tq_kv(llm: Any) -> dict[str, Any]:
    executor = extract_executor(llm)
    if executor is None:
        return {"freed_kv_bytes": None}

    def _free(worker):
        from turboquant.vllm_attn_backend import free_kv_cache

        return free_kv_cache(worker.model_runner)

    try:
        freed = executor.collective_rpc(_free)
    except Exception:
        return {"freed_kv_bytes": None}
    return {"freed_kv_bytes": int(freed[0]) if freed else None}


def phase_timeout_for(case: str, phase: str, context_len: int, strict_timeouts: bool = False) -> int:
    base = {
        "init": 420,
        "ttft": 900,
        "prefill_only": 900,
        "full": 1200,
        "decode_only": 1200,
        "quality": 600,
    }[phase]
    multiplier = max(1.0, context_len / 30000.0)
    timeout = int(base * multiplier)
    if case == "baseline" and context_len >= 200000:
        timeout = 1800 if strict_timeouts else 2400
    elif case == "tq" and context_len >= 200000:
        timeout = 2400 if strict_timeouts else 3000
    elif strict_timeouts:
        timeout = int(timeout * 0.85)
    return max(timeout, base)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def should_skip_phase(output_path: Path, skip_existing: bool, force: bool) -> bool:
    if not output_path.exists() or force:
        return False
    if skip_existing:
        return True
    try:
        payload = load_json(output_path)
    except Exception:
        return False
    return payload.get("status") == "ok"


def status_rank(status: str) -> int:
    order = {
        "ok": 0,
        "partial": 1,
        "timeout": 2,
        "killed": 3,
        "error": 4,
        "missing": 5,
    }
    return order.get(status, 99)


def collect_campaign_summary(campaign_root: Path) -> tuple[dict[str, Any], str]:
    config_path = campaign_root / "campaign_config.json"
    config = load_json(config_path) if config_path.exists() else {}
    contexts = config.get("contexts", [])
    cases = config.get("cases", [])
    phases = config.get("phases", [])

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    prompt_mismatches: list[dict[str, Any]] = []

    for context_len in contexts:
        prompt_hashes: dict[str, str | None] = {}
        for case in cases:
            case_dir = campaign_root / str(context_len) / case
            prompt_meta_path = campaign_root / str(context_len) / "prompt" / "prompt_meta.json"
            if prompt_meta_path.exists():
                prompt_meta = load_json(prompt_meta_path)
                prompt_hashes[case] = prompt_meta.get("prompt_hash")

            phase_statuses = {}
            for phase in phases:
                path = case_dir / f"{phase}.json"
                if not path.exists():
                    phase_statuses[phase] = "missing"
                    missing.append(
                        {
                            "context_len": context_len,
                            "case": case,
                            "phase": phase,
                            "path": str(path),
                        }
                    )
                    continue

                payload = load_json(path)
                status = payload.get("status", "error")
                phase_statuses[phase] = status
                if status != "ok":
                    failed.append(
                        {
                            "context_len": context_len,
                            "case": case,
                            "phase": phase,
                            "status": status,
                            "path": str(path),
                            "exit_code": payload.get("exit_code"),
                        }
                    )
                if case not in prompt_hashes and payload.get("prompt_hash"):
                    prompt_hashes[case] = payload.get("prompt_hash")

                rows.append(
                    {
                        "context_len": context_len,
                        "case": case,
                        "phase": phase,
                        "status": status,
                        "elapsed_s": payload.get("elapsed_s"),
                        "prompt_hash": payload.get("prompt_hash"),
                        "sample_text_present": bool(payload.get("sample_text")),
                    }
                )

        baseline_hash = prompt_hashes.get("baseline")
        tq_hash = prompt_hashes.get("tq")
        if baseline_hash and tq_hash and baseline_hash != tq_hash:
            prompt_mismatches.append(
                {
                    "context_len": context_len,
                    "baseline_prompt_hash": baseline_hash,
                    "tq_prompt_hash": tq_hash,
                }
            )

    ok_rows = sum(1 for row in rows if row["status"] == "ok")
    non_ok_rows = len(rows) - ok_rows
    exit_137_count = sum(1 for item in failed if item.get("exit_code") == 137 or item.get("status") == "killed")
    tq_200k_ok = any(
        row["context_len"] == 200000 and row["case"] == "tq" and row["phase"] == "full" and row["status"] == "ok"
        for row in rows
    )
    baseline_200k_ok = any(
        row["context_len"] == 200000 and row["case"] == "baseline" and row["phase"] == "full" and row["status"] == "ok"
        for row in rows
    )
    stable_through_120k = all(
        any(
            row["context_len"] == context_len
            and row["case"] == case
            and row["phase"] == phase
            and row["status"] == "ok"
            for row in rows
        )
        for context_len in [c for c in contexts if c <= 120000]
        for case in cases
        for phase in phases
    )

    requested_contexts = sorted(int(context) for context in contexts)
    only_sub_200k = bool(requested_contexts) and all(context < 200000 for context in requested_contexts)
    all_requested_complete = all(
        any(
            row["context_len"] == context_len
            and row["case"] == case
            and row["phase"] == phase
            and row["status"] == "ok"
            for row in rows
        )
        for context_len in requested_contexts
        for case in cases
        for phase in phases
    )

    if all_requested_complete and only_sub_200k and not prompt_mismatches:
        recommended_plan = "A"
    elif stable_through_120k and tq_200k_ok and not prompt_mismatches:
        recommended_plan = "A"
    elif tq_200k_ok and not baseline_200k_ok:
        recommended_plan = "C"
    elif exit_137_count > 0 or any(item["status"] in {"timeout", "killed"} for item in failed):
        recommended_plan = "B"
    else:
        recommended_plan = "investigate"

    manifest = {
        "campaign_id": config.get("campaign_id", campaign_root.name),
        "campaign_root": str(campaign_root),
        "generated_at": utc_now_iso(),
        "host": socket.gethostname(),
        "contexts": contexts,
        "cases": cases,
        "phases": phases,
        "rows": rows,
        "missing": missing,
        "failed": failed,
        "prompt_mismatches": prompt_mismatches,
        "ok_rows": ok_rows,
        "non_ok_rows": non_ok_rows,
        "recommended_plan": recommended_plan,
    }

    lines = [
        f"# TurboQuant Campaign Summary: {manifest['campaign_id']}",
        "",
        f"- Campaign root: `{campaign_root}`",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Recommended plan: `Plan {recommended_plan}`",
        f"- OK rows: `{ok_rows}`",
        f"- Non-OK rows: `{non_ok_rows}`",
        f"- Prompt mismatches: `{len(prompt_mismatches)}`",
        "",
        "| Context | Case | Phase | Status | Elapsed (s) | Sample Text |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in sorted(rows, key=lambda item: (item["context_len"], item["case"], item["phase"])):
        elapsed = "" if row["elapsed_s"] is None else f"{row['elapsed_s']:.3f}"
        lines.append(
            f"| {row['context_len']} | {row['case']} | {row['phase']} | {row['status']} | {elapsed} | "
            f"{'yes' if row['sample_text_present'] else 'no'} |"
        )
    if missing:
        lines.extend(
            [
                "",
                "## Missing",
            ]
        )
        for item in missing:
            lines.append(f"- {item['context_len']} / {item['case']} / {item['phase']} -> `{item['path']}`")
    if failed:
        lines.extend(
            [
                "",
                "## Failed",
            ]
        )
        for item in failed:
            lines.append(
                f"- {item['context_len']} / {item['case']} / {item['phase']} -> "
                f"{item['status']} (exit={item.get('exit_code')})"
            )
    if prompt_mismatches:
        lines.extend(
            [
                "",
                "## Prompt Mismatches",
            ]
        )
        for item in prompt_mismatches:
            lines.append(
                f"- {item['context_len']}: baseline `{item['baseline_prompt_hash']}` vs tq `{item['tq_prompt_hash']}`"
            )

    return manifest, "\n".join(lines) + "\n"
