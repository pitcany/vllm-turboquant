#!/usr/bin/env python3
"""Run exactly one TurboQuant telemetry phase and always emit a JSON artifact."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tq_harness_lib import (
    DEFAULT_QUALITY_PROMPT,
    atomic_write_json,
    ensure_dir,
    ensure_repo_import_path,
    inspect_runtime_stats,
    load_prompt_artifacts,
    maybe_free_tq_kv,
    probe_gpu_metrics,
    scrub_gpu_processes,
    tail_text,
    utc_now_iso,
)

PHASES = ("init", "ttft", "prefill_only", "full", "decode_only", "quality")
CASES = ("baseline", "tq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-s", type=int, required=True)
    parser.add_argument("--prompt-seed", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--heartbeat-interval-s", type=float, default=15.0)
    parser.add_argument("--worker-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-sleep-s", type=float, default=0.0, help=argparse.SUPPRESS)
    return parser.parse_args()


def status_path_for(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + ".status.json")


def worker_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        "--case",
        args.case,
        "--phase",
        args.phase,
        "--context-len",
        str(args.context_len),
        "--output",
        args.output,
        "--timeout-s",
        str(args.timeout_s),
        "--prompt-seed",
        str(args.prompt_seed),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--model-path",
        args.model_path,
        "--prompt-dir",
        args.prompt_dir,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--heartbeat-interval-s",
        str(args.heartbeat_interval_s),
    ]
    if args.campaign_id:
        cmd.extend(["--campaign-id", args.campaign_id])
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.dry_run_sleep_s:
        cmd.extend(["--dry-run-sleep-s", str(args.dry_run_sleep_s)])
    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    return cmd


def build_supervisor_payload(args: argparse.Namespace, output_path: Path) -> dict[str, Any]:
    return {
        "campaign_id": args.campaign_id,
        "host": socket.gethostname(),
        "case": args.case,
        "phase": args.phase,
        "context_len": args.context_len,
        "prompt_seed": args.prompt_seed,
        "max_output_tokens": args.max_output_tokens,
        "model_path": args.model_path,
        "prompt_dir": args.prompt_dir,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "output_path": str(output_path),
        "status": "running",
        "start_ts": utc_now_iso(),
    }


def write_heartbeat(status_path: Path, payload: dict[str, Any], process: subprocess.Popen[str]) -> None:
    heartbeat = dict(payload)
    heartbeat.update(
        {
            "status": "running",
            "supervisor_pid": os.getpid(),
            "worker_pid": process.pid,
            "heartbeat_ts": utc_now_iso(),
        }
    )
    atomic_write_json(status_path, heartbeat)


def parse_last_json_line(text: str) -> dict[str, Any] | None:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def run_supervisor(args: argparse.Namespace) -> int:
    output_path = Path(args.output).resolve()
    status_path = status_path_for(output_path)
    ensure_dir(output_path.parent)

    payload = build_supervisor_payload(args, output_path)
    process = subprocess.Popen(
        worker_command(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )

    start_monotonic = time.monotonic()
    next_heartbeat = start_monotonic
    timed_out = False

    while True:
        now = time.monotonic()
        if now >= next_heartbeat:
            write_heartbeat(status_path, payload, process)
            next_heartbeat = now + args.heartbeat_interval_s

        rc = process.poll()
        if rc is not None:
            break

        if now - start_monotonic > args.timeout_s:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            break
        time.sleep(0.5)

    stdout, stderr = process.communicate()
    worker_payload = parse_last_json_line(stdout)
    end_ts = utc_now_iso()
    elapsed_s = round(time.monotonic() - start_monotonic, 6)

    if worker_payload:
        payload.update(worker_payload)

    payload["stdout_tail"] = tail_text(stdout)
    payload["stderr_tail"] = tail_text(stderr)
    payload["exit_code"] = process.returncode
    payload["end_ts"] = end_ts
    payload["elapsed_s"] = payload.get("elapsed_s", elapsed_s)

    if timed_out:
        payload["status"] = "timeout"
        payload["error_type"] = "phase_timeout"
    elif process.returncode == 137 or process.returncode == -signal.SIGKILL:
        payload["status"] = "killed"
        payload["error_type"] = "exit_137"
    elif process.returncode and payload.get("status") == "ok":
        payload["status"] = "error"
        payload["error_type"] = payload.get("error_type", "worker_nonzero_exit")
    elif not worker_payload:
        payload["status"] = "error"
        payload["error_type"] = "missing_worker_json"

    atomic_write_json(output_path, payload)
    atomic_write_json(status_path, payload)
    return 0 if payload["status"] == "ok" else 1


def compute_max_model_len(args: argparse.Namespace) -> int:
    if args.max_model_len is not None:
        return args.max_model_len
    reserve = max(512, args.max_output_tokens + 256)
    return args.context_len + reserve


def worker_main(args: argparse.Namespace) -> int:
    try:
        ensure_repo_import_path()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        prompt_text, prompt_meta = load_prompt_artifacts(Path(args.prompt_dir))
        prompt_hash = prompt_meta.get("prompt_hash")
        prompt_tokens = prompt_meta.get("prompt_tokens")
        init_gpu = probe_gpu_metrics()

        result: dict[str, Any] = {
            "campaign_id": args.campaign_id,
            "host": socket.gethostname(),
            "case": args.case,
            "phase": args.phase,
            "context_len": args.context_len,
            "prompt_seed": args.prompt_seed,
            "prompt_hash": prompt_hash,
            "prompt_tokens": prompt_tokens,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": init_gpu.get("gpu_name"),
            "memory_used_mb": init_gpu.get("memory_used_mb"),
            "memory_free_mb": init_gpu.get("memory_free_mb"),
            "power_w": init_gpu.get("power_w"),
            "temp_c": init_gpu.get("temp_c"),
            "start_ts": utc_now_iso(),
            "status": "ok",
            "stdout_tail": "",
            "stderr_tail": "",
            "error_type": None,
        }

        if args.dry_run:
            if args.dry_run_sleep_s:
                time.sleep(args.dry_run_sleep_s)
            result.update(
                {
                    "init_s": 0.123,
                    "ttft_s": 0.456 if args.phase in {"ttft", "prefill_only"} else None,
                    "prefill_tok_s": float(prompt_tokens or 0) / 0.456 if args.phase in {"ttft", "prefill_only"} else None,
                    "gen_tok_s": float(args.max_output_tokens) / 1.5 if args.phase in {"full", "decode_only", "quality"} else None,
                    "sample_text": "dry run output" if args.phase in {"full", "decode_only", "quality"} else "",
                    "quality_prompt": DEFAULT_QUALITY_PROMPT if args.phase == "quality" else None,
                    "status": "ok",
                    "end_ts": utc_now_iso(),
                    "elapsed_s": 0.789,
                }
            )
            print(json.dumps(result))
            return 0

        init_start = time.perf_counter()
        if args.case == "tq":
            from turboquant.vllm_attn_backend import enable_no_alloc

            enable_no_alloc(key_bits=3, value_bits=2, buffer_size=128, initial_layers_count=4)

        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model_path,
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=compute_max_model_len(args),
            tensor_parallel_size=args.tensor_parallel_size,
            max_num_seqs=1,
            enforce_eager=args.enforce_eager,
        )
        init_elapsed = time.perf_counter() - init_start
        post_init_gpu = probe_gpu_metrics()
        runtime_stats = inspect_runtime_stats(llm)

        result.update(
            {
                "init_s": round(init_elapsed, 6),
                "memory_used_mb": post_init_gpu.get("memory_used_mb"),
                "memory_free_mb": post_init_gpu.get("memory_free_mb"),
                "power_w": post_init_gpu.get("power_w"),
                "temp_c": post_init_gpu.get("temp_c"),
                "gpu_name": post_init_gpu.get("gpu_name"),
                "tq_hooked_layers": runtime_stats.get("tq_hooked_layers"),
                "shared_kv_layers": runtime_stats.get("shared_kv_layers"),
                "kv_reserved_gb": runtime_stats.get("kv_reserved_gb"),
                "tq_total_memory_bytes": runtime_stats.get("tq_total_memory_bytes"),
                "tq_total_compressed_tokens": runtime_stats.get("tq_total_compressed_tokens"),
                "tq_mode": runtime_stats.get("tq_mode"),
            }
        )

        if args.phase == "init":
            result["elapsed_s"] = round(init_elapsed, 6)
            result["end_ts"] = utc_now_iso()
            print(json.dumps(result))
            return 0

        run_prompt = prompt_text
        max_tokens = args.max_output_tokens
        if args.phase == "quality":
            run_prompt = DEFAULT_QUALITY_PROMPT
            max_tokens = max(args.max_output_tokens, 64)
            result["quality_prompt"] = DEFAULT_QUALITY_PROMPT
        elif args.phase in {"ttft", "prefill_only"}:
            max_tokens = 1
            if args.phase == "prefill_only":
                result["phase_note"] = (
                    "Prefill-only probe requests a single output token so elapsed time is dominated by long-context prefill."
                )
        elif args.phase == "decode_only":
            result["phase_note"] = (
                "Decode-only is an isolated multi-token generate run. "
                "Because vLLM does not expose reusable prefill state across processes here, this is decode-dominant rather than perfectly decode-exclusive."
            )

        sample_start = time.perf_counter()
        outputs = llm.generate([run_prompt], SamplingParams(temperature=0, max_tokens=max_tokens))
        sample_elapsed = time.perf_counter() - sample_start
        response = outputs[0].outputs[0]
        post_phase_gpu = probe_gpu_metrics()
        result["sample_text"] = response.text
        result["output_tokens"] = len(response.token_ids)
        result["elapsed_s"] = round(sample_elapsed, 6)
        result["end_ts"] = utc_now_iso()
        result["activation_est_mb"] = (
            post_phase_gpu.get("memory_used_mb") - post_init_gpu.get("memory_used_mb")
            if post_phase_gpu.get("memory_used_mb") is not None
            and post_init_gpu.get("memory_used_mb") is not None
            else None
        )
        result["memory_used_mb"] = post_phase_gpu.get("memory_used_mb")
        result["memory_free_mb"] = post_phase_gpu.get("memory_free_mb")
        result["power_w"] = post_phase_gpu.get("power_w")
        result["temp_c"] = post_phase_gpu.get("temp_c")

        if args.phase in {"ttft", "prefill_only"}:
            result["ttft_s"] = round(sample_elapsed, 6)
            result["prefill_tok_s"] = round((prompt_tokens or 0) / sample_elapsed, 6) if sample_elapsed else None
        elif args.phase in {"full", "decode_only", "quality"}:
            result["gen_tok_s"] = round(len(response.token_ids) / sample_elapsed, 6) if sample_elapsed else None

        if args.case == "tq":
            result.update(maybe_free_tq_kv(llm))
            post_free_gpu = probe_gpu_metrics()
            result["memory_after_free_mb"] = post_free_gpu.get("memory_used_mb")
            result["memory_free_after_free_mb"] = post_free_gpu.get("memory_free_mb")

        print(json.dumps(result))
        return 0
    except Exception as exc:
        error_payload = {
            "campaign_id": args.campaign_id,
            "case": args.case,
            "phase": args.phase,
            "context_len": args.context_len,
            "prompt_seed": args.prompt_seed,
            "status": "error",
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
            "end_ts": utc_now_iso(),
        }
        print(json.dumps(error_payload))
        return 1


def main() -> int:
    args = parse_args()
    if args.worker_mode:
        return worker_main(args)
    return run_supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())
