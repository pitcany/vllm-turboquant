#!/usr/bin/env python3
"""Wait for remote recovery, sync the harness, and launch the Plan B campaign."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from tq_campaign import sync_remote
from tq_harness_lib import atomic_write_json, tail_text, utc_now_iso
from tq_remote_snapshot import build_remote_python, run_ssh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="your-gpu-host.example.com")
    parser.add_argument("--port", type=int, default=3003)
    parser.add_argument("--user", default="root")
    parser.add_argument("--identity-file", default="~/.ssh/id_rsa")
    parser.add_argument("--ssh-timeout-s", type=int, default=20)
    parser.add_argument("--probe-interval-s", type=int, default=30)
    parser.add_argument("--max-wait-s", type=int, default=300)
    parser.add_argument("--remote-scripts-dir", default="/5090-qwen3.5-27b/scripts")
    parser.add_argument("--remote-model-path", default="/5090-qwen3.5-27b/models/QuantTrio-Qwen3.5-27B-AWQ")
    parser.add_argument("--probe-campaign-root", default="/5090-qwen3.5-27b/logs/campaigns/smoke-50k-harness")
    parser.add_argument("--recovery-campaign-root", default="/5090-qwen3.5-27b/logs/campaigns/smoke-50k-planb")
    parser.add_argument("--contexts", default="50000")
    parser.add_argument("--cases", default="baseline,tq")
    parser.add_argument("--phases", default="init,prefill_only,decode_only,full")
    parser.add_argument("--prompt-seed", type=int, default=5090)
    parser.add_argument("--max-output-tokens", type=int, default=24)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def probe_remote(args: argparse.Namespace, mode: str, campaign_root: str) -> dict:
    payload = {
        "mode": mode,
        "campaign_root": campaign_root,
        "ok": False,
        "checked_at": utc_now_iso(),
    }
    remote_python = build_remote_python(campaign_root, mode)
    try:
        stdout = run_ssh(args, f"python3 - <<'PY'\n{remote_python}\nPY")
        data = json.loads(stdout)
        payload["ok"] = True
        payload["payload"] = data
    except subprocess.TimeoutExpired as exc:
        payload["error_type"] = "ssh_timeout"
        payload["error_message"] = str(exc)
    except subprocess.CalledProcessError as exc:
        payload["error_type"] = "ssh_failed"
        payload["error_message"] = str(exc)
        payload["stdout_tail"] = tail_text(exc.stdout)
        payload["stderr_tail"] = tail_text(exc.stderr)
    return payload


def build_remote_launch_command(args: argparse.Namespace) -> str:
    launch_log = f"{args.recovery_campaign_root}/launch.log"
    return (
        f"mkdir -p {args.recovery_campaign_root} && "
        f"cd {args.remote_scripts_dir} && "
        "nohup env "
        "CUDA_VISIBLE_DEVICES=0 "
        "VLLM_ENABLE_V1_MULTIPROCESSING=0 "
        "TOKENIZERS_PARALLELISM=false "
        f"/5090-qwen3.5-27b/.venv/bin/python tq_campaign.py "
        f"--model-path {args.remote_model_path} "
        f"--contexts {args.contexts} "
        f"--cases {args.cases} "
        f"--phases {args.phases} "
        f"--campaign-root {args.recovery_campaign_root} "
        f"--prompt-seed {args.prompt_seed} "
        f"--max-output-tokens {args.max_output_tokens} "
        f"--gpu-memory-utilization {args.gpu_memory_utilization} "
        f"--tensor-parallel-size {args.tensor_parallel_size} "
        "--strict-timeouts --force "
        f"> {launch_log} 2>&1 < /dev/null & echo $!"
    )


def run_remote_command(args: argparse.Namespace, remote_cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh",
            "-F",
            "/dev/null",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=12",
            "-o",
            f"IdentityFile={args.identity_file}",
            "-o",
            "IdentitiesOnly=yes",
            f"{args.user}@{args.host}",
            "-p",
            str(args.port),
            remote_cmd,
        ],
        capture_output=True,
        text=True,
        timeout=args.ssh_timeout_s,
        check=True,
    )


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    report = {
        "started_at": utc_now_iso(),
        "status": "error",
        "host": args.host,
        "port": args.port,
        "probe_campaign_root": args.probe_campaign_root,
        "recovery_campaign_root": args.recovery_campaign_root,
        "phases": args.phases.split(","),
        "checks": [],
        "sync": None,
        "launch": None,
    }

    attempts = max(1, math.ceil(args.max_wait_s / max(args.probe_interval_s, 1)))
    for attempt in range(1, attempts + 1):
        probe = probe_remote(args, "ping", args.probe_campaign_root)
        probe["attempt"] = attempt
        report["checks"].append(probe)
        if probe["ok"]:
            break
        if attempt < attempts and not args.dry_run:
            time.sleep(args.probe_interval_s)

    if not report["checks"][-1]["ok"]:
        report["error_type"] = report["checks"][-1].get("error_type", "probe_failed")
        report["error_message"] = report["checks"][-1].get("error_message", "Ping probe did not recover")
        report["ended_at"] = utc_now_iso()
        atomic_write_json(output_path, report)
        print(output_path)
        return 1

    summary_probe = probe_remote(args, "summary", args.probe_campaign_root)
    report["checks"].append(summary_probe)
    if not summary_probe["ok"]:
        report["error_type"] = summary_probe.get("error_type", "summary_probe_failed")
        report["error_message"] = summary_probe.get("error_message", "Summary probe failed")
        report["ended_at"] = utc_now_iso()
        atomic_write_json(output_path, report)
        print(output_path)
        return 1

    if args.dry_run:
        report["sync"] = {"status": "skipped", "reason": "dry_run"}
        report["launch"] = {
            "status": "skipped",
            "reason": "dry_run",
            "remote_command": build_remote_launch_command(args),
        }
        report["status"] = "ok"
        report["ended_at"] = utc_now_iso()
        atomic_write_json(output_path, report)
        print(output_path)
        return 0

    sync_args = SimpleNamespace(
        remote_host=args.host,
        remote_port=args.port,
        remote_user=args.user,
        remote_key=args.identity_file,
        remote_scripts_dir=args.remote_scripts_dir,
    )
    try:
        sync_remote(sync_args)
        report["sync"] = {"status": "ok", "synced_at": utc_now_iso()}
    except Exception as exc:  # noqa: BLE001
        report["sync"] = {"status": "error", "error_type": exc.__class__.__name__, "error_message": str(exc)}
        report["error_type"] = "sync_failed"
        report["error_message"] = str(exc)
        report["ended_at"] = utc_now_iso()
        atomic_write_json(output_path, report)
        print(output_path)
        return 1

    launch_cmd = build_remote_launch_command(args)
    try:
        proc = run_remote_command(args, launch_cmd)
        report["launch"] = {
            "status": "ok",
            "launched_at": utc_now_iso(),
            "remote_command": launch_cmd,
            "stdout_tail": tail_text(proc.stdout),
            "stderr_tail": tail_text(proc.stderr),
            "worker_pid": proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else None,
            "launch_log": f"{args.recovery_campaign_root}/launch.log",
        }
    except subprocess.TimeoutExpired as exc:
        report["launch"] = {"status": "error", "error_type": "ssh_timeout", "error_message": str(exc)}
        report["error_type"] = "launch_timeout"
        report["error_message"] = str(exc)
        report["ended_at"] = utc_now_iso()
        atomic_write_json(output_path, report)
        print(output_path)
        return 1
    except subprocess.CalledProcessError as exc:
        report["launch"] = {
            "status": "error",
            "error_type": "ssh_failed",
            "error_message": str(exc),
            "stdout_tail": tail_text(exc.stdout),
            "stderr_tail": tail_text(exc.stderr),
        }
        report["error_type"] = "launch_failed"
        report["error_message"] = str(exc)
        report["ended_at"] = utc_now_iso()
        atomic_write_json(output_path, report)
        print(output_path)
        return 1

    report["status"] = "ok"
    report["ended_at"] = utc_now_iso()
    atomic_write_json(output_path, report)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
