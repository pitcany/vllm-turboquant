#!/usr/bin/env python3
"""Compose and optionally execute staged TurboQuant autoresearch recovery runs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from tq_harness_lib import atomic_write_json, ensure_dir, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", default="50000,80000,120000,200000")
    parser.add_argument("--base-remote-root", default="/5090-qwen3.5-27b/logs/campaigns")
    parser.add_argument("--remote-model-path", default="/5090-qwen3.5-27b/models/QuantTrio-Qwen3.5-27B-AWQ")
    parser.add_argument("--host", default="your-gpu-host.example.com")
    parser.add_argument("--port", type=int, default=3003)
    parser.add_argument("--user", default="root")
    parser.add_argument("--identity-file", default="~/.ssh/id_rsa")
    parser.add_argument("--remote-scripts-dir", default="/5090-qwen3.5-27b/scripts")
    parser.add_argument("--probe-interval-s", type=int, default=30)
    parser.add_argument("--max-wait-s", type=int, default=300)
    parser.add_argument("--ssh-timeout-s", type=int, default=20)
    parser.add_argument("--prompt-seed", type=int, default=5090)
    parser.add_argument("--max-output-tokens", type=int, default=24)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--local-report-dir", default="test-output/autoresearch-driver")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_contexts(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def stage_name_for(context_len: int) -> str:
    if context_len == 50000:
        return "smoke-50k-planb"
    if context_len == 200000:
        return "smoke-200k-tq"
    return f"smoke-{context_len // 1000}k-planb"


def stage_cases_for(context_len: int) -> str:
    return "tq" if context_len >= 200000 else "baseline,tq"


def stage_phases_for(context_len: int) -> str:
    if context_len >= 200000:
        return "init,prefill_only,decode_only,full"
    return "init,prefill_only,decode_only,full"


def build_stage_specs(args: argparse.Namespace) -> list[dict]:
    contexts = parse_contexts(args.contexts)
    specs: list[dict] = []
    probe_root = f"{args.base_remote_root}/smoke-50k-harness"
    local_report_dir = Path(args.local_report_dir)

    for context_len in contexts:
        stage_name = stage_name_for(context_len)
        recovery_root = f"{args.base_remote_root}/{stage_name}"
        local_report_path = local_report_dir / f"{stage_name}-resume-report.json"
        specs.append(
            {
                "context_len": context_len,
                "stage_name": stage_name,
                "probe_campaign_root": probe_root,
                "recovery_campaign_root": recovery_root,
                "cases": stage_cases_for(context_len),
                "phases": stage_phases_for(context_len),
                "local_report_path": str(local_report_path),
                "notes": [
                    "200k stage is TQ-only by default" if context_len >= 200000 else "A/B stage uses baseline and TQ",
                    "Split phases isolate prefill/decode pressure",
                ],
            }
        )
        probe_root = recovery_root
    return specs


def build_resume_command(args: argparse.Namespace, stage: dict, dry_run: bool | None = None) -> list[str]:
    if dry_run is None:
        dry_run = args.dry_run
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "tq_resume_recovery.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--user",
        args.user,
        "--identity-file",
        args.identity_file,
        "--ssh-timeout-s",
        str(args.ssh_timeout_s),
        "--probe-interval-s",
        str(args.probe_interval_s),
        "--max-wait-s",
        str(args.max_wait_s),
        "--remote-scripts-dir",
        args.remote_scripts_dir,
        "--remote-model-path",
        args.remote_model_path,
        "--probe-campaign-root",
        stage["probe_campaign_root"],
        "--recovery-campaign-root",
        stage["recovery_campaign_root"],
        "--contexts",
        str(stage["context_len"]),
        "--cases",
        stage["cases"],
        "--phases",
        stage["phases"],
        "--prompt-seed",
        str(args.prompt_seed),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--output",
        stage["local_report_path"],
    ]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def command_to_shell(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    ensure_dir(Path(args.local_report_dir))

    stages = build_stage_specs(args)
    report = {
        "started_at": utc_now_iso(),
        "status": "ok",
        "dry_run": args.dry_run,
        "contexts": [stage["context_len"] for stage in stages],
        "stages": [],
    }

    for stage in stages:
        cmd = build_resume_command(args, stage)
        stage_report = {
            **stage,
            "command": cmd,
            "command_shell": command_to_shell(cmd),
            "status": "planned",
        }

        if not args.dry_run:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            stage_report["status"] = "ok" if proc.returncode == 0 else "error"
            stage_report["returncode"] = proc.returncode
            stage_report["stdout_tail"] = proc.stdout[-4000:] if proc.stdout else ""
            stage_report["stderr_tail"] = proc.stderr[-4000:] if proc.stderr else ""
            report["stages"].append(stage_report)
            if proc.returncode != 0:
                report["status"] = "error"
                report["error_stage"] = stage["stage_name"]
                break
        else:
            report["stages"].append(stage_report)

    report["ended_at"] = utc_now_iso()
    atomic_write_json(output_path, report)
    print(output_path)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
