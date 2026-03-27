#!/usr/bin/env python3
"""Run a resumable TurboQuant telemetry campaign across contexts, cases, and phases."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from tq_harness_lib import (
    DEFAULT_CASES,
    DEFAULT_CONTEXTS,
    DEFAULT_PHASES,
    atomic_write_json,
    atomic_write_text,
    build_python_launcher,
    collect_campaign_summary,
    ensure_dir,
    generate_prompt_artifacts,
    parse_csv_ints,
    parse_csv_strings,
    phase_timeout_for,
    scrub_gpu_processes,
    should_skip_phase,
    timestamp_slug,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", default=None)
    parser.add_argument("--contexts", default=None)
    parser.add_argument("--cases", default=None)
    parser.add_argument("--phases", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-seed", type=int, default=5090)
    parser.add_argument("--max-output-tokens", type=int, default=24)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-timeouts", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sync-remote", action="store_true")
    parser.add_argument("--remote-host", default="your-gpu-host.example.com")
    parser.add_argument("--remote-port", type=int, default=3003)
    parser.add_argument("--remote-user", default="root")
    parser.add_argument("--remote-key", default="~/.ssh/id_rsa")
    parser.add_argument("--remote-scripts-dir", default="/5090-qwen3.5-27b/scripts")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def phase_runner_path() -> Path:
    return Path(__file__).resolve().parent / "tq_phase_runner.py"


def collector_path() -> Path:
    return Path(__file__).resolve().parent / "tq_collect_report.py"


def scripts_to_sync() -> list[Path]:
    scripts_dir = Path(__file__).resolve().parent
    return [
        scripts_dir / "tq_harness_lib.py",
        scripts_dir / "tq_phase_runner.py",
        scripts_dir / "tq_campaign.py",
        scripts_dir / "tq_collect_report.py",
    ]


def sync_remote(args: argparse.Namespace) -> None:
    ssh_base = [
        "ssh",
        "-F",
        "/dev/null",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=12",
        "-o",
        f"IdentityFile={args.remote_key}",
        "-o",
        "IdentitiesOnly=yes",
        f"{args.remote_user}@{args.remote_host}",
        "-p",
        str(args.remote_port),
    ]
    subprocess.run(
        ssh_base + [f"mkdir -p {args.remote_scripts_dir}"],
        check=True,
    )

    for script_path in scripts_to_sync():
        subprocess.run(
            [
                "scp",
                "-F",
                "/dev/null",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                f"IdentityFile={args.remote_key}",
                "-o",
                "IdentitiesOnly=yes",
                "-P",
                str(args.remote_port),
                str(script_path),
                f"{args.remote_user}@{args.remote_host}:{args.remote_scripts_dir}/{script_path.name}",
            ],
            check=True,
        )


def build_campaign_root(args: argparse.Namespace) -> Path:
    if args.campaign_root:
        return Path(args.campaign_root).resolve()
    return Path("logs") / "campaigns" / timestamp_slug()


def run_phase(args: argparse.Namespace, campaign_root: Path, campaign_id: str, context_len: int, case: str, phase: str) -> int:
    output_path = campaign_root / str(context_len) / case / f"{phase}.json"
    if should_skip_phase(output_path, skip_existing=args.skip_existing or args.resume, force=args.force):
        return 0

    prompt_dir = campaign_root / str(context_len) / "prompt"
    timeout_s = phase_timeout_for(case, phase, context_len, strict_timeouts=args.strict_timeouts)
    cmd = build_python_launcher() + [
        str(phase_runner_path()),
        "--case",
        case,
        "--phase",
        phase,
        "--context-len",
        str(context_len),
        "--output",
        str(output_path),
        "--timeout-s",
        str(timeout_s),
        "--prompt-seed",
        str(args.prompt_seed),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--model-path",
        args.model_path,
        "--prompt-dir",
        str(prompt_dir),
        "--campaign-id",
        campaign_id,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)
    return proc.returncode


def main() -> int:
    args = parse_args()
    contexts = parse_csv_ints(args.contexts, DEFAULT_CONTEXTS)
    cases = parse_csv_strings(args.cases, DEFAULT_CASES)
    phases = parse_csv_strings(args.phases, DEFAULT_PHASES)
    campaign_root = build_campaign_root(args)
    campaign_root = ensure_dir(campaign_root)
    campaign_id = campaign_root.name

    if args.sync_remote:
        sync_remote(args)

    config = {
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root),
        "created_at": utc_now_iso(),
        "model_path": args.model_path,
        "prompt_seed": args.prompt_seed,
        "contexts": contexts,
        "cases": cases,
        "phases": phases,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "strict_timeouts": args.strict_timeouts,
        "dry_run": args.dry_run,
    }
    atomic_write_json(campaign_root / "campaign_config.json", config)

    gate_stop = False
    gate_reason = None

    for context_len in contexts:
        prompt_dir = campaign_root / str(context_len) / "prompt"
        generate_prompt_artifacts(
            model_path=args.model_path,
            context_len=context_len,
            seed=args.prompt_seed,
            prompt_dir=prompt_dir,
            force=args.force,
            dry_run=args.dry_run,
        )

        for case in cases:
            attempts = 2 if case == "baseline" and context_len >= 200000 else 1
            for phase in phases:
                if gate_stop:
                    break

                scrub_info = scrub_gpu_processes()
                atomic_write_json(
                    campaign_root / str(context_len) / case / f"{phase}.prescrub.json",
                    {
                        "campaign_id": campaign_id,
                        "context_len": context_len,
                        "case": case,
                        "phase": phase,
                        "scrub": scrub_info,
                        "timestamp": utc_now_iso(),
                    },
                )

                rc = 1
                for _ in range(attempts):
                    rc = run_phase(args, campaign_root, campaign_id, context_len, case, phase)
                    if rc == 0:
                        break

                manifest, summary = collect_campaign_summary(campaign_root)
                atomic_write_json(campaign_root / "manifest.json", manifest)
                atomic_write_text(campaign_root / "summary.md", summary)

                if rc != 0 and context_len <= 120000 and args.strict_timeouts:
                    gate_stop = True
                    gate_reason = (
                        f"Strict timeout gate tripped at context {context_len} "
                        f"case={case} phase={phase}; stopping higher-context execution."
                    )
                    break
            if gate_stop:
                break
        if gate_stop:
            break

    manifest, summary = collect_campaign_summary(campaign_root)
    if gate_reason:
        manifest["gate_stop_reason"] = gate_reason
        summary += f"\n## Gate Stop\n- {gate_reason}\n"
    atomic_write_json(campaign_root / "manifest.json", manifest)
    atomic_write_text(campaign_root / "summary.md", summary)

    collector_cmd = build_python_launcher() + [
        str(collector_path()),
        "--campaign-root",
        str(campaign_root),
    ]
    if args.dry_run:
        collector_cmd.append("--dry-run")
    subprocess.run(collector_cmd, cwd=Path(__file__).resolve().parent.parent, check=False)

    print(f"campaign_root={campaign_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
