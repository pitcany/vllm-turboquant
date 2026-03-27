#!/usr/bin/env python3
"""Watch the local autoresearch state and trigger staged recovery when ready."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from tq_harness_lib import atomic_write_json, ensure_dir, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-json", default="test-output/autoresearch-driver/status.json")
    parser.add_argument("--status-md", default="test-output/autoresearch-driver/status.md")
    parser.add_argument("--status-script", default="scripts/tq_autoresearch_status.py")
    parser.add_argument("--dashboard-script", default="scripts/tq_autoresearch_dashboard.py")
    parser.add_argument("--dashboard-output", default="test-output/autoresearch-driver/dashboard.html")
    parser.add_argument("--driver-script", default="scripts/tq_autoresearch_driver.py")
    parser.add_argument("--driver-output", default="test-output/autoresearch-driver/driver-report.json")
    parser.add_argument("--watch-report", required=True)
    parser.add_argument("--poll-interval-s", type=int, default=60)
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--run-driver-when-ready", action="store_true")
    parser.add_argument("--driver-dry-run", action="store_true")
    return parser.parse_args()


def build_status_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        args.status_script,
        "--output-json",
        args.status_json,
        "--output-md",
        args.status_md,
    ]


def build_driver_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        args.driver_script,
        "--output",
        args.driver_output,
    ]
    if args.driver_dry_run:
        cmd.append("--dry-run")
    return cmd


def build_dashboard_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        args.dashboard_script,
        "--status-json",
        args.status_json,
        "--watch-report",
        args.watch_report,
        "--output",
        args.dashboard_output,
    ]


def load_status(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def should_trigger_driver(status_payload: dict | None) -> bool:
    if not status_payload:
        return False
    return status_payload.get("blocker_code") == "ready"


def main() -> int:
    args = parse_args()
    watch_report_path = Path(args.watch_report)
    ensure_dir(watch_report_path.parent)
    history: list[dict] = []
    iteration = 0
    driver_triggered = False

    while True:
        iteration += 1
        step = {"iteration": iteration, "checked_at": utc_now_iso()}

        status_cmd = build_status_command(args)
        status_proc = subprocess.run(status_cmd, capture_output=True, text=True)
        step["status_command"] = status_cmd
        step["status_returncode"] = status_proc.returncode
        step["status_stdout_tail"] = status_proc.stdout[-4000:] if status_proc.stdout else ""
        step["status_stderr_tail"] = status_proc.stderr[-4000:] if status_proc.stderr else ""

        status_payload = load_status(Path(args.status_json))
        step["blocker_code"] = status_payload.get("blocker_code") if status_payload else None
        step["blocker_summary"] = status_payload.get("blocker_summary") if status_payload else None

        dashboard_cmd = build_dashboard_command(args)
        dashboard_proc = subprocess.run(dashboard_cmd, capture_output=True, text=True)
        step["dashboard_command"] = dashboard_cmd
        step["dashboard_returncode"] = dashboard_proc.returncode
        step["dashboard_stdout_tail"] = dashboard_proc.stdout[-4000:] if dashboard_proc.stdout else ""
        step["dashboard_stderr_tail"] = dashboard_proc.stderr[-4000:] if dashboard_proc.stderr else ""

        if args.run_driver_when_ready and should_trigger_driver(status_payload):
            driver_cmd = build_driver_command(args)
            driver_proc = subprocess.run(driver_cmd, capture_output=True, text=True)
            step["driver_command"] = driver_cmd
            step["driver_returncode"] = driver_proc.returncode
            step["driver_stdout_tail"] = driver_proc.stdout[-4000:] if driver_proc.stdout else ""
            step["driver_stderr_tail"] = driver_proc.stderr[-4000:] if driver_proc.stderr else ""
            driver_triggered = True

        history.append(step)
        report = {
            "started_at": history[0]["checked_at"],
            "ended_at": utc_now_iso(),
            "iterations": iteration,
            "driver_triggered": driver_triggered,
            "history": history,
        }
        atomic_write_json(watch_report_path, report)

        if driver_triggered:
            print(watch_report_path)
            return 0

        if args.max_iterations and iteration >= args.max_iterations:
            print(watch_report_path)
            return 0

        time.sleep(args.poll_interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
