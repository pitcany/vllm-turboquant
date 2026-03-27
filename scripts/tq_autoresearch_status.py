#!/usr/bin/env python3
"""Aggregate local TurboQuant autoresearch artifacts into a status packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tq_harness_lib import atomic_write_json, atomic_write_text, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-smoke-dir", default="test-output/remote-smoke-50k")
    parser.add_argument("--driver-report", default="test-output/autoresearch-driver/driver-report.json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def choose_latest_probe(remote_smoke_dir: Path) -> tuple[str | None, dict[str, Any] | None]:
    candidates = [
        ("ping_latest", remote_smoke_dir / "ping-snapshot-latest.json"),
        ("ping", remote_smoke_dir / "ping-snapshot.json"),
        ("summary", remote_smoke_dir / "health-snapshot-summary.json"),
        ("retry", remote_smoke_dir / "retry-snapshot.json"),
    ]
    for name, path in candidates:
        payload = load_json_if_exists(path)
        if payload is not None:
            return name, payload
    return None, None


def summarize_blocker(probe: dict[str, Any] | None, recovery: dict[str, Any] | None) -> tuple[str, str]:
    if probe and probe.get("status") == "error":
        error_type = probe.get("error_type", "unknown")
        stderr_tail = probe.get("stderr_tail", "")
        if "Operation timed out" in stderr_tail:
            return "ssh_outage", "Remote SSH is timing out on your-gpu-host.example.com:3003"
        return error_type, probe.get("error_message", "Remote probe failed")
    if recovery and recovery.get("status") == "error":
        return recovery.get("error_type", "recovery_error"), recovery.get("error_message", "Recovery driver failed")
    return "ready", "No current local blocker recorded"


def extract_next_command(driver_report: dict[str, Any] | None) -> str | None:
    if not driver_report:
        return None
    for stage in driver_report.get("stages", []):
        if stage.get("status") in {"planned", "error"}:
            command_shell = stage.get("command_shell")
            if not command_shell:
                return None
            return command_shell.replace(" --dry-run", "")
    return None


def build_status_payload(remote_smoke_dir: Path, driver_report_path: Path) -> dict[str, Any]:
    local_manifest = load_json_if_exists(remote_smoke_dir / "local-manifest.json")
    recovery_report = load_json_if_exists(remote_smoke_dir / "resume-recovery-report.json")
    driver_report = load_json_if_exists(driver_report_path)
    probe_name, probe_payload = choose_latest_probe(remote_smoke_dir)
    blocker_code, blocker_summary = summarize_blocker(probe_payload, recovery_report)
    next_command = extract_next_command(driver_report)

    return {
        "generated_at": utc_now_iso(),
        "remote_smoke_dir": str(remote_smoke_dir),
        "driver_report_path": str(driver_report_path),
        "campaign_plan": local_manifest.get("recommended_plan") if local_manifest else None,
        "campaign_failed": local_manifest.get("failed", []) if local_manifest else [],
        "campaign_missing": local_manifest.get("missing", []) if local_manifest else [],
        "latest_probe_name": probe_name,
        "latest_probe": probe_payload,
        "recovery_report": recovery_report,
        "driver_report": driver_report,
        "blocker_code": blocker_code,
        "blocker_summary": blocker_summary,
        "next_command": next_command,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# TurboQuant Autoresearch Status",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Blocker: `{payload['blocker_code']}`",
        f"- Summary: {payload['blocker_summary']}",
        f"- 50k campaign plan: `{payload.get('campaign_plan')}`",
        f"- Latest probe: `{payload.get('latest_probe_name')}`",
        "",
    ]

    latest_probe = payload.get("latest_probe") or {}
    if latest_probe:
        lines.extend(
            [
                "## Latest Probe",
                f"- Status: `{latest_probe.get('status', 'unknown')}`",
                f"- Error type: `{latest_probe.get('error_type')}`",
                f"- stderr tail: `{latest_probe.get('stderr_tail', '')}`",
                "",
            ]
        )

    failed = payload.get("campaign_failed") or []
    missing = payload.get("campaign_missing") or []
    if failed:
        lines.append("## Failed")
        for item in failed:
            lines.append(
                f"- {item.get('context_len')} / {item.get('case')} / {item.get('phase')} -> {item.get('status')} (exit={item.get('exit_code')})"
            )
        lines.append("")
    if missing:
        lines.append("## Missing")
        for item in missing:
            lines.append(f"- {item.get('context_len')} / {item.get('case')} / {item.get('phase')}")
        lines.append("")

    lines.append("## Next Command")
    if payload.get("next_command"):
        lines.append(f"```bash\n{payload['next_command']}\n```")
    else:
        lines.append("- No next command available")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    remote_smoke_dir = Path(args.remote_smoke_dir)
    driver_report_path = Path(args.driver_report)
    payload = build_status_payload(remote_smoke_dir, driver_report_path)
    markdown = render_markdown(payload)
    atomic_write_json(Path(args.output_json), payload)
    atomic_write_text(Path(args.output_md), markdown)
    print(args.output_json)
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
