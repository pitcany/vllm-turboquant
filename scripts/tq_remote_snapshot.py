#!/usr/bin/env python3
"""Capture a remote TurboQuant campaign snapshot over SSH into local JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from tq_harness_lib import atomic_write_json, atomic_write_text, ensure_dir, tail_text, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="your-gpu-host.example.com")
    parser.add_argument("--port", type=int, default=3003)
    parser.add_argument("--user", default="root")
    parser.add_argument("--identity-file", default="~/.ssh/id_rsa")
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mirror-root", default=None)
    parser.add_argument("--mode", choices=("ping", "summary", "full"), default="full")
    parser.add_argument("--ssh-timeout-s", type=int, default=30)
    return parser.parse_args()


def run_ssh(args: argparse.Namespace, remote_cmd: str) -> str:
    proc = subprocess.run(
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
        check=True,
        timeout=args.ssh_timeout_s,
    )
    return proc.stdout


def write_mirror_files(payload: dict, mirror_root: Path) -> None:
    ensure_dir(mirror_root)

    manifest = payload.get("manifest")
    if manifest is not None:
        atomic_write_json(mirror_root / "manifest.json", manifest)

    summary_text = payload.get("summary_text")
    if summary_text:
        atomic_write_text(mirror_root / "summary.md", summary_text)

    for entry in payload.get("files", []):
        rel_path = entry.get("relative_path")
        if not rel_path:
            continue
        target = mirror_root / rel_path
        text = entry.get("text")
        if text is None:
            continue
        if entry.get("type") == "json":
            atomic_write_json(target, json.loads(text))
        else:
            atomic_write_text(target, text)


def build_remote_python(root: str, mode: str) -> str:
    if mode == "ping":
        return f"""
import json
import socket
from pathlib import Path

root = Path({root!r})
payload = {{
    "campaign_root": str(root),
    "generated_at": None,
    "manifest": None,
    "summary_head": None,
    "summary_text": None,
    "phase_files": [],
    "files": [],
    "mode": "ping",
    "remote_host": socket.gethostname(),
    "campaign_root_exists": root.exists(),
}}
print(json.dumps(payload))
"""
    include_files = mode == "full"
    return f"""
import json
from pathlib import Path

root = Path({root!r})
include_files = {include_files!r}
    payload = {{
        "campaign_root": str(root),
        "generated_at": None,
        "manifest": None,
        "summary_head": None,
    "summary_text": None,
    "phase_files": [],
    "files": [],
    "mode": {mode!r},
}}

manifest_path = root / "manifest.json"
summary_path = root / "summary.md"
if manifest_path.exists():
    manifest_text = manifest_path.read_text()
    payload["manifest"] = json.loads(manifest_text)
    if include_files:
        payload["files"].append({{
            "path": str(manifest_path),
            "relative_path": str(manifest_path.relative_to(root)),
            "type": "json",
            "text": manifest_text,
        }})
if summary_path.exists():
    summary_text = summary_path.read_text()
    payload["summary_text"] = summary_text
    payload["summary_head"] = summary_text.splitlines()[:40]
    if include_files:
        payload["files"].append({{
            "path": str(summary_path),
            "relative_path": str(summary_path.relative_to(root)),
            "type": "text",
            "text": summary_text,
        }})

for path in sorted(root.glob("*/*/*.json")):
    if path.name.endswith(".prescrub.json"):
        continue
    entry = {{
        "path": str(path),
        "kind": "status" if path.name.endswith(".status.json") else "phase",
    }}
    try:
        text = path.read_text()
        data = json.loads(text)
        for key in ("status", "elapsed_s", "ttft_s", "prefill_tok_s", "gen_tok_s", "error_type", "exit_code", "worker_pid", "heartbeat_ts"):
            if key in data:
                entry[key] = data.get(key)
        if include_files:
            payload["files"].append({{
                "path": str(path),
                "relative_path": str(path.relative_to(root)),
                "type": "json",
                "text": text,
            }})
    except Exception as exc:
        entry["error"] = str(exc)
    payload["phase_files"].append(entry)

if include_files:
    for path in sorted(root.glob("*/*/*")):
        if not path.is_file():
            continue
        if path.suffix not in {{".txt", ".md"}}:
            continue
        payload["files"].append({{
            "path": str(path),
            "relative_path": str(path.relative_to(root)),
            "type": "text",
            "text": path.read_text(),
        }})

print(json.dumps(payload))
"""


def main() -> int:
    args = parse_args()
    root = args.campaign_root
    output_path = Path(args.output)
    fallback_payload = {
        "campaign_root": root,
        "generated_at": utc_now_iso(),
        "manifest": None,
        "summary_head": None,
        "summary_text": None,
        "phase_files": [],
        "files": [],
    }
    remote_python = build_remote_python(root, args.mode)
    try:
        stdout = run_ssh(args, f"python3 - <<'PY'\n{remote_python}\nPY")
        payload = json.loads(stdout)
        payload["generated_at"] = utc_now_iso()
        atomic_write_json(output_path, payload)
    except subprocess.TimeoutExpired as exc:
        fallback_payload.update(
            {
                "status": "error",
                "error_type": "ssh_timeout",
                "error_message": str(exc),
                "ssh_timeout_s": args.ssh_timeout_s,
            }
        )
        atomic_write_json(output_path, fallback_payload)
        print(args.output)
        return 1
    except subprocess.CalledProcessError as exc:
        fallback_payload.update(
            {
                "status": "error",
                "error_type": "ssh_failed",
                "error_message": str(exc),
                "stdout_tail": tail_text(exc.stdout),
                "stderr_tail": tail_text(exc.stderr),
            }
        )
        atomic_write_json(output_path, fallback_payload)
        print(args.output)
        return 1
    if args.mirror_root:
        write_mirror_files(payload, Path(args.mirror_root))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
