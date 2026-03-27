#!/usr/bin/env python3
"""Local contract tests for the TurboQuant telemetry harness."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

import sys

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from tq_harness_lib import (
    atomic_write_json,
    build_prompt_bundle_from_tokenizer,
    collect_campaign_summary,
    probe_gpu_metrics,
    should_skip_phase,
)
from tq_remote_snapshot import build_remote_python, run_ssh, write_mirror_files
from tq_autoresearch_driver import build_resume_command, build_stage_specs
from tq_autoresearch_dashboard import build_dashboard_payload, render_html
from tq_autoresearch_status import build_status_payload, render_markdown
from tq_autoresearch_watch import build_dashboard_command, build_driver_command, build_status_command, should_trigger_driver
from tq_resume_recovery import build_remote_launch_command


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(chr(token) for token in token_ids)


class HarnessTests(unittest.TestCase):
    def test_prompt_bundle_deterministic(self):
        tokenizer = FakeTokenizer()
        first = build_prompt_bundle_from_tokenizer(tokenizer, context_len=128, seed=42)
        second = build_prompt_bundle_from_tokenizer(tokenizer, context_len=128, seed=42)
        self.assertEqual(first.prompt_hash, second.prompt_hash)
        self.assertEqual(first.prompt_tokens, second.prompt_tokens)
        self.assertEqual(first.text, second.text)

    def test_atomic_write_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.json"
            atomic_write_json(path, {"status": "ok", "value": 7})
            self.assertTrue(path.exists())
            self.assertEqual(json.loads(path.read_text()), {"status": "ok", "value": 7})
            leftovers = list(path.parent.glob("*.tmp"))
            self.assertEqual(leftovers, [])

    def test_collect_campaign_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            atomic_write_json(
                root / "campaign_config.json",
                {
                    "campaign_id": "demo",
                    "contexts": [30000],
                    "cases": ["baseline", "tq"],
                    "phases": ["init", "full"],
                },
            )
            atomic_write_json(
                root / "30000" / "prompt" / "prompt_meta.json",
                {"prompt_hash": "abc", "prompt_tokens": 30000},
            )
            atomic_write_json(
                root / "30000" / "baseline" / "init.json",
                {"status": "ok", "prompt_hash": "abc", "elapsed_s": 1.2},
            )
            atomic_write_json(
                root / "30000" / "baseline" / "full.json",
                {"status": "ok", "prompt_hash": "abc", "elapsed_s": 3.4, "sample_text": "hi"},
            )
            atomic_write_json(
                root / "30000" / "tq" / "init.json",
                {"status": "timeout", "prompt_hash": "abc", "elapsed_s": 4.5, "exit_code": 137},
            )

            manifest, summary = collect_campaign_summary(root)
            self.assertEqual(manifest["recommended_plan"], "B")
            self.assertEqual(len(manifest["missing"]), 1)
            self.assertIn("30000", summary)
            self.assertIn("timeout", summary)

    def test_collect_campaign_summary_complete_sub_200k_is_plan_a(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            atomic_write_json(
                root / "campaign_config.json",
                {
                    "campaign_id": "demo-complete",
                    "contexts": [30000],
                    "cases": ["baseline", "tq"],
                    "phases": ["init", "ttft", "full"],
                },
            )
            atomic_write_json(
                root / "30000" / "prompt" / "prompt_meta.json",
                {"prompt_hash": "abc", "prompt_tokens": 30000},
            )
            for case in ("baseline", "tq"):
                for phase in ("init", "ttft", "full"):
                    atomic_write_json(
                        root / "30000" / case / f"{phase}.json",
                        {
                            "status": "ok",
                            "prompt_hash": "abc",
                            "elapsed_s": 1.0,
                            "sample_text": "done" if phase == "full" else "",
                        },
                    )

            manifest, _summary = collect_campaign_summary(root)
            self.assertEqual(manifest["recommended_plan"], "A")

    def test_should_skip_phase(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "full.json"
            atomic_write_json(path, {"status": "ok"})
            self.assertTrue(should_skip_phase(path, skip_existing=False, force=False))
            self.assertTrue(should_skip_phase(path, skip_existing=True, force=False))
            self.assertFalse(should_skip_phase(path, skip_existing=False, force=True))

    def test_should_not_skip_non_ok_phase_without_skip_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "full.json"
            atomic_write_json(path, {"status": "error"})
            self.assertFalse(should_skip_phase(path, skip_existing=False, force=False))
            self.assertTrue(should_skip_phase(path, skip_existing=True, force=False))

    def test_remote_snapshot_ssh_wrapper(self):
        with unittest.mock.patch("subprocess.run") as mocked:
            mocked.return_value = SimpleNamespace(stdout='{"ok":true}\n')
            args = SimpleNamespace(
                host="example.com",
                port=2222,
                user="root",
                identity_file="/tmp/key",
                ssh_timeout_s=30,
            )
            out = run_ssh(args, "echo hi")
            self.assertEqual(out, '{"ok":true}\n')

    def test_remote_snapshot_writes_local_mirror(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mirror_root = Path(tmpdir) / "mirror"
            payload = {
                "manifest": {"campaign_id": "demo"},
                "summary_text": "# Demo\n",
                "files": [
                    {
                        "relative_path": "30000/baseline/init.json",
                        "type": "json",
                        "text": json.dumps({"status": "ok", "elapsed_s": 1.23}),
                    },
                    {
                        "relative_path": "30000/prompt/prompt.txt",
                        "type": "text",
                        "text": "prompt body",
                    },
                ],
            }
            write_mirror_files(payload, mirror_root)
            self.assertEqual(json.loads((mirror_root / "manifest.json").read_text()), {"campaign_id": "demo"})
            self.assertEqual((mirror_root / "summary.md").read_text(), "# Demo\n")
            self.assertEqual(
                json.loads((mirror_root / "30000" / "baseline" / "init.json").read_text()),
                {"status": "ok", "elapsed_s": 1.23},
            )
            self.assertEqual((mirror_root / "30000" / "prompt" / "prompt.txt").read_text(), "prompt body")

    def test_remote_snapshot_summary_mode_skips_file_payloads(self):
        remote_python = build_remote_python("/tmp/demo-campaign", "summary")
        self.assertIn("include_files = False", remote_python)
        self.assertIn('"mode": \'summary\'', remote_python)

    def test_remote_snapshot_ping_mode_is_minimal(self):
        remote_python = build_remote_python("/tmp/demo-campaign", "ping")
        self.assertIn('"mode": "ping"', remote_python)
        self.assertIn('"campaign_root_exists": root.exists()', remote_python)
        self.assertNotIn("manifest_path = root / \"manifest.json\"", remote_python)

    def test_resume_recovery_builds_plan_b_launch_command(self):
        args = SimpleNamespace(
            recovery_campaign_root="/remote/logs/smoke-50k-planb",
            remote_scripts_dir="/remote/scripts",
            remote_model_path="/remote/models/model",
            contexts="50000",
            cases="baseline,tq",
            phases="init,prefill_only,decode_only,full",
            prompt_seed=5090,
            max_output_tokens=24,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,
        )
        cmd = build_remote_launch_command(args)
        self.assertIn("nohup env CUDA_VISIBLE_DEVICES=0", cmd)
        self.assertIn("--phases init,prefill_only,decode_only,full", cmd)
        self.assertIn("--campaign-root /remote/logs/smoke-50k-planb", cmd)
        self.assertIn("> /remote/logs/smoke-50k-planb/launch.log 2>&1 < /dev/null & echo $!", cmd)

    def test_autoresearch_driver_builds_stage_specs(self):
        args = SimpleNamespace(
            contexts="50000,80000,120000,200000",
            base_remote_root="/remote/logs",
            local_report_dir="test-output/autoresearch-driver",
        )
        stages = build_stage_specs(args)
        self.assertEqual([stage["context_len"] for stage in stages], [50000, 80000, 120000, 200000])
        self.assertEqual(stages[0]["stage_name"], "smoke-50k-planb")
        self.assertEqual(stages[0]["probe_campaign_root"], "/remote/logs/smoke-50k-harness")
        self.assertEqual(stages[1]["probe_campaign_root"], "/remote/logs/smoke-50k-planb")
        self.assertEqual(stages[-1]["cases"], "tq")

    def test_autoresearch_driver_builds_resume_command(self):
        args = SimpleNamespace(
            dry_run=True,
            host="your-gpu-host.example.com",
            port=3003,
            user="root",
            identity_file="/tmp/key",
            ssh_timeout_s=20,
            probe_interval_s=30,
            max_wait_s=300,
            remote_scripts_dir="/remote/scripts",
            remote_model_path="/remote/models/model",
            prompt_seed=5090,
            max_output_tokens=24,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,
        )
        stage = {
            "context_len": 80000,
            "probe_campaign_root": "/remote/logs/smoke-50k-planb",
            "recovery_campaign_root": "/remote/logs/smoke-80k-planb",
            "cases": "baseline,tq",
            "phases": "init,prefill_only,decode_only,full",
            "local_report_path": "test-output/autoresearch-driver/smoke-80k-planb-resume-report.json",
        }
        cmd = build_resume_command(args, stage)
        self.assertIn("--contexts", cmd)
        self.assertIn("80000", cmd)
        self.assertIn("--recovery-campaign-root", cmd)
        self.assertIn("/remote/logs/smoke-80k-planb", cmd)
        self.assertEqual(cmd[-1], "--dry-run")

    def test_autoresearch_status_builds_blocked_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            remote = root / "remote-smoke-50k"
            driver_dir = root / "autoresearch-driver"
            remote.mkdir()
            driver_dir.mkdir()
            atomic_write_json(
                remote / "local-manifest.json",
                {
                    "recommended_plan": "B",
                    "failed": [{"context_len": 50000, "case": "baseline", "phase": "ttft", "status": "killed", "exit_code": -9}],
                    "missing": [{"context_len": 50000, "case": "tq", "phase": "init"}],
                },
            )
            atomic_write_json(
                remote / "ping-snapshot-latest.json",
                {
                    "status": "error",
                    "error_type": "ssh_failed",
                    "stderr_tail": "ssh: connect to host your-gpu-host.example.com port 3003: Operation timed out",
                },
            )
            atomic_write_json(
                remote / "resume-recovery-report.json",
                {
                    "status": "error",
                    "error_type": "ssh_failed",
                    "error_message": "connect timed out",
                },
            )
            atomic_write_json(
                driver_dir / "driver-report.json",
                {
                    "stages": [
                        {
                            "status": "planned",
                            "command_shell": "uv run python scripts/tq_resume_recovery.py --output report.json --dry-run",
                        }
                    ]
                },
            )
            payload = build_status_payload(remote, driver_dir / "driver-report.json")
            self.assertEqual(payload["blocker_code"], "ssh_outage")
            self.assertEqual(payload["campaign_plan"], "B")
            self.assertIn("uv run python", payload["next_command"])
            self.assertNotIn("--dry-run", payload["next_command"])

    def test_autoresearch_status_renders_markdown(self):
        payload = {
            "generated_at": "2026-03-27T00:00:00+00:00",
            "blocker_code": "ssh_outage",
            "blocker_summary": "Remote SSH is timing out on your-gpu-host.example.com:3003",
            "campaign_plan": "B",
            "latest_probe_name": "ping_latest",
            "latest_probe": {"status": "error", "error_type": "ssh_failed", "stderr_tail": "timeout"},
            "campaign_failed": [{"context_len": 50000, "case": "baseline", "phase": "ttft", "status": "killed", "exit_code": -9}],
            "campaign_missing": [{"context_len": 50000, "case": "tq", "phase": "init"}],
            "next_command": "uv run python scripts/tq_autoresearch_driver.py --output report.json",
        }
        md = render_markdown(payload)
        self.assertIn("# TurboQuant Autoresearch Status", md)
        self.assertIn("ssh_outage", md)
        self.assertIn("50000 / baseline / ttft", md)
        self.assertIn("```bash", md)

    def test_autoresearch_watch_builds_commands(self):
        args = SimpleNamespace(
            status_script="scripts/tq_autoresearch_status.py",
            status_json="test-output/autoresearch-driver/status.json",
            status_md="test-output/autoresearch-driver/status.md",
            dashboard_script="scripts/tq_autoresearch_dashboard.py",
            dashboard_output="test-output/autoresearch-driver/dashboard.html",
            watch_report="test-output/autoresearch-driver/watch-report.json",
            driver_script="scripts/tq_autoresearch_driver.py",
            driver_output="test-output/autoresearch-driver/driver-report.json",
            driver_dry_run=True,
        )
        status_cmd = build_status_command(args)
        dashboard_cmd = build_dashboard_command(args)
        driver_cmd = build_driver_command(args)
        self.assertEqual(status_cmd[1], "scripts/tq_autoresearch_status.py")
        self.assertIn("--output-json", status_cmd)
        self.assertEqual(dashboard_cmd[1], "scripts/tq_autoresearch_dashboard.py")
        self.assertIn("--watch-report", dashboard_cmd)
        self.assertEqual(driver_cmd[1], "scripts/tq_autoresearch_driver.py")
        self.assertEqual(driver_cmd[-1], "--dry-run")

    def test_autoresearch_watch_trigger_gate(self):
        self.assertFalse(should_trigger_driver(None))
        self.assertFalse(should_trigger_driver({"blocker_code": "ssh_outage"}))
        self.assertTrue(should_trigger_driver({"blocker_code": "ready"}))

    def test_autoresearch_dashboard_payload(self):
        status_payload = {
            "blocker_code": "ssh_outage",
            "blocker_summary": "Remote SSH is timing out on your-gpu-host.example.com:3003",
            "campaign_plan": "B",
            "latest_probe": {"status": "error", "error_type": "ssh_failed", "stderr_tail": "timeout"},
            "next_command": "uv run python scripts/tq_resume_recovery.py --output report.json",
            "campaign_failed": [{"context_len": 50000, "case": "baseline", "phase": "ttft", "status": "killed", "exit_code": -9}],
            "campaign_missing": [{"context_len": 50000, "case": "tq", "phase": "init"}],
            "driver_report": {
                "dry_run": True,
                "stages": [{"stage_name": "smoke-50k-planb", "context_len": 50000, "cases": "baseline,tq", "phases": "init,prefill_only", "status": "planned"}],
            },
        }
        watch_payload = {
            "iterations": 1,
            "driver_triggered": False,
            "history": [{"blocker_code": "ssh_outage", "blocker_summary": "Remote SSH is timing out on your-gpu-host.example.com:3003"}],
        }
        payload = build_dashboard_payload(status_payload, watch_payload)
        self.assertEqual(payload["blocker_code"], "ssh_outage")
        self.assertEqual(payload["watch_iterations"], 1)
        self.assertEqual(len(payload["stage_rows"]), 1)

    def test_autoresearch_dashboard_renders_html(self):
        payload = {
            "generated_at": "2026-03-27T00:00:00+00:00",
            "blocker_code": "ssh_outage",
            "blocker_summary": "Remote SSH is timing out on your-gpu-host.example.com:3003",
            "campaign_plan": "B",
            "latest_probe": {"status": "error", "error_type": "ssh_failed", "stderr_tail": "timeout"},
            "next_command": "uv run python scripts/tq_resume_recovery.py --output report.json",
            "campaign_failed": [{"context_len": 50000, "case": "baseline", "phase": "ttft", "status": "killed", "exit_code": -9}],
            "campaign_missing": [{"context_len": 50000, "case": "tq", "phase": "init"}],
            "driver_dry_run": True,
            "stage_rows": [{"stage_name": "smoke-50k-planb", "context_len": 50000, "cases": "baseline,tq", "phases": "init,prefill_only", "status": "planned"}],
            "watch_iterations": 1,
            "watch_driver_triggered": False,
            "watch_latest_blocker": "ssh_outage",
            "watch_latest_summary": "Remote SSH is timing out on your-gpu-host.example.com:3003",
        }
        html_text = render_html(payload)
        self.assertIn("<!DOCTYPE html>", html_text)
        self.assertIn("TurboQuant Autoresearch Dashboard", html_text)
        self.assertIn("ssh_outage", html_text)
        self.assertIn("smoke-50k-planb", html_text)

    def test_phase_runner_timeout_writes_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_dir = Path(tmpdir) / "prompt"
            atomic_write_json(
                prompt_dir / "prompt_meta.json",
                {"prompt_hash": "dry", "prompt_tokens": 12},
            )
            (prompt_dir / "prompt.txt").write_text("dry run prompt", encoding="utf-8")
            output = Path(tmpdir) / "phase.json"
            cmd = [
                sys.executable,
                str(SCRIPTS / "tq_phase_runner.py"),
                "--case",
                "baseline",
                "--phase",
                "full",
                "--context-len",
                "30000",
                "--output",
                str(output),
                "--timeout-s",
                "1",
                "--prompt-seed",
                "7",
                "--max-output-tokens",
                "24",
                "--model-path",
                "unused",
                "--prompt-dir",
                str(prompt_dir),
                "--dry-run",
                "--dry-run-sleep-s",
                "2",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["status"], "timeout")
            self.assertEqual(payload["error_type"], "phase_timeout")

    def test_phase_runner_prefill_only_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_dir = Path(tmpdir) / "prompt"
            atomic_write_json(
                prompt_dir / "prompt_meta.json",
                {"prompt_hash": "dry", "prompt_tokens": 12},
            )
            (prompt_dir / "prompt.txt").write_text("dry run prompt", encoding="utf-8")
            output = Path(tmpdir) / "phase.json"
            cmd = [
                sys.executable,
                str(SCRIPTS / "tq_phase_runner.py"),
                "--case",
                "baseline",
                "--phase",
                "prefill_only",
                "--context-len",
                "30000",
                "--output",
                str(output),
                "--timeout-s",
                "30",
                "--prompt-seed",
                "7",
                "--max-output-tokens",
                "24",
                "--model-path",
                "unused",
                "--prompt-dir",
                str(prompt_dir),
                "--dry-run",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["status"], "ok")
            self.assertIsNotNone(payload["ttft_s"])
            self.assertIsNotNone(payload["prefill_tok_s"])

    def test_gpu_probe_payload_is_json_serializable(self):
        payload = probe_gpu_metrics()
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
