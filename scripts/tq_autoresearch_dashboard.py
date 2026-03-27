#!/usr/bin/env python3
"""Render a single-file HTML dashboard for the TurboQuant autoresearch control plane."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from tq_harness_lib import atomic_write_text, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-json", default="test-output/autoresearch-driver/status.json")
    parser.add_argument("--watch-report", default="test-output/autoresearch-driver/watch-report.json")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def summarize_stage_rows(driver_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not driver_report:
        return []
    rows: list[dict[str, Any]] = []
    for stage in driver_report.get("stages", []):
        rows.append(
            {
                "stage_name": stage.get("stage_name"),
                "context_len": stage.get("context_len"),
                "cases": stage.get("cases"),
                "phases": stage.get("phases"),
                "status": stage.get("status"),
                "probe_campaign_root": stage.get("probe_campaign_root"),
                "recovery_campaign_root": stage.get("recovery_campaign_root"),
            }
        )
    return rows


def build_dashboard_payload(status_payload: dict[str, Any] | None, watch_payload: dict[str, Any] | None) -> dict[str, Any]:
    status_payload = status_payload or {}
    watch_payload = watch_payload or {}
    latest_watch = (watch_payload.get("history") or [{}])[-1]
    driver_report = status_payload.get("driver_report")
    return {
        "generated_at": utc_now_iso(),
        "blocker_code": status_payload.get("blocker_code"),
        "blocker_summary": status_payload.get("blocker_summary"),
        "campaign_plan": status_payload.get("campaign_plan"),
        "latest_probe": status_payload.get("latest_probe") or {},
        "next_command": status_payload.get("next_command"),
        "campaign_failed": status_payload.get("campaign_failed") or [],
        "campaign_missing": status_payload.get("campaign_missing") or [],
        "driver_dry_run": driver_report.get("dry_run") if driver_report else None,
        "stage_rows": summarize_stage_rows(driver_report),
        "watch_iterations": watch_payload.get("iterations"),
        "watch_driver_triggered": watch_payload.get("driver_triggered"),
        "watch_latest_blocker": latest_watch.get("blocker_code"),
        "watch_latest_summary": latest_watch.get("blocker_summary"),
    }


def render_list(items: list[str]) -> str:
    if not items:
        return "<p class='muted'>None</p>"
    return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"


def render_stage_rows(stage_rows: list[dict[str, Any]]) -> str:
    if not stage_rows:
        return "<p class='muted'>No stage plan available.</p>"
    header = (
        "<tr><th>Stage</th><th>Context</th><th>Cases</th><th>Phases</th><th>Status</th></tr>"
    )
    rows = []
    for row in stage_rows:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('stage_name') or ''))}</td>"
            f"<td>{html.escape(str(row.get('context_len') or ''))}</td>"
            f"<td>{html.escape(str(row.get('cases') or ''))}</td>"
            f"<td>{html.escape(str(row.get('phases') or ''))}</td>"
            f"<td>{html.escape(str(row.get('status') or ''))}</td>"
            "</tr>"
        )
    return f"<table>{header}{''.join(rows)}</table>"


def render_html(payload: dict[str, Any]) -> str:
    failed_items = [
        f"{item.get('context_len')} / {item.get('case')} / {item.get('phase')} -> {item.get('status')} (exit={item.get('exit_code')})"
        for item in payload.get("campaign_failed", [])
    ]
    missing_items = [
        f"{item.get('context_len')} / {item.get('case')} / {item.get('phase')}"
        for item in payload.get("campaign_missing", [])
    ]
    next_command = html.escape(payload.get("next_command") or "No next command available")
    latest_probe = payload.get("latest_probe") or {}

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TurboQuant Autoresearch Dashboard</title>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #151c23;
      --panel-strong: #1b2530;
      --text: #edf2f7;
      --muted: #95a3b3;
      --accent: #6ee7b7;
      --warn: #f59e0b;
      --danger: #f87171;
      --border: #2a3642;
      --mono: "SFMono-Regular", "SF Mono", ui-monospace, Menlo, monospace;
      --sans: "IBM Plex Sans", "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top right, #163041 0%, var(--bg) 42%);
      color: var(--text);
      font-family: var(--sans);
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    h1, h2 {{ margin: 0 0 12px; line-height: 1.1; }}
    h1 {{ font-size: 34px; letter-spacing: -0.03em; }}
    h2 {{ font-size: 18px; }}
    p {{ margin: 0; }}
    .hero {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .card {{
      background: linear-gradient(180deg, var(--panel-strong), var(--panel));
      border: 1px solid var(--border);
      padding: 18px;
      border-radius: 14px;
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
    }}
    .badge {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      margin-bottom: 12px;
    }}
    .badge-danger {{ background: rgba(248, 113, 113, 0.14); color: var(--danger); }}
    .badge-ok {{ background: rgba(110, 231, 183, 0.14); color: var(--accent); }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .meta div {{
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid rgba(255, 255, 255, 0.04);
      border-radius: 10px;
    }}
    .label {{
      display: block;
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 5px;
    }}
    .value {{ font-size: 14px; line-height: 1.5; }}
    .muted {{ color: var(--muted); }}
    pre {{
      margin: 0;
      padding: 14px;
      overflow-x: auto;
      border-radius: 12px;
      background: #0b1116;
      border: 1px solid var(--border);
      color: #d7e3ef;
      font-size: 12px;
      line-height: 1.45;
      font-family: var(--mono);
      white-space: pre-wrap;
      word-break: break-word;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--border);
      padding: 10px 8px;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }}
    ul {{ margin: 0; padding-left: 18px; }}
    li + li {{ margin-top: 6px; }}
    @media (max-width: 900px) {{
      .hero, .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="card">
        <span class="badge {'badge-ok' if payload.get('blocker_code') == 'ready' else 'badge-danger'}">{html.escape(str(payload.get('blocker_code') or 'unknown'))}</span>
        <h1>TurboQuant Autoresearch Dashboard</h1>
        <p class="muted">{html.escape(str(payload.get('blocker_summary') or 'No summary available'))}</p>
        <div class="meta">
          <div><span class="label">Generated</span><span class="value">{html.escape(str(payload.get('generated_at') or ''))}</span></div>
          <div><span class="label">50k Plan</span><span class="value">{html.escape(str(payload.get('campaign_plan') or 'unknown'))}</span></div>
          <div><span class="label">Watch Iterations</span><span class="value">{html.escape(str(payload.get('watch_iterations') or 0))}</span></div>
          <div><span class="label">Driver Triggered</span><span class="value">{html.escape(str(payload.get('watch_driver_triggered')))}</span></div>
        </div>
      </div>
      <div class="card">
        <h2>Latest Probe</h2>
        <div class="meta">
          <div><span class="label">Status</span><span class="value">{html.escape(str(latest_probe.get('status') or 'unknown'))}</span></div>
          <div><span class="label">Error Type</span><span class="value">{html.escape(str(latest_probe.get('error_type') or 'none'))}</span></div>
        </div>
        <div style="margin-top:14px">
          <span class="label">stderr tail</span>
          <pre>{html.escape(str(latest_probe.get('stderr_tail') or ''))}</pre>
        </div>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Next Live Command</h2>
        <pre>{next_command}</pre>
      </div>
      <div class="card">
        <h2>Watch Gate</h2>
        <div class="meta">
          <div><span class="label">Latest Blocker</span><span class="value">{html.escape(str(payload.get('watch_latest_blocker') or 'unknown'))}</span></div>
          <div><span class="label">Latest Summary</span><span class="value">{html.escape(str(payload.get('watch_latest_summary') or ''))}</span></div>
          <div><span class="label">Driver Dry Run</span><span class="value">{html.escape(str(payload.get('driver_dry_run')))}</span></div>
        </div>
      </div>
      <div class="card">
        <h2>Failed Phases</h2>
        {render_list(failed_items)}
      </div>
      <div class="card">
        <h2>Missing Phases</h2>
        {render_list(missing_items)}
      </div>
      <div class="card" style="grid-column: 1 / -1;">
        <h2>Stage Plan</h2>
        {render_stage_rows(payload.get('stage_rows') or [])}
      </div>
    </section>
  </div>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    status_payload = load_json(Path(args.status_json))
    watch_payload = load_json(Path(args.watch_report))
    payload = build_dashboard_payload(status_payload, watch_payload)
    atomic_write_text(Path(args.output), render_html(payload))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
