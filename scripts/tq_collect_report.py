#!/usr/bin/env python3
"""Collect campaign JSON artifacts into a manifest and summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from tq_harness_lib import atomic_write_json, atomic_write_text, collect_campaign_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    manifest, summary = collect_campaign_summary(campaign_root)

    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else campaign_root / "manifest.json"
    summary_path = Path(args.summary_path).resolve() if args.summary_path else campaign_root / "summary.md"

    atomic_write_json(manifest_path, manifest)
    atomic_write_text(summary_path, summary)
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
