"""
scripts/run_pipeline.py
────────────────────────
CLI entry point for the compliance pipeline.

Examples:
  # Audit a YouTube video
  python scripts/run_pipeline.py --source "https://www.youtube.com/watch?v=XXXX"

  # Audit a local video file
  python scripts/run_pipeline.py --source ./data/raw_videos/my_video.mp4

  # Audit raw text directly
  python scripts/run_pipeline.py --type text --text "We collect personal data without consent."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.compliance.schemas import AuditReport
from app.orchestration.graph import run_pipeline
from app.telemetry.logging_config import setup_logging
from app.telemetry.tracing import setup_tracing
from app.utils.config import get_config

cfg = get_config()
setup_logging(cfg.telemetry.log_level)
setup_tracing(cfg.telemetry.langsmith_project, cfg.telemetry.enable_tracing)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the compliance auditing pipeline")
    parser.add_argument("--source", type=str, default="", help="YouTube URL or local file path")
    parser.add_argument("--type", type=str, default="video", choices=["video", "text", "image"],
                        help="Input type")
    parser.add_argument("--text", type=str, default="", help="Raw text input (for --type text)")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output file path")
    args = parser.parse_args()

    if not args.source and not args.text:
        parser.error("Provide --source (URL / file path) or --text with --type text")

    source = args.source or "cli_text_input"
    print(f"\n🔍  Starting compliance audit for: {source}\n")

    state = run_pipeline(source=source, input_type=args.type, raw_text=args.text)

    if state.get("error"):
        print(f"\n❌  Pipeline error: {state['error']}")
        sys.exit(1)

    report_data = state.get("audit_report")
    if not report_data:
        print("\n⚠️  No audit report was generated.")
        sys.exit(1)

    report = AuditReport(**report_data) if isinstance(report_data, dict) else report_data

    # ── Print report ──────────────────────────────────────────────────────────
    status_icon = "✅" if report.is_compliant else "❌"
    print(f"\n{'='*60}")
    print(f"  Compliance Report – {report.source}")
    print(f"{'='*60}")
    print(f"  Score     : {report.compliance_score:.1%}  {status_icon}")
    print(f"  Compliant : {report.is_compliant}")
    print(f"\n  Summary:\n{report.summary}")
    print(f"\n  Rule Results:")
    for r in report.results:
        icon = {"PASSES": "✅", "FAILS": "❌", "UNCLEAR": "⚠️"}.get(r.status, "?")
        print(f"    {icon}  [{r.severity.upper()}] {r.rule_id}: {r.reason}")
    print(f"{'='*60}\n")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(report.model_dump(), indent=2, default=str),
            encoding="utf-8",
        )
        print(f"📄  Report saved to {out_path}")


if __name__ == "__main__":
    main()
