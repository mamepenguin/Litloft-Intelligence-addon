"""CLI entry point for the citation eval harness.

Mirrors ``python -m app.evals`` but wires the citation-specific
loader / runner / report modules. The snapshot DB is the same one
the Ask harness uses (``../evals/test-drive/snapshot/search.db``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CASES = "../evals/citations/cases/"
DEFAULT_SNAPSHOT = "../evals/citations/snapshot/search.db"
DEFAULT_REPORTS_DIR = "../evals/citations/reports"

logger = logging.getLogger("app.evals_citations")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.evals_citations",
        description=(
            "Score detailed_summary citations against curated ground "
            "truth. Reads the snapshot's stored detailed_summary text "
            "and runs app.citations.compute_citations against it."
        ),
    )
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--drive", default="動画")
    parser.add_argument("--filter", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline sidecar (.json) for delta comparison.",
    )
    return parser


def _setup_snapshot(snapshot: Path) -> None:
    if not snapshot.exists():
        raise FileNotFoundError(f"Snapshot DB not found: {snapshot}")
    os.environ["INTELLIGENCE_SEARCH_DB_PATH"] = str(snapshot)


def _init_dbs() -> None:
    from app.database import init_search_db

    init_search_db()


def _run(args: argparse.Namespace) -> int:
    from app.evals_citations.loader import load_cases
    from app.evals_citations.report import (
        build_sidecar,
        compare_aggregates,
        render_markdown,
        write_report,
        write_sidecar,
    )
    from app.evals_citations.runner import aggregate, run_case

    cases_path = Path(args.cases).resolve()
    snapshot_path = Path(args.snapshot).resolve()

    _setup_snapshot(snapshot_path)
    _init_dbs()

    cases = load_cases(cases_path, filter_substr=args.filter)
    if not cases:
        logger.warning("No cases matched; nothing to score.")
        return 0
    logger.info("Loaded %d citation cases from %s", len(cases), cases_path)

    reports = [run_case(c, drive=args.drive) for c in cases]
    agg = aggregate(reports)

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = (Path(DEFAULT_REPORTS_DIR) / f"{ts}.md").resolve()

    markdown = render_markdown(reports, agg, label=args.label)
    sidecar = build_sidecar(reports, agg, label=args.label)

    if args.baseline:
        try:
            baseline = json.loads(
                Path(args.baseline).resolve().read_text(encoding="utf-8")
            )
            markdown += "\n" + compare_aggregates(
                baseline, sidecar, args.epsilon
            )
        except Exception as e:  # noqa: BLE001 — eval-time best effort
            logger.warning("Baseline comparison skipped: %s", e)
            markdown += f"\n<!-- baseline comparison failed: {e} -->\n"

    write_report(out_path, markdown)
    write_sidecar(out_path, sidecar)
    print(f"Citation eval report: {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
