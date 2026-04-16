"""CLI entry point for the Ask eval harness.

Wires loader → resolver → stages → report. The runner re-points the
intelligence service at a frozen snapshot DB by setting
``INTELLIGENCE_SEARCH_DB_PATH`` *before* the database module
initialises its engine, so the eval reads the snapshot's vectors and
transcripts instead of whatever live state the container happens to
have.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Default snapshot lives next to the case yamls inside the addon repo.
DEFAULT_CASES = "../evals/cases/"
DEFAULT_SNAPSHOT = "../evals/test-drive/snapshot/search.db"
DEFAULT_REPORTS_DIR = "../evals/reports"

logger = logging.getLogger("app.evals")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.evals",
        description="Dev-time eval harness for the Ask (RAG) pipeline.",
    )
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--runs-stage1", type=int, default=1)
    parser.add_argument("--runs-stage2", type=int, default=1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--drive", default="eval-drive")
    parser.add_argument("--filter", default=None)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override RAG top_k (default: from search-config).")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline report (.md or .json sidecar). "
             "When set, the current report gains a 'Pair comparison' section.",
    )
    return parser


def _setup_snapshot(snapshot: Path) -> None:
    """Set INTELLIGENCE_SEARCH_DB_PATH and verify the file exists.

    Must run before any ``app.database`` initialisation so the engine
    binds to the snapshot file. Inside ``python -m app.evals`` the
    only entry point that triggers init is our own ``_init_dbs``
    below — no production startup code path is on the import graph.
    """
    if not snapshot.exists():
        raise FileNotFoundError(f"Snapshot DB not found: {snapshot}")
    os.environ["INTELLIGENCE_SEARCH_DB_PATH"] = str(snapshot)
    logger.info("Eval snapshot DB: %s", snapshot)


def _init_dbs() -> None:
    """Initialise the search DB engine pointed at the snapshot.

    HomeVault DB is only needed for live indexing, not for retrieval —
    we skip ``init_homevault_db`` so the runner works against a snapshot
    even when the host DB file is absent or stale.
    """
    from app.database import init_search_db

    init_search_db()


def _init_llm() -> None:
    """Bring up the dependency container so RAG calls find the LLM client."""
    from app.config import settings
    from app import dependencies
    from app.llm import create_llm_client

    # No public setter exists in the production code path (clients are
    # bound during the FastAPI lifespan). Mutating the module attribute
    # is the established pattern in this codebase for dev-time tooling
    # — the eval runner is the only non-router caller of get_llm_client.
    dependencies._llm_client = create_llm_client(settings.llm)


async def _run(args: argparse.Namespace) -> int:
    from app.config import settings
    from app.evals.compare import compare_sidecars, render_comparison_md
    from app.evals.loader import load_cases
    from app.evals.report import (
        ReportMeta,
        build_sidecar,
        render,
        write_json_sidecar,
        write_report,
    )
    from app.evals.resolver import (
        build_path_to_file_id,
        hash_file_sha256,
        resolve_case,
        short_git_commit,
    )
    from app.evals.stages import run_case

    cases_path = Path(args.cases).resolve()
    snapshot_path = Path(args.snapshot).resolve()

    _setup_snapshot(snapshot_path)
    _init_dbs()
    _init_llm()

    cases = load_cases(cases_path, filter_substr=args.filter)
    logger.info("Loaded %d cases from %s", len(cases), cases_path)

    path_to_id = build_path_to_file_id(snapshot_path, args.drive)
    logger.info("Snapshot has %d files for drive %r", len(path_to_id), args.drive)

    top_k = args.top_k if args.top_k is not None else settings.rag.top_k

    case_reports = []
    for case in cases:
        resolved = resolve_case(case, path_to_id)
        report = await run_case(
            case=case,
            resolved=resolved,
            drive=args.drive,
            runs_stage3=args.runs,
            epsilon=args.epsilon,
            top_k=top_k,
        )
        case_reports.append(report)

    # Snapshot manifest
    manifest_path = snapshot_path.parent / "manifest.json"
    indexed_with: dict[str, str] = {}
    if manifest_path.exists():
        try:
            mf = json.loads(manifest_path.read_text(encoding="utf-8"))
            iw = mf.get("indexed_with") or {}
            indexed_with = {str(k): str(v) for k, v in iw.items()}
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to read manifest.json: %s", e)

    from app.evals.text_match import blocklist_sha256
    blocklists = {
        "question_words.txt": blocklist_sha256("question_words"),
        "file_type_words.txt": blocklist_sha256("file_type_words"),
    }

    meta = ReportMeta(
        label=args.label,
        git_commit=short_git_commit(),
        llm_model=settings.llm.model,
        llm_base_url=settings.llm.base_url,
        rag_top_k=top_k,
        rag_max_tokens=settings.rag.max_tokens,
        snapshot_path=str(snapshot_path),
        snapshot_sha256=hash_file_sha256(snapshot_path),
        indexed_with=indexed_with,
        runs_stage1=args.runs_stage1,
        runs_stage2=args.runs_stage2,
        runs_stage3=args.runs,
        epsilon=args.epsilon,
        drive=args.drive,
        blocklists=blocklists,
    )

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = (Path(DEFAULT_REPORTS_DIR) / f"{ts}.md").resolve()

    content = render(case_reports, meta)
    sidecar = build_sidecar(case_reports, meta)

    # --baseline: append pair-comparison section before writing the md.
    if args.baseline:
        try:
            comp = compare_sidecars(
                Path(args.baseline).resolve(), sidecar, args.epsilon
            )
            content = content + "\n" + render_comparison_md(comp)
        except Exception as e:  # noqa: BLE001 — eval-time best effort
            logger.warning("Baseline comparison skipped: %s", e)
            content = content + f"\n<!-- baseline comparison failed: {e} -->\n"

    write_report(out_path, content)
    write_json_sidecar(out_path, sidecar)
    logger.info("Wrote report → %s (+ .json sidecar)", out_path)
    print(f"Report written: {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
