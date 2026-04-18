"""Backfill ``detailed_summary_citations`` for pre-existing summaries.

The citations table is populated automatically for summaries generated
after the Phase 1 feature shipped, but existing ``detailed_summary``
rows — including ones edited by hand — have no citation data. Running
this script computes citations for every file that already has a
``detailed_summary`` body but lacks citations in the derived table.

Usage (inside the intelligence container)::

    docker compose exec intelligence python -m scripts.backfill_detailed_citations
    docker compose exec intelligence python -m scripts.backfill_detailed_citations --dry-run
    docker compose exec intelligence python -m scripts.backfill_detailed_citations --file-id abc123
    docker compose exec intelligence python -m scripts.backfill_detailed_citations --force

Flags:

* ``--dry-run``  — list candidate files but don't run embeddings / write rows.
* ``--file-id``  — only process this ID (useful for one-off repairs).
* ``--force``    — recompute even for files that already have citations.
                   By default existing citation rows are skipped so re-runs
                   are cheap.
"""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import text as sql_text


logger = logging.getLogger("backfill_detailed_citations")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _find_candidates(force: bool, only_file_id: str | None) -> list[str]:
    """Return file_ids needing citation backfill.

    When ``force`` is false we skip files that already have at least
    one citation row (cheapest common case: re-running the script after
    a partial failure).
    """
    from app.database import get_search_db

    with get_search_db() as session:
        if only_file_id:
            row = session.execute(
                sql_text(
                    "SELECT file_id FROM file_summaries "
                    "WHERE file_id = :fid "
                    "AND detailed_summary IS NOT NULL "
                    "AND length(detailed_summary) > 0"
                ),
                {"fid": only_file_id},
            ).fetchone()
            return [row[0]] if row else []

        base_sql = (
            "SELECT file_id FROM file_summaries "
            "WHERE detailed_summary IS NOT NULL "
            "AND length(detailed_summary) > 0"
        )
        if not force:
            base_sql += (
                " AND file_id NOT IN ("
                "  SELECT DISTINCT file_id FROM detailed_summary_citations"
                ")"
            )
        base_sql += " ORDER BY file_id"
        rows = session.execute(sql_text(base_sql)).fetchall()
        return [row[0] for row in rows]


def _load_summary(file_id: str) -> str | None:
    """Fetch the current ``detailed_summary`` body for ``file_id``."""
    from app.database import get_search_db

    with get_search_db() as session:
        row = session.execute(
            sql_text(
                "SELECT detailed_summary FROM file_summaries "
                "WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchone()
    if row is None:
        return None
    return row[0]


def run(*, dry_run: bool, only_file_id: str | None, force: bool) -> int:
    """Main entry point.

    Returns the count of files processed (including ``dry_run`` hits
    so the summary line reflects the full candidate set).
    """
    from app.citations import calculate_and_store

    candidates = _find_candidates(force=force, only_file_id=only_file_id)
    logger.info(
        "Found %d candidate files%s",
        len(candidates),
        " (dry-run)" if dry_run else "",
    )

    if dry_run:
        for fid in candidates:
            logger.info("would-process %s", fid)
        return len(candidates)

    processed = 0
    failed = 0
    for file_id in candidates:
        summary = _load_summary(file_id)
        if not summary:
            logger.warning("skipping %s: summary vanished mid-run", file_id)
            continue
        try:
            with_count, without_count = calculate_and_store(file_id, summary)
            logger.info(
                "ok %s — %d cited, %d warned",
                file_id, with_count, without_count,
            )
            processed += 1
        except Exception as e:  # noqa: BLE001 — keep the batch going
            logger.error("failed %s: %s", file_id, e)
            failed += 1

    logger.info(
        "backfill complete: processed=%d failed=%d total=%d",
        processed, failed, len(candidates),
    )
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill detailed_summary_citations for existing summaries."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List candidates without computing or writing anything.",
    )
    parser.add_argument(
        "--file-id",
        default=None,
        help="Only process this file_id.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even for files that already have citation rows.",
    )
    args = parser.parse_args()

    _configure_logging()

    # Initialise the DB engine in the same way main.py does so the
    # module-level ``_search_engine`` is populated before citations.py
    # tries to use it.
    from app.database import init_search_db

    init_search_db()

    run(
        dry_run=args.dry_run,
        only_file_id=args.file_id,
        force=args.force,
    )


if __name__ == "__main__":
    main()
