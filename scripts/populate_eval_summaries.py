"""One-off eval helper: populate file_summaries on the eval snapshot.

The committed eval snapshot (``evals/test-drive/snapshot/search.db``)
does not have summaries for most files. Phase 3's hierarchical
retrieval depends on per-file ``long_summary`` text — without it,
``coarse_retrieve`` falls back to title+filename+description+tags
metadata embeddings and the multi-query clue generator collapses to
the legacy single-keyword path. Phase 4 baseline measurement showed
zero observable Phase 3 effect on the as-shipped snapshot for this
exact reason (see hako ``WnEEUSY8EY5miq-_pDJ-n``).

This script bypasses the production worker queue and calls the
short/long summary generation pipeline directly against an arbitrary
search.db path. It is **not** intended for production use — the path
override + manual loop is for one-shot eval-data preparation.

Run inside the intelligence container:

    INTELLIGENCE_SEARCH_DB_PATH=/eval-data/test-drive/snapshot/search.db \\
        python scripts/populate_eval_summaries.py --drive eval-drive

The snapshot is git-tracked, so revert with
``git checkout evals/test-drive/snapshot/`` after the eval run.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("populate_eval_summaries")


def _setup(snapshot: Path) -> None:
    """Point app.database at the snapshot before any module imports it."""
    if not snapshot.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot}")
    os.environ["INTELLIGENCE_SEARCH_DB_PATH"] = str(snapshot)


async def _run(drive: str, force: bool) -> int:
    from sqlalchemy import text as sql_text

    from app.config import settings
    from app.database import get_search_db, init_search_db
    from app.llm import create_llm_client
    from app.workers.summaries import (
        _build_context,
        _classify_file_type,
        _get_indexed_file,
        SummariesWorker,
    )

    init_search_db()

    llm = create_llm_client(settings.llm)
    if not llm.enabled:
        logger.error(
            "LLM client is disabled (provider=%s) — set LLM_API_KEY / "
            "ensure ollama is reachable before running this script.",
            settings.llm.provider,
        )
        return 1

    # Inject the LLM client into the dependencies module so any worker
    # code path that calls ``get_llm_client()`` finds it. Mirrors the
    # eval runner's _init_llm.
    from app import dependencies
    dependencies._llm_client = llm

    worker = SummariesWorker(llm)

    # Pull the eval-drive's active file_ids. Skip files that already
    # have a summary unless --force is given.
    with get_search_db() as session:
        rows = session.execute(
            sql_text(
                "SELECT i.file_id, i.filename, s.file_id IS NOT NULL "
                "FROM indexed_files i "
                "LEFT JOIN file_summaries s ON s.file_id = i.file_id "
                "WHERE i.drive = :drive AND i.active = 1 "
                "ORDER BY i.filename"
            ),
            {"drive": drive},
        ).fetchall()

    targets = []
    for file_id, filename, has_summary in rows:
        if has_summary and not force:
            logger.info("skip %s (%s) — already has a summary", file_id, filename)
            continue
        targets.append((file_id, filename))

    if not targets:
        logger.info("Nothing to do — every active file in %r has a summary", drive)
        return 0

    logger.info("Generating summaries for %d files in drive %r", len(targets), drive)

    generated = 0
    skipped = 0
    failed = 0
    for file_id, filename in targets:
        indexed = _get_indexed_file(file_id)
        if indexed is None:
            logger.warning("skip %s (%s) — no IndexedFile row", file_id, filename)
            skipped += 1
            continue

        context_type = _classify_file_type(
            indexed["file_type"], indexed.get("mime_type")
        )
        if context_type is None:
            logger.warning(
                "skip %s (%s) — file_type=%s, mime=%s not supported by summaries worker",
                file_id, filename, indexed["file_type"], indexed.get("mime_type"),
            )
            skipped += 1
            continue

        raw = _build_context(indexed, context_type)
        if not raw:
            logger.warning(
                "skip %s (%s) — no transcript/text content available",
                file_id, filename,
            )
            skipped += 1
            continue

        try:
            await worker._generate_short_long(
                file_id, indexed, context_type, raw
            )
            generated += 1
        except Exception as e:  # noqa: BLE001 — script-level catch-all
            logger.error("FAIL %s (%s): %s", file_id, filename, e)
            failed += 1

    logger.info(
        "Done: generated=%d skipped=%d failed=%d", generated, skipped, failed,
    )
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", default="eval-drive")
    parser.add_argument(
        "--snapshot",
        default="/eval-data/test-drive/snapshot/search.db",
        help="Path to the snapshot search.db inside the container.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even when a summary row already exists.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _setup(Path(args.snapshot))
    return asyncio.run(_run(args.drive, args.force))


if __name__ == "__main__":
    sys.exit(main())
