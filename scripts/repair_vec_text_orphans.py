"""One-shot repair for files left without vec_text embeddings.

Background
----------
Before the migration fix to ``_migrate_vec_text_if_needed`` (branch (d)),
a text embedding model swap would:

  1. ``DROP TABLE vec_text`` (wiping every vector in it),
  2. delete only ``embedding_type='text_content'`` rows from
     ``embeddings``,
  3. reset only ``text_indexed=0`` on ``indexed_files``.

That left every ``whisper`` / ``tfidf_keywords`` embedding row pointing
at a vector that no longer existed. On the next boot
``cleanup_orphaned_embeddings`` mass-deleted those orphan metadata rows,
but ``whisper_indexed`` / ``tfidf_keywords_indexed`` stayed ``True`` —
so the indexer skipped re-embedding those files forever, and every
detailed-summary citation computed against those files came back with
``top_score=0`` / ``has_citation=False`` (the Verify toggle in the
detailed summary section then never appears).

This script repairs the data left behind by that old bug. It is safe
to run repeatedly; idempotent under the fixed migration. After running,
trigger an indexer scan and (optionally) re-run
``backfill_detailed_citations.py --force`` to recompute citations.

Usage (inside the intelligence container)::

    docker compose exec intelligence \
        python -m scripts.repair_vec_text_orphans --dry-run
    docker compose exec intelligence \
        python -m scripts.repair_vec_text_orphans

The script does NOT touch ``vec_clip`` / image embeddings.
"""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import text as sql_text

logger = logging.getLogger("repair_vec_text_orphans")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# Embedding types stored in ``vec_text`` whose presence we use to detect
# whether a file's text-side index is actually populated. Mirrors the
# purge list in ``_migrate_vec_text_if_needed`` branch (d).
_VEC_TEXT_EMBEDDING_TYPES = (
    "text_content",
    "whisper",
    "tfidf_keywords",
    "vision_description",
    "metadata",
)


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change, do not write.",
    )
    args = parser.parse_args(argv)

    from app.database import get_search_db, init_search_db

    init_search_db()

    with get_search_db() as session:
        # Files whose *_indexed flag claims completion but no
        # corresponding embedding row exists. We check whisper /
        # tfidf_keywords explicitly because text_content is already
        # covered by the existing reindex_text_content.py path.
        whisper_stale = session.execute(
            sql_text(
                "SELECT file_id FROM indexed_files i "
                "WHERE i.whisper_indexed = 1 "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM embeddings e "
                "  WHERE e.file_id = i.file_id "
                "  AND e.embedding_type = 'whisper'"
                ")"
            )
        ).fetchall()
        tfidf_stale = session.execute(
            sql_text(
                "SELECT file_id FROM indexed_files i "
                "WHERE i.tfidf_keywords_indexed = 1 "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM embeddings e "
                "  WHERE e.file_id = i.file_id "
                "  AND e.embedding_type = 'tfidf_keywords'"
                ")"
            )
        ).fetchall()

        whisper_ids = [r[0] for r in whisper_stale]
        tfidf_ids = [r[0] for r in tfidf_stale]
        affected = set(whisper_ids) | set(tfidf_ids)

        logger.info(
            "Stale flags found: whisper_indexed=%d, "
            "tfidf_keywords_indexed=%d, unique files=%d.",
            len(whisper_ids),
            len(tfidf_ids),
            len(affected),
        )

        # detailed_summary_citations rows computed while embeddings were
        # missing are all top_score=0 / has_citation=False — useless to
        # the UI. Recompute after the indexer re-embeds.
        stale_citations = 0
        if affected:
            stale_citations = session.execute(
                sql_text(
                    "SELECT COUNT(*) FROM detailed_summary_citations "
                    "WHERE file_id IN :ids"
                ).bindparams(
                    __import__("sqlalchemy").bindparam("ids", expanding=True)
                ),
                {"ids": list(affected)},
            ).scalar_one()
        logger.info(
            "Citation rows on affected files (will be cleared so the "
            "next backfill recomputes them): %d",
            stale_citations,
        )

        if args.dry_run:
            logger.info("Dry run: no changes written.")
            return 0

        if not affected:
            logger.info("Nothing to repair.")
            return 0

        # Reset stale flags so the indexer re-runs whisper / VTT /
        # TF-IDF paths on the next scan.
        if whisper_ids:
            session.execute(
                sql_text(
                    "UPDATE indexed_files SET whisper_indexed = 0 "
                    "WHERE file_id IN :ids"
                ).bindparams(
                    __import__("sqlalchemy").bindparam("ids", expanding=True)
                ),
                {"ids": whisper_ids},
            )
            logger.info(
                "Reset whisper_indexed=False for %d files.",
                len(whisper_ids),
            )
        if tfidf_ids:
            session.execute(
                sql_text(
                    "UPDATE indexed_files SET tfidf_keywords_indexed = 0 "
                    "WHERE file_id IN :ids"
                ).bindparams(
                    __import__("sqlalchemy").bindparam("ids", expanding=True)
                ),
                {"ids": tfidf_ids},
            )
            logger.info(
                "Reset tfidf_keywords_indexed=False for %d files.",
                len(tfidf_ids),
            )

        # Drop stale (zero-score) citation rows for affected files. The
        # detailed_summary itself is preserved; re-run
        # backfill_detailed_citations after the scan to repopulate.
        deleted_cites = session.execute(
            sql_text(
                "DELETE FROM detailed_summary_citations "
                "WHERE file_id IN :ids"
            ).bindparams(
                __import__("sqlalchemy").bindparam("ids", expanding=True)
            ),
            {"ids": list(affected)},
        ).rowcount
        logger.info(
            "Deleted %d stale citation rows on affected files.",
            deleted_cites,
        )

        session.commit()

    logger.info(
        "Done. Trigger an indexer scan to re-embed, then run "
        "backfill_detailed_citations --force (or regenerate the "
        "detailed summary from the UI) to repopulate citations."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
