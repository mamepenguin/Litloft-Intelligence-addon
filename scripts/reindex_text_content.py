"""Purge existing text_content embeddings and reset text_indexed flag.

After running this, the next indexer scan will re-index all text files
with the current (possibly updated) chunk size / overlap settings.
``detailed_summary_citations`` rows are also cleared because their
``citation_chunk_ids`` reference the old chunk indices and become stale
once the chunks are regenerated. Re-run
``backfill_detailed_citations.py --force`` after re-indexing completes
to recompute citations against the new chunks.

Usage (inside the intelligence container)::

    docker compose exec intelligence python -m scripts.reindex_text_content
    docker compose exec intelligence python -m scripts.reindex_text_content --dry-run

This script does NOT touch whisper / clip / metadata embeddings.
"""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import text as sql_text

logger = logging.getLogger("reindex_text_content")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts without deleting.",
    )
    args = parser.parse_args(argv)

    from app.database import (
        delete_fts_text_content,
        get_search_db,
        init_search_db,
    )
    from app.models import Embedding, IndexedFile

    init_search_db()

    with get_search_db() as session:
        emb_count = (
            session.query(Embedding)
            .filter_by(embedding_type="text_content")
            .count()
        )
        indexed_file_ids = [
            fid
            for (fid,) in session.query(IndexedFile.file_id)
            .filter(IndexedFile.text_indexed.is_(True))
            .all()
        ]
        citations_count = session.execute(
            sql_text("SELECT COUNT(*) FROM detailed_summary_citations")
        ).scalar_one()

        logger.info(
            "Found %d text_content embeddings, %d text_indexed files, "
            "%d citation rows.",
            emb_count,
            len(indexed_file_ids),
            citations_count,
        )

        if args.dry_run:
            logger.info("Dry run: no changes written.")
            return 0

        # 1. Drop vec_text rows linked to text_content embeddings, then
        #    delete the embedding rows themselves. Filter by vector_table
        #    to be safe — text_content always lives in vec_text.
        session.execute(
            sql_text(
                "DELETE FROM vec_text WHERE embedding_id IN ("
                "  SELECT id FROM embeddings WHERE embedding_type = 'text_content'"
                ")"
            )
        )
        deleted_emb = (
            session.query(Embedding)
            .filter_by(embedding_type="text_content")
            .delete(synchronize_session=False)
        )
        logger.info("Deleted %d text_content embeddings.", deleted_emb)

        # 2. Clear fts_text_content for each indexed file.
        for fid in indexed_file_ids:
            delete_fts_text_content(session, fid)
        logger.info(
            "Cleared fts_text_content for %d files.",
            len(indexed_file_ids),
        )

        # 3. Reset text_indexed flag so the next scan re-indexes.
        reset_count = (
            session.query(IndexedFile)
            .filter(IndexedFile.text_indexed.is_(True))
            .update({"text_indexed": False}, synchronize_session=False)
        )
        logger.info(
            "Reset text_indexed=False for %d files.", reset_count
        )

        # 4. Clear detailed_summary_citations — chunk_ids reference old
        #    chunk indices which are about to be regenerated.
        deleted_cites = session.execute(
            sql_text("DELETE FROM detailed_summary_citations")
        ).rowcount
        logger.info(
            "Deleted %d detailed_summary_citations rows.",
            deleted_cites,
        )

        session.commit()

    logger.info(
        "Done. Trigger a scan to re-index text content, then run "
        "backfill_detailed_citations --force to recompute citations."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
