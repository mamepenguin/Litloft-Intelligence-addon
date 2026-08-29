"""Measure the Related Passages hit rate at several ``min_z`` values.

Background
----------
The section pairs a passage of the file being read with a passage of a
verified file, and admits a pair only when it stands out from that
request's own score distribution (``related_passages.min_z``). No fixed
cosine floor works: on a measured drive, *unrelated* passages sit at a
median of 0.770 and a p99 of 0.852, while a genuinely related pair
scores 0.928 — the bands touch.

Where the bar belongs therefore depends on the library and the
embedding model, so it has to be measured rather than assumed. Re-run
this after a big change to the library, and after any change to
``models.text_embedding``.

**A higher hit rate is not the goal.** Read the example pairs and ask
whether they are ones you would have wanted to see. On the drive this
feature was developed against, dropping the bar from 5.0 to 4.0 took
the hit rate from 48% to 78% and every pair it added was noise.

Usage (inside the intelligence container)::

    docker compose exec intelligence \
        python -m scripts.tune_related_passages --drive main

    docker compose exec intelligence \
        python -m scripts.tune_related_passages --drive main \
            --sample 80 --z 5.0 --z 4.5

Read-only: it opens the search DB, scores in memory, and writes
nothing. Applying a new value means editing ``related_passages.min_z``
in ``search-config.yml`` and restarting the container — settings are
read once at startup, so no rebuild is needed.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import random
import sys
import time

from sqlalchemy import text as sql_text

logger = logging.getLogger("tune_related_passages")

#: Sampling is seeded so two runs against the same library are
#: comparable; a threshold that looks better only because the sample
#: moved is not evidence.
_SEED = 1


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", required=True, help="Drive to sample.")
    parser.add_argument(
        "--sample", type=int, default=40, help="Files to sample (default 40)."
    )
    parser.add_argument(
        "--z",
        type=float,
        action="append",
        dest="thresholds",
        help="A min_z to try; repeatable. Default: 5.0, 4.5, 4.0.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Top pairs to print per threshold (default 3).",
    )
    args = parser.parse_args(argv)
    thresholds = args.thresholds or [5.0, 4.5, 4.0]

    from app.database import get_search_engine, init_search_db

    init_search_db()

    from app import passages

    with get_search_engine().connect() as conn:
        files = [
            row[0]
            for row in conn.execute(
                sql_text(
                    "SELECT DISTINCT e.file_id FROM embeddings e "
                    "JOIN indexed_files f ON f.file_id = e.file_id "
                    "WHERE f.drive = :d AND f.active = 1 "
                    "  AND e.embedding_type IN ('text_content','whisper')"
                ),
                {"d": args.drive},
            ).fetchall()
        ]
        pending = conn.execute(
            sql_text(
                "SELECT COUNT(*) FROM embeddings "
                "WHERE embedding_type = 'text_content' "
                "  AND chunk_index IS NULL"
            )
        ).scalar()

    logger.info("drive=%s  files with passages: %d", args.drive, len(files))
    if not files:
        logger.error("Nothing indexed in that drive.")
        return 1
    if pending:
        # Without the chunk key a document passage cannot be resolved to
        # its full text, so it is dropped from the results. Measuring
        # now understates every threshold equally, but it understates.
        logger.warning(
            "%d document chunks still have no chunk_index — the re-index "
            "has not finished. Document passages cannot be resolved yet, "
            "so these numbers are a floor, not a reading.",
            pending,
        )

    random.seed(_SEED)
    sample = random.sample(files, min(args.sample, len(files)))
    base = passages.settings.related_passages
    original = passages.settings

    try:
        for threshold in thresholds:
            passages.settings = type(
                "TunedSettings",
                (),
                {"related_passages": dataclasses.replace(base, min_z=threshold)},
            )()

            hits = pairs_total = 0
            examples: list = []
            started = time.time()
            for file_id in sample:
                source, candidates = passages._source_and_candidates(
                    file_id, args.drive
                )
                if not source or not candidates:
                    continue
                found = passages._build_pairs(
                    source, candidates[: base.candidate_files], 5
                )
                if not found:
                    continue
                hits += 1
                pairs_total += len(found)
                if len(examples) < args.examples:
                    examples.append(found[0])

            logger.info(
                "\nmin_z=%.1f: %d/%d files (%.0f%%), %d pairs, %.2fs per file",
                threshold,
                hits,
                len(sample),
                100 * hits / len(sample),
                pairs_total,
                (time.time() - started) / len(sample),
            )
            for pair in examples:
                logger.info("    %.3f  -> %s", pair.score, pair.other_filename)
                logger.info("      mine : %r", pair.text[:80])
                logger.info("      their: %r", pair.other_text[:80])
    finally:
        passages.settings = original

    logger.info(
        "\nJudge by the pairs, not the hit rate. To apply a value, set "
        "related_passages.min_z in search-config.yml and restart the "
        "container."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
