"""Database initialization for the semantic search service.

Manages two SQLite connections:
1. Search DB (read-write): sqlite-vec enabled, stores embeddings and index state
2. Litloft DB (read-only): reads file metadata for indexing
"""

import json
import logging
import secrets
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import bindparam, event, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.config import settings

Base = declarative_base()

# Global write lock: SQLite allows only one writer at a time.
# All vector table mutations (INSERT/DELETE on vec_text, vec_clip) and
# their accompanying ORM flushes MUST be wrapped with this lock to
# prevent "database is locked" errors from concurrent worker threads.
_write_lock = threading.Lock()

# Search DB engine (read-write, with sqlite-vec)
_search_engine: Engine | None = None
_SearchSession: sessionmaker | None = None

# Litloft DB engine (read-only)
_litloft_engine: Engine | None = None
_HomevaultSession: sessionmaker | None = None


import os

_SQLITE_VEC_PATH = os.environ.get(
    "SQLITE_VEC_PATH", "/usr/local/lib/sqlite-vec/vec0"
)


def _load_sqlite_vec(dbapi_conn: sqlite3.Connection, _: object) -> None:
    """Load sqlite-vec extension on new connections."""
    dbapi_conn.enable_load_extension(True)
    dbapi_conn.load_extension(_SQLITE_VEC_PATH)
    dbapi_conn.enable_load_extension(False)


def _enable_wal_mode(dbapi_conn: sqlite3.Connection, _: object) -> None:
    """Enable WAL mode for better concurrent read performance."""
    dbapi_conn.execute("PRAGMA journal_mode=WAL")
    dbapi_conn.execute("PRAGMA foreign_keys=ON")


def _resolve_search_db_path() -> str:
    """Resolve the search.db path, honoring the eval-harness override.

    The eval harness sets ``INTELLIGENCE_SEARCH_DB_PATH`` to point the
    service at a frozen snapshot DB instead of the live runtime DB. The
    override path must already exist (we never copy or migrate); the
    caller is responsible for providing a complete snapshot.

    Returns the path string passed to SQLAlchemy's ``create_engine``.
    """
    override = os.environ.get("INTELLIGENCE_SEARCH_DB_PATH", "").strip()
    if override:
        return override
    return str(settings.search_db_path)


def init_search_db() -> None:
    """Initialize the search database with sqlite-vec extension."""
    global _search_engine, _SearchSession

    settings.intelligence_data_dir.mkdir(parents=True, exist_ok=True)

    db_path = _resolve_search_db_path()
    _search_engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False, "timeout": 120},
    )

    event.listen(_search_engine, "connect", _load_sqlite_vec)
    event.listen(_search_engine, "connect", _enable_wal_mode)

    Base.metadata.create_all(_search_engine)

    _SearchSession = sessionmaker(bind=_search_engine, expire_on_commit=False)

    # Migrate vec_clip if dimension changed (model swap), and vec_text
    # if the configured text embedding MODEL NAME changed (spec
    # 2026-05-20-gui-text-embedding-model §2.1-2 — model-name keyed,
    # not dimension keyed). Both run BEFORE _create_vec_tables so a
    # dropped table is recreated at the new float[dim].
    with _search_engine.connect() as conn:
        _migrate_vec_clip_if_needed(conn)
        _migrate_vec_text_if_needed(conn)
        _create_vec_tables(conn)
        conn.commit()

    # Migrate indexed_files for thumbnail CLIP support (idempotent).
    with _search_engine.begin() as conn:
        _migrate_indexed_files_thumbnail_columns(conn)
    # Phase 4: rebrand legacy image clip embeddings as clip_thumbnail.
    # Spec 2026-05-02-thumbnail-clip-default-shallow-search.md.
    _migrate_image_clip_to_clip_thumbnail()

    # Migrate transcript_chunks for AI refine columns (idempotent).
    with _search_engine.begin() as conn:
        _migrate_transcript_chunks_if_needed(conn)

    # Migrate transcript_words: drop legacy word_index column (idempotent).
    with _search_engine.begin() as conn:
        _migrate_transcript_words_if_needed(conn)

    # Phase 1A foundation of the cloud-transcription-providers spec:
    # add ``speaker_id`` to transcript_words / transcript_chunks for
    # diarization-capable providers, and create ``job_records`` for
    # fail-loud job lifecycle history. All idempotent.
    with _search_engine.begin() as conn:
        _migrate_transcript_speaker_id_columns(conn)
        _create_job_records_table(conn)

    # Create suggested_tags table for auto-tagging
    with _search_engine.connect() as conn:
        _create_suggested_tags_table(conn)
        conn.commit()

    # Create retrieval_keywords table for SIRA-style LLM keyword expansion.
    with _search_engine.connect() as conn:
        _create_retrieval_keywords_table(conn)
        conn.commit()

    # Create file_summaries table for AI summaries
    with _search_engine.connect() as conn:
        _create_file_summaries_table(conn)
        conn.commit()

    # Migrate file_summaries for user-edit support (idempotent).
    with _search_engine.begin() as conn:
        _migrate_file_summaries_if_needed(conn)

    # Create detailed_summary_citations table (idempotent).
    with _search_engine.connect() as conn:
        _create_detailed_summary_citations_table(conn)
        conn.commit()

    # Create file_insights table (AI-output history, Step 1).
    with _search_engine.connect() as conn:
        _create_file_insights_table(conn)
        conn.commit()

    # Create pdf_markdown table (one Markdown body per PDF).
    with _search_engine.connect() as conn:
        _create_pdf_markdown_table(conn)
        conn.commit()

    # Backfill file_insights from existing detailed_summary rows.
    # Runs BEFORE the Step 2b column drop so the legacy data is
    # captured in file_insights before the source columns vanish.
    _backfill_file_insights_from_detailed_summary()

    # Step 2b: drop the detailed_* columns that moved to file_insights.
    # Separate from _migrate_file_summaries_if_needed so it can run
    # strictly after the backfill above.
    with _search_engine.begin() as conn:
        _migrate_file_summaries_drop_legacy_detailed_columns(conn)

    # Backfill fts_transcripts from existing transcript_chunks
    _backfill_fts_transcripts()
    # Phase 3: backfill the word-tokenized parallel tables from the
    # existing trigram tables so legacy DBs gain the dual-index
    # benefit without a full reindex.
    _backfill_fts_word_tables()
    # Re-index fts_retrieval_keywords if it was built with the wrong
    # tokenizer (unicode61 → trigram). Japanese compound phrases like
    # "傾聴の技術" are one unicode61 token, making "傾聴" un-queryable.
    _migrate_fts_retrieval_keywords_to_trigram()

    with _search_engine.begin() as conn:
        _migrate_tfidf_keywords_indexed(conn)


def _migrate_fts_retrieval_keywords_to_trigram() -> None:
    """Re-create fts_retrieval_keywords with trigram tokenizer if needed.

    The initial Phase 1 implementation used unicode61, which does not
    split CJK compound phrases — multi-character Japanese terms stored
    as a single token are not reachable by shorter sub-word queries.
    Trigram indexes every 3-char window and supports CJK substring
    matching, consistent with the other FTS tables in this DB.

    Data is preserved: retrieval_keywords (the canonical table) is the
    source of truth; fts_retrieval_keywords is just its FTS mirror.
    Runs on every startup but only writes when the schema needs fixing.
    """
    if _search_engine is None:
        return

    import logging as _logging
    _logger = _logging.getLogger(__name__)

    with _search_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND name='fts_retrieval_keywords'"
            )
        ).fetchone()

        if row is None:
            return  # Table doesn't exist yet; _create_vec_tables will create it correctly.

        # If the CREATE statement already uses trigram, nothing to do.
        create_sql: str = row[0] or ""
        if "trigram" in create_sql.lower():
            return

        _logger.info(
            "Migrating fts_retrieval_keywords from unicode61 → trigram"
        )
        conn.execute(text("DROP TABLE fts_retrieval_keywords"))
        conn.execute(text(
            "CREATE VIRTUAL TABLE fts_retrieval_keywords "
            "USING fts5(file_id, keywords, tokenize='trigram')"
        ))
        conn.execute(text(
            "INSERT INTO fts_retrieval_keywords(file_id, keywords) "
            "SELECT file_id, keywords FROM retrieval_keywords "
            "WHERE status = 'generated'"
        ))
        conn.commit()
        count = conn.execute(
            text("SELECT COUNT(*) FROM fts_retrieval_keywords")
        ).scalar()
        _logger.info(
            "fts_retrieval_keywords migrated to trigram; %d rows re-indexed",
            count,
        )


def _backfill_fts_transcripts() -> None:
    """Backfill fts_transcripts from existing transcript_chunks.

    Runs on startup. Skips if fts_transcripts already has data.
    """
    import logging
    logger = logging.getLogger(__name__)

    if _search_engine is None:
        return

    with _search_engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM fts_transcripts")
        ).scalar()
        if count and count > 0:
            return

        # Copy all transcript_chunks into fts_transcripts
        result = conn.execute(
            text(
                "INSERT INTO fts_transcripts(file_id, chunk_index, text) "
                "SELECT file_id, chunk_index, text FROM transcript_chunks"
            )
        )
        conn.commit()
        inserted = result.rowcount
        if inserted > 0:
            logger.info("Backfilled %d transcript chunks into fts_transcripts", inserted)


def _backfill_fts_word_tables() -> None:
    """Copy existing trigram FTS rows into the word-tokenized parallel tables.

    Phase 3 of the required-keyword hard filter spec: legacy DBs only
    have the trigram tables. To avoid forcing every operator through
    a full reindex, the word tables are populated by copying from
    their trigram siblings on first start. Subsequent re-indexing
    keeps both in sync via the upsert helpers; the backfill only runs
    when the word table is empty.

    Each pair is copied independently so a partial migration (e.g.
    one table backfilled, others not) self-heals on the next startup.
    """
    import logging
    logger = logging.getLogger(__name__)

    if _search_engine is None:
        return

    pairs: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        (
            "fts_files",
            "fts_files_word",
            ("file_id", "filename", "title", "description", "tags_text"),
        ),
        (
            "fts_transcripts",
            "fts_transcripts_word",
            ("file_id", "chunk_index", "text"),
        ),
        (
            "fts_text_content",
            "fts_text_content_word",
            ("file_id", "chunk_index", "page", "text"),
        ),
    )

    with _search_engine.connect() as conn:
        for trigram_table, word_table, columns in pairs:
            count = conn.execute(
                text(f"SELECT COUNT(*) FROM {word_table}")
            ).scalar()
            if count and count > 0:
                continue
            cols_csv = ", ".join(columns)
            result = conn.execute(text(
                f"INSERT INTO {word_table}({cols_csv}) "
                f"SELECT {cols_csv} FROM {trigram_table}"
            ))
            inserted = result.rowcount
            if inserted and inserted > 0:
                logger.info(
                    "Backfilled %d rows from %s into %s",
                    inserted, trigram_table, word_table,
                )
        conn.commit()


def _get_text_embedding_dim() -> int:
    """Get the text embedding dimension from model config."""
    from app.workers.embedder import _MODEL_DIMS
    from app.config import settings
    return _MODEL_DIMS.get(settings.models.text_embedding, 384)


def _get_clip_embedding_dim() -> int:
    """Get the CLIP embedding dimension from model config."""
    from app.workers.clip import _CLIP_DIMS
    from app.config import settings
    return _CLIP_DIMS.get(settings.models.clip, 512)


def _migrate_vec_clip_if_needed(conn: object) -> None:
    """Drop and recreate vec_clip if its dimension doesn't match config.

    sqlite-vec virtual tables cannot be ALTERed, so a dimension change
    requires DROP + CREATE. All existing CLIP embeddings are removed and
    clip_indexed flags are reset so the indexer re-processes all files.
    """
    import logging
    logger = logging.getLogger(__name__)

    expected_dim = _get_clip_embedding_dim()

    # Check if vec_clip exists
    row = conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_clip'")
    ).fetchone()
    if row is None:
        return  # Table doesn't exist yet; _create_vec_tables will handle it

    # Probe current dimension by reading the table's SQL definition
    # sqlite-vec tables store their schema in sqlite_master.sql
    schema_row = conn.execute(
        text("SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_clip'")
    ).fetchone()

    if schema_row is None:
        return

    # Parse "vector float[N]" from the CREATE statement
    import re
    match = re.search(r"float\[(\d+)\]", schema_row[0] or "")
    if match is None:
        return

    current_dim = int(match.group(1))
    if current_dim == expected_dim:
        return

    logger.warning(
        "CLIP dimension changed (%d → %d). Dropping vec_clip and "
        "resetting clip_indexed flags for re-indexing.",
        current_dim, expected_dim,
    )

    # Remove all CLIP embedding records from the ORM table
    conn.execute(text("DELETE FROM embeddings WHERE embedding_type = 'clip'"))

    # Drop the old virtual table
    conn.execute(text("DROP TABLE vec_clip"))

    # Reset clip_indexed so the indexer re-processes all files
    conn.execute(text("UPDATE indexed_files SET clip_indexed = 0"))

    conn.commit()


def _probe_vec_text_dim(conn: object) -> int | None:
    """Return the dimension recorded in ``vec_text``'s CREATE SQL.

    Mirrors the ``float[N]`` regex probe used by
    ``_migrate_vec_clip_if_needed``. Returns ``None`` when the table
    is absent or its schema carries no parsable dimension.
    """
    import re

    schema_row = conn.execute(
        text(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name='vec_text'"
        )
    ).fetchone()
    if schema_row is None:
        return None
    match = re.search(r"float\[(\d+)\]", schema_row[0] or "")
    if match is None:
        return None
    return int(match.group(1))


def _read_index_meta(conn: object, key: str) -> str | None:
    """Read ``index_meta[key]``; tolerate the table being absent.

    On a DB that predates this feature ``index_meta`` may not exist
    yet (it is created by ``_create_vec_tables``, which runs *after*
    this migration in ``init_search_db``), so a missing table is a
    legitimate "recorded value absent" signal, not an error.
    """
    try:
        row = conn.execute(
            text("SELECT value FROM index_meta WHERE key = :k"),
            {"k": key},
        ).fetchone()
    except (OperationalError, sqlite3.OperationalError):
        return None
    return row[0] if row else None


def _upsert_index_meta(conn: object, key: str, value: str) -> None:
    """Idempotently set ``index_meta[key] = value``.

    Creates ``index_meta`` first so this is safe to call before
    ``_create_vec_tables`` has run (the migration may need to record
    the configured model on a brand-new / pre-feature DB).
    """
    conn.execute(text(
        "CREATE TABLE IF NOT EXISTS index_meta ("
        "  key TEXT PRIMARY KEY,"
        "  value TEXT"
        ")"
    ))
    conn.execute(
        text(
            "INSERT INTO index_meta(key, value) VALUES(:k, :v) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        ),
        {"k": key, "v": value},
    )


_TEXT_EMBEDDING_MODEL_KEY = "text_embedding_model"


def _migrate_vec_text_if_needed(conn: object) -> None:
    """Rebuild ``vec_text`` when the configured text embedding model
    differs from the one the current index was built with.

    The trigger keys on the recorded MODEL NAME
    (``index_meta['text_embedding_model']``) versus the configured
    ``settings.models.text_embedding`` — NOT the vector dimension.
    Two different models can share a dimension (e5-base 768 ↔
    ruri-130m 768) yet produce non-comparable vector spaces; cosine
    distance across a mixed index is meaningless. Dimension-only
    keying (the latent bug in ``_migrate_vec_clip_if_needed``) would
    wrongly skip that swap. Spec invariant §2.1-1 / §2.1-2.

    Branches:
      (a) recorded absent, vec_text present, probed dim == expected
          -> the index was built at baseline before this feature
             existed; seed ``index_meta`` only, NO purge
             (invariant §2.1-5, no surprise full re-embed on upgrade).
      (b) recorded absent, vec_text absent
          -> fresh DB; record the configured model and return.
      (c) recorded == configured
          -> no-op.
      (d) recorded != configured (any dimension)
          -> in one transaction: DROP vec_text (recreated at the new
             ``float[dim]`` by ``_create_vec_tables`` immediately
             after, see ``init_search_db``); purge text_content
             embeddings, both FTS text tables, detailed summary
             citations; reset ``text_indexed``; record the new model.

    ``settings`` is imported lazily inside the function because the
    migration tests rebind ``app.config.settings``; a module-top
    binding would read a stale snapshot.
    """
    import logging

    from app.config import settings

    logger = logging.getLogger(__name__)

    configured = settings.models.text_embedding
    recorded = _read_index_meta(conn, _TEXT_EMBEDDING_MODEL_KEY)
    # Defensive: a non-str / blank index_meta value (NULL, whitespace,
    # a rolled-back build's payload) must route through the safe
    # seed/fresh logic, never directly into the destructive branch (d)
    # on a spurious string mismatch (a false positive here mass-purges
    # the text index).
    if not isinstance(recorded, str) or not recorded.strip():
        recorded = None

    if recorded is None:
        probed_dim = _probe_vec_text_dim(conn)
        if probed_dim is None:
            # (b) fresh DB — no vec_text yet. Just record the model
            # so the next restart can detect a change.
            _upsert_index_meta(
                conn, _TEXT_EMBEDDING_MODEL_KEY, configured
            )
            return
        expected_dim = _get_text_embedding_dim()
        if probed_dim == expected_dim:
            # (a) upgrade path: an index built at the baseline model
            # before index_meta existed. Treat as already-built;
            # seed the record only, do NOT re-embed (§2.1-5).
            logger.info(
                "Seeding index_meta text_embedding_model=%s for an "
                "existing vec_text (dim %d) — upgrade migration, no "
                "re-index.",
                configured, probed_dim,
            )
            _upsert_index_meta(
                conn, _TEXT_EMBEDDING_MODEL_KEY, configured
            )
            return
        # recorded absent but the existing vec_text dimension does
        # not match the configured model: the model genuinely
        # changed on a pre-feature DB. Fall through to the rebuild.
        logger.warning(
            "vec_text dim %d != configured model %s dim %d and no "
            "recorded model — treating as a model change and "
            "rebuilding.",
            probed_dim, configured, expected_dim,
        )
    elif recorded == configured:
        # (c) nothing changed.
        return

    # (d) recorded != configured (model name keyed; dimension is
    # irrelevant to the decision — the regression-critical
    # same-dimension swap path).
    #
    # Atomicity: _migrate_vec_clip_if_needed may have issued its own
    # conn.commit() just before this runs (init_search_db), so the
    # outer connection's implicit transaction is NOT guaranteed to
    # span the whole rebuild. Wrap the destructive sequence + the
    # index_meta upsert in an explicit SAVEPOINT so it is strictly
    # all-or-nothing: a crash/exception mid-purge rolls back to the
    # pre-migration state (old vec_text + embeddings intact, recorded
    # model unchanged) and the next boot retries cleanly, instead of
    # leaving a half-purged index later recreated empty by
    # _create_vec_tables.
    logger.warning(
        "Text embedding model changed (%s -> %s). Dropping vec_text "
        "and purging all text_content so the indexer re-embeds every "
        "text-indexed file.",
        recorded, configured,
    )

    nested = conn.begin_nested()
    try:
        conn.execute(text("DROP TABLE IF EXISTS vec_text"))
        conn.execute(text(
            "DELETE FROM embeddings "
            "WHERE embedding_type = 'text_content'"
        ))

        # Full purge of both FTS text indices. The per-file owner of
        # the fts_text_content / fts_text_content_word pair
        # (delete_fts_text_content) takes a Session; inside this
        # SAVEPOINT a Session(bind=conn) rolls its own DELETEs back on
        # close(), which would leave FTS rows behind. Branch (d) wipes
        # ALL text_content regardless, so a blanket delete on the same
        # conn is the correct equivalent and additionally clears any
        # orphan FTS rows. Keep both table names in lockstep with
        # delete_fts_text_content.
        conn.execute(text("DELETE FROM fts_text_content"))
        conn.execute(text("DELETE FROM fts_text_content_word"))

        conn.execute(
            text("UPDATE indexed_files SET text_indexed = 0")
        )
        conn.execute(
            text("DELETE FROM detailed_summary_citations")
        )

        _upsert_index_meta(
            conn, _TEXT_EMBEDDING_MODEL_KEY, configured
        )
        nested.commit()
    except Exception:
        nested.rollback()
        raise


def _migrate_image_clip_to_clip_thumbnail() -> None:
    """One-shot rebrand: image rows' ``clip`` embeddings → ``clip_thumbnail``.

    Before this rollout images were stored with ``embedding_type="clip"``
    (single embedding per file). Spec
    ``2026-05-02-thumbnail-clip-default-shallow-search.md`` reserves
    ``"clip"`` for video scene frames going forward and tags the
    image's single representative embedding as ``"clip_thumbnail"``.

    Done in chunks of ``_IMAGE_MIGRATION_CHUNK`` so the SQLite write
    lock is not held for a multi-second UPDATE on large libraries.
    Each chunk is its own transaction; partial completion is safe and
    self-heals on the next startup. Idempotent: a run with no remaining
    work returns immediately.

    Side effect: also flips ``clip_thumbnail_indexed`` to ``True`` for
    every migrated row so the new backfill queue does not re-pick
    already-embedded images.
    """
    logger = logging.getLogger(__name__)
    if _search_engine is None:
        return

    image_mimes = (
        "image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp",
    )

    with _search_engine.connect() as conn:
        # Discover the universe up front so progress is bounded and
        # observable; chunking just paces the writes.
        rows = conn.execute(
            text(
                "SELECT DISTINCT e.file_id "
                "FROM embeddings e "
                "JOIN indexed_files i ON i.file_id = e.file_id "
                "WHERE e.embedding_type = 'clip' "
                "AND i.mime_type IN :mimes"
            ).bindparams(bindparam("mimes", expanding=True)),
            {"mimes": list(image_mimes)},
        ).fetchall()

    file_ids = [r[0] for r in rows]
    if not file_ids:
        return

    logger.info(
        "Migrating %d image rows from embedding_type=clip → clip_thumbnail",
        len(file_ids),
    )

    chunk = _IMAGE_MIGRATION_CHUNK
    migrated = 0
    for start in range(0, len(file_ids), chunk):
        batch = file_ids[start:start + chunk]
        with _search_engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE embeddings SET embedding_type = 'clip_thumbnail' "
                    "WHERE embedding_type = 'clip' "
                    "AND file_id IN :ids"
                ).bindparams(bindparam("ids", expanding=True)),
                {"ids": batch},
            )
            conn.execute(
                text(
                    "UPDATE indexed_files SET clip_thumbnail_indexed = 1 "
                    "WHERE file_id IN :ids"
                ).bindparams(bindparam("ids", expanding=True)),
                {"ids": batch},
            )
        migrated += len(batch)
        logger.debug(
            "clip → clip_thumbnail batch committed: %d/%d",
            migrated, len(file_ids),
        )

    logger.info(
        "Image clip → clip_thumbnail migration complete: %d files",
        migrated,
    )


# Tunable for tests. Production value chosen so the per-batch write
# lock is sub-second on a typical SSD: 500 files × ~1 row each ≈ ~500
# UPDATEs per transaction.
_IMAGE_MIGRATION_CHUNK = 500


def _migrate_indexed_files_thumbnail_columns(conn: object) -> None:
    """Add ``thumbnail_path`` and ``clip_thumbnail_indexed`` columns.

    Spec ``2026-05-02-thumbnail-clip-default-shallow-search.md``:
    - ``thumbnail_path`` projects core ``File.thumbnail_path`` so the
      CLIP worker can embed the representative frame without a
      per-file Internal API roundtrip.
    - ``clip_thumbnail_indexed`` tracks completion of the new
      ``embedding_type="clip_thumbnail"`` route, separate from
      ``clip_indexed`` (which now means "scene-detected video frames").

    Idempotent: skips columns that already exist. Image migration
    (clip → clip_thumbnail) and the bulk ``thumbnail_path`` projection
    happen later in Phase 4.
    """
    cols = {
        row[1]
        for row in conn.execute(
            text("PRAGMA table_info(indexed_files)")
        ).fetchall()
    }
    if "thumbnail_path" not in cols:
        conn.execute(
            text("ALTER TABLE indexed_files ADD COLUMN thumbnail_path TEXT")
        )
    if "clip_thumbnail_indexed" not in cols:
        conn.execute(
            text(
                "ALTER TABLE indexed_files ADD COLUMN "
                "clip_thumbnail_indexed BOOLEAN NOT NULL DEFAULT 0"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS "
                "idx_indexed_files_clip_thumbnail_indexed "
                "ON indexed_files(clip_thumbnail_indexed)"
            )
        )


def _migrate_tfidf_keywords_indexed(conn: object) -> None:
    """Add ``tfidf_keywords_indexed`` column for pre-computed keyword embeddings.

    Idempotent: skips if column already exists.
    """
    cols = {
        row[1]
        for row in conn.execute(
            text("PRAGMA table_info(indexed_files)")
        ).fetchall()
    }
    if "tfidf_keywords_indexed" not in cols:
        conn.execute(
            text(
                "ALTER TABLE indexed_files ADD COLUMN "
                "tfidf_keywords_indexed BOOLEAN NOT NULL DEFAULT 0"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS "
                "idx_indexed_files_tfidf_keywords_indexed "
                "ON indexed_files(tfidf_keywords_indexed)"
            )
        )


def _migrate_transcript_chunks_if_needed(conn: object) -> None:
    """Evolve ``transcript_chunks`` schema for AI refine + re-chunking.

    - Adds ``text_refined_at`` if missing (original refine feature).
    - Drops ``text_original`` if present: refine now re-chunks the
      transcript at sentence boundaries derived from LLM-inserted
      punctuation, which invalidates any per-chunk original snapshot.
      Revert is instead "re-run whisper from scratch"; dropping the
      column keeps the schema honest about what we can restore.

    Idempotent: running twice is a no-op. Requires SQLite 3.35+ for
    ``DROP COLUMN`` (Docker image ships 3.40+).
    """
    cols = {
        row[1]
        for row in conn.execute(
            text("PRAGMA table_info(transcript_chunks)")
        ).fetchall()
    }
    if "text_refined_at" not in cols:
        conn.execute(
            text(
                "ALTER TABLE transcript_chunks ADD COLUMN text_refined_at TIMESTAMP"
            )
        )
    if "text_original" in cols:
        conn.execute(
            text("ALTER TABLE transcript_chunks DROP COLUMN text_original")
        )


def _migrate_transcript_words_if_needed(conn: object) -> None:
    """Drop legacy ``word_index`` column from ``transcript_words``.

    ``word_index`` had no externally-defined invariant and caused
    overflow bugs in refine (hako GfJ-m48_jisu3dpMfRkcg). Ordering
    now relies on ``(timestamp_start, id)``. Idempotent: running twice
    is a no-op. Requires SQLite 3.35+ for ``DROP COLUMN`` (the Docker
    image ships 3.40+).
    """
    import logging

    logger = logging.getLogger(__name__)

    # PRAGMA table_info is safe if the table doesn't exist yet (returns []).
    cols = {
        row[1]
        for row in conn.execute(
            text("PRAGMA table_info(transcript_words)")
        ).fetchall()
    }
    if not cols:
        # Table will be created fresh by Base.metadata.create_all — nothing
        # to migrate and nothing to log.
        return

    idx_names = {
        row[1]
        for row in conn.execute(
            text("PRAGMA index_list(transcript_words)")
        ).fetchall()
    }

    did_migrate = False

    if "idx_transcript_words_file_idx" in idx_names:
        conn.execute(text("DROP INDEX IF EXISTS idx_transcript_words_file_idx"))
        did_migrate = True

    # Legacy composite index using old naming — drop if it slipped through.
    if "idx_transcript_words_file_time" in idx_names:
        conn.execute(text("DROP INDEX IF EXISTS idx_transcript_words_file_time"))
        did_migrate = True

    if "word_index" in cols:
        conn.execute(text("ALTER TABLE transcript_words DROP COLUMN word_index"))
        did_migrate = True

    # Recreate the timestamp-based index if absent (covers both legacy
    # rename and fresh-drop paths).
    if "idx_transcript_words_file_ts" not in idx_names:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_transcript_words_file_ts "
                "ON transcript_words(file_id, timestamp_start)"
            )
        )

    if did_migrate:
        logger.info(
            "Migrated transcript_words: dropped legacy word_index column + index"
        )


def _migrate_transcript_speaker_id_columns(conn: object) -> None:
    """Add ``speaker_id`` to ``transcript_words`` and ``transcript_chunks``.

    Phase 1A foundation of the cloud-transcription-providers spec
    (2026-05-07). Diarization-capable providers (Deepgram /
    ElevenLabs Scribe) populate the column; non-diarized providers
    (whisper_local / openai_compatible) leave it NULL.

    Both columns are nullable so pre-existing rows stay NULL without
    a data migration. ``ALTER TABLE ADD COLUMN`` is idempotent here
    via the ``PRAGMA table_info`` probe.
    """
    for table in ("transcript_words", "transcript_chunks"):
        cols = {
            row[1]
            for row in conn.execute(
                text(f"PRAGMA table_info({table})")
            ).fetchall()
        }
        if not cols:
            # Table will be created fresh by ``Base.metadata.create_all``
            # — the new column is part of that schema, no migration needed.
            continue
        if "speaker_id" not in cols:
            conn.execute(
                text(f"ALTER TABLE {table} ADD COLUMN speaker_id TEXT")
            )


def _create_job_records_table(conn: object) -> None:
    """Create the ``job_records`` table if it doesn't exist.

    Phase 1A of the cloud-transcription-providers spec. The table is
    the fail-loud persistence layer for asynchronous worker jobs:
    each transcription / embedding / summary attempt writes one row
    so operators can ``SELECT * FROM job_records WHERE status='failed'``
    to observe failures that previously vanished into worker logs.

    Mirrors the ``JobRecord`` ORM model in ``app.models``. CREATE
    only — no ALTER — because the model is brand new in this release.
    """
    conn.execute(text(
        "CREATE TABLE IF NOT EXISTS job_records ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  file_id TEXT NOT NULL,"
        "  job_kind TEXT NOT NULL,"
        "  provider TEXT,"
        "  status TEXT NOT NULL,"
        "  error_class TEXT,"
        "  error_message TEXT,"
        "  attempted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  completed_at TIMESTAMP,"
        "  FOREIGN KEY (file_id) REFERENCES indexed_files(file_id)"
        "    ON DELETE CASCADE"
        ")"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_job_records_file_kind "
        "ON job_records(file_id, job_kind)"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_job_records_status "
        "ON job_records(status)"
    ))


def _create_vec_tables(conn: object) -> None:
    """Create sqlite-vec virtual tables if they don't exist."""
    text_dim = _get_text_embedding_dim()
    # Text embedding vectors (dimension depends on configured model)
    conn.execute(text(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_text "
        f"USING vec0(embedding_id TEXT PRIMARY KEY, vector float[{text_dim}])"
    ))

    # CLIP embedding vectors (dimension depends on configured model)
    clip_dim = _get_clip_embedding_dim()
    conn.execute(text(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_clip "
        f"USING vec0(embedding_id TEXT PRIMARY KEY, vector float[{clip_dim}])"
    ))

    # Generic index-level metadata key/value store. Created alongside
    # the vec tables so ``_migrate_vec_text_if_needed`` can read the
    # recorded ``text_embedding_model`` on a brand-new DB. The rebuild
    # trigger keys on the recorded MODEL NAME, not the vector
    # dimension (spec invariant §2.1-2; vec_clip's dimension-only
    # keying is a latent same-dim-swap bug we deliberately do not
    # replicate here).
    conn.execute(text(
        "CREATE TABLE IF NOT EXISTS index_meta ("
        "  key TEXT PRIMARY KEY,"
        "  value TEXT"
        ")"
    ))

    # FTS5 trigram index for keyword search on indexed_files
    conn.execute(text(
        "CREATE VIRTUAL TABLE IF NOT EXISTS fts_files "
        "USING fts5(file_id, filename, title, description, tags_text, "
        "tokenize='trigram')"
    ))

    # FTS5 trigram index for keyword search on transcript chunks
    conn.execute(text(
        "CREATE VIRTUAL TABLE IF NOT EXISTS fts_transcripts "
        "USING fts5(file_id, chunk_index, text, tokenize='trigram')"
    ))

    # FTS5 trigram index for keyword search on text content (PDF, etc.)
    conn.execute(text(
        "CREATE VIRTUAL TABLE IF NOT EXISTS fts_text_content "
        "USING fts5(file_id, chunk_index, page, text, tokenize='trigram')"
    ))

    # Phase 3 of the required-keyword hard filter spec
    # (2026-04-30-required-semantic-hybrid-retrieval): word-level FTS
    # parallel to the trigram tables. Trigram is excellent for CJK
    # substring matching but shatters Latin/Cyrillic/Hangul tokens
    # (e.g. "ViT" → "Vi"/"iT") so word-boundary languages return weak
    # matches. The unicode61 tokenizer with diacritic folding is the
    # natural complement: each table is queried in addition to its
    # trigram sibling and results UNION'd, so any single hit in either
    # tokenization is enough to pass the required-keyword filter.
    #
    # ``remove_diacritics 2`` strips combining marks AND Unicode
    # decomposable diacritics — "Café" matches "Cafe" without the
    # caller having to do anything special. Lower-case folding is
    # implicit in unicode61's default behaviour.
    conn.execute(text(
        "CREATE VIRTUAL TABLE IF NOT EXISTS fts_files_word "
        "USING fts5(file_id, filename, title, description, tags_text, "
        "tokenize=\"unicode61 remove_diacritics 2\")"
    ))
    conn.execute(text(
        "CREATE VIRTUAL TABLE IF NOT EXISTS fts_transcripts_word "
        "USING fts5(file_id, chunk_index, text, "
        "tokenize=\"unicode61 remove_diacritics 2\")"
    ))
    conn.execute(text(
        "CREATE VIRTUAL TABLE IF NOT EXISTS fts_text_content_word "
        "USING fts5(file_id, chunk_index, page, text, "
        "tokenize=\"unicode61 remove_diacritics 2\")"
    ))

    # fts5vocab auxiliary tables expose per-term document frequency for
    # the SIRA-inspired rarity filter (app/rag/rarity_filter.py). They
    # stay in sync with the host word-FTS tables automatically — no
    # backfill needed. ``'row'`` aggregates counts across all columns
    # so a single SELECT returns the term's DF in the chunk corpus.
    conn.execute(text(
        "CREATE VIRTUAL TABLE IF NOT EXISTS fts_transcripts_word_vocab "
        "USING fts5vocab('fts_transcripts_word', 'row')"
    ))
    conn.execute(text(
        "CREATE VIRTUAL TABLE IF NOT EXISTS fts_text_content_word_vocab "
        "USING fts5vocab('fts_text_content_word', 'row')"
    ))

    # SIRA-style LLM-generated keywords per file: synonyms, abbreviations,
    # alternate names the LLM predicts the user might search by. Lives in
    # its own FTS surface (NOT mixed into fts_files) so the UI can render
    # "expanded keyword" hits distinctly from body hits, and so the data
    # stays out of citation paths (3-tier design: tier-3 retrieval signal
    # only).
    #
    # Tokenizer: trigram (not unicode61). Japanese compound keywords
    # generated by the LLM may span multiple morphemes without spaces,
    # making them un-queryable by shorter sub-word queries under unicode61.
    # Trigram indexes 3-char windows for CJK substring matching, consistent
    # with fts_transcripts. Latin alphanumeric terms also match correctly.
    # Spec: docs/superpowers/specs/2026-05-14-sira-retrieval-keywords.md.
    conn.execute(text(
        "CREATE VIRTUAL TABLE IF NOT EXISTS fts_retrieval_keywords "
        "USING fts5(file_id, keywords, tokenize='trigram')"
    ))


def _create_suggested_tags_table(conn: object) -> None:
    """Create the suggested_tags table for auto-tagging if it doesn't exist."""
    conn.execute(text(
        "CREATE TABLE IF NOT EXISTS suggested_tags ("
        "  file_id TEXT PRIMARY KEY,"
        "  tags TEXT NOT NULL,"
        "  model TEXT NOT NULL,"
        "  context_type TEXT NOT NULL,"
        "  created_at TEXT NOT NULL,"
        "  status TEXT NOT NULL DEFAULT 'pending'"
        ")"
    ))


def _create_retrieval_keywords_table(conn: object) -> None:
    """Create the retrieval_keywords table for SIRA-style LLM keyword expansion.

    One row per file. ``keywords`` is a single whitespace-separated
    string (the canonical form the FTS engine indexes). ``model`` lets
    a later regenerate pass detect the row was produced by a previous
    model. ``context_type`` matches the existing summaries / auto_tags
    classification ('transcript' / 'document') so regeneration can
    pick the right extractor. ``status`` is either ``'generated'``
    (active, queried) or ``'hidden'`` (user-suppressed; not queried).

    No FK to indexed_files: matches the suggested_tags / file_summaries
    convention — purge is handled explicitly in ``_purge_file``.

    Spec: docs/superpowers/specs/2026-05-14-sira-retrieval-keywords.md.
    """
    conn.execute(text(
        "CREATE TABLE IF NOT EXISTS retrieval_keywords ("
        "  file_id TEXT PRIMARY KEY,"
        "  keywords TEXT NOT NULL,"
        "  model TEXT NOT NULL,"
        "  context_type TEXT NOT NULL,"
        "  status TEXT NOT NULL DEFAULT 'generated',"
        "  created_at TEXT NOT NULL,"
        "  updated_at TEXT"
        ")"
    ))


def upsert_retrieval_keywords(
    session: Session,
    *,
    file_id: str,
    keywords: str,
    model: str,
    context_type: str,
    status: str = "generated",
) -> None:
    """Write or replace a retrieval_keywords row + its FTS mirror.

    Mirrors the ``upsert_fts_files`` / ``upsert_fts_transcripts``
    pattern: the canonical table row and the FTS row are written
    together so they cannot drift. The caller is responsible for the
    surrounding transaction; this helper only runs the SQL.

    Args:
        session: Active search-DB session.
        file_id: The file id (also the PK).
        keywords: Whitespace-separated keyword string after blocklist +
            rarity filtering. Empty strings are accepted (the caller
            decides whether to skip writes for empty results).
        model: Generation model label, e.g. ``"gemini-2.5-flash"``.
        context_type: Either ``"transcript"`` or ``"document"`` to
            match the summaries / auto_tags classification.
        status: ``"generated"`` (default, active) or ``"hidden"``
            (user-suppressed; FTS row is still dropped so the data
            cannot accidentally re-enter queries).
    """
    now = datetime.now(UTC).isoformat()
    session.execute(
        text(
            "INSERT INTO retrieval_keywords"
            "(file_id, keywords, model, context_type, status, "
            " created_at, updated_at) "
            "VALUES(:file_id, :keywords, :model, :context_type, "
            "       :status, :created_at, :updated_at) "
            "ON CONFLICT(file_id) DO UPDATE SET "
            "  keywords=excluded.keywords,"
            "  model=excluded.model,"
            "  context_type=excluded.context_type,"
            "  status=excluded.status,"
            "  updated_at=excluded.updated_at"
        ),
        {
            "file_id": file_id,
            "keywords": keywords,
            "model": model,
            "context_type": context_type,
            "status": status,
            "created_at": now,
            "updated_at": now,
        },
    )
    # FTS: always rewrite. ``hidden`` rows leave the FTS empty so they
    # do not surface in search (the table row is kept as the audit
    # trail of what the LLM generated for this file).
    session.execute(
        text("DELETE FROM fts_retrieval_keywords WHERE file_id = :file_id"),
        {"file_id": file_id},
    )
    if status == "generated" and keywords:
        session.execute(
            text(
                "INSERT INTO fts_retrieval_keywords(file_id, keywords) "
                "VALUES(:file_id, :keywords)"
            ),
            {"file_id": file_id, "keywords": keywords},
        )


def delete_retrieval_keywords(session: Session, file_id: str) -> None:
    """Remove the retrieval_keywords row and its FTS mirror for ``file_id``."""
    session.execute(
        text("DELETE FROM retrieval_keywords WHERE file_id = :file_id"),
        {"file_id": file_id},
    )
    session.execute(
        text("DELETE FROM fts_retrieval_keywords WHERE file_id = :file_id"),
        {"file_id": file_id},
    )


def _create_file_summaries_table(conn: object) -> None:
    """Create the file_summaries table for AI summaries if it doesn't exist.

    Stores a single summary per file with two layers:
    - short_summary: ~1 sentence (30-80 chars)
    - long_summary: 3-5 sentences (200-400 chars)

    Unlike suggested_tags, summaries have no approve/dismiss workflow —
    the intelligence DB stays self-contained and the host DB is never touched.
    Status is either 'generated' (displayed) or 'hidden' (user opted out).

    ``short_original`` / ``long_original`` hold the AI output snapshot taken
    on first edit so the user can revert. ``edited_at`` flags a user-edited
    row (NULL = AI-generated, timestamp = last edit time).

    ``detailed_status`` / ``detailed_error`` carry the transient workflow
    state for the Markdown long-form summary (generating / generated /
    failed + error message). The full body, metadata, and edit history
    live on ``file_insights`` (Step 2b removed the redundant columns).
    """
    conn.execute(text(
        "CREATE TABLE IF NOT EXISTS file_summaries ("
        "  file_id TEXT PRIMARY KEY,"
        "  short_summary TEXT NOT NULL,"
        "  long_summary TEXT NOT NULL,"
        "  model TEXT NOT NULL,"
        "  context_type TEXT NOT NULL,"
        "  context_chars INTEGER NOT NULL,"
        "  was_truncated INTEGER NOT NULL DEFAULT 0,"
        "  status TEXT NOT NULL DEFAULT 'generated',"
        "  created_at TEXT NOT NULL,"
        "  edited_at TEXT,"
        "  short_original TEXT,"
        "  long_original TEXT,"
        "  detailed_status TEXT,"
        "  detailed_error TEXT,"
        "  visual_description TEXT,"
        "  visual_description_generated_at TEXT,"
        "  visual_description_model TEXT,"
        "  visual_description_status TEXT"
        ")"
    ))


def _migrate_file_summaries_if_needed(conn: object) -> None:
    """Evolve ``file_summaries`` schema for user-edit support.

    Adds ``edited_at`` / ``short_original`` / ``long_original`` if missing.
    Idempotent: running twice is a no-op.

    Unlike ``transcript_chunks`` (where refine re-chunks and 1:1 mapping
    is lost), summaries are one row per file and ``*_original`` can safely
    hold the last AI version for revert. Snapshot is taken lazily on the
    first edit so AI-only rows stay lean.
    """
    cols = {
        row[1]
        for row in conn.execute(
            text("PRAGMA table_info(file_summaries)")
        ).fetchall()
    }
    if "edited_at" not in cols:
        conn.execute(
            text("ALTER TABLE file_summaries ADD COLUMN edited_at TEXT")
        )
    if "short_original" not in cols:
        conn.execute(
            text("ALTER TABLE file_summaries ADD COLUMN short_original TEXT")
        )
    if "long_original" not in cols:
        conn.execute(
            text("ALTER TABLE file_summaries ADD COLUMN long_original TEXT")
        )
    # Detailed-summary workflow columns. Step 2b removed the body /
    # metadata / edit-history columns — that data now lives on
    # ``file_insights``. ``detailed_status`` + ``detailed_error``
    # remain as the transient generating / generated / failed marker.
    # The actual drop of the removed columns happens in
    # ``_migrate_file_summaries_drop_legacy_detailed_columns`` AFTER
    # the FileInsight backfill has had a chance to read them.
    if "detailed_status" not in cols:
        conn.execute(
            text("ALTER TABLE file_summaries ADD COLUMN detailed_status TEXT")
        )
    if "detailed_error" not in cols:
        conn.execute(
            text("ALTER TABLE file_summaries ADD COLUMN detailed_error TEXT")
        )
    # Vision-describe columns. Nullable — the row may pre-exist purely
    # for short/long/detailed summary data. ``visual_description_status``
    # uses plain TEXT (no CHECK constraint) so the app layer owns value
    # validation; spec values are NULL / "pending" / "success" / "failed"
    # / "unsupported".
    if "visual_description" not in cols:
        conn.execute(
            text("ALTER TABLE file_summaries ADD COLUMN visual_description TEXT")
        )
    if "visual_description_generated_at" not in cols:
        conn.execute(
            text(
                "ALTER TABLE file_summaries "
                "ADD COLUMN visual_description_generated_at TEXT"
            )
        )
    if "visual_description_model" not in cols:
        conn.execute(
            text(
                "ALTER TABLE file_summaries "
                "ADD COLUMN visual_description_model TEXT"
            )
        )
    if "visual_description_status" not in cols:
        conn.execute(
            text(
                "ALTER TABLE file_summaries "
                "ADD COLUMN visual_description_status TEXT"
            )
        )


def _create_detailed_summary_citations_table(conn: object) -> None:
    """Create ``detailed_summary_citations`` if it doesn't exist.

    One row per (file_id, section_path) — ``section_path`` comes from
    ``app.summary_parser`` and uniquely identifies a bullet / paragraph
    / table row within a detailed_summary document. ``has_citation``
    is True when the top-1 ``top_score`` crosses the configured
    threshold; rows with ``has_citation = False`` are still stored so
    the UI can render the "⚠ no source found" badge.

    ``citation_chunk_ids`` is a JSON-encoded array of chunk identifiers
    (e.g. ``["wh_abc_0", "txt_abc_3"]``). The prefix disambiguates
    transcript vs document chunks so the frontend can jump appropriately.
    """
    conn.execute(text(
        "CREATE TABLE IF NOT EXISTS detailed_summary_citations ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  file_id TEXT NOT NULL,"
        "  section_path TEXT NOT NULL,"
        "  segment_type TEXT NOT NULL,"
        "  segment_text TEXT NOT NULL,"
        "  citation_chunk_ids TEXT NOT NULL,"
        "  top_score REAL NOT NULL,"
        "  has_citation BOOLEAN NOT NULL,"
        "  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  UNIQUE (file_id, section_path)"
        ")"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_detailed_citations_file "
        "ON detailed_summary_citations(file_id)"
    ))


_STEP_2B_DROPPED_COLUMNS = (
    "detailed_summary",
    "detailed_model",
    "detailed_generated_at",
    "detailed_context_chars",
    "detailed_was_truncated",
    "detailed_original",
    "detailed_edited_at",
)


def _migrate_file_summaries_drop_legacy_detailed_columns(conn: object) -> None:
    """Drop the ``detailed_*`` columns superseded by ``file_insights``.

    Step 2b removed seven columns from ``file_summaries`` after Step 2a
    made ``file_insights`` the source of truth for detailed-summary
    content, metadata, and edit history. This runs **after**
    ``_backfill_file_insights_from_detailed_summary`` so any legacy
    rows have already been mirrored into ``file_insights`` before the
    columns vanish.

    SQLite ≥ 3.35 supports ``ALTER TABLE ... DROP COLUMN``; the addon's
    runtime (3.46) and the test harness (3.40+) both clear that bar.
    Idempotent: the drop short-circuits once the column is gone.
    """
    cols = {
        row[1]
        for row in conn.execute(
            text("PRAGMA table_info(file_summaries)")
        ).fetchall()
    }
    for col in _STEP_2B_DROPPED_COLUMNS:
        if col in cols:
            conn.execute(
                text(f"ALTER TABLE file_summaries DROP COLUMN {col}")
            )


def _create_file_insights_table(conn: object) -> None:
    """Create ``file_insights`` (AI-output history) if it doesn't exist.

    Step 1 of the FileInsight rollout (see hako ``vdPrMz0_adP-3C6Ogkjds``).
    Initially populated only for ``kind='detailed_summary'``; other kinds
    (``auto_tags``, ``key_questions``) will follow in later steps.

    Columns:
    - ``id``             PK (12-char URL-safe, generated app-side)
    - ``file_id``        cross-DB reference to core File.id (no FK)
    - ``kind``           free-form ``"detailed_summary"`` / ``"auto_tags"`` / ...
    - ``content``        raw LLM output (Markdown for summaries, JSON for tags)
    - ``metadata_json``  per-kind generation metadata (model, tokens, etc.)
    - ``status``         ``"active"`` | ``"superseded"`` | ``"invalidated"``
    - ``created_by``     ``"intelligence"`` | ``"manual"`` | ``"knowledge"``
    - ``created_at``     UTC timestamp (ISO string, SQLite TEXT)
    - ``invalidated_at`` UTC timestamp or NULL (reserved for chain invalidation)
    """
    conn.execute(text(
        "CREATE TABLE IF NOT EXISTS file_insights ("
        "  id TEXT PRIMARY KEY,"
        "  file_id TEXT NOT NULL,"
        "  kind TEXT NOT NULL,"
        "  content TEXT NOT NULL,"
        "  metadata_json TEXT,"
        "  status TEXT NOT NULL DEFAULT 'active',"
        "  created_by TEXT NOT NULL,"
        "  created_at TIMESTAMP NOT NULL,"
        "  invalidated_at TIMESTAMP"
        ")"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_file_insights_file_kind_status "
        "ON file_insights(file_id, kind, status)"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_file_insights_kind_status "
        "ON file_insights(kind, status)"
    ))


def _create_pdf_markdown_table(conn: object) -> None:
    """Create ``pdf_markdown`` (one Markdown body per PDF) if missing.

    Holds the PyMuPDF4LLM-generated Markdown rendering of each indexed
    PDF (see spec ``2026-04-27-intelligence-pdf-markdown-indexing.md``).
    The body is regenerated whenever a PDF is (re-)indexed; the row is
    removed automatically when the parent ``indexed_files`` row is
    deleted (FK CASCADE — requires ``PRAGMA foreign_keys=ON``, which the
    addon's connect listener already sets).

    Columns:
    - ``file_id``       PK + FK ``indexed_files.file_id`` ON DELETE CASCADE
    - ``markdown``      Markdown body (TEXT NOT NULL)
    - ``page_count``    page count of the source PDF (INTEGER NOT NULL)
    - ``extractor``     ``"pymupdf4llm"`` or ``"fitz_fallback"`` (TEXT NOT NULL)
    - ``generated_at``  first creation time (TIMESTAMP NOT NULL)
    - ``updated_at``    last write time, advanced by UPSERT (TIMESTAMP NOT NULL)
    """
    conn.execute(text(
        "CREATE TABLE IF NOT EXISTS pdf_markdown ("
        "  file_id TEXT PRIMARY KEY,"
        "  markdown TEXT NOT NULL,"
        "  page_count INTEGER NOT NULL,"
        "  extractor TEXT NOT NULL,"
        "  generated_at TIMESTAMP NOT NULL,"
        "  updated_at TIMESTAMP NOT NULL,"
        "  FOREIGN KEY (file_id) REFERENCES indexed_files(file_id)"
        "    ON DELETE CASCADE"
        ")"
    ))


def _generate_insight_id_local() -> str:
    """Mirror of ``app.models.generate_insight_id`` for the backfill path.

    Duplicated here so the backfill can run before the models module is
    fully imported (``Base.metadata.create_all`` runs inside
    ``init_search_db`` before the backfill helper).
    """
    return secrets.token_urlsafe(9)[:12]


def _backfill_file_insights_from_detailed_summary() -> None:
    """Populate ``file_insights`` from existing ``file_summaries`` rows.

    One-shot, idempotent: skipped entirely when any
    ``kind='detailed_summary'`` row already exists. Safe to run on
    every startup.

    Mapping:

    - Row with ``detailed_edited_at IS NULL``:
        1 insight row — status=active, created_by=intelligence,
        content=detailed_summary.

    - Row with ``detailed_edited_at IS NOT NULL``:
        2 insight rows:
          * status=superseded, created_by=intelligence,
            content=detailed_original (the pre-edit AI snapshot)
          * status=active, created_by=manual,
            content=detailed_summary (the edited body),
            metadata.edited_at populated.

    Rows whose ``detailed_summary`` is NULL or whose ``detailed_status``
    is not 'generated' are skipped — they have no body to log.
    """
    logger = logging.getLogger(__name__)
    if _search_engine is None:
        return

    with _search_engine.begin() as conn:
        # Safety: table exists on fresh installs via the CREATE call
        # above, but an older DB that somehow missed the create (test
        # harness shortcut) would raise. Probe first to avoid aborting
        # startup — the create call ran two steps earlier and the
        # probe is cheap.
        has_table = conn.execute(text(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='file_insights'"
        )).fetchone()
        if has_table is None:
            return

        existing = conn.execute(text(
            "SELECT 1 FROM file_insights "
            "WHERE kind = 'detailed_summary' LIMIT 1"
        )).fetchone()
        if existing is not None:
            return

        # ``file_summaries`` table may not exist in narrow test harnesses
        # that construct the search DB from scratch without summaries.
        has_source = conn.execute(text(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='file_summaries'"
        )).fetchone()
        if has_source is None:
            return

        # Fresh installs on the Step 2b schema no longer carry the
        # legacy detailed_* columns. Skip the backfill query in that
        # case — the SELECT would raise "no such column" and, by
        # definition, there is nothing to migrate from this table.
        source_cols = {
            row[1]
            for row in conn.execute(
                text("PRAGMA table_info(file_summaries)")
            ).fetchall()
        }
        if "detailed_summary" not in source_cols:
            return

        rows = conn.execute(text(
            "SELECT file_id, detailed_summary, detailed_model, "
            "detailed_generated_at, detailed_context_chars, "
            "detailed_was_truncated, detailed_original, detailed_edited_at "
            "FROM file_summaries "
            "WHERE detailed_summary IS NOT NULL "
            "  AND detailed_status = 'generated'"
        )).fetchall()

        if not rows:
            return

        inserted = 0
        for r in rows:
            (file_id, content, model, gen_at, ctx_chars,
             was_trunc, original, edited_at) = r

            base_meta = {
                "model": model,
                "context_chars": ctx_chars,
                "was_truncated": (
                    bool(was_trunc) if was_trunc is not None else None
                ),
            }

            if edited_at:
                # Edited row: preserve the original AI body as a
                # superseded entry so the edit history is traceable.
                if original:
                    conn.execute(text(
                        "INSERT INTO file_insights "
                        "(id, file_id, kind, content, metadata_json, "
                        " status, created_by, created_at) "
                        "VALUES (:id, :fid, 'detailed_summary', :c, :m, "
                        " 'superseded', 'intelligence', :ca)"
                    ), {
                        "id": _generate_insight_id_local(),
                        "fid": file_id,
                        "c": original,
                        "m": json.dumps(base_meta),
                        "ca": gen_at or datetime.now(UTC).isoformat(),
                    })
                    inserted += 1
                conn.execute(text(
                    "INSERT INTO file_insights "
                    "(id, file_id, kind, content, metadata_json, "
                    " status, created_by, created_at) "
                    "VALUES (:id, :fid, 'detailed_summary', :c, :m, "
                    " 'active', 'manual', :ca)"
                ), {
                    "id": _generate_insight_id_local(),
                    "fid": file_id,
                    "c": content,
                    "m": json.dumps({**base_meta, "edited_at": edited_at}),
                    "ca": edited_at,
                })
                inserted += 1
            else:
                conn.execute(text(
                    "INSERT INTO file_insights "
                    "(id, file_id, kind, content, metadata_json, "
                    " status, created_by, created_at) "
                    "VALUES (:id, :fid, 'detailed_summary', :c, :m, "
                    " 'active', 'intelligence', :ca)"
                ), {
                    "id": _generate_insight_id_local(),
                    "fid": file_id,
                    "c": content,
                    "m": json.dumps(base_meta),
                    "ca": gen_at or datetime.now(UTC).isoformat(),
                })
                inserted += 1

        if inserted:
            logger.info(
                "Backfilled %d detailed_summary rows into file_insights",
                inserted,
            )


def init_litloft_db() -> None:
    """Initialize read-only connection to Litloft's SQLite database."""
    global _litloft_engine, _HomevaultSession

    if not settings.litloft_db_path.exists():
        raise FileNotFoundError(
            f"Litloft database not found: {settings.litloft_db_path}"
        )

    _litloft_engine = create_engine(
        f"sqlite:///file:{settings.litloft_db_path}?mode=ro&uri=true",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    _HomevaultSession = sessionmaker(
        bind=_litloft_engine, expire_on_commit=False
    )


@contextmanager
def get_search_db() -> Generator[Session, None, None]:
    """Get a search database session.

    The write lock is acquired before yielding to serialize all writes
    (including flush() calls within workers) and held through commit.
    This prevents 'database is locked' errors from concurrent SQLite writers.
    """
    if _SearchSession is None:
        raise RuntimeError("Search database not initialized")
    session = _SearchSession()
    try:
        with _write_lock:
            yield session
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_search_db_read() -> Generator[Session, None, None]:
    """Get a read-only search database session.

    Does not acquire the write lock. Safe for concurrent SELECT queries
    under WAL mode — SQLite allows multiple simultaneous readers.
    Use only for operations that do not modify the database.
    """
    if _SearchSession is None:
        raise RuntimeError("Search database not initialized")
    session = _SearchSession()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_litloft_db() -> Generator[Session, None, None]:
    """Get a read-only Litloft database session."""
    if _HomevaultSession is None:
        raise RuntimeError("Litloft database not initialized")
    session = _HomevaultSession()
    try:
        yield session
    finally:
        session.close()


def get_search_engine() -> Engine:
    """Get the raw search database engine for direct SQL operations."""
    if _search_engine is None:
        raise RuntimeError("Search database not initialized")
    return _search_engine


ALLOWED_VECTOR_TABLES = frozenset({"vec_text", "vec_clip"})
ALLOWED_FTS_TABLES = frozenset({
    "fts_files",
    "fts_transcripts",
    "fts_text_content",
    "fts_files_word",
    "fts_transcripts_word",
    "fts_text_content_word",
})


def validate_vector_table(table_name: str) -> str:
    """Validate that a vector table name is in the allowed set.

    Prevents SQL injection via interpolated table names in sqlite-vec queries.
    """
    if table_name not in ALLOWED_VECTOR_TABLES:
        raise ValueError(f"Invalid vector table: {table_name}")
    return table_name


def get_write_lock() -> threading.Lock:
    """Get the global DB write lock for serializing vector table mutations."""
    return _write_lock


def upsert_fts_file(session: Session, file_id: str, filename: str,
                     title: str, description: str, tags_text: str) -> None:
    """Insert or replace a row in both FTS5 indices (trigram + word).

    The word-tokenized parallel table is part of the Phase 3 dual-index
    setup so word-boundary languages (Latin / Cyrillic / Hangul) get a
    real word match in addition to the trigram substring match.
    """
    payload = {
        "file_id": file_id,
        "filename": filename,
        "title": title,
        "description": description,
        "tags_text": tags_text,
    }
    session.execute(text(
        "INSERT OR REPLACE INTO fts_files(file_id, filename, title, description, tags_text) "
        "VALUES(:file_id, :filename, :title, :description, :tags_text)"
    ), payload)
    session.execute(text(
        "INSERT OR REPLACE INTO fts_files_word"
        "(file_id, filename, title, description, tags_text) "
        "VALUES(:file_id, :filename, :title, :description, :tags_text)"
    ), payload)


def delete_fts_file(session: Session, file_id: str) -> None:
    """Remove a row from both FTS5 indices (trigram + word)."""
    session.execute(text(
        "DELETE FROM fts_files WHERE file_id = :file_id"
    ), {"file_id": file_id})
    session.execute(text(
        "DELETE FROM fts_files_word WHERE file_id = :file_id"
    ), {"file_id": file_id})


def upsert_fts_transcripts(
    session: Session, file_id: str, chunks: list[dict]
) -> None:
    """Replace all transcript chunks in both FTS5 indices for a file.

    Args:
        session: Database session.
        file_id: The file ID.
        chunks: List of dicts with keys: chunk_index, text.
    """
    delete_fts_transcripts(session, file_id)
    for chunk in chunks:
        payload = {
            "file_id": file_id,
            "chunk_index": str(chunk["chunk_index"]),
            "text": chunk["text"],
        }
        session.execute(text(
            "INSERT INTO fts_transcripts(file_id, chunk_index, text) "
            "VALUES(:file_id, :chunk_index, :text)"
        ), payload)
        session.execute(text(
            "INSERT INTO fts_transcripts_word(file_id, chunk_index, text) "
            "VALUES(:file_id, :chunk_index, :text)"
        ), payload)


def delete_fts_transcripts(session: Session, file_id: str) -> None:
    """Remove all transcript chunks from both FTS5 indices for a file."""
    session.execute(text(
        "DELETE FROM fts_transcripts WHERE file_id = :file_id"
    ), {"file_id": file_id})
    session.execute(text(
        "DELETE FROM fts_transcripts_word WHERE file_id = :file_id"
    ), {"file_id": file_id})


def upsert_fts_text_content(
    session: Session, file_id: str, chunks: list[dict]
) -> None:
    """Replace all text content chunks in both FTS5 indices for a file.

    Args:
        session: Database session.
        file_id: The file ID.
        chunks: List of dicts with keys: chunk_index, page (int|None), text.
    """
    delete_fts_text_content(session, file_id)
    for chunk in chunks:
        payload = {
            "file_id": file_id,
            "chunk_index": str(chunk["chunk_index"]),
            "page": str(chunk["page"]) if chunk["page"] is not None else "",
            "text": chunk["text"],
        }
        session.execute(text(
            "INSERT INTO fts_text_content(file_id, chunk_index, page, text) "
            "VALUES(:file_id, :chunk_index, :page, :text)"
        ), payload)
        session.execute(text(
            "INSERT INTO fts_text_content_word(file_id, chunk_index, page, text) "
            "VALUES(:file_id, :chunk_index, :page, :text)"
        ), payload)


def delete_fts_text_content(session: Session, file_id: str) -> None:
    """Remove all text content chunks from both FTS5 indices for a file."""
    session.execute(text(
        "DELETE FROM fts_text_content WHERE file_id = :file_id"
    ), {"file_id": file_id})
    session.execute(text(
        "DELETE FROM fts_text_content_word WHERE file_id = :file_id"
    ), {"file_id": file_id})
