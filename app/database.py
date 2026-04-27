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

from sqlalchemy import event, create_engine, text
from sqlalchemy.engine import Engine
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

    # Migrate vec_clip if dimension changed (model swap), then create tables
    with _search_engine.connect() as conn:
        _migrate_vec_clip_if_needed(conn)
        _create_vec_tables(conn)
        conn.commit()

    # Migrate transcript_chunks for AI refine columns (idempotent).
    with _search_engine.begin() as conn:
        _migrate_transcript_chunks_if_needed(conn)

    # Migrate transcript_words: drop legacy word_index column (idempotent).
    with _search_engine.begin() as conn:
        _migrate_transcript_words_if_needed(conn)

    # Create suggested_tags table for auto-tagging
    with _search_engine.connect() as conn:
        _create_suggested_tags_table(conn)
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

    # Create pdf_markdown table and force re-index of existing PDFs so
    # the new Markdown-based extractor populates it (idempotent: the
    # reset only flips active PDFs that are still flagged text_indexed).
    with _search_engine.connect() as conn:
        _create_pdf_markdown_table(conn)
        _reset_text_indexed_for_pdfs(conn)
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


def _reset_text_indexed_for_pdfs(conn: object) -> None:
    """Force re-indexing of active PDFs so they pick up the new path.

    The PDF extractor is being switched from fitz raw-text to
    PyMuPDF4LLM Markdown (spec ``2026-04-27-intelligence-pdf-markdown-indexing``).
    Because ``index_text_content`` is invoked from the metadata worker
    loop (no standalone TEXT_CONTENT worker exists), we reset both
    ``text_indexed`` and ``metadata_indexed`` so the metadata worker
    picks up the file and triggers ``index_text_content`` for it.
    Re-running metadata embedding for a PDF is a no-op in content terms
    (filename / title / description don't change), only the embedding
    is recomputed.

    Scoped to ``mime_type = 'application/pdf' AND active = 1`` to avoid
    touching transcript-driven indexes or already soft-deleted rows.
    Idempotent: a no-op once every PDF has been re-indexed.
    """
    conn.execute(text(
        "UPDATE indexed_files "
        "SET text_indexed = 0, metadata_indexed = 0 "
        "WHERE mime_type = 'application/pdf' "
        "  AND active = 1"
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
ALLOWED_FTS_TABLES = frozenset({"fts_files", "fts_transcripts", "fts_text_content"})


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
    """Insert or replace a row in the FTS5 trigram index."""
    session.execute(text(
        "INSERT OR REPLACE INTO fts_files(file_id, filename, title, description, tags_text) "
        "VALUES(:file_id, :filename, :title, :description, :tags_text)"
    ), {
        "file_id": file_id,
        "filename": filename,
        "title": title,
        "description": description,
        "tags_text": tags_text,
    })


def delete_fts_file(session: Session, file_id: str) -> None:
    """Remove a row from the FTS5 trigram index."""
    session.execute(text(
        "DELETE FROM fts_files WHERE file_id = :file_id"
    ), {"file_id": file_id})


def upsert_fts_transcripts(
    session: Session, file_id: str, chunks: list[dict]
) -> None:
    """Replace all transcript chunks in the FTS5 trigram index for a file.

    Args:
        session: Database session.
        file_id: The file ID.
        chunks: List of dicts with keys: chunk_index, text.
    """
    delete_fts_transcripts(session, file_id)
    for chunk in chunks:
        session.execute(text(
            "INSERT INTO fts_transcripts(file_id, chunk_index, text) "
            "VALUES(:file_id, :chunk_index, :text)"
        ), {
            "file_id": file_id,
            "chunk_index": str(chunk["chunk_index"]),
            "text": chunk["text"],
        })


def delete_fts_transcripts(session: Session, file_id: str) -> None:
    """Remove all transcript chunks from the FTS5 trigram index for a file."""
    session.execute(text(
        "DELETE FROM fts_transcripts WHERE file_id = :file_id"
    ), {"file_id": file_id})


def upsert_fts_text_content(
    session: Session, file_id: str, chunks: list[dict]
) -> None:
    """Replace all text content chunks in the FTS5 trigram index for a file.

    Args:
        session: Database session.
        file_id: The file ID.
        chunks: List of dicts with keys: chunk_index, page (int|None), text.
    """
    delete_fts_text_content(session, file_id)
    for chunk in chunks:
        session.execute(text(
            "INSERT INTO fts_text_content(file_id, chunk_index, page, text) "
            "VALUES(:file_id, :chunk_index, :page, :text)"
        ), {
            "file_id": file_id,
            "chunk_index": str(chunk["chunk_index"]),
            "page": str(chunk["page"]) if chunk["page"] is not None else "",
            "text": chunk["text"],
        })


def delete_fts_text_content(session: Session, file_id: str) -> None:
    """Remove all text content chunks from the FTS5 trigram index for a file."""
    session.execute(text(
        "DELETE FROM fts_text_content WHERE file_id = :file_id"
    ), {"file_id": file_id})
