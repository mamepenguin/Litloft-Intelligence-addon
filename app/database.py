"""Database initialization for the semantic search service.

Manages two SQLite connections:
1. Search DB (read-write): sqlite-vec enabled, stores embeddings and index state
2. HomeVault DB (read-only): reads file metadata for indexing
"""

import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager

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

# HomeVault DB engine (read-only)
_homevault_engine: Engine | None = None
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
    """Add AI-refine columns (``text_original``, ``text_refined_at``).

    SQLite's ALTER TABLE ADD COLUMN is idempotent via a preflight
    PRAGMA table_info check; running this twice is a no-op.
    """
    cols = {
        row[1]
        for row in conn.execute(
            text("PRAGMA table_info(transcript_chunks)")
        ).fetchall()
    }
    if "text_original" not in cols:
        conn.execute(
            text("ALTER TABLE transcript_chunks ADD COLUMN text_original TEXT")
        )
    if "text_refined_at" not in cols:
        conn.execute(
            text(
                "ALTER TABLE transcript_chunks ADD COLUMN text_refined_at TIMESTAMP"
            )
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
        "  created_at TEXT NOT NULL"
        ")"
    ))


def init_homevault_db() -> None:
    """Initialize read-only connection to HomeVault's SQLite database."""
    global _homevault_engine, _HomevaultSession

    if not settings.homevault_db_path.exists():
        raise FileNotFoundError(
            f"HomeVault database not found: {settings.homevault_db_path}"
        )

    _homevault_engine = create_engine(
        f"sqlite:///file:{settings.homevault_db_path}?mode=ro&uri=true",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    _HomevaultSession = sessionmaker(
        bind=_homevault_engine, expire_on_commit=False
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
def get_homevault_db() -> Generator[Session, None, None]:
    """Get a read-only HomeVault database session."""
    if _HomevaultSession is None:
        raise RuntimeError("HomeVault database not initialized")
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
