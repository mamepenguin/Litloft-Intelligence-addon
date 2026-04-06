"""Database initialization for the semantic search service.

Manages two SQLite connections:
1. Search DB (read-write): sqlite-vec enabled, stores embeddings and index state
2. HomeVault DB (read-only): reads file metadata for indexing
"""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import event, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.config import settings

Base = declarative_base()

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


def init_search_db() -> None:
    """Initialize the search database with sqlite-vec extension."""
    global _search_engine, _SearchSession

    settings.search_data_dir.mkdir(parents=True, exist_ok=True)

    _search_engine = create_engine(
        f"sqlite:///{settings.search_db_path}",
        echo=False,
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    event.listen(_search_engine, "connect", _load_sqlite_vec)
    event.listen(_search_engine, "connect", _enable_wal_mode)

    Base.metadata.create_all(_search_engine)

    _SearchSession = sessionmaker(bind=_search_engine, expire_on_commit=False)

    # Create virtual tables for vector search (sqlite-vec)
    with _search_engine.connect() as conn:
        _create_vec_tables(conn)
        conn.commit()


def _create_vec_tables(conn: object) -> None:
    """Create sqlite-vec virtual tables if they don't exist."""
    # Text embedding vectors (384 dimensions for multilingual-e5-small)
    conn.execute(text(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_text "
        "USING vec0(embedding_id TEXT PRIMARY KEY, vector float[384])"
    ))

    # CLIP embedding vectors (512 dimensions for ViT-B/32)
    conn.execute(text(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_clip "
        "USING vec0(embedding_id TEXT PRIMARY KEY, vector float[512])"
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
    """Get a search database session."""
    if _SearchSession is None:
        raise RuntimeError("Search database not initialized")
    session = _SearchSession()
    try:
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


def validate_vector_table(table_name: str) -> str:
    """Validate that a vector table name is in the allowed set.

    Prevents SQL injection via interpolated table names in sqlite-vec queries.
    """
    if table_name not in ALLOWED_VECTOR_TABLES:
        raise ValueError(f"Invalid vector table: {table_name}")
    return table_name
