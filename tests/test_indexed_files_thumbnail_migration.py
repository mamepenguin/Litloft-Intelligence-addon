"""Schema migration tests for ``indexed_files`` thumbnail columns.

Spec: ``2026-05-02-thumbnail-clip-default-shallow-search.md``.

``_migrate_indexed_files_thumbnail_columns`` must:

* add ``thumbnail_path`` (nullable TEXT) and ``clip_thumbnail_indexed``
  (NOT NULL DEFAULT 0 BOOLEAN) to a pre-existing ``indexed_files`` table
* preserve any existing rows (default new boolean to 0, ``thumbnail_path``
  to NULL)
* be idempotent (running twice is a no-op)
* create the ``idx_indexed_files_clip_thumbnail_indexed`` index
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# Mirror other intelligence migration tests: stub heavy ML deps that
# may not import cleanly in the bare test env.
for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine, text  # noqa: E402


@pytest.fixture()
def legacy_engine(tmp_path):
    """Build a DB with a pre-thumbnail-CLIP indexed_files schema."""
    db_path = tmp_path / "search.db"
    engine = create_engine(f"sqlite:///{db_path}")

    # Pre-thumbnail-CLIP CREATE: no thumbnail_path / clip_thumbnail_indexed.
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE indexed_files ("
            "  file_id TEXT PRIMARY KEY,"
            "  drive TEXT NOT NULL,"
            "  filename TEXT NOT NULL,"
            "  file_path TEXT NOT NULL,"
            "  file_type TEXT NOT NULL,"
            "  mime_type TEXT NOT NULL,"
            "  file_size INTEGER NOT NULL,"
            "  duration REAL,"
            "  active BOOLEAN NOT NULL DEFAULT 1,"
            "  metadata_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  clip_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  whisper_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  text_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  title TEXT NOT NULL DEFAULT '',"
            "  description TEXT NOT NULL DEFAULT '',"
            "  tags_text TEXT NOT NULL DEFAULT '',"
            "  indexed_at TEXT,"
            "  updated_at TEXT"
            ")"
        ))
        conn.execute(
            text(
                "INSERT INTO indexed_files "
                "(file_id, drive, filename, file_path, file_type, "
                "mime_type, file_size, clip_indexed) "
                "VALUES (:fid, :d, :fn, :fp, :ft, :m, :sz, :ci)"
            ),
            {
                "fid": "legacy-1",
                "d": "default",
                "fn": "video.mp4",
                "fp": "/mnt/default/video.mp4",
                "ft": "video",
                "m": "video/mp4",
                "sz": 12345,
                "ci": 1,  # already CLIP-indexed (scene frames)
            },
        )
    return engine


def _columns(engine) -> set[str]:
    with engine.connect() as conn:
        return {
            row[1]
            for row in conn.execute(
                text("PRAGMA table_info(indexed_files)")
            ).fetchall()
        }


def _indexes(engine) -> set[str]:
    with engine.connect() as conn:
        return {
            row[1]
            for row in conn.execute(
                text("PRAGMA index_list(indexed_files)")
            ).fetchall()
        }


def test_migration_adds_thumbnail_path_column(legacy_engine):
    from app.database import _migrate_indexed_files_thumbnail_columns

    with legacy_engine.begin() as conn:
        _migrate_indexed_files_thumbnail_columns(conn)

    assert "thumbnail_path" in _columns(legacy_engine)


def test_migration_adds_clip_thumbnail_indexed_column(legacy_engine):
    from app.database import _migrate_indexed_files_thumbnail_columns

    with legacy_engine.begin() as conn:
        _migrate_indexed_files_thumbnail_columns(conn)

    assert "clip_thumbnail_indexed" in _columns(legacy_engine)


def test_migration_creates_index_on_clip_thumbnail_indexed(legacy_engine):
    from app.database import _migrate_indexed_files_thumbnail_columns

    with legacy_engine.begin() as conn:
        _migrate_indexed_files_thumbnail_columns(conn)

    assert "idx_indexed_files_clip_thumbnail_indexed" in _indexes(legacy_engine)


def test_migration_preserves_existing_row_with_null_thumbnail(legacy_engine):
    """Pre-existing rows: thumbnail_path = NULL, clip_thumbnail_indexed = 0."""
    from app.database import _migrate_indexed_files_thumbnail_columns

    with legacy_engine.begin() as conn:
        _migrate_indexed_files_thumbnail_columns(conn)

    with legacy_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT clip_indexed, thumbnail_path, clip_thumbnail_indexed "
                "FROM indexed_files WHERE file_id = :fid"
            ),
            {"fid": "legacy-1"},
        ).fetchone()

    assert row is not None
    assert row[0] == 1  # clip_indexed survived
    assert row[1] is None
    assert row[2] == 0


def test_migration_is_idempotent(legacy_engine):
    from app.database import _migrate_indexed_files_thumbnail_columns

    with legacy_engine.begin() as conn:
        _migrate_indexed_files_thumbnail_columns(conn)
    # Second run must not raise (no duplicate ALTER TABLE).
    with legacy_engine.begin() as conn:
        _migrate_indexed_files_thumbnail_columns(conn)

    cols = _columns(legacy_engine)
    assert "thumbnail_path" in cols
    assert "clip_thumbnail_indexed" in cols


def test_migration_skips_when_columns_already_present(tmp_path):
    """Fresh schema (created via SQLAlchemy / Base.metadata) skips ALTERs."""
    from app.database import _migrate_indexed_files_thumbnail_columns

    db_path = tmp_path / "fresh.db"
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.begin() as conn:
        # Mimic a freshly-created table that already has both columns.
        conn.execute(text(
            "CREATE TABLE indexed_files ("
            "  file_id TEXT PRIMARY KEY,"
            "  drive TEXT NOT NULL,"
            "  filename TEXT NOT NULL,"
            "  file_path TEXT NOT NULL,"
            "  file_type TEXT NOT NULL,"
            "  mime_type TEXT NOT NULL,"
            "  file_size INTEGER NOT NULL,"
            "  duration REAL,"
            "  thumbnail_path TEXT,"
            "  active BOOLEAN NOT NULL DEFAULT 1,"
            "  metadata_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  clip_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  clip_thumbnail_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  whisper_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  text_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  title TEXT NOT NULL DEFAULT '',"
            "  description TEXT NOT NULL DEFAULT '',"
            "  tags_text TEXT NOT NULL DEFAULT '',"
            "  indexed_at TEXT,"
            "  updated_at TEXT"
            ")"
        ))

    with engine.begin() as conn:
        _migrate_indexed_files_thumbnail_columns(conn)

    cols = _columns(engine)
    assert "thumbnail_path" in cols
    assert "clip_thumbnail_indexed" in cols
