"""RED-phase tests for the transcript_chunks schema migration.

Spec: docs/superpowers/specs/2026-04-15-intelligence-transcript-refine.md

After migration, ``transcript_chunks`` must have two additional nullable
columns:

* ``text_original TEXT NULL``
* ``text_refined_at TIMESTAMP NULL``

Existing rows (inserted before migration) must see ``NULL`` in both.

These tests run against an in-memory SQLite instance; they exercise the
initialisation code path that also runs in production startup.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

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
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.database import Base  # noqa: E402
from app.models import TranscriptChunk  # noqa: E402


@pytest.fixture()
def search_engine(tmp_path, monkeypatch):
    """Initialise a standalone search DB at a temp path.

    The intelligence addon exposes ``init_search_db()`` which creates
    tables + applies migrations. We point ``settings.search_db_path``
    at a temp file and let the normal init path run.
    """
    db_path = tmp_path / "search.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    # Insert a pre-migration row to verify NULL backfill semantics.
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO transcript_chunks "
                "(file_id, chunk_index, text, language, "
                "timestamp_start, timestamp_end, created_at) "
                "VALUES (:fid, :idx, :t, :lang, :ts, :te, :ca)"
            ),
            {
                "fid": "fileabc",
                "idx": 0,
                "t": "existing row",
                "lang": "ja",
                "ts": 0.0,
                "te": 5.0,
                "ca": datetime.now(UTC).isoformat(),
            },
        )

    # Invoke the migration entrypoint. Function name follows the
    # existing ``_migrate_vec_clip_if_needed`` convention — the import
    # is EXPECTED to fail during RED phase.
    from app.database import _migrate_transcript_chunks_if_needed

    with engine.begin() as conn:
        _migrate_transcript_chunks_if_needed(conn)

    return engine


def test_migration_adds_text_original_column(search_engine):
    with search_engine.connect() as conn:
        cols = {
            row[1]
            for row in conn.execute(
                text("PRAGMA table_info(transcript_chunks)")
            ).fetchall()
        }
    assert "text_original" in cols


def test_migration_adds_text_refined_at_column(search_engine):
    with search_engine.connect() as conn:
        cols = {
            row[1]
            for row in conn.execute(
                text("PRAGMA table_info(transcript_chunks)")
            ).fetchall()
        }
    assert "text_refined_at" in cols


def test_existing_rows_have_null_refine_columns(search_engine):
    with search_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT text_original, text_refined_at "
                "FROM transcript_chunks WHERE file_id = :fid"
            ),
            {"fid": "fileabc"},
        ).fetchone()

    assert row is not None
    assert row[0] is None  # text_original
    assert row[1] is None  # text_refined_at


def test_model_exposes_new_columns():
    """The SQLAlchemy model must declare the new columns so queries /
    ORM updates work. A column defined only at DDL level would force
    every caller into raw SQL.
    """
    assert hasattr(TranscriptChunk, "text_original"), (
        "TranscriptChunk.text_original must be an ORM-mapped column"
    )
    assert hasattr(TranscriptChunk, "text_refined_at"), (
        "TranscriptChunk.text_refined_at must be an ORM-mapped column"
    )


def test_migration_is_idempotent(search_engine):
    """Running the migration twice must not raise (columns already exist)."""
    from app.database import _migrate_transcript_chunks_if_needed

    with search_engine.begin() as conn:
        _migrate_transcript_chunks_if_needed(conn)  # should no-op cleanly
