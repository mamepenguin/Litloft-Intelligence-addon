"""Schema migration tests for ``transcript_chunks``.

After the re-chunking refine design lands, ``transcript_chunks`` must:

* expose ``text_refined_at TIMESTAMP NULL`` (marks AI-refined rows)
* NOT expose ``text_original`` (refine re-chunks on punctuation, so
  per-chunk originals no longer align; revert == re-run whisper).

The migration ``_migrate_transcript_chunks_if_needed`` drops a legacy
``text_original`` column if present, and adds ``text_refined_at`` to
pre-existing DBs that haven't seen it yet.
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

from app.database import Base  # noqa: E402
from app.models import TranscriptChunk  # noqa: E402


@pytest.fixture()
def search_engine(tmp_path):
    db_path = tmp_path / "search.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    # Simulate a legacy DB that still has text_original from a prior
    # refine design. The migration must drop it.
    with engine.begin() as conn:
        conn.execute(
            text("ALTER TABLE transcript_chunks ADD COLUMN text_original TEXT")
        )
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

    from app.database import _migrate_transcript_chunks_if_needed

    with engine.begin() as conn:
        _migrate_transcript_chunks_if_needed(conn)

    return engine


def _columns(engine):
    with engine.connect() as conn:
        return {
            row[1]
            for row in conn.execute(
                text("PRAGMA table_info(transcript_chunks)")
            ).fetchall()
        }


def test_migration_keeps_refined_at(search_engine):
    assert "text_refined_at" in _columns(search_engine)


def test_migration_drops_text_original(search_engine):
    assert "text_original" not in _columns(search_engine)


def test_model_has_only_refined_at():
    assert hasattr(TranscriptChunk, "text_refined_at"), (
        "TranscriptChunk.text_refined_at must be an ORM-mapped column"
    )
    assert not hasattr(TranscriptChunk, "text_original"), (
        "TranscriptChunk.text_original should have been removed with "
        "the re-chunking refine design"
    )


def test_migration_is_idempotent(search_engine):
    """Running the migration twice must not raise."""
    from app.database import _migrate_transcript_chunks_if_needed

    with search_engine.begin() as conn:
        _migrate_transcript_chunks_if_needed(conn)
