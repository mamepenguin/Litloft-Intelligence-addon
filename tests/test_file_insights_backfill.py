"""Backfill tests for ``_backfill_file_insights_from_detailed_summary``.

Verifies:
- Pristine AI rows → 1 active intelligence insight
- Edited rows → 1 superseded intelligence + 1 active manual
- NULL / non-'generated' detailed_summary rows → skipped
- Idempotent: skipped entirely on second run (no duplicates)
- Missing ``file_summaries`` / ``file_insights`` table → no-op (no raise)
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

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


@pytest.fixture()
def engine_with_summaries(tmp_path):
    """Build a search DB with both tables + sample file_summaries rows."""
    from app.database import (
        _create_file_insights_table,
        _create_file_summaries_table,
    )

    db_path = tmp_path / "search.db"
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.begin() as conn:
        _create_file_summaries_table(conn)
        _create_file_insights_table(conn)

    return engine


def _insert_summary(engine, **kwargs):
    defaults = {
        "file_id": "file-1",
        "short_summary": "short",
        "long_summary": "long",
        "model": "test-model",
        "context_type": "video",
        "context_chars": 500,
        "was_truncated": 0,
        "status": "generated",
        "created_at": "2026-04-22T12:00:00",
        "detailed_summary": "# Detailed\n\nBody.",
        "detailed_status": "generated",
        "detailed_model": "detailed-model",
        "detailed_generated_at": "2026-04-22T13:00:00",
        "detailed_context_chars": 1200,
        "detailed_was_truncated": 0,
        "detailed_original": None,
        "detailed_edited_at": None,
    }
    defaults.update(kwargs)

    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO file_summaries "
            "(file_id, short_summary, long_summary, model, context_type, "
            " context_chars, was_truncated, status, created_at, "
            " detailed_summary, detailed_status, detailed_model, "
            " detailed_generated_at, detailed_context_chars, "
            " detailed_was_truncated, detailed_original, detailed_edited_at) "
            "VALUES (:file_id, :short_summary, :long_summary, :model, "
            " :context_type, :context_chars, :was_truncated, :status, "
            " :created_at, :detailed_summary, :detailed_status, "
            " :detailed_model, :detailed_generated_at, "
            " :detailed_context_chars, :detailed_was_truncated, "
            " :detailed_original, :detailed_edited_at)"
        ), defaults)


def _fetch_insights(engine) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT file_id, kind, content, metadata_json, status, "
            "created_by, created_at "
            "FROM file_insights ORDER BY created_at, id"
        )).fetchall()
    return [
        {
            "file_id": r[0],
            "kind": r[1],
            "content": r[2],
            "metadata_json": r[3],
            "status": r[4],
            "created_by": r[5],
            "created_at": r[6],
        }
        for r in rows
    ]


def test_backfill_pristine_row_maps_to_single_active(engine_with_summaries):
    """AI-only detailed_summary → 1 active intelligence insight."""
    _insert_summary(engine_with_summaries, file_id="file-clean")

    with patch("app.database._search_engine", engine_with_summaries):
        from app.database import _backfill_file_insights_from_detailed_summary
        _backfill_file_insights_from_detailed_summary()

    insights = _fetch_insights(engine_with_summaries)
    assert len(insights) == 1
    assert insights[0]["file_id"] == "file-clean"
    assert insights[0]["kind"] == "detailed_summary"
    assert insights[0]["status"] == "active"
    assert insights[0]["created_by"] == "intelligence"
    assert insights[0]["content"] == "# Detailed\n\nBody."
    meta = json.loads(insights[0]["metadata_json"])
    assert meta["model"] == "detailed-model"
    assert meta["context_chars"] == 1200
    assert meta["was_truncated"] is False


def test_backfill_edited_row_maps_to_two_rows(engine_with_summaries):
    """Edited row → 1 superseded AI + 1 active manual (2 rows total)."""
    _insert_summary(
        engine_with_summaries,
        file_id="file-edited",
        detailed_summary="# Edited\n\nUser-modified body.",
        detailed_original="# Detailed\n\nOriginal AI body.",
        detailed_edited_at="2026-04-23T09:00:00",
    )

    with patch("app.database._search_engine", engine_with_summaries):
        from app.database import _backfill_file_insights_from_detailed_summary
        _backfill_file_insights_from_detailed_summary()

    insights = _fetch_insights(engine_with_summaries)
    assert len(insights) == 2

    # Created_at order: original AI timestamp < edited timestamp.
    superseded = insights[0]
    active = insights[1]

    assert superseded["status"] == "superseded"
    assert superseded["created_by"] == "intelligence"
    assert superseded["content"] == "# Detailed\n\nOriginal AI body."

    assert active["status"] == "active"
    assert active["created_by"] == "manual"
    assert active["content"] == "# Edited\n\nUser-modified body."
    meta = json.loads(active["metadata_json"])
    assert meta["edited_at"] == "2026-04-23T09:00:00"


def test_backfill_edited_without_original_skips_superseded_row(
    engine_with_summaries,
):
    """Edge case: edited_at set but detailed_original is NULL → only active row."""
    _insert_summary(
        engine_with_summaries,
        file_id="file-edit-no-snapshot",
        detailed_summary="# Edited only",
        detailed_original=None,
        detailed_edited_at="2026-04-23T10:00:00",
    )

    with patch("app.database._search_engine", engine_with_summaries):
        from app.database import _backfill_file_insights_from_detailed_summary
        _backfill_file_insights_from_detailed_summary()

    insights = _fetch_insights(engine_with_summaries)
    assert len(insights) == 1
    assert insights[0]["status"] == "active"
    assert insights[0]["created_by"] == "manual"


def test_backfill_skips_rows_without_detailed_summary(engine_with_summaries):
    """short/long only rows are ignored."""
    _insert_summary(
        engine_with_summaries,
        file_id="file-no-detailed",
        detailed_summary=None,
        detailed_status=None,
        detailed_model=None,
        detailed_generated_at=None,
        detailed_context_chars=None,
        detailed_was_truncated=None,
    )

    with patch("app.database._search_engine", engine_with_summaries):
        from app.database import _backfill_file_insights_from_detailed_summary
        _backfill_file_insights_from_detailed_summary()

    insights = _fetch_insights(engine_with_summaries)
    assert insights == []


def test_backfill_skips_non_generated_status(engine_with_summaries):
    """pending / failed detailed_status rows are ignored."""
    _insert_summary(
        engine_with_summaries,
        file_id="file-pending",
        detailed_status="pending",
    )
    _insert_summary(
        engine_with_summaries,
        file_id="file-failed",
        detailed_status="failed",
    )

    with patch("app.database._search_engine", engine_with_summaries):
        from app.database import _backfill_file_insights_from_detailed_summary
        _backfill_file_insights_from_detailed_summary()

    insights = _fetch_insights(engine_with_summaries)
    assert insights == []


def test_backfill_is_idempotent(engine_with_summaries):
    """Second run must not duplicate insights."""
    _insert_summary(engine_with_summaries, file_id="file-1")

    with patch("app.database._search_engine", engine_with_summaries):
        from app.database import _backfill_file_insights_from_detailed_summary
        _backfill_file_insights_from_detailed_summary()
        _backfill_file_insights_from_detailed_summary()

    insights = _fetch_insights(engine_with_summaries)
    assert len(insights) == 1


def test_backfill_noop_when_insights_table_missing(tmp_path):
    """No file_insights table → silently return, no raise."""
    db_path = tmp_path / "search.db"
    engine = create_engine(f"sqlite:///{db_path}")
    from app.database import _create_file_summaries_table
    with engine.begin() as conn:
        _create_file_summaries_table(conn)

    with patch("app.database._search_engine", engine):
        from app.database import _backfill_file_insights_from_detailed_summary
        # Should not raise.
        _backfill_file_insights_from_detailed_summary()


def test_backfill_noop_when_summaries_table_missing(tmp_path):
    """No file_summaries table → silently return, no raise."""
    db_path = tmp_path / "search.db"
    engine = create_engine(f"sqlite:///{db_path}")
    from app.database import _create_file_insights_table
    with engine.begin() as conn:
        _create_file_insights_table(conn)

    with patch("app.database._search_engine", engine):
        from app.database import _backfill_file_insights_from_detailed_summary
        _backfill_file_insights_from_detailed_summary()

    # Insights table remains empty.
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM file_insights")
        ).scalar()
    assert count == 0
