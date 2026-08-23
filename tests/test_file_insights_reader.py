"""Reader-path tests for the Step 2a migration.

After Step 2a, ``_get_detailed_summary`` and ``_fetch_detailed_edit_state``
source content + versioning metadata from ``file_insights`` (keeping
workflow status in ``file_summaries``). These tests assert the
equivalence of the read response after each write path: fresh save,
resave, edit, revert, and purge.

The reader returns the same shape as before migration so downstream
API response shaping (``_detailed_row_to_response``) is unaffected.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
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

from app.database import (  # noqa: E402
    Base,
    _create_detailed_summary_citations_table,
    _create_file_insights_table,
    _create_file_summaries_table,
)
from app.models import IndexedFile  # noqa: E402,F401


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    db_path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        _create_file_summaries_table(conn)
        _create_detailed_summary_citations_table(conn)
        _create_file_insights_table(conn)

    seed_session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        seed_session.add(
            IndexedFile(
                file_id="file001",
                drive="drive1",
                filename="lecture.mp4",
                file_path="/drives/drive1/lecture.mp4",
                file_type="video",
                mime_type="video/mp4",
                file_size=1000,
                active=True,
            )
        )
        seed_session.commit()
    finally:
        seed_session.close()

    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def _get_search_db():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr("app.database.get_search_db", _get_search_db)
    monkeypatch.setattr(
        "app.routers.summaries.get_search_db", _get_search_db
    )
    monkeypatch.setattr(
        "app.workers.summaries.get_search_db", _get_search_db
    )
    monkeypatch.setattr("app.indexer.get_search_db", _get_search_db)
    return engine, Session


def _seed_file_summary_row(engine, file_id: str) -> None:
    """Seed the short/long placeholder row so _save_detailed_summary UPDATE hits."""
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO file_summaries "
            "(file_id, short_summary, long_summary, model, context_type, "
            " context_chars, was_truncated, status, created_at) "
            "VALUES (:fid, '', '', '', '', 0, 0, 'hidden', :ca)"
        ), {"fid": file_id, "ca": now})


def test_reader_returns_none_when_no_summary(search_db):
    """No insights + no file_summaries row → None."""
    from app.workers.summaries import _get_detailed_summary

    assert _get_detailed_summary("file001") is None


def test_reader_returns_workflow_state_without_body(search_db):
    """file_summaries has status='generating' but no active insight → transient state."""
    engine, _ = search_db
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO file_summaries "
            "(file_id, short_summary, long_summary, model, context_type, "
            " context_chars, was_truncated, status, created_at, "
            " detailed_status) "
            "VALUES ('file001', '', '', '', '', 0, 0, 'hidden', :ca, "
            " 'generating')"
        ), {"ca": now})

    from app.workers.summaries import _get_detailed_summary
    row = _get_detailed_summary("file001")
    assert row is not None
    assert row["detailed_status"] == "generating"
    assert row["detailed_summary"] is None


def test_reader_returns_active_intelligence_body(search_db):
    """After _save_detailed_summary, reader returns the body + metadata."""
    engine, _ = search_db
    _seed_file_summary_row(engine, "file001")

    from app.workers.summaries import (
        _get_detailed_summary,
        _save_detailed_summary,
    )

    _save_detailed_summary(
        file_id="file001",
        detailed_summary="# AI body",
        model="gpt-4o-mini",
        context_chars=1500,
        was_truncated=False,
    )

    row = _get_detailed_summary("file001")
    assert row is not None
    assert row["detailed_summary"] == "# AI body"
    assert row["detailed_status"] == "generated"
    assert row["detailed_model"] == "gpt-4o-mini"
    assert row["detailed_context_chars"] == 1500
    assert row["detailed_was_truncated"] is False
    assert row["detailed_edited_at"] is None
    assert row["detailed_original"] is None
    assert row["detailed_generated_at"] is not None


def test_reader_returns_edited_body_with_ai_snapshot(
    search_db, monkeypatch,
):
    """After edit endpoint, reader maps active manual + superseded AI."""
    engine, _ = search_db
    _seed_file_summary_row(engine, "file001")

    from app.workers.summaries import (
        _get_detailed_summary,
        _save_detailed_summary,
    )
    from app.routers import summaries as router_mod

    initial = "## 章1\nAI 本文。\n\n## 章2\n別本文。"
    _save_detailed_summary(
        file_id="file001",
        detailed_summary=initial,
        model="m",
        context_chars=100,
        was_truncated=False,
    )

    monkeypatch.setattr(router_mod, "_require_detailed_feature_enabled",
                        lambda: None)
    monkeypatch.setattr(router_mod, "_require_file_in_drive",
                        lambda file_id, drive: None)

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(router_mod, "_emit_ws_event", _noop)
    monkeypatch.setattr(router_mod, "_recalculate_citations", _noop)

    from app.schemas import DetailedSummaryEditRequest
    from fastapi import BackgroundTasks
    import asyncio

    asyncio.run(
        router_mod.edit_detailed_summary_section(
            file_id="file001",
            body=DetailedSummaryEditRequest(
                section_heading="章1",
                subsection_heading=None,
                new_content="## 章1\n編集済み本文。",
            ),
            background_tasks=BackgroundTasks(),
            drive="drive1",
        )
    )

    row = _get_detailed_summary("file001")
    assert row is not None
    # Current body is the edited version.
    assert "編集済み本文" in row["detailed_summary"]
    # Snapshot for revert is the original AI body.
    assert row["detailed_original"] == initial
    assert row["detailed_edited_at"] is not None
    # Model / generated_at come from the AI row, not the edit event.
    assert row["detailed_model"] == "m"


def test_fetch_detailed_edit_state_matches_reader(search_db, monkeypatch):
    """_fetch_detailed_edit_state returns (body, original, edited_at) consistently."""
    engine, _ = search_db
    _seed_file_summary_row(engine, "file001")

    from app.database import get_search_db
    from app.routers.summaries import _fetch_detailed_edit_state
    from app.workers.summaries import _save_detailed_summary

    # Clean AI generation.
    _save_detailed_summary(
        file_id="file001",
        detailed_summary="ai body",
        model="m",
        context_chars=100,
        was_truncated=False,
    )

    with get_search_db() as session:
        state = _fetch_detailed_edit_state(session, "file001")
    assert state == ("ai body", None, None)


def test_fetch_detailed_edit_state_after_edit_returns_original(
    search_db, monkeypatch,
):
    """After an edit, the helper exposes the AI snapshot via position 1."""
    engine, _ = search_db
    _seed_file_summary_row(engine, "file001")

    from app.database import get_search_db
    from app.routers import summaries as router_mod
    from app.routers.summaries import _fetch_detailed_edit_state
    from app.workers.summaries import _save_detailed_summary

    initial = "## 章1\nAI 本文。\n\n## 章2\n別本文。"
    _save_detailed_summary(
        file_id="file001",
        detailed_summary=initial,
        model="m",
        context_chars=100,
        was_truncated=False,
    )

    monkeypatch.setattr(router_mod, "_require_detailed_feature_enabled",
                        lambda: None)
    monkeypatch.setattr(router_mod, "_require_file_in_drive",
                        lambda file_id, drive: None)

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(router_mod, "_emit_ws_event", _noop)
    monkeypatch.setattr(router_mod, "_recalculate_citations", _noop)

    from app.schemas import DetailedSummaryEditRequest
    from fastapi import BackgroundTasks
    import asyncio

    asyncio.run(
        router_mod.edit_detailed_summary_section(
            file_id="file001",
            body=DetailedSummaryEditRequest(
                section_heading="章1",
                subsection_heading=None,
                new_content="## 章1\n編集。",
            ),
            background_tasks=BackgroundTasks(),
            drive="drive1",
        )
    )

    with get_search_db() as session:
        state = _fetch_detailed_edit_state(session, "file001")
    assert state is not None
    body, original, edited_at = state
    assert "編集" in body
    assert original == initial
    assert edited_at is not None


def test_reader_returns_none_after_delete(search_db):
    """_delete_detailed_summary wipes both tables → reader returns None."""
    engine, _ = search_db
    _seed_file_summary_row(engine, "file001")

    from app.workers.summaries import (
        _delete_detailed_summary,
        _get_detailed_summary,
        _save_detailed_summary,
    )

    _save_detailed_summary(
        file_id="file001",
        detailed_summary="body",
        model="m",
        context_chars=100,
        was_truncated=False,
    )
    assert _get_detailed_summary("file001") is not None

    _delete_detailed_summary("file001")
    assert _get_detailed_summary("file001") is None


def test_reader_was_truncated_preserves_bool(search_db):
    """JSON metadata round-trip preserves the boolean type."""
    engine, _ = search_db
    _seed_file_summary_row(engine, "file001")

    from app.workers.summaries import (
        _get_detailed_summary,
        _save_detailed_summary,
    )

    _save_detailed_summary(
        file_id="file001",
        detailed_summary="body",
        model="m",
        context_chars=100,
        was_truncated=True,
    )
    row = _get_detailed_summary("file001")
    assert row["detailed_was_truncated"] is True
