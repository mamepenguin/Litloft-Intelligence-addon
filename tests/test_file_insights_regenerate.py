"""End-to-end regenerate regression test.

Reproduces the full regenerate lifecycle against real SQLite:

1. Seed an AI-generated detailed summary (file_summaries + file_insights).
2. Call ``regenerate_detailed_summary`` — this wipes both tables for the
   detailed kind.
3. Simulate the background worker: ``_set_detailed_status('generating')``,
   then ``_save_detailed_summary`` with the new body.
4. Call ``_get_detailed_summary`` and confirm it returns the new body.

This is the path the user walks through the UI: "Regenerate" button →
poll status → view the new summary. If the reader returns ``None`` at
step 4, the detail slot would disappear.
"""

from __future__ import annotations

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

    seed = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        seed.add(
            IndexedFile(
                file_id="abc123",
                drive="drive1",
                filename="lecture.mp4",
                file_path="/drives/drive1/lecture.mp4",
                file_type="video",
                mime_type="video/mp4",
                file_size=1000,
                active=True,
            )
        )
        seed.commit()
    finally:
        seed.close()

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
    return engine, Session


@pytest.mark.asyncio
async def test_regenerate_then_save_yields_readable_summary(
    search_db, monkeypatch,
):
    """Full round-trip: regenerate → set_status → save → read."""
    engine, _ = search_db

    from app.routers import summaries as router_mod
    from app.workers.summaries import (
        _get_detailed_summary,
        _save_detailed_summary,
        _set_detailed_status,
        DETAILED_STATUS_GENERATING,
    )

    # Seed the pre-regenerate state exactly the way the worker does:
    # status marker first (creates the file_summaries row with empty
    # short/long placeholders), then the body save. Both tables must
    # end up populated.
    _set_detailed_status("abc123", DETAILED_STATUS_GENERATING, model="m1")
    _save_detailed_summary(
        file_id="abc123",
        detailed_summary="# Old AI body",
        model="m1",
        context_chars=100,
        was_truncated=False,
    )
    assert _get_detailed_summary("abc123") is not None

    monkeypatch.setattr(router_mod, "_require_detailed_enabled", lambda: None)
    monkeypatch.setattr(
        router_mod, "_require_file_in_drive", lambda file_id, drive: None
    )
    monkeypatch.setattr(
        router_mod, "classify_detailed_missing_reason",
        lambda file_id: "not_generated",
    )

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(router_mod, "_clear_core_active_summary", _noop)

    # Inhibit the actual generation — we drive save manually so the
    # test is deterministic (no LLM stub needed).
    monkeypatch.setattr(
        router_mod, "generate_detailed_summary",
        lambda file_id, client: None,
    )
    monkeypatch.setattr(router_mod, "get_llm_client", lambda: MagicMock())

    from fastapi import BackgroundTasks
    result = await router_mod.regenerate_detailed_summary(
        "abc123", BackgroundTasks(), None, "drive1",
    )
    assert result.status == "accepted"

    # Post-clear state: the prior active row is now superseded, not
    # deleted — history is preserved across regenerate so a future
    # UI can surface or restore it.
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT content, status FROM file_insights "
            "WHERE file_id = 'abc123' AND kind = 'detailed_summary'"
        )).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "# Old AI body"
    assert rows[0][1] == "superseded"

    # Simulate the background worker: the pending/generating marker.
    _set_detailed_status("abc123", DETAILED_STATUS_GENERATING, model="m2")

    mid_flight = _get_detailed_summary("abc123")
    # During generation the slot should be visible as "generating", not
    # hidden. This was the symptom the user reported.
    assert mid_flight is not None
    assert mid_flight["detailed_status"] == "generating"
    assert mid_flight["detailed_summary"] is None

    # Save the new body — mirrors the worker's happy path.
    _save_detailed_summary(
        file_id="abc123",
        detailed_summary="# New AI body",
        model="m2",
        context_chars=200,
        was_truncated=False,
    )

    final = _get_detailed_summary("abc123")
    assert final is not None, (
        "Reader returned None after save — this is the 'slot disappears' bug"
    )
    assert final["detailed_summary"] == "# New AI body"
    assert final["detailed_status"] == "generated"
    assert final["detailed_model"] == "m2"


@pytest.mark.asyncio
async def test_regenerate_endpoint_keeps_slot_visible_during_background_window(
    search_db, monkeypatch,
):
    """Between regenerate's clear and the worker's first status write,
    the reader must not return ``None``. Otherwise the file-detail
    frontend hides the detailed-summary slot until generation completes,
    which is the user-visible symptom: "スロットごと非表示になってしまいます".

    After the fix, the regenerate endpoint marks the file as
    ``generating`` synchronously so the slot flashes the spinner
    instead of disappearing.
    """
    engine, _ = search_db

    from app.routers import summaries as router_mod
    from app.workers.summaries import (
        _get_detailed_summary,
        _save_detailed_summary,
        _set_detailed_status,
        DETAILED_STATUS_GENERATING,
    )

    # Pre-regenerate: a clean generated summary.
    _set_detailed_status("abc123", DETAILED_STATUS_GENERATING, model="m1")
    _save_detailed_summary(
        file_id="abc123",
        detailed_summary="# Old body",
        model="m1",
        context_chars=100,
        was_truncated=False,
    )

    monkeypatch.setattr(router_mod, "_require_detailed_enabled", lambda: None)
    monkeypatch.setattr(
        router_mod, "_require_file_in_drive", lambda file_id, drive: None
    )
    monkeypatch.setattr(
        router_mod, "classify_detailed_missing_reason",
        lambda file_id: "not_generated",
    )

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(router_mod, "_clear_core_active_summary", _noop)

    # Crucially, do NOT let the background task run — we want to see
    # the state the frontend would observe *between* the regenerate
    # response and the worker's first side effect.
    scheduled: list = []
    monkeypatch.setattr(
        router_mod, "generate_detailed_summary",
        lambda file_id, client: scheduled.append(file_id) or None,
    )
    monkeypatch.setattr(router_mod, "get_llm_client", lambda: MagicMock())

    from fastapi import BackgroundTasks
    await router_mod.regenerate_detailed_summary(
        "abc123", BackgroundTasks(), None, "drive1",
    )

    # Snapshot the reader *right after* the endpoint returned, before
    # the background task has done anything.
    between = _get_detailed_summary("abc123")
    assert between is not None, (
        "Reader returned None in the regenerate-to-worker window — "
        "slot would be hidden on the frontend"
    )
    # Status should indicate work is in flight.
    assert between["detailed_status"] == "generating"
    # No body yet; the frontend renders a spinner, not a hidden slot.
    assert between["detailed_summary"] is None


def test_regenerate_supersedes_active_even_without_file_summaries_row(
    search_db, monkeypatch,
):
    """Defensive invariant: regenerate must demote the active row
    regardless of whether file_summaries has a matching row.
    Otherwise a partial prior state (e.g. an orphaned insight from a
    bug) would remain as the active row when the new generation row
    is inserted, violating the one-active-per-(file, kind) rule.
    """
    engine, _ = search_db

    # Seed an orphan: file_insights populated without a file_summaries
    # row. This mirrors any edge case where the two tables drift.
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO file_insights "
            "(id, file_id, kind, content, metadata_json, "
            " status, created_by, created_at) "
            "VALUES ('orphan-id-0', 'abc123', 'detailed_summary', "
            " '# Stale body', NULL, 'active', 'intelligence', "
            " '2026-04-23T00:00:00')"
        ))

    from app.routers import summaries as router_mod

    monkeypatch.setattr(router_mod, "_require_detailed_enabled", lambda: None)
    monkeypatch.setattr(
        router_mod, "_require_file_in_drive", lambda file_id, drive: None
    )
    monkeypatch.setattr(
        router_mod, "classify_detailed_missing_reason",
        lambda file_id: "not_generated",
    )

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(router_mod, "_clear_core_active_summary", _noop)
    monkeypatch.setattr(
        router_mod, "generate_detailed_summary",
        lambda file_id, client: None,
    )
    monkeypatch.setattr(router_mod, "get_llm_client", lambda: MagicMock())

    from fastapi import BackgroundTasks
    import asyncio

    asyncio.get_event_loop().run_until_complete(
        router_mod.regenerate_detailed_summary(
            "abc123", BackgroundTasks(), None, "drive1",
        )
    )

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, status FROM file_insights "
            "WHERE file_id = 'abc123' AND kind = 'detailed_summary'"
        )).fetchall()
    # The row survives for history, but its status was demoted so a
    # subsequent _save_detailed_summary INSERT can claim the active
    # slot without violating the one-active invariant.
    assert len(rows) == 1
    assert rows[0][0] == "orphan-id-0"
    assert rows[0][1] == "superseded"


@pytest.mark.asyncio
async def test_regenerate_preserves_history_across_multiple_runs(
    search_db, monkeypatch,
):
    """End-to-end: every regenerate appends a new active row while
    demoting the previous one. After N regenerates the lineage
    contains one active and N-1 superseded rows keyed by descending
    created_at."""
    engine, _ = search_db

    from app.routers import summaries as router_mod
    from app.workers.summaries import (
        _save_detailed_summary,
        _set_detailed_status,
        DETAILED_STATUS_GENERATING,
    )

    monkeypatch.setattr(router_mod, "_require_detailed_enabled", lambda: None)
    monkeypatch.setattr(
        router_mod, "_require_file_in_drive", lambda file_id, drive: None
    )
    monkeypatch.setattr(
        router_mod, "classify_detailed_missing_reason",
        lambda file_id: "not_generated",
    )

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(router_mod, "_clear_core_active_summary", _noop)
    monkeypatch.setattr(
        router_mod, "generate_detailed_summary",
        lambda file_id, client: None,
    )
    monkeypatch.setattr(router_mod, "get_llm_client", lambda: MagicMock())

    # First generation.
    _set_detailed_status("abc123", DETAILED_STATUS_GENERATING, model="m1")
    _save_detailed_summary(
        file_id="abc123",
        detailed_summary="body v1",
        model="m1",
        context_chars=100,
        was_truncated=False,
    )

    from fastapi import BackgroundTasks

    # Regenerate twice more; each time feed a distinct body.
    for label, model in (("v2", "m2"), ("v3", "m3")):
        await router_mod.regenerate_detailed_summary(
            "abc123", BackgroundTasks(), None, "drive1",
        )
        _save_detailed_summary(
            file_id="abc123",
            detailed_summary=f"body {label}",
            model=model,
            context_chars=100,
            was_truncated=False,
        )

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT content, status FROM file_insights "
            "WHERE file_id = 'abc123' AND kind = 'detailed_summary' "
            "ORDER BY created_at"
        )).fetchall()

    # Three rows total, only the newest active.
    assert [r[0] for r in rows] == ["body v1", "body v2", "body v3"]
    assert [r[1] for r in rows] == ["superseded", "superseded", "active"]
