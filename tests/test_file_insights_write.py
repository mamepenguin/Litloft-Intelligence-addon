"""Write-path integration for ``file_insights``.

Covers the full lifecycle through the production code paths:

- ``_save_detailed_summary`` → 1 active intelligence insight.
- Second generation → 1 superseded + 1 active (not re-generate; just
  the direct worker save call to prove the append semantics).
- Edit endpoint → superseded intelligence + active manual.
- Revert endpoint → superseded manual + active intelligence with
  ``reverted_from_manual`` metadata.
- ``_delete_detailed_summary`` → all insights for that file/kind gone.
- ``_purge_file`` → insights for the file removed along with other rows.

Fixtures mirror ``test_detailed_summary_endpoints.py`` so the same
shim pattern patches ``get_search_db`` across router + worker.
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

# Import models at module import-time so Base.metadata.create_all()
# inside the fixture sees every declarative table (IndexedFile, etc.).
# Without this, the first test in the file runs create_all against an
# empty metadata registry and the seed INSERT below would fail with
# "no such table: indexed_files".
from app.models import IndexedFile  # noqa: E402,F401


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    """Real SQLite with file_summaries + file_insights + indexed_files seeded."""
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
        # FTS mirrors used by _purge_file — create minimal rooms so the
        # purge test can run the helper end-to-end without SQL errors.
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_files "
            "USING fts5(file_id UNINDEXED, filename, description, tags)"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_transcripts "
            "USING fts5(file_id UNINDEXED, chunk_index UNINDEXED, text)"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_text_content "
            "USING fts5(file_id UNINDEXED, chunk_index UNINDEXED, text)"
        ))
        # Minimal suggested_tags table (referenced by _purge_file).
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS suggested_tags ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  file_id TEXT NOT NULL,"
            "  tag TEXT NOT NULL,"
            "  status TEXT NOT NULL DEFAULT 'suggested'"
            ")"
        ))

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
    """Pre-create the row so UPDATE-based save path has something to modify."""
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO file_summaries "
            "(file_id, short_summary, long_summary, model, context_type, "
            " context_chars, was_truncated, status, created_at, "
            " detailed_status) "
            "VALUES (:fid, '', '', '', '', 0, 0, 'hidden', :ca, 'generated')"
        ), {"fid": file_id, "ca": now})


def _fetch_insights(engine, file_id: str) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, file_id, kind, content, metadata_json, status, "
            "created_by, created_at "
            "FROM file_insights WHERE file_id = :fid "
            "ORDER BY created_at, id"
        ), {"fid": file_id}).fetchall()
    return [
        {
            "id": r[0],
            "file_id": r[1],
            "kind": r[2],
            "content": r[3],
            "metadata_json": r[4],
            "status": r[5],
            "created_by": r[6],
            "created_at": r[7],
        }
        for r in rows
    ]


def test_save_detailed_summary_inserts_active_insight(search_db):
    """First save → 1 active intelligence row."""
    engine, _ = search_db
    _seed_file_summary_row(engine, "file001")

    from app.workers.summaries import _save_detailed_summary

    _save_detailed_summary(
        file_id="file001",
        detailed_summary="# Body\nContent.",
        model="gpt-4o-mini",
        context_chars=1500,
        was_truncated=False,
    )

    insights = _fetch_insights(engine, "file001")
    assert len(insights) == 1
    assert insights[0]["kind"] == "detailed_summary"
    assert insights[0]["status"] == "active"
    assert insights[0]["created_by"] == "intelligence"
    assert insights[0]["content"] == "# Body\nContent."
    meta = json.loads(insights[0]["metadata_json"])
    assert meta["model"] == "gpt-4o-mini"
    assert meta["context_chars"] == 1500
    assert meta["was_truncated"] is False


def test_save_detailed_summary_supersedes_previous_active(search_db):
    """Second save (same file) → previous active moves to superseded."""
    engine, _ = search_db
    _seed_file_summary_row(engine, "file001")

    from app.workers.summaries import _save_detailed_summary

    _save_detailed_summary(
        file_id="file001",
        detailed_summary="v1",
        model="m1",
        context_chars=100,
        was_truncated=False,
    )
    _save_detailed_summary(
        file_id="file001",
        detailed_summary="v2",
        model="m2",
        context_chars=200,
        was_truncated=True,
    )

    insights = _fetch_insights(engine, "file001")
    assert len(insights) == 2
    # Exactly one active row.
    active = [i for i in insights if i["status"] == "active"]
    superseded = [i for i in insights if i["status"] == "superseded"]
    assert len(active) == 1
    assert len(superseded) == 1
    assert active[0]["content"] == "v2"
    assert superseded[0]["content"] == "v1"


def test_delete_detailed_summary_wipes_insights(search_db):
    """_delete_detailed_summary purges the full history for that file."""
    engine, _ = search_db
    _seed_file_summary_row(engine, "file001")

    from app.workers.summaries import (
        _delete_detailed_summary,
        _save_detailed_summary,
    )

    _save_detailed_summary(
        file_id="file001",
        detailed_summary="v1",
        model="m",
        context_chars=100,
        was_truncated=False,
    )
    _save_detailed_summary(
        file_id="file001",
        detailed_summary="v2",
        model="m",
        context_chars=200,
        was_truncated=False,
    )
    assert len(_fetch_insights(engine, "file001")) == 2

    ok = _delete_detailed_summary("file001")
    assert ok is True
    assert _fetch_insights(engine, "file001") == []


def test_edit_detailed_summary_appends_manual_active(search_db, monkeypatch):
    """edit_detailed_summary_section path → superseded intelligence + active manual."""
    engine, _ = search_db
    _seed_file_summary_row(engine, "file001")

    # Pre-populate an intelligence-generated detailed summary.
    from app.workers.summaries import _save_detailed_summary

    initial = "## 章1\n本文。\n\n## 章2\n別の本文。"
    _save_detailed_summary(
        file_id="file001",
        detailed_summary=initial,
        model="m",
        context_chars=100,
        was_truncated=False,
    )

    # Call the edit endpoint through the router. We bypass the feature
    # / drive guards to isolate the insight-append behaviour.
    from app.routers import summaries as router_mod

    monkeypatch.setattr(router_mod, "_require_detailed_feature_enabled",
                        lambda: None)
    monkeypatch.setattr(router_mod, "_require_file_in_drive",
                        lambda file_id, drive: None)

    # Bypass WebSocket and citation recompute (background work).
    async def _noop(*args, **kwargs):
        return None
    monkeypatch.setattr(router_mod, "_emit_ws_event", _noop)
    monkeypatch.setattr(router_mod, "_recalculate_citations", _noop)

    from app.schemas import DetailedSummaryEditRequest

    body = DetailedSummaryEditRequest(
        section_heading="章1",
        subsection_heading=None,
        new_content="## 章1\n編集済み本文。",
    )

    from fastapi import BackgroundTasks
    import asyncio

    asyncio.get_event_loop().run_until_complete(
        router_mod.edit_detailed_summary_section(
            file_id="file001",
            body=body,
            background_tasks=BackgroundTasks(),
            drive="drive1",
        )
    )

    insights = _fetch_insights(engine, "file001")
    assert len(insights) == 2
    active = [i for i in insights if i["status"] == "active"]
    superseded = [i for i in insights if i["status"] == "superseded"]
    assert len(active) == 1
    assert len(superseded) == 1
    assert active[0]["created_by"] == "manual"
    assert "編集済み本文" in active[0]["content"]
    assert superseded[0]["created_by"] == "intelligence"
    meta = json.loads(active[0]["metadata_json"])
    assert "edited_at" in meta


def test_revert_detailed_summary_records_reverted_intelligence_row(
    search_db, monkeypatch
):
    """revert_detailed_summary → manual superseded, new active intelligence
    with reverted_from_manual=true metadata."""
    engine, _ = search_db
    _seed_file_summary_row(engine, "file001")

    # Prime: save AI → edit → now revert.
    from app.workers.summaries import _save_detailed_summary
    from app.routers import summaries as router_mod

    monkeypatch.setattr(router_mod, "_require_detailed_feature_enabled",
                        lambda: None)
    monkeypatch.setattr(router_mod, "_require_file_in_drive",
                        lambda file_id, drive: None)

    async def _noop(*args, **kwargs):
        return None
    monkeypatch.setattr(router_mod, "_emit_ws_event", _noop)
    monkeypatch.setattr(router_mod, "_recalculate_citations", _noop)

    initial = "## 章1\n本文。\n\n## 章2\n別の本文。"
    _save_detailed_summary(
        file_id="file001",
        detailed_summary=initial,
        model="m",
        context_chars=100,
        was_truncated=False,
    )

    # Edit to create a manual active row.
    from app.schemas import DetailedSummaryEditRequest
    from fastapi import BackgroundTasks
    import asyncio

    asyncio.get_event_loop().run_until_complete(
        router_mod.edit_detailed_summary_section(
            file_id="file001",
            body=DetailedSummaryEditRequest(
                section_heading="章1",
                subsection_heading=None,
                new_content="## 章1\n改変。",
            ),
            background_tasks=BackgroundTasks(),
            drive="drive1",
        )
    )

    # Now revert.
    asyncio.get_event_loop().run_until_complete(
        router_mod.revert_detailed_summary(
            file_id="file001",
            background_tasks=BackgroundTasks(),
            drive="drive1",
        )
    )

    insights = _fetch_insights(engine, "file001")
    # Expect 3 rows: initial AI (superseded), manual edit (superseded),
    # reverted AI restore (active, reverted_from_manual=true).
    active = [i for i in insights if i["status"] == "active"]
    assert len(active) == 1
    assert active[0]["created_by"] == "intelligence"
    meta = json.loads(active[0]["metadata_json"])
    assert meta.get("reverted_from_manual") is True
    # Content matches the original (pre-edit) AI body.
    assert active[0]["content"] == initial


def test_purge_file_removes_all_insights(search_db):
    """_purge_file wipes file_insights rows along with everything else."""
    engine, _ = search_db
    _seed_file_summary_row(engine, "file001")

    from app.workers.summaries import _save_detailed_summary
    _save_detailed_summary(
        file_id="file001",
        detailed_summary="body",
        model="m",
        context_chars=100,
        was_truncated=False,
    )
    assert len(_fetch_insights(engine, "file001")) == 1

    from app.indexer import _purge_file
    _purge_file("file001")

    assert _fetch_insights(engine, "file001") == []
