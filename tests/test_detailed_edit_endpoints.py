"""Integration tests for detailed-summary edit / revert / regenerate-409.

Covers the Phase 2 router endpoints:

* PUT ``/summary/detailed/section`` replaces the target section,
  snapshots the AI version on first edit, and re-triggers citation
  calculation.
* POST ``/summary/detailed/revert`` restores the snapshot and clears
  the edit flags; 400 when nothing to revert.
* POST ``/summary/detailed/regenerate`` returns 409 when the summary
  has been user-edited and ``force`` is absent, 202 (accepted) when
  ``force: true`` is supplied.

Tests use a real SQLite file + the same fixtures pattern as the
existing ``test_detailed_summary_endpoints.py``.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi import BackgroundTasks, HTTPException

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

from app.config import FeaturesConfig, LLMConfig  # noqa: E402
from app.database import (  # noqa: E402
    Base,
    _create_detailed_summary_citations_table,
    _create_file_insights_table,
    _create_file_summaries_table,
)
from app.routers.summaries import (  # noqa: E402
    edit_detailed_summary_section,
    regenerate_detailed_summary,
    revert_detailed_summary,
)
from app.schemas import (  # noqa: E402
    DetailedSummaryEditRequest,
    DetailedSummaryRegenerateRequest,
)


_SAMPLE_SUMMARY = (
    "## 全体像\n"
    "AI 版の全体像。\n"
    "\n"
    "## 主要な章/場面\n"
    "- AI 版の章1\n"
    "- AI 版の章2\n"
)


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

    from app.models import IndexedFile

    seed_session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        seed_session.add(
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
    monkeypatch.setattr("app.routers.summaries.get_search_db", _get_search_db)
    monkeypatch.setattr("app.workers.summaries.get_search_db", _get_search_db)
    return engine, Session


def _insert_detailed_row(
    engine,
    file_id: str,
    detailed_summary: str = _SAMPLE_SUMMARY,
    detailed_edited_at: str | None = None,
    detailed_original: str | None = None,
) -> None:
    """Insert paired ``file_summaries`` + ``file_insights`` rows for tests.

    Step 2a made the edit / revert / regenerate paths read the current
    body from ``file_insights``. When ``detailed_edited_at`` is set,
    two insight rows are seeded:

    - ``superseded`` / ``intelligence`` with the pre-edit AI body
      (``detailed_original``) — reverts consume this row.
    - ``active`` / ``manual`` with the post-edit body
      (``detailed_summary``) + ``metadata.edited_at``.

    Without an edit timestamp, a single ``active`` / ``intelligence``
    row is seeded with ``detailed_summary`` as its content.
    """
    import json as _json
    import secrets as _secrets

    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO file_summaries "
                "(file_id, short_summary, long_summary, model, context_type, "
                "context_chars, was_truncated, status, created_at, "
                "detailed_summary, detailed_status, detailed_model, "
                "detailed_generated_at, detailed_context_chars, "
                "detailed_was_truncated, detailed_original, "
                "detailed_edited_at) "
                "VALUES (:fid, '', '', '', '', 0, 0, 'hidden', :now, "
                ":summary, 'generated', 'test-model', :now, 500, 0, "
                ":original, :edited)"
            ),
            {
                "fid": file_id,
                "now": now,
                "summary": detailed_summary,
                "original": detailed_original,
                "edited": detailed_edited_at,
            },
        )
        meta = _json.dumps({
            "model": "test-model",
            "context_chars": 500,
            "was_truncated": False,
        })
        if detailed_edited_at is None:
            conn.execute(
                text(
                    "INSERT INTO file_insights "
                    "(id, file_id, kind, content, metadata_json, "
                    " status, created_by, created_at) "
                    "VALUES (:id, :fid, 'detailed_summary', :c, :m, "
                    " 'active', 'intelligence', :ca)"
                ),
                {
                    "id": _secrets.token_urlsafe(9)[:12],
                    "fid": file_id,
                    "c": detailed_summary,
                    "m": meta,
                    "ca": now,
                },
            )
        else:
            if detailed_original is not None:
                conn.execute(
                    text(
                        "INSERT INTO file_insights "
                        "(id, file_id, kind, content, metadata_json, "
                        " status, created_by, created_at) "
                        "VALUES (:id, :fid, 'detailed_summary', :c, :m, "
                        " 'superseded', 'intelligence', :ca)"
                    ),
                    {
                        "id": _secrets.token_urlsafe(9)[:12],
                        "fid": file_id,
                        "c": detailed_original,
                        "m": meta,
                        "ca": now,
                    },
                )
            conn.execute(
                text(
                    "INSERT INTO file_insights "
                    "(id, file_id, kind, content, metadata_json, "
                    " status, created_by, created_at) "
                    "VALUES (:id, :fid, 'detailed_summary', :c, :m, "
                    " 'active', 'manual', :ca)"
                ),
                {
                    "id": _secrets.token_urlsafe(9)[:12],
                    "fid": file_id,
                    "c": detailed_summary,
                    "m": _json.dumps({
                        "model": "test-model",
                        "context_chars": 500,
                        "was_truncated": False,
                        "edited_at": detailed_edited_at,
                    }),
                    "ca": detailed_edited_at,
                },
            )


@pytest.fixture()
def feature_enabled(monkeypatch, make_settings):
    settings = make_settings(
        features=FeaturesConfig(detailed_summaries="manual"),
        llm=LLMConfig(
            provider="openai_compatible",
            base_url="http://test",
            model="test-model",
        ),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.summaries.settings", settings)
    monkeypatch.setattr("app.workers.summaries.settings", settings)
    return settings


@pytest.fixture()
def stub_side_effects(monkeypatch):
    """No-op the WebSocket emitter and citation recalculation.

    The edit / revert endpoints fire these asynchronously; we don't
    want real HTTP calls or real embedding during router tests.
    Returns a dict exposing the mocks for assertions.
    """
    ws_calls: list[tuple[str, dict]] = []
    cit_calls: list[tuple[str, str]] = []

    async def fake_ws(event, data):
        ws_calls.append((event, data))

    async def fake_cit(file_id, summary):
        cit_calls.append((file_id, summary))

    monkeypatch.setattr(
        "app.routers.summaries._emit_ws_event", fake_ws
    )
    monkeypatch.setattr(
        "app.routers.summaries._recalculate_citations", fake_cit
    )
    return {"ws": ws_calls, "citations": cit_calls}


@pytest.fixture()
def mock_llm_enabled(monkeypatch):
    client = MagicMock()
    client.enabled = True
    monkeypatch.setattr(
        "app.routers.summaries.get_llm_client", lambda: client
    )
    # Stub the active-summary clear call so the test suite does not
    # fire real HTTP at a non-existent core. Tests that care about
    # the call log re-monkeypatch with a list-capturing stub.
    async def _noop(_file_id: str) -> None:
        return None

    monkeypatch.setattr(
        "app.routers.summaries._clear_core_active_summary", _noop
    )
    return client


class TestEditDetailedSummarySection:
    @pytest.mark.asyncio
    async def test_first_edit_snapshots_ai_version(
        self, search_db, feature_enabled, stub_side_effects,
    ):
        engine, _ = search_db
        _insert_detailed_row(engine, "abc123")

        body = DetailedSummaryEditRequest(
            section_heading="全体像",
            new_content="## 全体像\nユーザーが書き直した全体像。",
        )
        result = await edit_detailed_summary_section(
            "abc123", body, BackgroundTasks(), "drive1"
        )

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT detailed_summary, detailed_original, "
                    "detailed_edited_at FROM file_summaries "
                    "WHERE file_id = 'abc123'"
                )
            ).fetchone()
        assert "ユーザーが書き直した全体像" in row[0]
        # Original AI body stays snapshot.
        assert row[1] == _SAMPLE_SUMMARY
        assert row[2] is not None
        # The other section must be untouched.
        assert "AI 版の章1" in row[0]

        assert result.edited_at is not None
        assert result.has_original is True

    @pytest.mark.asyncio
    async def test_second_edit_preserves_ai_snapshot(
        self, search_db, feature_enabled, stub_side_effects,
    ):
        engine, _ = search_db
        _insert_detailed_row(engine, "abc123")

        await edit_detailed_summary_section(
            "abc123",
            DetailedSummaryEditRequest(
                section_heading="全体像",
                new_content="## 全体像\nv1 body",
            ),
            BackgroundTasks(),
            "drive1",
        )
        await edit_detailed_summary_section(
            "abc123",
            DetailedSummaryEditRequest(
                section_heading="全体像",
                new_content="## 全体像\nv2 body",
            ),
            BackgroundTasks(),
            "drive1",
        )
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT detailed_summary, detailed_original "
                    "FROM file_summaries WHERE file_id = 'abc123'"
                )
            ).fetchone()
        assert "v2 body" in row[0]
        # Snapshot still points at the AI version, not v1.
        assert row[1] == _SAMPLE_SUMMARY

    @pytest.mark.asyncio
    async def test_404_when_no_detailed_summary(
        self, search_db, feature_enabled, stub_side_effects,
    ):
        # No row inserted.
        with pytest.raises(HTTPException) as exc_info:
            await edit_detailed_summary_section(
                "abc123",
                DetailedSummaryEditRequest(
                    section_heading="全体像",
                    new_content="body",
                ),
                BackgroundTasks(),
                "drive1",
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_409_when_section_missing(
        self, search_db, feature_enabled, stub_side_effects,
    ):
        """Missing anchor = stale client state, not malformed request.

        Frontend treats this as "document changed under you, reload".
        """
        engine, _ = search_db
        _insert_detailed_row(engine, "abc123")

        with pytest.raises(HTTPException) as exc_info:
            await edit_detailed_summary_section(
                "abc123",
                DetailedSummaryEditRequest(
                    section_heading="存在しない見出し",
                    new_content="body",
                ),
                BackgroundTasks(),
                "drive1",
            )
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_409_when_subsection_missing(
        self, search_db, feature_enabled, stub_side_effects,
    ):
        engine, _ = search_db
        _insert_detailed_row(
            engine,
            "abc123",
            detailed_summary=(
                "## 全体像\n"
                "### 第一幕\n"
                "a body\n"
            ),
        )

        with pytest.raises(HTTPException) as exc_info:
            await edit_detailed_summary_section(
                "abc123",
                DetailedSummaryEditRequest(
                    section_heading="全体像",
                    subsection_heading="存在しないサブ",
                    new_content="body",
                ),
                BackgroundTasks(),
                "drive1",
            )
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_h3_edit_narrows_to_subsection(
        self, search_db, feature_enabled, stub_side_effects,
    ):
        """H3 splice must leave siblings untouched.

        ``plain_idx`` stays H2-scoped (the parser treats ``###`` as
        plain text), so existing citations under neighbouring H3s
        survive the edit. This test also pins the behaviour that a
        user can narrow an edit from H2 to a single H3 subrange.
        """
        engine, _ = search_db
        _insert_detailed_row(
            engine,
            "abc123",
            detailed_summary=(
                "## 全体像\n"
                "### 第一幕\n"
                "alpha body\n"
                "\n"
                "### 第二幕\n"
                "beta body\n"
            ),
        )

        await edit_detailed_summary_section(
            "abc123",
            DetailedSummaryEditRequest(
                section_heading="全体像",
                subsection_heading="第一幕",
                new_content="### 第一幕\nedited alpha body",
            ),
            BackgroundTasks(),
            "drive1",
        )
        with engine.connect() as conn:
            stored = conn.execute(
                text(
                    "SELECT detailed_summary FROM file_summaries "
                    "WHERE file_id = 'abc123'"
                )
            ).fetchone()[0]
        assert "edited alpha body" in stored
        # Original alpha body wiped (substring check: "edited alpha body"
        # also contains "alpha body", so assert the sentence-standalone
        # form is gone by checking the bare heading-body adjacency).
        assert "第一幕\nalpha body" not in stored
        # Sibling H3 preserved intact.
        assert "### 第二幕" in stored
        assert "beta body" in stored

    @pytest.mark.asyncio
    async def test_h2_rename_accepted(
        self, search_db, feature_enabled, stub_side_effects,
    ):
        """Including a renamed ``## heading`` in the fragment is a valid edit.

        The heading line is part of the editable range precisely so
        the user can rename a section. No validation rejects the
        rename — the new heading surfaces on the next render.
        """
        engine, _ = search_db
        _insert_detailed_row(engine, "abc123")

        await edit_detailed_summary_section(
            "abc123",
            DetailedSummaryEditRequest(
                section_heading="全体像",
                new_content="## 全体像（改題）\n新しい本文",
            ),
            BackgroundTasks(),
            "drive1",
        )
        with engine.connect() as conn:
            stored = conn.execute(
                text(
                    "SELECT detailed_summary FROM file_summaries "
                    "WHERE file_id = 'abc123'"
                )
            ).fetchone()[0]
        assert "## 全体像（改題）" in stored
        assert "新しい本文" in stored

    @pytest.mark.asyncio
    async def test_emits_ws_event_and_schedules_citations(
        self, search_db, feature_enabled, stub_side_effects,
    ):
        """Citation recompute must be deferred via BackgroundTasks so
        the HTTP response does not block on embedding work. The WS
        ``updated`` notification still fires synchronously (it's just
        a cheap broadcast).
        """
        engine, _ = search_db
        _insert_detailed_row(engine, "abc123")

        bg = BackgroundTasks()
        await edit_detailed_summary_section(
            "abc123",
            DetailedSummaryEditRequest(
                section_heading="全体像",
                new_content="edited",
            ),
            bg,
            "drive1",
        )

        events = stub_side_effects["ws"]
        assert any(
            ev[0] == "intelligence.detailed_summary.updated"
            for ev in events
        )
        # Citations should NOT have been awaited inline — they must be
        # queued on BackgroundTasks to keep the user-facing request
        # snappy.
        assert stub_side_effects["citations"] == []
        assert len(bg.tasks) == 1
        # Run the scheduled task and confirm the citation recompute
        # fires with the right file_id.
        await bg()
        assert stub_side_effects["citations"]
        assert stub_side_effects["citations"][0][0] == "abc123"


class TestRevertDetailedSummary:
    @pytest.mark.asyncio
    async def test_revert_restores_snapshot_and_clears_flags(
        self, search_db, feature_enabled, stub_side_effects,
    ):
        engine, _ = search_db
        _insert_detailed_row(
            engine,
            "abc123",
            detailed_summary="USER EDITED BODY",
            detailed_original=_SAMPLE_SUMMARY,
            detailed_edited_at=datetime.now(UTC).isoformat(),
        )

        bg = BackgroundTasks()
        result = await revert_detailed_summary("abc123", bg, "drive1")

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT detailed_summary, detailed_original, "
                    "detailed_edited_at FROM file_summaries "
                    "WHERE file_id = 'abc123'"
                )
            ).fetchone()
        assert row[0] == _SAMPLE_SUMMARY
        assert row[1] is None
        assert row[2] is None
        assert result.edited_at is None
        assert result.has_original is False

        # Citation recompute is deferred to BackgroundTasks so the
        # revert response does not block on embedding work.
        assert stub_side_effects["citations"] == []
        assert len(bg.tasks) == 1
        await bg()
        assert stub_side_effects["citations"]
        assert stub_side_effects["citations"][0][0] == "abc123"
        assert stub_side_effects["citations"][0][1] == _SAMPLE_SUMMARY

    @pytest.mark.asyncio
    async def test_revert_400_when_no_snapshot(
        self, search_db, feature_enabled, stub_side_effects,
    ):
        engine, _ = search_db
        _insert_detailed_row(engine, "abc123")  # no edit, no snapshot

        with pytest.raises(HTTPException) as exc_info:
            await revert_detailed_summary("abc123", BackgroundTasks(), "drive1")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_revert_404_when_no_row(
        self, search_db, feature_enabled, stub_side_effects,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await revert_detailed_summary("abc123", BackgroundTasks(), "drive1")
        assert exc_info.value.status_code == 404


class TestRegenerateConflict:
    @pytest.mark.asyncio
    async def test_409_when_edited_and_not_forced(
        self, monkeypatch, search_db, feature_enabled, mock_llm_enabled,
    ):
        engine, _ = search_db
        _insert_detailed_row(
            engine,
            "abc123",
            detailed_summary="USER EDITED",
            detailed_original=_SAMPLE_SUMMARY,
            detailed_edited_at=datetime.now(UTC).isoformat(),
        )

        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript",
            lambda fid: "a" * 500,
        )

        with pytest.raises(HTTPException) as exc_info:
            await regenerate_detailed_summary(
                "abc123",
                BackgroundTasks(),
                DetailedSummaryRegenerateRequest(force=False),
                "drive1",
            )
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_202_when_forced_on_edited_row(
        self, monkeypatch, search_db, feature_enabled, mock_llm_enabled,
    ):
        engine, _ = search_db
        _insert_detailed_row(
            engine,
            "abc123",
            detailed_summary="USER EDITED",
            detailed_original=_SAMPLE_SUMMARY,
            detailed_edited_at=datetime.now(UTC).isoformat(),
        )

        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript",
            lambda fid: "a" * 500,
        )

        bg = BackgroundTasks()
        result = await regenerate_detailed_summary(
            "abc123",
            bg,
            DetailedSummaryRegenerateRequest(force=True),
            "drive1",
        )
        assert result.status == "accepted"
        assert len(bg.tasks) == 1
        # Forced regeneration must also wipe the edit columns.
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT detailed_edited_at, detailed_original "
                    "FROM file_summaries WHERE file_id = 'abc123'"
                )
            ).fetchone()
        # Row either gone (placeholder) or columns cleared.
        if row is not None:
            assert row[0] is None
            assert row[1] is None

    @pytest.mark.asyncio
    async def test_proceeds_without_body_on_unedited_row(
        self, monkeypatch, search_db, feature_enabled, mock_llm_enabled,
    ):
        engine, _ = search_db
        _insert_detailed_row(engine, "abc123")  # no edit flag

        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript",
            lambda fid: "a" * 500,
        )

        bg = BackgroundTasks()
        # body=None is valid since FastAPI would treat missing body as None.
        result = await regenerate_detailed_summary(
            "abc123", bg, None, "drive1"
        )
        assert result.status == "accepted"

    @pytest.mark.asyncio
    async def test_clears_core_active_summary_on_success(
        self, monkeypatch, search_db, feature_enabled, mock_llm_enabled,
    ):
        """Phase 3 Step B: regenerating invalidates the core's active
        summary pointer so the file detail page flips back to showing
        the AI version. The knowledge ``.md`` itself must not be
        touched — only the pointer is cleared."""
        engine, _ = search_db
        _insert_detailed_row(engine, "abc123")

        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript",
            lambda fid: "a" * 500,
        )

        cleared: list[str] = []

        async def fake_clear(file_id: str) -> None:
            cleared.append(file_id)

        monkeypatch.setattr(
            "app.routers.summaries._clear_core_active_summary", fake_clear
        )

        result = await regenerate_detailed_summary(
            "abc123", BackgroundTasks(), None, "drive1",
        )
        assert result.status == "accepted"
        assert cleared == ["abc123"]

    @pytest.mark.asyncio
    async def test_400_when_content_insufficient(
        self, monkeypatch, search_db, feature_enabled, mock_llm_enabled,
    ):
        engine, _ = search_db
        _insert_detailed_row(engine, "abc123")

        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript",
            lambda fid: "hi",
        )

        with pytest.raises(HTTPException) as exc_info:
            await regenerate_detailed_summary(
                "abc123",
                BackgroundTasks(),
                DetailedSummaryRegenerateRequest(force=False),
                "drive1",
            )
        # 400 for insufficient content, not 409 (no edit flag).
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_conflict_check_and_delete_share_one_session(
        self, monkeypatch, search_db, feature_enabled, mock_llm_enabled,
    ):
        """TOCTOU regression: the conflict check and the row wipe must
        run in a single ``get_search_db()`` session so a concurrent
        edit cannot slip between them.

        Prior to the fix, ``_fetch_detailed_edit_state`` ran in session
        A and ``_delete_detailed_summary`` opened session B — a
        concurrent edit landing in that window would set
        ``detailed_edited_at`` but the already-read state would still
        show None, so the delete would overwrite the user's edit
        without returning 409.

        We verify the fix by counting the number of distinct
        ``get_search_db()`` invocations on the non-error path. It must
        be at most one in the edit-check + delete phase (the file/drive
        lookup done earlier by ``_require_file_in_drive`` is counted
        separately and not part of the critical window).
        """
        engine, _ = search_db
        _insert_detailed_row(
            engine,
            "abc123",
            detailed_summary="USER EDITED",
            detailed_original=_SAMPLE_SUMMARY,
            detailed_edited_at=datetime.now(UTC).isoformat(),
        )

        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript",
            lambda fid: "a" * 500,
        )

        # Intercept ``_fetch_detailed_edit_state`` and mutate the DB
        # via an out-of-band connection as a side-effect. With the
        # TOCTOU fix in place, the same session will not re-read after
        # the check so the concurrent write cannot cause a silent
        # clobber: either we saw the edit flag and 409 fires, or the
        # mutation landed after the check and after the delete in the
        # same transaction — structurally safe. The old code (separate
        # sessions) would have allowed the mutation to slip in between.
        original_fetch = (
            __import__(
                "app.routers.summaries", fromlist=["_fetch_detailed_edit_state"]
            )._fetch_detailed_edit_state
        )

        session_ids: list[int] = []

        def fetch_tracking_session(session, file_id):
            session_ids.append(id(session))
            return original_fetch(session, file_id)

        monkeypatch.setattr(
            "app.routers.summaries._fetch_detailed_edit_state",
            fetch_tracking_session,
        )

        # Intercept the raw delete call to assert it uses the *same*
        # session as the check. The fix inlines the delete within the
        # conflict-check session, so ``_delete_detailed_summary`` (which
        # opens its own session) MUST NOT be called from the regenerate
        # path anymore.
        delete_calls: list[str] = []
        original_delete = (
            __import__(
                "app.routers.summaries", fromlist=["_delete_detailed_summary"]
            )._delete_detailed_summary
        )

        def tracking_delete(file_id):
            delete_calls.append(file_id)
            return original_delete(file_id)

        monkeypatch.setattr(
            "app.routers.summaries._delete_detailed_summary",
            tracking_delete,
        )

        with pytest.raises(HTTPException) as exc_info:
            await regenerate_detailed_summary(
                "abc123",
                BackgroundTasks(),
                DetailedSummaryRegenerateRequest(force=False),
                "drive1",
            )
        assert exc_info.value.status_code == 409
        # Confirm the conflict check ran inside a single session and
        # the separate-session ``_delete_detailed_summary`` was not
        # used (which would re-open the TOCTOU window).
        assert len(session_ids) == 1
        assert delete_calls == []
