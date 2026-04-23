"""Tests for the detailed-summary router endpoints.

Covers:

* GET returns the stored row, maps status → available correctly, and
  surfaces a ``reason`` when no row exists.
* POST rejects when feature or LLM is off, when the file type/content
  cannot be summarised, and when a detailed summary is already in flight.
* POST schedules the generation task via BackgroundTasks without
  blocking the response.
* DELETE clears only the detailed_* columns when short/long are present
  and removes the row entirely when it was a detailed-only placeholder.
* GET /*.md returns the Markdown with a download-friendly filename and
  404s for disabled feature / missing row / non-generated status.

Tests use a real SQLite file so the full INSERT/UPDATE path exercises
the same SQL the router will run against production.
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
    delete_detailed_summary_route,
    download_detailed_summary,
    get_detailed_summary_route,
    start_detailed_summary,
)


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    """Real SQLite DB wired into the summaries module.

    Seeds a video file so the indexed_files pre-flight passes. Tests
    insert file_summaries rows as needed; the detailed path tolerates
    both "file_summaries row exists with short/long" and "no row at
    all" start states.
    """
    db_path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        _create_file_summaries_table(conn)
        # Phase 1 citations table — _delete_detailed_summary now also
        # wipes citation rows, so the table must exist even when the
        # test doesn't write any citations of its own.
        _create_detailed_summary_citations_table(conn)
        # Step 2a: _get_detailed_summary reads content from
        # file_insights, so the table must exist for the router tests
        # that seed data via ``_insert_detailed_row``.
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

    # Both the router and the worker module bind get_search_db at
    # import-time, so patch both surfaces.
    monkeypatch.setattr("app.database.get_search_db", _get_search_db)
    monkeypatch.setattr("app.routers.summaries.get_search_db", _get_search_db)
    monkeypatch.setattr("app.workers.summaries.get_search_db", _get_search_db)
    return engine, Session


def _insert_detailed_row(engine, file_id: str, **overrides) -> None:
    """Insert paired ``file_summaries`` + ``file_insights`` rows for tests.

    Step 2a made the reader consume ``file_insights`` for the body /
    versioning metadata. Tests that pre-seed must populate both tables
    or the reader will return None.
    """
    import json as _json
    import secrets as _secrets

    now = datetime.now(UTC).isoformat()
    defaults = {
        "fid": file_id,
        "s": "",
        "l": "",
        "m": "",
        "ct": "",
        "cc": 0,
        "wt": 0,
        "st": "hidden",
        "ca": now,
        "detailed_summary": "## 導入\n本動画は…",
        "detailed_status": "generated",
        "detailed_model": "test-model",
        "detailed_generated_at": now,
        "detailed_context_chars": 500,
        "detailed_was_truncated": 0,
        "detailed_error": None,
    }
    params = {**defaults, **overrides}
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO file_summaries "
                "(file_id, short_summary, long_summary, model, context_type, "
                "context_chars, was_truncated, status, created_at, "
                "detailed_summary, detailed_status, detailed_model, "
                "detailed_generated_at, detailed_context_chars, "
                "detailed_was_truncated, detailed_error) "
                "VALUES (:fid, :s, :l, :m, :ct, :cc, :wt, :st, :ca, "
                ":detailed_summary, :detailed_status, :detailed_model, "
                ":detailed_generated_at, :detailed_context_chars, "
                ":detailed_was_truncated, :detailed_error)"
            ),
            params,
        )
        # Mirror into file_insights so the reader finds the body.
        # Only seed when we actually wrote a body + generated status.
        if (
            params["detailed_summary"] is not None
            and params["detailed_status"] == "generated"
        ):
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
                    "c": params["detailed_summary"],
                    "m": _json.dumps({
                        "model": params["detailed_model"],
                        "context_chars": params["detailed_context_chars"],
                        "was_truncated": bool(params["detailed_was_truncated"])
                        if params["detailed_was_truncated"] is not None
                        else None,
                    }),
                    "ca": params["detailed_generated_at"],
                },
            )


@pytest.fixture()
def feature_enabled(monkeypatch, make_settings):
    """Enable detailed_summaries + configured LLM provider."""
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
def feature_disabled(monkeypatch, make_settings):
    """Feature flag off — endpoints should 400 / 404 depending on route."""
    settings = make_settings(
        features=FeaturesConfig(detailed_summaries="false"),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.summaries.settings", settings)
    monkeypatch.setattr("app.workers.summaries.settings", settings)
    return settings


@pytest.fixture()
def mock_llm_enabled(monkeypatch):
    """Pretend the LLM client is ready so router gates pass."""
    client = MagicMock()
    client.enabled = True
    monkeypatch.setattr(
        "app.routers.summaries.get_llm_client", lambda: client
    )
    return client


class TestGetDetailedSummary:
    """GET /files/{id}/summary/detailed response shaping."""

    @pytest.mark.asyncio
    async def test_returns_unavailable_when_feature_disabled(
        self, search_db, feature_disabled,
    ):
        result = await get_detailed_summary_route("abc123", "drive1")
        assert result.available is False
        assert result.detailed_summary is None

    @pytest.mark.asyncio
    async def test_returns_reason_when_no_row(
        self, monkeypatch, search_db, feature_enabled,
    ):
        # Stub content so classify_detailed_missing_reason reports
        # "not_generated" rather than insufficient_content.
        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript",
            lambda fid: "a" * 500,
        )

        result = await get_detailed_summary_route("abc123", "drive1")
        assert result.available is False
        assert result.reason == "not_generated"
        assert result.status is None

    @pytest.mark.asyncio
    async def test_returns_generated_row(
        self, search_db, feature_enabled,
    ):
        engine, _ = search_db
        _insert_detailed_row(engine, "abc123")

        result = await get_detailed_summary_route("abc123", "drive1")
        assert result.available is True
        assert result.status == "generated"
        assert result.detailed_summary.startswith("## 導入")
        assert result.model == "test-model"

    @pytest.mark.asyncio
    async def test_generating_status_reports_unavailable(
        self, search_db, feature_enabled,
    ):
        engine, _ = search_db
        _insert_detailed_row(
            engine, "abc123",
            detailed_summary=None, detailed_status="generating",
        )

        result = await get_detailed_summary_route("abc123", "drive1")
        assert result.available is False
        assert result.status == "generating"
        # Body intentionally not exposed until generation is complete.
        assert result.detailed_summary is None

    @pytest.mark.asyncio
    async def test_failed_status_exposes_error(
        self, search_db, feature_enabled,
    ):
        engine, _ = search_db
        _insert_detailed_row(
            engine, "abc123",
            detailed_summary=None, detailed_status="failed",
            detailed_error="LLM error: boom",
        )

        result = await get_detailed_summary_route("abc123", "drive1")
        assert result.available is False
        assert result.status == "failed"
        assert result.error == "LLM error: boom"

    @pytest.mark.asyncio
    async def test_404_when_file_in_other_drive(
        self, search_db, feature_enabled,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await get_detailed_summary_route("abc123", "otherdrive")
        assert exc_info.value.status_code == 404


class TestStartDetailedSummary:
    """POST /files/{id}/summary/detailed validation and scheduling."""

    @pytest.mark.asyncio
    async def test_400_when_feature_disabled(
        self, search_db, feature_disabled, mock_llm_enabled,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await start_detailed_summary(
                "abc123", BackgroundTasks(), "drive1",
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_400_when_llm_disabled(
        self, monkeypatch, search_db, feature_enabled,
    ):
        client = MagicMock()
        client.enabled = False
        monkeypatch.setattr(
            "app.routers.summaries.get_llm_client", lambda: client
        )

        with pytest.raises(HTTPException) as exc_info:
            await start_detailed_summary(
                "abc123", BackgroundTasks(), "drive1",
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_400_when_content_insufficient(
        self, monkeypatch, search_db, feature_enabled, mock_llm_enabled,
    ):
        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript",
            lambda fid: "hi",  # below min_context_chars
        )

        with pytest.raises(HTTPException) as exc_info:
            await start_detailed_summary(
                "abc123", BackgroundTasks(), "drive1",
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "insufficient_content"

    @pytest.mark.asyncio
    async def test_409_when_already_present(
        self, monkeypatch, search_db, feature_enabled, mock_llm_enabled,
    ):
        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript",
            lambda fid: "a" * 500,
        )
        engine, _ = search_db
        _insert_detailed_row(engine, "abc123")

        with pytest.raises(HTTPException) as exc_info:
            await start_detailed_summary(
                "abc123", BackgroundTasks(), "drive1",
            )
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_happy_path_schedules_background_task(
        self, monkeypatch, search_db, feature_enabled, mock_llm_enabled,
    ):
        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript",
            lambda fid: "a" * 500,
        )
        bg = BackgroundTasks()

        result = await start_detailed_summary("abc123", bg, "drive1")
        assert result.status == "accepted"
        # FastAPI BackgroundTasks stores callables in .tasks.
        assert len(bg.tasks) == 1


class TestDeleteDetailedSummary:
    """DELETE /files/{id}/summary/detailed behaviour."""

    @pytest.mark.asyncio
    async def test_404_when_no_row(
        self, search_db, feature_enabled,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await delete_detailed_summary_route("abc123", "drive1")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_clears_detailed_but_keeps_row_when_short_long_present(
        self, search_db, feature_enabled,
    ):
        engine, _ = search_db
        now = datetime.now(UTC).isoformat()
        # Insert a row with short/long AND detailed — delete should
        # leave short/long intact and only NULL out detailed_*.
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO file_summaries "
                    "(file_id, short_summary, long_summary, model, "
                    "context_type, context_chars, was_truncated, status, "
                    "created_at, detailed_summary, detailed_status) "
                    "VALUES (:fid, 's', 'l', 'm', 'video', 1, 0, "
                    "'generated', :now, 'detailed body', 'generated')"
                ),
                {"fid": "abc123", "now": now},
            )

        result = await delete_detailed_summary_route("abc123", "drive1")
        assert result.status == "ok"

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT short_summary, detailed_summary, detailed_status "
                    "FROM file_summaries WHERE file_id = :fid"
                ),
                {"fid": "abc123"},
            ).fetchone()
        assert row is not None
        assert row[0] == "s"
        assert row[1] is None
        assert row[2] is None

    @pytest.mark.asyncio
    async def test_removes_row_when_detailed_only(
        self, search_db, feature_enabled,
    ):
        engine, _ = search_db
        now = datetime.now(UTC).isoformat()
        # Placeholder row with empty short/long (as written by
        # _set_detailed_status for a file that only ever had detailed).
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO file_summaries "
                    "(file_id, short_summary, long_summary, model, "
                    "context_type, context_chars, was_truncated, status, "
                    "created_at, detailed_summary, detailed_status) "
                    "VALUES (:fid, '', '', '', '', 0, 0, 'hidden', "
                    ":now, 'body', 'generated')"
                ),
                {"fid": "abc123", "now": now},
            )

        await delete_detailed_summary_route("abc123", "drive1")

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT 1 FROM file_summaries WHERE file_id = :fid"
                ),
                {"fid": "abc123"},
            ).fetchone()
        assert row is None


class TestDownloadDetailedSummary:
    """GET /files/{id}/summary/detailed.md download response."""

    @pytest.mark.asyncio
    async def test_404_when_feature_disabled(
        self, search_db, feature_disabled,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await download_detailed_summary("abc123", "drive1")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_404_when_no_row(
        self, search_db, feature_enabled,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await download_detailed_summary("abc123", "drive1")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_404_when_status_not_generated(
        self, search_db, feature_enabled,
    ):
        engine, _ = search_db
        _insert_detailed_row(
            engine, "abc123",
            detailed_status="generating", detailed_summary=None,
        )

        with pytest.raises(HTTPException) as exc_info:
            await download_detailed_summary("abc123", "drive1")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_markdown_with_filename(
        self, search_db, feature_enabled,
    ):
        engine, _ = search_db
        _insert_detailed_row(
            engine, "abc123",
            detailed_summary="# Hello\n\nBody",
        )

        response = await download_detailed_summary("abc123", "drive1")
        assert response.status_code == 200
        assert response.media_type.startswith("text/markdown")
        body = response.body
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        assert body.startswith("# Hello")
        # The filename stem comes from the seeded IndexedFile.
        disposition = response.headers["content-disposition"]
        assert 'filename="lecture_summary.md"' in disposition
        # RFC 5987 fallback is present so Unicode filenames survive.
        assert "filename*=UTF-8''" in disposition

    @pytest.mark.asyncio
    async def test_unicode_filename_has_ascii_fallback(
        self, monkeypatch, search_db, feature_enabled,
    ):
        engine, _ = search_db

        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE indexed_files SET filename = :name "
                    "WHERE file_id = :fid"
                ),
                {"name": "日本語講義.mp4", "fid": "abc123"},
            )
        _insert_detailed_row(engine, "abc123", detailed_summary="body")

        response = await download_detailed_summary("abc123", "drive1")
        disposition = response.headers["content-disposition"]
        # ASCII fallback must contain no raw non-ASCII bytes.
        ascii_segment = disposition.split(";")[1].strip()
        assert ascii_segment.startswith('filename=')
        ascii_name = ascii_segment[len('filename="'):-1]
        assert ascii_name.isascii()
        # Full Unicode name is carried via the filename* parameter.
        assert "filename*=UTF-8''" in disposition
