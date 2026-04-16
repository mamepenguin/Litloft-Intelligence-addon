"""Tests for the summary edit / revert router endpoints.

Covers:

* GET returns ``edited_at`` / ``has_original`` from the stored row
* POST /edit snapshots the AI output on the first edit
* POST /edit preserves the snapshot on subsequent edits
* POST /edit 404s when the row doesn't exist (nothing to edit)
* POST /revert restores the snapshot and clears the edit flags
* POST /revert 400s when no snapshot is present
* Edits are allowed even when ``features.summaries = "false"``
  (disabling the feature only gates new generation)
* ``status = "hidden"`` is preserved across edits

Tests use an in-memory-style SQLite file so the full INSERT/UPDATE
path is exercised end-to-end without touching the real addon DB.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

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

from app.config import FeaturesConfig  # noqa: E402
from app.database import Base, _create_file_summaries_table  # noqa: E402
from app.routers.summaries import (  # noqa: E402
    edit_summary,
    get_summary,
    revert_summary,
)
from app.schemas import SummaryEditRequest  # noqa: E402


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    """Real SQLite DB wired into ``app.routers.summaries.get_search_db``.

    Seeds a single IndexedFile + file_summary row so GET / edit / revert
    have something to act on. Returns the session factory so tests can
    peek at DB state between calls.
    """
    db_path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    # Create ORM tables (incl. indexed_files). file_summaries is raw SQL.
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        _create_file_summaries_table(conn)

    # Seed an indexed_files row so _require_file_in_drive succeeds.
    # Use the ORM so server_default / default values populate correctly —
    # raw INSERTs would need to enumerate every NOT NULL column.
    from app.models import IndexedFile

    seed_session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        seed_session.add(
            IndexedFile(
                file_id="abc123",
                drive="drive1",
                filename="video.mp4",
                file_path="/drives/drive1/video.mp4",
                file_type="video",
                mime_type="video/mp4",
                file_size=1000,
                active=True,
            )
        )
        seed_session.commit()
    finally:
        seed_session.close()

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO file_summaries "
                "(file_id, short_summary, long_summary, model, context_type, "
                "context_chars, was_truncated, status, created_at) "
                "VALUES (:fid, :s, :l, :m, :ct, :cc, :wt, 'generated', :ca)"
            ),
            {
                "fid": "abc123",
                "s": "AI short",
                "l": "AI long content",
                "m": "gemma:e4b",
                "ct": "video",
                "cc": 500,
                "wt": 0,
                "ca": datetime.now(UTC).isoformat(),
            },
        )

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

    # The router reads get_search_db from the module-level import, so we
    # patch both the database module's binding and the router's alias.
    monkeypatch.setattr("app.database.get_search_db", _get_search_db)
    monkeypatch.setattr("app.routers.summaries.get_search_db", _get_search_db)
    return engine, Session


def _row(engine, file_id: str) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT short_summary, long_summary, status, edited_at, "
                "short_original, long_original FROM file_summaries "
                "WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchone()
    if row is None:
        return None
    return {
        "short_summary": row[0],
        "long_summary": row[1],
        "status": row[2],
        "edited_at": row[3],
        "short_original": row[4],
        "long_original": row[5],
    }


@pytest.fixture()
def feature_manual(monkeypatch, make_settings):
    """Enable summaries feature so ``get_summary`` doesn't early-return.

    Edit/revert endpoints build their own response and don't need this
    — GET is the one with a feature gate. Tests that only exercise
    edit/revert skip this fixture.
    """
    settings = make_settings(features=FeaturesConfig(summaries="manual"))
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.summaries.settings", settings)
    return settings


class TestGetSummaryNewFields:
    """GET must surface edited_at and has_original from the stored row."""

    @pytest.mark.asyncio
    async def test_fresh_ai_summary_has_no_edit_flags(
        self, search_db, feature_manual
    ):
        result = await get_summary("abc123", "drive1")
        assert result.available is True
        assert result.edited_at is None
        assert result.has_original is False

    @pytest.mark.asyncio
    async def test_edited_summary_exposes_flags(
        self, search_db, feature_manual
    ):
        engine, _ = search_db
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE file_summaries SET "
                    "short_summary = 'user short', long_summary = 'user long', "
                    "short_original = 'AI short', long_original = 'AI long', "
                    "edited_at = :now WHERE file_id = :fid"
                ),
                {
                    "now": datetime.now(UTC).isoformat(),
                    "fid": "abc123",
                },
            )
        result = await get_summary("abc123", "drive1")
        assert result.short_summary == "user short"
        assert result.has_original is True
        assert result.edited_at is not None


class TestEditSummary:
    """First edit snapshots the AI output; subsequent edits preserve it."""

    @pytest.mark.asyncio
    async def test_first_edit_snapshots_originals(self, search_db):
        engine, _ = search_db
        body = SummaryEditRequest(
            short_summary="human short",
            long_summary="human long",
        )
        await edit_summary("abc123", body, "drive1")

        row = _row(engine, "abc123")
        assert row is not None
        assert row["short_summary"] == "human short"
        assert row["long_summary"] == "human long"
        assert row["short_original"] == "AI short"
        assert row["long_original"] == "AI long content"
        assert row["edited_at"] is not None

    @pytest.mark.asyncio
    async def test_second_edit_preserves_ai_snapshot(self, search_db):
        engine, _ = search_db
        await edit_summary(
            "abc123",
            SummaryEditRequest(short_summary="v1 s", long_summary="v1 l"),
            "drive1",
        )
        await edit_summary(
            "abc123",
            SummaryEditRequest(short_summary="v2 s", long_summary="v2 l"),
            "drive1",
        )
        row = _row(engine, "abc123")
        assert row["short_summary"] == "v2 s"
        assert row["long_summary"] == "v2 l"
        # Snapshot still points at the AI output, not the v1 edit.
        assert row["short_original"] == "AI short"
        assert row["long_original"] == "AI long content"

    @pytest.mark.asyncio
    async def test_edit_404_when_row_missing(self, search_db):
        with pytest.raises(HTTPException) as excinfo:
            await edit_summary(
                "missing-id",
                SummaryEditRequest(short_summary="s", long_summary="l"),
                "drive1",
            )
        # The file isn't in indexed_files either, so _require_file_in_drive
        # 404s first. That's still the expected shape for the frontend.
        assert excinfo.value.status_code == 404

    @pytest.mark.asyncio
    async def test_edit_404_when_only_summary_row_missing(
        self, search_db
    ):
        """indexed_files row present, file_summaries row absent — 404."""
        engine, _ = search_db
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM file_summaries WHERE file_id = :fid"),
                {"fid": "abc123"},
            )
        with pytest.raises(HTTPException) as excinfo:
            await edit_summary(
                "abc123",
                SummaryEditRequest(short_summary="s", long_summary="l"),
                "drive1",
            )
        assert excinfo.value.status_code == 404

    @pytest.mark.asyncio
    async def test_edit_allowed_when_feature_disabled(
        self, search_db, monkeypatch, make_settings
    ):
        """features.summaries = 'false' must still allow edits.

        Disabling the feature only gates new generation; stored rows
        remain editable so data curation doesn't require re-enabling
        the LLM path.
        """
        settings = make_settings(features=FeaturesConfig(summaries="false"))
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.routers.summaries.settings", settings)

        engine, _ = search_db
        await edit_summary(
            "abc123",
            SummaryEditRequest(short_summary="ok", long_summary="ok long"),
            "drive1",
        )
        row = _row(engine, "abc123")
        assert row["short_summary"] == "ok"
        assert row["edited_at"] is not None

    @pytest.mark.asyncio
    async def test_edit_returns_updated_response(self, search_db):
        """The edit response should hydrate the UI without a second GET."""
        result = await edit_summary(
            "abc123",
            SummaryEditRequest(short_summary="ret s", long_summary="ret l"),
            "drive1",
        )
        assert result.short_summary == "ret s"
        assert result.long_summary == "ret l"
        assert result.has_original is True
        assert result.edited_at is not None


class TestRevertSummary:
    @pytest.mark.asyncio
    async def test_revert_restores_ai_and_clears_flags(self, search_db):
        engine, _ = search_db
        await edit_summary(
            "abc123",
            SummaryEditRequest(short_summary="edit s", long_summary="edit l"),
            "drive1",
        )
        result = await revert_summary("abc123", "drive1")

        assert result.short_summary == "AI short"
        assert result.long_summary == "AI long content"
        assert result.edited_at is None
        assert result.has_original is False

        row = _row(engine, "abc123")
        assert row["short_original"] is None
        assert row["long_original"] is None
        assert row["edited_at"] is None

    @pytest.mark.asyncio
    async def test_revert_400_when_no_snapshot(self, search_db):
        with pytest.raises(HTTPException) as excinfo:
            await revert_summary("abc123", "drive1")
        assert excinfo.value.status_code == 400

    @pytest.mark.asyncio
    async def test_revert_404_when_row_missing(self, search_db):
        engine, _ = search_db
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM file_summaries WHERE file_id = :fid"),
                {"fid": "abc123"},
            )
        with pytest.raises(HTTPException) as excinfo:
            await revert_summary("abc123", "drive1")
        assert excinfo.value.status_code == 404
