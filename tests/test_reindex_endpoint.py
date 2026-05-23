"""Tests for the per-file × per-task reindex endpoint.

Spec: ``docs/superpowers/specs/2026-05-24-intelligence-reindex-controls.md``
§2.1.

The endpoint is ``POST /api/addons/intelligence/files/{file_id}/reindex``
with body ``{"tasks": ["whisper"|"clip"|"text"|"metadata", ...]}``. The
handler must:

* Reset the matching ``*_indexed`` flags on ``IndexedFile`` to ``False``
  and re-enqueue via ``IndexManager.reconcile()`` / ``_enqueue`` so the
  existing incomplete-resume path picks them up.
* Reject unknown task names with 422 and an ``error`` shape that
  matches the other admin endpoints (``{"detail": <message>}``).
* Return 404 when the file does not exist, is soft-deleted, or lives in
  another drive (per the project-wide drive_boundary rule — never leak
  cross-drive file_ids).
* Return 202 + ``"status": "already_queued"`` when the same file × task
  is already in the queue (avoids double-False flag flips and double
  enqueue from Modal connect-clicks).

RED-phase: the handler does not exist yet. Importing the symbol will
raise ImportError, which pytest surfaces as a collection failure on
every test in this file — exactly the signal a TDD GREEN step expects.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub heavy ML deps before importing app modules (mirrors conftest).
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

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.database import Base  # noqa: E402
from app.models import IndexedFile  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    """Real SQLite engine + seed IndexedFile rows for in-drive and
    cross-drive coverage."""
    db_path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine, expire_on_commit=False)

    seed = Session()
    try:
        seed.add_all([
            IndexedFile(
                file_id="active_one",
                drive="drive1",
                filename="movie.mp4",
                file_path="/drives/drive1/movie.mp4",
                file_type="video",
                mime_type="video/mp4",
                file_size=1000,
                active=True,
                metadata_indexed=True,
                clip_indexed=True,
                whisper_indexed=True,
                text_indexed=True,
            ),
            IndexedFile(
                file_id="other_drv",
                drive="other",
                filename="secret.mp4",
                file_path="/drives/other/secret.mp4",
                file_type="video",
                mime_type="video/mp4",
                file_size=999,
                active=True,
                metadata_indexed=True,
                clip_indexed=True,
                whisper_indexed=True,
                text_indexed=True,
            ),
            IndexedFile(
                file_id="soft_dele1",
                drive="drive1",
                filename="gone.mp4",
                file_path="/drives/drive1/gone.mp4",
                file_type="video",
                mime_type="video/mp4",
                file_size=500,
                active=False,
                metadata_indexed=True,
                clip_indexed=True,
                whisper_indexed=True,
                text_indexed=True,
            ),
        ])
        seed.commit()
    finally:
        seed.close()

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
        "app.routers.files.get_search_db", _get_search_db, raising=False,
    )
    return engine, Session


@pytest.fixture()
def fake_index_manager(monkeypatch):
    """An IndexManager double recording enqueue calls and exposing
    queue-membership state so 202 ``already_queued`` is observable."""

    class _FakeManager:
        def __init__(self) -> None:
            # (file_id, task_type) → True when the request resulted in
            # an enqueue. Tests assert against ``enqueued`` to verify
            # tasks_reset was honoured.
            self.enqueued: list[tuple[str, str]] = []
            # Pre-populated by the test to simulate "already in the
            # queue" — handler should observe this and return 202.
            self.queued_pairs: set[tuple[str, str]] = set()

        def is_queued(self, file_id: str, task: str) -> bool:
            return (file_id, task) in self.queued_pairs

        async def enqueue_task_for_file(self, file_id: str, task: str) -> None:
            self.enqueued.append((file_id, task))

    fake = _FakeManager()
    monkeypatch.setattr(
        "app.dependencies.get_index_manager",
        lambda: fake,
    )
    monkeypatch.setattr(
        "app.routers.files.get_index_manager",
        lambda: fake,
        raising=False,
    )
    return fake


# ---------------------------------------------------------------------------
# Happy path: each task resets its flag and triggers enqueue
# ---------------------------------------------------------------------------


class TestReindexHappyPath:
    @pytest.mark.asyncio
    async def test_whisper_resets_flag_and_enqueues(
        self, search_db, fake_index_manager,
    ):
        from app.routers.files import reindex_file

        result = await reindex_file(
            "active_one", body={"tasks": ["whisper"]}, drive="drive1",
        )

        # Response shape
        payload = result if isinstance(result, dict) else result.model_dump()
        assert payload["status"] == "accepted"
        assert payload["file_id"] == "active_one"
        assert payload["tasks_reset"] == ["whisper"]

        # Flag flipped to False in DB
        _, Session = search_db
        with Session() as s:
            row = s.query(IndexedFile).filter_by(file_id="active_one").one()
            assert row.whisper_indexed is False
            # Other flags untouched
            assert row.metadata_indexed is True
            assert row.clip_indexed is True
            assert row.text_indexed is True

        # Enqueue fired
        assert ("active_one", "whisper") in fake_index_manager.enqueued

    @pytest.mark.asyncio
    async def test_clip_resets_flag(self, search_db, fake_index_manager):
        from app.routers.files import reindex_file

        await reindex_file("active_one", body={"tasks": ["clip"]}, drive="drive1")

        _, Session = search_db
        with Session() as s:
            row = s.query(IndexedFile).filter_by(file_id="active_one").one()
            assert row.clip_indexed is False

    @pytest.mark.asyncio
    async def test_text_resets_flag(self, search_db, fake_index_manager):
        from app.routers.files import reindex_file

        await reindex_file("active_one", body={"tasks": ["text"]}, drive="drive1")

        _, Session = search_db
        with Session() as s:
            row = s.query(IndexedFile).filter_by(file_id="active_one").one()
            assert row.text_indexed is False

    @pytest.mark.asyncio
    async def test_metadata_resets_flag(self, search_db, fake_index_manager):
        from app.routers.files import reindex_file

        await reindex_file(
            "active_one", body={"tasks": ["metadata"]}, drive="drive1",
        )

        _, Session = search_db
        with Session() as s:
            row = s.query(IndexedFile).filter_by(file_id="active_one").one()
            assert row.metadata_indexed is False

    @pytest.mark.asyncio
    async def test_multiple_tasks_reset_together(
        self, search_db, fake_index_manager,
    ):
        from app.routers.files import reindex_file

        result = await reindex_file(
            "active_one",
            body={"tasks": ["whisper", "clip"]},
            drive="drive1",
        )
        payload = result if isinstance(result, dict) else result.model_dump()
        assert set(payload["tasks_reset"]) == {"whisper", "clip"}

        _, Session = search_db
        with Session() as s:
            row = s.query(IndexedFile).filter_by(file_id="active_one").one()
            assert row.whisper_indexed is False
            assert row.clip_indexed is False
            # metadata / text untouched
            assert row.metadata_indexed is True
            assert row.text_indexed is True

        enqueued = set(fake_index_manager.enqueued)
        assert ("active_one", "whisper") in enqueued
        assert ("active_one", "clip") in enqueued


# ---------------------------------------------------------------------------
# Validation errors → 422
# ---------------------------------------------------------------------------


class TestReindexValidation:
    @pytest.mark.asyncio
    async def test_unknown_task_returns_422(
        self, search_db, fake_index_manager,
    ):
        from fastapi import HTTPException
        from app.routers.files import reindex_file

        with pytest.raises(HTTPException) as exc:
            await reindex_file(
                "active_one", body={"tasks": ["foo"]}, drive="drive1",
            )

        assert exc.value.status_code == 422
        # Spec §2.1 demands a specific error shape; the message must
        # enumerate the allowed names so the operator can correct the
        # payload without grepping source.
        msg = str(exc.value.detail)
        assert "foo" in msg
        for allowed in ("metadata", "clip", "whisper", "text"):
            assert allowed in msg

    @pytest.mark.asyncio
    async def test_empty_tasks_array_returns_422(
        self, search_db, fake_index_manager,
    ):
        from fastapi import HTTPException
        from app.routers.files import reindex_file

        with pytest.raises(HTTPException) as exc:
            await reindex_file(
                "active_one", body={"tasks": []}, drive="drive1",
            )
        assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_unknown_task_does_not_mutate_db(
        self, search_db, fake_index_manager,
    ):
        """Validation must fire *before* any flag is touched — partial
        progress on a rejected payload would corrupt state."""
        from fastapi import HTTPException
        from app.routers.files import reindex_file

        with pytest.raises(HTTPException):
            await reindex_file(
                "active_one",
                body={"tasks": ["whisper", "bogus"]},
                drive="drive1",
            )

        _, Session = search_db
        with Session() as s:
            row = s.query(IndexedFile).filter_by(file_id="active_one").one()
            assert row.whisper_indexed is True  # untouched
            assert row.metadata_indexed is True
            assert row.clip_indexed is True
            assert row.text_indexed is True

        assert fake_index_manager.enqueued == []


# ---------------------------------------------------------------------------
# 404 paths
# ---------------------------------------------------------------------------


class TestReindexNotFound:
    @pytest.mark.asyncio
    async def test_unknown_file_id_404(self, search_db, fake_index_manager):
        from fastapi import HTTPException
        from app.routers.files import reindex_file

        with pytest.raises(HTTPException) as exc:
            await reindex_file(
                "ghost", body={"tasks": ["whisper"]}, drive="drive1",
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_soft_deleted_file_404(self, search_db, fake_index_manager):
        from fastapi import HTTPException
        from app.routers.files import reindex_file

        with pytest.raises(HTTPException) as exc:
            await reindex_file(
                "soft_dele1", body={"tasks": ["whisper"]}, drive="drive1",
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_cross_drive_file_404(
        self, search_db, fake_index_manager,
    ):
        """Drive-boundary rule: never reveal that ``other_drv`` belongs
        to a drive the caller doesn't see. Must surface as 404 (not 403)."""
        from fastapi import HTTPException
        from app.routers.files import reindex_file

        with pytest.raises(HTTPException) as exc:
            await reindex_file(
                "other_drv", body={"tasks": ["whisper"]}, drive="drive1",
            )
        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# 202 "already_queued" — connect-clicks must be safe
# ---------------------------------------------------------------------------


class TestReindexAlreadyQueued:
    @pytest.mark.asyncio
    async def test_already_queued_returns_202(
        self, search_db, fake_index_manager,
    ):
        """When the file × task pair is already in the queue the handler
        must short-circuit with ``status='already_queued'`` and skip the
        flag flip + enqueue (spec §2.1 重複 enqueue 対策)."""
        from app.routers.files import reindex_file

        fake_index_manager.queued_pairs.add(("active_one", "whisper"))

        result = await reindex_file(
            "active_one", body={"tasks": ["whisper"]}, drive="drive1",
        )

        # FastAPI lets handlers return either a Response or a dict; the
        # 202 status is asserted via the JSONResponse's status_code or a
        # tuple shape — accept either by inspecting the payload directly.
        if hasattr(result, "status_code"):
            assert result.status_code == 202
            import json
            payload = json.loads(result.body)
        elif isinstance(result, dict):
            payload = result
        else:
            payload = result.model_dump()

        assert payload["status"] == "already_queued"

        # Flag NOT flipped, enqueue NOT called for that task.
        _, Session = search_db
        with Session() as s:
            row = s.query(IndexedFile).filter_by(file_id="active_one").one()
            assert row.whisper_indexed is True

        assert ("active_one", "whisper") not in fake_index_manager.enqueued

    @pytest.mark.asyncio
    async def test_mixed_queued_and_fresh_only_resets_fresh(
        self, search_db, fake_index_manager,
    ):
        """Spec doesn't lock the response shape for mixed batches, but
        the invariant is clear: queued tasks must not double-enqueue,
        and fresh tasks must still progress."""
        from app.routers.files import reindex_file

        fake_index_manager.queued_pairs.add(("active_one", "whisper"))

        await reindex_file(
            "active_one",
            body={"tasks": ["whisper", "clip"]},
            drive="drive1",
        )

        _, Session = search_db
        with Session() as s:
            row = s.query(IndexedFile).filter_by(file_id="active_one").one()
            # whisper was already queued — left untouched
            assert row.whisper_indexed is True
            # clip was fresh — reset
            assert row.clip_indexed is False

        enqueued = set(fake_index_manager.enqueued)
        assert ("active_one", "whisper") not in enqueued
        assert ("active_one", "clip") in enqueued
