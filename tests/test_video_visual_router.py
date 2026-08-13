"""Router endpoint tests for the video visual index feature.

``app.routers.video_visual`` exposes:

* ``GET  /files/{file_id}/visual-index``          — read state
* ``POST /files/{file_id}/visual-index/generate``  — manual trigger
* ``POST /files/{file_id}/visual-index/retry``     — retry failed scenes

Access gates (design doc §11, mirrors ``app.routers.vision``):

1. ``features.video_visual_index == "false"`` → 404 on every route
2. ``llm.vision_model`` empty → 404 on generate/retry
3. per-drive policy OFF → 404 on every route
4. cross-drive access / not a native video → 404
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

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

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.config import FeaturesConfig, LLMConfig  # noqa: E402
from app.database import Base  # noqa: E402
from app.models import IndexedFile, VideoVisualRun, VideoVisualScene  # noqa: E402

pytest.importorskip(
    "app.routers.video_visual",
    reason="video_visual router not yet implemented",
)

from app.routers.video_visual import (  # noqa: E402
    generate_visual_index,
    get_visual_index,
    retry_visual_index,
)


@pytest.fixture()
def search_db(monkeypatch, tmp_path):
    db_path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    seed = Session()
    try:
        seed.add_all([
            IndexedFile(
                file_id="vid-abc", drive="family", filename="clip.mp4",
                file_path="/drives/family/clip.mp4", file_type="video",
                mime_type="video/mp4", file_size=1000, active=True,
            ),
            IndexedFile(
                file_id="img-abc", drive="family", filename="cat.jpg",
                file_path="/drives/family/cat.jpg", file_type="image",
                mime_type="image/jpeg", file_size=1000, active=True,
            ),
        ])
        seed.commit()
    finally:
        seed.close()

    @contextmanager
    def _get_search_db_read():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr(
        "app.routers.video_visual.get_search_db_read", _get_search_db_read
    )
    monkeypatch.setattr(
        "app.workers.video_visual._load_candidates",
        lambda file_id: ([], None),
    )
    return engine, Session


@pytest.fixture()
def feature_manual(monkeypatch, make_settings):
    settings = make_settings(
        features=FeaturesConfig(video_visual_index="manual"),  # type: ignore[call-arg]
        llm=LLMConfig(
            provider="openai_compatible", base_url="http://test",
            model="gemma2:27b", vision_model="llava:13b",
        ),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.video_visual.settings", settings)
    monkeypatch.setattr(
        "app.routers.video_visual.is_feature_enabled",
        AsyncMock(return_value=True),
    )
    return settings


@pytest.fixture()
def feature_off(monkeypatch, make_settings):
    settings = make_settings(
        features=FeaturesConfig(video_visual_index="false"),  # type: ignore[call-arg]
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.video_visual.settings", settings)
    return settings


@pytest.fixture()
def stub_worker(monkeypatch):
    worker = MagicMock()
    worker.enqueue = AsyncMock(return_value={"accepted": True, "run_id": "vvr_1"})
    worker.retry = AsyncMock(return_value={"accepted": True, "run_id": "vvr_1", "reset_count": 1})

    async def _get():
        return worker

    monkeypatch.setattr("app.routers.video_visual.get_video_visual_worker", _get)
    return worker


# ---------------------------------------------------------------------------
# GET /files/{file_id}/visual-index
# ---------------------------------------------------------------------------


class TestGetVisualIndex:
    @pytest.mark.asyncio
    async def test_feature_off_returns_404(self, feature_off, search_db):
        with pytest.raises(HTTPException) as exc:
            await get_visual_index(file_id="vid-abc", drive="family")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_policy_off_returns_404(self, feature_manual, search_db, monkeypatch):
        monkeypatch.setattr(
            "app.routers.video_visual.is_feature_enabled",
            AsyncMock(return_value=False),
        )
        with pytest.raises(HTTPException) as exc:
            await get_visual_index(file_id="vid-abc", drive="family")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_unknown_file_returns_404(self, feature_manual, search_db):
        with pytest.raises(HTTPException) as exc:
            await get_visual_index(file_id="ghost", drive="family")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_cross_drive_returns_404(self, feature_manual, search_db):
        with pytest.raises(HTTPException) as exc:
            await get_visual_index(file_id="vid-abc", drive="other")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_non_video_returns_ineligible(self, feature_manual, search_db):
        result = await get_visual_index(file_id="img-abc", drive="family")
        assert result.eligible is False

    @pytest.mark.asyncio
    async def test_never_generated_has_no_active_or_staged_run(
        self, feature_manual, search_db,
    ):
        result = await get_visual_index(file_id="vid-abc", drive="family")
        assert result.eligible is True
        assert result.available is True
        assert result.active_run is None
        assert result.staged_run is None
        assert result.scenes == []

    @pytest.mark.asyncio
    async def test_active_run_returns_its_scenes(self, feature_manual, search_db):
        _, Session = search_db
        with Session() as s:
            s.add(
                VideoVisualRun(
                    id="vvr_active", file_id="vid-abc", status="succeeded", is_active=True,
                    requested_by="manual", priority=100, vision_model="llava:13b",
                    pipeline_version=1, candidate_fingerprint="fp",
                    selected_count=1, completed_count=1, succeeded_count=1, failed_count=0,
                )
            )
            s.add(
                VideoVisualScene(
                    run_id="vvr_active", ordering=0, clip_embedding_id="c0",
                    start_time=5.0, status="succeeded",
                    visual_description="A cat.", visible_text=None, scene_type="object",
                )
            )
            s.commit()

        result = await get_visual_index(file_id="vid-abc", drive="family")
        assert result.active_run is not None
        assert result.active_run.status == "succeeded"
        assert len(result.scenes) == 1
        assert result.scenes[0].visual_description == "A cat."
        assert result.staged_run is None

    @pytest.mark.asyncio
    async def test_staged_run_surfaces_alongside_active(self, feature_manual, search_db):
        _, Session = search_db
        with Session() as s:
            s.add(
                VideoVisualRun(
                    id="vvr_active", file_id="vid-abc", status="succeeded", is_active=True,
                    requested_by="manual", priority=100, vision_model="llava:13b",
                    pipeline_version=1, candidate_fingerprint="fp-old",
                    selected_count=1, completed_count=1, succeeded_count=1, failed_count=0,
                )
            )
            s.add(
                VideoVisualRun(
                    id="vvr_staged", file_id="vid-abc", status="running", is_active=False,
                    requested_by="manual", priority=100, vision_model="llava:13b",
                    pipeline_version=1, candidate_fingerprint="fp-new",
                    selected_count=4, completed_count=2, succeeded_count=2, failed_count=0,
                )
            )
            s.commit()

        result = await get_visual_index(file_id="vid-abc", drive="family")
        assert result.active_run.run_id == "vvr_active"
        assert result.staged_run.run_id == "vvr_staged"
        assert result.staged_run.status == "running"


# ---------------------------------------------------------------------------
# POST /files/{file_id}/visual-index/generate
# ---------------------------------------------------------------------------


class TestGenerateVisualIndex:
    @pytest.mark.asyncio
    async def test_feature_off_returns_404(self, feature_off, search_db, stub_worker):
        with pytest.raises(HTTPException) as exc:
            await generate_visual_index(file_id="vid-abc", drive="family")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_non_video_returns_404(self, feature_manual, search_db, stub_worker):
        with pytest.raises(HTTPException) as exc:
            await generate_visual_index(file_id="img-abc", drive="family")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_accepted_enqueues_manual(self, feature_manual, search_db, stub_worker):
        result = await generate_visual_index(file_id="vid-abc", drive="family")
        assert result["status"] == "accepted"
        stub_worker.enqueue.assert_awaited_once_with("vid-abc", requested_by="manual")

    @pytest.mark.asyncio
    async def test_waiting_clip_returns_409(self, feature_manual, search_db, stub_worker):
        stub_worker.enqueue = AsyncMock(
            return_value={"accepted": False, "reason": "waiting_clip"}
        )
        with pytest.raises(HTTPException) as exc:
            await generate_visual_index(file_id="vid-abc", drive="family")
        assert exc.value.status_code == 409

    @pytest.mark.asyncio
    async def test_unsupported_sticky_returns_409(self, feature_manual, search_db, stub_worker):
        stub_worker.enqueue = AsyncMock(
            return_value={"accepted": False, "reason": "unsupported_sticky"}
        )
        with pytest.raises(HTTPException) as exc:
            await generate_visual_index(file_id="vid-abc", drive="family")
        assert exc.value.status_code == 409

    @pytest.mark.asyncio
    async def test_already_queued_returns_200_status(self, feature_manual, search_db, stub_worker):
        stub_worker.enqueue = AsyncMock(
            return_value={"accepted": False, "reason": "already_queued"}
        )
        result = await generate_visual_index(file_id="vid-abc", drive="family")
        assert result["status"] == "already_queued"


# ---------------------------------------------------------------------------
# POST /files/{file_id}/visual-index/retry
# ---------------------------------------------------------------------------


class TestRetryVisualIndex:
    @pytest.mark.asyncio
    async def test_no_failed_scenes_returns_404(self, feature_manual, search_db, stub_worker):
        stub_worker.retry = AsyncMock(return_value={"accepted": False, "reason": "no_run"})
        with pytest.raises(HTTPException) as exc:
            await retry_visual_index(file_id="vid-abc", drive="family")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_accepted_returns_reset_count(self, feature_manual, search_db, stub_worker):
        result = await retry_visual_index(file_id="vid-abc", drive="family")
        assert result["status"] == "accepted"
        assert result["reset_count"] == 1
        stub_worker.retry.assert_awaited_once_with("vid-abc")
