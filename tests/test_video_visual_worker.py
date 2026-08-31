"""Tests for VideoVisualWorker: enqueue gating, retry, activation rules,
and startup recovery.

Design doc "Video Visual Index" §5.3 (activation rules), §9 (queue /
restart), §12 (failure handling). Mirrors the fixture pattern used by
``test_vision_worker.py`` — a real (vec-less) SQLite DB for ORM tables,
with the LLM / frame-cache / embedding paths stubbed out.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

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

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.config import FeaturesConfig, LLMConfig  # noqa: E402
from app.llm import (  # noqa: E402
    FAILURE_IMAGE_REJECTED,
    FAILURE_TOKEN_BUDGET,
    JsonGeneration,
)
from app.database import Base  # noqa: E402
from app.models import (  # noqa: E402
    Embedding,
    IndexedFile,
    VideoVisualRun,
    VideoVisualScene,
)

pytest.importorskip(
    "app.workers.video_visual",
    reason="VideoVisualWorker not yet implemented",
)

from app.workers.video_visual import (  # noqa: E402
    VideoVisualWorker,
    _load_frame_bytes,
    _read_or_extract_frame,
    _validate_scene_output,
    recover_on_startup,
)


@pytest.fixture()
def search_db(monkeypatch, tmp_path):
    """Real SQLite with all ORM tables (indexed_files, video_visual_*, embeddings)."""
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
                file_id="vid-ok",
                drive="family",
                filename="clip.mp4",
                file_path="/drives/family/clip.mp4",
                file_type="video",
                mime_type="video/mp4",
                file_size=200_000,
                duration=600.0,
                active=True,
            ),
            IndexedFile(
                file_id="vid-no-clip",
                drive="family",
                filename="noclip.mp4",
                file_path="/drives/family/noclip.mp4",
                file_type="video",
                mime_type="video/mp4",
                file_size=200_000,
                duration=600.0,
                active=True,
            ),
            IndexedFile(
                file_id="img-not-video",
                drive="family",
                filename="cat.jpg",
                file_path="/drives/family/cat.jpg",
                file_type="image",
                mime_type="image/jpeg",
                file_size=1000,
                active=True,
            ),
        ])
        seed.commit()
        # Give vid-ok a scene-CLIP candidate pool (embeddings metadata
        # only — no real vec_clip table needed for these tests).
        seed.add(
            Embedding(
                id="clip_vid-ok_1",
                file_id="vid-ok",
                embedding_type="clip",
                timestamp_start=5.0,
                vector_table="vec_clip",
            )
        )
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

    @contextmanager
    def _get_search_db_read():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("app.database.get_search_db", _get_search_db)
    monkeypatch.setattr("app.workers.video_visual.get_search_db", _get_search_db)
    monkeypatch.setattr(
        "app.workers.video_visual.get_search_db_read", _get_search_db_read
    )
    return engine, Session


@pytest.fixture()
def feature_manual(monkeypatch, make_settings):
    settings = make_settings(
        features=FeaturesConfig(video_visual_index="manual"),  # type: ignore[call-arg]
        llm=LLMConfig(
            provider="openai_compatible",
            base_url="http://test",
            model="gemma2:27b",
            vision_model="llava:13b",
        ),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.workers.video_visual.settings", settings)
    return settings


@pytest.fixture()
def policy_allow_all(monkeypatch):
    policy = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "app.workers.video_visual.is_file_feature_enabled",
        policy,
        raising=False,
    )
    return policy


@pytest.fixture()
def no_emit(monkeypatch):
    """Silence the best-effort core WS bridge for every test."""
    monkeypatch.setattr(
        "app.workers.video_visual.emit_video_visual_event",
        AsyncMock(return_value=None),
    )


# ---------------------------------------------------------------------------
# _validate_scene_output
# ---------------------------------------------------------------------------


class TestValidateSceneOutput:
    def test_valid_full_object(self):
        result = _validate_scene_output(
            {"scene_label": "Cat on a table", "visible_text": "MENU", "scene_type": "object"}
        )
        assert result == {
            "scene_label": "Cat on a table",
            "visible_text": "MENU",
            "scene_type": "object",
        }

    def test_blank_label_is_invalid(self):
        assert _validate_scene_output({"scene_label": "   "}) is None

    def test_missing_label_is_invalid(self):
        assert _validate_scene_output({"visible_text": "x"}) is None

    def test_legacy_visual_description_does_not_satisfy_new_contract(self):
        assert _validate_scene_output({"visual_description": "A cat on a table."}) is None

    def test_non_dict_is_invalid(self):
        assert _validate_scene_output("not a dict") is None
        assert _validate_scene_output(None) is None
        assert _validate_scene_output(["a", "list"]) is None

    def test_unknown_scene_type_is_dropped_not_rejected(self):
        result = _validate_scene_output(
            {"scene_label": "x", "scene_type": "spaceship"}
        )
        assert result is not None
        assert result["scene_type"] is None

    def test_missing_visible_text_defaults_to_empty(self):
        result = _validate_scene_output({"scene_label": "x"})
        assert result["visible_text"] == ""

    def test_oversized_fields_are_truncated(self):
        from app.workers.video_visual import (
            MAX_SCENE_LABEL_CHARS,
            MAX_VISIBLE_TEXT_CHARS,
        )

        result = _validate_scene_output(
            {"scene_label": "x" * 10_000, "visible_text": "y" * 10_000}
        )
        assert len(result["scene_label"]) == MAX_SCENE_LABEL_CHARS
        assert len(result["visible_text"]) == MAX_VISIBLE_TEXT_CHARS


# ---------------------------------------------------------------------------
# _process_scene scene-label persistence
# ---------------------------------------------------------------------------


class TestProcessSceneLabel:
    @pytest.mark.asyncio
    async def test_persists_and_embeds_scene_label(
        self, search_db, feature_manual, monkeypatch,
    ):
        _, Session = search_db
        with Session() as s:
            run = VideoVisualRun(
                id="vvr_label", file_id="vid-ok", status="running",
                is_active=False, requested_by="manual", priority=100,
                vision_model="llava:13b", pipeline_version=2,
                candidate_fingerprint="fp", selected_count=1,
            )
            scene = VideoVisualScene(
                run_id="vvr_label", ordering=0, clip_embedding_id="c0",
                start_time=5.0, status="pending",
            )
            s.add_all([run, scene])
            s.commit()
            scene_id = scene.id

        monkeypatch.setattr(
            "app.workers.video_visual.config.validate_file_path", lambda path: True,
        )
        monkeypatch.setattr(
            "app.workers.video_visual._load_frame_bytes",
            AsyncMock(return_value=b"raw-frame"),
        )
        monkeypatch.setattr(
            "app.workers.vision._preprocess_image",
            lambda data, mime: (b"processed-frame", "image/jpeg"),
        )
        monkeypatch.setattr(
            "app.workers.video_visual._select_transcript_excerpt",
            lambda *args: "nearby transcript",
        )
        embed_scene = MagicMock()
        monkeypatch.setattr("app.workers.video_visual._embed_scene", embed_scene)

        llm = MagicMock()
        llm.generate_video_scene_json = AsyncMock(return_value=JsonGeneration(
            {
                "scene_label": "Chicken marinade added",
                "visible_text": "",
                "scene_type": "demonstration",
            },
            None,
        ))

        outcome = await VideoVisualWorker(llm)._process_scene(
            scene_id, "vvr_label", "vid-ok", "/drives/family/clip.mp4", "clip.mp4"
        )

        assert outcome == "succeeded"
        with Session() as s:
            stored = s.query(VideoVisualScene).filter_by(id=scene_id).one()
            assert stored.scene_label == "Chicken marinade added"
            assert stored.visual_description is None
            assert stored.transcript_excerpt == "nearby transcript"
        embed_scene.assert_called_once_with(
            "vid-ok", scene_id, "Chicken marinade added", "", 5.0, None,
        )

    @pytest.mark.asyncio
    async def test_a_truncated_scene_reports_the_budget_not_the_prompt(
        self, search_db, feature_manual, policy_allow_all, monkeypatch,
    ):
        """The remedy differs, so the error class must too.

        Truncation can leave a syntactically complete object that the
        scene validator still rejects for missing scene_label. Calling
        that malformed output points the operator at the prompt; the
        fix is the token budget.
        """
        _, Session = search_db
        with Session() as s:
            run = VideoVisualRun(
                id="vvr_trunc", file_id="vid-ok", status="running",
                is_active=False, requested_by="manual", priority=100,
                vision_model="llava:13b", pipeline_version=2,
                candidate_fingerprint="fp", selected_count=1,
            )
            scene = VideoVisualScene(
                run_id="vvr_trunc", ordering=0, clip_embedding_id="c0",
                start_time=5.0, status="pending",
            )
            s.add_all([run, scene])
            s.commit()
            scene_id = scene.id

        monkeypatch.setattr(
            "app.workers.video_visual.config.validate_file_path", lambda path: True,
        )
        monkeypatch.setattr(
            "app.workers.video_visual._load_frame_bytes",
            AsyncMock(return_value=b"raw-frame"),
        )
        monkeypatch.setattr(
            "app.workers.vision._preprocess_image",
            lambda data, mime: (b"processed-frame", "image/jpeg"),
        )
        monkeypatch.setattr(
            "app.workers.video_visual._select_transcript_excerpt",
            lambda *args: "",
        )

        llm = MagicMock()
        # Parses, but carries no scene_label — the validator rejects it.
        llm.generate_video_scene_json = AsyncMock(
            return_value=JsonGeneration(
                {"visible_text": "partial"}, FAILURE_TOKEN_BUDGET
            )
        )

        outcome = await VideoVisualWorker(llm)._process_scene(
            scene_id, "vvr_trunc", "vid-ok", "/drives/family/clip.mp4", "clip.mp4"
        )

        assert outcome != "succeeded"
        with Session() as s:
            stored = s.query(VideoVisualScene).filter_by(id=scene_id).one()
            assert stored.error_class == "TokenBudget"

    @pytest.mark.asyncio
    async def test_a_rejected_frame_does_not_end_the_whole_run(
        self, search_db, feature_manual, policy_allow_all, monkeypatch,
    ):
        """Only a probed verdict about the model ends a run.

        One frame the provider would not read says nothing about the
        remaining scenes.
        """
        _, Session = search_db
        with Session() as s:
            run = VideoVisualRun(
                id="vvr_rej", file_id="vid-ok", status="running",
                is_active=False, requested_by="manual", priority=100,
                vision_model="llava:13b", pipeline_version=2,
                candidate_fingerprint="fp", selected_count=1,
            )
            scene = VideoVisualScene(
                run_id="vvr_rej", ordering=0, clip_embedding_id="c0",
                start_time=5.0, status="pending",
            )
            s.add_all([run, scene])
            s.commit()
            scene_id = scene.id

        monkeypatch.setattr(
            "app.workers.video_visual.config.validate_file_path", lambda path: True,
        )
        monkeypatch.setattr(
            "app.workers.video_visual._load_frame_bytes",
            AsyncMock(return_value=b"raw-frame"),
        )
        monkeypatch.setattr(
            "app.workers.vision._preprocess_image",
            lambda data, mime: (b"processed-frame", "image/jpeg"),
        )
        monkeypatch.setattr(
            "app.workers.video_visual._select_transcript_excerpt",
            lambda *args: "",
        )

        llm = MagicMock()
        llm.generate_video_scene_json = AsyncMock(
            return_value=JsonGeneration(None, FAILURE_IMAGE_REJECTED)
        )

        outcome = await VideoVisualWorker(llm)._process_scene(
            scene_id, "vvr_rej", "vid-ok", "/drives/family/clip.mp4", "clip.mp4"
        )

        assert outcome != "unsupported"
        with Session() as s:
            stored = s.query(VideoVisualScene).filter_by(id=scene_id).one()
            assert stored.error_class == "ImageRejected"



# ---------------------------------------------------------------------------
# enqueue()
# ---------------------------------------------------------------------------


class TestEnqueue:
    @pytest.mark.asyncio
    async def test_eligible_video_with_clip_is_accepted(
        self, search_db, feature_manual, policy_allow_all,
    ):
        worker = VideoVisualWorker(MagicMock())
        result = await worker.enqueue("vid-ok", requested_by="manual")
        assert result["accepted"] is True
        assert result["reason"] == "queued"
        assert result.get("run_id")

        engine, Session = search_db
        with Session() as s:
            run = s.query(VideoVisualRun).filter_by(id=result["run_id"]).first()
            assert run is not None
            assert run.status == "queued"
            assert run.priority == 100  # manual
            assert run.requested_by == "manual"
            assert run.pipeline_version == 2
        policy_allow_all.assert_awaited_once_with(
            "vid-ok", "video_visual_index", default_on_failure=False,
        )

    @pytest.mark.asyncio
    async def test_on_index_uses_priority_zero(
        self, search_db, feature_manual, policy_allow_all,
    ):
        worker = VideoVisualWorker(MagicMock())
        result = await worker.enqueue("vid-ok", requested_by="on_index")
        assert result["accepted"] is True
        _, Session = search_db
        with Session() as s:
            run = s.query(VideoVisualRun).filter_by(id=result["run_id"]).first()
            assert run.priority == 0

    @pytest.mark.asyncio
    async def test_non_video_mime_is_rejected(
        self, search_db, feature_manual, policy_allow_all,
    ):
        worker = VideoVisualWorker(MagicMock())
        result = await worker.enqueue("img-not-video")
        assert result["accepted"] is False
        assert result["reason"] == "not_eligible"

    @pytest.mark.asyncio
    async def test_missing_file_is_rejected(
        self, search_db, feature_manual, policy_allow_all,
    ):
        worker = VideoVisualWorker(MagicMock())
        result = await worker.enqueue("ghost")
        assert result["accepted"] is False
        assert result["reason"] == "not_found"

    @pytest.mark.asyncio
    async def test_feature_disabled_is_rejected(
        self, search_db, monkeypatch, make_settings, policy_allow_all,
    ):
        settings = make_settings(
            features=FeaturesConfig(video_visual_index="false"),  # type: ignore[call-arg]
            llm=LLMConfig(vision_model="llava:13b"),
        )
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.video_visual.settings", settings)

        worker = VideoVisualWorker(MagicMock())
        result = await worker.enqueue("vid-ok")
        assert result["accepted"] is False
        assert result["reason"] == "disabled"

    @pytest.mark.asyncio
    async def test_missing_vision_model_is_rejected(
        self, search_db, monkeypatch, make_settings, policy_allow_all,
    ):
        settings = make_settings(
            features=FeaturesConfig(video_visual_index="manual"),  # type: ignore[call-arg]
            llm=LLMConfig(provider="openai_compatible", base_url="http://x", model="m", vision_model=""),
        )
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.video_visual.settings", settings)

        worker = VideoVisualWorker(MagicMock())
        result = await worker.enqueue("vid-ok")
        assert result["accepted"] is False
        assert result["reason"] == "disabled"

    @pytest.mark.asyncio
    async def test_drive_policy_off_is_rejected(
        self, search_db, feature_manual, monkeypatch,
    ):
        monkeypatch.setattr(
            "app.workers.video_visual.is_file_feature_enabled",
            AsyncMock(return_value=False),
            raising=False,
        )
        worker = VideoVisualWorker(MagicMock())
        result = await worker.enqueue("vid-ok")
        assert result["accepted"] is False
        assert result["reason"] == "disabled"

    @pytest.mark.asyncio
    async def test_no_clip_candidates_rejected_and_manual_prioritizes(
        self, search_db, feature_manual, policy_allow_all, monkeypatch,
    ):
        index_manager = MagicMock()
        index_manager.prioritize = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "app.dependencies.get_index_manager", lambda: index_manager,
        )
        worker = VideoVisualWorker(MagicMock())
        result = await worker.enqueue("vid-no-clip", requested_by="manual")
        assert result["accepted"] is False
        assert result["reason"] == "waiting_clip"
        index_manager.prioritize.assert_awaited_once_with("vid-no-clip")

    @pytest.mark.asyncio
    async def test_no_clip_candidates_on_index_does_not_prioritize(
        self, search_db, feature_manual, policy_allow_all, monkeypatch,
    ):
        index_manager = MagicMock()
        monkeypatch.setattr(
            "app.dependencies.get_index_manager", lambda: index_manager,
        )
        worker = VideoVisualWorker(MagicMock())
        result = await worker.enqueue("vid-no-clip", requested_by="on_index")
        assert result["accepted"] is False
        assert result["reason"] == "waiting_clip"
        index_manager.prioritize.assert_not_called()

    @pytest.mark.asyncio
    async def test_already_in_flight_is_rejected(
        self, search_db, feature_manual, policy_allow_all,
    ):
        worker = VideoVisualWorker(MagicMock())
        first = await worker.enqueue("vid-ok")
        assert first["accepted"] is True
        second = await worker.enqueue("vid-ok")
        assert second["accepted"] is False
        assert second["reason"] == "already_queued"

    @pytest.mark.asyncio
    async def test_unsupported_sticky_for_same_model(
        self, search_db, feature_manual, policy_allow_all,
    ):
        _, Session = search_db
        with Session() as s:
            s.add(
                VideoVisualRun(
                    id="vvr_prior",
                    file_id="vid-ok",
                    status="failed",
                    is_active=False,
                    requested_by="manual",
                    priority=100,
                    vision_model="llava:13b",
                    pipeline_version=1,
                    candidate_fingerprint="fp",
                    error_class="Unsupported",
                )
            )
            s.commit()

        worker = VideoVisualWorker(MagicMock())
        result = await worker.enqueue("vid-ok")
        assert result["accepted"] is False
        assert result["reason"] == "unsupported_sticky"

    @pytest.mark.asyncio
    async def test_unsupported_sticky_clears_on_model_change(
        self, search_db, feature_manual, policy_allow_all,
    ):
        _, Session = search_db
        with Session() as s:
            s.add(
                VideoVisualRun(
                    id="vvr_prior",
                    file_id="vid-ok",
                    status="failed",
                    is_active=False,
                    requested_by="manual",
                    priority=100,
                    vision_model="old-model",
                    pipeline_version=1,
                    candidate_fingerprint="fp",
                    error_class="Unsupported",
                )
            )
            s.commit()

        worker = VideoVisualWorker(MagicMock())  # configured model is llava:13b
        result = await worker.enqueue("vid-ok")
        assert result["accepted"] is True

    @pytest.mark.asyncio
    async def test_on_index_skips_when_fingerprint_unchanged(
        self, search_db, feature_manual, policy_allow_all, monkeypatch,
    ):
        from app.workers.video_visual import PIPELINE_VERSION

        _, Session = search_db
        with Session() as s:
            s.add(
                VideoVisualRun(
                    id="vvr_active",
                    file_id="vid-ok",
                    status="succeeded",
                    is_active=True,
                    requested_by="manual",
                    priority=100,
                    vision_model="llava:13b",
                    pipeline_version=PIPELINE_VERSION,
                    candidate_fingerprint="same-fp",
                )
            )
            s.commit()

        monkeypatch.setattr(
            "app.workers.video_visual._load_candidates",
            lambda file_id: ([], None),
        )
        monkeypatch.setattr(
            "app.workers.video_visual.compute_candidate_fingerprint",
            lambda candidates: "same-fp",
        )

        worker = VideoVisualWorker(MagicMock())
        result = await worker.enqueue("vid-ok", requested_by="on_index")
        assert result["accepted"] is False
        assert result["reason"] == "up_to_date"

    @pytest.mark.asyncio
    async def test_on_index_regenerates_an_old_pipeline_contract(
        self, search_db, feature_manual, policy_allow_all, monkeypatch,
    ):
        _, Session = search_db
        with Session() as s:
            s.add(
                VideoVisualRun(
                    id="vvr_active", file_id="vid-ok", status="succeeded",
                    is_active=True, requested_by="manual", priority=100,
                    vision_model="llava:13b", pipeline_version=1,
                    candidate_fingerprint="same-fp",
                )
            )
            s.commit()

        monkeypatch.setattr(
            "app.workers.video_visual._load_candidates",
            lambda file_id: ([], None),
        )
        monkeypatch.setattr(
            "app.workers.video_visual.compute_candidate_fingerprint",
            lambda candidates: "same-fp",
        )

        result = await VideoVisualWorker(MagicMock()).enqueue(
            "vid-ok", requested_by="on_index"
        )

        assert result["accepted"] is True

    @pytest.mark.asyncio
    async def test_manual_generate_again_ignores_fingerprint_match(
        self, search_db, feature_manual, policy_allow_all, monkeypatch,
    ):
        _, Session = search_db
        with Session() as s:
            s.add(
                VideoVisualRun(
                    id="vvr_active",
                    file_id="vid-ok",
                    status="succeeded",
                    is_active=True,
                    requested_by="manual",
                    priority=100,
                    vision_model="llava:13b",
                    pipeline_version=1,
                    candidate_fingerprint="same-fp",
                )
            )
            s.commit()

        monkeypatch.setattr(
            "app.workers.video_visual.compute_candidate_fingerprint",
            lambda candidates: "same-fp",
        )

        worker = VideoVisualWorker(MagicMock())
        result = await worker.enqueue("vid-ok", requested_by="manual")
        assert result["accepted"] is True


# ---------------------------------------------------------------------------
# retry()
# ---------------------------------------------------------------------------


class TestRetry:
    @pytest.mark.asyncio
    async def test_no_run_rejected(self, search_db, feature_manual, policy_allow_all):
        worker = VideoVisualWorker(MagicMock())
        result = await worker.retry("vid-ok")
        assert result["accepted"] is False
        assert result["reason"] == "no_run"

    @pytest.mark.asyncio
    async def test_run_with_no_failed_scenes_rejected(
        self, search_db, feature_manual, policy_allow_all,
    ):
        _, Session = search_db
        with Session() as s:
            s.add(
                VideoVisualRun(
                    id="vvr_1", file_id="vid-ok", status="partial", is_active=True,
                    requested_by="manual", priority=100, vision_model="llava:13b",
                    pipeline_version=1, candidate_fingerprint="fp",
                    selected_count=2, completed_count=2, succeeded_count=2, failed_count=0,
                )
            )
            s.commit()
        worker = VideoVisualWorker(MagicMock())
        result = await worker.retry("vid-ok")
        assert result["accepted"] is False
        assert result["reason"] == "no_failed_scenes"

    @pytest.mark.asyncio
    async def test_failed_scenes_reset_to_pending(
        self, search_db, feature_manual, policy_allow_all,
    ):
        _, Session = search_db
        with Session() as s:
            s.add(
                VideoVisualRun(
                    id="vvr_1", file_id="vid-ok", status="partial", is_active=True,
                    requested_by="manual", priority=0, vision_model="llava:13b",
                    pipeline_version=1, candidate_fingerprint="fp",
                    selected_count=2, completed_count=2, succeeded_count=1, failed_count=1,
                )
            )
            s.add(
                VideoVisualScene(
                    run_id="vvr_1", ordering=0, clip_embedding_id="c0",
                    start_time=1.0, status="succeeded",
                )
            )
            s.add(
                VideoVisualScene(
                    run_id="vvr_1", ordering=1, clip_embedding_id="c1",
                    start_time=2.0, status="failed", error_class="MalformedOutput",
                    error_message="bad json",
                )
            )
            s.commit()

        worker = VideoVisualWorker(MagicMock())
        result = await worker.retry("vid-ok")
        assert result["accepted"] is True
        assert result["reset_count"] == 1

        with Session() as s:
            run = s.query(VideoVisualRun).filter_by(id="vvr_1").first()
            assert run.status == "queued"
            assert run.priority == 100
            assert run.failed_count == 0
            assert run.completed_count == 1

            scenes = {
                sc.ordering: sc
                for sc in s.query(VideoVisualScene).filter_by(run_id="vvr_1").all()
            }
            assert scenes[0].status == "succeeded"  # untouched
            assert scenes[1].status == "pending"
            assert scenes[1].error_class is None
            assert scenes[1].error_message is None

    @pytest.mark.asyncio
    async def test_obsolete_failed_run_older_than_active_is_not_retryable(
        self, search_db, feature_manual, policy_allow_all,
    ):
        _, Session = search_db
        with Session() as s:
            s.add(
                VideoVisualRun(
                    id="vvr_old_failed", file_id="vid-ok", status="failed",
                    is_active=False, requested_by="manual", priority=100,
                    vision_model="llava:13b", pipeline_version=1,
                    candidate_fingerprint="fp-old", selected_count=1,
                    completed_count=1, succeeded_count=0, failed_count=1,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )
            s.add(
                VideoVisualScene(
                    run_id="vvr_old_failed", ordering=0,
                    clip_embedding_id="old-c0", start_time=1.0,
                    status="failed",
                )
            )
            s.add(
                VideoVisualRun(
                    id="vvr_active_new", file_id="vid-ok", status="succeeded",
                    is_active=True, requested_by="manual", priority=100,
                    vision_model="llava:13b", pipeline_version=1,
                    candidate_fingerprint="fp-new", selected_count=1,
                    completed_count=1, succeeded_count=1, failed_count=0,
                    created_at=datetime(2026, 1, 2, tzinfo=UTC),
                )
            )
            s.commit()

        result = await VideoVisualWorker(MagicMock()).retry("vid-ok")

        assert result == {"accepted": False, "reason": "no_run"}
        with Session() as s:
            old = s.query(VideoVisualRun).filter_by(id="vvr_old_failed").one()
            assert old.status == "failed"


# ---------------------------------------------------------------------------
# _finalize_run activation rules (design doc §5.3)
# ---------------------------------------------------------------------------


class TestFinalizeRunSucceededActivation:
    @pytest.mark.asyncio
    async def test_succeeded_with_no_prior_active_becomes_active(
        self, search_db, feature_manual, no_emit,
    ):
        _, Session = search_db
        with Session() as s:
            s.add(
                VideoVisualRun(
                    id="vvr_new", file_id="vid-ok", status="running", is_active=False,
                    requested_by="manual", priority=100, vision_model="llava:13b",
                    pipeline_version=1, candidate_fingerprint="fp",
                    selected_count=2, completed_count=2, succeeded_count=2, failed_count=0,
                )
            )
            s.commit()

        worker = VideoVisualWorker(MagicMock())
        await worker._finalize_run("vvr_new", "vid-ok", "family")

        with Session() as s:
            run = s.query(VideoVisualRun).filter_by(id="vvr_new").first()
            assert run.status == "succeeded"
            assert run.is_active is True

    @pytest.mark.asyncio
    async def test_succeeded_supersedes_prior_active_and_purges_its_embeddings(
        self, search_db, feature_manual, no_emit,
    ):
        _, Session = search_db
        with Session() as s:
            s.add(
                VideoVisualRun(
                    id="vvr_old", file_id="vid-ok", status="succeeded", is_active=True,
                    requested_by="manual", priority=100, vision_model="llava:13b",
                    pipeline_version=1, candidate_fingerprint="fp-old",
                    selected_count=1, completed_count=1, succeeded_count=1, failed_count=0,
                )
            )
            old_scene = VideoVisualScene(
                run_id="vvr_old", ordering=0, clip_embedding_id="c0",
                start_time=1.0, status="succeeded",
            )
            s.add(old_scene)
            s.flush()
            s.add(
                Embedding(
                    id=f"vvs_{old_scene.id}_aaaaaaaa",
                    file_id="vid-ok",
                    embedding_type="video_visual_scene",
                    content_preview="old scene",
                    vector_table="vec_text",
                )
            )
            s.add(
                VideoVisualRun(
                    id="vvr_new", file_id="vid-ok", status="running", is_active=False,
                    requested_by="manual", priority=100, vision_model="llava:13b",
                    pipeline_version=1, candidate_fingerprint="fp-new",
                    selected_count=1, completed_count=1, succeeded_count=1, failed_count=0,
                )
            )
            stale_run = VideoVisualRun(
                id="vvr_stale_partial", file_id="vid-ok", status="partial",
                is_active=False, requested_by="manual", priority=100,
                vision_model="llava:13b", pipeline_version=1,
                candidate_fingerprint="fp-stale", selected_count=1,
                completed_count=1, succeeded_count=1, failed_count=0,
            )
            s.add(stale_run)
            stale_scene = VideoVisualScene(
                run_id="vvr_stale_partial", ordering=0,
                clip_embedding_id="stale-c0", start_time=2.0,
                status="succeeded",
            )
            s.add(stale_scene)
            s.flush()
            s.add(
                Embedding(
                    id=f"vvs_{stale_scene.id}_bbbbbbbb",
                    file_id="vid-ok", embedding_type="video_visual_scene",
                    content_preview="stale scene", vector_table="vec_text",
                )
            )
            s.commit()

        worker = VideoVisualWorker(MagicMock())
        await worker._finalize_run("vvr_new", "vid-ok", "family")

        with Session() as s:
            old_run = s.query(VideoVisualRun).filter_by(id="vvr_old").first()
            stale_run = s.query(VideoVisualRun).filter_by(id="vvr_stale_partial").first()
            new_run = s.query(VideoVisualRun).filter_by(id="vvr_new").first()
            assert old_run.status == "superseded"
            assert old_run.is_active is False
            assert stale_run.status == "superseded"
            assert stale_run.is_active is False
            assert new_run.status == "succeeded"
            assert new_run.is_active is True

            remaining_embeddings = (
                s.query(Embedding)
                .filter_by(file_id="vid-ok", embedding_type="video_visual_scene")
                .count()
            )
            assert remaining_embeddings == 0


class TestFinalizeRunPartialAndFailedActivation:
    @pytest.mark.asyncio
    async def test_first_partial_with_no_active_becomes_active(
        self, search_db, feature_manual, no_emit,
    ):
        _, Session = search_db
        with Session() as s:
            s.add(
                VideoVisualRun(
                    id="vvr_1", file_id="vid-ok", status="running", is_active=False,
                    requested_by="manual", priority=100, vision_model="llava:13b",
                    pipeline_version=1, candidate_fingerprint="fp",
                    selected_count=2, completed_count=2, succeeded_count=1, failed_count=1,
                )
            )
            s.commit()

        worker = VideoVisualWorker(MagicMock())
        await worker._finalize_run("vvr_1", "vid-ok", "family")

        with Session() as s:
            run = s.query(VideoVisualRun).filter_by(id="vvr_1").first()
            assert run.status == "partial"
            assert run.is_active is True

    @pytest.mark.asyncio
    async def test_partial_with_existing_active_stays_staged(
        self, search_db, feature_manual, no_emit,
    ):
        _, Session = search_db
        with Session() as s:
            s.add(
                VideoVisualRun(
                    id="vvr_active", file_id="vid-ok", status="succeeded", is_active=True,
                    requested_by="manual", priority=100, vision_model="llava:13b",
                    pipeline_version=1, candidate_fingerprint="fp-old",
                    selected_count=1, completed_count=1, succeeded_count=1, failed_count=0,
                )
            )
            s.add(
                VideoVisualRun(
                    id="vvr_staged", file_id="vid-ok", status="running", is_active=False,
                    requested_by="manual", priority=100, vision_model="llava:13b",
                    pipeline_version=1, candidate_fingerprint="fp-new",
                    selected_count=2, completed_count=2, succeeded_count=1, failed_count=1,
                )
            )
            s.commit()

        worker = VideoVisualWorker(MagicMock())
        await worker._finalize_run("vvr_staged", "vid-ok", "family")

        with Session() as s:
            active = s.query(VideoVisualRun).filter_by(id="vvr_active").first()
            staged = s.query(VideoVisualRun).filter_by(id="vvr_staged").first()
            assert active.is_active is True
            assert active.status == "succeeded"  # untouched
            assert staged.status == "partial"
            assert staged.is_active is False

    @pytest.mark.asyncio
    async def test_failed_never_changes_active_run(
        self, search_db, feature_manual, no_emit,
    ):
        _, Session = search_db
        with Session() as s:
            s.add(
                VideoVisualRun(
                    id="vvr_active", file_id="vid-ok", status="succeeded", is_active=True,
                    requested_by="manual", priority=100, vision_model="llava:13b",
                    pipeline_version=1, candidate_fingerprint="fp-old",
                    selected_count=1, completed_count=1, succeeded_count=1, failed_count=0,
                )
            )
            s.add(
                VideoVisualRun(
                    id="vvr_staged", file_id="vid-ok", status="running", is_active=False,
                    requested_by="manual", priority=100, vision_model="llava:13b",
                    pipeline_version=1, candidate_fingerprint="fp-new",
                    selected_count=2, completed_count=2, succeeded_count=0, failed_count=2,
                )
            )
            s.commit()

        worker = VideoVisualWorker(MagicMock())
        await worker._finalize_run("vvr_staged", "vid-ok", "family")

        with Session() as s:
            active = s.query(VideoVisualRun).filter_by(id="vvr_active").first()
            staged = s.query(VideoVisualRun).filter_by(id="vvr_staged").first()
            assert active.is_active is True
            assert staged.status == "failed"
            assert staged.is_active is False


class TestInterruptedRunRecovery:
    def test_requeues_run_and_resets_only_running_scenes(self, search_db):
        _, Session = search_db
        with Session() as s:
            s.add(
                VideoVisualRun(
                    id="vvr_interrupted", file_id="vid-ok", status="running",
                    is_active=False, requested_by="manual", priority=100,
                    vision_model="llava:13b", pipeline_version=1,
                    candidate_fingerprint="fp",
                )
            )
            s.add_all([
                VideoVisualScene(
                    run_id="vvr_interrupted", ordering=0,
                    clip_embedding_id="c0", start_time=1.0, status="running",
                ),
                VideoVisualScene(
                    run_id="vvr_interrupted", ordering=1,
                    clip_embedding_id="c1", start_time=2.0, status="succeeded",
                ),
            ])
            s.commit()

        VideoVisualWorker(MagicMock())._requeue_interrupted_run("vvr_interrupted")

        with Session() as s:
            run = s.query(VideoVisualRun).filter_by(id="vvr_interrupted").one()
            statuses = {
                scene.ordering: scene.status
                for scene in s.query(VideoVisualScene)
                .filter_by(run_id="vvr_interrupted")
                .all()
            }
            assert run.status == "queued"
            assert statuses == {0: "pending", 1: "succeeded"}

    @pytest.mark.asyncio
    async def test_frame_extraction_is_offloaded_from_event_loop(self, monkeypatch):
        to_thread = AsyncMock(return_value=b"frame")
        monkeypatch.setattr("app.workers.video_visual.asyncio.to_thread", to_thread)

        result = await _load_frame_bytes("vid-ok", 3.5, "/drives/family/clip.mp4")

        assert result == b"frame"
        to_thread.assert_awaited_once_with(
            _read_or_extract_frame,
            "vid-ok",
            3.5,
            "/drives/family/clip.mp4",
        )


# ---------------------------------------------------------------------------
# recover_on_startup
# ---------------------------------------------------------------------------


class TestRecoverOnStartup:
    def test_resets_running_scenes_and_runs(self, search_db):
        _, Session = search_db
        with Session() as s:
            s.add(
                VideoVisualRun(
                    id="vvr_1", file_id="vid-ok", status="running", is_active=False,
                    requested_by="manual", priority=100, vision_model="llava:13b",
                    pipeline_version=1, candidate_fingerprint="fp",
                )
            )
            s.add(
                VideoVisualScene(
                    run_id="vvr_1", ordering=0, clip_embedding_id="c0",
                    start_time=1.0, status="running",
                )
            )
            s.add(
                VideoVisualScene(
                    run_id="vvr_1", ordering=1, clip_embedding_id="c1",
                    start_time=2.0, status="succeeded",
                )
            )
            s.commit()

        counts = recover_on_startup()
        assert counts["runs_reset"] == 1
        assert counts["scenes_reset"] == 1

        with Session() as s:
            run = s.query(VideoVisualRun).filter_by(id="vvr_1").first()
            assert run.status == "queued"
            scenes = {
                sc.ordering: sc.status
                for sc in s.query(VideoVisualScene).filter_by(run_id="vvr_1").all()
            }
            assert scenes[0] == "pending"
            assert scenes[1] == "succeeded"  # never touched
