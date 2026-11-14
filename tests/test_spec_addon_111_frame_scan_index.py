"""SPEC-ADDON-111 / SPEC-ADDON-113: the frame scan's flag, tables and queue, the
indexing status, the CLIP regenerate, and the Visual Index's readiness and scene
sections.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from sqlalchemy import text

import app.database as database
from app.config import FeaturesConfig, LLMConfig
from app.models import Embedding, IndexedFile, VideoVisualRun, VideoVisualScene
from tests.ocr_fixtures import (
    SearchDB,
    build_search_db,
    pillow_real,  # noqa: F401  - the usefixtures marker resolves it
    seed_indexed_file,
)
from tests.vec_partition_db import search_index, start  # noqa: F401 - fixture

LOFT_MIME = "application/vnd.litloft.loft+json"
SCAN = "frame_scan"
ALL_DONE = dict(
    metadata_indexed=True,
    clip_indexed=True,
    clip_thumbnail_indexed=True,
    whisper_indexed=True,
    text_indexed=True,
    tfidf_keywords_indexed=True,
    frame_ocr_read=True,
    frame_ocr_indexed=True,
)


def _add_file(file_id: str, mime: str, *, file_type: str | None = None) -> None:
    with database.get_search_db() as s:
        s.add(IndexedFile(
            file_id=file_id, drive="family", filename=file_id,
            file_path=f"/drives/family/{file_id}",
            file_type=file_type or mime.split("/")[0], mime_type=mime,
            file_size=1, active=True,
        ))


def _flag(file_id: str):
    with database._search_engine.connect() as conn:
        return conn.execute(
            text("SELECT frame_scan_indexed FROM indexed_files WHERE file_id = :f"),
            {"f": file_id},
        ).scalar()


# ---------------------------------------------------------------------------
# SPEC-ADDON-111 item 1, §4 - schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_spec_addon_111_the_migration_marks_everything_but_native_videos_done(
        self, search_index,
    ):
        _add_file("video", "video/mp4")
        _add_file("loft", LOFT_MIME, file_type="video")
        _add_file("picture", "image/jpeg")
        _add_file("pdf", "application/pdf", file_type="document")
        with database._search_engine.begin() as conn:
            for (name,) in conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type = 'index' "
                "AND tbl_name = 'indexed_files' AND sql LIKE '%frame_scan_indexed%'"
            )).fetchall():
                conn.execute(text(f"DROP INDEX {name}"))
        search_index.sql("ALTER TABLE indexed_files DROP COLUMN frame_scan_indexed")

        start()

        assert [_flag(f) for f in ("video", "loft", "picture", "pdf")] == [0, 1, 1, 1]

    def test_spec_addon_111_a_new_row_starts_not_scanned(self, search_index):
        _add_file("video", "video/mp4")

        assert _flag("video") == 0

    def test_spec_addon_111_change_frames_have_their_own_vector_table_shaped_like_vec_clip(
        self, search_index,
    ):
        assert search_index.dim("vec_clip_change") == search_index.dim("vec_clip")
        assert search_index.is_partitioned("vec_clip_change")


def _change_frame(index, file_id: str, second: int, dim: int) -> str:
    emb_id = f"chg_{file_id}_{second}"
    with database.get_search_db() as s:
        s.add(Embedding(
            id=emb_id, file_id=file_id, embedding_type="clip_change",
            vector_table="vec_clip_change", timestamp_start=float(second),
            content_preview="x",
        ))
    index.put_vector(
        "vec_clip_change", emb_id, np.ones(dim, dtype=np.float32) / np.sqrt(dim),
        partition="clip_change",
    )
    return emb_id


def _scan_row(index, file_id: str) -> None:
    index.sql(
        "INSERT INTO video_frame_scans(file_id, version, seconds, hashes, sizes, "
        "created_at) VALUES (:f, 1, 1, :h, :s, :now)",
        f=file_id, h=b"\0" * 32, s=b"\0\0", now=datetime.now(UTC),
    )


class TestStates:
    def test_spec_addon_111_a_change_table_of_another_dimension_is_rebuilt_and_rescanned(
        self, search_index,
    ):
        """§3 States: compared on its own, independently of vec_clip."""
        dim = search_index.dim("vec_clip")
        _add_file("video", "video/mp4")
        _add_file("picture", "image/jpeg")
        search_index.sql("UPDATE indexed_files SET frame_scan_indexed = 1")
        search_index.add_embedding("clip_video_1", "video", "clip", "vec_clip")
        search_index.put_vector("vec_clip", "clip_video_1", np.ones(dim), partition="clip")
        _scan_row(search_index, "video")
        search_index.sql("DROP TABLE vec_clip_change")
        search_index.sql(
            "CREATE VIRTUAL TABLE vec_clip_change USING vec0(embedding_id TEXT PRIMARY KEY, "
            "embedding_type text partition key, vector float[8])"
        )
        search_index.add_embedding("chg_video_3", "video", "clip_change", "vec_clip_change")
        search_index.put_vector("vec_clip_change", "chg_video_3", np.ones(8),
                                partition="clip_change")

        start()

        assert search_index.dim("vec_clip_change") == dim
        assert search_index.vectors("vec_clip_change") == {}
        assert [e for e in search_index.embeddings("video") if e[1] == "clip_change"] == []
        assert set(search_index.vectors("vec_clip")) == {"clip_video_1"}
        assert (_flag("video"), _flag("picture")) == (0, 1)
        with database._search_engine.connect() as conn:
            assert conn.execute(text("SELECT file_id FROM video_frame_scans")).scalars().all() \
                == ["video"]

    def test_spec_addon_111_purging_a_file_removes_its_scan_and_change_frames(
        self, search_index,
    ):
        """I8."""
        from app.indexer import _purge_file

        dim = search_index.dim("vec_clip")
        _add_file("gone", "video/mp4")
        _add_file("kept", "video/mp4")
        for file_id in ("gone", "kept"):
            _change_frame(search_index, file_id, 0, dim)
            _scan_row(search_index, file_id)

        _purge_file("gone")

        assert set(search_index.vectors("vec_clip_change")) == {"chg_kept_0"}
        assert [e[0] for e in search_index.embeddings()] == ["chg_kept_0"]
        with database._search_engine.connect() as conn:
            assert conn.execute(text("SELECT file_id FROM video_frame_scans")).scalars().all() \
                == ["kept"]


# ---------------------------------------------------------------------------
# SPEC-ADDON-111 items 2, 9, 10 - the queue
# ---------------------------------------------------------------------------


@pytest.fixture()
def ocr_db(tmp_path, monkeypatch) -> SearchDB:
    return build_search_db(tmp_path, monkeypatch)


def _manager():
    from app.indexer import IndexManager

    return IndexManager()


async def _scans(manager) -> list[tuple[str, int]]:
    from app.indexer import TaskType

    return [
        (t.file_id, t.priority)
        for t in await manager._collect_batch(TaskType(SCAN), 10)
    ]


def _seed_video(db, tmp_path, file_id="vid", mime="video/mp4", **flags) -> None:
    seed_indexed_file(
        db, file_id, mime, tmp_path / f"{file_id}.mp4", file_type="video",
        **{**ALL_DONE, **flags},
    )


@pytest.mark.usefixtures("pillow_real")
class TestQueue:
    async def test_spec_addon_111_a_new_native_video_is_scanned_at_priority_zero(self, ocr_db):
        manager = _manager()

        await manager._queue_file_tasks({"id": "f1", "mime_type": "video/mp4"})

        assert await _scans(manager) == [("f1", 0)]

    @pytest.mark.parametrize("mime", [LOFT_MIME, "image/jpeg", "audio/mpeg"])
    async def test_spec_addon_111_a_new_file_that_is_not_a_native_video_is_not_scanned(
        self, ocr_db, mime,
    ):
        manager = _manager()

        await manager._queue_file_tasks({"id": "f1", "mime_type": mime})

        assert await _scans(manager) == []

    async def test_spec_addon_111_the_backlog_is_scanned_at_minus_ten(self, ocr_db, tmp_path):
        _seed_video(ocr_db, tmp_path, frame_scan_indexed=False)
        manager = _manager()

        await manager._resume_incomplete()

        assert await _scans(manager) == [("vid", -10)]

    async def test_spec_addon_111_an_inactive_or_scanned_video_is_not_queued(
        self, ocr_db, tmp_path,
    ):
        _seed_video(ocr_db, tmp_path, "done", frame_scan_indexed=True)
        _seed_video(ocr_db, tmp_path, "gone", active=False, frame_scan_indexed=False)
        manager = _manager()

        await manager._resume_incomplete()

        assert await _scans(manager) == []

    @pytest.mark.parametrize(
        "mime, file_type",
        [(LOFT_MIME, "video"), ("image/png", "image"), ("text/plain", "document")],
    )
    async def test_spec_addon_111_a_file_that_is_not_a_native_video_is_marked_done(
        self, ocr_db, tmp_path, mime, file_type,
    ):
        _seed_video(ocr_db, tmp_path, "other", mime=mime, frame_scan_indexed=False)
        with ocr_db.Session() as s:
            s.query(IndexedFile).filter_by(file_id="other").update({"file_type": file_type})
            s.commit()
        manager = _manager()

        await manager._resume_incomplete()

        assert bool(ocr_db.flag("other", "frame_scan_indexed")) is True
        assert await _scans(manager) == []

    async def test_spec_addon_111_prioritize_bumps_a_pending_scan(self, ocr_db, tmp_path):
        _seed_video(ocr_db, tmp_path, frame_scan_indexed=False)
        manager = _manager()
        await manager._resume_incomplete()

        await manager.prioritize("vid")

        queued = await _scans(manager)
        assert queued and all(f == "vid" for f, _ in queued)
        assert max(p for _, p in queued) > -10


@pytest.mark.usefixtures("pillow_real")
class TestClipRegenerate:
    async def _reindex(self, manager, monkeypatch):
        from app.routers import files as files_router

        monkeypatch.setattr(files_router, "get_index_manager", lambda: manager)
        result = await files_router.reindex_file(
            "vid", body={"tasks": ["clip"]}, drive="drive1",
        )
        return result if isinstance(result, dict) else result.model_dump()

    async def test_spec_addon_111_regenerating_clip_rescans_a_native_video(
        self, ocr_db, tmp_path, monkeypatch,
    ):
        _seed_video(ocr_db, tmp_path, frame_scan_indexed=True)
        manager = _manager()

        payload = await self._reindex(manager, monkeypatch)

        assert bool(ocr_db.flag("vid", "frame_scan_indexed")) is False
        assert await _scans(manager) == [("vid", 100)]
        assert manager.is_queued("vid", SCAN)
        assert SCAN not in repr(payload)

    async def test_spec_addon_111_a_second_regenerate_queues_no_second_scan(
        self, ocr_db, tmp_path, monkeypatch,
    ):
        _seed_video(ocr_db, tmp_path, frame_scan_indexed=True)
        manager = _manager()

        await self._reindex(manager, monkeypatch)
        await self._reindex(manager, monkeypatch)

        assert await _scans(manager) == [("vid", 100)]

    async def test_spec_addon_111_the_scan_is_queued_when_clip_is_already_queued(
        self, ocr_db, tmp_path, monkeypatch,
    ):
        _seed_video(ocr_db, tmp_path, clip_indexed=False, frame_scan_indexed=True)
        manager = _manager()
        await manager._resume_incomplete()

        await self._reindex(manager, monkeypatch)

        assert bool(ocr_db.flag("vid", "frame_scan_indexed")) is False
        assert await _scans(manager) == [("vid", 100)]

    async def test_spec_addon_111_only_the_clip_regenerate_resets_the_scan(
        self, ocr_db, tmp_path, monkeypatch,
    ):
        """§3 States: other reindex tasks leave the flag as it is."""
        from app.routers import files as files_router

        _seed_video(ocr_db, tmp_path, frame_scan_indexed=True)
        manager = _manager()
        monkeypatch.setattr(files_router, "get_index_manager", lambda: manager)

        await files_router.reindex_file("vid", body={"tasks": ["whisper"]}, drive="drive1")
        after_whisper = bool(ocr_db.flag("vid", "frame_scan_indexed"))
        await self._reindex(manager, monkeypatch)

        assert after_whisper is True
        assert bool(ocr_db.flag("vid", "frame_scan_indexed")) is False
        assert await _scans(manager) == [("vid", 100)]


    async def test_spec_addon_111_a_claimed_scan_is_left_as_it_is(
        self, ocr_db, tmp_path, monkeypatch,
    ):
        """Item 9: a running (claimed) scan keeps the flag and its result stands."""
        from app.routers import files as files_router

        _seed_video(ocr_db, tmp_path, frame_scan_indexed=True)
        manager = _manager()
        claims = set(files_router._reindex_in_flight) | {("vid", SCAN)}
        monkeypatch.setattr(files_router, "_reindex_in_flight", claims)

        await self._reindex(manager, monkeypatch)

        assert bool(ocr_db.flag("vid", "frame_scan_indexed")) is True
        assert await _scans(manager) == []


# ---------------------------------------------------------------------------
# SPEC-ADDON-113 item 6 - indexing status
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("pillow_real")
class TestStatus:
    def _status(self, monkeypatch):
        from app import main

        manager = main.IndexManager()
        monkeypatch.setattr(main.dependencies, "get_index_manager", lambda: manager)
        monkeypatch.setattr(main.dependencies, "_video_visual_worker", None)
        return main.status_endpoint(), manager

    def test_spec_addon_113_status_counts_native_videos_by_their_scan(
        self, ocr_db, tmp_path, monkeypatch,
    ):
        _seed_video(ocr_db, tmp_path, "done", frame_scan_indexed=True)
        _seed_video(ocr_db, tmp_path, "todo", frame_scan_indexed=False)
        _seed_video(ocr_db, tmp_path, "gone", active=False, frame_scan_indexed=False)
        _seed_video(ocr_db, tmp_path, "loft", mime=LOFT_MIME, frame_scan_indexed=False)
        seed_indexed_file(
            ocr_db, "pic", "image/png", tmp_path / "pic.png", **ALL_DONE,
            frame_scan_indexed=False,
        )

        pending, manager = self._status(monkeypatch)
        ocr_db.set_flag("todo", "frame_scan_indexed", True)
        scanned, _ = self._status(monkeypatch)

        assert (pending.indexed[SCAN], pending.pending[SCAN]) == (1, 1)
        assert (scanned.indexed[SCAN], scanned.pending[SCAN]) == (2, 0)
        assert pending.pending["total"] - scanned.pending["total"] == 1
        assert manager.get_queue_breakdown()[SCAN] == {"waiting": 0, "processing": []}


# ---------------------------------------------------------------------------
# SPEC-ADDON-113 items 1, 2 - Visual Index readiness and sections
# ---------------------------------------------------------------------------


@pytest.fixture()
def visual_db(monkeypatch, search_index, make_settings, use_llm):  # noqa: F811
    for name in ("app.workers.video_visual", "app.routers.video_visual"):
        monkeypatch.setattr(f"{name}.get_search_db", database.get_search_db, raising=False)
        monkeypatch.setattr(f"{name}.get_search_db_read", database.get_search_db_read)

    settings = make_settings(
        features=FeaturesConfig(video_visual_index="manual"),  # type: ignore[call-arg]
        llm=LLMConfig(
            provider="openai_compatible", base_url="http://test",
            model="gemma2:27b", vision_model="llava:13b",
        ),
    )
    for name in ("app.config", "app.workers.video_visual", "app.routers.video_visual"):
        monkeypatch.setattr(f"{name}.settings", settings)
    monkeypatch.setattr(
        "app.workers.video_visual.is_file_feature_enabled",
        AsyncMock(return_value=True), raising=False,
    )
    monkeypatch.setattr(
        "app.routers.video_visual.is_feature_enabled", AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "app.workers.video_visual.emit_video_visual_event", AsyncMock(return_value=None),
    )
    index_manager = MagicMock()
    index_manager.prioritize = AsyncMock(return_value=True)
    monkeypatch.setattr("app.dependencies.get_index_manager", lambda: index_manager)
    use_llm(MagicMock())

    def _video(file_id: str, *, scanned: bool) -> None:
        with database.get_search_db() as s:
            s.add(IndexedFile(
                file_id=file_id, drive="family", filename=f"{file_id}.mp4",
                file_path=f"/drives/family/{file_id}.mp4", file_type="video",
                mime_type="video/mp4", file_size=1, duration=600.0, active=True,
                frame_scan_indexed=scanned,
            ))
            s.add(Embedding(
                id=f"clip_{file_id}_1", file_id=file_id, embedding_type="clip",
                timestamp_start=5.0, vector_table="vec_clip",
            ))
        search_index.put_vector(
            "vec_clip", f"clip_{file_id}_1",
            np.ones(search_index.dim("vec_clip"), dtype=np.float32), partition="clip",
        )

    return database.get_search_db, _video


class TestVisualIndex:
    @pytest.mark.parametrize("requested_by", ["manual", "on_index"])
    async def test_spec_addon_113_a_video_waits_for_its_scan(self, visual_db, requested_by):
        from app.workers.video_visual import VideoVisualWorker

        _, video = visual_db
        video("waiting", scanned=False)
        video("ready", scanned=True)
        worker = VideoVisualWorker()

        waiting = await worker.enqueue("waiting", requested_by=requested_by)
        ready = await worker.enqueue("ready", requested_by=requested_by)

        assert (waiting["accepted"], waiting["reason"]) == (False, "waiting_clip")
        assert ready["accepted"] is True

    async def test_spec_addon_113_scene_items_carry_their_section(self, visual_db):
        from app.routers.video_visual import get_visual_index

        Session, video = visual_db
        video("new", scanned=True)
        with Session() as s:
            s.add(VideoVisualRun(
                id="vvr_new", file_id="new", status="succeeded", is_active=True,
                requested_by="manual", priority=100, vision_model="llava:13b",
                pipeline_version=3, candidate_fingerprint="fp",
                selected_count=2, completed_count=2, succeeded_count=2, failed_count=0,
            ))
            s.add(VideoVisualScene(
                run_id="vvr_new", ordering=0, clip_embedding_id="c0", start_time=6.0,
                section_start=0.0, section_end=75.5, status="succeeded", scene_label="A",
            ))
            s.add(VideoVisualScene(
                run_id="vvr_new", ordering=1, clip_embedding_id="c1", start_time=80.0,
                status="succeeded", scene_label="B",
            ))

        result = await get_visual_index(file_id="new", drive="family")

        items = [s.model_dump() for s in result.scenes]
        assert [(i["start_time"], i["section_start"], i["section_end"]) for i in items] == [
            (6.0, 0.0, 75.5), (80.0, None, None),
        ]


def _run(Session, run_id, file_id, *, status="succeeded", version=2, active=True):
    with Session() as s:
        s.add(VideoVisualRun(
            id=run_id, file_id=file_id, status=status, is_active=active,
            requested_by="on_index", priority=0, vision_model="llava:13b",
            pipeline_version=version, candidate_fingerprint="fp",
            selected_count=1, completed_count=1,
            succeeded_count=1 if status == "succeeded" else 0,
            failed_count=1 if status == "failed" else 0,
        ))


def _run_ids(file_id):
    with database._search_engine.connect() as conn:
        return conn.execute(
            text("SELECT id FROM video_visual_runs WHERE file_id = :f ORDER BY id"),
            {"f": file_id},
        ).scalars().all()


class TestAutomaticRequest:
    """SPEC-ADDON-113 item 1, I9: what an on_index request may start."""

    @pytest.mark.parametrize(
        "status, active",
        [
            pytest.param("succeeded", True, id="older-pipeline-active-run-kept"),
            pytest.param("failed", False, id="failed-run-not-retried"),
        ],
    )
    async def test_spec_addon_113_an_automatic_request_starts_only_a_first_run(
        self, visual_db, status, active,
    ):
        from app.workers.video_visual import VideoVisualWorker

        Session, video = visual_db
        video("old", scanned=True)
        video("fresh", scanned=True)
        _run(Session, "vvr_old", "old", status=status, version=2, active=active)
        worker = VideoVisualWorker()

        old = await worker.enqueue("old", requested_by="on_index")
        fresh = await worker.enqueue("fresh", requested_by="on_index")

        assert old["accepted"] is False
        assert _run_ids("old") == ["vvr_old"]
        assert fresh["accepted"] is True
        assert len(_run_ids("fresh")) == 1

    async def test_spec_addon_113_an_older_run_stays_until_the_user_regenerates_it(
        self, visual_db,
    ):
        from app.workers.video_visual import VideoVisualWorker

        Session, video = visual_db
        video("old", scanned=True)
        _run(Session, "vvr_old", "old", version=2)
        worker = VideoVisualWorker()

        automatic = await worker.enqueue("old", requested_by="on_index")
        manual = await worker.enqueue("old", requested_by="manual")

        assert automatic["accepted"] is False
        assert manual["accepted"] is True
        assert len(_run_ids("old")) == 2


class TestBuildScenes:
    async def test_spec_addon_113_a_run_is_built_from_the_navigation_list(
        self, visual_db, search_index,  # noqa: F811
    ):
        """SPEC-ADDON-113 item 2, I9: frame time in start_time, the section beside it."""
        import inspect

        from app.workers.video_visual import VideoVisualWorker
        from tests.scene_sections_db import Video, basis

        Session, _ = visual_db
        video = Video(search_index, "built")
        for t in [5.0 + 10 * k for k in range(60)]:
            video.clip(t, basis(video.clip_dim, int(t // 75)))
        video.commit()
        search_index.sql("UPDATE indexed_files SET frame_scan_indexed = 1")
        with Session() as s:
            s.add(VideoVisualRun(
                id="vvr_built", file_id="built", status="queued", is_active=False,
                requested_by="manual", priority=100, vision_model="llava:13b",
                pipeline_version=2, candidate_fingerprint="",
            ))

        built = VideoVisualWorker()._build_scenes("vvr_built", "built")
        if inspect.isawaitable(built):
            built = await built

        assert built is True
        with database._search_engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT start_time, end_time, section_start, section_end, clip_embedding_id "
                "FROM video_visual_scenes WHERE run_id = 'vvr_built' ORDER BY ordering"
            )).fetchall()
            version = conn.execute(text(
                "SELECT pipeline_version FROM video_visual_runs WHERE id = 'vvr_built'"
            )).scalar()
        frames = [5.0, 85.0, 165.0, 235.0, 315.0, 385.0, 465.0, 535.0]
        bounds = [0.0, 75.0, 155.0, 225.0, 305.0, 375.0, 455.0, 525.0, 600.0]
        assert [tuple(r) for r in rows] == [
            (frame, None, start, end, f"clip_built_{int(frame // 10) + 1}")
            for frame, start, end in zip(frames, bounds, bounds[1:])
        ]
        assert version == 3


class TestOrphans:
    def test_spec_addon_111_orphan_cleanup_removes_scans_and_change_vectors(
        self, search_index,  # noqa: F811
    ):
        """I8, touch points: scans without an active file, vectors without a row."""
        from app.indexer import cleanup_orphaned_embeddings

        dim = search_index.dim("vec_clip")
        _add_file("kept", "video/mp4")
        _add_file("trashed", "video/mp4")
        search_index.sql("UPDATE indexed_files SET active = 0 WHERE file_id = 'trashed'")
        for file_id in ("kept", "trashed", "deleted"):
            _scan_row(search_index, file_id)
        _change_frame(search_index, "kept", 0, dim)
        search_index.put_vector(
            "vec_clip_change", "chg_deleted_7", np.ones(dim, dtype=np.float32),
            partition="clip_change",
        )

        cleanup_orphaned_embeddings()

        with database._search_engine.connect() as conn:
            assert conn.execute(text("SELECT file_id FROM video_frame_scans")).scalars().all() \
                == ["kept"]
        assert set(search_index.vectors("vec_clip_change")) == {"chg_kept_0"}
