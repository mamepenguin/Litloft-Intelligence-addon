"""SPEC-ADDON-118 items 1-3, I8: the Visual Index of a YouTube ``.loft``.

Eligibility, the route answers and the scene frame source. The drive policy and
the routed model are the stand-ins the native Visual Index tests use.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import HTTPException

import app.database as database
from app.config import FeaturesConfig, LLMConfig
from app.frame_cache import frame_cache_path
from app.models import Embedding, IndexedFile
from tests.ocr_fixtures import pillow_real  # noqa: F401 - the usefixtures marker resolves it
from tests.search_endpoint_db import patch_everywhere
from tests.vec_partition_db import search_index  # noqa: F401 - fixture
from tests.youtube_frame_doubles import describe, needs_ffmpeg, youtube  # noqa: F401

pytestmark = pytest.mark.usefixtures("pillow_real")

LOFT_MIME = "application/vnd.litloft.loft+json"


@pytest.fixture()
def visual(monkeypatch, search_index, make_settings, use_llm, tmp_path):  # noqa: F811
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

    def add(file_id: str, *, mime: str = LOFT_MIME, source: str | None = "youtube",
            failures: int = 0, scanned: bool = True) -> str:
        path = tmp_path / f"{file_id}.loft"
        path.write_text('{"provider": "youtube", "url": "https://www.youtube.com/watch?v=x"}')
        with database.get_search_db() as s:
            s.add(IndexedFile(
                file_id=file_id, drive="family", filename=path.name,
                file_path=str(path), file_type="video", mime_type=mime,
                file_size=1, duration=600.0, active=True, frame_scan_indexed=scanned,
            ))
            s.add(Embedding(
                id=f"clip_{file_id}_1", file_id=file_id, embedding_type="clip",
                timestamp_start=5.0, vector_table="vec_clip",
            ))
        search_index.put_vector(
            "vec_clip", f"clip_{file_id}_1",
            np.ones(search_index.dim("vec_clip"), dtype=np.float32), partition="clip",
        )
        if mime == LOFT_MIME:
            search_index.sql(
                "UPDATE indexed_files SET loft_source = :s, loft_fetch_failures = :n "
                "WHERE file_id = :f",
                s=source, n=failures, f=file_id,
            )
        return str(path)

    return add


# ---------------------------------------------------------------------------
# Item 1 - eligibility (I8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("requested_by", ["manual", "on_index"])
async def test_spec_addon_118_i8_only_an_uncapped_youtube_loft_is_eligible(
    visual, requested_by,
):
    from app.workers.video_visual import VideoVisualWorker

    visual("youtube")
    visual("capped", failures=3)
    visual("two-failures", failures=2)
    visual("other", source="other")
    visual("unclassified", source=None)
    worker = VideoVisualWorker()

    answers = {
        f: await worker.enqueue(f, requested_by=requested_by)
        for f in ("youtube", "two-failures", "capped", "other", "unclassified")
    }

    assert answers["youtube"]["accepted"] is True
    assert answers["two-failures"]["accepted"] is True
    for refused in ("capped", "other", "unclassified"):
        assert (answers[refused]["accepted"], answers[refused]["reason"]) == (
            False, "not_eligible",
        ), refused


async def test_spec_addon_118_a_youtube_loft_waits_for_its_scan_as_a_native_video_does(visual):
    from app.workers.video_visual import VideoVisualWorker

    visual("waiting", scanned=False)

    result = await VideoVisualWorker().enqueue("waiting", requested_by="manual")

    assert (result["accepted"], result["reason"]) == (False, "waiting_clip")


# ---------------------------------------------------------------------------
# Item 3 - the routes (I8)
# ---------------------------------------------------------------------------


async def test_spec_addon_118_i8_visual_index_is_eligible_only_for_an_uncapped_youtube_loft(
    visual,
):
    from app.routers.video_visual import get_visual_index

    visual("youtube")
    visual("capped", failures=3)
    visual("other", source="other")
    visual("unclassified", source=None)

    answers = {
        f: (await get_visual_index(file_id=f, drive="family")).eligible
        for f in ("youtube", "capped", "other", "unclassified")
    }

    assert answers == {
        "youtube": True, "capped": False, "other": False, "unclassified": False,
    }


async def test_spec_addon_118_i8_generate_accepts_an_uncapped_youtube_loft_and_404s_others(
    visual, monkeypatch,
):
    from app.routers import video_visual as router

    worker = MagicMock()
    worker.enqueue = AsyncMock(return_value={"accepted": True, "run_id": "vvr_1"})

    async def _get():
        return worker

    monkeypatch.setattr("app.routers.video_visual.get_video_visual_worker", _get)
    visual("youtube")
    visual("capped", failures=3)
    visual("other", source="other")
    visual("unclassified", source=None)

    accepted = await router.generate_visual_index(file_id="youtube", drive="family")
    refused = {}
    for file_id in ("capped", "other", "unclassified"):
        with pytest.raises(HTTPException) as excinfo:
            await router.generate_visual_index(file_id=file_id, drive="family")
        refused[file_id] = excinfo.value.status_code

    assert accepted["status"] == "accepted"
    worker.enqueue.assert_awaited_once_with("youtube", requested_by="manual")
    assert refused == {"capped": 404, "other": 404, "unclassified": 404}


# ---------------------------------------------------------------------------
# Item 2 - the scene frame (I8)
# ---------------------------------------------------------------------------


@needs_ffmpeg
async def test_spec_addon_118_i8_without_a_kept_picture_the_scene_frame_is_read_from_youtube(
    visual, youtube, monkeypatch,  # noqa: F811
):
    from app.workers.video_visual import _load_frame_bytes

    patch_everywhere(monkeypatch, "resolve_file_path", lambda _drive, path: path)
    patch_everywhere(monkeypatch, "validate_file_path", lambda _path: True)
    youtube.set_ytdlp(describe=describe([youtube.format("f720", 720)]))
    path = visual("youtube")

    data = await _load_frame_bytes("youtube", 5.0, path)

    assert data
    assert frame_cache_path("youtube", 5.0).read_bytes() == data
    assert len(youtube.lookups()) == 1
    assert not any(path in " ".join(e["argv"]) for e in youtube.ffmpeg_events())


async def test_spec_addon_118_the_sweep_enqueues_youtube_lofts_beside_native_videos(
    visual, monkeypatch,
):
    import dataclasses

    import app.config as config
    from app.models import VideoVisualRun
    from app.workers.video_visual import VideoVisualWorker

    settings = dataclasses.replace(
        config.settings,
        features=dataclasses.replace(config.settings.features, video_visual_index="on_index"),
    )
    for name in ("app.config", "app.workers.video_visual", "app.routers.video_visual"):
        monkeypatch.setattr(f"{name}.settings", settings)
    visual("native", mime="video/mp4")
    visual("youtube")
    visual("capped", failures=3)
    visual("other", source="other")
    visual("unclassified", source=None)

    await VideoVisualWorker().enqueue_unprocessed()

    with database.get_search_db_read() as s:
        swept = sorted({r.file_id for r in s.query(VideoVisualRun).all()})
    assert swept == ["native", "youtube"]
