"""SPEC-ADDON-117 items 1.2, 1.7, 1.8 and §3: queuing a ``.loft``'s fetch, the
requests that open it, the status counts and the per-file dialog's payload.
"""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from tests.ocr_fixtures import (
    SearchDB,
    build_search_db,
    patch_features,
    pillow_real,  # noqa: F401  - the usefixtures marker resolves it
    seed_indexed_file,
)

pytestmark = pytest.mark.usefixtures("pillow_real")

LOFT_MIME = "application/vnd.litloft.loft+json"
FETCH = "loft_media"
ALL_DONE = dict(
    metadata_indexed=True,
    clip_indexed=True,
    clip_thumbnail_indexed=True,
    whisper_indexed=True,
    text_indexed=True,
    tfidf_keywords_indexed=True,
    frame_scan_indexed=True,
    frame_ocr_read=True,
    frame_ocr_indexed=True,
)


@pytest.fixture()
def db(tmp_path, monkeypatch) -> SearchDB:
    return build_search_db(tmp_path, monkeypatch)


def _manager():
    from app.indexer import IndexManager

    return IndexManager()


async def _queued(manager, task: str) -> list[tuple[str, int]]:
    from app.indexer import TaskType

    return [(t.file_id, t.priority) for t in await manager._collect_batch(TaskType(task), 10)]


def _loft(db, tmp_path, file_id: str, source: str | None, *, failures: int = 0, **flags):
    path = tmp_path / f"{file_id}.loft"
    path.write_text('{"provider": "youtube", "url": "https://www.youtube.com/watch?v=x"}')
    seed_indexed_file(db, file_id, LOFT_MIME, path, file_type="video", **{**ALL_DONE, **flags})
    db.execute(
        "UPDATE indexed_files SET loft_source = :s, loft_fetch_failures = :n "
        "WHERE file_id = :f",
        s=source, n=failures, f=file_id,
    )


def _video(db, tmp_path, file_id: str = "vid", **flags):
    seed_indexed_file(
        db, file_id, "video/mp4", tmp_path / f"{file_id}.mp4", file_type="video",
        **{**ALL_DONE, **flags},
    )


async def _reindex(manager, monkeypatch, file_id: str, task: str):
    from app.routers import files as files_router

    monkeypatch.setattr(files_router, "get_index_manager", lambda: manager)
    result = await files_router.reindex_file(file_id, body={"tasks": [task]}, drive="drive1")
    if isinstance(result, dict):
        return result
    if isinstance(result, JSONResponse):
        return json.loads(result.body)
    return result.model_dump()


# ---------------------------------------------------------------------------
# Item 1.2 - queuing
# ---------------------------------------------------------------------------


async def test_spec_addon_117_a_new_loft_gets_one_fetch_at_zero_and_no_frame_task(
    db, monkeypatch,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    manager = _manager()

    await manager._queue_file_tasks({"id": "new1", "mime_type": LOFT_MIME})

    assert await _queued(manager, FETCH) == [("new1", 0)]
    assert await _queued(manager, "frame_scan") == []
    assert await _queued(manager, "frame_ocr") == []


@pytest.mark.parametrize(
    "source, flags",
    [
        pytest.param(None, dict(frame_scan_indexed=False), id="unclassified-scan-open"),
        pytest.param(None, dict(frame_ocr_read=False, frame_ocr_indexed=False),
                     id="unclassified-ocr-open"),
        pytest.param("youtube", dict(frame_scan_indexed=False), id="youtube-scan-wanted"),
        pytest.param("youtube", dict(frame_ocr_read=False, frame_ocr_indexed=False),
                     id="youtube-ocr-wanted"),
        pytest.param("web", dict(frame_scan_indexed=False), id="web-scan-wanted"),
        pytest.param("web", dict(frame_ocr_read=False, frame_ocr_indexed=False),
                     id="web-ocr-wanted"),
    ],
)
async def test_spec_addon_117_the_backlog_fetch_is_queued_at_minus_ten_and_flags_stay_open(
    db, tmp_path, monkeypatch, source, flags,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _loft(db, tmp_path, "loft1", source, **flags)
    manager = _manager()

    await manager._resume_incomplete()

    assert await _queued(manager, FETCH) == [("loft1", -10)]
    assert await _queued(manager, "frame_scan") == []
    assert await _queued(manager, "frame_ocr") == []
    for flag, value in flags.items():
        assert bool(db.flag("loft1", flag)) is value


@pytest.mark.parametrize("mode", ["manual", "false"])
async def test_spec_addon_117_outside_on_index_an_ocr_read_alone_queues_no_fetch(
    db, tmp_path, monkeypatch, mode,
):
    patch_features(monkeypatch, video_frame_ocr=mode)
    # Candidates already settled: this test measures the OCR read alone.
    _loft(db, tmp_path, "yt1", "youtube", frame_ocr_read=False, frame_ocr_indexed=False,
          loft_candidates_done=True)
    _loft(db, tmp_path, "yt2", "youtube", frame_scan_indexed=False,
          loft_candidates_done=True)
    manager = _manager()

    await manager._resume_incomplete()

    assert await _queued(manager, FETCH) == [("yt2", -10)]
    assert bool(db.flag("yt1", "frame_ocr_read")) is False


async def test_spec_addon_117_i1_the_sweep_closes_a_loft_of_source_other_and_queues_nothing(
    db, tmp_path, monkeypatch,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _loft(db, tmp_path, "other1", "other", frame_scan_indexed=False,
          frame_ocr_read=False, frame_ocr_indexed=False)
    _loft(db, tmp_path, "yt1", "youtube", frame_scan_indexed=False)
    manager = _manager()

    await manager._resume_incomplete()

    assert [bool(db.flag("other1", f)) for f in
            ("frame_scan_indexed", "frame_ocr_read", "frame_ocr_indexed")] == [True] * 3
    assert await _queued(manager, FETCH) == [("yt1", -10)]


@pytest.mark.parametrize("mode", ["on_index", "manual", "false"])
@pytest.mark.parametrize("source", ["youtube", "web"])
async def test_spec_addon_117_i7_a_stored_read_is_rebuilt_by_the_rebuild_only_task(
    db, tmp_path, monkeypatch, mode, source,
):
    patch_features(monkeypatch, video_frame_ocr=mode)
    _loft(db, tmp_path, "yt1", source, frame_ocr_read=True, frame_ocr_indexed=False,
          loft_candidates_done=True)

    manager = _manager()
    await manager._resume_incomplete()

    assert await _queued(manager, "frame_ocr") == [("yt1", -10)]
    assert await _queued(manager, FETCH) == []


# ---------------------------------------------------------------------------
# §3 - reindex frame_ocr, CLIP regenerate, prioritize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["youtube", "web", None])
async def test_spec_addon_117_a_frame_ocr_reindex_of_a_loft_opens_the_read_and_fetches_at_100(
    db, tmp_path, monkeypatch, source,
):
    patch_features(monkeypatch, video_frame_ocr="manual")
    _loft(db, tmp_path, "loft1", source, failures=3)
    manager = _manager()

    payload = await _reindex(manager, monkeypatch, "loft1", "frame_ocr")

    assert payload["status"] == "accepted"
    assert "frame_ocr" in payload["tasks_reset"]
    assert bool(db.flag("loft1", "frame_ocr_read")) is False
    assert bool(db.flag("loft1", "frame_ocr_indexed")) is False
    assert db.flag("loft1", "loft_fetch_failures") == 0
    assert await _queued(manager, FETCH) == [("loft1", 100)]
    assert await _queued(manager, "frame_ocr") == []


async def test_spec_addon_117_a_frame_ocr_reindex_of_a_loft_of_source_other_is_422(
    db, tmp_path, monkeypatch,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _loft(db, tmp_path, "other1", "other")
    _loft(db, tmp_path, "yt1", "youtube")
    manager = _manager()

    await _reindex(manager, monkeypatch, "yt1", "frame_ocr")
    with pytest.raises(HTTPException) as excinfo:
        await _reindex(manager, monkeypatch, "other1", "frame_ocr")

    assert excinfo.value.status_code == 422
    assert bool(db.flag("other1", "frame_ocr_read")) is True
    assert await _queued(manager, FETCH) == [("yt1", 100)]


async def test_spec_addon_117_i13_a_reindex_while_the_fetch_is_queued_raises_it_to_100(
    db, tmp_path, monkeypatch,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _loft(db, tmp_path, "yt1", "youtube", frame_scan_indexed=False, failures=2)
    manager = _manager()
    await manager._resume_incomplete()

    payload = await _reindex(manager, monkeypatch, "yt1", "frame_ocr")

    assert payload["status"] == "already_queued"
    assert bool(db.flag("yt1", "frame_ocr_read")) is False
    assert db.flag("yt1", "loft_fetch_failures") == 0
    assert await _queued(manager, FETCH) == [("yt1", 100)]


@pytest.mark.parametrize("source", ["youtube", "other", None])
async def test_spec_addon_117_i13_a_clip_regenerate_of_a_loft_unclassifies_it_and_fetches_at_100(
    db, tmp_path, monkeypatch, source,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _loft(db, tmp_path, "loft1", source, failures=3)
    manager = _manager()

    await _reindex(manager, monkeypatch, "loft1", "clip")

    assert db.flag("loft1", "loft_source") is None
    assert bool(db.flag("loft1", "frame_scan_indexed")) is False
    assert db.flag("loft1", "loft_fetch_failures") == 0
    assert await _queued(manager, FETCH) == [("loft1", 100)]
    assert await _queued(manager, "frame_scan") == []


async def test_spec_addon_117_i13_a_clip_regenerate_raises_a_queued_backlog_fetch(
    db, tmp_path, monkeypatch,
):
    """R5-2: one fetch at 100, not a second beside the -10 entry."""
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _loft(db, tmp_path, "loft1", None, frame_scan_indexed=False)
    manager = _manager()
    await manager._resume_incomplete()

    payload = await _reindex(manager, monkeypatch, "loft1", "clip")

    assert payload["status"] == "already_queued"
    assert db.flag("loft1", "loft_source") is None
    assert bool(db.flag("loft1", "frame_scan_indexed")) is False
    assert db.flag("loft1", "loft_fetch_failures") == 0
    assert await _queued(manager, FETCH) == [("loft1", 100)]


async def test_spec_addon_117_i13_a_clip_regenerate_fetches_although_the_clip_task_is_queued(
    db, tmp_path, monkeypatch,
):
    """R5-3: the queued CLIP task must not swallow the fetch the request asks for."""
    from app.indexer import IndexTask, TaskType

    patch_features(monkeypatch, video_frame_ocr="on_index")
    _loft(db, tmp_path, "loft1", "youtube", frame_scan_indexed=True, failures=3)
    manager = _manager()
    await manager._enqueue(
        IndexTask(file_id="loft1", task_type=TaskType.CLIP, priority=0), force=True,
    )

    payload = await _reindex(manager, monkeypatch, "loft1", "clip")

    assert payload["status"] == "accepted"
    assert "clip" in payload["tasks_reset"]
    assert db.flag("loft1", "loft_source") is None
    assert bool(db.flag("loft1", "frame_scan_indexed")) is False
    assert db.flag("loft1", "loft_fetch_failures") == 0
    assert await _queued(manager, FETCH) == [("loft1", 100)]


@pytest.mark.parametrize("source", ["youtube", "web"])
async def test_spec_addon_117_prioritize_of_a_stored_read_queues_the_rebuild_only_task(
    db, tmp_path, monkeypatch, source,
):
    patch_features(monkeypatch, video_frame_ocr="manual")
    _loft(db, tmp_path, "yt1", source, frame_ocr_read=True, frame_ocr_indexed=False,
          loft_candidates_done=True)
    manager = _manager()

    await manager.prioritize("yt1")

    assert await _queued(manager, "frame_ocr") == [("yt1", 100)]
    assert await _queued(manager, FETCH) == []


async def test_spec_addon_117_prioritize_of_a_wanted_scan_queues_a_fetch_at_100(
    db, tmp_path, monkeypatch,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _loft(db, tmp_path, "yt1", "youtube", frame_scan_indexed=False)
    manager = _manager()

    await manager.prioritize("yt1")

    assert await _queued(manager, FETCH) == [("yt1", 100)]
    assert await _queued(manager, "frame_scan") == []


async def test_spec_addon_117_is_queued_for_a_lofts_frame_tasks_answers_from_its_fetch(
    db, tmp_path, monkeypatch,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _loft(db, tmp_path, "yt1", "youtube", frame_scan_indexed=False,
          loft_candidates_done=True)
    _loft(db, tmp_path, "yt2", "youtube", loft_candidates_done=True)
    manager = _manager()
    await manager._resume_incomplete()

    assert manager.is_queued("yt1", "frame_scan") is True
    assert manager.is_queued("yt1", "frame_ocr") is True
    assert manager.is_queued("yt2", "frame_scan") is False


# ---------------------------------------------------------------------------
# Item 1.7 - status (I10)
# ---------------------------------------------------------------------------


def _status(monkeypatch):
    from app import main

    manager = main.IndexManager()
    monkeypatch.setattr(main.dependencies, "get_index_manager", lambda: manager)
    monkeypatch.setattr(main.dependencies, "_video_visual_worker", None)
    return main.status_endpoint(), manager


@pytest.mark.parametrize("mode, ocr_pending", [("on_index", 1), ("manual", 0)])
def test_spec_addon_117_i10_status_counts_youtube_web_and_unclassified_lofts_never_other(
    db, tmp_path, monkeypatch, mode, ocr_pending,
):
    patch_features(monkeypatch, video_frame_ocr=mode)
    _video(db, tmp_path, "base")
    baseline, _ = _status(monkeypatch)
    _loft(db, tmp_path, "yt-done", "youtube")
    _loft(db, tmp_path, "yt-todo", "youtube", frame_scan_indexed=False)
    _loft(db, tmp_path, "web-todo", "web", frame_scan_indexed=False,
          frame_ocr_read=False, frame_ocr_indexed=False)
    _loft(db, tmp_path, "unclassified", None, frame_ocr_read=False, frame_ocr_indexed=False)
    _loft(db, tmp_path, "other", "other", frame_scan_indexed=False,
          frame_ocr_read=False, frame_ocr_indexed=False)

    status, _ = _status(monkeypatch)

    assert status.indexed["frame_scan"] - baseline.indexed["frame_scan"] == 2
    assert status.pending["frame_scan"] - baseline.pending["frame_scan"] == 2
    assert status.pending["frame_ocr"] - baseline.pending["frame_ocr"] == 2 * ocr_pending
    assert status.pending["total"] - baseline.pending["total"] == 2 + 2 * ocr_pending


def test_spec_addon_117_the_queue_breakdown_has_a_loft_media_row(db, monkeypatch):
    _, manager = _status(monkeypatch)

    assert manager.get_queue_breakdown()[FETCH] == {"waiting": 0, "processing": []}


# ---------------------------------------------------------------------------
# Item 1.8 - index-details (I12)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "file_id, source, expected",
    [
        ("yt1", "youtube", "youtube"),
        ("web1", "web", "web"),
        ("other1", "other", "other"),
        ("loft1", None, None),
        ("vid", None, None),
    ],
)
async def test_spec_addon_117_i12_index_details_carries_the_loft_source(
    db, tmp_path, file_id, source, expected,
):
    from app.routers.files import get_index_details

    if file_id == "vid":
        _video(db, tmp_path, "vid")
    else:
        _loft(db, tmp_path, file_id, source)

    response = await get_index_details(file_id=file_id, drive="drive1")

    assert "loft_source" in type(response).model_fields
    assert response.loft_source == expected
