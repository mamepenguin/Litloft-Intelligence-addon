"""Queueing, modes, priorities and manual reindex for the two new OCR passes.

SPEC-ADDON-066 A2/A11/A12/A13 (video frames) and SPEC-ADDON-069 B2/B9/B10/B11
(scanned PDF pages): the two features, their task types, the on_index / manual /
false modes, backlog priority, manual reindex, pending counts, and the rule that
a file of the wrong type is marked done and never queued.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from tests.ocr_fixtures import (
    SearchDB,
    build_search_db,
    patch_features,
    pillow_real,  # noqa: F401  - the usefixtures marker resolves it
    seed_indexed_file,
)

pytestmark = pytest.mark.usefixtures("pillow_real")

ALL_DONE = dict(
    metadata_indexed=True,
    clip_indexed=True,
    clip_thumbnail_indexed=True,
    whisper_indexed=True,
    text_indexed=True,
    tfidf_keywords_indexed=True,
)

VIDEO_MIMES = ["video/mp4", "video/webm", "video/quicktime", "video/x-matroska"]

# (task name on TaskType, flag the mode gates, mime type, request name)
CASES = [
    pytest.param("FRAME_OCR", "frame_ocr_indexed", mime, "frame_ocr", id=f"video-{mime}")
    for mime in VIDEO_MIMES
] + [
    pytest.param(
        "PAGE_OCR", "page_ocr_indexed", "application/pdf", "page_ocr", id="pdf"
    ),
]


@pytest.fixture()
def search_db(tmp_path, monkeypatch) -> SearchDB:
    return build_search_db(tmp_path, monkeypatch)


def make_manager():
    from app.indexer import IndexManager

    return IndexManager()


def task_type(name: str):
    from app.indexer import TaskType

    return getattr(TaskType, name)


async def collect(manager, name: str) -> list:
    return await manager._collect_batch(task_type(name), 10)


def file_data(file_id: str, mime_type: str) -> dict:
    return {"id": file_id, "mime_type": mime_type}


def seed_video(search_db, tmp_path, file_id="vid", **flags) -> None:
    seed_indexed_file(
        search_db, file_id, "video/mp4", tmp_path / f"{file_id}.mp4",
        file_type="video", **{**ALL_DONE, **flags},
    )


def seed_pdf(search_db, tmp_path, file_id="pdf", **flags) -> None:
    path = tmp_path / f"{file_id}.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    seed_indexed_file(
        search_db, file_id, "application/pdf", path,
        file_type="document", **{**ALL_DONE, **flags},
    )


# ---------------------------------------------------------------------------
# SPEC-ADDON-066 A12 / SPEC-ADDON-069 B10 - the settings
# ---------------------------------------------------------------------------


def test_spec_addon_066_a12_video_frame_ocr_defaults_to_on_index():
    from app.config import FeaturesConfig

    assert FeaturesConfig().video_frame_ocr == "on_index"


def test_spec_addon_069_b10_scanned_pdf_ocr_defaults_to_on_index():
    from app.config import FeaturesConfig

    assert FeaturesConfig().scanned_pdf_ocr == "on_index"


def test_spec_addon_066_a2_the_two_task_types_exist():
    assert task_type("FRAME_OCR").value == "frame_ocr"
    assert task_type("PAGE_OCR").value == "page_ocr"


# ---------------------------------------------------------------------------
# SPEC-ADDON-066 A2 / SPEC-ADDON-069 B2 - a new file is queued at 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task, flag, mime, request_name", CASES)
async def test_spec_addon_066_a2_a_new_file_is_queued_at_priority_zero(
    search_db, monkeypatch, task, flag, mime, request_name,
):
    """SPEC-ADDON-066 A2, SPEC-ADDON-069 B2: on_index queues the new pass."""
    patch_features(
        monkeypatch, video_frame_ocr="on_index", scanned_pdf_ocr="on_index"
    )
    manager = make_manager()

    await manager._queue_file_tasks(file_data("f1", mime))

    assert [
        (t.file_id, t.priority) for t in await collect(manager, task)
    ] == [("f1", 0)]


@pytest.mark.parametrize("mode", ["manual", "false"])
@pytest.mark.parametrize("task, flag, mime, request_name", CASES)
async def test_spec_addon_066_a2_outside_on_index_a_new_file_is_not_queued(
    search_db, monkeypatch, mode, task, flag, mime, request_name,
):
    """SPEC-ADDON-066 I7: manual and false never queue a new file."""
    patch_features(monkeypatch, video_frame_ocr=mode, scanned_pdf_ocr=mode)
    manager = make_manager()

    await manager._queue_file_tasks(file_data("f1", mime))

    assert await collect(manager, task) == []


# ---------------------------------------------------------------------------
# SPEC-ADDON-066 A2/A6 / SPEC-ADDON-069 B2 - the backlog
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task, flag, mime, request_name", CASES)
async def test_spec_addon_066_a2_the_backlog_is_queued_at_minus_ten(
    search_db, tmp_path, monkeypatch, task, flag, mime, request_name,
):
    """SPEC-ADDON-066 A2, SPEC-ADDON-069 B2: an old file waits at -10."""
    patch_features(
        monkeypatch, video_frame_ocr="on_index", scanned_pdf_ocr="on_index"
    )
    if mime == "application/pdf":
        seed_pdf(search_db, tmp_path, **{flag: False})
    else:
        seed_video(search_db, tmp_path, **{flag: False})
    manager = make_manager()

    await manager._resume_incomplete()

    assert [
        (t.file_id, t.priority) for t in await collect(manager, task)
    ] == [("vid" if mime != "application/pdf" else "pdf", -10)]


@pytest.mark.parametrize("mode", ["manual", "false"])
async def test_spec_addon_066_a6_a_rebuild_only_task_is_queued_under_every_mode(
    search_db, tmp_path, monkeypatch, mode,
):
    """SPEC-ADDON-066 A6/I7: stored text is rebuilt without reading frames."""
    patch_features(monkeypatch, video_frame_ocr=mode)
    seed_video(search_db, tmp_path, frame_ocr_read=True, frame_ocr_indexed=False)
    manager = make_manager()

    await manager._resume_incomplete()

    assert [
        (t.file_id, t.priority) for t in await collect(manager, "FRAME_OCR")
    ] == [("vid", -10)]


# ---------------------------------------------------------------------------
# SPEC-ADDON-066 A13 / SPEC-ADDON-069 B11 - a file of the wrong type
# ---------------------------------------------------------------------------


async def test_spec_addon_066_a13_a_non_video_is_marked_done_and_never_queued(
    search_db, tmp_path, monkeypatch,
):
    """SPEC-ADDON-066 A13: the frame flags are only meaningful for videos."""
    patch_features(monkeypatch, video_frame_ocr="on_index")
    document = tmp_path / "notes.txt"
    document.write_text("plain text")
    seed_indexed_file(
        search_db, "doc1", "text/plain", document, **ALL_DONE,
        frame_ocr_read=False, frame_ocr_indexed=False,
    )
    manager = make_manager()

    await manager._resume_incomplete()

    assert bool(search_db.flag("doc1", "frame_ocr_read")) is True
    assert bool(search_db.flag("doc1", "frame_ocr_indexed")) is True
    assert await collect(manager, "FRAME_OCR") == []


async def test_spec_addon_069_b11_a_non_pdf_is_marked_done_and_never_queued(
    search_db, tmp_path, monkeypatch,
):
    """SPEC-ADDON-069 B11: page_ocr_indexed is only meaningful for PDFs."""
    patch_features(monkeypatch, scanned_pdf_ocr="on_index")
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    seed_indexed_file(
        search_db, "vid1", "video/mp4", video, file_type="video", **ALL_DONE,
        page_ocr_indexed=False,
    )
    manager = make_manager()

    await manager._resume_incomplete()

    assert bool(search_db.flag("vid1", "page_ocr_indexed")) is True
    assert await collect(manager, "PAGE_OCR") == []


# ---------------------------------------------------------------------------
# SPEC-ADDON-066 A11 / SPEC-ADDON-069 B9 - manual run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task, flag, mime, request_name", CASES)
async def test_spec_addon_066_a11_prioritize_includes_the_task_when_it_is_pending(
    search_db, tmp_path, monkeypatch, task, flag, mime, request_name,
):
    """SPEC-ADDON-066 A11, SPEC-ADDON-069 B9: prioritize queues it at 100."""
    patch_features(
        monkeypatch, video_frame_ocr="manual", scanned_pdf_ocr="manual"
    )
    if mime == "application/pdf":
        seed_pdf(search_db, tmp_path, **{flag: False})
    else:
        seed_video(search_db, tmp_path, **{flag: False})
    manager = make_manager()

    assert await manager.prioritize("pdf" if mime == "application/pdf" else "vid")
    assert [
        (t.file_id, t.priority) for t in await collect(manager, task)
    ] == [("pdf" if mime == "application/pdf" else "vid", 100)]


@pytest.mark.parametrize("task, flag, mime, request_name", CASES)
async def test_spec_addon_066_a11_reindex_accepts_the_task_and_resets_its_flags(
    search_db, tmp_path, monkeypatch, task, flag, mime, request_name,
):
    """SPEC-ADDON-066 A11, SPEC-ADDON-069 B9: a manual request resets and queues."""
    from app.routers import files as files_router

    patch_features(
        monkeypatch, video_frame_ocr="on_index", scanned_pdf_ocr="on_index"
    )
    file_id = "pdf" if mime == "application/pdf" else "vid"
    if mime == "application/pdf":
        seed_pdf(search_db, tmp_path, **{flag: True})
    else:
        seed_video(search_db, tmp_path, **{flag: True})
    manager = make_manager()
    monkeypatch.setattr(files_router, "get_index_manager", lambda: manager)

    result = await files_router.reindex_file(
        file_id, body={"tasks": [request_name]}, drive="drive1"
    )

    payload = result if isinstance(result, dict) else result.model_dump()
    assert payload["status"] == "accepted"
    assert request_name in payload["tasks_reset"]
    assert bool(search_db.flag(file_id, flag)) is False
    if request_name == "frame_ocr":
        assert bool(search_db.flag(file_id, "frame_ocr_read")) is False
    assert [
        (t.file_id, t.priority) for t in await collect(manager, task)
    ] == [(file_id, 100)]


@pytest.mark.parametrize("task, flag, mime, request_name", CASES)
async def test_spec_addon_066_a11_reindex_refuses_the_task_when_the_mode_is_false(
    search_db, tmp_path, monkeypatch, task, flag, mime, request_name,
):
    """SPEC-ADDON-066 I7, SPEC-ADDON-069 B9: off means 422, not ignored."""
    from app.routers import files as files_router

    patch_features(
        monkeypatch, video_frame_ocr="false", scanned_pdf_ocr="false"
    )
    file_id = "pdf" if mime == "application/pdf" else "vid"
    if mime == "application/pdf":
        seed_pdf(search_db, tmp_path, **{flag: True})
    else:
        seed_video(search_db, tmp_path, **{flag: True})
    manager = make_manager()
    monkeypatch.setattr(files_router, "get_index_manager", lambda: manager)

    with pytest.raises(HTTPException) as excinfo:
        await files_router.reindex_file(
            file_id, body={"tasks": [request_name]}, drive="drive1"
        )

    assert excinfo.value.status_code == 422
    assert bool(search_db.flag(file_id, flag)) is True
    assert await collect(manager, task) == []


# ---------------------------------------------------------------------------
# SPEC-ADDON-066 §3 / SPEC-ADDON-069 §3 - pending counts and mode switches
# ---------------------------------------------------------------------------


def test_spec_addon_066_pending_frame_and_page_counts_only_under_on_index(
    search_db, tmp_path, monkeypatch,
):
    """SPEC-ADDON-066 §3, SPEC-ADDON-069 §3: pending only while on_index."""
    patch_features(
        monkeypatch, video_frame_ocr="on_index", scanned_pdf_ocr="on_index"
    )
    seed_video(search_db, tmp_path, frame_ocr_indexed=False)
    seed_pdf(search_db, tmp_path, page_ocr_indexed=False)
    manager = make_manager()

    status = manager.get_index_status()
    assert (status.pending_frame_ocr, status.pending_page_ocr) == (1, 1)

    patch_features(
        monkeypatch, video_frame_ocr="manual", scanned_pdf_ocr="manual"
    )
    off = make_manager().get_index_status()
    assert (off.pending_frame_ocr, off.pending_page_ocr) == (0, 0)


@pytest.mark.parametrize("task, flag, mime, request_name", CASES)
async def test_spec_addon_066_a12_a_mode_switch_keeps_the_stored_text(
    search_db, tmp_path, monkeypatch, task, flag, mime, request_name,
):
    """SPEC-ADDON-066 A12/I7, SPEC-ADDON-069 B10: off removes nothing."""
    patch_features(
        monkeypatch, video_frame_ocr="false", scanned_pdf_ocr="false"
    )
    file_id = "pdf" if mime == "application/pdf" else "vid"
    if mime == "application/pdf":
        seed_pdf(search_db, tmp_path, **{flag: True})
        search_db.execute(
            "INSERT INTO pdf_page_ocr_texts(file_id, page, text) "
            "VALUES('pdf', 1, 'kept page text')"
        )
    else:
        seed_video(search_db, tmp_path, **{flag: True})
        search_db.execute(
            "INSERT INTO video_frame_texts"
            "(file_id, chunk_index, start_time, end_time, text) "
            "VALUES('vid', 0, 0.0, 10.0, 'kept frame text')"
        )
    manager = make_manager()

    await manager._resume_incomplete()

    assert bool(search_db.flag(file_id, flag)) is True
    if mime == "application/pdf":
        assert search_db.rows(
            "SELECT text FROM pdf_page_ocr_texts WHERE file_id = 'pdf'"
        ) == [("kept page text",)]
    else:
        assert search_db.rows(
            "SELECT text FROM video_frame_texts WHERE file_id = 'vid'"
        ) == [("kept frame text",)]
    assert await collect(manager, task) == []
