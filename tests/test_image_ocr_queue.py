"""Queueing, manual reindex and mode tests for picture OCR (SPEC-ADDON-050)."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from tests.ocr_fixtures import (
    SearchDB,
    build_search_db,
    patch_features,
    pillow_real,  # noqa: F401  - the usefixtures marker resolves it
    picture,
    seed_indexed_file,
    seed_ocr_rows,
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


@pytest.fixture()
def search_db(tmp_path, monkeypatch) -> SearchDB:
    return build_search_db(tmp_path, monkeypatch)


def make_manager():
    from app.indexer import IndexManager

    return IndexManager()


def file_data(file_id: str, mime_type: str) -> dict:
    return {"id": file_id, "mime_type": mime_type}


async def ocr_tasks(manager) -> list:
    from app.indexer import TaskType

    return await manager._collect_batch(TaskType.OCR, 10)


def seed_picture(search_db, tmp_path, file_id="pic1", *, path=None) -> None:
    seed_indexed_file(
        search_db, file_id, "image/png", path or picture(tmp_path / f"{file_id}.png"),
        **ALL_DONE,
    )


# ---------------------------------------------------------------------------
# I4 - queueing and priorities
# ---------------------------------------------------------------------------


def test_the_default_mode_is_on_index(monkeypatch):
    """SPEC-ADDON-050 I4: a picture is read as part of indexing by default."""
    from app.config import FeaturesConfig

    assert FeaturesConfig().image_ocr == "on_index"


async def test_a_new_picture_is_queued_at_priority_zero(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-050 I4: on_index queues OCR alongside CLIP, at 0."""
    patch_features(monkeypatch, image_ocr="on_index")
    manager = make_manager()

    await manager._queue_file_tasks(file_data("pic1", "image/png"))

    tasks = await ocr_tasks(manager)
    assert [(task.file_id, task.priority) for task in tasks] == [("pic1", 0)]


@pytest.mark.parametrize("mime", sorted([
    "image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp",
    "image/heic", "image/heif",
]))
async def test_every_picture_mime_is_queued(search_db, monkeypatch, mime):
    """SPEC-ADDON-050 I4: the whole picture set, HEIC and HEIF included."""
    patch_features(monkeypatch, image_ocr="on_index")
    manager = make_manager()

    await manager._queue_file_tasks(file_data("pic1", mime))

    assert [task.file_id for task in await ocr_tasks(manager)] == ["pic1"]


async def test_the_backlog_is_queued_at_minus_ten(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-050 I4: an existing picture waits behind new work."""
    patch_features(monkeypatch, image_ocr="on_index")
    seed_picture(search_db, tmp_path)
    manager = make_manager()

    await manager._resume_incomplete()

    tasks = await ocr_tasks(manager)
    assert [(task.file_id, task.priority) for task in tasks] == [("pic1", -10)]


async def test_tasks_are_taken_a_hundred_then_zero_then_minus_ten(
    search_db, monkeypatch
):
    """SPEC-ADDON-050 I4: the three priorities in the order they are taken."""
    from app.indexer import IndexTask, TaskType

    patch_features(monkeypatch, image_ocr="on_index")
    manager = make_manager()
    await manager._enqueue(
        IndexTask(file_id="backlog", task_type=TaskType.OCR, priority=-10)
    )
    await manager._enqueue(
        IndexTask(file_id="new", task_type=TaskType.OCR, priority=0)
    )
    await manager._enqueue(
        IndexTask(file_id="requested", task_type=TaskType.OCR, priority=100)
    )

    assert [task.file_id for task in await ocr_tasks(manager)] == [
        "requested", "new", "backlog",
    ]


@pytest.mark.parametrize("mode", ["manual", "false"])
async def test_other_modes_do_not_queue_on_index_or_the_backlog(
    search_db, tmp_path, monkeypatch, mode
):
    """SPEC-ADDON-050 I4/I10: only a request queues in manual; never in false."""
    patch_features(monkeypatch, image_ocr=mode)
    seed_picture(search_db, tmp_path)
    manager = make_manager()

    await manager._queue_file_tasks(file_data("pic1", "image/png"))
    await manager._resume_incomplete()

    assert await ocr_tasks(manager) == []
    assert bool(search_db.flag("pic1", "ocr_indexed")) is False


async def test_turning_it_off_keeps_the_stored_text(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-050 I10: a mode switch removes nothing already written."""
    path = picture(tmp_path / "pic.png")
    seed_picture(search_db, tmp_path, path=path)
    seed_ocr_rows(search_db, "pic1", ("already searchable",))
    search_db.set_flag("pic1", "ocr_indexed", True)
    patch_features(monkeypatch, image_ocr="false")
    manager = make_manager()

    await manager._resume_incomplete()

    assert bool(search_db.flag("pic1", "ocr_indexed")) is True
    assert search_db.ocr_fts("pic1") == [(0, "3", "already searchable")]
    assert await ocr_tasks(manager) == []


# ---------------------------------------------------------------------------
# I4 - the per-file reindex path
# ---------------------------------------------------------------------------


async def test_reindex_in_manual_mode_resets_the_flag_and_queues_at_100(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-050 I4: a request is the manual way in, at priority 100."""
    from app.routers import files as files_router

    patch_features(monkeypatch, image_ocr="manual")
    seed_picture(search_db, tmp_path)
    search_db.set_flag("pic1", "ocr_indexed", True)
    manager = make_manager()
    monkeypatch.setattr(files_router, "get_index_manager", lambda: manager)

    result = await files_router.reindex_file(
        "pic1", body={"tasks": ["ocr"]}, drive="drive1"
    )

    payload = result if isinstance(result, dict) else result.model_dump()
    assert payload["status"] == "accepted"
    assert payload["tasks_reset"] == ["ocr"]
    assert bool(search_db.flag("pic1", "ocr_indexed")) is False
    assert [(task.file_id, task.priority) for task in await ocr_tasks(manager)] == [
        ("pic1", 100)
    ]


async def test_reindex_answers_422_for_ocr_when_the_mode_is_false(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-050 I4/I10: off means the request is refused, not ignored."""
    from app.routers import files as files_router

    patch_features(monkeypatch, image_ocr="false")
    seed_picture(search_db, tmp_path)
    search_db.set_flag("pic1", "ocr_indexed", True)
    manager = make_manager()
    monkeypatch.setattr(files_router, "get_index_manager", lambda: manager)

    with pytest.raises(HTTPException) as excinfo:
        await files_router.reindex_file(
            "pic1", body={"tasks": ["ocr"]}, drive="drive1"
        )

    assert excinfo.value.status_code == 422
    assert bool(search_db.flag("pic1", "ocr_indexed")) is True
    assert await ocr_tasks(manager) == []


async def test_reindex_answers_422_for_ocr_on_a_file_that_is_not_a_picture(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-050 I4: OCR has no meaning for a document."""
    from app.routers import files as files_router

    patch_features(monkeypatch, image_ocr="on_index")
    document = tmp_path / "notes.txt"
    document.write_text("plain text")
    seed_indexed_file(search_db, "doc1", "text/plain", document, **ALL_DONE)
    manager = make_manager()
    monkeypatch.setattr(files_router, "get_index_manager", lambda: manager)

    with pytest.raises(HTTPException) as excinfo:
        await files_router.reindex_file(
            "doc1", body={"tasks": ["ocr"]}, drive="drive1"
        )

    assert excinfo.value.status_code == 422
    assert await ocr_tasks(manager) == []


async def test_prioritize_includes_ocr_only_when_it_is_on(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-050 I4: prioritize is a way in for manual, not for false."""
    patch_features(monkeypatch, image_ocr="manual")
    seed_picture(search_db, tmp_path)
    manager = make_manager()

    assert await manager.prioritize("pic1") is True
    assert [(task.file_id, task.priority) for task in await ocr_tasks(manager)] == [
        ("pic1", 100)
    ]

    patch_features(monkeypatch, image_ocr="false")
    off_manager = make_manager()
    assert await off_manager.prioritize("pic1") is True
    assert await ocr_tasks(off_manager) == []


# ---------------------------------------------------------------------------
# I9 - files that are not pictures
# ---------------------------------------------------------------------------


async def test_a_non_picture_is_marked_done_by_the_sweep_and_never_queued(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-050 I9: nothing pending for a file OCR cannot read."""
    patch_features(monkeypatch, image_ocr="on_index")
    document = tmp_path / "notes.txt"
    document.write_text("plain text")
    seed_indexed_file(search_db, "doc1", "text/plain", document, **ALL_DONE)
    manager = make_manager()

    await manager._resume_incomplete()
    await manager._queue_file_tasks(file_data("doc2", "application/pdf"))

    assert bool(search_db.flag("doc1", "ocr_indexed")) is True
    assert await ocr_tasks(manager) == []


@pytest.mark.parametrize(
    "mime",
    ["text/plain", "text/markdown", "application/pdf", "video/mp4", "audio/mpeg"],
)
async def test_no_document_or_media_type_gets_an_ocr_task(
    search_db, monkeypatch, mime
):
    """SPEC-ADDON-050 I9: only the picture set is OCR work."""
    patch_features(monkeypatch, image_ocr="on_index")
    manager = make_manager()

    await manager._queue_file_tasks(file_data("f1", mime))

    assert await ocr_tasks(manager) == []
