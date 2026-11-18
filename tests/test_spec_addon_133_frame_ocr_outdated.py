"""SPEC-ADDON-133: a frame read stored under the old selection rule is outdated
under ``on_index``, and is queued, counted, shown and reindexed as pending.

Item 1 steps 6, 7 and 9, §3, §4, I6, I7, I10, and the revised SPEC-ADDON-117 I10.
"""

from __future__ import annotations

import json

import pytest
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text

from tests.ocr_fixtures import (
    SearchDB,
    build_search_db,
    patch_features,
    pillow_real,  # noqa: F401  - the usefixtures marker resolves it
    seed_indexed_file,
)
from tests.test_frame_page_ocr_schema import init_database  # noqa: F401 - fixture

pytestmark = pytest.mark.usefixtures("pillow_real")

LOFT_MIME = "application/vnd.litloft.loft+json"
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


async def _queued(manager, task: str = "frame_ocr") -> list[tuple[str, int]]:
    from app.indexer import TaskType

    return sorted(
        (t.file_id, t.priority)
        for t in await manager._collect_batch(TaskType(task), 10)
    )


def _video(db, tmp_path, file_id: str = "vid", **flags) -> None:
    seed_indexed_file(
        db, file_id, "video/mp4", tmp_path / f"{file_id}.mp4", file_type="video",
        **{**ALL_DONE, **flags},
    )


def _loft(db, tmp_path, file_id: str, **flags) -> None:
    path = tmp_path / f"{file_id}.loft"
    path.write_text('{"provider": "youtube", "url": "https://www.youtube.com/watch?v=x"}')
    seed_indexed_file(db, file_id, LOFT_MIME, path, file_type="video", **{**ALL_DONE, **flags})
    db.execute(
        "UPDATE indexed_files SET loft_source = 'youtube' WHERE file_id = :f", f=file_id,
    )


async def _reindex(manager, monkeypatch, file_id: str):
    from app.routers import files as files_router

    monkeypatch.setattr(files_router, "get_index_manager", lambda: manager)
    result = await files_router.reindex_file(
        file_id, body={"tasks": ["frame_ocr"]}, drive="drive1",
    )
    if isinstance(result, dict):
        return result
    if isinstance(result, JSONResponse):
        return json.loads(result.body)
    return result.model_dump()


def _status(monkeypatch):
    from app import main

    manager = main.IndexManager()
    monkeypatch.setattr(main.dependencies, "get_index_manager", lambda: manager)
    monkeypatch.setattr(main.dependencies, "_video_visual_worker", None)
    return main.status_endpoint()


async def _details_frame_ocr(file_id: str) -> bool:
    from app.routers.files import get_index_details

    response = await get_index_details(file_id=file_id, drive="drive1")
    payload = response if isinstance(response, dict) else response.model_dump()
    return bool(payload["status"]["frame_ocr"])


def _rule_flags(db, file_id: str) -> tuple:
    return (
        bool(db.flag(file_id, "frame_ocr_read")),
        bool(db.flag(file_id, "frame_ocr_indexed")),
        db.flag(file_id, "frame_ocr_rule"),
    )


# ---------------------------------------------------------------------------
# §4, I6 - the column and its migration
# ---------------------------------------------------------------------------


def _columns(engine) -> dict[str, tuple[str, int]]:
    with engine.connect() as conn:
        return {
            row[1]: (str(row[2]).upper(), row[3])
            for row in conn.execute(text("PRAGMA table_info(indexed_files)")).fetchall()
        }


def test_spec_addon_133_i6_the_migration_adds_a_nullable_rule_column_twice_leaving_rows_null(
    init_database,  # noqa: F811
):
    start, path = init_database
    engine = create_engine(f"sqlite:///{path}")
    from app.database import Base

    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        if "frame_ocr_rule" in _columns(engine):
            conn.execute(text("ALTER TABLE indexed_files DROP COLUMN frame_ocr_rule"))
        conn.execute(text(
            "INSERT INTO indexed_files(file_id, drive, filename, file_path, "
            "file_type, mime_type, file_size, active, title, description, "
            "tags_text, indexed_at, updated_at, frame_ocr_read, frame_ocr_indexed) "
            "VALUES('vid1', 'drive1', 'v.mp4', '/drives/drive1/v.mp4', "
            "'video', 'video/mp4', 10, 1, '', '', '', "
            "'2026-11-06 10:00:00', '2026-11-06 10:00:00', 1, 1)"
        ))
    engine.dispose()

    start()

    reader = create_engine(f"sqlite:///{path}")
    columns = _columns(reader)
    assert "frame_ocr_rule" in columns
    assert columns["frame_ocr_rule"] == ("INTEGER", 0)
    with reader.begin() as conn:
        assert conn.execute(text(
            "SELECT frame_ocr_rule, frame_ocr_read, frame_ocr_indexed "
            "FROM indexed_files WHERE file_id = 'vid1'"
        )).fetchone() == (None, 1, 1)
        conn.execute(text("UPDATE indexed_files SET frame_ocr_rule = 2"))
    reader.dispose()

    start()

    reader = create_engine(f"sqlite:///{path}")
    with reader.connect() as conn:
        assert conn.execute(text(
            "SELECT frame_ocr_rule, frame_ocr_read, frame_ocr_indexed "
            "FROM indexed_files WHERE file_id = 'vid1'"
        )).fetchone() == (2, 1, 1)
    reader.dispose()


# ---------------------------------------------------------------------------
# Item 1 step 7, I7 - the backlog
# ---------------------------------------------------------------------------


async def test_spec_addon_133_i7_an_outdated_native_video_is_queued_at_minus_ten(
    db, tmp_path, monkeypatch,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _video(db, tmp_path, "old")
    manager = _manager()

    await manager._resume_incomplete()

    assert await _queued(manager) == [("old", -10)]
    assert _rule_flags(db, "old") == (True, True, None)


async def test_spec_addon_133_i7_a_rule_two_read_a_loft_and_an_inactive_video_are_not_outdated(
    db, tmp_path, monkeypatch,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _video(db, tmp_path, "old")
    _video(db, tmp_path, "new", frame_ocr_rule=2)
    _video(db, tmp_path, "below", frame_ocr_rule=1)
    _video(db, tmp_path, "gone", active=False)
    _loft(db, tmp_path, "yt1")
    manager = _manager()

    await manager._resume_incomplete()

    assert await _queued(manager) == [("below", -10), ("old", -10)]


@pytest.mark.parametrize("mode", ["manual", "false"])
async def test_spec_addon_133_i7_outside_on_index_an_old_read_is_not_queued_until_the_mode_changes(
    db, tmp_path, monkeypatch, mode,
):
    _video(db, tmp_path, "old")

    patch_features(monkeypatch, video_frame_ocr=mode)
    off = _manager()
    await off._resume_incomplete()
    assert await _queued(off) == []

    patch_features(monkeypatch, video_frame_ocr="on_index")
    on = _manager()
    await on._resume_incomplete()
    assert await _queued(on) == [("old", -10)]
    assert _rule_flags(db, "old") == (True, True, None)


# ---------------------------------------------------------------------------
# Item 1 step 9, I7, revised SPEC-ADDON-117 I10 - the counts
# ---------------------------------------------------------------------------


def test_spec_addon_133_i7_get_status_counts_an_outdated_read_as_pending_only_under_on_index(
    db, tmp_path, monkeypatch,
):
    _video(db, tmp_path, "old")

    patch_features(monkeypatch, video_frame_ocr="on_index")
    assert _manager().get_index_status().pending_frame_ocr == 1

    patch_features(monkeypatch, video_frame_ocr="manual")
    assert _manager().get_index_status().pending_frame_ocr == 0


def test_spec_addon_133_i7_status_moves_an_outdated_read_from_indexed_to_pending(
    db, tmp_path, monkeypatch,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _video(db, tmp_path, "base", frame_ocr_read=False, frame_ocr_indexed=False)
    baseline = _status(monkeypatch)
    _video(db, tmp_path, "old")
    _loft(db, tmp_path, "yt1")

    status = _status(monkeypatch)

    assert status.pending["frame_ocr"] - baseline.pending["frame_ocr"] == 1
    assert status.pending["total"] - baseline.pending["total"] == 1
    assert status.indexed["frame_ocr"] - baseline.indexed["frame_ocr"] == 1


def test_spec_addon_133_i7_status_counts_a_rule_two_read_as_indexed(
    db, tmp_path, monkeypatch,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _video(db, tmp_path, "old")
    baseline = _status(monkeypatch)
    _video(db, tmp_path, "new", frame_ocr_rule=2)

    status = _status(monkeypatch)

    assert baseline.pending["frame_ocr"] == 1
    assert status.pending["frame_ocr"] == 1
    assert status.indexed["frame_ocr"] - baseline.indexed["frame_ocr"] == 1


def test_spec_addon_133_i7_outside_on_index_an_old_read_counts_as_indexed(
    db, tmp_path, monkeypatch,
):
    _video(db, tmp_path, "old")

    patch_features(monkeypatch, video_frame_ocr="on_index")
    on = _status(monkeypatch)
    patch_features(monkeypatch, video_frame_ocr="manual")
    off = _status(monkeypatch)

    assert (on.pending["frame_ocr"], on.indexed["frame_ocr"]) == (1, 0)
    assert (off.pending["frame_ocr"], off.indexed["frame_ocr"]) == (0, 1)


# ---------------------------------------------------------------------------
# Item 1 step 9, I7 - index details and prioritize
# ---------------------------------------------------------------------------


async def test_spec_addon_133_i7_index_details_show_an_outdated_read_as_not_done(
    db, tmp_path, monkeypatch,
):
    _video(db, tmp_path, "old")
    _loft(db, tmp_path, "yt1")

    patch_features(monkeypatch, video_frame_ocr="on_index")
    assert await _details_frame_ocr("old") is False
    assert await _details_frame_ocr("yt1") is True

    patch_features(monkeypatch, video_frame_ocr="manual")
    assert await _details_frame_ocr("old") is True


async def test_spec_addon_133_i7_index_details_show_a_rule_two_read_as_done(
    db, tmp_path, monkeypatch,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _video(db, tmp_path, "new", frame_ocr_rule=2)

    assert await _details_frame_ocr("new") is True


async def test_spec_addon_133_i7_prioritize_queues_an_outdated_read_at_100(
    db, tmp_path, monkeypatch,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _video(db, tmp_path, "old")
    manager = _manager()

    assert await manager.prioritize("old")
    assert await _queued(manager) == [("old", 100)]
    assert _rule_flags(db, "old") == (True, True, None)


async def test_spec_addon_133_i7_prioritize_leaves_a_rule_two_read_and_a_manual_old_read_alone(
    db, tmp_path, monkeypatch,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _video(db, tmp_path, "new", frame_ocr_rule=2)
    manager = _manager()
    await manager.prioritize("new")
    assert await _queued(manager) == []

    patch_features(monkeypatch, video_frame_ocr="manual")
    _video(db, tmp_path, "old")
    manual = _manager()
    await manual.prioritize("old")
    assert await _queued(manual) == []


# ---------------------------------------------------------------------------
# Item 1 step 9, I10 - reindex
# ---------------------------------------------------------------------------


async def test_spec_addon_133_i10_a_reindex_of_a_queued_outdated_read_raises_it_to_100(
    db, tmp_path, monkeypatch,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _video(db, tmp_path, "old")
    manager = _manager()
    await manager._resume_incomplete()

    payload = await _reindex(manager, monkeypatch, "old")

    assert payload["status"] == "already_queued"
    assert await _queued(manager) == [("old", 100)]
    assert _rule_flags(db, "old") == (True, True, None)


async def test_spec_addon_133_item_9_a_reindex_with_no_queued_task_resets_r_and_x_and_keeps_f(
    db, tmp_path, monkeypatch,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _video(db, tmp_path, "old")
    _video(db, tmp_path, "new", frame_ocr_rule=2)
    manager = _manager()

    for file_id in ("old", "new"):
        payload = await _reindex(manager, monkeypatch, file_id)
        assert payload["status"] == "accepted"
        assert "frame_ocr" in payload["tasks_reset"]

    assert _rule_flags(db, "old") == (False, False, None)
    assert _rule_flags(db, "new") == (False, False, 2)
    assert await _queued(manager) == [("new", 100), ("old", 100)]
