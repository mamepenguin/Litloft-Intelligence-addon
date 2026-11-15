"""SPEC-ADDON-111 item 9: a CLIP regenerate queues the scan of a native video
whose flag is false even when no scan of it is in any queue yet.

The sweep that normally queues an unscanned video may not have run, so the
regenerate is the only thing that would ever ask for the scan. It must queue it
at priority 100 and leave the flag false, exactly as it does for a video that
was scanned before.
"""

from __future__ import annotations

import pytest

from app.indexer import IndexManager, TaskType
from tests.ocr_fixtures import (
    SearchDB,
    build_search_db,
    pillow_real,  # noqa: F401 - the usefixtures marker resolves it
    seed_indexed_file,
)

pytestmark = pytest.mark.usefixtures("pillow_real")

SCAN = TaskType.FRAME_SCAN
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


@pytest.fixture()
def search_db(tmp_path, monkeypatch) -> SearchDB:
    return build_search_db(tmp_path, monkeypatch)


async def test_spec_addon_111_a_regenerate_queues_the_scan_of_a_never_queued_video(
    search_db, tmp_path, monkeypatch,
):
    from app.routers import files as files_router

    path = tmp_path / "vid.mp4"
    path.write_bytes(b"x")
    seed_indexed_file(
        search_db, "vid", "video/mp4", path, file_type="video",
        **{**ALL_DONE, "frame_scan_indexed": False},
    )
    manager = IndexManager()
    monkeypatch.setattr(files_router, "get_index_manager", lambda: manager)

    await files_router.reindex_file(
        "vid", body={"tasks": ["clip"]}, drive="drive1",
    )

    queued = [
        (t.file_id, t.priority)
        for t in await manager._collect_batch(SCAN, 10)
    ]
    assert queued == [("vid", 100)]
    assert bool(search_db.flag("vid", "frame_scan_indexed")) is False
