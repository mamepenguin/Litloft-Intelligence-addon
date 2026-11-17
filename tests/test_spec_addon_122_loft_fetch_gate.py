"""SPEC-ADDON-122 item 1 step 10, I11 (revising SPEC-ADDON-117 I3): a queued
re-read of an outdated native read does not hold a backlog ``.loft`` fetch, and
a native first read still does.
"""

from __future__ import annotations

import shutil

import pytest

from tests.loft_media_world import LoftWorld
from tests.ocr_fixtures import pillow_real  # noqa: F401 - the usefixtures marker resolves it
from tests.test_spec_addon_119_loft_fetch_counting import (  # noqa: F401 - fixtures
    Fetches,
    fresh_worker_state,
    gradient,
)
from tests.test_spec_addon_119_loft_fetch_pacing import BACKLOG, Worker
from tests.vec_partition_db import search_index  # noqa: F401 - fixture
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    clock,
    needs_ffmpeg,
    youtube,
)

pytestmark = [needs_ffmpeg, pytest.mark.usefixtures("pillow_real")]


@pytest.fixture()
def world(search_index, youtube, monkeypatch, tmp_path, make_settings):  # noqa: F811
    return LoftWorld(search_index, youtube, monkeypatch, tmp_path, make_settings)


@pytest.fixture()
def fetches(world, gradient, caplog):  # noqa: F811
    caplog.set_level("INFO")
    return Fetches(world, gradient, caplog)


@pytest.fixture()
async def worker(fetches):
    w = Worker(fetches)
    yield w
    await w.stop()


def _on_index(fetches) -> None:
    fetches.prepare("disk-guard")
    fetches.world.configure(frame_ocr="on_index")
    fetches.world.monkeypatch.setattr(shutil, "disk_usage", fetches.free)


async def _queue_native_frame_task(worker, file_id: str) -> None:
    from app.indexer import IndexTask, TaskType

    await worker.manager._enqueue(
        IndexTask(file_id=file_id, task_type=TaskType.FRAME_OCR, priority=BACKLOG),
    )


async def test_spec_addon_122_i11_a_queued_reread_does_not_hold_a_backlog_fetch_and_a_first_read_does(
    fetches, worker, gradient,  # noqa: F811
):
    _on_index(fetches)
    world = fetches.world
    world.add_native(
        "old", gradient, frame_scan_indexed=True, frame_ocr_read=True, frame_ocr_indexed=True,
    )
    await _queue_native_frame_task(worker, "old")
    a, b = fetches.loft(), fetches.loft()

    await worker.enqueue(a, BACKLOG)
    worker.start()
    await worker.settled(1)
    assert worker.looked_up() == [a]

    world.add_native(
        "fresh", gradient, frame_scan_indexed=True, frame_ocr_read=False, frame_ocr_indexed=False,
    )
    await _queue_native_frame_task(worker, "fresh")
    await worker.enqueue(b, BACKLOG)
    await worker.stays(1)
