"""SPEC-ADDON-122: the frame OCR worker's re-read bookkeeping.

A re-read is decided by a lookup that can fail on its own, and while it runs
its stored row is what the `.loft` gate must not read as native frame work.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from tests.loft_media_world import LoftWorld
from tests.ocr_fixtures import pillow_real  # noqa: F401 - the usefixtures marker resolves it
from tests.test_spec_addon_119_loft_fetch_counting import (  # noqa: F401 - fixtures
    Fetches,
    fresh_worker_state,
    gradient,
)
from tests.test_spec_addon_119_loft_fetch_pacing import BACKLOG, Worker
from tests.test_spec_addon_122_loft_fetch_gate import (  # noqa: F401 - fixtures
    _on_index,
    _queue_native_frame_task,
    fetches,
    world,
)
from tests.vec_partition_db import search_index  # noqa: F401 - fixture
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    clock,
    needs_ffmpeg,
    youtube,
)

pytestmark = [needs_ffmpeg, pytest.mark.usefixtures("pillow_real")]


@pytest.fixture()
async def worker(fetches):  # noqa: F811
    w = Worker(fetches)
    yield w
    await w.stop()


async def test_spec_addon_122_i11_a_failed_reread_lookup_still_runs_the_read(
    monkeypatch, caplog,
):
    from app.indexer import IndexManager, IndexTask, TaskType

    manager = IndexManager()
    calls: list[str] = []

    def fail_lookup(_file_id: str) -> bool:
        raise RuntimeError("search.db is busy")

    def read(file_id: str, is_paused=None) -> bool:
        calls.append(file_id)
        return True

    monkeypatch.setattr("app.indexer._is_outdated_frame_read", fail_lookup)
    monkeypatch.setattr("app.indexer.index_frame_ocr", read)
    caplog.set_level("WARNING")
    await manager._enqueue(
        IndexTask(file_id="vid1", task_type=TaskType.FRAME_OCR, priority=0),
    )

    loop = asyncio.create_task(IndexManager._frame_ocr_worker(manager))
    try:
        limit = time.perf_counter() + 10
        while not calls and time.perf_counter() < limit:
            await asyncio.sleep(0.05)
    finally:
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)

    assert calls == ["vid1"]
    assert "search.db is busy" in caplog.text


async def test_spec_addon_122_i11_a_running_reread_past_its_store_does_not_hold_a_backlog_fetch(
    fetches, worker, gradient,  # noqa: F811
):
    _on_index(fetches)
    world = fetches.world
    world.add_native(
        "old", gradient, frame_scan_indexed=True,
        frame_ocr_read=True, frame_ocr_indexed=True,
    )
    await _queue_native_frame_task(worker, "old")

    entered = threading.Event()
    release = threading.Event()

    def store_then_wait(file_id: str, is_paused=None) -> bool:
        # The store step of a re-read: the row is no longer outdated while the
        # read is still the work this task is doing.
        world.set(
            file_id, frame_ocr_read=1, frame_ocr_indexed=0, frame_ocr_rule=2,
        )
        entered.set()
        release.wait(timeout=30)
        return True

    world.monkeypatch.setattr("app.indexer.index_frame_ocr", store_then_wait)
    loop = asyncio.create_task(
        type(worker.manager)._frame_ocr_worker(worker.manager)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        assert await worker.manager._native_frame_busy() is False

        b = fetches.loft()
        await worker.enqueue(b, BACKLOG)
        worker.start()
        await worker.settled(1)
        assert worker.looked_up() == [b]
    finally:
        release.set()
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)
