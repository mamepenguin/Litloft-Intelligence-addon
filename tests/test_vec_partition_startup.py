"""SPEC-ADDON-022 steps 5a / 5b, I9 and I10: what the lifespan does with the
partition migration and the NULL-partition repair."""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import pytest

from tests.vec_partition_db import (  # noqa: F401
    random_vector,
    search_index,
)


async def _until(predicate, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return predicate()


@pytest.fixture()
def lifespan(search_index, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "init_litloft_db", lambda: None)
    return main


@pytest.mark.asyncio
async def test_spec_addon_022_the_event_loop_keeps_running_while_the_search_db_starts(
    lifespan, monkeypatch,
):
    """A start that takes a while (a long migration) runs off the event loop, so the
    loop keeps running callbacks through it, as the LoopWatchdog requires."""
    real_init = lifespan.init_search_db
    window: dict[str, float] = {}
    loop_thread = threading.get_ident()
    ran_on: list[int] = []

    def slow_init():
        ran_on.append(threading.get_ident())
        window["start"] = time.monotonic()
        time.sleep(1.5)
        window["end"] = time.monotonic()
        real_init()

    monkeypatch.setattr(lifespan, "init_search_db", slow_init)
    ticks: list[float] = []

    async def ticker():
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.05)

    task = asyncio.create_task(ticker())
    try:
        async with lifespan.lifespan(lifespan.app):
            pass
    finally:
        task.cancel()

    assert ran_on and ran_on[0] != loop_thread
    inside = [t for t in ticks if window["start"] <= t <= window["end"]]
    assert len(inside) >= 10, f"{len(inside)} loop callbacks ran during a 1.5 s start"


@pytest.mark.asyncio
async def test_spec_addon_022_a_start_repairs_rows_stored_without_a_partition_value(
    lifespan, search_index,
):
    dim = search_index.dim("vec_text")
    search_index.add_file("vid1")
    vec = random_vector(dim, 7)
    search_index.add_embedding("t0", "vid1", "whisper", "vec_text")
    search_index.put_vector("vec_text", "t0", vec)

    async with lifespan.lifespan(lifespan.app):
        assert await _until(
            lambda: search_index.keyed("vec_text").get("t0", (None,))[0] == "whisper"
        )

    assert search_index.keyed("vec_text") == {"t0": ("whisper", vec.tobytes())}


class _RepairBroke(Exception):
    pass


@pytest.mark.asyncio
async def test_spec_addon_022_a_repair_that_raises_is_logged_and_startup_continues(
    lifespan, monkeypatch, caplog,
):
    import app.indexer as indexer

    def broken():
        raise _RepairBroke("disk I/O error")

    monkeypatch.setattr(indexer, "repair_unkeyed_vec_rows", broken)
    monkeypatch.setattr(lifespan, "repair_unkeyed_vec_rows", broken, raising=False)
    caplog.set_level(logging.ERROR)

    def logged() -> bool:
        return any(
            r.levelno == logging.ERROR and r.exc_info and r.exc_info[0] is _RepairBroke
            for r in caplog.records
        )

    entered = False
    async with lifespan.lifespan(lifespan.app):
        entered = True
        assert await _until(logged)

    assert entered
