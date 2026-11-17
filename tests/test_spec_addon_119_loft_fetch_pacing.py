"""SPEC-ADDON-119 items 1.1-1.2, §2, §3, I1-I3: the ``loft_media`` worker holds
the backlog for the pause and the backoff, never holds a new fetch, and puts an
uncounted failure back behind the backlog.

The clock moves forward a little more or less than each limit rather than
exactly onto it: the result time is taken from a real clock the test cannot
stop, so only the two sides of a limit are observable.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.loft_media_world import LoftWorld
from tests.ocr_fixtures import pillow_real  # noqa: F401 - the usefixtures marker resolves it
from tests.test_spec_addon_117_loft_media_start import _start
from tests.test_spec_addon_119_loft_fetch_counting import (  # noqa: F401 - fixtures
    Fetches,
    _line,
    fresh_worker_state,
    gradient,
)
from tests.vec_partition_db import search_index  # noqa: F401 - fixture
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    clock,
    needs_ffmpeg,
    youtube,
)

pytestmark = [needs_ffmpeg, pytest.mark.usefixtures("pillow_real")]

PROMPT = 8.0
BACKLOG = -10


@pytest.fixture()
def world(search_index, youtube, monkeypatch, tmp_path, make_settings):  # noqa: F811
    return LoftWorld(search_index, youtube, monkeypatch, tmp_path, make_settings)


@pytest.fixture()
def fetches(world, gradient, caplog):  # noqa: F811
    caplog.set_level("INFO")
    return Fetches(world, gradient, caplog)


class Worker:
    """The ``loft_media`` worker loop of one manager, observed through the fake yt-dlp."""

    def __init__(self, fetches: Fetches, manager=None):
        from app.indexer import IndexManager

        self.fetches = fetches
        self.manager = manager or IndexManager()
        self.task: asyncio.Task | None = None

    async def enqueue(self, file_id: str, priority: int) -> None:
        from app.indexer import IndexTask, TaskType

        await self.manager._enqueue(
            IndexTask(file_id=file_id, task_type=TaskType("loft_media"), priority=priority),
        )

    def start(self) -> None:
        from app.indexer import IndexManager

        self.task = asyncio.create_task(IndexManager._loft_media_worker(self.manager))

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    def looked_up(self) -> list[str]:
        by_video = {url.rsplit("=", 1)[1]: f for f, url in self.fetches.urls.items()}
        seen = []
        for entry in self.fetches.world.youtube.lookups():
            argv = " ".join(entry["argv"])
            seen.extend(f for video, f in by_video.items() if video in argv)
        return seen

    def _idle(self) -> bool:
        return not self.manager.get_queue_breakdown()["loft_media"]["processing"]

    async def settled(self, lookups: int, seconds: float = 30.0) -> None:
        """Wait until ``lookups`` lookups have happened and the fetch has returned."""
        limit = time.perf_counter() + seconds
        while time.perf_counter() < limit:
            if len(self.looked_up()) >= lookups and self._idle():
                await asyncio.sleep(0.1)
                if self._idle():
                    return
            await asyncio.sleep(0.05)
        raise AssertionError(
            f"expected {lookups} lookups within {seconds}s, saw {self.looked_up()}"
        )

    async def stays(self, lookups: int, seconds: float = 3.0) -> None:
        """Assert no further lookup starts for ``seconds`` of real time."""
        limit = time.perf_counter() + seconds
        while time.perf_counter() < limit:
            assert len(self.looked_up()) == lookups, (
                f"a fetch started while it should be held: {self.looked_up()}"
            )
            await asyncio.sleep(0.05)


@pytest.fixture()
async def worker(fetches):
    w = Worker(fetches)
    yield w
    await w.stop()


async def _fail_three_times(fetches) -> None:
    for _ in range(3):
        await asyncio.to_thread(fetches.run, "lookup")
    assert fetches.backoffs() == [_line(30, 3)]


# ---------------------------------------------------------------------------
# I1 - the pause interval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("priority", [0, 100])
async def test_spec_addon_119_i1_a_backlog_fetch_waits_30_s_after_a_result_and_a_new_one_does_not(
    fetches, worker, clock, priority,
):
    fetches.prepare("no-rendition")
    a, b, c = fetches.loft(), fetches.loft(), fetches.loft()
    await worker.enqueue(a, BACKLOG)
    worker.start()
    await worker.settled(1)

    await worker.enqueue(b, BACKLOG)
    await worker.stays(1)
    clock.advance(20)
    await worker.stays(1)
    await worker.enqueue(c, priority)
    await worker.settled(2, PROMPT)
    assert worker.looked_up() == [a, c]

    clock.advance(15)
    await worker.stays(2)
    clock.advance(16)
    await worker.settled(3)
    assert worker.looked_up() == [a, c, b]


async def test_spec_addon_119_i1_a_fetch_with_no_youtube_result_starts_no_pause(
    fetches, worker, clock,
):
    fetches.prepare("no-rendition")
    a, b, c = fetches.loft(), fetches.loft(), fetches.loft()
    await worker.enqueue(a, BACKLOG)
    worker.start()
    await worker.settled(1)

    fetches.prepare("disk-guard")
    await worker.enqueue(b, BACKLOG)
    await worker.stays(1)
    clock.advance(31)
    await worker.settled(2)

    await worker.enqueue(c, BACKLOG)
    await worker.settled(3, PROMPT)
    assert worker.looked_up() == [a, b, c]


# ---------------------------------------------------------------------------
# I1, I3 - the backoff holds the backlog only
# ---------------------------------------------------------------------------


async def test_spec_addon_119_i3_the_backlog_is_held_for_the_whole_30_minute_backoff(
    fetches, worker, clock,
):
    await _fail_three_times(fetches)
    b = fetches.loft()
    fetches.prepare("no-rendition")
    await worker.enqueue(b, BACKLOG)
    worker.start()

    await worker.stays(3)
    clock.advance(29 * 60)
    await worker.stays(3)
    clock.advance(2 * 60)
    await worker.settled(4)
    assert worker.looked_up()[-1] == b


@pytest.mark.parametrize("priority", [0, 100])
async def test_spec_addon_119_i1_a_new_fetch_runs_during_a_backoff_and_its_success_ends_it(
    fetches, worker, clock, priority,
):
    await _fail_three_times(fetches)
    b, c = fetches.loft(), fetches.loft()
    fetches.prepare("lookup")
    await worker.enqueue(b, BACKLOG)
    worker.start()
    await worker.stays(3)
    clock.advance(60)
    await worker.stays(3)

    fetches.prepare("success")
    await worker.enqueue(c, priority)
    await worker.settled(4, 60.0)
    assert worker.looked_up()[-1] == c
    assert fetches.world.flags(c)[0] is True

    clock.advance(31)
    await worker.settled(5)
    assert worker.looked_up()[-1] == b


@pytest.mark.parametrize("priority", [0, 100])
async def test_spec_addon_119_section_3_a_new_fetch_failing_in_a_backoff_restarts_it_doubled(
    fetches, worker, clock, priority,
):
    await _fail_three_times(fetches)
    clock.advance(20 * 60)
    b, c = fetches.loft(), fetches.loft()
    await worker.enqueue(c, priority)
    worker.start()
    await worker.settled(4, PROMPT)
    assert fetches.backoffs() == [_line(30, 3), _line(60, 4)]

    await worker.enqueue(b, BACKLOG)
    clock.advance(59 * 60)
    await worker.stays(4)
    clock.advance(2 * 60)
    await worker.settled(5)


# ---------------------------------------------------------------------------
# Step 2 / I2 - an uncounted failure goes back behind the backlog
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("priority", [BACKLOG, 100])
async def test_spec_addon_119_i2_an_uncounted_failure_is_queued_again_at_minus_10_behind_the_backlog(
    fetches, worker, clock, priority,
):
    fetches.prepare("lookup")
    a, b = fetches.loft(), fetches.loft()
    if priority == BACKLOG:
        await worker.enqueue(a, BACKLOG)
        await worker.enqueue(b, BACKLOG)
    else:
        await worker.enqueue(b, BACKLOG)
        await worker.enqueue(a, priority)
    worker.start()

    await worker.settled(1)
    assert worker.looked_up() == [a]
    clock.advance(31)
    await worker.settled(2)
    assert worker.looked_up() == [a, b]
    clock.advance(31)
    await worker.settled(3)

    assert worker.looked_up() == [a, b, a]
    assert fetches.world.failures(a) == 0
    assert fetches.world.failures(b) == 0
    assert fetches.world.flags(a)[0] is False


# ---------------------------------------------------------------------------
# Restart (§2): no backoff and no pause carried over
# ---------------------------------------------------------------------------


async def test_spec_addon_119_i3_a_restart_holds_nothing_from_before_it(fetches, worker):
    from app.indexer import IndexManager

    fetches.prepare("no-rendition")
    a, b = fetches.loft(), fetches.loft()
    await worker.enqueue(a, BACKLOG)
    worker.start()
    await worker.settled(1)
    await _fail_three_times(fetches)
    fetches.prepare("no-rendition")
    await worker.enqueue(b, BACKLOG)
    await worker.stays(4)
    await worker.stop()

    restarted = IndexManager()
    stop = await _start(restarted)
    after = Worker(fetches, restarted)
    try:
        assert restarted.get_queue_breakdown()["loft_media"]["waiting"] == 0
        await after.enqueue(b, BACKLOG)
        after.start()
        await after.settled(5, PROMPT)
    finally:
        await after.stop()
        stop.set()
        await restarted.stop()

    assert after.looked_up()[-1] == b
