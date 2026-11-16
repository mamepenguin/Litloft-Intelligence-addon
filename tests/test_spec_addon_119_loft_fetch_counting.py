"""SPEC-ADDON-119 items 1.2-1.4, §2, §3, I2-I4: which failed YouTube fetches are
counted, and when the backoff starts and how long it is.
"""

from __future__ import annotations

import asyncio
import shutil

import pytest

from tests.loft_media_world import GIB, FreeSpace, LoftWorld, fetch, index_off
from tests.ocr_fixtures import pillow_real  # noqa: F401 - the usefixtures marker resolves it
from tests.test_spec_addon_111_frame_scan_task import _encode
from tests.test_spec_addon_117_loft_media_start import _start
from tests.vec_partition_db import search_index  # noqa: F401 - fixture
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    clock,
    describe,
    everything_logged,
    needs_ffmpeg,
    youtube,
)

pytestmark = [needs_ffmpeg, pytest.mark.usefixtures("pillow_real")]

VIDEO_SECONDS = 40.0
REFUSAL = "ERROR: [youtube] Sign in to confirm you're not a bot"
BACKING_OFF = "loft_media backing off"


@pytest.fixture(scope="module")
def gradient(tmp_path_factory):
    root = tmp_path_factory.mktemp("loft-pacing")
    return _encode(root / "switches.mp4")


@pytest.fixture(autouse=True)
def fresh_worker_state():
    from app.workers import loft_media

    loft_media.reset_worker_state()
    yield
    loft_media.reset_worker_state()


@pytest.fixture()
def world(search_index, youtube, monkeypatch, tmp_path, make_settings):  # noqa: F811
    return LoftWorld(search_index, youtube, monkeypatch, tmp_path, make_settings)


class Fetches:
    """Runs one fetch of a fresh ``.loft`` whose YouTube answer is chosen by kind."""

    def __init__(self, world, gradient, caplog):
        self.world = world
        self.gradient = gradient
        self.caplog = caplog
        self.free = FreeSpace(100 * GIB)
        world.monkeypatch.setattr(shutil, "disk_usage", self.free)
        self.made = 0
        self.urls: dict[str, str] = {}

    def loft(self, file_id: str | None = None, *, failures: int = 0) -> str:
        self.made += 1
        file_id = file_id or f"ytloft{self.made:06d}"
        video_id = f"v{self.made:010d}"
        url = f"https://www.youtube.com/watch?v={video_id}"
        self.urls[file_id] = url
        self.world.add_loft(file_id, {"provider": "youtube", "url": url})
        if failures:
            self.world.set(file_id, loft_fetch_failures=failures)
        return file_id

    def _reset(self) -> None:
        self.world.configure()
        self.world.monkeypatch.setattr(shutil, "disk_usage", self.free)
        self.free.free = 100 * GIB
        self.world.youtube.set_download(hold=0)

    def prepare(self, kind: str) -> None:
        """Make every fetch from now on get this YouTube answer."""
        self._reset()
        youtube = self.world.youtube
        if kind == "success":
            self.world.serve(self.gradient, duration=VIDEO_SECONDS, name="ok")
        elif kind == "no-rendition":
            youtube.set_ytdlp(describe=describe([], duration=VIDEO_SECONDS))
        elif kind == "lookup":
            youtube.set_ytdlp(fail=REFUSAL)
        elif kind == "download-403":
            self.world.serve(self.gradient, duration=VIDEO_SECONDS, name="refused")
            youtube.override("/v/refused", status=403)
        elif kind == "download-exit":
            self.world.serve(self.gradient, duration=VIDEO_SECONDS, name="ok")
            youtube.set_download(exit=1)
        elif kind == "download-short":
            self.world.serve(self.gradient, duration=100.0, name="ok")
        elif kind == "disk-guard":
            self.world.serve(self.gradient, duration=VIDEO_SECONDS, name="ok")
            self.free.free = 1 * GIB
        elif kind == "index-off":
            self.world.serve(self.gradient, duration=VIDEO_SECONDS, name="ok")
            self.world.configure(policy=index_off)
            self.world.monkeypatch.setattr(shutil, "disk_usage", self.free)
        else:
            raise AssertionError(kind)

    def run(self, kind: str, file_id: str | None = None, *, failures: int = 0) -> str:
        file_id = file_id or self.loft(failures=failures)
        self.prepare(kind)
        with self.caplog.at_level("INFO"):
            fetch(file_id)
        if kind == "success":
            assert self.world.flags(file_id)[0] is True, "the success fixture did not scan"
        return file_id

    def backoffs(self) -> list[str]:
        return [
            r.getMessage() for r in self.caplog.records if BACKING_OFF in r.getMessage()
        ]


@pytest.fixture()
def fetches(world, gradient, caplog):
    return Fetches(world, gradient, caplog)


def _line(minutes: int, failures: int) -> str:
    return f"loft_media backing off {minutes} min after {failures} failures in a row"


FAILURE_KINDS = ["lookup", "download-403", "download-exit", "download-short"]


# ---------------------------------------------------------------------------
# Step 2 / I2 - only a failure right after a success is counted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("first", ["download-403", "download-exit"])
def test_spec_addon_119_i2_a_failure_with_no_success_before_it_in_the_process_is_not_counted(
    fetches, world, first,
):
    a = fetches.run(first, failures=2)
    b = fetches.run("lookup", failures=2)

    assert world.failures(a) == 2
    assert world.flags(a)[0] is False
    assert world.failures(b) == 2
    assert world.flags(b)[0] is False


@pytest.mark.parametrize("first", FAILURE_KINDS)
@pytest.mark.parametrize("second", ["lookup", "download-403"])
def test_spec_addon_119_i2_after_a_success_only_the_first_failure_of_a_run_is_counted(
    fetches, world, first, second,
):
    fetches.run("success")
    a = fetches.run(first)
    b = fetches.run(second, failures=2)

    assert world.failures(a) == 1
    assert world.failures(b) == 2
    assert world.flags(b) == (False, True, True)


@pytest.mark.parametrize("success", ["success", "no-rendition"])
def test_spec_addon_119_i2_a_success_between_failures_makes_the_next_failure_counted_again(
    fetches, world, success,
):
    x = fetches.run("lookup")
    fetches.run(success)
    a = fetches.run("download-403")
    b = fetches.run("lookup")

    assert world.failures(x) == 0
    assert world.failures(a) == 1
    assert world.failures(b) == 0


@pytest.mark.parametrize("none", ["disk-guard", "index-off"])
def test_spec_addon_119_i2_a_fetch_with_no_youtube_result_is_skipped_by_the_counting_rule(
    fetches, world, none,
):
    fetches.run("success")
    fetches.run(none)
    a = fetches.run("lookup")
    fetches.run(none)
    b = fetches.run("download-403")

    assert world.failures(a) == 1
    assert world.failures(b) == 0


# ---------------------------------------------------------------------------
# Backoff (I3) and its log line (I4)
# ---------------------------------------------------------------------------


def test_spec_addon_119_i3_the_backoff_starts_at_three_failures_and_doubles_up_to_six_hours(
    fetches,
):
    fetches.run("success")
    for _ in range(8):
        fetches.run("lookup")

    assert fetches.backoffs() == [
        _line(30, 3), _line(60, 4), _line(120, 5), _line(240, 6), _line(360, 7),
        _line(360, 8),
    ]


@pytest.mark.parametrize("kind", FAILURE_KINDS)
def test_spec_addon_119_i3_every_kind_of_failure_result_extends_the_run(fetches, kind):
    fetches.run("lookup")
    fetches.run("lookup")
    assert fetches.backoffs() == []

    fetches.run(kind)

    assert fetches.backoffs() == [_line(30, 3)]


@pytest.mark.parametrize("success", ["success", "no-rendition"])
def test_spec_addon_119_i3_a_success_resets_the_run_and_the_next_length(fetches, success):
    for _ in range(4):
        fetches.run("lookup")
    fetches.run(success)
    fetches.run("lookup")
    fetches.run("lookup")
    assert fetches.backoffs() == [_line(30, 3), _line(60, 4)]

    fetches.run("lookup")

    assert fetches.backoffs() == [_line(30, 3), _line(60, 4), _line(30, 3)]


@pytest.mark.parametrize("none", ["disk-guard", "index-off"])
def test_spec_addon_119_i3_a_fetch_with_no_youtube_result_leaves_the_run_as_it_is(
    fetches, none,
):
    fetches.run("lookup")
    fetches.run("lookup")
    fetches.run(none)
    fetches.run("lookup")

    assert fetches.backoffs() == [_line(30, 3)]


def test_spec_addon_119_section_3_a_backoff_that_elapses_doubles_the_next_one(fetches, clock):
    for _ in range(3):
        fetches.run("lookup")
    clock.advance(31 * 60)

    fetches.run("lookup")

    assert fetches.backoffs() == [_line(30, 3), _line(60, 4)]


def test_spec_addon_119_i4_the_backoff_line_names_no_file_url_or_video(fetches, world):
    ids = [fetches.run("lookup") for _ in range(3)]

    lines = [
        f"{r.name} {r.getMessage()} {r.args!r}"
        for r in fetches.caplog.records if BACKING_OFF in r.getMessage()
    ]
    assert len(lines) == 1
    for file_id in ids:
        url = fetches.urls[file_id]
        for secret in (file_id, url, url.rsplit("=", 1)[1], "dQw4w9WgXcQ"):
            assert secret not in lines[0]


# ---------------------------------------------------------------------------
# Restart (§2): the run, the backoff and the success before it start fresh
# ---------------------------------------------------------------------------


def _restart(world) -> None:
    from app.indexer import IndexManager

    async def run():
        manager = IndexManager()
        stop = await _start(manager)
        stop.set()
        await manager.stop()

    asyncio.run(run())


def test_spec_addon_119_i3_a_restart_starts_with_no_run_and_the_first_length(fetches, world):
    for _ in range(5):
        fetches.run("lookup")
    before = [_line(30, 3), _line(60, 4), _line(120, 5)]
    assert fetches.backoffs() == before

    _restart(world)
    fetches.run("lookup")
    fetches.run("lookup")
    assert fetches.backoffs() == before
    fetches.run("lookup")

    assert fetches.backoffs() == [*before, _line(30, 3)]


def test_spec_addon_119_i2_a_restart_forgets_the_success_before_it(fetches, world):
    fetches.run("success")

    _restart(world)
    a = fetches.run("download-403", failures=2)

    assert world.failures(a) == 2
    assert world.flags(a)[0] is False
