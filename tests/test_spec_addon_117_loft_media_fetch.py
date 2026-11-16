"""SPEC-ADDON-117 items 1.3-1.4, §2, I1-I7, I11: one fetch of a YouTube ``.loft``.

The rendition is the gradient video of the SPEC-ADDON-111 scan tests, so the
change seconds a native scan finds in it are known: 0, 10, 20 and 30.
"""

from __future__ import annotations

import asyncio
import shutil
import threading
import time

import pytest

from app.frame_cache import frame_cache_path
from tests.loft_media_world import (
    FLAGS,
    GIB,
    FreeSpace,
    LoftWorld,
    disk_usage_with,
    fetch,
    index_off,
)
from tests.ocr_fixtures import pillow_real  # noqa: F401 - the usefixtures marker resolves it
from tests.test_spec_addon_111_frame_scan_task import CHANGE_SECONDS, _encode
from tests.vec_partition_db import search_index  # noqa: F401 - fixture
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    HEADER_TOKEN,
    SIGNED_TOKEN,
    YOUTUBE_URL,
    clock,
    describe,
    everything_logged,
    log_everything,
    needs_ffmpeg,
    webp_size,
    youtube,
)

pytestmark = [needs_ffmpeg, pytest.mark.usefixtures("pillow_real")]

VIDEO_SECONDS = 40.0
COOKIE_OPTIONS = (
    "--cookies", "--cookies-from-browser", "--username", "--password",
    "--netrc", "--video-password", "--ap-username", "--add-header",
)


@pytest.fixture(scope="module")
def gradient(tmp_path_factory):
    root = tmp_path_factory.mktemp("loft-media")
    return _encode(root / "switches.mp4")


@pytest.fixture()
def world(search_index, youtube, monkeypatch, tmp_path, make_settings):  # noqa: F811
    return LoftWorld(search_index, youtube, monkeypatch, tmp_path, make_settings)


def _succeed_once(world, gradient, file_id: str = "ytwarm000001") -> None:
    """A fetch whose lookup succeeds, so a later lookup failure is a counted one."""
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.add_loft(file_id)
    fetch(file_id)
    assert world.flags(file_id)[0] is True


def _wait_until(condition, seconds: float = 10.0) -> None:
    limit = time.monotonic() + seconds
    while not condition() and time.monotonic() < limit:
        time.sleep(0.02)


def _after_download_starts(youtube, action) -> threading.Thread:
    """Run ``action`` once the download has written its first bytes, so a stop rule, not the guard, decides."""
    def run():
        _wait_until(lambda: youtube.partials())
        action()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _advance_while_growing(world, clock, seconds: float) -> threading.Thread:
    """Move the clock in steps shorter than the stall limit, each after the file grew.

    One jump past 120 s would read as a stall whatever the stream does.
    """
    def size() -> int:
        return sum(p.stat().st_size for p in world.fetch_dir.glob("*") if p.is_file())

    def run():
        _wait_until(lambda: world.youtube.partials())
        moved = 0.0
        while moved < seconds:
            step = min(100.0, seconds - moved)
            before = size()
            _wait_until(lambda: size() > before, 2.0)
            clock.advance(step)
            moved += step

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Normal flow: the scan of the download
# ---------------------------------------------------------------------------


def test_spec_addon_117_a_youtube_loft_is_scanned_from_its_download_as_a_native_video_is(
    world, gradient,
):
    """I4: the same change frames, scan row and flag as a native scan of that video."""
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000001")

    fetch("ytloft000001")

    assert world.source("ytloft000001") == "youtube"
    assert world.change_seconds("ytloft000001") == [float(s) for s in CHANGE_SECONDS]
    assert world.scan_rows("ytloft000001") == 1
    assert world.flags("ytloft000001")[0] is True
    assert world.failures("ytloft000001") == 0
    assert world.fetch_dir_files() == []


def test_spec_addon_117_i11_the_scan_of_a_loft_keeps_a_320px_picture_of_each_change_frame(
    world, gradient, tmp_path,
):
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000001")
    kept = frame_cache_path("ytloft000001", 10.0)
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_bytes(b"KEPT")

    fetch("ytloft000001")

    assert kept.read_bytes() == b"KEPT"
    for second in (0, 20, 30):
        picture = frame_cache_path("ytloft000001", float(second))
        assert picture.exists(), f"no small picture at {second}s"
        assert webp_size(picture.read_bytes(), tmp_path)[0] == 320


def test_spec_addon_117_i11_a_native_scan_keeps_no_small_pictures(world, gradient):
    """Item 7 / I11: only the scan of a ``.loft`` writes them."""
    from app.workers import frame_scan

    world.serve(gradient, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000001")
    world.add_native("native000001", gradient)

    fetch("ytloft000001")
    frame_scan.index_frame_scan("native000001")

    assert frame_cache_path("ytloft000001", 10.0).exists()
    assert world.change_seconds("native000001") == [float(s) for s in CHANGE_SECONDS]
    assert not any(
        frame_cache_path("native000001", float(s)).exists() for s in CHANGE_SECONDS
    )


def test_spec_addon_117_i4_the_ocr_read_of_the_download_writes_what_a_native_read_writes(
    world, gradient,
):
    from app.workers import frame_ocr

    world.configure(frame_ocr="on_index")
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000001", frame_scan_indexed=True,
                   frame_ocr_read=False, frame_ocr_indexed=False)
    world.add_native("native000001", gradient, frame_scan_indexed=True,
                     frame_ocr_read=False, frame_ocr_indexed=False)

    fetch("ytloft000001")
    assert frame_ocr.index_frame_ocr("native000001") is True

    native = world.frame_texts("native000001")
    assert native, "the native read found no text; the fixture does not exercise OCR"
    assert world.frame_texts("ytloft000001") == native
    assert world.flags("ytloft000001") == (True, True, True)
    assert world.fetch_dir_files() == []


def test_spec_addon_117_one_fetch_runs_the_scan_and_then_the_ocr_read(world, gradient):
    world.configure(frame_ocr="on_index")
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000001", frame_ocr_read=False, frame_ocr_indexed=False)

    fetch("ytloft000001")

    assert world.flags("ytloft000001") == (True, True, True)
    assert world.change_seconds("ytloft000001") == [float(s) for s in CHANGE_SECONDS]
    assert world.frame_texts("ytloft000001") != []
    assert world.lookups() == 1
    assert len({r["path"] for r in world.downloads()}) == 1


# ---------------------------------------------------------------------------
# Item 1.3 step 1 - reading the .loft (I1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"provider": "vimeo", "url": "https://vimeo.com/1"}, id="another-provider"),
        pytest.param("this is not json", id="not-json"),
        pytest.param({"provider": "youtube"}, id="no-url"),
        pytest.param({"url": YOUTUBE_URL}, id="no-provider"),
    ],
)
def test_spec_addon_117_i1_a_loft_with_no_youtube_reference_becomes_other_with_flags_closed(
    world, gradient, body,
):
    world.configure(frame_ocr="on_index")
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.add_loft("other0000001", body, frame_ocr_read=False, frame_ocr_indexed=False)

    fetch("other0000001")

    assert world.source("other0000001") == "other"
    assert world.flags("other0000001") == (True, True, True)
    assert world.lookups() == 0
    assert world.downloads() == []


def test_spec_addon_117_i1_a_loft_that_cannot_be_read_stays_unclassified_and_uncounted(
    world, gradient,
):
    world.serve(gradient, duration=VIDEO_SECONDS)
    path = world.add_loft("gone00000001")
    path.unlink()
    world.add_loft("ytloft000001")

    fetch("gone00000001")
    fetch("ytloft000001")

    assert world.source("gone00000001") is None
    assert world.flags("gone00000001")[0] is False
    assert world.failures("gone00000001") == 0
    assert world.source("ytloft000001") == "youtube"
    assert world.lookups() == 1


# ---------------------------------------------------------------------------
# Item 1.3 step 2 - ends before any network call (I3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode, flags",
    [
        pytest.param("manual", dict(frame_scan_indexed=True, frame_ocr_read=False,
                                    frame_ocr_indexed=False), id="manual-not-requested"),
        pytest.param("false", dict(frame_scan_indexed=True, frame_ocr_read=False,
                                   frame_ocr_indexed=False), id="ocr-off"),
        pytest.param("on_index", dict(frame_scan_indexed=True, frame_ocr_read=True,
                                      frame_ocr_indexed=False), id="read-stored"),
        pytest.param("on_index", dict(frame_scan_indexed=True, frame_ocr_read=True,
                                      frame_ocr_indexed=True), id="all-done"),
    ],
)
def test_spec_addon_117_i3_a_fetch_with_no_stage_wanted_classifies_and_downloads_nothing(
    world, gradient, mode, flags,
):
    world.configure(frame_ocr=mode)
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000001", **flags)

    fetch("ytloft000001")

    assert world.source("ytloft000001") == "youtube"
    assert world.lookups() == 0
    assert world.downloads() == []
    assert world.flags("ytloft000001") == tuple(flags[f] for f in FLAGS)
    assert world.failures("ytloft000001") == 0


def test_spec_addon_117_i3_the_index_feature_off_ends_before_the_lookup(world, gradient):
    world.configure(policy=index_off)
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000001")

    fetch("ytloft000001")

    assert world.source("ytloft000001") == "youtube"
    assert world.lookups() == 0
    assert world.flags("ytloft000001")[0] is False
    assert world.failures("ytloft000001") == 0


def test_spec_addon_117_i3_a_clip_model_that_cannot_load_ends_before_the_lookup(
    world, gradient,
):
    from tests.search_endpoint_db import patch_everywhere

    def _cannot_load(*_a, **_k):
        raise RuntimeError("the CLIP weights are not on disk")

    world.serve(gradient, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000001")
    patch_everywhere(world.monkeypatch, "load_image_model", _cannot_load)

    fetch("ytloft000001")

    assert world.source("ytloft000001") == "youtube"
    assert world.lookups() == 0
    assert world.flags("ytloft000001")[0] is False
    assert world.failures("ytloft000001") == 0


def test_spec_addon_117_i3_an_ocr_engine_that_cannot_be_created_ends_before_the_lookup(
    world, gradient,
):
    from tests.ocr_fixtures import EngineFactory
    from tests.search_endpoint_db import patch_everywhere

    def _no_engine(*_a, **_k):
        raise RuntimeError("rapidocr is not installed")

    world.configure(frame_ocr="on_index")
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000001", frame_scan_indexed=True,
                   frame_ocr_read=False, frame_ocr_indexed=False)
    patch_everywhere(world.monkeypatch, "_get_engine", _no_engine)
    patch_everywhere(world.monkeypatch, "_create_engine",
                     EngineFactory(error=RuntimeError("rapidocr is not installed")))

    fetch("ytloft000001")

    assert world.source("ytloft000001") == "youtube"
    assert world.lookups() == 0
    assert world.flags("ytloft000001") == (True, False, False)
    assert world.failures("ytloft000001") == 0


# ---------------------------------------------------------------------------
# Item 1.3 step 3 - the duration (I5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stored, looked_up, expected",
    [
        pytest.param(None, VIDEO_SECONDS, VIDEO_SECONDS, id="null-is-filled"),
        pytest.param(12.5, VIDEO_SECONDS, 12.5, id="set-is-kept"),
        pytest.param(None, None, None, id="no-lookup-duration"),
        pytest.param(None, 0.0, None, id="zero-lookup-duration"),
    ],
)
def test_spec_addon_117_i5_the_duration_is_written_only_from_null_with_a_positive_lookup(
    world, gradient, stored, looked_up, expected,
):
    world.serve(gradient, duration=looked_up)
    world.add_loft("ytloft000001", duration=stored)

    fetch("ytloft000001")

    assert world.lookups() == 1
    assert world.value("ytloft000001", "duration") == expected


# ---------------------------------------------------------------------------
# Counted and uncounted ends (I6), the cap (item 4)
# ---------------------------------------------------------------------------


def test_spec_addon_117_i6_no_rendition_is_a_success_result_and_is_logged(
    world, caplog,
):
    """SPEC-ADDON-119 Terms: no rendition is a success, so it is not counted."""
    world.youtube.set_ytdlp(describe=describe([], duration=VIDEO_SECONDS))
    world.add_loft("ytloft000001")

    with caplog.at_level("INFO"):
        fetch("ytloft000001")

    assert world.failures("ytloft000001") == 0
    assert world.flags("ytloft000001")[0] is False
    assert world.downloads() == []
    assert "loft_media no rendition (file ytloft000001)" in everything_logged(caplog)


def test_spec_addon_117_i6_a_lookup_failure_after_a_successful_lookup_is_counted(
    world, gradient, caplog,
):
    _succeed_once(world, gradient)
    world.youtube.set_ytdlp(fail="ERROR: Private video")
    world.add_loft("ytloft000002")

    with caplog.at_level("INFO"):
        fetch("ytloft000002")

    assert world.failures("ytloft000002") == 1
    assert world.flags("ytloft000002")[0] is False
    assert "loft_media lookup failed (file ytloft000002)" in everything_logged(caplog)


def test_spec_addon_117_i6_a_missing_yt_dlp_is_never_counted(world, gradient):
    _succeed_once(world, gradient)
    world.youtube.remove_ytdlp()
    world.add_loft("ytloft000002")

    fetch("ytloft000002")

    assert world.source("ytloft000002") == "youtube"
    assert world.failures("ytloft000002") == 0
    assert world.flags("ytloft000002")[0] is False


def test_spec_addon_117_i6_a_download_that_fails_is_counted_and_leaves_nothing(
    world, gradient, caplog,
):
    """SPEC-ADDON-119 I2: a failure is counted only after a success."""
    _succeed_once(world, gradient)
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.youtube.override("/v/rendition", status=403)
    world.add_loft("ytloft000001")

    with caplog.at_level("INFO"):
        fetch("ytloft000001")

    assert world.youtube.partials() != []
    assert world.downloads() != []
    assert world.failures("ytloft000001") == 1
    assert world.scan_rows("ytloft000001") == 0
    assert world.flags("ytloft000001")[0] is False
    assert world.fetch_dir_files() == []
    assert "loft_media download failed (file ytloft000001)" in everything_logged(caplog)


def test_spec_addon_117_i6_a_nonzero_exit_after_a_whole_file_is_a_counted_failure(
    world, gradient, caplog,
):
    """yt-dlp can fail after the transfer; the file it wrote must not be scanned.

    SPEC-ADDON-119 I2: the failure is counted because a lookup just succeeded.
    """
    _succeed_once(world, gradient)
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.youtube.set_download(exit=1)
    world.add_loft("ytloft000001")

    with caplog.at_level("INFO"):
        fetch("ytloft000001")

    assert world.youtube.download_notes() != []
    assert world.failures("ytloft000001") == 1
    assert world.flags("ytloft000001")[0] is False
    assert world.scan_rows("ytloft000001") == 0
    assert world.change_seconds("ytloft000001") == []
    assert world.fetch_dir_files() == []
    assert "loft_media download failed (file ytloft000001)" in everything_logged(caplog)


def test_spec_addon_117_i6_a_download_that_writes_an_empty_file_is_counted(world, gradient):
    """With no lookup duration nothing probes the file, so only the size check refuses it.

    SPEC-ADDON-119 I2: the failure follows a success, so it is counted.
    """
    _succeed_once(world, gradient)
    world.serve(gradient, duration=None)
    world.youtube.set_download(empty=True)
    world.add_loft("ytloft000001")

    fetch("ytloft000001")

    assert world.failures("ytloft000001") == 1
    assert world.flags("ytloft000001")[0] is False
    assert world.scan_rows("ytloft000001") == 0
    assert world.fetch_dir_files() == []


@pytest.mark.parametrize(
    "looked_up, counted",
    [
        pytest.param(100.0, True, id="40-of-100-is-short"),
        pytest.param(43.0, True, id="40-of-43-is-below-95-percent"),
        pytest.param(41.0, False, id="40-of-41-is-whole"),
    ],
)
def test_spec_addon_117_i6_a_download_shorter_than_95_percent_of_the_lookup_is_counted(
    world, gradient, looked_up, counted,
):
    """SPEC-ADDON-119 I2: a counted row is counted because a success came first."""
    _succeed_once(world, gradient)
    world.serve(gradient, duration=looked_up)
    world.add_loft("ytloft000001")

    fetch("ytloft000001")

    assert world.downloads() != []
    assert world.failures("ytloft000001") == (1 if counted else 0)
    assert world.flags("ytloft000001")[0] is (not counted)
    assert world.scan_rows("ytloft000001") == (0 if counted else 1)
    assert world.fetch_dir_files() == []


def test_spec_addon_117_a_truncated_download_is_counted(world, gradient):
    """SPEC-ADDON-119 I2: the failure follows a success, so it is counted."""
    _succeed_once(world, gradient)
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.youtube.override("/v/rendition", truncate=0.5)
    world.add_loft("ytloft000001")

    fetch("ytloft000001")

    assert world.failures("ytloft000001") == 1
    assert world.fetch_dir_files() == []


def test_spec_addon_117_item_4_the_third_counted_failure_closes_the_wanted_stages(
    world, gradient,
):
    """SPEC-ADDON-119 I2: each counted failure here directly follows a success."""
    world.configure(frame_ocr="on_index")
    world.add_loft("ytloft000001", frame_ocr_read=False, frame_ocr_indexed=False)
    world.add_loft("ytloft000002", frame_ocr_read=False, frame_ocr_indexed=False)
    world.set("ytloft000001", loft_fetch_failures=2)
    world.set("ytloft000002", loft_fetch_failures=1)

    _succeed_once(world, gradient, "ytwarm000001")
    world.youtube.set_ytdlp(fail="ERROR: Private video")
    fetch("ytloft000001")

    _succeed_once(world, gradient, "ytwarm000002")
    world.youtube.set_ytdlp(fail="ERROR: Private video")
    fetch("ytloft000002")

    assert world.failures("ytloft000001") == 3
    assert world.flags("ytloft000001") == (True, True, True)
    assert world.scan_rows("ytloft000001") == 0
    assert world.frame_texts("ytloft000001") == []
    assert world.failures("ytloft000002") == 2
    assert world.flags("ytloft000002") == (False, False, False)


def test_spec_addon_117_i6_a_fetch_whose_stages_all_set_their_flags_resets_the_count(
    world, gradient,
):
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000001")
    world.set("ytloft000001", loft_fetch_failures=2)

    fetch("ytloft000001")

    assert world.flags("ytloft000001")[0] is True
    assert world.failures("ytloft000001") == 0


# ---------------------------------------------------------------------------
# Item 1.3 step 5 - the disk guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filesize, free, proceeds",
    [
        pytest.param(2 * GIB, 4 * GIB, True, id="known-size-leaves-2-GiB"),
        pytest.param(2 * GIB, 4 * GIB - 1, False, id="known-size-leaves-under-2-GiB"),
        pytest.param(None, 4 * GIB, True, id="unknown-size-with-4-GiB"),
        pytest.param(None, 4 * GIB - 1, False, id="unknown-size-under-4-GiB"),
    ],
)
def test_spec_addon_117_the_disk_guard_is_an_uncounted_end_before_the_download(
    world, gradient, caplog, filesize, free, proceeds,
):
    """The free space is read through ``shutil.disk_usage``."""
    world.monkeypatch.setattr(shutil, "disk_usage", disk_usage_with(free))
    world.serve(gradient, duration=VIDEO_SECONDS, filesize=filesize)
    world.add_loft("ytloft000001")

    with caplog.at_level("INFO"):
        fetch("ytloft000001")

    assert world.lookups() == 1
    assert (world.downloads() != []) is proceeds
    assert world.flags("ytloft000001")[0] is proceeds
    assert world.failures("ytloft000001") == 0
    logged = "loft_media no space (file ytloft000001)" in everything_logged(caplog)
    assert logged is (not proceeds)


# ---------------------------------------------------------------------------
# Item 1.3 step 6 - the download's stop rules (I2)
# ---------------------------------------------------------------------------


def test_spec_addon_117_a_download_with_no_growth_for_120_seconds_is_stopped_and_counted(
    world, gradient, clock,
):
    """The clock jumps 121 s while yt-dlp holds a partial it already wrote.

    SPEC-ADDON-119 I2: the failure follows a success, so it is counted.
    """
    _succeed_once(world, gradient)
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.youtube.set_download(hold=30, stall_bytes=8192)
    world.add_loft("ytloft000001")

    def stall_until_stopped():
        # The fetch may see the first bytes only after the clock moved, which
        # restarts its timer; keep moving it until the stall rule fires.
        while not world.youtube.downloads_stopped():
            clock.advance(121)
            time.sleep(0.2)

    _after_download_starts(world.youtube, stall_until_stopped)
    started = time.perf_counter()
    fetch("ytloft000001")

    assert time.perf_counter() - started < 25
    assert world.youtube.partials() != []
    assert world.failures("ytloft000001") == 1
    assert world.flags("ytloft000001")[0] is False
    assert world.fetch_dir_files() == []
    assert world.youtube.downloads_stopped()


def test_spec_addon_117_i2_a_download_is_stopped_when_free_space_falls_below_1_gib(
    world, gradient,
):
    """The guard checks once before the start; a download that fills the disk stops mid-way.

    SPEC-ADDON-119 Terms: the free-space stop is no YouTube result, so it is not counted.
    """
    free = FreeSpace(50 * GIB)
    world.monkeypatch.setattr(shutil, "disk_usage", free)
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.youtube.set_download(hold=5, stall_bytes=4096)
    world.add_loft("ytloft000001")

    _after_download_starts(world.youtube, lambda: setattr(free, "free", GIB // 2))
    fetch("ytloft000001")

    assert world.youtube.partials() != []
    assert world.failures("ytloft000001") == 0
    assert world.flags("ytloft000001")[0] is False
    assert world.scan_rows("ytloft000001") == 0
    assert world.fetch_dir_files() == []
    assert world.youtube.downloads_stopped()


def test_spec_addon_117_i2_a_download_is_stopped_at_the_time_limit(world, gradient, clock):
    """The stream keeps growing, so only the time limit can stop it.

    SPEC-ADDON-119 I2: the failure follows a success, so it is counted.
    """
    _succeed_once(world, gradient)
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.youtube.set_download(grow=[64, 3.0])
    world.add_loft("ytloft000001")

    _after_download_starts(world.youtube, lambda: clock.advance(601))
    fetch("ytloft000001")

    assert world.youtube.partials() != []
    assert world.failures("ytloft000001") == 1
    assert world.flags("ytloft000001")[0] is False
    assert world.scan_rows("ytloft000001") == 0
    assert world.fetch_dir_files() == []
    assert world.youtube.downloads_stopped()


def test_spec_addon_117_i2_a_download_under_the_600_s_floor_is_not_stopped(world, gradient):
    """3 s of lookup would give a 4.5 s limit without the floor; this download takes 5 s."""
    world.serve(gradient, duration=3.0)
    world.youtube.set_download(hold=5)
    world.add_loft("ytloft000001")

    fetch("ytloft000001")

    assert world.failures("ytloft000001") == 0
    assert world.flags("ytloft000001")[0] is True
    assert world.scan_rows("ytloft000001") == 1
    assert world.fetch_dir_files() == []


def test_spec_addon_117_i2_a_download_of_unknown_duration_is_not_stopped_at_600_s(
    world, gradient, clock,
):
    world.serve(gradient, duration=None)
    world.youtube.set_download(grow=[64, 3.0])
    world.add_loft("ytloft000001")

    _advance_while_growing(world, clock, 700)
    fetch("ytloft000001")

    assert world.failures("ytloft000001") == 0
    assert world.flags("ytloft000001")[0] is True
    assert world.scan_rows("ytloft000001") == 1
    assert world.fetch_dir_files() == []


def test_spec_addon_117_i2_a_download_of_unknown_duration_is_stopped_at_7200_s(
    world, gradient, clock,
):
    """SPEC-ADDON-119 I2: the failure follows a success, so it is counted."""
    _succeed_once(world, gradient)
    world.serve(gradient, duration=None)
    world.youtube.set_download(grow=[64, 3.0])
    world.add_loft("ytloft000001")

    _after_download_starts(world.youtube, lambda: clock.advance(7201))
    fetch("ytloft000001")

    assert world.youtube.partials() != []
    assert world.failures("ytloft000001") == 1
    assert world.flags("ytloft000001")[0] is False
    assert world.fetch_dir_files() == []
    assert world.youtube.downloads_stopped()


# ---------------------------------------------------------------------------
# I2 - what the tools are given, what is left, what is logged
# ---------------------------------------------------------------------------


def test_spec_addon_117_i2_the_download_takes_one_video_only_rendition_from_the_stored_url(
    world, gradient,
):
    """SPEC-ADDON-117 I2 (revised): yt-dlp, not ffmpeg, reads the stored URL."""
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000001")

    fetch("ytloft000001")

    (lookup,) = world.youtube.lookups()
    for option in ("-J", "--no-playlist", "--ignore-config"):
        assert option in lookup["argv"]
    assert not any(a.split("=")[0] in COOKIE_OPTIONS for a in lookup["argv"])

    (download,) = world.youtube.download_commands()
    argv = download["argv"]
    assert argv[argv.index("-f") + 1] == "bv*[height<=720][protocol=https]"
    assert argv[argv.index("--http-chunk-size") + 1] == "10M"
    for option in ("--no-playlist", "--ignore-config"):
        assert option in argv
    assert argv[argv.index("-o") + 1] == str(
        world.fetch_dir / "ytloft000001.part.%(ext)s"
    )
    assert argv[-1] == YOUTUBE_URL
    assert not any(a.split("=")[0] in COOKIE_OPTIONS for a in argv)
    assert not any(
        any("/v/rendition" in a for a in event["argv"])
        for event in world.youtube.ffmpeg_events()
    )
    assert all(
        r["headers"].get("x-fake-auth") == HEADER_TOKEN for r in world.downloads()
    )
    assert world.fetch_dir_files() == []


def test_spec_addon_117_i2_a_leftover_of_the_same_file_id_is_removed_before_the_download(
    world, gradient,
):
    """A crash after the start-up cleanup can leave an output yt-dlp would take as done."""
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.fetch_dir.mkdir(parents=True)
    (world.fetch_dir / "ytloft000001.part.mp4").write_bytes(b"left by a killed fetch")
    world.add_loft("ytloft000001")

    fetch("ytloft000001")

    assert world.youtube.reused_outputs() == []
    assert world.downloads() != []
    assert world.change_seconds("ytloft000001") == [float(s) for s in CHANGE_SECONDS]
    assert world.failures("ytloft000001") == 0
    assert world.flags("ytloft000001")[0] is True
    assert world.fetch_dir_files() == []


def test_spec_addon_117_i2_a_temporary_left_by_yt_dlp_is_not_the_finished_file(
    world, gradient,
):
    """yt-dlp writes ``<out>.part`` first; an exit 0 that leaves only that is no download.

    SPEC-ADDON-119 I2: the failure follows a success, so it is counted.
    """
    _succeed_once(world, gradient)
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.youtube.set_download(temp=True)
    world.add_loft("ytloft000001")

    fetch("ytloft000001")

    assert world.youtube.download_notes() != []
    assert world.failures("ytloft000001") == 1
    assert world.flags("ytloft000001")[0] is False
    assert world.scan_rows("ytloft000001") == 0
    assert world.fetch_dir_files() == []


@pytest.mark.parametrize("outcome", ["success", "no-rendition", "download-failure"])
def test_spec_addon_117_i2_no_log_line_names_a_url_a_video_id_or_a_header(
    world, gradient, caplog, outcome,
):
    log_everything(caplog)
    world.serve(gradient, duration=VIDEO_SECONDS)
    if outcome == "no-rendition":
        world.youtube.set_ytdlp(describe=describe([], duration=VIDEO_SECONDS))
    if outcome == "download-failure":
        world.youtube.override("/v/rendition", status=500)
    world.add_loft("ytloft000001")

    fetch("ytloft000001")

    assert world.source("ytloft000001") == "youtube"
    logged = everything_logged(caplog)
    for secret in (SIGNED_TOKEN, HEADER_TOKEN, "dQw4w9WgXcQ", world.youtube.base):
        assert secret not in logged



# ---------------------------------------------------------------------------
# I7 - OCR requested, and the workers without a downloaded source
# ---------------------------------------------------------------------------


async def test_spec_addon_117_i7_under_manual_a_frame_ocr_reindex_makes_the_next_fetch_read(
    world, gradient,
):
    from app.indexer import IndexManager
    from app.routers import files as files_router
    from tests.search_endpoint_db import patch_everywhere

    world.configure(frame_ocr="manual")
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000001", frame_scan_indexed=True)
    manager = IndexManager()
    patch_everywhere(world.monkeypatch, "get_index_manager", lambda: manager)

    await files_router.reindex_file(
        "ytloft000001", body={"tasks": ["frame_ocr"]}, drive=world.value("ytloft000001", "drive"),
    )
    await asyncio.to_thread(fetch, "ytloft000001")

    assert world.lookups() == 1
    assert world.frame_texts("ytloft000001") != []
    assert world.flags("ytloft000001") == (True, True, True)


def test_spec_addon_117_i7_the_ocr_worker_given_a_loft_without_a_download_changes_nothing(
    world,
):
    from app.workers import frame_ocr

    world.configure(frame_ocr="on_index")
    world.add_loft("ytloft000001", frame_scan_indexed=True,
                   frame_ocr_read=False, frame_ocr_indexed=False)

    frame_ocr.index_frame_ocr("ytloft000001")

    assert world.flags("ytloft000001") == (True, False, False)
    assert world.frame_texts("ytloft000001") == []
    assert world.engine.calls == 0



def test_spec_addon_117_i2_a_known_duration_download_is_stopped_at_one_and_a_half_times_it(
    world, gradient, clock,
):
    """800 s of lookup gives a 1200 s limit: running at 1000 s, stopped and counted past 1200 s.

    SPEC-ADDON-119 I2: the failure follows a success, so it is counted.
    """
    _succeed_once(world, gradient)
    world.serve(gradient, duration=800.0)
    world.youtube.set_download(grow=[16, 10.0])
    world.add_loft("ytloft000001")
    seen: dict[str, bool] = {}

    def size() -> int:
        return sum(p.stat().st_size for p in world.fetch_dir.glob("*") if p.is_file())

    def drive():
        _wait_until(lambda: world.youtube.partials())
        for moved in (100.0,) * 10:
            before = size()
            _wait_until(lambda: size() > before, 2.0)
            clock.advance(moved)
        # The fetch reads the clock every 0.2 s; let it read 1000 s at least
        # twice while the stream still grows.
        time.sleep(0.6)
        seen["running_at_1000"] = not world.youtube.downloads_stopped()
        for moved in (100.0, 100.0, 1.0):
            before = size()
            _wait_until(lambda: size() > before, 2.0)
            clock.advance(moved)

    threading.Thread(target=drive, daemon=True).start()
    fetch("ytloft000001")

    assert seen == {"running_at_1000": True}
    assert world.failures("ytloft000001") == 1
    assert world.flags("ytloft000001")[0] is False
    assert world.fetch_dir_files() == []
    assert world.youtube.downloads_stopped()
