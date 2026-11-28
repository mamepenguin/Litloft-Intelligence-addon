"""SPEC-ADDON-193 I1-I3, I6, I12, I13: a fetch of a web provider's ``.loft``.

The rendition, the lookup and the download are the doubles of the YouTube fetch
tests; the reference says ``example-video``, which is no real site.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app.frame_cache import frame_cache_path
from tests.loft_media_world import LoftWorld, fetch
from tests.ocr_fixtures import pillow_real  # noqa: F401 - the usefixtures marker resolves it
from tests.test_spec_addon_079_scene_tiles import tile_names, tiles_dir
from tests.test_spec_addon_111_frame_scan_task import _encode
from tests.test_spec_addon_119_loft_fetch_counting import (  # noqa: F401 - fixtures
    fresh_worker_state,
    gradient,
)
from tests.vec_partition_db import search_index  # noqa: F401 - fixture
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    HEADER_TOKEN,
    SIGNED_TOKEN,
    clock,
    describe,
    everything_logged,
    log_everything,
    needs_ffmpeg,
    storyboard_format,
    video_format,
    youtube,
)

pytestmark = [needs_ffmpeg, pytest.mark.usefixtures("pillow_real")]

WEB_PROVIDER = "example-video"
WEB_URL = "https://videos.example-video.test/watch/abcdef123456?token=zzzzzzz9"
VIDEO_SECONDS = 40.0
REFUSAL = "ERROR: [example-video] this video is private"
WARM_ID = "webloft000001"


@pytest.fixture()
def world(search_index, youtube, monkeypatch, tmp_path, make_settings):  # noqa: F811
    return LoftWorld(search_index, youtube, monkeypatch, tmp_path, make_settings)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    root = tmp_path_factory.mktemp("loft-web")
    return _encode(root / "switches.mp4")


def _body(provider: str = WEB_PROVIDER, url: str = WEB_URL) -> dict:
    return {"provider": provider, "url": url}


def _add_web(world, file_id: str, **flags):
    return world.add_loft(file_id, _body(), **flags)


def _succeed(world, video, file_id: str = WARM_ID) -> None:
    """A successful web fetch, so the next failure directly follows a success."""
    world.serve(video, duration=VIDEO_SECONDS)
    _add_web(world, file_id)
    fetch(file_id)
    assert world.source(file_id) == "web"
    assert world.flags(file_id)[0] is True


def _clip_seconds(world, file_id: str) -> list[float]:
    with world.index.engine.connect() as conn:
        return [r[0] for r in conn.execute(text(
            "SELECT timestamp_start FROM embeddings WHERE file_id = :f "
            "AND embedding_type = 'clip' ORDER BY timestamp_start"
        ), {"f": file_id})]


# ---------------------------------------------------------------------------
# I1 - classification before any network call
# ---------------------------------------------------------------------------


def test_spec_addon_193_i1_a_web_provider_is_written_before_any_network_call(world, video):
    world.serve(video, duration=VIDEO_SECONDS)
    _add_web(world, "webloft000001", frame_scan_indexed=True, loft_candidates_done=True)

    fetch("webloft000001")

    assert world.source("webloft000001") == "web"
    assert world.lookups() == 0
    assert world.downloads() == []
    assert world.flags("webloft000001") == (True, True, True)
    assert world.failures("webloft000001") == 0


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="no-body"),
        pytest.param("this is not json", id="not-json"),
        pytest.param({"provider": WEB_PROVIDER}, id="no-url"),
        pytest.param({"url": WEB_URL}, id="no-provider"),
        pytest.param({"provider": "generic", "url": WEB_URL}, id="generic"),
    ],
)
def test_spec_addon_193_i1_a_reference_with_no_video_provider_is_other_and_uncalled(
    world, video, body,
):
    world.serve(video, duration=VIDEO_SECONDS)
    world.add_loft(
        "other0000001", body, frame_ocr_read=False, frame_ocr_indexed=False,
    )

    fetch("other0000001")

    assert world.source("other0000001") == "other"
    assert world.flags("other0000001") == (True, True, True)
    assert world.value("other0000001", "loft_candidates_done") in (True, 1)
    assert world.lookups() == 0
    assert world.downloads() == []


def test_spec_addon_193_i1_a_youtube_provider_is_still_written_youtube(world, video):
    world.serve(video, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000001")

    fetch("ytloft000001")

    assert world.source("ytloft000001") == "youtube"


def test_spec_addon_193_i1_a_reference_that_cannot_be_read_writes_nothing(world, video):
    world.serve(video, duration=VIDEO_SECONDS)
    path = _add_web(world, "gone00000001", frame_ocr_read=False, frame_ocr_indexed=False)
    path.unlink()

    fetch("gone00000001")

    assert world.source("gone00000001") is None
    assert world.flags("gone00000001") == (False, False, False)
    assert world.failures("gone00000001") == 0
    assert world.lookups() == 0


# ---------------------------------------------------------------------------
# I2 - a lookup with no usable format
# ---------------------------------------------------------------------------


def test_spec_addon_193_i2_a_web_lookup_with_no_rendition_closes_the_file_uncounted(
    world, caplog,
):
    world.youtube.set_ytdlp(describe=describe([], duration=VIDEO_SECONDS))
    _add_web(world, "webloft000002", frame_ocr_read=False, frame_ocr_indexed=False)
    world.set("webloft000002", loft_fetch_failures=1)

    with caplog.at_level("INFO"):
        fetch("webloft000002")

    assert world.source("webloft000002") == "other"
    assert world.flags("webloft000002") == (True, True, True)
    assert world.value("webloft000002", "loft_candidates_done") in (True, 1)
    assert world.failures("webloft000002") == 1
    assert world.lookups() == 1
    assert world.downloads() == []
    assert "loft_media no rendition (file webloft000002)" in everything_logged(caplog)


def test_spec_addon_193_i2_the_same_case_for_a_youtube_loft_is_a_counted_failure(world):
    world.youtube.set_ytdlp(describe=describe([], duration=VIDEO_SECONDS))
    world.add_loft("ytloft000002", frame_ocr_read=False, frame_ocr_indexed=False)

    fetch("ytloft000002")

    assert world.source("ytloft000002") == "youtube"
    assert world.failures("ytloft000002") == 1
    assert world.downloads() == []


# ---------------------------------------------------------------------------
# I3 - what is never looked up or read
# ---------------------------------------------------------------------------


def test_spec_addon_193_i3_a_generic_loft_is_never_looked_up(world, video):
    world.serve(video, duration=VIDEO_SECONDS)
    world.add_loft(
        "generic00001", _body("generic"), frame_ocr_read=False, frame_ocr_indexed=False,
    )

    fetch("generic00001")

    assert world.source("generic00001") == "other"
    assert world.lookups() == 0
    assert world.downloads() == []


def test_spec_addon_193_i3_an_other_loft_of_a_web_provider_is_never_looked_up(
    world, video,
):
    world.serve(video, duration=VIDEO_SECONDS)
    _add_web(world, "otherweb0001", frame_ocr_read=False, frame_ocr_indexed=False)
    world.set("otherweb0001", loft_source="other")

    fetch("otherweb0001")

    assert world.source("otherweb0001") == "other"
    assert world.lookups() == 0
    assert world.downloads() == []


def test_spec_addon_193_i3_a_youtube_loft_of_source_other_is_still_read_live(world, video):
    world.serve(video, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000003")
    world.set("ytloft000003", loft_source="other")

    fetch("ytloft000003")

    assert world.source("ytloft000003") == "youtube"
    assert world.lookups() == 1
    assert world.scan_rows("ytloft000003") == 1


def test_spec_addon_193_i3_a_web_fetch_makes_no_storyboard_call(world, video, caplog):
    board = storyboard_format(world.youtube.base, [("sheet.png", VIDEO_SECONDS)])
    world.youtube.server.files["/sb/sheet.png"] = b"a sheet the fetch must not ask for"
    _add_web(world, "webloft000003", loft_candidates_done=True)
    path = "/v/rendition"
    world.youtube.server.files[path] = video.read_bytes()
    fmt = {
        "format_id": "rendition", "ext": "mp4", "protocol": "https",
        "vcodec": "avc1.4d401f", "acodec": "none", "height": 720, "width": 1280,
        "url": f"{world.youtube.base}{path}?sig={SIGNED_TOKEN}",
        "http_headers": {"X-Fake-Auth": HEADER_TOKEN},
    }
    world.youtube.set_ytdlp(describe=describe([fmt, board], duration=VIDEO_SECONDS))

    with caplog.at_level("INFO"):
        fetch("webloft000003")

    assert world.source("webloft000003") == "web"
    assert world.lookups() == 1
    assert world.youtube.fetched("/sb/") == []
    assert "youtube storyboard" not in everything_logged(caplog)
    assert tile_names("webloft000003") == []
    assert not tiles_dir("webloft000003").exists()


def test_spec_addon_193_i3_the_storyboard_leg_of_a_web_loft_reads_nothing(world, video):
    from app.workers import clip as clip_worker

    _add_web(world, "webloft000004", loft_candidates_done=True)
    lookups = world.lookups()

    assert clip_worker._index_clip_sync("webloft000004") is True

    assert world.lookups() == lookups
    assert world.youtube.fetched("/sb/") == []
    assert tile_names("webloft000004") == []


def test_spec_addon_193_section_1_4_a_muxed_rendition_is_downloaded_and_deleted(
    world, video,
):
    """Revised SPEC-ADDON-117 I2: with no video-only format, the muxed one is fetched."""
    path = "/v/muxed"
    world.youtube.server.files[path] = video.read_bytes()
    fmt = video_format(
        "muxed", 720, acodec="mp4a.40.2", url=f"{world.youtube.base}{path}?sig=x",
    )
    world.youtube.set_ytdlp(describe=describe([fmt], duration=VIDEO_SECONDS))
    _add_web(world, "webloft000010")

    fetch("webloft000010")

    assert world.source("webloft000010") == "web"
    assert world.lookups() == 1
    assert {r["path"] for r in world.downloads()} == {path}
    assert world.flags("webloft000010")[0] is True
    assert world.scan_rows("webloft000010") == 1
    assert world.failures("webloft000010") == 0
    assert world.fetch_dir_files() == []


# ---------------------------------------------------------------------------
# I6 - the same rows, pictures and flags as a YouTube reference
# ---------------------------------------------------------------------------


def test_spec_addon_193_i6_a_web_fetch_writes_what_a_youtube_fetch_writes(world, video):
    world.configure(frame_ocr="on_index")
    world.serve(video, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000004", frame_ocr_read=False, frame_ocr_indexed=False)
    _add_web(world, "webloft000005", frame_ocr_read=False, frame_ocr_indexed=False)

    fetch("ytloft000004")
    fetch("webloft000005")

    youtube_id, web_id = "ytloft000004", "webloft000005"
    assert world.source(youtube_id) == "youtube"
    assert world.source(web_id) == "web"
    assert world.flags(web_id) == world.flags(youtube_id) == (True, True, True)
    assert world.failures(web_id) == world.failures(youtube_id) == 0
    assert world.scan_rows(web_id) == world.scan_rows(youtube_id) == 1
    assert world.change_seconds(web_id) == world.change_seconds(youtube_id) == []
    assert world.frame_texts(web_id) == world.frame_texts(youtube_id) != []
    assert world.value(web_id, "loft_candidates_done") == world.value(
        youtube_id, "loft_candidates_done"
    ) in (True, 1)
    assert world.value(web_id, "clip_indexed") == world.value(youtube_id, "clip_indexed")
    assert _clip_seconds(web_id) == _clip_seconds(youtube_id) != []
    for timestamp in _clip_seconds(web_id):
        assert frame_cache_path(web_id, timestamp).exists()
        assert frame_cache_path(youtube_id, timestamp).exists()
    assert world.fetch_dir_files() == []


# ---------------------------------------------------------------------------
# I12 - one pacing state for web and YouTube fetches
# ---------------------------------------------------------------------------


def test_spec_addon_193_i12_a_web_success_makes_the_next_lookup_failure_counted(world, video):
    _succeed(world, video)
    world.youtube.set_ytdlp(fail=REFUSAL)
    world.add_loft("ytloft000005", frame_ocr_read=False, frame_ocr_indexed=False)

    fetch("ytloft000005")

    assert world.failures("ytloft000005") == 1


def test_spec_addon_193_i12_a_youtube_success_makes_the_next_web_failure_counted(
    world, video, caplog,
):
    world.serve(video, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000006")
    fetch("ytloft000006")
    assert world.flags("ytloft000006")[0] is True
    world.youtube.set_ytdlp(fail=REFUSAL)
    _add_web(world, "webloft000006", frame_ocr_read=False, frame_ocr_indexed=False)

    with caplog.at_level("INFO"):
        fetch("webloft000006")

    assert world.source("webloft000006") == "web"
    assert world.failures("webloft000006") == 1
    assert "loft_media lookup failed (file webloft000006)" in everything_logged(caplog)


def test_spec_addon_193_i12_a_web_lookup_with_no_rendition_is_an_answered_request(world):
    world.youtube.set_ytdlp(describe=describe([], duration=VIDEO_SECONDS))
    _add_web(world, "webloft000007", frame_ocr_read=False, frame_ocr_indexed=False)
    fetch("webloft000007")
    assert world.source("webloft000007") == "other"

    world.youtube.set_ytdlp(fail=REFUSAL)
    world.add_loft("ytloft000007", frame_ocr_read=False, frame_ocr_indexed=False)
    fetch("ytloft000007")

    assert world.failures("ytloft000007") == 1


def test_spec_addon_193_i12_the_pause_holds_a_youtube_fetch_after_a_web_success(
    world, video, clock,
):
    from app.workers import loft_media

    _succeed(world, video)

    assert loft_media.backlog_held() is True
    clock.advance(31)
    assert loft_media.backlog_held() is False


def test_spec_addon_193_i12_three_web_refusals_in_a_row_start_the_backoff(
    world, caplog,
):
    world.youtube.set_ytdlp(fail=REFUSAL)
    with caplog.at_level("INFO"):
        for n in range(3):
            _add_web(
                world, f"webloft00001{n}",
                frame_ocr_read=False, frame_ocr_indexed=False,
            )
            fetch(f"webloft00001{n}")

    lines = [
        record.getMessage() for record in caplog.records
        if "loft_media backing off" in record.getMessage()
    ]
    assert lines == ["loft_media backing off 30 min after 3 failures in a row"]


# ---------------------------------------------------------------------------
# I13 - nothing secret in a log line or a response
# ---------------------------------------------------------------------------


def test_spec_addon_193_i13_a_failed_web_lookup_logs_neither_url_nor_its_tokens(
    world, caplog,
):
    log_everything(caplog)
    world.youtube.set_ytdlp(fail=(
        "WARNING: first line\n"
        f"ERROR: [example-video] {WEB_URL} refused"
    ))
    _add_web(world, "webloft000008", frame_ocr_read=False, frame_ocr_indexed=False)

    fetch("webloft000008")

    logged = everything_logged(caplog)
    for secret in (WEB_URL, "videos.example-video.test", "abcdef123456",
                   "zzzzzzz9"):
        assert secret not in logged, secret
    assert "loft_media lookup failed (file webloft000008)" in logged


def test_spec_addon_193_i13_a_failed_web_download_logs_neither_signed_url_nor_header(
    world, video, caplog,
):
    log_everything(caplog)
    world.serve(video, duration=VIDEO_SECONDS)
    world.youtube.override("/v/rendition", status=403)
    _add_web(world, "webloft000009", frame_ocr_read=False, frame_ocr_indexed=False)

    fetch("webloft000009")

    logged = everything_logged(caplog)
    for secret in (SIGNED_TOKEN, HEADER_TOKEN, WEB_URL, "abcdef123456",
                   world.youtube.base):
        assert secret not in logged, secret
    assert "loft_media download failed (file webloft000009)" in logged
