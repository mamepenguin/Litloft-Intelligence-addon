"""SPEC-ADDON-076: ``GET /files/{file_id}/frame`` serves a YouTube ``.loft``.

``yt-dlp`` and ``ffmpeg`` are the executables of ``tests.youtube_frame_doubles``;
the stream is a local HTTP server, so each frame is a real decode.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from app.frame_cache import frame_cache_path
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    DRIVE,
    HEADER_TOKEN,
    LOFT_MIME,
    REAL_FFMPEG,
    SIGNED_TOKEN,
    YOUTUBE_URL,
    clock,
    describe,
    everything_logged,
    frame_api,
    needs_ffmpeg,
    webp_size,
    youtube,
)

pytestmark = needs_ffmpeg

DEADLINE = 15.0
COOKIE_OPTIONS = (
    "--cookies", "--cookies-from-browser", "--username", "--password",
    "--netrc", "--video-password", "--ap-username", "--add-header",
)


def new_id() -> str:
    return "yt" + uuid.uuid4().hex[:10]


def standard(youtube):
    """A description with one usable 720p stream; returns the format."""
    fmt = youtube.format("f720", 720)
    youtube.set_ytdlp(describe=describe([fmt]))
    return fmt


class TestNormalFlow:
    async def test_small_frame_is_320_wide_and_cached_at_the_native_path(
        self, youtube, frame_api, tmp_path,
    ):
        """SPEC-ADDON-076 I1: small is 320 px wide and lands on the native cache path."""
        standard(youtube)
        fid = new_id()
        frame_api.add_loft(fid)

        response = await frame_api.frame(fid, 1.0)

        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("image/webp")
        assert webp_size(response.content, tmp_path)[0] == 320
        cached = frame_cache_path(fid, 1.0)
        assert cached.read_bytes() == response.content
        assert frame_api.cached(fid) == ["1000.webp"]

    async def test_large_frame_fits_1280_and_has_its_own_cache_file(
        self, youtube, frame_api, tmp_path,
    ):
        """SPEC-ADDON-076 I1: large keeps the stream's own size up to 1280 px."""
        standard(youtube)
        fid = new_id()
        frame_api.add_loft(fid)

        small = await frame_api.frame(fid, 1.0, "small")
        large = await frame_api.frame(fid, 1.0, "large")

        assert large.status_code == 200, large.text
        assert webp_size(large.content, tmp_path) == (1280, 720)
        assert frame_cache_path(fid, 1.0, "large").read_bytes() == large.content
        assert frame_cache_path(fid, 1.0).read_bytes() == small.content
        assert frame_api.cached(fid) == ["1000.large.webp", "1000.webp"]

    async def test_large_is_never_enlarged_past_the_stream(
        self, youtube, frame_api, tmp_path,
    ):
        """SPEC-ADDON-076 I1: a 360 p stream gives a 640 px wide large frame."""
        fmt = youtube.format("f360", 360)
        youtube.set_ytdlp(describe=describe([fmt]))
        fid = new_id()
        frame_api.add_loft(fid)

        response = await frame_api.frame(fid, 1.0, "large")

        assert webp_size(response.content, tmp_path) == (640, 360)

    async def test_the_response_headers_are_those_of_a_native_frame(
        self, youtube, frame_api, tmp_path,
    ):
        """SPEC-ADDON-076 step 6: the same headers as for a native frame."""
        standard(youtube)
        native_path = tmp_path / "native.mp4"
        shutil.copy(youtube.videos_dir / "1280x720.mp4", native_path)
        native = new_id()
        frame_api.add_row(native, native_path, mime="video/mp4", duration=4.0)
        fid = new_id()
        frame_api.add_loft(fid)

        from_native = await frame_api.frame(native, 1.0, "large")
        from_loft = await frame_api.frame(fid, 1.0, "large")

        assert from_loft.status_code == from_native.status_code == 200
        drop = {"content-length", "etag", "last-modified"}
        assert {k: v for k, v in from_loft.headers.items() if k not in drop} == {
            k: v for k, v in from_native.headers.items() if k not in drop
        }

    async def test_a_repeated_request_is_answered_from_the_cache(self, youtube, frame_api):
        """SPEC-ADDON-076 step 2: a hit runs neither the lookup nor ffmpeg."""
        standard(youtube)
        fid = new_id()
        frame_api.add_loft(fid)

        first = await frame_api.frame(fid, 2.0)
        second = await frame_api.frame(fid, 2.0)

        assert second.status_code == 200
        assert second.content == first.content
        assert len(youtube.lookups()) == 1
        assert len([e for e in youtube.ffmpeg_events() if e["kind"] == "start"]) == 1

    async def test_two_requests_for_one_time_leave_one_whole_file(self, youtube, frame_api):
        """SPEC-ADDON-076 failure table: unique temp names, last replace wins."""
        standard(youtube)
        fid = new_id()
        frame_api.add_loft(fid)

        a, b = await asyncio.gather(frame_api.frame(fid, 1.0), frame_api.frame(fid, 1.0))

        assert a.status_code == b.status_code == 200
        assert a.content == b.content
        assert frame_api.cached(fid) == ["1000.webp"]


class TestWhatIsGivenToTheTools:
    async def test_yt_dlp_gets_only_the_stored_url_with_no_playlist_and_no_credentials(
        self, youtube, frame_api,
    ):
        """SPEC-ADDON-076 I5: the lookup is ``-J --no-playlist <url>`` and nothing more."""
        standard(youtube)
        fid = new_id()
        frame_api.add_loft(fid)

        await frame_api.frame(fid, 1.0)

        (lookup,) = youtube.lookups()
        argv = lookup["argv"]
        assert "-J" in argv
        assert "--no-playlist" in argv
        assert argv.count(YOUTUBE_URL) == 1
        for option in COOKIE_OPTIONS:
            assert not any(a == option or a.startswith(option + "=") for a in argv)
        assert all(a.startswith("-") or a == YOUTUBE_URL for a in argv)

    async def test_ffmpeg_reads_the_lookups_address_with_the_formats_headers(
        self, youtube, frame_api,
    ):
        """SPEC-ADDON-076 step 5/I5: only the returned address, with its request headers."""
        fmt = standard(youtube)
        fid = new_id()
        frame_api.add_loft(fid)

        await frame_api.frame(fid, 3.0)

        (start,) = [e for e in youtube.ffmpeg_events() if e["kind"] == "start"]
        argv = start["argv"]
        assert fmt["url"] in argv
        assert "-ss" in argv
        assert not any(a.startswith("http") and a != fmt["url"] for a in argv)
        stream_requests = youtube.fetched("/v/")
        assert stream_requests
        assert all(r["path"] == "/v/f720" for r in stream_requests)
        assert all(r["headers"].get("x-fake-auth") == HEADER_TOKEN for r in stream_requests)

    @pytest.mark.parametrize(
        ("label", "formats", "expected"),
        [
            ("greatest height not above 720", [("f480", 480, {}), ("f720", 720, {}), ("f1080", 1080, {})], "f720"),
            ("h264 over vp9 at one height", [("vp9", 720, {"vcodec": "vp09.00.40.08", "ext": "mp4"}), ("avc", 720, {})], "avc"),
            ("mp4 over webm at one height", [("webm", 720, {"vcodec": "vp09.00.40.08", "ext": "webm"}), ("mp4", 720, {"vcodec": "vp09.00.40.08", "ext": "mp4"})], "mp4"),
            ("h264 before container", [("avcwebm", 720, {"ext": "webm"}), ("vp9mp4", 720, {"vcodec": "vp09.00.40.08", "ext": "mp4"})], "avcwebm"),
            ("height before codec", [("avc480", 480, {}), ("vp9720", 720, {"vcodec": "vp09.00.40.08", "ext": "webm"})], "vp9720"),
            ("plain https only", [("hls", 720, {"protocol": "m3u8_native"}), ("dash", 720, {"protocol": "http_dash_segments"}), ("plain", 360, {})], "plain"),
            ("video only", [("muxed", 720, {"acodec": "mp4a.40.2"}), ("silent", 480, {})], "silent"),
        ],
    )
    async def test_the_chosen_format(self, label, formats, expected, youtube, frame_api):
        """SPEC-ADDON-076 step 5: video-only plain HTTPS, <= 720 p, H.264 then mp4."""
        built = [youtube.format(fid, height, **extra) for fid, height, extra in formats]
        youtube.set_ytdlp(describe=describe(built))
        fid = new_id()
        frame_api.add_loft(fid)

        response = await frame_api.frame(fid, 1.0, "large")

        assert response.status_code == 200, (label, response.text)
        assert {r["path"] for r in youtube.fetched("/v/")} == {f"/v/{expected}"}, label


class TestNoFrameSource:
    @pytest.mark.parametrize(
        "body",
        [
            {"provider": "vimeo", "url": "https://vimeo.com/1"},
            {"provider": "generic", "url": "https://example.com/v"},
            "not json at all",
            "[1, 2]",
            {"provider": "youtube"},
            {"provider": "youtube", "url": ""},
            {"provider": "youtube", "url": 42},
        ],
        ids=["vimeo", "generic", "not-json", "array", "no-url", "empty-url", "numeric-url"],
    )
    async def test_a_loft_without_a_youtube_source_is_400_and_caches_nothing(
        self, body, youtube, frame_api, caplog,
    ):
        """SPEC-ADDON-076 I1: 400 ``No frame source``, no lookup, no cache."""
        standard(youtube)
        fid = new_id()
        frame_api.add_loft(fid, body)

        with caplog.at_level("DEBUG"):
            response = await frame_api.frame(fid, 1.0)

        assert response.status_code == 400
        assert response.json()["detail"] == "No frame source"
        assert frame_api.cached(fid) == []
        assert youtube.lookups() == []
        assert youtube.ffmpeg_events() == []

    async def test_a_missing_loft_file_is_400(self, youtube, frame_api):
        """SPEC-ADDON-076 I1: an unreadable ``.loft`` is 400 too."""
        standard(youtube)
        fid = new_id()
        path = frame_api.add_loft(fid)
        path.unlink()

        response = await frame_api.frame(fid, 1.0)

        assert response.status_code == 400
        assert response.json()["detail"] == "No frame source"
        assert youtube.lookups() == []

    async def test_the_other_provider_is_logged_as_not_youtube(self, youtube, frame_api, caplog):
        """SPEC-ADDON-076 item 8: one WARNING with the file id and ``not_youtube``."""
        fid = new_id()
        frame_api.add_loft(fid, {"provider": "vimeo", "url": "https://vimeo.com/1"})

        with caplog.at_level("DEBUG"):
            await frame_api.frame(fid, 1.0)

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any(fid in r.getMessage() and "not_youtube" in r.getMessage() for r in warnings)


class TestLookupFailure:
    @pytest.mark.parametrize(
        "how", ["fails", "missing", "no-format", "only-hls", "only-above-720"],
    )
    async def test_a_lookup_that_gives_no_stream_is_502(self, how, youtube, frame_api):
        """SPEC-ADDON-076 I2: 502 ``Frame source unavailable``, no cache file."""
        if how == "fails":
            youtube.set_ytdlp(fail="ERROR: [youtube] x: Video unavailable")
        elif how == "missing":
            youtube.remove_ytdlp()
        elif how == "no-format":
            youtube.set_ytdlp(describe=describe([]))
        elif how == "only-hls":
            youtube.set_ytdlp(describe=describe(
                [youtube.format("hls", 720, protocol="m3u8_native")]
            ))
        else:
            youtube.set_ytdlp(describe=describe([youtube.format("f1080", 1080)]))
        fid = new_id()
        frame_api.add_loft(fid)

        response = await frame_api.frame(fid, 1.0)

        assert response.status_code == 502
        assert response.json()["detail"] == "Frame source unavailable"
        assert frame_api.cached(fid) == []
        assert youtube.ffmpeg_events() == []

    async def test_a_failed_lookup_is_not_kept(self, youtube, frame_api):
        """SPEC-ADDON-076 I4: the next request looks up again."""
        youtube.set_ytdlp(fail="ERROR: [youtube] x: Video unavailable")
        fid = new_id()
        frame_api.add_loft(fid)
        assert (await frame_api.frame(fid, 1.0)).status_code == 502

        standard(youtube)
        second = await frame_api.frame(fid, 1.0)

        assert second.status_code == 200
        assert len(youtube.lookups()) == 2

    async def test_the_failure_is_logged_by_class_without_urls_or_headers(
        self, youtube, frame_api, caplog,
    ):
        """SPEC-ADDON-076 item 8/I5: class, file id, yt-dlp's last line with URLs removed."""
        youtube.set_ytdlp(fail=(
            "WARNING: first line\n"
            f"ERROR: [youtube] x: Video unavailable, see {YOUTUBE_URL} for details"
        ))
        fid = new_id()
        frame_api.add_loft(fid)

        with caplog.at_level("DEBUG"):
            await frame_api.frame(fid, 1.0)

        logged = everything_logged(caplog)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any(fid in r.getMessage() and "lookup_failed" in r.getMessage() for r in warnings)
        assert "Video unavailable" in logged
        assert "youtube.com/watch" not in logged
        assert "dQw4w9WgXcQ" not in logged.replace(fid, "")

    async def test_no_usable_format_is_logged_as_no_format(self, youtube, frame_api, caplog):
        """SPEC-ADDON-076 item 8: ``no_format`` with the file id."""
        youtube.set_ytdlp(describe=describe([]))
        fid = new_id()
        frame_api.add_loft(fid)

        with caplog.at_level("DEBUG"):
            await frame_api.frame(fid, 1.0)

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any(fid in r.getMessage() and "no_format" in r.getMessage() for r in warnings)


class TestFfmpegFailure:
    @pytest.mark.parametrize("how", ["forbidden", "not-a-video"])
    async def test_a_stream_ffmpeg_cannot_read_is_500_with_nothing_left(
        self, how, youtube, frame_api, caplog,
    ):
        """SPEC-ADDON-076 I2: 500, no cache file, no temp file, no other frame."""
        fmt = standard(youtube)
        if how == "forbidden":
            youtube.override("/v/f720", status=403)
        else:
            youtube.server.files["/v/f720"] = b"this is not a video" * 100
        fid = new_id()
        frame_api.add_loft(fid)
        other = frame_cache_path(fid, 0.0, "large")
        other.parent.mkdir(parents=True)
        other.write_bytes(b"A DIFFERENT TIME")

        with caplog.at_level("DEBUG"):
            response = await frame_api.frame(fid, 1.0, "large")

        assert response.status_code == 500
        assert response.content != b"A DIFFERENT TIME"
        assert frame_api.cached(fid) == ["0.large.webp"]
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any(fid in r.getMessage() and "ffmpeg_failed" in r.getMessage() for r in warnings)
        assert fmt["url"] not in everything_logged(caplog)

    async def test_the_description_is_dropped_after_an_ffmpeg_failure(self, youtube, frame_api):
        """SPEC-ADDON-076 I4: the next request looks the stream up again."""
        standard(youtube)
        youtube.override("/v/f720", status=403)
        fid = new_id()
        frame_api.add_loft(fid)
        assert (await frame_api.frame(fid, 1.0)).status_code == 500
        youtube.override("/v/f720")

        second = await frame_api.frame(fid, 1.0)

        assert second.status_code == 200
        assert len(youtube.lookups()) == 2

    async def test_no_other_format_is_tried(self, youtube, frame_api):
        """SPEC-ADDON-076 step 5: a decode failure is not retried on another stream."""
        built = [youtube.format("avc", 720), youtube.format("vp9", 720, vcodec="vp09.00.40.08", ext="webm")]
        youtube.set_ytdlp(describe=describe(built))
        youtube.override("/v/avc", status=403)
        fid = new_id()
        frame_api.add_loft(fid)

        response = await frame_api.frame(fid, 1.0)

        assert response.status_code == 500
        assert youtube.fetched("/v/avc")
        assert youtube.fetched("/v/vp9") == []
        assert len(youtube.lookups()) == 1
        assert frame_api.cached(fid) == []


class TestDescriptionReuse:
    async def test_a_description_is_reused_for_five_minutes(self, youtube, frame_api, clock):
        """SPEC-ADDON-076 I4: one lookup serves frames of the same file inside 5 minutes."""
        standard(youtube)
        fid = new_id()
        frame_api.add_loft(fid)

        assert (await frame_api.frame(fid, 1.0)).status_code == 200
        clock.advance(299)
        assert (await frame_api.frame(fid, 2.0)).status_code == 200
        assert len(youtube.lookups()) == 1

        clock.advance(2)
        assert (await frame_api.frame(fid, 3.0)).status_code == 200
        assert len(youtube.lookups()) == 2

    async def test_files_do_not_share_a_description(self, youtube, frame_api):
        """SPEC-ADDON-076 I4: the description is kept per file id."""
        standard(youtube)
        a, b = new_id(), new_id()
        frame_api.add_loft(a)
        frame_api.add_loft(b)

        await frame_api.frame(a, 1.0)
        await frame_api.frame(b, 1.0)

        assert len(youtube.lookups()) == 2


class TestConcurrency:
    async def test_requests_for_one_file_share_one_lookup(self, youtube, frame_api):
        """SPEC-ADDON-076 I3: single flight."""
        standard(youtube)
        youtube.set_ytdlp(describe=describe([youtube.format("f720", 720)]), delay=1.0)
        fid = new_id()
        frame_api.add_loft(fid)

        responses = await asyncio.gather(
            *(frame_api.frame(fid, float(t)) for t in range(5))
        )

        assert [r.status_code for r in responses] == [200] * 5
        assert len(youtube.lookups()) == 1

    async def test_a_failed_shared_lookup_fails_every_waiter_with_502(self, youtube, frame_api):
        """SPEC-ADDON-076 failure table: one lookup, all 502."""
        youtube.set_ytdlp(fail="ERROR: [youtube] x: Video unavailable", delay=1.0)
        fid = new_id()
        frame_api.add_loft(fid)

        responses = await asyncio.gather(
            *(frame_api.frame(fid, float(t)) for t in range(4))
        )

        assert [r.status_code for r in responses] == [502] * 4
        assert len(youtube.lookups()) == 1

    async def test_more_requests_than_slots_all_finish_with_four_ffmpeg_at_most(
        self, youtube, frame_api,
    ):
        """SPEC-ADDON-076 I3: six requests, at most four ffmpeg runs at once, all served."""
        standard(youtube)
        youtube.set_ffmpeg(hold=1.5)
        fid = new_id()
        frame_api.add_loft(fid)

        responses = await asyncio.gather(
            *(frame_api.frame(fid, float(t) / 2) for t in range(6))
        )

        assert [r.status_code for r in responses] == [200] * 6
        assert youtube.peak_ffmpeg_concurrency() <= 4
        assert len([e for e in youtube.ffmpeg_events() if e["kind"] == "start"]) == 6

    async def test_the_lookup_does_not_wait_for_an_ffmpeg_slot(self, youtube, frame_api):
        """SPEC-ADDON-076 I3: a full set of ffmpeg runs does not hold up another file's lookup."""
        standard(youtube)
        youtube.set_ffmpeg(hold=6.0)
        busy, other = new_id(), new_id()
        frame_api.add_loft(busy)
        frame_api.add_loft(other)

        holders = [asyncio.create_task(frame_api.frame(busy, float(t))) for t in range(4)]
        await asyncio.sleep(2.0)
        started = time.time()
        waiting = asyncio.create_task(frame_api.frame(other, 1.0))
        await asyncio.sleep(1.5)

        lookups = youtube.lookups()
        assert len(lookups) == 2
        assert lookups[-1]["t"] - started < 1.5
        results = await asyncio.gather(*holders, waiting)
        assert [r.status_code for r in results] == [200] * 5


class TestTheDeadline:
    async def test_lookup_and_ffmpeg_share_one_15_second_deadline(
        self, youtube, frame_api, caplog,
    ):
        """SPEC-ADDON-076 I3/I2: 504 after 15 s in all, ffmpeg killed, nothing cached or kept."""
        standard(youtube)
        youtube.set_ytdlp(describe=describe([youtube.format("f720", 720)]), delay=8.0)
        youtube.set_ffmpeg(hold=8.0)
        fid = new_id()
        frame_api.add_loft(fid)

        started = time.monotonic()
        with caplog.at_level("DEBUG"):
            response = await frame_api.frame(fid, 1.0)
        elapsed = time.monotonic() - started

        assert response.status_code == 504
        assert response.json()["detail"] == "Frame extraction timed out"
        assert DEADLINE - 1.5 < elapsed < DEADLINE + 3.0
        assert frame_api.cached(fid) == []
        (start,) = [e for e in youtube.ffmpeg_events() if e["kind"] == "start"]
        await asyncio.sleep(0.5)
        with pytest.raises(ProcessLookupError):
            os.kill(start["pid"], 0)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any(fid in r.getMessage() and "timeout" in r.getMessage() for r in warnings)

        youtube.set_ytdlp(describe=describe([youtube.format("f720", 720)]))
        youtube.set_ffmpeg(hold=0)
        again = await frame_api.frame(fid, 1.0)
        assert again.status_code == 200
        assert len(youtube.lookups()) == 2

    async def test_a_request_that_cannot_take_a_slot_in_time_is_504(self, youtube, frame_api):
        """SPEC-ADDON-076 I3: four runs hold the slots, the fifth request waits and times out."""
        standard(youtube)
        youtube.set_ffmpeg(hold=60.0)
        fid = new_id()
        frame_api.add_loft(fid)

        started = time.monotonic()
        responses = await asyncio.gather(
            *(frame_api.frame(fid, float(t)) for t in range(5))
        )
        elapsed = time.monotonic() - started

        assert [r.status_code for r in responses] == [504] * 5
        assert elapsed < DEADLINE + 3.0
        starts = [e for e in youtube.ffmpeg_events() if e["kind"] == "start"]
        assert len(starts) == 4
        await asyncio.sleep(0.5)
        for event in starts:
            with pytest.raises(ProcessLookupError):
                os.kill(event["pid"], 0)
        assert frame_api.cached(fid) == []
        assert len(youtube.lookups()) == 1


class TestNothingSecretIsLogged:
    async def test_no_log_line_has_a_url_or_a_header_on_success_or_failure(
        self, youtube, frame_api, caplog,
    ):
        """SPEC-ADDON-076 I5: neither the stored URL, a signed URL nor a request header value."""
        fmt = standard(youtube)
        ok, bad = new_id(), new_id()
        frame_api.add_loft(ok)
        frame_api.add_loft(bad)

        with caplog.at_level("DEBUG"):
            served = await frame_api.frame(ok, 1.0, "large")
            youtube.override("/v/f720", status=403)
            refused = await frame_api.frame(bad, 1.0, "large")

        assert served.status_code == 200
        assert refused.status_code == 500

        logged = everything_logged(caplog)
        for secret in (YOUTUBE_URL, "dQw4w9WgXcQ", SIGNED_TOKEN, HEADER_TOKEN, fmt["url"]):
            assert secret not in logged, secret
