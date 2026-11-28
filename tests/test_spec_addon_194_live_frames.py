"""SPEC-ADDON-194 I16, with SPEC-ADDON-193 I13: a live frame of a web reference.

A `web` `.loft` below the failure cap reads its frame from the rendition
``pick_format`` returns, through ffmpeg with that format's request headers. A
capped one, an `other` one and an unclassified one answer 400 and run nothing.
"""

from __future__ import annotations

import uuid

import pytest

from app.frame_cache import frame_cache_path
from app.models import IndexedFile
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    HEADER_TOKEN,
    SIGNED_TOKEN,
    YOUTUBE_URL,
    clock,
    describe,
    everything_logged,
    frame_api,
    log_everything,
    needs_ffmpeg,
    video_format,
    webp_size,
    youtube,
)

pytestmark = needs_ffmpeg

WEB_BODY = {
    "provider": "example-video",
    "url": "https://videos.example-video.test/watch/abcdef123456?token=zzzzzzz9",
}


def new_id() -> str:
    return "wf" + uuid.uuid4().hex[:10]


def _set(frame_api, file_id: str, **columns) -> None:
    with frame_api.Session() as session:
        session.query(IndexedFile).filter_by(file_id=file_id).update(columns)
        session.commit()


def _web_loft(frame_api, *, source: str | None = "web", failures: int = 0) -> str:
    file_id = new_id()
    frame_api.add_loft(file_id, WEB_BODY, duration=600.0)
    _set(frame_api, file_id, loft_source=source, loft_fetch_failures=failures)
    return file_id


def _serve(youtube, *, acodec: str = "none") -> dict:
    fmt = youtube.format("f720", 720, acodec=acodec)
    youtube.set_ytdlp(describe=describe([fmt]))
    return fmt


async def test_spec_addon_194_i16_a_web_loft_below_the_cap_reads_its_frame_live(
    youtube, frame_api, tmp_path,
):
    fmt = _serve(youtube)
    file_id = _web_loft(frame_api)

    response = await frame_api.frame(file_id, 1.0)

    assert response.status_code == 200, response.text
    assert webp_size(response.content, tmp_path)[0] == 320
    assert frame_cache_path(file_id, 1.0).read_bytes() == response.content
    (lookup,) = youtube.lookups()
    assert WEB_BODY["url"] in lookup["argv"]
    (start,) = [e for e in youtube.ffmpeg_events() if e["kind"] == "start"]
    assert fmt["url"] in start["argv"]
    assert not any("abcdef123456" in arg for arg in start["argv"])
    stream = youtube.fetched("/v/")
    assert stream
    assert all(r["headers"].get("x-fake-auth") == HEADER_TOKEN for r in stream)


async def test_spec_addon_194_i16_a_muxed_rendition_gives_a_web_lofts_frame(
    youtube, frame_api, tmp_path,
):
    path = youtube.serve_video("muxed", 1280, 720)
    fmt = video_format(
        "muxed", 720, acodec="mp4a.40.2",
        url=f"{youtube.base}{path}?sig={SIGNED_TOKEN}",
    )
    youtube.set_ytdlp(describe=describe([fmt]))
    file_id = _web_loft(frame_api)

    response = await frame_api.frame(file_id, 1.0)

    assert response.status_code == 200, response.text
    assert webp_size(response.content, tmp_path)[0] == 320
    assert {r["path"] for r in youtube.fetched("/v/")} == {"/v/muxed"}


@pytest.mark.parametrize("failures", [0, 2])
async def test_spec_addon_194_i16_a_web_loft_below_the_cap_is_a_frame_source(
    failures, youtube, frame_api,
):
    _serve(youtube)
    file_id = _web_loft(frame_api, failures=failures)

    response = await frame_api.frame(file_id, 1.0)

    assert response.status_code == 200, response.text


async def test_spec_addon_194_i16_a_capped_web_loft_answers_no_frame_source(
    youtube, frame_api,
):
    _serve(youtube)
    file_id = _web_loft(frame_api, failures=3)

    response = await frame_api.frame(file_id, 1.0)

    assert response.status_code == 400
    assert response.json()["detail"] == "No frame source"
    assert frame_api.cached(file_id) == []
    assert youtube.lookups() == []
    assert youtube.ffmpeg_events() == []


@pytest.mark.parametrize(
    ("source", "body", "failures"),
    [
        pytest.param("other", WEB_BODY, 0, id="other"),
        pytest.param(None, WEB_BODY, 0, id="unclassified-web-provider"),
        pytest.param("web", {"provider": "generic", "url": WEB_BODY["url"]}, 0,
                     id="generic"),
    ],
)
async def test_spec_addon_194_i16_a_loft_that_is_no_frame_source_runs_nothing(
    source, body, failures, youtube, frame_api,
):
    _serve(youtube)
    file_id = new_id()
    frame_api.add_loft(file_id, body, duration=600.0)
    _set(frame_api, file_id, loft_source=source, loft_fetch_failures=failures)

    response = await frame_api.frame(file_id, 1.0)

    assert response.status_code == 400
    assert response.json()["detail"] == "No frame source"
    assert frame_api.cached(file_id) == []
    assert youtube.lookups() == []
    assert youtube.ffmpeg_events() == []


async def test_spec_addon_194_i16_a_youtube_loft_of_any_source_is_read_live(
    youtube, frame_api,
):
    _serve(youtube)
    file_id = new_id()
    frame_api.add_loft(file_id, {"provider": "youtube", "url": YOUTUBE_URL})
    _set(frame_api, file_id, loft_source="other")

    response = await frame_api.frame(file_id, 1.0)

    assert response.status_code == 200, response.text
    assert len(youtube.lookups()) == 1


async def test_spec_addon_194_i16_a_web_lofts_description_is_reused_for_five_minutes(
    youtube, frame_api, clock,
):
    _serve(youtube)
    file_id = _web_loft(frame_api)

    assert (await frame_api.frame(file_id, 1.0)).status_code == 200
    clock.advance(299)
    assert (await frame_api.frame(file_id, 2.0)).status_code == 200
    assert len(youtube.lookups()) == 1

    clock.advance(2)
    assert (await frame_api.frame(file_id, 3.0)).status_code == 200
    assert len(youtube.lookups()) == 2


async def test_spec_addon_194_i13_a_failed_live_read_carries_no_url_or_header(
    youtube, frame_api, caplog,
):
    log_everything(caplog)
    _serve(youtube)
    youtube.override("/v/f720", status=403)
    file_id = _web_loft(frame_api)

    response = await frame_api.frame(file_id, 1.0, "large")

    assert response.status_code == 500
    logged = everything_logged(caplog)
    for secret in (WEB_BODY["url"], "abcdef123456", "zzzzzzz9", SIGNED_TOKEN,
                   HEADER_TOKEN, youtube.base):
        assert secret not in logged, secret
        assert secret not in response.text, secret
    assert frame_api.cached(file_id) == []
