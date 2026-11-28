"""SPEC-ADDON-193 I4 (r5)/I5: which format a fetch downloads and reads a frame from.

I4: ``pick_format`` returns the tallest video-only plain-HTTPS format up to 720,
otherwise the tallest muxed one by the same ranking, otherwise ``None``; it never
returns a format without a video codec, a URL, ``https`` or an integer height in
1..720.

I5: yt-dlp's own selector, given ``DOWNLOAD_FORMAT``, selects a format whenever
``pick_format`` returns one, and a video-only one whenever the description has a
qualifying video-only format.
"""

from __future__ import annotations

import pytest

from app.youtube_source import DOWNLOAD_FORMAT, pick_format

_KEEP_URL = object()


def _format(
    fid: str,
    height,
    *,
    acodec="none",
    vcodec="avc1.4d401f",
    ext="mp4",
    protocol="https",
    url=_KEEP_URL,
) -> dict:
    width = height * 16 // 9 // 2 * 2 if isinstance(height, int) and not isinstance(
        height, bool
    ) else 0
    return {
        "format_id": fid,
        "ext": ext,
        "protocol": protocol,
        "vcodec": vcodec,
        "acodec": acodec,
        "height": height,
        "width": width,
        "url": (
            f"https://cdn.example-video.test/{fid}?sig=abc123"
            if url is _KEEP_URL else url
        ),
    }


def video(fid: str, height, **kwargs) -> dict:
    return _format(fid, height, **kwargs)


def muxed(fid: str, height, **kwargs) -> dict:
    return _format(fid, height, acodec="mp4a.40.2", **kwargs)


def audio(fid: str, height: int = 0) -> dict:
    return _format(fid, height, acodec="mp4a.40.2", vcodec="none")


def _description(formats: list) -> dict:
    return {"id": "dQw4w9WgXcQ", "title": "fixture", "formats": formats}


# ---------------------------------------------------------------------------
# I4 - the rendition pick_format returns
# ---------------------------------------------------------------------------

PICKS = [
    pytest.param(
        "only muxed", [muxed("m360", 360)], "m360", id="only-muxed",
    ),
    pytest.param(
        "a null audio codec is muxed",
        [_format("m720", 720, acodec=None)], "m720", id="acodec-null",
    ),
    pytest.param(
        "an unknown audio codec is muxed",
        [_format("m720", 720, acodec="unknown")], "m720", id="acodec-unknown",
    ),
    pytest.param(
        "both kinds: the video-only format wins though the muxed one is taller",
        [video("v480", 480), muxed("m720", 720)], "v480", id="both-kinds",
    ),
    pytest.param(
        "the tallest video-only format",
        [video("v480", 480), video("v720", 720)], "v720", id="tallest-video-only",
    ),
    pytest.param(
        "avc1 wins a tie between containers",
        [video("vp9webm", 720, vcodec="vp09.00.40.08", ext="webm"),
         video("avcwebm", 720, ext="webm")],
        "avcwebm", id="tie-avc1-before-container",
    ),
    pytest.param(
        "mp4 wins a tie once the codec is equal",
        [video("vp9webm", 720, vcodec="vp09.00.40.08", ext="webm"),
         video("vp9mp4", 720, vcodec="vp09.00.40.08", ext="mp4")],
        "vp9mp4", id="tie-mp4",
    ),
    pytest.param(
        "avc1 before mp4",
        [video("avcwebm", 720, ext="webm"),
         video("vp9mp4", 720, vcodec="vp09.00.40.08", ext="mp4")],
        "avcwebm", id="tie-avc1-before-mp4",
    ),
    pytest.param(
        "height before codec",
        [video("avc480", 480),
         video("vp9720", 720, vcodec="vp09.00.40.08", ext="webm")],
        "vp9720", id="height-before-codec",
    ),
    pytest.param(
        "the muxed ranking is the same",
        [muxed("vp9webm", 720, vcodec="vp09.00.40.08", ext="webm"),
         muxed("avcwebm", 720, ext="webm")],
        "avcwebm", id="muxed-tie-avc1",
    ),
    pytest.param(
        "a muxed mp4 wins its tie",
        [muxed("vp9webm", 720, vcodec="vp09.00.40.08", ext="webm"),
         muxed("vp9mp4", 720, vcodec="vp09.00.40.08", ext="mp4")],
        "vp9mp4", id="muxed-tie-mp4",
    ),
    pytest.param(
        "an audio-only format is no video",
        [audio("a128", 720)], None, id="only-audio",
    ),
    pytest.param(
        "an audio-only format does not outrank a muxed one",
        [audio("a720", 720), muxed("m360", 360)], "m360", id="audio-beside-muxed",
    ),
    pytest.param("height zero", [video("v0", 0)], None, id="height-zero"),
    pytest.param("height above the cap", [video("v721", 721)], None, id="height-721"),
    pytest.param("height is a bool", [video("vtrue", True)], None, id="height-bool"),
    pytest.param("height is a string", [video("vstr", "720")], None, id="height-string"),
    pytest.param("height is a float", [video("vfloat", 720.0)], None, id="height-float"),
    pytest.param(
        "only HLS formats",
        [video("hls", 720, protocol="m3u8_native")], None, id="only-hls",
    ),
    pytest.param(
        "only DASH formats",
        [video("dash", 720, protocol="http_dash_segments")], None, id="only-dash",
    ),
    pytest.param(
        "the video codec is the string none",
        [video("vc", 720, vcodec="none")], None, id="vcodec-none",
    ),
    pytest.param(
        "the video codec is null",
        [video("vc", 720, vcodec=None)], "vc", id="vcodec-null",
    ),
    pytest.param(
        "progressive formats with both codecs unknown beside HLS, as one provider lists them",
        [
            _format("p240", 240, vcodec=None, acodec=None),
            _format("p720", 720, vcodec=None, acodec=None),
            _format("p1080", 1080, vcodec=None, acodec=None),
            muxed("hls720", 720, protocol="m3u8_native"),
        ],
        "p720",
        id="unknown-codecs-progressive",
    ),
    pytest.param(
        "a muxed format of known codec outranks a taller one of unknown codec",
        [muxed("m360", 360), _format("u720", 720, vcodec=None, acodec=None)],
        "m360",
        id="known-muxed-before-unknown",
    ),
    pytest.param(
        "an unknown codec without a height",
        [_format("u", None, vcodec=None, acodec=None)], None, id="unknown-codec-no-height",
    ),
    pytest.param("the URL is null", [video("u", 720, url=None)], None, id="url-null"),
    pytest.param("the URL is empty", [video("u", 720, url="")], None, id="url-empty"),
    pytest.param(
        "a video-only format above the cap leaves only the muxed one",
        [video("v1080", 1080), muxed("m360", 360)], "m360", id="tall-video-only-above-cap",
    ),
    pytest.param(
        "a playlist with no formats", [], None, id="playlist-no-formats",
    ),
]


@pytest.mark.parametrize(("_label", "formats", "expected"), PICKS)
def test_spec_addon_193_i4_pick_format_returns_the_ranked_rendition(
    _label, formats, expected,
):
    chosen = pick_format(_description(formats))

    assert (chosen or {}).get("format_id") == expected


def test_spec_addon_193_i4_pick_format_without_a_formats_key_returns_none():
    assert pick_format({"id": "x", "title": "fixture"}) is None


def test_spec_addon_193_i4_pick_format_skips_entries_that_are_not_objects():
    description = _description(["junk", None, video("v720", 720)])

    assert pick_format(description)["format_id"] == "v720"


# ---------------------------------------------------------------------------
# I5 - the same descriptions through yt-dlp's own format selector
# ---------------------------------------------------------------------------

def _has_video_only(formats: list) -> bool:
    """A video-only format as the spec's Terms define one."""
    return any(
        isinstance(fmt, dict)
        and fmt.get("protocol") == "https"
        and fmt.get("acodec") == "none"
        and fmt.get("vcodec") not in (None, "none")
        and isinstance(fmt.get("url"), str) and fmt["url"]
        and isinstance(fmt.get("height"), int)
        and not isinstance(fmt.get("height"), bool)
        and 0 < fmt["height"] <= 720
        for fmt in formats
    )


def _ytdlp_picks(formats: list) -> list[dict]:
    """What yt-dlp's real selector picks from this description and DOWNLOAD_FORMAT."""
    from yt_dlp import YoutubeDL

    selector = YoutubeDL({"quiet": True}).build_format_selector(DOWNLOAD_FORMAT)
    context = {
        "formats": formats,
        "has_merged_format": True,
        "incomplete_formats": False,
        "has_audio": True,
        "duration": 60,
    }
    return list(selector(context))


SELECTIONS = [
    pytest.param(
        "only muxed", [muxed("m720", 720)], True, id="only-muxed",
    ),
    pytest.param(
        "a muxed format with a null audio codec",
        [_format("m720", 720, acodec=None)], True, id="muxed-acodec-null",
    ),
    pytest.param(
        "both kinds", [video("v480", 480), muxed("m720", 720)], True,
        id="both-kinds",
    ),
    pytest.param(
        "both kinds with the muxed one taller",
        [video("v360", 360), muxed("m720", 720)], True, id="muxed-taller",
    ),
    pytest.param(
        "two video-only formats", [video("v480", 480), video("v720", 720)], True,
        id="two-video-only",
    ),
    pytest.param(
        "a video-only format above the cap beside a muxed one",
        [video("v1080", 1080), muxed("m360", 360)], True, id="above-cap-beside-muxed",
    ),
    pytest.param(
        "progressive formats with both codecs unknown",
        [_format("p480", 480, vcodec=None, acodec=None), _format("p720", 720, vcodec=None, acodec=None)],
        True, id="unknown-codecs-progressive",
    ),
    pytest.param(
        "only audio", [audio("a128")], False, id="only-audio",
    ),
    pytest.param(
        "only HLS", [video("hls", 720, protocol="m3u8_native")], False,
        id="only-hls",
    ),
    pytest.param(
        "no video codec", [video("vc", 720, vcodec="none")], False,
        id="no-video-codec",
    ),
    pytest.param(
        "height above the cap", [video("v721", 721)], False, id="height-721",
    ),
    pytest.param(
        "a playlist with no formats", [], False, id="playlist-no-formats",
    ),
]


@pytest.mark.parametrize(("_label", "formats", "picked"), SELECTIONS)
def test_spec_addon_193_i5_the_download_selector_agrees_with_pick_format(
    _label, formats, picked,
):
    chosen = pick_format(_description(formats))
    selected = _ytdlp_picks(formats)

    if not picked:
        assert chosen is None
        assert selected == []
        return

    assert chosen is not None
    assert selected, "DOWNLOAD_FORMAT selected nothing for a description pick_format accepts"
    if _has_video_only(formats):
        assert selected[0].get("acodec") == "none", "a video-only format was available"
    else:
        assert selected[0].get("acodec") != "none"
