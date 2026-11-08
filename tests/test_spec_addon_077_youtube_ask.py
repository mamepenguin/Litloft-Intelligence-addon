"""SPEC-ADDON-077: Ask attaches the frame at the player position for a YouTube ``.loft``."""

from __future__ import annotations

import base64
import json
import time

import pytest

import app.config as config
import app.database as database
from app.frame_cache import frame_cache_path
from app.models import IndexedFile
from tests.file_question_doubles import chunk
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    DRIVE,
    addon,
    ask,
    by_kind,
    call,
    db,
    events,
    llm,
    vision_client,
    write_large_frame,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    HEADER_TOKEN,
    LOFT_MIME,
    SIGNED_TOKEN,
    YOUTUBE_URL,
    describe,
    everything_logged,
    needs_ffmpeg,
    webp_size,
    youtube,
)

pytestmark = needs_ffmpeg

VIDEO = "yta00000077"


def _loft(db, tmp_path, *, body=None, duration=60.0, file_id=VIDEO):
    path = tmp_path / f"{file_id}.loft"
    path.write_text(json.dumps(
        body if body is not None else {"provider": "youtube", "url": YOUTUBE_URL}
    ))
    db.file(file_id, mime=LOFT_MIME, file_path=str(path), duration=duration)
    with database.get_search_db() as s:
        s.query(IndexedFile).filter_by(file_id=file_id).update({"file_type": "video"})
    db.transcript(chunk("on screen the speaker shows a slide", 30), start=0.0)


def _serve(youtube):
    fmt = youtube.format("f720", 720)
    youtube.set_ytdlp(describe=describe([fmt]))
    return fmt


def _decoded(wire) -> bytes:
    payload = wire.images[0]
    if payload.startswith("data:"):
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


class TestTheFrameAtThePosition:
    async def test_the_large_frame_of_the_stream_is_attached_once(
        self, db, addon, llm, call, tmp_path, youtube,
    ):
        """SPEC-ADDON-077 I6: exactly one ``large`` frame at the position, cached as ``large``."""
        _loft(db, tmp_path)
        _serve(youtube)
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO, position=12.345)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert len(wire.images) == 1
        assert wire.images[0].startswith("data:image/webp;base64,")
        cached = frame_cache_path(VIDEO, 12.345, "large")
        assert _decoded(wire) == cached.read_bytes()
        assert sorted(p.name for p in cached.parent.iterdir()) == ["12345.large.webp"]
        assert len(youtube.lookups()) == 1

    async def test_the_attached_frame_is_the_large_size_of_the_stream(
        self, db, addon, llm, call, tmp_path, youtube,
    ):
        """SPEC-ADDON-077 I6: the image is the ``large`` frame (1280 px), not a ``small`` one."""
        _loft(db, tmp_path)
        _serve(youtube)
        _, wire = vision_client(llm)

        await ask(call, VIDEO, position=12.0)

        assert webp_size(_decoded(wire), tmp_path) == (1280, 720)


class TestWhenTheFrameCannotBeRead:
    @pytest.mark.parametrize("how", ["lookup", "ffmpeg", "no-format"])
    async def test_any_frame_failure_is_unavailable_with_the_text_only_retry(
        self, how, db, addon, llm, call, tmp_path, youtube,
    ):
        """SPEC-ADDON-077 I6: ``unavailable`` and a text-only answer, never another frame."""
        _loft(db, tmp_path)
        if how == "lookup":
            youtube.set_ytdlp(fail="ERROR: [youtube] x: Video unavailable")
        elif how == "no-format":
            youtube.set_ytdlp(describe=describe([]))
        else:
            _serve(youtube)
            youtube.override("/v/f720", status=403)
        write_large_frame(VIDEO, 0.0, b"ANOTHER-TIME")
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO, position=5.0)

        assert by_kind(response)["meta"]["image"] == "unavailable"
        assert wire.image_bodies == []
        assert len(wire.text_bodies) == 1
        assert not frame_cache_path(VIDEO, 5.0, "large").exists()
        assert len(youtube.lookups()) == 1
        assert bool(youtube.fetched("/v/f720")) == (how == "ffmpeg")

    async def test_a_position_past_the_known_duration_reads_nothing(
        self, db, addon, llm, call, tmp_path, youtube,
    ):
        """SPEC-ADDON-077 I6: past the indexed duration is unavailable, with no lookup."""
        _loft(db, tmp_path, duration=60.0)
        _serve(youtube)
        _, wire = vision_client(llm)
        inside = await ask(call, VIDEO, position=59.0)
        assert by_kind(inside)["meta"]["image"] == "attached"
        lookups = len(youtube.lookups())

        response = await ask(call, VIDEO, position=61.0)

        assert by_kind(response)["meta"]["image"] == "unavailable"
        assert len(wire.image_bodies) == 1
        assert len(youtube.lookups()) == lookups

    async def test_a_lookup_slower_than_15_seconds_ends_as_unavailable(
        self, db, addon, llm, call, tmp_path, youtube,
    ):
        """SPEC-ADDON-077: the frame route's 15 s limit bounds the image step."""
        _loft(db, tmp_path)
        youtube.set_ytdlp(describe=describe([youtube.format("f720", 720)]), delay=40.0)
        _, wire = vision_client(llm)

        started = time.monotonic()
        response = await ask(call, VIDEO, position=5.0)
        elapsed = time.monotonic() - started

        assert by_kind(response)["meta"]["image"] == "unavailable"
        assert wire.image_bodies == []
        assert 13.0 < elapsed < 20.0


class TestWhatIsLogged:
    async def test_no_url_or_header_reaches_a_log_line(
        self, db, addon, llm, call, tmp_path, youtube, caplog,
    ):
        """SPEC-ADDON-077 I6/I5: nothing of the stream is logged."""
        _loft(db, tmp_path)
        fmt = _serve(youtube)
        _, wire = vision_client(llm)

        with caplog.at_level("DEBUG"):
            response = await ask(call, VIDEO, position=5.0)

        assert by_kind(response)["meta"]["image"] == "attached"
        logged = everything_logged(caplog)
        for secret in (YOUTUBE_URL, "dQw4w9WgXcQ", SIGNED_TOKEN, HEADER_TOKEN, fmt["url"]):
            assert secret not in logged, secret
