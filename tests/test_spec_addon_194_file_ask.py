"""SPEC-ADDON-194 I17: a one-file Ask about a web reference.

A `web` `.loft` that is a frame source takes the same frames path as a YouTube
one, reading its `large` frames live from the stream; a capped one is answered
as a `.loft` with no frame source.
"""

from __future__ import annotations

import json
import logging

import pytest

import app.database as database
import app.frame_cache as frame_cache
from app.models import IndexedFile
from tests.file_question_doubles import chunk
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    DRIVE,
    VISION_MODEL,
    Wire,
    addon,
    ask,
    by_kind,
    call,
    db,
    events,
    llm,
    openai_config,
    profiles,
    vision_client,
)
from tests.video_frames_doubles import scenes
from tests.vec_partition_db import search_index  # noqa: F401 - fixture
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    LOFT_MIME,
    describe,
    needs_ffmpeg,
    youtube,
)

WEB_PROVIDER = "example-video"
WEB_URL = "https://videos.example-video.test/watch/abcdef123456?token=zzzzzzz9"
LOFT = "wfloft000001"
CAPPED = "wfloft000002"
FRESH = "wfloft000003"
PASSAGE = chunk("SPOKEN-PASSAGE-ALPHA", 40)
QUESTION = "QUESTION-MARKER what does it show"


def _loft(
    db, tmp_path, file_id: str, *, source: str | None = "web", failures: int = 0,
    duration: float | None = 60.0, passage: str | None = None,
):
    path = tmp_path / f"{file_id}.loft"
    path.write_text(json.dumps({"provider": WEB_PROVIDER, "url": WEB_URL}))
    db.file(file_id, mime=LOFT_MIME, file_path=str(path), duration=duration)
    with database.get_search_db() as session:
        session.query(IndexedFile).filter_by(file_id=file_id).update({
            "file_type": "video",
            "loft_source": source,
            "loft_fetch_failures": failures,
        })
    if passage is not None:
        db.on(file_id).transcript(passage, start=0.0)


async def _available(call, file_id) -> bool:
    response = await call("GET", f"/files/{file_id}/ask")
    assert response.status_code == 200, response.text
    return response.json()["available"]


def _answer(response) -> str:
    return "".join(
        data.get("delta", "") for kind, data in events(response) if kind == "answer_chunk"
    )


@needs_ffmpeg
async def test_spec_addon_194_i17_a_web_loft_with_text_sends_frames_read_live(
    db, addon, llm, call, tmp_path, youtube, monkeypatch,
):
    _loft(db, tmp_path, LOFT, passage=PASSAGE)
    scenes(monkeypatch, [10.0, 20.0], duration=None)
    youtube.set_ytdlp(describe=describe([youtube.format("f720", 720)]))
    _, wire = vision_client(llm)

    response = await ask(call, LOFT, position=12.0)

    assert by_kind(response)["meta"]["image"] == "attached"
    assert len(wire.image_bodies) == 1
    assert PASSAGE in wire.text_of(wire.image_bodies[0])
    directory = frame_cache.frame_cache_dir(LOFT)
    assert sorted(p.name for p in directory.iterdir()) == [
        "10000.large.webp", "12000.large.webp", "20000.large.webp",
    ]
    assert youtube.lookups()
    assert not any(
        str(tmp_path / f"{LOFT}.loft") in " ".join(e["argv"])
        for e in youtube.ffmpeg_events()
    )


@needs_ffmpeg
async def test_spec_addon_194_i17_a_capped_web_loft_attempts_no_image(
    db, addon, llm, call, tmp_path, youtube, monkeypatch, caplog,
):
    _loft(db, tmp_path, CAPPED, failures=3, passage=PASSAGE)
    scenes(monkeypatch, [10.0, 20.0], duration=None)
    youtube.set_ytdlp(describe=describe([youtube.format("f720", 720)]))
    _, wire = vision_client(llm)

    with caplog.at_level(logging.WARNING):
        response = await ask(call, CAPPED, position=12.0)

    assert by_kind(response)["meta"].get("image", "none") == "none"
    assert wire.image_bodies == []
    assert wire.text_bodies
    assert PASSAGE in wire.text_of(wire.text_bodies[0])
    assert youtube.lookups() == []
    directory = frame_cache.frame_cache_dir(CAPPED)
    assert not directory.exists() or list(directory.iterdir()) == []
    assert [
        r.getMessage() for r in caplog.records
        if "file question image failed" in r.getMessage()
    ] == []


@needs_ffmpeg
async def test_spec_addon_194_i17_a_text_less_web_loft_is_answered_from_live_frames(
    db, addon, llm, call, tmp_path, youtube, monkeypatch,
):
    _loft(db, tmp_path, FRESH, passage=None)
    scenes(monkeypatch, [], duration=60.0)
    youtube.set_ytdlp(describe=describe([youtube.format("f720", 720)]))
    addon(profiles=profiles())
    _, wire = vision_client(llm)

    assert await _available(call, FRESH) is True
    response = await ask(call, FRESH, position=12.0)

    assert by_kind(response)["meta"]["image"] == "attached"
    assert len(wire.requests) == 1
    cached = frame_cache.frame_cache_path(FRESH, 12.0, "large")
    assert cached.exists()
    assert youtube.lookups()


@needs_ffmpeg
async def test_spec_addon_194_i17_a_capped_text_less_web_loft_is_unavailable(
    db, addon, llm, call, tmp_path, youtube, monkeypatch,
):
    _loft(db, tmp_path, CAPPED, failures=3, passage=None)
    scenes(monkeypatch, [], duration=60.0)
    youtube.set_ytdlp(describe=describe([youtube.format("f720", 720)]))
    addon(profiles=profiles())
    _, wire = vision_client(llm)

    assert await _available(call, CAPPED) is False
    response = await ask(call, CAPPED)
    assert response.status_code == 404
    assert wire.requests == []
    assert youtube.lookups() == []


@needs_ffmpeg
async def test_spec_addon_194_i17_a_web_loft_with_text_and_a_blind_profile_reads_nothing(
    db, addon, llm, call, tmp_path, youtube, monkeypatch,
):
    _loft(db, tmp_path, LOFT, passage=PASSAGE)
    scenes(monkeypatch, [10.0, 20.0], duration=None)
    youtube.set_ytdlp(describe=describe([youtube.format("f720", 720)]))
    _, wire = vision_client(llm, config=openai_config(vision_model=""))

    response = await ask(call, LOFT, position=12.0)

    assert "image" not in by_kind(response)["meta"]
    assert wire.image_bodies == []
    assert wire.text_bodies
    assert youtube.lookups() == []
    assert _answer(response) == Wire.text_answer
