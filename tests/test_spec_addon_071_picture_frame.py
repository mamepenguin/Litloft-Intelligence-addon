"""SPEC-ADDON-071: a picture's pixels and the playing video's frame.

The image is read from what the provider would have received, because the
addon's real client carries it out through its own transport.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

import app.config as config
import app.frame_cache as frame_cache
from tests.file_question_doubles import chunk
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    DRIVE,
    OTHER,
    PICTURE_BYTES,
    VISION_MODEL,
    addon,
    ask,
    by_kind,
    call,
    db,
    events,
    ffmpeg_fails,
    ffmpeg_times_out,
    ffmpeg_writes,
    llm,
    openai_config,
    seed_picture,
    seed_video,
    vision_client,
    write_large_frame,
)
from tests.search_endpoint_db import patch_everywhere
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.video_frames_doubles import FFmpeg, decoded, ms, sent_keys


PICTURE = "pic000000001"
PICTURE_TWO = "pic000000002"
VIDEO = "vid000000011"
TEXT = chunk("receipt total is 42", 40)

# tests/test_file_question.py::test_rag_off_answers_as_ask_does pins this status.
RAG_OFF_STATUS = 400


def _decoded_image(wire) -> bytes:
    payload = wire.images[0]
    if payload.startswith("data:"):
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


def _attach_picture(db, file_id=PICTURE, *, text=TEXT, data=PICTURE_BYTES, **kwargs):
    base = Path(config.settings.intelligence_data_dir).parent / "pictures"
    path = kwargs.pop("path", base / f"{file_id}.png")
    return seed_picture(
        db, file_id, text=text, path=path, data=data, **kwargs,
    )


class TestTheRequestedPicture:
    async def test_a_picture_with_text_reaches_the_vision_model_with_its_passage(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-071 I1/I2: one image, the requested picture's, on the vision model."""
        _attach_picture(db)
        _, wire = vision_client(llm)

        response = await ask(call, PICTURE)

        assert response.status_code == 200
        events = by_kind(response)
        assert events["meta"]["image"] == "attached"
        assert "".join(
            data["delta"] for kind, data in response_events(response)
            if kind == "answer_chunk"
        ) == "VISION ANSWER"
        assert len(wire.image_bodies) == 1
        assert wire.image_bodies[0]["model"] == VISION_MODEL
        assert len(wire.images) == 1
        assert _decoded_image(wire) == PICTURE_BYTES
        assert TEXT in wire.text_of(wire.image_bodies[0])
        prompt = wire.text_of(wire.image_bodies[0]).lower()
        assert "image" in prompt
        assert "exact" in prompt
        assert "passage" in prompt
        assert llm.asked == [(DRIVE, "rag", None)]

    async def test_only_the_requested_pictures_bytes_are_sent(self, db, addon, llm, call):
        """SPEC-ADDON-071 I1: a sibling picture in the same drive is not material."""
        _attach_picture(db, PICTURE, data=b"FIRST-PICTURE")
        _attach_picture(db, PICTURE_TWO, data=b"SECOND-PICTURE")
        _, wire = vision_client(llm)

        await ask(call, PICTURE)

        assert _decoded_image(wire) == b"FIRST-PICTURE"

    @pytest.mark.parametrize("case", ["another drive", "inactive"])
    async def test_a_picture_outside_the_request_drive_is_404(
        self, case, db, addon, llm, call,
    ):
        """SPEC-ADDON-071 I1: no other drive's picture, and no inactive one."""
        if case == "another drive":
            _attach_picture(db, drive=OTHER)
        else:
            _attach_picture(db, active=False)
        _attach_picture(db, "pic000000090", data=b"ASKABLE-PICTURE")
        _, wire = vision_client(llm)

        response = await ask(call, PICTURE)
        control = await ask(call, "pic000000090")

        assert response.status_code == 404
        assert by_kind(control)["meta"]["image"] == "attached"
        assert wire.image_bodies

    async def test_rag_off_answers_as_the_route_does_and_reads_no_picture(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-071 I1: rag off closes the route before the picture is read."""
        _attach_picture(db)
        addon(rag=False)
        _, wire = vision_client(llm)

        response = await ask(call, PICTURE)

        assert response.status_code == RAG_OFF_STATUS
        assert wire.requests == []

        addon(rag=True)
        served = await ask(call, PICTURE)
        assert by_kind(served)["meta"]["image"] == "attached"


class TestThePictureImageStep:
    async def test_the_picture_is_reencoded_through_the_description_helpers(
        self, db, addon, llm, call, monkeypatch,
    ):
        """SPEC-ADDON-071 I1: the 1024-px re-encode is the worker's own helper."""
        _attach_picture(db)
        seen: list[tuple] = []

        def _load(file_id):
            seen.append(("load", file_id))
            return b"RAW-PICTURE", "image/png"

        def _preprocess(raw, mime):
            seen.append(("preprocess", raw, mime))
            return b"REENCODED-PICTURE", "image/jpeg"

        patch_everywhere(monkeypatch, "_load_image_bytes", _load)
        patch_everywhere(monkeypatch, "_preprocess_image", _preprocess)
        _, wire = vision_client(llm)

        response = await ask(call, PICTURE)

        assert response.status_code == 200
        assert ("load", PICTURE) in seen
        assert ("preprocess", b"RAW-PICTURE", "image/png") in seen
        assert _decoded_image(wire) == b"REENCODED-PICTURE"
        assert wire.images[0].startswith("data:image/jpeg;base64,")

    async def test_an_unreadable_picture_is_unavailable_and_the_text_answer_stands(
        self, db, addon, llm, call, tmp_path,
    ):
        """SPEC-ADDON-071 I3: a missing picture file reports unavailable, not an error."""
        missing = tmp_path / "pictures" / "gone.png"
        db.file(
            PICTURE, mime="image/png", file_path=str(missing), file_size=10,
        )
        db.document(TEXT, chunk_index=0)
        _, wire = vision_client(llm)

        response = await ask(call, PICTURE)

        assert response.status_code == 200
        assert by_kind(response)["meta"]["image"] == "unavailable"
        assert answer_of(response) == "TEXT ONLY ANSWER"
        assert wire.image_bodies == []

    async def test_a_picture_over_the_size_limit_is_unavailable(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-071 I3: over 50 MB is not loaded or sent."""
        _attach_picture(db, file_size=50 * 1024 * 1024 + 1)
        _, wire = vision_client(llm)

        response = await ask(call, PICTURE)

        assert response.status_code == 200
        assert by_kind(response)["meta"]["image"] == "unavailable"
        assert wire.image_bodies == []

    async def test_an_undecodable_picture_is_unavailable(
        self, db, addon, llm, call, monkeypatch,
    ):
        """SPEC-ADDON-071 I3: a decode failure is reported as an unavailable image."""
        _attach_picture(db)
        patch_everywhere(monkeypatch, "_preprocess_image", lambda raw, mime: None)
        _, wire = vision_client(llm)

        response = await ask(call, PICTURE)

        assert by_kind(response)["meta"]["image"] == "unavailable"
        assert wire.image_bodies == []

class TestTheVideoFrame:
    """SPEC-ADDON-114: a video with a text chunk gets a set of frames, its position frame among them."""

    async def test_the_position_frame_is_read_from_its_millisecond_cache_key(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I1/I5 (071 I1 revised): the frame at the position is the cached one, no other time stands in."""
        seed_video(
            db, VIDEO, text=chunk("at that moment", 30),
            path=tmp_path / "video.mp4", duration=60.0,
        )
        write_large_frame(VIDEO, 12.345, b"FRAME-AT-12345")
        write_large_frame(VIDEO, 12.000, b"OTHER-FRAME")
        ffmpeg = FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO, position=12.345)

        assert by_kind(response)["meta"]["image"] == "attached"
        images = [decoded(payload) for payload in wire.images]
        assert b"FRAME-AT-12345" in images
        assert b"OTHER-FRAME" not in images
        assert wire.images[0].startswith("data:image/webp;base64,")
        assert ms(12.345) not in ffmpeg.starts

    async def test_a_video_without_a_position_sends_the_scene_frames(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I5 §1.3/§7 (071 I3 revised): no position frame, only the scene set."""
        seed_video(
            db, VIDEO, text=chunk("plain video", 30),
            path=tmp_path / "video.mp4", duration=60.0,
        )
        ffmpeg = FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert len(wire.images) == 8
        assert ms(0.0) not in sent_keys(wire)
        assert len(ffmpeg.starts) == 8

    async def test_a_blind_profile_sends_no_image_for_a_video_without_a_position(
        self, db, addon, llm, call, tmp_path,
    ):
        """SPEC-ADDON-114 I8 §1.7 (071 I3 kept): a profile that cannot see keeps today's text path."""
        seed_video(
            db, VIDEO, text=chunk("plain video", 30),
            path=tmp_path / "video.mp4", duration=60.0,
        )
        _, wire = vision_client(llm, config=openai_config(vision_model=""))

        response = await ask(call, VIDEO)

        assert by_kind(response)["meta"] == {"partial": False}
        assert wire.image_bodies == []
        assert answer_of(response) == "TEXT ONLY ANSWER"

    async def test_a_frame_extraction_failure_is_unavailable_with_the_text_answer(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I4 (071 I3/I7 revised): no frame is prepared, so the passages answer."""
        seed_video(
            db, VIDEO, text=chunk("video", 30),
            path=tmp_path / "video.mp4", duration=None,
        )
        monkeypatch.setattr(frame_cache.subprocess, "run", ffmpeg_fails())
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO, position=5.0)

        assert by_kind(response)["meta"]["image"] == "unavailable"
        assert by_kind(response)["done"].get("error") is None
        assert wire.image_bodies == []
        assert answer_of(response) == "TEXT ONLY ANSWER"

    async def test_a_frame_extraction_timeout_is_unavailable(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I4 (071 I3/I7 revised): the 15 s bound leaves the frame out, no substitute."""
        seed_video(
            db, VIDEO, text=chunk("video", 30),
            path=tmp_path / "video.mp4", duration=None,
        )
        monkeypatch.setattr(frame_cache.subprocess, "run", ffmpeg_times_out())
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO, position=5.0)

        assert by_kind(response)["meta"]["image"] == "unavailable"
        assert wire.image_bodies == []
        assert answer_of(response) == "TEXT ONLY ANSWER"

    async def test_a_position_past_the_indexed_duration_reads_no_frame_at_that_time(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I5 (071 I7 revised): no frame past the duration, and none substituted for it."""
        seed_video(
            db, VIDEO, text=chunk("video", 30),
            path=tmp_path / "video.mp4", duration=60.0,
        )
        ffmpeg = FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO, position=61.0)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert sent_keys(wire)
        assert ms(61.0) not in sent_keys(wire)
        assert ms(61.0) not in ffmpeg.starts
        assert "1:01" in wire.prompts[-1]

    async def test_without_a_stored_duration_the_extraction_decides(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I5 (071 I3): no duration means try, and ffmpeg's result decides."""
        seed_video(
            db, VIDEO, text=chunk("video", 30),
            path=tmp_path / "video.mp4", duration=None,
        )
        calls: list = []
        monkeypatch.setattr(
            frame_cache.subprocess, "run", ffmpeg_writes(b"FRESH-FRAME", calls),
        )
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO, position=5.0)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert _decoded_image(wire) == b"FRESH-FRAME"
        assert len(calls) == 1
        assert frame_cache.frame_cache_path(VIDEO, 5.0, "large").exists()


class TestTheImageOnlyPicture:
    async def test_a_picture_with_no_text_gets_the_image_only_wording(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-071 I5: no citation instruction and an empty citation list."""
        _attach_picture(db, text=None)
        _, wire = vision_client(llm)

        response = await ask(call, PICTURE)

        assert response.status_code == 200
        frames = by_kind(response)
        assert frames["meta"]["image"] == "attached"
        assert frames["citations"] == []
        prompt = wire.text_of(wire.image_bodies[0]).lower()
        assert "image" in prompt
        assert "cite" not in prompt
        assert "passages from the file" not in prompt


# ---------------------------------------------------------------------------
# Helpers local to this module
# ---------------------------------------------------------------------------


def response_events(response):
    return events(response)


def answer_of(response) -> str:
    return "".join(
        data["delta"] for kind, data in events(response) if kind == "answer_chunk"
    )
