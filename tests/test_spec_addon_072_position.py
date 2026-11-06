"""SPEC-ADDON-072: the player's position in the file question.

The position reaches the prompt only for the kinds a viewer plays, and the
range check applies to every kind before a model is called.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.file_question_doubles import chunk
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    addon,
    ask,
    by_kind,
    call,
    db,
    llm,
    seed_audio,
    seed_document,
    seed_picture,
    seed_video,
    vision_client,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture

VIDEO = "vid000000021"
AUDIO = "aud000000021"
PICTURE = "pic000000021"
DOCUMENT = "pdf000000021"
BOOK = "epub000000021"
NOTE = "note000000021"


def _file(kind: str, db, tmp_path: Path) -> str:
    text = chunk(f"{kind} body", 30)
    if kind == "video":
        seed_video(
            db, VIDEO, text=text, path=tmp_path / "video.mp4", duration=3600.0,
        )
        return VIDEO
    if kind == "audio":
        seed_audio(db, AUDIO, text=text, path=tmp_path / "audio.mp3")
        return AUDIO
    if kind == "picture":
        seed_picture(db, PICTURE, text=text, path=tmp_path / "picture.png")
        return PICTURE
    if kind == "pdf":
        seed_document(
            db, DOCUMENT, text=text, path=tmp_path / "doc.pdf",
            mime="application/pdf",
        )
        return DOCUMENT
    if kind == "epub":
        seed_document(
            db, BOOK, text=text, path=tmp_path / "book.epub",
            mime="application/epub+zip",
        )
        return BOOK
    seed_document(
        db, NOTE, text=text, path=tmp_path / "note.md", mime="text/markdown",
    )
    return NOTE


class TestThePositionInThePrompt:
    @pytest.mark.parametrize(
        ("kind", "position", "stamp"),
        [
            ("video", 1705.0, "28:25"),
            ("video", 5.0, "0:05"),
            ("video", 3661.0, "1:01:01"),
            ("audio", 65.0, "1:05"),
            ("audio", 3725.0, "1:02:05"),
        ],
        ids=["video-minutes", "video-seconds", "video-over-an-hour", "audio", "audio-hours"],
    )
    async def test_a_position_is_written_in_the_viewers_clock(
        self, kind, position, stamp, db, addon, llm, call, tmp_path,
    ):
        """SPEC-ADDON-072 I5: h:mm:ss for a long media file, m:ss under an hour."""
        file_id = _file(kind, db, tmp_path)
        _, wire = vision_client(llm)

        response = await ask(call, file_id, position=position)

        assert response.status_code == 200
        assert stamp in wire.prompts[-1]

    @pytest.mark.parametrize("kind", ["pdf", "epub", "note", "picture"])
    async def test_a_position_is_absent_from_a_kind_that_ignores_it(
        self, kind, db, addon, llm, call, tmp_path,
    ):
        """SPEC-ADDON-072 I5/I7: a valid position is accepted and ignored."""
        file_id = _file(kind, db, tmp_path)
        _, wire = vision_client(llm)

        response = await ask(call, file_id, position=65.0)

        assert response.status_code == 200
        assert "1:05" not in wire.prompts[-1]
        assert by_kind(response)["meta"]["image"] == (
            "attached" if kind == "picture" else "none"
        )


class TestThePositionRange:
    @pytest.mark.parametrize(
        "kind", ["video", "audio", "picture", "pdf", "epub", "note"],
    )
    @pytest.mark.parametrize(
        ("raw", "name"),
        [
            (b"-1.0", "negative"),
            (b"NaN", "nan"),
            (b"Infinity", "infinity"),
            (b"-Infinity", "negative-infinity"),
            (b"1000001.0", "over-a-million"),
        ],
        ids=["negative", "nan", "infinity", "negative-infinity", "over-a-million"],
    )
    async def test_a_position_outside_the_range_is_422_before_a_model(
        self, kind, raw, name, db, addon, llm, call, tmp_path,
    ):
        """SPEC-ADDON-072 I7: the range is every kind's, and nothing is called."""
        file_id = _file(kind, db, tmp_path)
        _, wire = vision_client(llm)

        body = b'{"question":"what is this","position_seconds":' + raw + b"}"
        response = await call("POST", f"/files/{file_id}/ask", content=body)

        assert response.status_code == 422
        assert wire.requests == []
        assert llm.asked == []

    @pytest.mark.parametrize(
        ("position", "stamp"),
        [
            (0.0, "0:00"),
            (1.0, "0:01"),
            (999_999.0, "277:46:39"),
            (1_000_000.0, "277:46:40"),
        ],
        ids=["zero", "one", "just-under", "upper-bound"],
    )
    async def test_a_position_inside_the_range_reaches_the_prompt(
        self, position, stamp, db, addon, llm, call, tmp_path,
    ):
        """SPEC-ADDON-072 I5/I7: the bounds themselves are valid and written."""
        seed_video(
            db, VIDEO, text=chunk("video", 30),
            path=tmp_path / "video.mp4", duration=1_000_001.0,
        )
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO, position=position)

        assert response.status_code == 200
        assert stamp in wire.prompts[-1]
