"""SPEC-ADDON-114 I11: every frame sent with a question is introduced by its own label.

The label ``[n] <time>`` comes right before its image, for a video with text
and for a text-less one, and the prompt text pairs no frame number with a time.
Frame bytes name their millisecond key, so each label is checked against the
image that follows it. The passages hold no digit, so every clock stamp in a
prompt belongs to a frame or to the position.
"""

from __future__ import annotations

import pytest

from tests.file_question_doubles import chunk, numbered_in_order
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    addon,
    ask,
    by_kind,
    call,
    db,
    llm,
    vision_client,
)
from tests.frame_label_layout import assert_labelled
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.video_frames_doubles import FFmpeg, ms, scenes, seed_text_less_video

VIDEO = "vid000000116"
QUESTION = "QUESTION-MARKER what does it show"
PASSAGE = chunk("SPOKEN-PASSAGE-ALPHA", 40)
PASSAGE_TWO = chunk("SPOKEN-PASSAGE-BRAVO", 40)
EVEN_80 = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]
EVEN_70_AROUND_33 = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0]
PROVIDERS = ["openai_compatible", "ollama"]
KINDS = ["text-video", "text-less-video"]


def _video(db, tmp_path, kind: str, *, duration, passages=(PASSAGE,)) -> int:
    """Seed the video and return the number its first frame takes."""
    seed_text_less_video(
        db, VIDEO, path=tmp_path / f"{VIDEO}.mp4", duration=duration,
    )
    if kind == "text-less-video":
        return 1
    for index, text in enumerate(passages):
        db.on(VIDEO).transcript(text, start=float(2 * index))
    return len(passages) + 1


class TestEachFrameCarriesItsLabel:
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_a_text_video_labels_its_frames_after_the_passages(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I11 I3 I1 §1.5: ``[P+1] m:ss`` / ``h:mm:ss`` right before each image."""
        first = _video(
            db, tmp_path, "text-video", duration=None, passages=(PASSAGE, PASSAGE_TWO),
        )
        scenes(monkeypatch, [10.0, 75.0, 3725.0], duration=None)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm, provider=provider)

        response = await ask(call, VIDEO, question=QUESTION)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert len(wire.image_bodies) == 1
        layout = assert_labelled(
            wire, wire.image_bodies[0], first=first, times=[10.0, 75.0, 3725.0],
        )
        assert [label for label, _ in layout.frames] == ["[3] 0:10", "[4] 1:15", "[5] 1:02:05"]
        assert QUESTION in layout.prompt
        assert numbered_in_order(layout.prompt, [PASSAGE, PASSAGE_TWO])

    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_a_text_less_video_gets_the_same_layout_from_one(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I11 §1.5 §7 (108 I4 revised): ``P = 0``, labels from ``[1]``."""
        first = _video(db, tmp_path, "text-less-video", duration=80.0)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm, provider=provider)

        response = await ask(call, VIDEO, question=QUESTION)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert len(wire.image_bodies) == 1
        layout = assert_labelled(wire, wire.image_bodies[0], first=first, times=EVEN_80)
        assert QUESTION in layout.prompt

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_a_kept_position_frame_is_labelled_at_its_time_and_the_position_stated(
        self, provider, kind, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I11 I5 (108 I5 revised): the position sentence is not a label line."""
        first = _video(db, tmp_path, kind, duration=70.0)
        scenes(monkeypatch, [], duration=70.0)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm, provider=provider)

        await ask(call, VIDEO, question=QUESTION, position=33.0)

        assert len(wire.image_bodies) == 1
        layout = assert_labelled(
            wire, wire.image_bodies[0], first=first,
            times=sorted([33.0, *EVEN_70_AROUND_33]),
        )
        assert "0:33" in layout.prompt

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_a_failed_position_frame_has_no_label_and_the_position_is_still_stated(
        self, provider, kind, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I11 I5 (108 I5 revised): no frame is labelled at the position's time."""
        first = _video(db, tmp_path, kind, duration=70.0)
        scenes(monkeypatch, [], duration=70.0)
        FFmpeg(fail={ms(33.0)}).install(monkeypatch)
        _, wire = vision_client(llm, provider=provider)

        await ask(call, VIDEO, question=QUESTION, position=33.0)

        assert len(wire.image_bodies) == 1
        layout = assert_labelled(
            wire, wire.image_bodies[0], first=first, times=EVEN_70_AROUND_33,
        )
        assert "0:33" in layout.prompt

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_failed_frames_leave_no_label_and_the_kept_ones_are_numbered_in_turn(
        self, provider, kind, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I11 I3: labels follow the frames sent, not the frames chosen."""
        first = _video(db, tmp_path, kind, duration=80.0)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg(fail={ms(15.0)}, timeout={ms(55.0)}).install(monkeypatch)
        _, wire = vision_client(llm, provider=provider)

        await ask(call, VIDEO, question=QUESTION)

        assert len(wire.image_bodies) == 1
        assert_labelled(
            wire, wire.image_bodies[0], first=first,
            times=[5.0, 25.0, 35.0, 45.0, 65.0, 75.0],
        )

    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_with_history_the_labels_follow_this_questions_prompt(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I11 §1.5: earlier turns come before; the frames follow this prompt."""
        first = _video(db, tmp_path, "text-video", duration=None)
        scenes(monkeypatch, [10.0, 20.0], duration=None)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm, provider=provider)

        response = await call(
            "POST", f"/files/{VIDEO}/ask",
            json={
                "question": QUESTION,
                "history": [{"question": "EARLIER-QUESTION", "answer": "EARLIER-ANSWER"}],
            },
        )

        assert by_kind(response)["meta"]["image"] == "attached"
        assert len(wire.image_bodies) == 1
        body = wire.image_bodies[0]
        assert "EARLIER-QUESTION" in wire.text_of(body)
        layout = assert_labelled(wire, body, first=first, times=[10.0, 20.0])
        assert QUESTION in layout.prompt
        assert numbered_in_order(layout.prompt, [PASSAGE])


class TestTheSystemPrompt:
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_it_calls_the_frames_the_numbered_frames_that_follow_the_question(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 §1.5 I10: "follow the question", not "attached to this message"."""
        _video(db, tmp_path, "text-video", duration=80.0)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm, provider=provider)

        await ask(call, VIDEO, question=QUESTION)

        assert len(wire.image_bodies) == 1
        system = " ".join(wire.system_of(wire.image_bodies[0]).lower().split())
        assert "numbered frames that follow the question" in system
        assert "attached to this message" not in system
