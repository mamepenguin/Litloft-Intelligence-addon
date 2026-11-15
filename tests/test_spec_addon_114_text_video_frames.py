"""SPEC-ADDON-114: a question about a video with text sends its frames with its passages.

The passages are transcript chunks whose text holds no digit, so every clock
stamp in a prompt belongs to a frame or to the position. Frame bytes name the
millisecond key they were cached under, so the images a provider received read
back as the times they were taken at, in the order they were sent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

import app.database as database
import app.frame_cache as frame_cache
from app.models import IndexedFile
from app.workers.video_visual_selection import (
    SelectedFrame,
    Selection,
    SelectionInputs,
)
from tests.file_question_doubles import chunk, mentioned, numbered_in_order
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
    ffmpeg_fails,
    llm,
    openai_config,
    statements,
    toward,
    vision_client,
)
from tests.search_endpoint_db import patch_everywhere
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.video_frames_doubles import (
    MARKER,
    FFmpeg,
    decoded,
    ms,
    scenes,
    seed_text_less_video,
    sent_keys,
    user_prompt,
)
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    LOFT_MIME,
    YOUTUBE_URL,
    describe,
    needs_ffmpeg,
    youtube,
)

VIDEO = "vid000000114"
SECOND = "vid000000115"
LOFT = "lft000000114"
OTHER_LOFT = "lft000000115"
QUESTION = "QUESTION-MARKER what does it show"
PASSAGE = chunk("SPOKEN-PASSAGE-ALPHA", 40)
PASSAGE_TWO = chunk("SPOKEN-PASSAGE-BRAVO", 40)
EVEN_80 = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]
EVEN_70_AROUND_33 = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0]
FRAME_ERRORS = ("frames_unusable", "frames_rejected", "image_unusable")


def _text_video(
    db, tmp_path, file_id=VIDEO, *, duration=80.0, passages=((PASSAGE, 0.0),),
):
    seed_text_less_video(
        db, file_id, path=tmp_path / f"{file_id}.mp4", duration=duration,
    )
    for text, start in passages:
        db.on(file_id).transcript(text, start=start)


def _loft(db, tmp_path, file_id, *, body, duration=60.0, passage=PASSAGE, raw=None):
    path = tmp_path / f"{file_id}.loft"
    path.write_text(raw if raw is not None else json.dumps(body))
    db.file(file_id, mime=LOFT_MIME, file_path=str(path), duration=duration)
    with database.get_search_db() as s:
        s.query(IndexedFile).filter_by(file_id=file_id).update({"file_type": "video"})
    db.on(file_id).transcript(passage, start=0.0)


def _answer(response) -> str:
    return "".join(
        data.get("delta", "") for kind, data in events(response) if kind == "answer_chunk"
    )


def _cited(response) -> list[tuple[int, dict | None]]:
    return sorted(
        (entry["n"], entry["position"])
        for entry in by_kind(response).get("citations", [])
    )


def _answering(text: str):
    def script(body):
        return 200, text
    return script


def _refusing_images(wire: Wire, status: int):
    def script(body):
        if wire.has_image(body):
            return status, ""
        return 200, wire.text_answer
    return script


def _listed_from(prompt: str, first: int, stamps: list[str]) -> bool:
    """True when ``[first]`` .. ``[first+m-1]`` each appear followed by their own stamp."""
    cursor = 0
    for number, stamp in enumerate(stamps, start=first):
        marker = prompt.find(f"[{number}]", cursor)
        if marker == -1:
            return False
        at = prompt.find(stamp, marker)
        following = prompt.find(f"[{number + 1}]", marker)
        if at == -1 or (following != -1 and at > following):
            return False
        cursor = at
    return True


def _scenes_per_file(monkeypatch, plan: dict[str, tuple[list[float], float | None]]):
    """Each file's Ask-list frame times and selection duration, keyed by file id."""
    current: list[str] = []

    def _load(file_id):
        current.append(file_id)
        return SelectionInputs(duration=plan[file_id][1])

    def _select(*_args, **_kwargs):
        times, duration = plan[current[-1]]
        items = tuple(
            SelectedFrame(
                section_start=0.0,
                section_end=duration if duration is not None else 0.0,
                frame_time=float(t),
                embedding_id=f"clip{i:03d}",
                embedding_type="clip",
                kind="visual",
                textual=False,
            )
            for i, t in enumerate(times)
        )
        return Selection(navigation=items, ask=items, candidate_count=len(items))

    patch_everywhere(monkeypatch, "load_selection_inputs", _load)
    patch_everywhere(monkeypatch, "select_scenes", _select)


def _frame_warnings(caplog, file_id: str) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and (file_id in record.getMessage() or "frames_" in record.getMessage())
    ]


# ---------------------------------------------------------------------------
# Normal flow: passages and frames in one request
# ---------------------------------------------------------------------------


class TestPassagesAndFramesGoTogether:
    @pytest.mark.parametrize("provider", ["openai_compatible", "ollama"])
    async def test_one_request_carries_the_passages_and_the_frame_set(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I1/I9 §1.3-1.6: eight frames, in order, to the resolved vision model."""
        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm, provider=provider)

        response = await ask(call, VIDEO, question=QUESTION)

        assert len(wire.requests) == 1
        body = wire.requests[0]
        assert body["model"] == VISION_MODEL
        assert sent_keys(wire) == [ms(t) for t in EVEN_80]
        assert PASSAGE in wire.text_of(body)
        assert llm.asked == [(DRIVE, "rag", None)]
        meta = by_kind(response)["meta"]
        assert meta["image"] == "attached"
        assert meta["partial"] is False
        assert _answer(response) == Wire.vision_answer
        assert "error" not in by_kind(response)["done"]
        if provider == "openai_compatible":
            user = [m for m in body["messages"] if m.get("role") == "user"][-1]
            kinds = [part["type"] for part in user["content"]]
            assert kinds == ["text"] + ["image_url"] * 8

    async def test_the_passages_are_numbered_first_and_the_frames_follow_with_their_times(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I3 §1.5: ``[1]..[P]`` passages, then ``[P+1]..[P+F]`` frames in time order."""
        _text_video(
            db, tmp_path, duration=None,
            passages=((PASSAGE, 0.0), (PASSAGE_TWO, 2.0)),
        )
        scenes(monkeypatch, [3725.9, 5.9, 75.9], duration=None)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        await ask(call, VIDEO, question=QUESTION)

        assert len(wire.image_bodies) == 1
        prompt = user_prompt(wire)
        assert QUESTION in prompt
        assert numbered_in_order(prompt, [PASSAGE, PASSAGE_TWO])
        assert prompt.index(PASSAGE_TWO) < prompt.index("[3]")
        assert _listed_from(prompt, 3, ["0:05", "1:15", "1:02:05"])
        assert "[6]" not in prompt
        assert sent_keys(wire) == [ms(5.9), ms(75.9), ms(3725.9)]

    async def test_the_system_prompt_lets_both_passages_and_frames_be_cited(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I10: passages and frames, exact characters from the passage."""
        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        await ask(call, VIDEO)

        assert len(wire.image_bodies) == 1
        system = wire.system_of(wire.image_bodies[0]).lower()
        assert "passage" in system
        assert "frame" in system
        assert "exact" in system
        assert "[n]" in system or "cite" in system

    async def test_with_history_the_numbers_still_start_at_this_messages_passages(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 §1.5: earlier turns are context; this message's passages and frames are numbered."""
        _text_video(db, tmp_path, duration=None)
        scenes(monkeypatch, [10.0, 20.0], duration=None)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm, script=_answering("See [1] and [3] and [4]."))

        response = await call(
            "POST", f"/files/{VIDEO}/ask",
            json={
                "question": "and then what",
                "history": [{"question": "EARLIER-QUESTION", "answer": "EARLIER-ANSWER [7]"}],
            },
        )

        assert by_kind(response)["meta"]["image"] == "attached"
        assert sent_keys(wire) == [ms(10.0), ms(20.0)]
        assert "EARLIER-QUESTION" in "\n".join(wire.prompts)
        prompt = user_prompt(wire)
        assert numbered_in_order(prompt, [PASSAGE])
        assert _listed_from(prompt, 2, ["0:10", "0:20"])
        assert _cited(response) == [(1, {"seconds": 0.0}), (3, {"seconds": 20.0})]

    async def test_a_cut_passage_list_reports_partial_with_the_frames_attached(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I9: ``partial`` is the passages' own cut, frames or not."""
        addon(budget=80)
        _text_video(
            db, tmp_path,
            passages=((chunk("FIRST-PASSAGE", 50), 0.0), (chunk("SECOND-PASSAGE", 50), 2.0)),
        )
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO)

        meta = by_kind(response)["meta"]
        assert meta["image"] == "attached"
        assert meta["partial"] is True
        assert len(wire.images) == 8


# ---------------------------------------------------------------------------
# The frame set
# ---------------------------------------------------------------------------


SELECTION = [
    pytest.param([], 80.0, None, EVEN_80, id="no-candidates-evenly-spaced"),
    pytest.param(
        [10.0 * (i + 1) for i in range(10)], 110.0, 55.0,
        [10.0, 30.0, 40.0, 55.0, 60.0, 70.0, 90.0, 100.0],
        id="position-first-plus-seven-thinned",
    ),
    pytest.param(
        [10.0, 20.0, 30.5, 31.0], 70.0, 30.0,
        [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 31.0, 35.0],
        id="within-a-second-of-the-position-removed",
    ),
    pytest.param([12.0, 47.0], None, None, [12.0, 47.0], id="unknown-duration-no-fill"),
    pytest.param([], 80.0, 0.0, EVEN_80, id="position-zero-is-no-position"),
]


class TestTheFrameSet:
    @pytest.mark.parametrize(("selected", "duration", "position", "expected"), SELECTION)
    async def test_the_frames_sent_are_the_frame_set_of_a_text_less_video(
        self, selected, duration, position, expected,
        db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I5 §7: the same selection as SPEC-ADDON-108, budget 8 or 7 plus the position."""
        _text_video(db, tmp_path, duration=duration)
        scenes(monkeypatch, selected, duration=duration)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO, position=position)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert len(wire.requests) == 1
        assert sent_keys(wire) == [ms(t) for t in expected]

    async def test_a_position_of_zero_sends_the_same_request_as_no_position(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I5 §1.3/§7: 0 is no position; no frame at 0 and no position stated."""
        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        await ask(call, VIDEO, question=QUESTION)
        await ask(call, VIDEO, question=QUESTION, position=0.0)

        assert len(wire.image_bodies) == 2
        assert wire.prompts[0] == wire.prompts[1]
        assert wire.system_prompts[0] == wire.system_prompts[1]
        assert sent_keys(wire) == [ms(t) for t in EVEN_80] * 2

    async def test_the_position_frame_is_numbered_in_time_order_and_the_position_stated(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I5 §7: the position frame is part of the set, listed at its time."""
        _text_video(db, tmp_path, duration=70.0)
        scenes(monkeypatch, [], duration=70.0)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        await ask(call, VIDEO, position=33.0)

        expected = sorted([33.0, *EVEN_70_AROUND_33])
        assert sent_keys(wire) == [ms(t) for t in expected]
        assert _listed_from(
            user_prompt(wire), 2,
            ["0:05", "0:15", "0:25", "0:33", "0:35", "0:45", "0:55", "1:05"],
        )

    async def test_a_failed_position_frame_is_left_out_stated_and_never_replaced(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I5 §2 (071 I7 revised): the scene frames are still sent, ``attached``."""
        _text_video(db, tmp_path, duration=70.0)
        scenes(monkeypatch, [], duration=70.0)
        FFmpeg(fail={ms(33.0)}).install(monkeypatch)
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO, position=33.0)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert sent_keys(wire) == [ms(t) for t in EVEN_70_AROUND_33]
        prompt = user_prompt(wire)
        assert "0:33" in prompt
        assert _listed_from(
            prompt, 2, ["0:05", "0:15", "0:25", "0:35", "0:45", "0:55", "1:05"],
        )
        assert "[9]" not in prompt

    async def test_failed_frames_are_dropped_and_the_rest_keep_numbering_after_the_passages(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 §2: a failed or timed-out frame is left out, no note, ``attached``."""
        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg(fail={ms(15.0)}, timeout={ms(55.0)}).install(monkeypatch)
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO)

        kept = [5.0, 25.0, 35.0, 45.0, 65.0, 75.0]
        assert sent_keys(wire) == [ms(t) for t in kept]
        assert by_kind(response)["meta"]["image"] == "attached"
        assert "error" not in by_kind(response)["done"]
        assert _listed_from(
            user_prompt(wire), 2, ["0:05", "0:25", "0:35", "0:45", "1:05", "1:15"],
        )

    @pytest.mark.parametrize("raises", ["load_selection_inputs", "select_scenes"])
    async def test_unreadable_selection_inputs_fall_back_to_evenly_spaced_frames(
        self, raises, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-114 §2/§8: evenly spaced filling, ``scene_candidates_failed`` logged."""
        _text_video(db, tmp_path)
        scenes(monkeypatch, [12.0, 47.0], duration=80.0)

        def _boom(*_args, **_kwargs):
            raise RuntimeError("vec_clip unreadable")

        patch_everywhere(monkeypatch, raises, _boom)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        with caplog.at_level("DEBUG"):
            response = await ask(call, VIDEO)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert sent_keys(wire) == [ms(t) for t in EVEN_80]
        assert "scene_candidates_failed" in caplog.text


# ---------------------------------------------------------------------------
# Citations across passages and frames
# ---------------------------------------------------------------------------


class TestCitations:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            (
                "A [1]. B [3]. C [5]. D [6]. E [2]. F [0].",
                [(1, 0.0), (2, 2.0), (3, 10.0), (5, 30.0)],
            ),
            ("Only the frame [4].", [(4, 20.0)]),
            ("Out of range [6] and [0].", []),
        ],
        ids=["passages-and-frames", "one-frame", "only-out-of-range"],
    )
    async def test_a_number_is_a_passage_up_to_p_and_a_frame_up_to_p_plus_f(
        self, answer, expected, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I3 §1.6 (053 I3 revised): any other number is dropped."""
        _text_video(
            db, tmp_path, duration=None,
            passages=((PASSAGE, 0.0), (PASSAGE_TWO, 2.0)),
        )
        scenes(monkeypatch, [10.0, 20.0, 30.0], duration=None)
        FFmpeg().install(monkeypatch)
        vision_client(llm, script=_answering(answer))

        response = await ask(call, VIDEO)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert _cited(response) == [(n, {"seconds": t}) for n, t in expected]

    async def test_a_dropped_frame_shifts_the_frame_numbers_to_the_frames_sent(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I3: ``[P+2]`` is the second frame kept, not the second chosen."""
        _text_video(db, tmp_path, duration=None)
        scenes(monkeypatch, [10.0, 20.0, 30.0], duration=None)
        FFmpeg(fail={ms(20.0)}).install(monkeypatch)
        _, wire = vision_client(llm, script=_answering("X [3]. Y [4]. Z [2]."))

        response = await ask(call, VIDEO)

        assert sent_keys(wire) == [ms(10.0), ms(30.0)]
        assert _cited(response) == [(2, {"seconds": 10.0}), (3, {"seconds": 30.0})]

    async def test_a_cited_position_frame_carries_the_position(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I3: the position frame takes its number in time order."""
        _text_video(db, tmp_path, duration=None)
        scenes(monkeypatch, [10.0, 30.0], duration=None)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm, script=_answering("At that moment [3]."))

        response = await ask(call, VIDEO, position=21.5)

        assert sent_keys(wire) == [ms(10.0), ms(21.5), ms(30.0)]
        assert _cited(response) == [(3, {"seconds": 21.5})]


# ---------------------------------------------------------------------------
# When frames cannot be used
# ---------------------------------------------------------------------------


class TestWhenFramesCannotBeUsed:
    @pytest.mark.parametrize("position", [None, 0.0], ids=["no-position", "position-zero"])
    async def test_no_frame_time_chosen_answers_from_the_passages_with_none_and_no_log(
        self, position, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-114 §2 I9: nothing tried; ``none``, no notice, no log line."""
        _text_video(db, tmp_path, duration=None)
        _text_video(db, tmp_path, SECOND, duration=80.0)
        _scenes_per_file(monkeypatch, {VIDEO: ([], None), SECOND: ([], 80.0)})
        ffmpeg = FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        with caplog.at_level("DEBUG"):
            response = await ask(call, VIDEO, position=position)
        warnings = _frame_warnings(caplog, VIDEO)

        assert ffmpeg.starts == []
        assert by_kind(response)["meta"]["image"] == "none"
        assert "error" not in by_kind(response)["done"]
        assert _answer(response) == Wire.text_answer
        assert wire.image_bodies == []
        assert len(wire.text_bodies) == 1
        assert PASSAGE in wire.text_of(wire.text_bodies[0])
        assert warnings == []

        framed = await ask(call, SECOND)
        assert by_kind(framed)["meta"]["image"] == "attached"
        assert sent_keys(wire) == [ms(t) for t in EVEN_80]

    async def test_no_frame_prepared_answers_from_the_passages_as_unavailable(
        self, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-114 I4 §2/§8: one text-only request, ``unavailable``, ``frames_unusable`` logged."""
        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        monkeypatch.setattr(frame_cache.subprocess, "run", ffmpeg_fails())
        _, wire = vision_client(llm)

        with caplog.at_level("DEBUG"):
            response = await ask(call, VIDEO)

        assert by_kind(response)["meta"]["image"] == "unavailable"
        assert by_kind(response)["done"].get("error") not in FRAME_ERRORS
        assert "error" not in by_kind(response)["done"]
        assert _answer(response) == Wire.text_answer
        assert wire.image_bodies == []
        assert len(wire.text_bodies) == 1
        assert PASSAGE in wire.text_of(wire.text_bodies[0])
        assert any(
            record.levelno == logging.WARNING
            and "frames_unusable" in record.getMessage()
            and VIDEO in record.getMessage()
            for record in caplog.records
        )

    @pytest.mark.parametrize("provider", ["openai_compatible", "ollama"])
    @pytest.mark.parametrize("status", [400, 404])
    async def test_a_refusal_retries_once_from_the_passages_and_writes_no_verdict(
        self, status, provider, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-114 I4 §2/§8: text-only retry, ``unavailable``, ``frames_rejected`` logged, no verdict."""
        from app.llm import reset_vision_capability_cache
        from app.llm_routing import profile_can_see

        reset_vision_capability_cache()
        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = Wire(provider)
        wire.script = _refusing_images(wire, status)
        vision_client(llm, provider=provider, wire=wire)

        with caplog.at_level("DEBUG"):
            response = await ask(call, VIDEO)

        assert len(wire.image_bodies) == 1
        assert sent_keys(wire) == [ms(t) for t in EVEN_80]
        assert len(wire.text_bodies) == 1
        assert PASSAGE in wire.text_of(wire.text_bodies[0])
        assert by_kind(response)["meta"]["image"] == "unavailable"
        assert "error" not in by_kind(response)["done"]
        assert _answer(response) == Wire.text_answer
        assert any(
            record.levelno == logging.WARNING
            and "frames_rejected" in record.getMessage()
            and VIDEO in record.getMessage()
            for record in caplog.records
        )

        assert profile_can_see(llm.result) is True
        again = await ask(call, VIDEO)
        assert by_kind(again)["meta"]["image"] == "unavailable"
        assert len(wire.image_bodies) == 2

    async def test_the_passages_are_the_same_with_frames_without_and_after_a_refusal(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I2: one budget and ranking for every path."""
        addon(budget=80)
        first = chunk("FIRST-PASSAGE", 50)
        second = chunk("SECOND-PASSAGE", 50)
        seed_text_less_video(db, VIDEO, path=tmp_path / f"{VIDEO}.mp4", duration=80.0)
        db.on(VIDEO).transcript(first, start=0.0, cos=0.2)
        db.on(VIDEO).transcript(second, start=2.0, cos=0.9)
        patch_everywhere(
            monkeypatch, "embed_query", lambda _text: toward(db.db.dim, 1.0),
        )
        scenes(monkeypatch, [], duration=80.0)

        FFmpeg().install(monkeypatch)
        _, framed = vision_client(llm)
        attached = await ask(call, VIDEO)
        assert by_kind(attached)["meta"]["image"] == "attached"
        assert len(framed.image_bodies) == 1
        assert mentioned(framed.text_of(framed.image_bodies[0]), [first, second]) == [second]

        refused = Wire("openai_compatible")
        refused.script = _refusing_images(refused, 400)
        vision_client(llm, wire=refused)
        retried = await ask(call, VIDEO)
        assert by_kind(retried)["meta"]["image"] == "unavailable"
        assert len(refused.image_bodies) == 1
        assert mentioned(refused.text_of(refused.image_bodies[0]), [first, second]) == [second]
        assert mentioned(refused.text_of(refused.text_bodies[0]), [first, second]) == [second]

        monkeypatch.setattr(frame_cache.subprocess, "run", ffmpeg_fails())
        frame_cache_dir = frame_cache.frame_cache_dir(VIDEO)
        for entry in list(frame_cache_dir.iterdir()) if frame_cache_dir.exists() else []:
            entry.unlink()
        _, unprepared = vision_client(llm)
        unavailable = await ask(call, VIDEO)
        assert by_kind(unavailable)["meta"]["image"] == "unavailable"
        assert unprepared.image_bodies == []
        assert mentioned(unprepared.text_of(unprepared.text_bodies[0]), [first, second]) == [second]

        _, blind = vision_client(llm, config=openai_config(vision_model=""))
        await ask(call, VIDEO)
        assert mentioned(blind.prompts[-1], [first, second]) == [second]


class TestGenerationFailures:
    @pytest.mark.parametrize(
        "how", [500, "timeout", "empty"], ids=["500", "timeout", "empty-answer"],
    )
    async def test_a_failure_while_answering_with_frames_is_generic_and_attached(
        self, how, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 §2 I9: the generic failure line, ``attached``, no text retry."""
        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = Wire("openai_compatible")

        def script(body):
            if not wire.has_image(body):
                return 200, wire.text_answer
            if how == "timeout":
                return httpx.ReadTimeout("timed out")
            if how == "empty":
                return 200, ""
            return how, ""

        wire.script = script
        vision_client(llm, wire=wire)

        response = await ask(call, VIDEO)

        error = by_kind(response)["done"].get("error")
        assert error
        assert error not in FRAME_ERRORS
        assert by_kind(response)["meta"]["image"] == "attached"
        assert len(wire.image_bodies) == 1
        assert wire.text_bodies == []

    async def test_a_failure_during_the_text_only_retry_is_generic_and_unavailable(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 §2/§3: one retry only; its failure is the generic line, ``unavailable``."""
        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = Wire("openai_compatible")
        wire.script = lambda body: (400, "") if wire.has_image(body) else (500, "")
        vision_client(llm, wire=wire)

        response = await ask(call, VIDEO)

        error = by_kind(response)["done"].get("error")
        assert error
        assert error not in FRAME_ERRORS
        assert by_kind(response)["meta"]["image"] == "unavailable"
        assert len(wire.image_bodies) == 1
        assert len(wire.text_bodies) == 1


# ---------------------------------------------------------------------------
# Paths this change does not touch, beside the one it does
# ---------------------------------------------------------------------------


class TestTheProfileDecides:
    async def test_a_blind_profile_keeps_todays_text_path_while_a_seeing_one_gets_frames(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I1/I8 §1.7: no image, position stated as today, ``meta`` as today."""
        _text_video(db, tmp_path, duration=4000.0, passages=((PASSAGE, 3725.0),))
        scenes(monkeypatch, [], duration=4000.0)
        ffmpeg = FFmpeg().install(monkeypatch)

        _, blind = vision_client(llm, config=openai_config(vision_model=""))
        at_33 = await ask(call, VIDEO, position=33.0)
        at_0 = await ask(call, VIDEO, position=0.0)
        assert blind.image_bodies == []
        assert ffmpeg.starts == []
        assert "0:33" in blind.prompts[0]
        assert "0:00" in blind.prompts[1]
        for response in (at_33, at_0):
            assert by_kind(response)["meta"] == {"partial": False, "image": "none"}
            assert _answer(response) == Wire.text_answer

        _, seeing = vision_client(llm)
        framed = await ask(call, VIDEO)
        assert by_kind(framed)["meta"]["image"] == "attached"
        assert len(seeing.images) == 8

    @pytest.mark.parametrize(
        ("body", "raw"),
        [({"provider": "vimeo", "url": "https://vimeo.com/1"}, None), (None, "{not json")],
        ids=["other-provider", "unreadable"],
    )
    async def test_a_loft_that_is_not_a_text_video_keeps_its_position_image_attempt(
        self, body, raw, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I8 §1.7 (053 I1 revised): no frame set; its attempt ends ``unavailable``."""
        _loft(db, tmp_path, OTHER_LOFT, body=body, raw=raw)
        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        other = await ask(call, OTHER_LOFT, position=5.0)
        assert by_kind(other)["meta"]["image"] == "unavailable"
        assert wire.image_bodies == []
        assert len(wire.text_bodies) == 1
        assert PASSAGE in wire.text_of(wire.text_bodies[0])

        native = await ask(call, VIDEO)
        assert by_kind(native)["meta"]["image"] == "attached"
        assert sent_keys(wire) == [ms(t) for t in EVEN_80]


class TestWhenTextArrives:
    async def test_the_question_after_a_transcript_lands_sends_passages_and_frames(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 §7 (108 I1 revised): frames from ``[1]`` before, after the passage once text lands."""
        seed_text_less_video(db, VIDEO, path=tmp_path / f"{VIDEO}.mp4", duration=80.0)
        scenes(monkeypatch, [12.0, 47.0], duration=80.0)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        before = await ask(call, VIDEO)
        assert by_kind(before)["meta"]["image"] == "attached"
        assert len(wire.image_bodies) == 1
        expected = [5.0, 12.0, 15.0, 25.0, 35.0, 45.0, 47.0, 55.0]

        db.on(VIDEO).transcript(PASSAGE, start=0.0)
        after = await ask(call, VIDEO)

        assert by_kind(after)["meta"]["image"] == "attached"
        assert len(wire.image_bodies) == 2
        assert sent_keys(wire)[8:] == [ms(t) for t in expected]
        prompt = user_prompt(wire)
        assert numbered_in_order(prompt, [PASSAGE])
        assert _listed_from(
            prompt, 2, ["0:05", "0:12", "0:15", "0:25", "0:35", "0:45", "0:47", "0:55"],
        )


# ---------------------------------------------------------------------------
# A YouTube reference with text
# ---------------------------------------------------------------------------


@needs_ffmpeg
class TestAYouTubeLoftWithText:
    async def test_its_frame_set_is_read_through_the_large_frame_route(
        self, db, addon, llm, call, tmp_path, youtube, monkeypatch,
    ):
        """SPEC-ADDON-114 §7 (077 I6 revised): every frame of the set, cached as ``large``."""
        _loft(
            db, tmp_path, LOFT, body={"provider": "youtube", "url": YOUTUBE_URL},
            duration=None,
        )
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
        sent = [decoded(p) for p in wire.images]
        assert sent == [
            frame_cache.frame_cache_path(LOFT, t, "large").read_bytes()
            for t in (10.0, 12.0, 20.0)
        ]

    async def test_when_no_frame_can_be_read_it_answers_from_the_text_as_unavailable(
        self, db, addon, llm, call, tmp_path, youtube, monkeypatch,
    ):
        """SPEC-ADDON-114 §2 (077 I6 revised): every frame failing ends ``unavailable`` with the text."""
        _loft(db, tmp_path, LOFT, body={"provider": "youtube", "url": YOUTUBE_URL})
        scenes(monkeypatch, [10.0, 20.0], duration=None)
        youtube.set_ytdlp(fail="ERROR: [youtube] x: Video unavailable")
        _, wire = vision_client(llm)

        response = await ask(call, LOFT)

        assert by_kind(response)["meta"]["image"] == "unavailable"
        assert wire.image_bodies == []
        assert len(wire.text_bodies) == 1
        assert PASSAGE in wire.text_of(wire.text_bodies[0])
        assert youtube.lookups()


# ---------------------------------------------------------------------------
# What is stored and logged
# ---------------------------------------------------------------------------


class TestWhatIsStoredOrLogged:
    async def test_only_the_large_frame_cache_entries_are_written(
        self, db, addon, llm, call, tmp_path, monkeypatch, statements,
    ):
        """SPEC-ADDON-114 I7 §4: no row, and no file but ``{ms}.large.webp``."""
        settings = addon()
        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)
        data_dir = Path(settings.intelligence_data_dir)
        before = {p for p in data_dir.rglob("*") if p.is_file()} if data_dir.exists() else set()
        statements.reset()

        await ask(call, VIDEO, question=QUESTION)

        assert len(wire.image_bodies) == 1
        assert statements.writes() == []
        new = {p for p in data_dir.rglob("*") if p.is_file()} - before
        assert sorted(p.name for p in new) == sorted(f"{ms(t)}.large.webp" for t in EVEN_80)
        assert {p.parent for p in new} == {data_dir / "frames" / VIDEO}

    async def test_no_question_answer_or_frame_byte_is_logged(
        self, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-114 I7 §8: ids and classes only."""
        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg(fail={ms(15.0)}).install(monkeypatch)
        _, wire = vision_client(llm, script=_answering("SECRET-ANSWER [1]"))

        with caplog.at_level("DEBUG"):
            response = await ask(call, VIDEO, question=QUESTION)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert len(wire.images) == 7
        logged = caplog.text
        assert "QUESTION-MARKER" not in logged
        assert "SECRET-ANSWER" not in logged
        assert MARKER.decode() not in logged
        assert "SPOKEN-PASSAGE" not in logged
