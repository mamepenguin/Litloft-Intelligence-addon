"""SPEC-ADDON-174: the result cap, an error while building, and a stored row."""

from __future__ import annotations

from app.routers import rag as rag_router
from tests.file_question_doubles import chunk
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    Reply,
    addon,
    ask,
    call,
    db,
    events,
    llm,
    vision_client,
)
from tests.frame_label_layout import assert_labelled, tool_messages
from tests.frame_repeat_doubles import ask_frame_chunk
from tests.frame_request_doubles import (
    ANSWER,
    BOTH_TOOLS,
    CUT_BUDGET,
    CUT_ROW_TEXTS,
    CUT_ROWS,
    EVEN_80,
    QUESTION,
    VIDEO,
    answer_text,
    get_transcript,
    offers_tool,
    plays,
    rank_in_file_order,
    seed_cut_video,
    tool_names,
    transcript_asks,
    wire_for,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.video_frames_doubles import (
    FFmpeg,
    scenes,
    seed_text_less_video,
)

TODAY_ANSWER = "TODAYS-ANSWER-MARKER"
LIMIT_NOTE = "was not sent: the limit of one call was reached."


def _cut(db, tmp_path, monkeypatch):
    seed_cut_video(db, tmp_path, CUT_ROWS)
    scenes(monkeypatch, [], duration=80.0)
    FFmpeg().install(monkeypatch)
    rank_in_file_order(monkeypatch, db)


class TestTheResultCap:
    async def test_a_row_at_the_boundary_leaves_room_for_the_note_newline(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I3: the stop note and its newline fit in 3,000."""
        seed_text_less_video(
            db, VIDEO, path=tmp_path / f"{VIDEO}.mp4", duration=36_200.0,
        )
        # A transcript row written as "[10]\n" + text + "\n" is 2,930
        # characters; the longest stop note is 70, so the note's own newline
        # is what puts the result over 3,000.
        first = chunk("HOUR-A", 2_924)
        later = chunk("HOUR-B", 10)
        db.on(VIDEO).transcript(first, start=36_000.0, end=36_005.0)
        db.on(VIDEO).transcript(later, start=36_010.0, end=36_015.0)
        scenes(monkeypatch, [], duration=36_200.0)
        FFmpeg().install(monkeypatch)
        rank_in_file_order(monkeypatch, db)
        addon(budget=1_500)
        wire = wire_for(llm, "openai_compatible", plays(
            transcript_asks(36_000.0, 36_030.0),
        ))

        await ask(call, VIDEO, question=QUESTION)

        result = tool_messages(wire.requests[1])[0]["content"]
        assert len(result) <= 3000
        assert LIMIT_NOTE in result


class TestAnUnexpectedErrorWhileBuilding:
    async def test_it_sends_todays_request_once_and_answers_from_it(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 §3: a build error goes to the fallback, as cutting's does."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)

        def explode(*_args, **_kwargs):
            raise RuntimeError("transcript build failed")

        monkeypatch.setattr(rag_router, "_serve_transcript", explode)

        def script(wire, body):
            if offers_tool(wire, body):
                return Reply(calls=(get_transcript(20.0, 40.0),))
            return Reply(text=TODAY_ANSWER)

        wire = wire_for(llm, "openai_compatible", script)

        pairs = events(await ask(call, VIDEO, question=QUESTION))

        assert len(wire.requests) == 2
        assert tool_names(wire.requests[0]) == BOTH_TOOLS
        todays = wire.requests[1]
        assert "tools" not in todays and "tool_choice" not in todays
        assert not any(
            message.get("role") in ("tool", "assistant")
            for message in todays.get("messages", [])
        )
        assert_labelled(wire, todays, first=2, times=EVEN_80)
        assert CUT_ROW_TEXTS[0] in wire.text_of(todays)
        for requested in CUT_ROW_TEXTS[1:]:
            assert requested not in wire.text_of(todays)
        assert answer_text(pairs) == TODAY_ANSWER


class TestAFirstSegmentRowSentWithItsStoredText:
    async def test_a_later_call_names_it_and_sends_no_removed_part(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I10: only a filtered first-segment row gets a supplement."""
        seed_text_less_video(
            db, VIDEO, path=tmp_path / f"{VIDEO}.mp4", duration=80.0,
        )
        # The line repeats from the first segment into the second, so the
        # second row's stored text loses it to the filter.
        ask_frame_chunk(db, VIDEO, 0, 0.0, 10.0, "R-UNIQUE-REMOVED\nEARLY")
        ask_frame_chunk(db, VIDEO, 1, 20.0, 30.0, "R-UNIQUE-REMOVED\nKEPT")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        rank_in_file_order(monkeypatch, db)
        addon(budget=1)
        wire = wire_for(llm, "openai_compatible", plays(
            transcript_asks(20.0, 30.0),
            transcript_asks(20.0, 30.0),
        ))

        await ask(call, VIDEO, question=QUESTION)

        first = tool_messages(wire.requests[1])[-1]["content"]
        later = tool_messages(wire.requests[2])[-1]["content"]
        assert "[10]\nR-UNIQUE-REMOVED\nKEPT" in first
        assert later == "already sent as [10]"
        assert "R-UNIQUE-REMOVED" not in later
        assert "[11]" not in later
