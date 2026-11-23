"""SPEC-ADDON-174: what a get_transcript result sends, in what order and shape."""

from __future__ import annotations

import re

import pytest

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
from tests.frame_label_layout import tool_messages
from tests.frame_repeat_doubles import ask_frame_chunk
from tests.frame_request_doubles import (
    CUT_BUDGET,
    CUT_ROW_TEXTS,
    CUT_ROWS,
    PROVIDERS,
    QUESTION,
    VIDEO,
    is_final,
    plays,
    rank_in_file_order,
    seed_cut_video,
    tool_names,
    transcript_asks,
    wire_for,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.video_frames_doubles import FFmpeg, scenes, seed_text_less_video

BAD_ARGUMENTS = (
    "No text follows: get_transcript takes a JSON object with start and end, "
    "two times with start before end."
)
CLOCK = re.compile(r"\b\d+:\d\d\b")
LIMIT_NOTE = "the limit of one call was reached."

#: Frame-text rows of the range's first segment and later segments: a line
#: repeated from an earlier segment, and one row the filter removes whole.
FRAME_ROWS = [
    (0.0, 10.0, "HEADER\nearly"),
    (60.0, 70.0, "HEADER\nA-KEEP"),
    (60.0, 70.0, "HEADER\nB-KEEP"),
    (60.0, 70.0, "HEADER"),
    (130.0, 140.0, "HEADER\nlate-A"),
    (130.0, 140.0, "HEADER\nlate-B"),
    (190.0, 200.0, "HEADER"),
]
SPOKEN_AT_SIXTY = chunk("SPOKEN-AT-SIXTY", 24)


def _cut(db, tmp_path, monkeypatch):
    seed_cut_video(db, tmp_path, CUT_ROWS)
    scenes(monkeypatch, [], duration=80.0)
    FFmpeg().install(monkeypatch)
    rank_in_file_order(monkeypatch, db)


def _frame_text_cut(db, tmp_path, monkeypatch, apply_addon, *, budget):
    seed_text_less_video(
        db, VIDEO, path=tmp_path / f"{VIDEO}.mp4", duration=80.0,
    )
    db.on(VIDEO).transcript(SPOKEN_AT_SIXTY, start=60.0, end=65.0)
    for index, (start, end, body) in enumerate(FRAME_ROWS):
        ask_frame_chunk(db, VIDEO, index, start, end, body)
    scenes(monkeypatch, [], duration=80.0)
    FFmpeg().install(monkeypatch)
    rank_in_file_order(monkeypatch, db)
    apply_addon(budget=budget)


def _result(wire, index: int = 1) -> str:
    return tool_messages(wire.requests[index])[0]["content"]


class TestTheWalkOfARange:
    async def test_rows_come_in_time_order_after_the_range_start(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I3: already-sent rows are named, the rest numbered on."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(
            transcript_asks(0.0, 40.0),
        ))

        await ask(call, VIDEO, question=QUESTION)

        result = _result(wire)
        assert result.startswith("already sent as [1]")
        assert f"[10]\n{CUT_ROW_TEXTS[1]}" in result
        assert f"[11]\n{CUT_ROW_TEXTS[2]}" in result
        first = result.index(f"[10]\n{CUT_ROW_TEXTS[1]}")
        second = result.index(f"[11]\n{CUT_ROW_TEXTS[2]}")
        assert result.index("already sent as [1]") < first < second
        assert not result.endswith("\n")

    async def test_a_transcript_row_comes_before_frame_text_at_the_same_start(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I10: the first segment is complete, later rows keep the filtered text."""
        _frame_text_cut(db, tmp_path, monkeypatch, addon, budget=24)
        wire = wire_for(llm, "openai_compatible", plays(
            transcript_asks(60.0, 200.0),
        ))

        await ask(call, VIDEO, question=QUESTION)

        result = _result(wire)
        starts = [
            "already sent as [1]",
            "[10]\nHEADER\nA-KEEP",
            "[11]\nHEADER\nB-KEEP",
            "[12]\nHEADER",
            "[13]\nlate-A",
            "[14]\nlate-B",
        ]
        for text in starts:
            assert text in result
        positions = [result.index(text) for text in starts]
        assert positions == sorted(positions)
        assert "[15]" not in result
        assert not result.endswith("\n")

    async def test_a_row_that_does_not_fit_is_not_cut_and_no_row_after_it_is_sent(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I3: the stop note names the row it stops at."""
        seed_text_less_video(
            db, VIDEO, path=tmp_path / f"{VIDEO}.mp4", duration=80.0,
        )
        long_rows = [
            chunk("LONG-A", 1500), chunk("LONG-B", 1500), chunk("LONG-C", 1500),
        ]
        for index, (text, start) in enumerate(zip(long_rows, (0.0, 10.0, 20.0))):
            db.on(VIDEO).transcript(text, start=start, end=start + 5.0)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        rank_in_file_order(monkeypatch, db)
        addon(budget=1500)
        wire = wire_for(llm, "openai_compatible", plays(
            transcript_asks(0.0, 30.0),
        ))

        await ask(call, VIDEO, question=QUESTION)

        result = _result(wire)
        stop = "Text from 0:20 on was not sent: " + LIMIT_NOTE
        assert result.endswith(stop)
        assert long_rows[1] in result
        assert "LONG-C" not in result
        assert "[11]" not in result
        assert len(result) <= 3000

    @pytest.mark.parametrize(
        ("base", "stop"),
        [
            (0.0, "Text from 0:20 on was not sent: " + LIMIT_NOTE),
            (3700.0, "Text from 1:02:00 on was not sent: " + LIMIT_NOTE),
        ],
        ids=["under-an-hour", "from-an-hour"],
    )
    async def test_the_stop_note_writes_the_row_start_as_a_clock_time(
        self, base, stop, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I5: a clock time in the stop note, never beside a number."""
        seed_text_less_video(
            db, VIDEO, path=tmp_path / f"{VIDEO}.mp4", duration=80.0,
        )
        for index, start in enumerate((base, base + 10.0, base + 20.0)):
            db.on(VIDEO).transcript(
                chunk(f"FAR-{index}", 1500), start=start, end=start + 5.0,
            )
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        rank_in_file_order(monkeypatch, db)
        addon(budget=1500)
        wire = wire_for(llm, "openai_compatible", plays(
            transcript_asks(base, base + 30.0),
        ))

        await ask(call, VIDEO, question=QUESTION)

        result = _result(wire)
        assert result.endswith(stop)
        clocked = [line for line in result.splitlines() if CLOCK.search(line)]
        assert clocked == [stop]
        assert not any("[" in line for line in clocked)

    @pytest.mark.parametrize(
        ("row_length", "written", "stopped_at"),
        [(2880, True, "0:20"), (2920, False, "0:10")],
        ids=["room-for-the-note", "no-room-for-the-note"],
    )
    async def test_the_limit_keeps_room_for_the_longest_stop_note(
        self, row_length, written, stopped_at,
        db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I3: a row is written only if it fits with the note."""
        seed_text_less_video(
            db, VIDEO, path=tmp_path / f"{VIDEO}.mp4", duration=80.0,
        )
        first = chunk("BOUND-A", 1500)
        middle = chunk("BOUND-B", row_length)
        last = chunk("BOUND-C", 100)
        for index, (text, start) in enumerate(zip((first, middle, last), (0.0, 10.0, 20.0))):
            db.on(VIDEO).transcript(text, start=start, end=start + 5.0)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        rank_in_file_order(monkeypatch, db)
        addon(budget=1500)
        wire = wire_for(llm, "openai_compatible", plays(
            transcript_asks(0.0, 30.0),
        ))

        await ask(call, VIDEO, question=QUESTION)

        result = _result(wire)
        if written:
            assert "[10]\n" + middle in result
        else:
            assert "[10]" not in result
        assert result.endswith(f"Text from {stopped_at} on was not sent: " + LIMIT_NOTE)
        assert len(result) <= 3000

    async def test_the_result_does_not_end_with_an_empty_line(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I3: the empty line after the last row is dropped."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(
            transcript_asks(20.0, 40.0),
        ))

        await ask(call, VIDEO, question=QUESTION)

        result = _result(wire)
        assert f"[10]\n{CUT_ROW_TEXTS[1]}" in result
        assert f"[11]\n{CUT_ROW_TEXTS[2]}" in result
        first = result.index(f"[10]\n{CUT_ROW_TEXTS[1]}")
        second = result.index(f"[11]\n{CUT_ROW_TEXTS[2]}")
        assert first < second
        assert not result.endswith("\n")


class TestTheFirstSegment:
    async def test_every_segment_at_the_smallest_start_is_given_its_stored_text(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I10: two segments share a start, so both are the first segment."""
        seed_text_less_video(
            db, VIDEO, path=tmp_path / f"{VIDEO}.mp4", duration=80.0,
        )
        db.on(VIDEO).transcript(SPOKEN_AT_SIXTY, start=60.0, end=65.0)
        for index, (start, end, body) in enumerate([
            (0.0, 10.0, "HEADER\nearly"),
            (60.0, 70.0, "HEADER\nfirst-end"),
            (60.0, 68.0, "HEADER\nsecond-end"),
        ]):
            ask_frame_chunk(db, VIDEO, index, start, end, body)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        rank_in_file_order(monkeypatch, db)
        addon(budget=24)
        wire = wire_for(llm, "openai_compatible", plays(
            transcript_asks(60.0, 75.0),
        ))

        await ask(call, VIDEO, question=QUESTION)

        result = _result(wire)
        assert "[10]\nHEADER\nfirst-end" in result
        assert "[11]\nHEADER\nsecond-end" in result
        first = result.index("[10]\nHEADER\nfirst-end")
        second = result.index("[11]\nHEADER\nsecond-end")
        assert result.index("already sent as [1]") < first < second

    async def test_a_sent_first_segment_row_is_named_and_gets_its_removed_parts(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I10: stored text for the first segment, filtered text after it."""
        _frame_text_cut(db, tmp_path, monkeypatch, addon, budget=42)
        wire = wire_for(llm, "openai_compatible", plays(
            transcript_asks(60.0, 200.0),
            then=Reply(text=f"{ANSWER} [12]"),
        ))

        pairs = events(await ask(call, VIDEO, question=QUESTION))

        result = _result(wire)
        assert result.startswith("already sent as [1]")
        assert "already sent as [3]" in result
        assert "[12]" in result
        sent = result.index("already sent as [3]")
        supplement = result.index("[12]", sent)
        removed = result.index("HEADER", supplement)
        assert "[13]" in result[removed:]
        row_two = result.index("[13]", removed)
        assert "A-KEEP" not in result[supplement:row_two]
        assert "HEADER\nB-KEEP" in result[row_two:]
        assert "[14]" in result[row_two:] and "HEADER" in result[row_two:]
        assert "[15]\nlate-A" in result
        assert "[16]\nlate-B" in result
        assert "[17]" not in result
        assert dict(pairs)["citations"] == [
            {"n": 12, "position": {"seconds": 60.0}},
        ]
        assert dict(pairs)["meta"]["partial"] is True

    async def test_a_later_call_does_not_send_the_removed_parts_again(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I10: a requested passage's removed parts count as sent."""
        _frame_text_cut(db, tmp_path, monkeypatch, addon, budget=42)
        wire = wire_for(llm, "openai_compatible", plays(
            transcript_asks(60.0, 200.0),
            transcript_asks(60.0, 70.0),
        ))

        await ask(call, VIDEO, question=QUESTION)

        result = _result(wire, 2)
        sent = [
            line for line in result.splitlines()
            if line.startswith("already sent")
        ]
        assert sent == [
            "already sent as [1]",
            "already sent as [3]",
            "already sent as [13]",
            "already sent as [14]",
        ]
        assert "HEADER" not in result
        assert "[12]" not in result
        assert not result.endswith("\n")


class TestTheArguments:
    @pytest.mark.parametrize("provider", PROVIDERS)
    @pytest.mark.parametrize(
        "arguments",
        [
            "not json",
            {"start": 20.0},
            {"start": "abc", "end": 30.0},
            {"start": 30.0, "end": 20.0},
            {"start": 20.0, "end": 20.0},
        ],
        ids=["not-an-object", "missing-end", "unreadable-start", "start-after-end", "start-equals-end"],
    )
    async def test_bad_arguments_get_the_exact_text_and_send_no_row(
        self, provider, arguments, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I3: an object with two readable times, start below end."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)
        from tests.file_question_vision_doubles import ToolCall

        bad = ToolCall(arguments, name="get_transcript")
        wire = wire_for(llm, provider, plays(Reply(calls=(bad,))))

        await ask(call, VIDEO, question=QUESTION)

        assert tool_names(wire.requests[0]) == ["get_frames", "get_transcript"]
        assert _result(wire) == BAD_ARGUMENTS
        assert "[" not in _result(wire)

    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_three_bad_calls_are_served_and_the_next_request_is_final(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I2: a bad call counts as served."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)
        from tests.file_question_vision_doubles import ToolCall

        bad = ToolCall({"start": 20.0}, name="get_transcript")
        wire = wire_for(llm, provider, plays(Reply(calls=(bad, bad, bad))))

        await ask(call, VIDEO, question=QUESTION)

        assert len(wire.requests) == 2
        assert [m["content"] for m in tool_messages(wire.requests[1])] == [BAD_ARGUMENTS] * 3
        assert is_final(wire, wire.requests[1])

    @pytest.mark.parametrize(
        ("start", "end", "text"),
        [
            (200.0, 210.0, "Nothing is said or shown between 3:20 and 3:30."),
            (3700.0, 3720.0, "Nothing is said or shown between 1:01:40 and 1:02:00."),
        ],
        ids=["under-an-hour", "from-an-hour"],
    )
    async def test_a_range_with_no_row_says_nothing_with_clock_times(
        self, start, end, text, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I3: a range past the end holds no row."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(
            transcript_asks(start, end),
        ))

        await ask(call, VIDEO, question=QUESTION)

        assert _result(wire) == text
        assert "[" not in text


class TestTheLinesTheToolWrites:
    async def test_row_text_is_sent_unchanged_with_its_own_clock_stamps(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I5: only the tool's own lines follow the clock rule."""
        spoken = "shown at 0:15 and again at 1:02:03"
        rows = [
            (CUT_ROW_TEXTS[0], 0.0, 5.0),
            (spoken, 20.0, 25.0),
            (CUT_ROW_TEXTS[2], 30.0, 35.0),
        ]
        seed_cut_video(db, tmp_path, rows)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        rank_in_file_order(monkeypatch, db)
        addon(budget=CUT_BUDGET)
        wire = wire_for(llm, "openai_compatible", plays(
            transcript_asks(0.0, 40.0),
        ))

        await ask(call, VIDEO, question=QUESTION)

        result = _result(wire)
        assert f"[10]\n{spoken}" in result
        own = [line for line in result.splitlines() if line.startswith("[") or line.startswith("already sent")]
        assert [line for line in own if CLOCK.search(line)] == []
