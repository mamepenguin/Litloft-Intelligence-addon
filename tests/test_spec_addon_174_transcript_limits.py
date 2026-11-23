"""SPEC-ADDON-174: the shared limit, the numbering, the fallback and the log line."""

from __future__ import annotations

import pytest

from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    Reply,
    ToolCall,
    addon,
    ask,
    call,
    db,
    events,
    llm,
    vision_client,
)
from tests.frame_label_layout import (
    assert_labelled,
    labels_for,
    requested_frames,
    tool_messages,
)
from tests.frame_request_doubles import (
    ANSWER,
    BOTH_TOOLS,
    CUT_BUDGET,
    CUT_ROW_TEXTS,
    CUT_ROWS,
    EVEN_80,
    PROVIDERS,
    QUESTION,
    VIDEO,
    get_frames,
    get_transcript,
    is_final,
    offers_tool,
    plays,
    rank_in_file_order,
    seed_cut_video,
    tool_names,
    wire_for,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.video_frames_doubles import FFmpeg, ms, scenes

LIMIT = "No frame or text follows: this question has already been served its 3 tool calls."
TODAY_ANSWER = "TODAYS-ANSWER-MARKER"


def _cut(db, tmp_path, monkeypatch):
    seed_cut_video(db, tmp_path, CUT_ROWS)
    scenes(monkeypatch, [], duration=80.0)
    ffmpeg = FFmpeg()
    ffmpeg.install(monkeypatch)
    rank_in_file_order(monkeypatch, db)
    return ffmpeg


def _failing(failure):
    if failure == "empty":
        return Reply(text="")
    return Reply(status=failure)


FAILURES = [400, 500, "empty"]


def _assert_todays_request(wire, body) -> None:
    """A request with no tools and no exchange: the passages and initial frames."""
    assert "tools" not in body and "tool_choice" not in body
    assert not any(
        message.get("role") in ("tool", "assistant")
        for message in body.get("messages", [])
    )
    assert_labelled(wire, body, first=2, times=EVEN_80)
    assert CUT_ROW_TEXTS[0] in wire.text_of(body)
    for requested in CUT_ROW_TEXTS[1:]:
        assert requested not in wire.text_of(body)


class TestTheSharedLimit:
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_the_first_three_calls_are_served_and_the_fourth_gets_the_limit(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I2: get_frames and get_transcript share one limit of three."""
        addon(budget=CUT_BUDGET)
        ffmpeg = _cut(db, tmp_path, monkeypatch)
        wire = wire_for(llm, provider, plays(Reply(calls=(
            get_frames(12.0),
            get_transcript(20.0, 40.0),
            get_frames(22.0),
            get_transcript(0.0, 5.0),
        ))))

        await ask(call, VIDEO, question=QUESTION)

        assert len(wire.requests) == 2
        assert tool_names(wire.requests[0]) == BOTH_TOOLS
        assert is_final(wire, wire.requests[1])
        results = [message["content"] for message in tool_messages(wire.requests[1])]
        assert len(results) == 4
        assert results[:3] != [LIMIT] * 3
        assert CUT_ROW_TEXTS[1] in results[1]
        assert CUT_ROW_TEXTS[2] in results[1]
        assert results[3] == LIMIT
        assert sorted(ffmpeg.starts[-2:]) == [ms(12.0), ms(22.0)]
        assert len(ffmpeg.starts) == len(EVEN_80) + 2

    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_the_fourth_call_of_a_later_reply_gets_the_limit(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I2: calls are counted across the replies of the question."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)
        wire = wire_for(llm, provider, plays(
            Reply(calls=(get_frames(12.0), get_transcript(20.0, 40.0))),
            Reply(calls=(get_transcript(0.0, 5.0), get_frames(22.0))),
        ))

        await ask(call, VIDEO, question=QUESTION)

        assert len(wire.requests) == 3
        assert offers_tool(wire, wire.requests[1])
        assert is_final(wire, wire.requests[2])
        results = [message["content"] for message in tool_messages(wire.requests[2])]
        assert len(results) == 4
        assert results[3] == LIMIT


class TestTheNumbering:
    async def test_requested_passages_take_the_next_numbers_before_the_replys_frames(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I4: passages, then requested passages, then the reply's frames."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(
            Reply(calls=(get_frames(12.0), get_transcript(20.0, 40.0))),
            then=Reply(text=f"{ANSWER} [10] [12]"),
        ))

        pairs = events(await ask(call, VIDEO, question=QUESTION))

        result = tool_messages(wire.requests[1])[1]["content"]
        assert f"[10]\n{CUT_ROW_TEXTS[1]}" in result
        assert f"[11]\n{CUT_ROW_TEXTS[2]}" in result
        assert requested_frames(wire, wire.requests[1]).frames == labels_for(
            [12.0], first=12,
        )
        assert dict(pairs)["citations"] == [
            {"n": 10, "position": {"seconds": 20.0}},
            {"n": 12, "position": {"seconds": 12.0}},
        ]
        assert dict(pairs)["meta"]["partial"] is True

    async def test_a_failed_frame_takes_no_number_and_the_next_frame_closes_up(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I4: a frame that fails leaves no gap; a later frame takes its place."""
        addon(budget=CUT_BUDGET)
        seed_cut_video(db, tmp_path, CUT_ROWS)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg(fail={ms(12.0)}).install(monkeypatch)
        rank_in_file_order(monkeypatch, db)
        wire = wire_for(llm, "openai_compatible", plays(
            Reply(calls=(get_frames(12.0), get_transcript(20.0, 40.0))),
            Reply(calls=(get_frames(22.0),)),
            then=Reply(text=f"{ANSWER} [10] [11] [12]"),
        ))

        pairs = events(await ask(call, VIDEO, question=QUESTION))

        assert len(wire.requests) == 3
        assert requested_frames(wire, wire.requests[1]).frames == []
        assert requested_frames(wire, wire.requests[2]).frames == labels_for(
            [22.0], first=12,
        )
        assert {entry["n"]: entry["position"] for entry in dict(pairs)["citations"]} == {
            10: {"seconds": 20.0},
            11: {"seconds": 30.0},
            12: {"seconds": 22.0},
        }


class TestTheFallback:
    @pytest.mark.parametrize("provider", PROVIDERS)
    @pytest.mark.parametrize("failure", FAILURES, ids=str)
    async def test_a_failed_tool_request_is_followed_by_todays_request_once(
        self, provider, failure, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I6: refused, failed or empty, then the passages and initial frames."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)

        def script(wire, body):
            return _failing(failure) if offers_tool(wire, body) else Reply(text=TODAY_ANSWER)

        wire = wire_for(llm, provider, script)

        pairs = events(await ask(call, VIDEO, question=QUESTION))

        assert len(wire.requests) == 2
        assert tool_names(wire.requests[0]) == BOTH_TOOLS
        _assert_todays_request(wire, wire.requests[1])
        assert dict(pairs)["meta"]["image"] == "attached"
        assert "".join(
            data.get("delta", "") for kind, data in pairs if kind == "answer_chunk"
        ) == TODAY_ANSWER

    @pytest.mark.parametrize("provider", PROVIDERS)
    @pytest.mark.parametrize("failure", FAILURES, ids=str)
    async def test_a_final_request_failing_before_its_first_chunk_falls_back_once(
        self, provider, failure, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I6: a final request's early failure also sends today's request once."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)

        def script(wire, body):
            if is_final(wire, body):
                return _failing(failure)
            if offers_tool(wire, body):
                return Reply(calls=(
                    get_transcript(20.0, 25.0),
                    get_transcript(30.0, 35.0),
                    get_transcript(0.0, 5.0),
                ))
            return Reply(text=TODAY_ANSWER)

        wire = wire_for(llm, provider, script)

        pairs = events(await ask(call, VIDEO, question=QUESTION))

        assert len(wire.requests) == 3
        assert tool_names(wire.requests[0]) == BOTH_TOOLS
        assert is_final(wire, wire.requests[1])
        results = [message["content"] for message in tool_messages(wire.requests[1])]
        assert CUT_ROW_TEXTS[1] in results[0]
        _assert_todays_request(wire, wire.requests[2])
        assert "".join(
            data.get("delta", "") for kind, data in pairs if kind == "answer_chunk"
        ) == TODAY_ANSWER


class TestTheLogLine:
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_it_counts_the_served_calls_frames_and_passages_once(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-174 I9: another name and an over-limit call are counted in neither."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)
        wire = wire_for(llm, provider, plays(Reply(calls=(
            get_frames(12.0),
            ToolCall({"times": ["TOOL-ARG-MARKER"]}, name="other_tool"),
            get_transcript(20.0, 40.0),
            get_transcript(0.0, 5.0),
        ))))

        with caplog.at_level("DEBUG"):
            await ask(call, VIDEO, question=QUESTION)

        line = (
            f"tool requests: 1 frame calls, 1 frames, 1 transcript calls, "
            f"2 passages (file {VIDEO})"
        )
        assert caplog.text.count(line) == 1
        for private in (QUESTION, ANSWER, "TOOL-ARG-MARKER", "CUT-ROW"):
            assert private not in caplog.text

    async def test_bad_arguments_are_counted_under_the_name_given(
        self, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-174 I9: a served call with bad arguments still counts for its name."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(Reply(calls=(
            ToolCall({"times": "BAD-FRAME-ARGUMENT"}, name="get_frames"),
            ToolCall("BAD-TRANSCRIPT-ARGUMENT", name="get_transcript"),
            get_transcript(20.0, 40.0),
        ))))

        with caplog.at_level("DEBUG"):
            await ask(call, VIDEO, question=QUESTION)

        assert len(wire.requests) == 2
        line = (
            f"tool requests: 1 frame calls, 0 frames, 2 transcript calls, "
            f"2 passages (file {VIDEO})"
        )
        assert caplog.text.count(line) == 1
        for private in (QUESTION, "BAD-FRAME-ARGUMENT", "BAD-TRANSCRIPT-ARGUMENT", "CUT-ROW"):
            assert private not in caplog.text

    async def test_a_question_that_serves_no_call_writes_no_cut_line(
        self, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-174 I9: the cut line belongs to a question that served a tool call."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)
        wire_for(llm, "openai_compatible", plays(then=Reply(text=ANSWER)))

        with caplog.at_level("DEBUG"):
            await ask(call, VIDEO, question=QUESTION)

        assert "tool requests:" not in caplog.text
        assert "frame requests:" not in caplog.text
