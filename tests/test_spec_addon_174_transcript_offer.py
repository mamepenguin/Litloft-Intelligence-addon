"""SPEC-ADDON-174: a cut video's question offers get_transcript beside get_frames."""

from __future__ import annotations

import json

import pytest

from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    Reply,
    addon,
    ask,
    call,
    db,
    events,
    llm,
    statements,
    vision_client,
)
from tests.frame_label_layout import tool_messages
from tests.frame_request_doubles import (
    ANSWER,
    BOTH_TOOLS,
    CUT_BUDGET,
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
    seed_video,
    tool_names,
    transcript_asks,
    wire_for,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.video_frames_doubles import FFmpeg, scenes

GET_TRANSCRIPT_DESCRIPTION = (
    "Return what is said and shown in this video between two times, apart from "
    "what you already have."
)
TIME_DESCRIPTION = 'Times in the video, as seconds (83.5) or m:ss ("1:23")'

OPENING = (
    "the numbered passages below, the numbered passages get_transcript returns, "
    "and the numbered frames that follow the question"
)
MUST_CALL = (
    "If the answer may be in a part of the video the passages do not include, "
    "you must call get_transcript for that part before answering, using the "
    "times the contents give. Never say that something is not said in the video "
    "unless you have called get_transcript for the part where it would be."
)
PASSAGES_RULE = (
    "The passages below are part of the file's text, the parts closest to the "
    "question; the frames that follow the question are moments of the video, in "
    "time order after the passages."
)
TOOL_SECTION = [
    "More frames and text:",
    "- The frames you are given are a sample of the video, not all of it, and "
    "the passages are part of its text, not all of it.",
    "- The tool get_frames returns frames of this video at the times you give "
    "it: up to 4 times per call.",
    "- The tool get_transcript returns what is said and shown in this video "
    "between two times, up to 3,000 characters per call; a part you already "
    "have is named by its number instead.",
    "- You may make at most 3 calls of get_frames and get_transcript together "
    "for this question.",
    "- Call get_frames when the answer depends on what is shown at moments the "
    "frames you have do not show, and get_transcript when it may be said or "
    "shown in a part the passages do not include.",
    "- Never ask the viewer for times; choose them yourself from the times the "
    "prompt gives you.",
    "- The frames and passages these tools return are numbered and cited "
    "exactly as the ones you already have: [n], right after the sentence they "
    "support.",
]


def _cut(db, tmp_path, monkeypatch):
    seed_cut_video(db, tmp_path, CUT_ROWS)
    scenes(monkeypatch, [], duration=80.0)
    FFmpeg().install(monkeypatch)
    rank_in_file_order(monkeypatch, db)


def _system_sentences(system: str) -> list[str]:
    return [sentence.strip() for sentence in system.split(".") if sentence.strip()]


class TestTheToolOfferOfCutMaterial:
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_every_tool_request_offers_both_tools_and_the_final_request_lists_them_again(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I1: get_frames then get_transcript, and the final request offers none."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)
        wire = wire_for(llm, provider, plays(
            Reply(calls=(get_transcript(20.0, 25.0), get_transcript(30.0, 35.0))),
            Reply(calls=(get_transcript(0.0, 5.0),)),
        ))

        await ask(call, VIDEO, question=QUESTION)

        assert len(wire.requests) == 3
        for body in wire.requests[:2]:
            assert offers_tool(wire, body)
            assert tool_names(body) == BOTH_TOOLS
        final = wire.requests[2]
        assert is_final(wire, final)
        if provider == "openai_compatible":
            assert tool_names(final) == BOTH_TOOLS
            assert final["tool_choice"] == "none"
        else:
            assert not final.get("tools")

    async def test_the_get_transcript_definition_is_the_one_the_prompt_describes(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I1: name, description and the two required time strings."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(then=Reply(text=ANSWER)))

        await ask(call, VIDEO, question=QUESTION)

        tools = {
            tool["function"]["name"]: tool["function"]
            for tool in wire.requests[0]["tools"]
        }
        definition = tools["get_transcript"]
        assert definition["description"] == GET_TRANSCRIPT_DESCRIPTION
        assert definition["parameters"] == {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": TIME_DESCRIPTION},
                "end": {"type": "string", "description": TIME_DESCRIPTION},
            },
            "required": ["start", "end"],
        }

    async def test_material_that_is_not_cut_is_offered_get_frames_alone(
        self, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-174 I1: an uncut video keeps SPEC-ADDON-116's requests and log line."""
        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(
            Reply(calls=(get_frames(12.0), get_frames(22.0), get_frames(32.0))),
            then=Reply(text=ANSWER),
        ))

        with caplog.at_level("DEBUG"):
            pairs = events(await ask(call, VIDEO, question=QUESTION))

        assert len(wire.requests) == 2
        for body in wire.requests:
            assert tool_names(body) == ["get_frames"]
            assert "get_transcript" not in json.dumps(body)
        assert wire.requests[1]["tool_choice"] == "none"
        assert wire.requests[1]["tools"] == wire.requests[0]["tools"]
        assert [
            message["content"] for message in tool_messages(wire.requests[1])
        ] == ["1 frame follow."] * 3
        assert dict(pairs)["meta"]["partial"] is False
        line = f"frame requests: 3 calls, 3 frames (file {VIDEO})"
        assert caplog.text.count(line) == 1
        assert "tool requests:" not in caplog.text


class TestTheSystemPromptOfCutMaterial:
    async def test_it_names_the_transcript_passages_where_the_answer_uses_them(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I8: the opening and the rules name the transcript's passages."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(then=Reply(text=ANSWER)))

        await ask(call, VIDEO, question=QUESTION)

        system = wire.system_of(wire.requests[0])
        opening = _system_sentences(system)[0]
        assert OPENING in opening
        assert PASSAGES_RULE in system
        assert MUST_CALL in system

    async def test_with_history_the_citable_sentence_names_the_same_three(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I8: the history sentence counts transcript passages as citable."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(then=Reply(text=ANSWER)))

        await call(
            "POST", f"/files/{VIDEO}/ask",
            json={
                "question": QUESTION,
                "history": [{"question": "EARLIER-QUESTION", "answer": "EARLIER-ANSWER"}],
            },
        )

        system = wire.system_of(wire.requests[0])
        citable = next(
            sentence for sentence in _system_sentences(system)
            if "may be cited" in sentence
        )
        assert "numbered passages below" in citable
        assert "get_transcript returns" in citable
        assert "numbered frames that follow the question" in citable

    async def test_its_tool_section_reads_the_lines_the_spec_words(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-174 I8: the "More frames and text:" section line for line."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(then=Reply(text=ANSWER)))

        await ask(call, VIDEO, question=QUESTION)

        system = wire.system_of(wire.requests[0])
        for line in TOOL_SECTION:
            assert line in system
        assert "More frames:" not in system


class TestWhatTheToolPathWrites:
    async def test_a_question_with_transcript_calls_writes_nothing(
        self, db, addon, llm, call, tmp_path, monkeypatch, statements,
    ):
        """SPEC-ADDON-174 I7: the transcript is read from the question's own material."""
        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(
            transcript_asks(20.0, 40.0),
        ))
        statements.reset()

        await ask(call, VIDEO, question=QUESTION)

        assert tool_names(wire.requests[0]) == BOTH_TOOLS
        assert statements.writes() == []

    async def test_a_frame_cache_entry_is_written_and_no_row(
        self, db, addon, llm, call, tmp_path, monkeypatch, statements,
    ):
        """SPEC-ADDON-174 I7: a requested frame adds its cache entry and no row."""
        from app.frame_cache import frame_cache_path

        addon(budget=CUT_BUDGET)
        _cut(db, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(
            Reply(calls=(get_frames(12.0), get_transcript(20.0, 40.0))),
        ))
        statements.reset()

        await ask(call, VIDEO, question=QUESTION)

        assert tool_names(wire.requests[0]) == BOTH_TOOLS
        assert frame_cache_path(VIDEO, 12.0, "large").exists()
        assert statements.writes() == []
