"""SPEC-ADDON-116: a question about a video lets the model ask for more frames.

The provider is the addon's real client over ``httpx.MockTransport``; its
script answers a request that offers ``get_frames`` with tool calls, and the
requests that follow are read back as the provider received them. Frame bytes
name their millisecond key, so every label is checked against its image.
"""

from __future__ import annotations

import json

import pytest

from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    VISION_MODEL,
    Reply,
    ToolCall,
    Wire,
    addon,
    ask,
    ask_slots,
    call,
    db,
    events,
    llm,
    statements,
    vision_client,
)
from tests.frame_label_layout import (
    all_frames,
    assert_labelled,
    labels_for,
    requested_frames,
    tool_messages,
)
from tests.frame_request_doubles import (
    ANSWER,
    ANSWER_TOKENS,
    EVEN_80,
    KINDS,
    PROVIDERS,
    QUESTION,
    VIDEO,
    answer_text,
    asks,
    assistant_messages,
    cited,
    get_frames,
    is_final,
    kinds,
    offers_tool,
    seed_video,
    tool_names,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.video_frames_doubles import FFmpeg, ms, scenes

AFTER_INITIAL = len(EVEN_80)


def _wire(llm, provider: str, script) -> Wire:
    wire = Wire(provider)
    wire.script = lambda body: script(wire, body)
    vision_client(llm, provider=provider, wire=wire)
    return wire


def _plays(*replies: Reply, then: Reply = Reply(text=ANSWER)):
    """Answer the tool requests with ``replies`` in turn, everything else with ``then``."""
    queue = list(replies)

    def script(wire, body):
        if offers_tool(wire, body) and queue:
            return queue.pop(0)
        return then

    return script


def _keys_after_initial(ffmpeg: FFmpeg) -> list[int]:
    return ffmpeg.starts[AFTER_INITIAL:]


class TestTheFirstRequest:
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_it_offers_get_frames_with_the_initial_frames_and_a_plain_reply_is_the_answer(
        self, provider, kind, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §1.2 §1.3 I5: a tool request to the vision model; no call, so its text answers."""
        first = seed_video(db, tmp_path, kind)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = _wire(llm, provider, _plays(then=Reply(text=f"{ANSWER} [{first}]")))

        pairs = events(await ask(call, VIDEO, question=QUESTION))

        assert len(wire.requests) == 1
        body = wire.requests[0]
        assert offers_tool(wire, body)
        assert tool_names(body) == ["get_frames"]
        if provider == "openai_compatible":
            assert body["tool_choice"] == "auto"
        assert body["model"] == VISION_MODEL
        assert_labelled(wire, body, first=first, times=EVEN_80)
        assert answer_text(pairs) == f"{ANSWER} [{first}]"
        answered = kinds(pairs).count("answer_chunk")
        assert answered >= 1
        assert kinds(pairs)[-3:] == ["citations", "meta", "done"]
        assert dict(pairs)["meta"]["image"] == "attached"
        assert cited(pairs) == [(first, {"seconds": 5.0})]

    async def test_an_ollama_profile_thinks_with_twice_the_answer_budget(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §1.2 I5: ``think`` on and ``num_predict`` of twice ``rag.max_tokens``."""
        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = _wire(llm, "ollama", _plays(asks(12.0)))

        await ask(call, VIDEO, question=QUESTION)

        assert len(wire.requests) == 2
        for body in wire.requests:
            assert body.get("think") is True
            assert body["options"]["num_predict"] == 2 * ANSWER_TOKENS

    async def test_thinking_text_is_never_written_or_sent_back(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §1.3 §1.5 I5: the reply's thinking is dropped from the stream and the exchange."""
        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = _wire(llm, "ollama", _plays(
            asks(12.0, thinking="THINKING-MARKER-ONE"),
            then=Reply(text=ANSWER, thinking="THINKING-MARKER-TWO"),
        ))

        pairs = events(await ask(call, VIDEO, question=QUESTION))

        assert len(wire.requests) == 2
        assert offers_tool(wire, wire.requests[1])
        assert "THINKING-MARKER" not in json.dumps(wire.requests[1])
        assert answer_text(pairs) == ANSWER


class TestServingAFrameRequest:
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_requested_frames_follow_the_tool_result_numbered_on_in_time_order(
        self, provider, kind, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §1.4 §1.5 §1.7 I3 (114 I11 revised): the exchange, then labelled frames from ``[P+F+1]``."""
        first = seed_video(db, tmp_path, kind)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        n = first + AFTER_INITIAL
        wire = _wire(llm, provider, _plays(
            asks(47.0, 12.0, text="PREAMBLE-MARKER"),
            then=Reply(text=f"{ANSWER} [{n + 1}] [{n}] [{first}] [{n + 2}]"),
        ))

        pairs = events(await ask(call, VIDEO, question=QUESTION))

        assert len(wire.requests) == 2
        second = wire.requests[1]
        assert offers_tool(wire, second)
        assert second["model"] == VISION_MODEL
        replies = assistant_messages(second)
        assert len(replies) == 1
        assert "PREAMBLE-MARKER" in (replies[0].get("content") or "")
        calls = replies[0].get("tool_calls") or []
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_frames"
        arguments = calls[0]["function"]["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        assert arguments == {"times": [47.0, 12.0]}
        results = tool_messages(second)
        assert len(results) == 1
        if provider == "openai_compatible":
            assert results[0]["tool_call_id"] == calls[0]["id"]
        later = requested_frames(wire, second)
        assert later.problem is None, later.problem
        assert later.frames == labels_for([12.0, 47.0], first=n)
        everything = all_frames(wire, second)
        assert everything.problem is None, everything.problem
        assert everything.frames == (
            labels_for(EVEN_80, first=first) + labels_for([12.0, 47.0], first=n)
        )
        assert "PREAMBLE-MARKER" not in answer_text(pairs)
        assert dict(pairs)["meta"]["image"] == "attached"
        assert cited(pairs) == [
            (n + 1, {"seconds": 47.0}), (n, {"seconds": 12.0}), (first, {"seconds": 5.0}),
        ]

    async def test_a_requested_time_is_rounded_to_the_millisecond_everywhere(
        self, db, addon, llm, call, tmp_path, monkeypatch, statements, ask_slots,
    ):
        """SPEC-ADDON-116 Terms (requested frame) I3 §4 (114 I6 I7 revised): key, label, citation; only the cache written; one slot."""
        from app.frame_cache import frame_cache_path
        from app.routers import rag as rag_router

        first = seed_video(db, tmp_path, "text-less-video")
        scenes(monkeypatch, [], duration=80.0)
        ffmpeg = FFmpeg().install(monkeypatch)
        n = first + AFTER_INITIAL
        slots: list[int] = []
        plays = _plays(asks(32.3456), then=Reply(text=f"See [{n}]."))

        def script(wire, body):
            slots.append(ask_slots._value)
            return plays(wire, body)

        wire = _wire(llm, "openai_compatible", script)
        statements.reset()

        pairs = events(await ask(call, VIDEO, question=QUESTION))

        assert _keys_after_initial(ffmpeg) == [32346]
        assert requested_frames(wire, wire.requests[1]).frames == [(f"[{n}] 0:32", 32346)]
        assert cited(pairs) == [(n, {"seconds": 32.346})]
        assert frame_cache_path(VIDEO, 32.346, "large").exists()
        assert statements.writes() == []
        assert slots == [rag_router._MAX_CONCURRENT_ASK - 1] * 2
        assert ask_slots._value == rag_router._MAX_CONCURRENT_ASK

    async def test_a_cached_large_frame_is_sent_without_ffmpeg(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §1.4.4: a requested time already in the ``large`` cache is read from it."""
        from tests.file_question_vision_doubles import write_large_frame
        from tests.video_frames_doubles import frame_bytes

        first = seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        write_large_frame(VIDEO, 12.0, frame_bytes(ms(12.0)))
        ffmpeg = FFmpeg().install(monkeypatch)
        wire = _wire(llm, "openai_compatible", _plays(asks(12.0, 52.0)))

        await ask(call, VIDEO, question=QUESTION)

        assert _keys_after_initial(ffmpeg) == [ms(52.0)]
        assert requested_frames(wire, wire.requests[1]).frames == labels_for(
            [12.0, 52.0], first=first + AFTER_INITIAL,
        )

    async def test_the_first_four_usable_times_are_cut_and_sent_in_time_order(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §1.4.3 §1.4.4 I1: four of the given order are used, sent by time."""
        first = seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        ffmpeg = FFmpeg(hold=0.1).install(monkeypatch)
        wire = _wire(llm, "openai_compatible", _plays(asks(70.0, 12.0, 47.0, 30.0, 60.0, 22.0)))

        await ask(call, VIDEO, question=QUESTION)

        assert sorted(_keys_after_initial(ffmpeg)) == [ms(t) for t in [12.0, 30.0, 47.0, 70.0]]
        assert ffmpeg.peak <= 4
        assert requested_frames(wire, wire.requests[1]).frames == labels_for(
            [12.0, 30.0, 47.0, 70.0], first=first + AFTER_INITIAL,
        )

    async def test_duplicates_are_dropped_before_the_limit_of_four(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §1.4.3: a duplicate in the same reply does not take one of the four."""
        first = seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = _wire(llm, "openai_compatible", _plays(asks(10.0, 10.5, 20.0, 30.0, 40.0, 50.0)))

        await ask(call, VIDEO, question=QUESTION)

        assert requested_frames(wire, wire.requests[1]).frames == labels_for(
            [10.0, 20.0, 30.0, 40.0], first=first + AFTER_INITIAL,
        )

    @pytest.mark.parametrize(
        ("given", "key"),
        [
            (32, 32000),
            (32.5, 32500),
            ("32", 32000),
            ("32.5", 32500),
            ("  32  ", 32000),
            ("1:02", 62000),
            ("1:02.5", 62500),
            ("0:01:02", 62000),
            ("0:00:32.25", 32250),
            (True, None),
            (False, None),
            (None, None),
            ("", None),
            ("abc", None),
            ("32s", None),
            ("1:60", None),
            ("0:60:00", None),
            ("0:00:60", None),
            (-3, None),
            ("-3", None),
            (85, None),
            ("1:25", None),
        ],
        ids=lambda value: repr(value),
    )
    async def test_only_a_readable_time_within_the_known_duration_is_cut(
        self, given, key, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 Terms (readable time) §1.4.3 I1: unreadable, negative or past the end is not used."""
        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        ffmpeg = FFmpeg().install(monkeypatch)
        wire = _wire(llm, "openai_compatible", _plays(asks(given, 52.0)))

        await ask(call, VIDEO, question=QUESTION)

        expected = sorted([ms(52.0)] + ([key] if key is not None else []))
        assert sorted(_keys_after_initial(ffmpeg)) == expected
        assert [k for _, k in requested_frames(wire, wire.requests[1]).frames] == expected

    @pytest.mark.parametrize(
        ("given", "used"),
        [
            (14.0, True),
            (14.001, False),
            (15.999, False),
            (16.0, True),
        ],
    )
    async def test_a_time_within_one_second_of_a_frame_already_sent_is_not_used(
        self, given, used, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 Terms (within 1 s) §1.4.3 I1: exactly 1.0 s from the frame at 0:15 is used."""
        first = seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        ffmpeg = FFmpeg().install(monkeypatch)
        wire = _wire(llm, "openai_compatible", _plays(asks(given)))

        await ask(call, VIDEO, question=QUESTION)

        assert _keys_after_initial(ffmpeg) == ([ms(given)] if used else [])
        result = tool_messages(wire.requests[1])[0]["content"]
        if not used:
            assert f"[{first + 1}]" in result

    async def test_a_time_within_one_second_of_an_earlier_time_of_the_reply_is_not_used(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §1.4.3: 30.9 after 30 is dropped, 31.0 is not."""
        first = seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = _wire(llm, "openai_compatible", _plays(asks(30.0, 30.9, 31.0)))

        await ask(call, VIDEO, question=QUESTION)

        assert requested_frames(wire, wire.requests[1]).frames == labels_for(
            [30.0, 31.0], first=first + AFTER_INITIAL,
        )

    @pytest.mark.parametrize("position_frame", ["kept", "failed"])
    async def test_a_time_within_one_second_of_the_position_is_not_used(
        self, position_frame, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §1.4.3 I1 (108 I5): the stated position counts even when its frame failed."""
        seed_video(db, tmp_path, "text-video", duration=70.0)
        scenes(monkeypatch, [], duration=70.0)
        fail = {ms(33.0)} if position_frame == "failed" else set()
        ffmpeg = FFmpeg(fail=fail).install(monkeypatch)
        wire = _wire(llm, "openai_compatible", _plays(asks(33.5, 32.2, 34.0)))

        await ask(call, VIDEO, question=QUESTION, position=33.0)

        assert ffmpeg.starts[AFTER_INITIAL:] == [ms(34.0)]
        assert [k for _, k in requested_frames(wire, wire.requests[1]).frames] == [ms(34.0)]

    async def test_a_time_already_sent_by_an_earlier_frame_request_is_not_cut_again(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §1.4.3 I1 I7: the tool result names the frame it duplicates, and only that."""
        first = seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        ffmpeg = FFmpeg().install(monkeypatch)
        n = first + AFTER_INITIAL
        wire = _wire(llm, "openai_compatible", _plays(asks(12.0), asks(12.5, 40.0)))

        await ask(call, VIDEO, question=QUESTION)

        assert len(wire.requests) == 3
        assert _keys_after_initial(ffmpeg) == [ms(12.0), ms(40.0)]
        assert requested_frames(wire, wire.requests[2]).frames == labels_for([40.0], first=n + 1)
        results = [m["content"] for m in tool_messages(wire.requests[2])]
        assert f"[{n}]" in results[1]
        assert re_numbers(results[0]) == []
        assert re_numbers(results[1]) == [n]

    async def test_a_failed_frame_is_not_replaced_and_the_numbers_close_up(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §1.4.4 I3: a dropped frame leaves no gap and no substitute."""
        first = seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        ffmpeg = FFmpeg(fail={ms(47.0)}, timeout={ms(30.0)}).install(monkeypatch)
        n = first + AFTER_INITIAL
        wire = _wire(llm, "openai_compatible", _plays(asks(12.0, 47.0, 30.0), asks(62.0)))

        await ask(call, VIDEO, question=QUESTION)

        assert sorted(_keys_after_initial(ffmpeg)) == [ms(t) for t in [12.0, 30.0, 47.0, 62.0]]
        assert requested_frames(wire, wire.requests[1]).frames == labels_for([12.0], first=n)
        assert requested_frames(wire, wire.requests[2]).frames == labels_for([62.0], first=n + 1)

    async def test_with_an_unknown_duration_no_time_is_past_the_end(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §2 (duration unknown) I1: a late time is still cut."""
        first = seed_video(db, tmp_path, "text-video", duration=None)
        scenes(monkeypatch, [10.0, 20.0], duration=None)
        ffmpeg = FFmpeg().install(monkeypatch)
        wire = _wire(llm, "openai_compatible", _plays(asks(5000.0)))

        await ask(call, VIDEO, question=QUESTION)

        assert ffmpeg.starts[2:] == [ms(5000.0)]
        assert requested_frames(wire, wire.requests[1]).frames == labels_for(
            [5000.0], first=first + 2,
        )

    async def test_a_requested_frame_is_cut_from_the_requested_file(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §6 I1: the requested frame's ffmpeg input is the asked file."""
        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        import app.frame_cache as frame_cache

        commands: list[list[str]] = []
        ffmpeg = FFmpeg()
        real_run = frame_cache.subprocess.run

        def _run(command, **kwargs):
            # The module is the shared ``subprocess``: the HTTP client's own
            # platform probe goes through it too and is not frame work.
            if not command or command[0] != "ffmpeg":
                return real_run(command, **kwargs)
            commands.append([str(part) for part in command])
            return ffmpeg.run(command, **kwargs)

        monkeypatch.setattr(frame_cache.subprocess, "run", _run)
        _wire(llm, "openai_compatible", _plays(asks(12.0)))

        await ask(call, VIDEO, question=QUESTION)

        source = str(tmp_path / f"{VIDEO}.mp4")
        assert all(source in command for command in commands)
        assert len(commands) == AFTER_INITIAL + 1


class TestTheLimitOfThree:
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_after_three_served_requests_the_next_request_is_final(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-116 §1.5 §1.6 I2 I5 §8: three served, then no tool; one count line."""
        first = seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        n = first + AFTER_INITIAL
        wire = _wire(llm, provider, _plays(
            asks(12.0, 13.5, "ARG-MARKER"), asks(22.0, 23.5), asks(32.0, 33.5), asks(42.0),
            then=Reply(text=f"{ANSWER} [{n + 4}]"),
        ))

        with caplog.at_level("DEBUG"):
            pairs = events(await ask(call, VIDEO, question=QUESTION))

        assert len(wire.requests) == 4
        assert [offers_tool(wire, b) for b in wire.requests] == [True, True, True, False]
        final = wire.requests[3]
        assert is_final(wire, final)
        if provider == "openai_compatible":
            assert final["tool_choice"] == "none"
            assert final["tools"] == wire.requests[0]["tools"]
        else:
            assert final.get("think") is True
            assert final["options"]["num_predict"] == 2 * ANSWER_TOKENS
        assert len(assistant_messages(final)) == 3
        assert len(tool_messages(final)) == 3
        times = [12.0, 13.5, 22.0, 23.5, 32.0, 33.5]
        assert all_frames(wire, final).frames == (
            labels_for(EVEN_80, first=first) + labels_for(times, first=n)
        )
        assert answer_text(pairs) == f"{ANSWER} [{n + 4}]"
        assert cited(pairs) == [(n + 4, {"seconds": 32.0})]
        assert dict(pairs)["meta"]["image"] == "attached"
        line = f"frame requests: 3 calls, 6 frames (file {VIDEO})"
        assert caplog.text.count(line) == 1
        for private in (QUESTION, ANSWER, "ARG-MARKER", "FRAME@", "RlJBTUVA"):
            assert private not in caplog.text

    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_calls_past_the_third_get_no_frame_and_the_next_request_is_final(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §1.4.1 §1.5 I2: four calls in one reply, three served, call order kept."""
        first = seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        ffmpeg = FFmpeg().install(monkeypatch)
        wire = _wire(llm, provider, _plays(Reply(calls=(
            get_frames(32.0), get_frames(12.0), get_frames(22.0), get_frames(42.0),
        ))))

        await ask(call, VIDEO, question=QUESTION)

        assert len(wire.requests) == 2
        second = wire.requests[1]
        assert is_final(wire, second)
        assert len(tool_messages(second)) == 4
        assert ms(42.0) not in ffmpeg.starts
        assert requested_frames(wire, second).frames == labels_for(
            [32.0, 12.0, 22.0], first=first + AFTER_INITIAL,
        )

    @pytest.mark.parametrize(
        "bad",
        [
            ToolCall({"times": [62.0]}, name="get_frame"),
            ToolCall({"time": [62.0]}),
            ToolCall({"times": 62.0}),
            ToolCall({"times": "62"}),
            ToolCall([62.0]),
            ToolCall("not json"),
        ],
        ids=["another-tool", "no-times", "times-a-number", "times-a-string", "a-list", "not-json"],
    )
    async def test_a_malformed_call_gets_a_result_and_counts_as_served(
        self, bad, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §1.4.2 §2: it cuts nothing, is answered, and spends one of the three."""
        first = seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        ffmpeg = FFmpeg().install(monkeypatch)
        wire = _wire(llm, "openai_compatible", _plays(
            Reply(calls=(bad,)), asks(12.0), asks(22.0),
        ))

        await ask(call, VIDEO, question=QUESTION)

        assert len(wire.requests) == 4
        assert is_final(wire, wire.requests[3])
        assert _keys_after_initial(ffmpeg) == [ms(12.0), ms(22.0)]
        assert len(tool_messages(wire.requests[1])) == 1
        assert all_frames(wire, wire.requests[3]).frames[AFTER_INITIAL:] == labels_for(
            [12.0, 22.0], first=first + AFTER_INITIAL,
        )

    async def test_a_request_that_yields_no_frame_counts_as_served(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §2 (no frame): every time unusable or failed still spends a request."""
        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg(fail={ms(62.0)}).install(monkeypatch)
        wire = _wire(llm, "openai_compatible", _plays(asks(15.2, 99.0), asks(62.0), asks("x")))

        await ask(call, VIDEO, question=QUESTION)

        assert len(wire.requests) == 4
        assert is_final(wire, wire.requests[3])
        assert len(tool_messages(wire.requests[3])) == 3
        assert all_frames(wire, wire.requests[3]).frames[AFTER_INITIAL:] == []


class TestStreaming:
    async def test_a_tool_requests_reply_is_written_only_once_complete(
        self, db, addon, llm, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §1.3 I6 §9: the answer appears after the reply's last byte."""
        from app.main import app
        from tests.video_frames_doubles import timed_post

        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = _wire(llm, "openai_compatible", _plays(then=Reply(text=ANSWER, gap=0.5)))

        _started, received, status = await timed_post(
            app, path=f"/files/{VIDEO}/ask", body={"question": QUESTION},
        )

        assert status == 200
        assert offers_tool(wire, wire.requests[0])
        first = next(at for at, data in received if b"event: answer_chunk" in data)
        assert first >= wire.yielded[-1][0]

    async def test_the_final_requests_reply_streams_as_it_arrives(
        self, db, addon, llm, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §1.6 I6: the first ``answer_chunk`` precedes the reply's second half."""
        from app.main import app
        from tests.video_frames_doubles import timed_post

        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = _wire(llm, "openai_compatible", _plays(
            asks(12.0), asks(22.0), asks(32.0), then=Reply(text=ANSWER, gap=0.5),
        ))

        _started, received, status = await timed_post(
            app, path=f"/files/{VIDEO}/ask", body={"question": QUESTION},
        )

        assert status == 200
        assert len(wire.requests) == 4
        assert is_final(wire, wire.requests[3])
        first = next(at for at, data in received if b"event: answer_chunk" in data)
        assert first < wire.yielded[-1][0]

    async def test_the_line_is_kept_alive_while_frames_are_requested(
        self, db, addon, llm, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §9 (114 I6 revised): comments between the tool request and the next one."""
        import threading
        import time

        from app.main import app
        from app.routers import rag as rag_router
        from tests.video_frames_doubles import timed_post

        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        ffmpeg = FFmpeg(hang={ms(12.0)}).install(monkeypatch)
        monkeypatch.setattr(rag_router, "_KEEPALIVE_SECONDS", 0.05)
        sent_at: list[float] = []
        plays = _plays(asks(12.0))

        def script(wire, body):
            sent_at.append(time.monotonic())
            if len(sent_at) == 1:
                threading.Timer(0.5, ffmpeg.release, args=(ms(12.0),)).start()
            return plays(wire, body)

        wire = _wire(llm, "openai_compatible", script)

        try:
            _started, received, status = await timed_post(
                app, path=f"/files/{VIDEO}/ask", body={"question": QUESTION},
            )
        finally:
            ffmpeg.release_all.set()

        assert status == 200
        assert len(sent_at) == 2
        assert offers_tool(wire, wire.requests[0])
        keepalives = [at for at, data in received if data == b": keepalive\n\n"]
        assert any(sent_at[0] < at < sent_at[1] for at in keepalives)

    async def test_no_silence_spans_a_tool_reply_a_cut_and_the_next_request(
        self, db, addon, llm, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §9 (114 I6 revised): each phase just under the interval adds no gap."""
        import threading

        from app.main import app
        from app.routers import rag as rag_router
        from tests.video_frames_doubles import timed_post

        interval = 0.2
        phase = 0.17
        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        ffmpeg = FFmpeg(hang={ms(12.0)}).install(monkeypatch)
        monkeypatch.setattr(rag_router, "_KEEPALIVE_SECONDS", interval)
        requests: list[int] = []
        plays = _plays(
            Reply(calls=asks(12.0).calls, delay=phase),
            then=Reply(text=ANSWER, delay=phase),
        )

        def script(wire, body):
            requests.append(1)
            if len(requests) == 1:
                threading.Timer(
                    2 * phase, ffmpeg.release, args=(ms(12.0),),
                ).start()
            return plays(wire, body)

        _wire(llm, "openai_compatible", script)

        try:
            started, received, status = await timed_post(
                app, path=f"/files/{VIDEO}/ask", body={"question": QUESTION},
            )
        finally:
            ffmpeg.release_all.set()

        assert status == 200
        assert len(requests) == 2
        times = [started, *(at for at, _data in received)]
        gaps = [later - earlier for earlier, later in zip(times, times[1:])]
        assert max(gaps) < 1.5 * interval


class TestBetweenQuestions:
    async def test_a_follow_up_starts_again_from_the_initial_frames(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §3: nothing kept; history carries only the earlier answer text."""
        first = seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = _wire(llm, "openai_compatible", _plays(asks(12.0)))

        await ask(call, VIDEO, question=QUESTION)
        assert len(wire.requests) == 2
        wire.requests.clear()
        again = _plays(asks(12.0))
        wire.script = lambda body: again(wire, body)
        response = await call(
            "POST", f"/files/{VIDEO}/ask",
            json={
                "question": QUESTION,
                "history": [{"question": "EARLIER-QUESTION", "answer": "EARLIER-ANSWER"}],
            },
        )

        assert dict(events(response))["meta"]["image"] == "attached"
        opening = wire.requests[0]
        assert offers_tool(wire, opening)
        assert tool_messages(opening) == []
        assert "EARLIER-ANSWER" in wire.text_of(opening)
        assert all_frames(wire, opening).frames == labels_for(EVEN_80, first=first)
        assert requested_frames(wire, wire.requests[1]).frames == labels_for(
            [12.0], first=first + AFTER_INITIAL,
        )


class TestTheClosedTab:
    async def test_a_disconnect_while_cutting_sends_and_cuts_nothing_more(
        self, db, addon, llm, ask_slots, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §2 (tab closed) §3: no further request, no further frame, slot freed."""
        import asyncio
        import threading

        from app.main import app
        from app.routers import rag as rag_router
        from tests.file_question_vision_doubles import DRIVE, disconnect_during_preparation

        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        reached = threading.Event()

        class _Watch(FFmpeg):
            def run(self, command, **kwargs):
                from tests.video_frames_doubles import key_of

                if key_of(command) == ms(12.0):
                    reached.set()
                return super().run(command, **kwargs)

        ffmpeg = _Watch(hang={ms(12.0)}).install(monkeypatch)
        wire = _wire(llm, "openai_compatible", _plays(asks(12.0, 52.0), asks(62.0)))

        try:
            await disconnect_during_preparation(
                app,
                path=f"/files/{VIDEO}/ask",
                body=json.dumps({"question": QUESTION}).encode(),
                headers=[
                    (b"content-type", b"application/json"),
                    (b"x-lit-drive", DRIVE.encode()),
                ],
                entered=reached,
                release=ffmpeg.release_all,
                timeout=5.0,
            )
        except TimeoutError:
            pass
        assert reached.is_set(), "the requested frame was never cut"
        started = len(ffmpeg.starts)
        await asyncio.sleep(1.0)

        assert len(wire.requests) == 1
        assert len(ffmpeg.starts) == started
        assert ms(62.0) not in ffmpeg.starts
        assert ask_slots._value == rag_router._MAX_CONCURRENT_ASK


def re_numbers(text: str) -> list[int]:
    import re

    return [int(n) for n in re.findall(r"\[(\d+)\]", text)]

