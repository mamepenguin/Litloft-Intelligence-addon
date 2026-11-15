"""SPEC-ADDON-116 failure cases: the fallback to today's request, the limits in time.

Today's request is the body the frames path sent for the same question before
``get_frames`` existed, recorded in ``tests/golden_prompts``. The 120 s window
and the 300 s deadline are shrunk through their module constants.
"""

from __future__ import annotations

import httpx
import pytest

from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    Reply,
    Wire,
    addon,
    ask,
    call,
    db,
    events,
    llm,
    openai_config,
    ollama_config,
    seed_audio,
    seed_document,
    seed_picture,
    vision_client,
)
from tests.frame_label_layout import requested_frames, tool_messages
from tests.frame_request_doubles import (
    ANSWER,
    DEADLINE_SECONDS,
    EVEN_80,
    FAILED,
    KINDS,
    PROVIDERS,
    QUESTION,
    VIDEO,
    WINDOW_SECONDS,
    answer_text,
    asks,
    is_final,
    kinds,
    offers_tool,
    seed_video,
    shrink,
    todays_request,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.video_frames_doubles import FFmpeg, ms, scenes

TODAY_ANSWER = "TODAYS-ANSWER-MARKER"
FALLBACK_LOG = f"frame_tool_fallback (file {VIDEO})"
DEADLINE_LOG = f"frame_tool_deadline (file {VIDEO})"


def _wire(llm, provider: str, script) -> Wire:
    wire = Wire(provider)
    wire.script = lambda body: script(wire, body)
    vision_client(llm, provider=provider, wire=wire)
    return wire


def _verdicts() -> dict:
    from app.llm import _vision_capability_cache

    return dict(_vision_capability_cache._verdicts)


def _carries_exchange(body: dict) -> bool:
    return any(m.get("role") in ("tool", "assistant") for m in body.get("messages", []))


def _today(wire, body) -> bool:
    return not offers_tool(wire, body) and not _carries_exchange(body)


def _failing(failure):
    """A reply, or an exception to raise, standing for one kind of failure."""
    if failure == "timeout":
        return httpx.ReadTimeout("timed out")
    if failure == "empty":
        return Reply(text="")
    return Reply(status=failure)


FAILURES = [400, 404, 500, 503, "timeout", "empty"]


def _serves_three(failure_on_final, *, today=None):
    """Three tool requests served, the final request answered by ``failure_on_final``."""
    queue = [asks(12.0), asks(22.0), asks(32.0)]

    def script(wire, body):
        if offers_tool(wire, body):
            return queue.pop(0) if queue else asks(42.0)
        if is_final(wire, body):
            return failure_on_final
        if today is not None and wire.has_image(body):
            return today
        return Reply(text=TODAY_ANSWER)

    return script


def _ended_failed(pairs) -> None:
    assert kinds(pairs)[-2:] == ["meta", "done"]
    assert dict(pairs)["meta"]["image"] == "attached"
    assert pairs[-1][1] == FAILED


class TestAFailedToolRequest:
    @pytest.mark.parametrize("failure", FAILURES, ids=str)
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_it_is_followed_by_todays_request_once(
        self, provider, kind, failure, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-116 §2 I4: refused, failed or empty, then exactly today's request; no verdict."""
        seed_video(db, tmp_path, kind)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = _wire(llm, provider, lambda wire, body: (
            _failing(failure) if offers_tool(wire, body) else Reply(text=TODAY_ANSWER)
        ))

        with caplog.at_level("DEBUG"):
            pairs = events(await ask(call, VIDEO, question=QUESTION))

        assert len(wire.requests) == 2
        assert offers_tool(wire, wire.requests[0])
        assert wire.requests[1] == todays_request(provider, kind)
        assert answer_text(pairs) == TODAY_ANSWER
        assert dict(pairs)["meta"]["image"] == "attached"
        assert caplog.text.count(FALLBACK_LOG) == 1
        assert _verdicts() == {}

    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_a_refusal_of_todays_request_too_ends_in_the_text_only_answer(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §2 I4 (114 I4 revised): today's refusal handling takes over; no verdict."""
        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = _wire(llm, provider, lambda wire, body: (
            Reply(status=400) if offers_tool(wire, body) or wire.has_image(body)
            else Reply(text=TODAY_ANSWER)
        ))

        pairs = events(await ask(call, VIDEO, question=QUESTION))

        assert len(wire.requests) == 3
        assert offers_tool(wire, wire.requests[0])
        assert wire.requests[1] == todays_request(provider, "text-video")
        assert not wire.has_image(wire.requests[2])
        assert answer_text(pairs) == TODAY_ANSWER
        assert dict(pairs)["meta"]["image"] == "unavailable"
        assert _verdicts() == {}


class TestAFailedFinalRequest:
    @pytest.mark.parametrize(
        "failure", [400, 500, "timeout", "empty", "only-a-call"], ids=str,
    )
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_before_its_first_chunk_it_is_followed_by_todays_request_once(
        self, provider, failure, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-116 §2 I4: a final request that leaves no answer falls back like a tool request."""
        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        reply = asks(42.0) if failure == "only-a-call" else _failing(failure)
        wire = _wire(llm, provider, _serves_three(reply))

        with caplog.at_level("DEBUG"):
            pairs = events(await ask(call, VIDEO, question=QUESTION))

        assert len(wire.requests) == 5
        assert is_final(wire, wire.requests[3])
        assert wire.requests[4] == todays_request(provider, "text-video")
        assert answer_text(pairs) == TODAY_ANSWER
        assert dict(pairs)["meta"]["image"] == "attached"
        assert caplog.text.count(FALLBACK_LOG) == 1
        assert _verdicts() == {}

    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_after_its_first_chunk_the_answer_fails_with_nothing_sent_after(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §2 §3 (``final`` → ``failed``): meta ``attached``, then the failed ``done``."""
        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = _wire(llm, provider, _serves_three(Reply(text="PARTIAL-MARKER", cut=True)))

        pairs = events(await ask(call, VIDEO, question=QUESTION))

        assert len(wire.requests) == 4
        assert is_final(wire, wire.requests[3])
        assert "PARTIAL-MARKER" in answer_text(pairs)
        _ended_failed(pairs)

    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_at_most_six_model_requests_are_sent(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §5 §10 I2 I4: three tool, one final, today's, then 114's text-only request."""
        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = _wire(llm, provider, _serves_three(Reply(status=400), today=Reply(status=400)))

        pairs = events(await ask(call, VIDEO, question=QUESTION))

        assert len(wire.requests) == 6
        assert [offers_tool(wire, b) for b in wire.requests] == [True] * 3 + [False] * 3
        assert is_final(wire, wire.requests[3])
        assert wire.requests[4] == todays_request(provider, "text-video")
        assert not wire.has_image(wire.requests[5])
        assert dict(pairs)["meta"]["image"] == "unavailable"


class TestWhenNoToolIsOffered:
    @pytest.mark.parametrize("kind", KINDS)
    async def test_no_initial_frame_means_no_tool_request(
        self, kind, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §2 (no initial frame) §3: today's path; the same file with frames gets the tool."""
        seed_video(db, tmp_path, kind)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg(fail={ms(t) for t in EVEN_80}).install(monkeypatch)
        wire = _wire(llm, "openai_compatible", lambda wire, body: Reply(text=ANSWER))

        await ask(call, VIDEO, question=QUESTION)
        without = list(wire.requests)
        wire.requests.clear()
        FFmpeg().install(monkeypatch)
        await ask(call, VIDEO, question=QUESTION)

        assert not any(offers_tool(wire, body) or body.get("tools") for body in without)
        assert offers_tool(wire, wire.requests[0])

    @pytest.mark.parametrize(
        "subject", ["picture", "audio", "document", "blind-openai", "blind-ollama"],
    )
    async def test_other_files_and_blind_profiles_are_never_offered_the_tool(
        self, subject, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §7 (114 I8 revised): only a video with a seeing profile gets ``get_frames``."""
        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        other = "oth000000216"
        if subject == "picture":
            seed_picture(db, other, text="PICTURE-TEXT", path=tmp_path / "p.png")
        elif subject == "audio":
            seed_audio(db, other, text="AUDIO-TEXT", path=tmp_path / "a.mp3")
        elif subject == "document":
            seed_document(db, other, text="DOCUMENT-TEXT", path=tmp_path / "d.pdf")
        else:
            other = VIDEO
        config = None
        provider = "ollama" if subject == "blind-ollama" else "openai_compatible"
        if subject == "blind-openai":
            config = openai_config(vision_model="")
        elif subject == "blind-ollama":
            config = ollama_config(vision_model="")
        wire = Wire(provider)
        wire.script = lambda body: Reply(text=ANSWER)
        vision_client(llm, provider=provider, wire=wire, config=config)

        await ask(call, other, question=QUESTION)
        unchanged = list(wire.requests)
        wire.requests.clear()
        vision_client(llm, provider=provider, wire=wire)
        await ask(call, VIDEO, question=QUESTION)

        assert unchanged
        for body in unchanged:
            assert "tools" not in body and "tool_choice" not in body
            assert body.get("think") is not True
        assert offers_tool(wire, wire.requests[0])


class TestTheRequestBody:
    @pytest.mark.parametrize("kind", KINDS)
    async def test_openai_tool_and_final_requests_differ_from_today_only_by_the_exchange(
        self, kind, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 I5 (revised): ``stream``, temperature and model are today's; no max tokens."""
        seed_video(db, tmp_path, kind)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = _wire(llm, "openai_compatible", _serves_three(Reply(text=ANSWER)))

        await ask(call, VIDEO, question=QUESTION)

        assert len(wire.requests) == 4
        exchange = ("messages", "tools", "tool_choice")
        limits = ("max_tokens", "max_completion_tokens")
        todays = todays_request("openai_compatible", kind)
        assert any(k in todays for k in limits)
        today = {k: v for k, v in todays.items() if k not in exchange + limits}
        for body in wire.requests:
            assert not any(k in body for k in limits)
            assert {k: v for k, v in body.items() if k not in exchange} == today
        assert [b["tool_choice"] for b in wire.requests] == ["auto"] * 3 + ["none"]

    @pytest.mark.parametrize("kind", KINDS)
    async def test_the_tool_requests_system_prompt_names_get_frames_and_todays_does_not(
        self, kind, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 I8: the tool instructions are in the tool request's system prompt only."""
        seed_video(db, tmp_path, kind)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = _wire(llm, "openai_compatible", lambda wire, body: (
            Reply(status=400) if offers_tool(wire, body) else Reply(text=TODAY_ANSWER)
        ))

        await ask(call, VIDEO, question=QUESTION)

        assert len(wire.requests) == 2
        assert "get_frames" in wire.system_of(wire.requests[0])
        assert "get_frames" not in wire.system_of(wire.requests[1])


class TestTheToolWindow:
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_once_the_window_has_passed_the_next_request_is_final(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §1.5 I2 §10: past the window (120 s, shrunk) the tool is not offered again."""
        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        shrink(monkeypatch, WINDOW_SECONDS, 1.0)
        slow = Reply(calls=asks(12.0).calls, delay=1.5)
        queue = [slow, asks(22.0)]
        wire = _wire(llm, provider, lambda wire, body: (
            queue.pop(0) if offers_tool(wire, body) and queue else Reply(text=ANSWER)
        ))

        pairs = events(await ask(call, VIDEO, question=QUESTION))

        assert len(wire.requests) == 2
        assert is_final(wire, wire.requests[1])
        assert len(tool_messages(wire.requests[1])) == 1
        assert [k for _, k in requested_frames(wire, wire.requests[1]).frames] == [ms(12.0)]
        assert answer_text(pairs) == ANSWER


class TestTheTotalDeadline:
    @pytest.mark.parametrize(
        "stage", ["tool-request", "cutting", "fallback", "final-after-first-chunk"],
    )
    async def test_it_stops_what_is_in_flight_and_ends_the_answer_failed(
        self, stage, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-116 §2 (deadline) §3 I9: nothing started after 300 s (shrunk); meta, failed ``done``."""
        import time

        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        ffmpeg = FFmpeg(hang={ms(12.0)} if stage == "cutting" else ()).install(monkeypatch)
        shrink(monkeypatch, DEADLINE_SECONDS, 2.0)
        late = 5.0
        if stage == "tool-request":
            script = lambda wire, body: Reply(text=ANSWER, delay=late)  # noqa: E731
        elif stage == "cutting":
            script = lambda wire, body: asks(12.0)  # noqa: E731
        elif stage == "fallback":
            script = lambda wire, body: (  # noqa: E731
                Reply(status=400) if offers_tool(wire, body)
                else Reply(text=TODAY_ANSWER, delay=late)
            )
        else:
            script = _serves_three(Reply(text=ANSWER, gap=late))
        wire = _wire(llm, "openai_compatible", script)

        started = time.monotonic()
        try:
            with caplog.at_level("DEBUG"):
                pairs = events(await ask(call, VIDEO, question=QUESTION))
        finally:
            ffmpeg.release_all.set()
        elapsed = time.monotonic() - started

        assert elapsed < late
        _ended_failed(pairs)
        assert caplog.text.count(DEADLINE_LOG) == 1
        expected_requests = {
            "tool-request": 1, "cutting": 1, "fallback": 2, "final-after-first-chunk": 4,
        }[stage]
        assert len(wire.requests) == expected_requests
        assert offers_tool(wire, wire.requests[0])
        if stage == "final-after-first-chunk":
            assert answer_text(pairs)
        else:
            assert "answer_chunk" not in kinds(pairs)
