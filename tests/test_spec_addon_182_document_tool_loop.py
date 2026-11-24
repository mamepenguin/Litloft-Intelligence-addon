"""SPEC-ADDON-182: the document tool loop, its limits, its fallback and its log lines.

The 120 s window, the 300 s deadline and the keep-alive interval are shrunk
through the frames path's module constants, which the document path shares.
Today's text request is read from a run with a profile that cannot see.
"""

from __future__ import annotations

import time

import httpx
import pytest

from tests.document_tool_doubles import (
    ANSWER,
    DEADLINE_SECONDS,
    EPUB_BUDGET,
    EPUB_ID,
    FAILED,
    GET_TEXT_BAD_PDF,
    KEEPALIVE_SECONDS,
    OVER_THREE,
    PDF_BUDGET,
    PDF_ID,
    PDF_TEXT,
    PROVIDERS,
    QUESTION,
    TODAY_ANSWER,
    WINDOW_SECONDS,
    answer_text,
    calls,
    cannot_serve,
    cited,
    get_page_images,
    get_text,
    images_of,
    is_final,
    is_today,
    kinds,
    offers_tool,
    page_images_head,
    plays,
    results,
    seed_epub,
    seed_pdf,
    todays_body,
    tool_names,
    wire_for,
)
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    VISION_MODEL,
    Reply,
    ToolCall,
    addon,
    ask,
    call,
    db,
    events,
    llm,
)
from tests.frame_request_doubles import assistant_messages, shrink
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture

FALLBACK_LOG = f"document_tool_fallback (file {PDF_ID})"
DEADLINE_LOG = f"document_tool_deadline (file {PDF_ID})"
EMPTY_LOG = f"file question failed: EmptyAnswer (file {PDF_ID})"
TOOLS_FAILED = "file question document tools failed: "
BOTH = ["get_text", "get_page_images"]


def _pdf(db, addon, tmp_path, monkeypatch, budget=PDF_BUDGET):
    addon(budget=budget)
    seed_pdf(db, tmp_path, monkeypatch)


def _events(pairs) -> list[tuple[str, dict]]:
    return [(kind, data) for kind, data in pairs if kind]


def _verdicts() -> dict:
    from app.llm import _vision_capability_cache

    return dict(_vision_capability_cache._verdicts)


def _failing(failure):
    if failure == "timeout":
        return httpx.ReadTimeout("timed out")
    if failure == "empty":
        return Reply(text="")
    if failure == "blank":
        return Reply(text="  \n ")
    if failure == "only-a-call":
        return calls(get_text(page=1))
    return Reply(status=failure)


def _serves_three(final, *, today=None):
    """Three served tool requests, ``final`` for the final request, ``today`` for today's."""
    queue = [
        calls(get_text(page=1)), calls(get_page_images(pages=[1])), calls(get_text(page=5)),
    ]

    def script(wire, body):
        if offers_tool(wire, body):
            return queue.pop(0) if queue else calls(get_text(page=2))
        if is_final(wire, body):
            return final
        return today if today is not None else Reply(text=TODAY_ANSWER)

    return script


class TestTheLoop:
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_a_reply_with_no_call_is_the_answer_in_one_chunk(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.6 I2: one ``answer_chunk``, then citations, meta and done."""
        _pdf(db, addon, tmp_path, monkeypatch)
        wire = wire_for(llm, provider, plays(then=Reply(text=f"{ANSWER} [1].", gap=0.05)))

        pairs = _events(events(await ask(call, PDF_ID, question=QUESTION)))

        assert len(wire.requests) == 1
        assert offers_tool(wire, wire.requests[0])
        assert kinds(pairs) == ["answer_chunk", "citations", "meta", "done"]
        assert answer_text(pairs) == f"{ANSWER} [1]."
        assert cited(pairs) == [(1, {"page": 2})]
        assert dict(pairs)["meta"] == {"partial": True, "image": "none"}
        assert dict(pairs)["done"] == {}

    async def test_a_tool_requests_reply_is_written_only_once_complete(
        self, db, addon, llm, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.6 I2: no text of a tool reply before its last byte."""
        from app.main import app
        from tests.video_frames_doubles import timed_post

        _pdf(db, addon, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(then=Reply(text=ANSWER, gap=0.5)))

        _started, received, status = await timed_post(
            app, path=f"/files/{PDF_ID}/ask", body={"question": QUESTION},
        )

        assert status == 200
        assert offers_tool(wire, wire.requests[0])
        first = next(at for at, data in received if b"event: answer_chunk" in data)
        assert first >= wire.yielded[-1][0]

    async def test_the_final_requests_answer_streams_as_it_arrives(
        self, db, addon, llm, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.6: the final request's first chunk precedes its last byte."""
        from app.main import app
        from tests.video_frames_doubles import timed_post

        _pdf(db, addon, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", _serves_three(Reply(text=ANSWER, gap=0.5)))

        _started, received, status = await timed_post(
            app, path=f"/files/{PDF_ID}/ask", body={"question": QUESTION},
        )

        assert status == 200
        assert len(wire.requests) == 4
        assert is_final(wire, wire.requests[3])
        first = next(at for at, data in received if b"event: answer_chunk" in data)
        assert first < wire.yielded[-1][0]

    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_after_three_served_calls_the_next_request_is_final(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.6 I2: same tools with ``tool_choice`` none, or none at all (ollama)."""
        _pdf(db, addon, tmp_path, monkeypatch)
        wire = wire_for(llm, provider, _serves_three(Reply(text=f"{ANSWER} [4].")))

        pairs = events(await ask(call, PDF_ID, question=QUESTION))

        assert len(wire.requests) == 4
        assert [offers_tool(wire, b) for b in wire.requests] == [True, True, True, False]
        final = wire.requests[3]
        assert is_final(wire, final)
        assert final["model"] == VISION_MODEL
        if provider == "openai_compatible":
            assert final["tool_choice"] == "none"
            assert tool_names(final) == BOTH
        else:
            assert not final.get("tools")
            assert final.get("think") is True
            assert final["options"]["num_predict"] == 8192
        assert len(assistant_messages(final)) == 3
        assert len(results(final)) == 3
        assert answer_text(pairs) == f"{ANSWER} [4]."
        assert cited(pairs) == [(4, {"page": 1})]
        assert dict(pairs)["meta"] == {"partial": True, "image": "attached"}

    async def test_calls_past_the_third_get_nothing_and_spend_nothing(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.7 I2: four calls in one reply, three served in order."""
        _pdf(db, addon, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(calls(
            get_text(page=5), get_page_images(pages=[1]), get_text(page=0),
            get_page_images(pages=[2]),
        )))

        await ask(call, PDF_ID, question=QUESTION)

        assert len(wire.requests) == 2
        second = wire.requests[1]
        assert is_final(wire, second)
        assert results(second) == [
            f"Page 5 begins.\n[4]\n{PDF_TEXT[8]}\nThe file ends here.",
            page_images_head(1),
            GET_TEXT_BAD_PDF.format(n=5),
            OVER_THREE,
        ]
        assert images_of(wire, second) == [("[5] page 1", 1)]

    @pytest.mark.parametrize(
        ("file", "name", "offered"),
        [
            ("pdf-cut", "get_frames", BOTH),
            ("pdf-cut", "", BOTH),
            ("pdf-not-cut", "get_text", ["get_page_images"]),
            ("epub-cut", "get_page_images", ["get_text"]),
        ],
        ids=["another-tool", "no-name", "text-on-a-whole-pdf", "pages-on-an-epub"],
    )
    async def test_a_call_of_a_tool_not_offered_is_answered_and_spends_one(
        self, file, name, offered, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.7: ``Nothing follows: <name> cannot be served. Only ... are available.``"""
        if file == "epub-cut":
            addon(budget=EPUB_BUDGET)
            seed_epub(db, tmp_path, monkeypatch)
            file_id = EPUB_ID
        else:
            _pdf(db, addon, tmp_path, monkeypatch, budget=PDF_BUDGET if file == "pdf-cut" else 16000)
            file_id = PDF_ID
        bad = ToolCall({"pages": [1], "page": 1}, name=name)
        wire = wire_for(llm, "openai_compatible", plays(
            calls(bad), calls(bad), calls(bad),
        ))

        await ask(call, file_id, question=QUESTION)

        assert len(wire.requests) == 4
        assert tool_names(wire.requests[0]) == offered
        assert is_final(wire, wire.requests[3])
        assert results(wire.requests[3]) == [cannot_serve(name, offered)] * 3
        assert images_of(wire, wire.requests[3]) == []

    async def test_bad_arguments_spend_a_call(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.8.1: the call is still served."""
        _pdf(db, addon, tmp_path, monkeypatch)
        bad = calls(get_text(page=0))
        wire = wire_for(llm, "openai_compatible", plays(bad, bad, bad))

        await ask(call, PDF_ID, question=QUESTION)

        assert len(wire.requests) == 4
        assert is_final(wire, wire.requests[3])
        assert results(wire.requests[3]) == [GET_TEXT_BAD_PDF.format(n=5)] * 3

    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_once_the_window_has_passed_the_next_request_is_final(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.6 I2: 120 s from the slot (shrunk), then the final request."""
        _pdf(db, addon, tmp_path, monkeypatch)
        shrink(monkeypatch, WINDOW_SECONDS, 1.0)
        slow = Reply(calls=(get_text(page=5),), delay=1.5)
        wire = wire_for(llm, provider, plays(slow, calls(get_text(page=1))))

        pairs = events(await ask(call, PDF_ID, question=QUESTION))

        assert len(wire.requests) == 2
        assert offers_tool(wire, wire.requests[0])
        assert is_final(wire, wire.requests[1])
        assert len(results(wire.requests[1])) == 1
        assert answer_text(pairs) == ANSWER

    async def test_a_follow_up_question_starts_again(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §3 (nothing kept between questions): numbers and sent pages start over."""
        _pdf(db, addon, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(
            calls(get_page_images(pages=[1])), Reply(text=ANSWER),
            calls(get_page_images(pages=[1])), Reply(text=ANSWER),
        ))

        await ask(call, PDF_ID, question=QUESTION)
        await ask(call, PDF_ID, question=QUESTION)

        assert len(wire.requests) == 4
        for body in (wire.requests[1], wire.requests[3]):
            assert results(body) == [page_images_head(1)]
            assert images_of(wire, body) == [("[4] page 1", 1)]


class TestTheFallback:
    @pytest.mark.parametrize(
        "failure", [400, 404, 500, 503, "timeout", "empty", "blank"], ids=str,
    )
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_a_failed_tool_request_is_followed_by_todays_request_once(
        self, provider, failure, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-182 §2 I6: exactly today's request; no verdict; meta image none."""
        _pdf(db, addon, tmp_path, monkeypatch)
        today = await todays_body(llm, call, provider, PDF_ID)
        wire = wire_for(llm, provider, lambda wire, body: (
            _failing(failure) if offers_tool(wire, body) else Reply(text=TODAY_ANSWER)
        ))

        with caplog.at_level("DEBUG"):
            pairs = events(await ask(call, PDF_ID, question=QUESTION))

        assert len(wire.requests) == 2
        assert offers_tool(wire, wire.requests[0])
        assert wire.requests[1] == today
        assert answer_text(pairs) == TODAY_ANSWER
        assert dict(pairs)["meta"] == {"partial": True, "image": "none"}
        assert caplog.text.count(FALLBACK_LOG) == 1
        if failure in ("empty", "blank"):
            assert caplog.text.count(EMPTY_LOG) == 1
        elif failure in (500, 503, "timeout"):
            failed = [line for line in caplog.text.splitlines() if TOOLS_FAILED in line]
            assert len(failed) == 1 and failed[0].endswith(f"(file {PDF_ID})")
        assert _verdicts() == {}

    @pytest.mark.parametrize(
        "failure", [400, 500, "timeout", "empty", "blank", "only-a-call"], ids=str,
    )
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_a_final_request_that_leaves_no_answer_falls_back_without_requested_material(
        self, provider, failure, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-182 §2 I6 I4: today's request with no requested passage or image; a taken page makes it unavailable."""
        _pdf(db, addon, tmp_path, monkeypatch)
        today = await todays_body(llm, call, provider, PDF_ID)
        wire = wire_for(llm, provider, _serves_three(_failing(failure)))

        with caplog.at_level("DEBUG"):
            pairs = events(await ask(call, PDF_ID, question=QUESTION))

        assert len(wire.requests) == 5
        assert is_final(wire, wire.requests[3])
        assert wire.requests[4] == today
        assert answer_text(pairs) == TODAY_ANSWER
        assert dict(pairs)["meta"] == {"partial": True, "image": "unavailable"}
        assert all(n <= 3 for n, _position in cited(pairs))
        assert caplog.text.count(FALLBACK_LOG) == 1
        if failure in ("empty", "blank", "only-a-call"):
            assert caplog.text.count(EMPTY_LOG) == 1
        assert _verdicts() == {}

    async def test_a_fallback_with_no_page_taken_reports_no_image(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.11: ``none`` when no page was taken before the fallback."""
        _pdf(db, addon, tmp_path, monkeypatch)
        queue = [calls(get_text(page=1))]
        wire = wire_for(llm, "openai_compatible", lambda wire, body: (
            (queue.pop(0) if queue else Reply(status=500)) if offers_tool(wire, body)
            else Reply(text=TODAY_ANSWER) if is_today(wire, body)
            else Reply(status=500)
        ))

        pairs = events(await ask(call, PDF_ID, question=QUESTION))

        assert len(wire.requests) == 3
        assert offers_tool(wire, wire.requests[1])
        assert is_today(wire, wire.requests[2])
        assert dict(pairs)["meta"] == {"partial": True, "image": "none"}

    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_at_most_five_model_requests_are_sent(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §5 §3: three tool requests, one final, one fallback; a failed fallback ends it."""
        _pdf(db, addon, tmp_path, monkeypatch)
        wire = wire_for(llm, provider, _serves_three(Reply(status=500), today=Reply(status=500)))

        pairs = events(await ask(call, PDF_ID, question=QUESTION))

        assert len(wire.requests) == 5
        assert [offers_tool(wire, b) for b in wire.requests] == [True] * 3 + [False] * 2
        assert is_today(wire, wire.requests[4])
        assert pairs[-1][0] == "done" and pairs[-1][1].get("error")

    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_a_final_request_that_fails_after_its_first_chunk_ends_failed(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-182 §2 §3 (final → failed) §8: meta, then the failed done; no fallback."""
        _pdf(db, addon, tmp_path, monkeypatch)
        wire = wire_for(llm, provider, _serves_three(Reply(text="PARTIAL-MARKER", cut=True)))

        with caplog.at_level("DEBUG"):
            pairs = _events(events(await ask(call, PDF_ID, question=QUESTION)))

        assert len(wire.requests) == 4
        assert is_final(wire, wire.requests[3])
        assert "PARTIAL-MARKER" in answer_text(pairs)
        assert kinds(pairs)[-2:] == ["meta", "done"]
        assert dict(pairs)["meta"] == {"partial": True, "image": "attached"}
        assert pairs[-1][1] == FAILED
        failed = [
            line for line in caplog.text.splitlines()
            if "file question failed: " in line and "EmptyAnswer" not in line
        ]
        assert len(failed) == 1 and failed[0].endswith(f"(file {PDF_ID})")
        assert FALLBACK_LOG not in caplog.text


class TestTheDeadline:
    @pytest.mark.parametrize("stage", ["tool-request", "fallback", "final-after-first-chunk"])
    async def test_it_stops_what_is_in_flight_and_ends_the_answer_failed(
        self, stage, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-182 §2 §3 I2: nothing after 300 s from the slot (shrunk); meta, failed done."""
        _pdf(db, addon, tmp_path, monkeypatch)
        shrink(monkeypatch, DEADLINE_SECONDS, 2.0)
        late = 5.0
        if stage == "tool-request":
            script = lambda wire, body: Reply(text=ANSWER, delay=late)  # noqa: E731
            image = "none"
        elif stage == "fallback":
            script = lambda wire, body: (  # noqa: E731
                Reply(status=400) if offers_tool(wire, body)
                else Reply(text=TODAY_ANSWER, delay=late)
            )
            image = "none"
        else:
            script = _serves_three(Reply(text=ANSWER, gap=late))
            image = "attached"
        wire = wire_for(llm, "openai_compatible", script)

        started = time.monotonic()
        with caplog.at_level("DEBUG"):
            pairs = _events(events(await ask(call, PDF_ID, question=QUESTION)))
        elapsed = time.monotonic() - started

        assert offers_tool(wire, wire.requests[0])
        assert elapsed < late
        assert kinds(pairs)[-2:] == ["meta", "done"]
        assert dict(pairs)["meta"] == {"partial": True, "image": image}
        assert pairs[-1][1] == FAILED
        assert caplog.text.count(DEADLINE_LOG) == 1
        assert len(wire.requests) == {
            "tool-request": 1, "fallback": 2, "final-after-first-chunk": 4,
        }[stage]


class TestKeepAlive:
    @pytest.mark.parametrize("stage", ["tool-request", "fallback"])
    async def test_the_line_is_kept_alive_until_the_first_answer_chunk(
        self, stage, db, addon, llm, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.6: ``: keepalive`` while a tool reply or the fallback is awaited."""
        from app.main import app
        from app.routers import rag as rag_router
        from tests.video_frames_doubles import timed_post

        _pdf(db, addon, tmp_path, monkeypatch)
        monkeypatch.setattr(rag_router, KEEPALIVE_SECONDS, 0.05)
        sent_at: list[float] = []

        def script(wire, body):
            sent_at.append(time.monotonic())
            if stage == "tool-request":
                return Reply(text=ANSWER, delay=0.5)
            if offers_tool(wire, body):
                return Reply(status=400)
            return Reply(text=TODAY_ANSWER, delay=0.5)

        wire = wire_for(llm, "openai_compatible", script)

        _started, received, status = await timed_post(
            app, path=f"/files/{PDF_ID}/ask", body={"question": QUESTION},
        )

        assert status == 200
        assert offers_tool(wire, wire.requests[0])
        first = next(at for at, data in received if b"event: answer_chunk" in data)
        keepalives = [at for at, data in received if data == b": keepalive\n\n"]
        assert any(sent_at[-1] < at < first for at in keepalives)


class TestTheLog:
    async def test_one_summary_line_counts_served_calls_by_name_and_what_they_sent(
        self, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-182 §8 I9: bad arguments count as calls; nothing private is logged."""
        _pdf(db, addon, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(
            calls(get_text(page=1), get_page_images(pages=[1, 2])),
            calls(get_text(page="ARG-MARKER"), ToolCall({}, name="get_frames")),
            then=Reply(text=f"{ANSWER} [4]."),
        ))

        with caplog.at_level("DEBUG"):
            await ask(call, PDF_ID, question=QUESTION)

        assert len(wire.requests) == 3
        assert offers_tool(wire, wire.requests[0])
        line = f"document tool requests: 2 text calls, 6 passages, 1 page calls, 2 pages (file {PDF_ID})"
        assert caplog.text.count(line) == 1
        assert caplog.text.count("document tool requests:") == 1
        for private in (QUESTION, ANSWER, "ARG-MARKER", "PDF-ROW-", "/9j/"):
            assert private not in caplog.text

    async def test_a_question_that_served_no_call_logs_no_summary(
        self, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-182 §8: the summary is for a question that served a tool call."""
        _pdf(db, addon, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(then=Reply(text=ANSWER)))

        with caplog.at_level("DEBUG"):
            await ask(call, PDF_ID, question=QUESTION)

        assert offers_tool(wire, wire.requests[0])
        assert "document tool requests:" not in caplog.text
