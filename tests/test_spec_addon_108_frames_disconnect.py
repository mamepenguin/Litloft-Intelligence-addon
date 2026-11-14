"""SPEC-ADDON-108 I7/I8: hanging up on a frames answer, and waiting for its first token.

The provider is the addon's real client over ``httpx.MockTransport``. Its
stream stops after the tokens the case asks for and then waits, so the test can
hang up while the model is still inside it and read back whether it was torn
down.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx

from app.routers import rag as rag_router
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    DRIVE,
    VISION_MODEL,
    addon,
    ask_slots,
    db,
    llm,
    openai_config,
    resolved,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.video_frames_doubles import (
    FFmpeg,
    seed_text_less_video,
    timed_post,
)

VIDEO = "vid000000108"
QUESTION = "QUESTION-MARKER what happens here"
MAX = rag_router._MAX_CONCURRENT_ASK

KEEPALIVE = b": keepalive\n\n"


def _video(db, tmp_path):
    seed_text_less_video(
        db, VIDEO, path=tmp_path / f"{VIDEO}.mp4", duration=80.0,
    )


def _delta(text: str) -> bytes:
    chunk = {
        "id": "1", "object": "chat.completion.chunk", "created": 0,
        "model": VISION_MODEL,
        "choices": [{
            "index": 0, "delta": {"role": "assistant", "content": text},
            "finish_reason": None,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


def _serve(llm, stream, requests: list[dict]) -> None:
    """Hand the routed client ``stream`` as every vision response."""
    from openai import AsyncOpenAI

    from app.llm import LLMClient

    settings = openai_config()
    client = LLMClient(settings)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream,
        )

    client._client = AsyncOpenAI(  # noqa: SLF001 - the test replaces the transport
        base_url=settings.base_url,
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    llm.serve(resolved(client, settings))


class _HeldOpen(httpx.AsyncByteStream):
    """A provider stream that sends ``tokens`` deltas and then never ends."""

    def __init__(self, tokens: int = 0):
        self.tokens = tokens
        self.started = asyncio.Event()
        self.closed = False
        self.cancelled = False

    async def __aiter__(self):
        self.started.set()
        for index in range(self.tokens):
            yield _delta(f"TOKEN-{index}")
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def aclose(self) -> None:
        self.closed = True


class _FirstTokenLate(httpx.AsyncByteStream):
    """A provider stream whose first token arrives after ``delay`` seconds."""

    def __init__(self, delay: float):
        self.delay = delay

    async def __aiter__(self):
        await asyncio.sleep(self.delay)
        yield _delta("LATE-TOKEN")

    async def aclose(self) -> None:
        return None


async def _wait_until(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.02)


async def _hang_up_after(app, *, until, timeout: float = 10.0) -> list[dict]:
    """Drive the app over ASGI and send ``http.disconnect`` once ``until`` holds."""
    queue: asyncio.Queue[dict] = asyncio.Queue()
    await queue.put({
        "type": "http.request",
        "body": json.dumps({"question": QUESTION}).encode(),
        "more_body": False,
    })
    sent: list[dict] = []

    async def receive() -> dict:
        return await queue.get()

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": f"/files/{VIDEO}/ask",
        "raw_path": f"/files/{VIDEO}/ask".encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"x-lit-drive", DRIVE.encode()),
        ],
        "server": ("intelligence", 8100),
        "client": ("test", 1),
    }
    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(until(sent), timeout)
        await queue.put({"type": "http.disconnect"})
        try:
            await asyncio.wait_for(task, timeout)
        except asyncio.CancelledError:
            # Starlette cancels the response task on the disconnect; that is
            # the outcome the test is provoking.
            pass
    finally:
        if not task.done():
            task.cancel()
    return sent


def _body(message: dict) -> bytes:
    return message.get("body", b"")


class TestADisconnectWhileTheModelAnswers:
    async def test_hanging_up_after_the_first_chunk_ends_the_response_cleanly(
        self, db, addon, llm, ask_slots, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 I7: stopped, torn down, released, no error."""
        from app.main import app

        _video(db, tmp_path)
        FFmpeg().install(monkeypatch)
        stream = _HeldOpen(tokens=1)
        requests: list[dict] = []
        _serve(llm, stream, requests)

        sent = await _hang_up_after(
            app,
            until=lambda seen: _wait_until(
                lambda: any(b"event: answer_chunk" in _body(m) for m in seen),
            ),
        )

        assert any(b"event: answer_chunk" in _body(m) for m in sent)
        assert stream.closed or stream.cancelled
        assert ask_slots._value == MAX

    async def test_hanging_up_before_the_first_token_ends_the_response_cleanly(
        self, db, addon, llm, ask_slots, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 I7: a model already started is torn down on a disconnect."""
        from app.main import app

        _video(db, tmp_path)
        FFmpeg().install(monkeypatch)
        stream = _HeldOpen(tokens=0)
        requests: list[dict] = []
        _serve(llm, stream, requests)

        sent = await _hang_up_after(app, until=lambda _seen: stream.started.wait())

        assert requests
        assert not any(b"event: answer_chunk" in _body(m) for m in sent)
        assert stream.closed or stream.cancelled
        assert ask_slots._value == MAX


class TestTheKeepAliveWhileTheModelThinks:
    async def test_a_comment_is_written_while_waiting_for_the_first_token(
        self, db, addon, llm, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 I8: no 10 s of silence before the first chunk, either."""
        from openai import AsyncOpenAI

        from app.llm import LLMClient
        from app.main import app

        _video(db, tmp_path)
        FFmpeg().install(monkeypatch)
        monkeypatch.setattr(rag_router, "_KEEPALIVE_SECONDS", 0.05)
        requested_at: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested_at.append(time.monotonic())
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"},
                stream=_FirstTokenLate(delay=0.5),
            )

        settings = openai_config()
        client = LLMClient(settings)
        client._client = AsyncOpenAI(  # noqa: SLF001 - the test replaces the transport
            base_url=settings.base_url,
            api_key="test-key",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        llm.serve(resolved(client, settings))

        _started, received, status = await timed_post(
            app, path=f"/files/{VIDEO}/ask", body={"question": QUESTION},
        )

        body = b"".join(data for _, data in received).decode()
        assert status == 200
        chunks = [at for at, data in received if b"event: answer_chunk" in data]
        keepalives = [at for at, data in received if data == KEEPALIVE]
        assert "LATE-TOKEN" in body
        assert requested_at
        assert any(
            requested_at[0] <= at <= chunks[0] for at in keepalives
        ), (requested_at, keepalives, chunks)
