"""Tests for app.routers.rag (SSE streaming).

Covers the ``POST /ask`` endpoint:

* Feature + LLM + query-length gating still raise HTTPException
  *before* the stream opens, so clients get a normal JSON 4xx
  instead of an empty event stream.
* The happy path returns a ``StreamingResponse`` with the correct
  ``text/event-stream`` content type and well-formed SSE frames.
* The frame format matches the backend contract (``event: <kind>``
  + ``data: <json>`` + blank line terminator).
* Mid-stream exceptions are caught and converted to a final
  ``done`` event with an ``error`` field rather than crashing the
  ASGI task.

Handlers are invoked directly as async functions; no TestClient
required. We exhaust the ``StreamingResponse.body_iterator`` to
collect the emitted SSE frames and parse them.
"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.config import LLMConfig, RagConfig  # noqa: E402
from app.rag.service import AnswerEvent  # noqa: E402
from app.routers.rag import ask_endpoint  # noqa: E402
from app.schemas import AskRequest  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures (unchanged from the pre-streaming router tests)
# ---------------------------------------------------------------------------


@pytest.fixture()
def rag_enabled(monkeypatch, make_settings):
    settings = make_settings(
        features=type(make_settings().features)(rag=True),
        llm=LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
        ),
        rag=RagConfig(),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.rag.settings", settings)

    llm_stub = MagicMock()
    llm_stub.enabled = True
    monkeypatch.setattr(
        "app.routers.rag.get_llm_client", lambda: llm_stub
    )
    return settings


@pytest.fixture()
def rag_disabled(monkeypatch, make_settings):
    settings = make_settings(
        features=type(make_settings().features)(rag=False),
        llm=LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
        ),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.rag.settings", settings)

    llm_stub = MagicMock()
    llm_stub.enabled = True
    monkeypatch.setattr(
        "app.routers.rag.get_llm_client", lambda: llm_stub
    )
    return settings


@pytest.fixture()
def llm_disabled(monkeypatch, make_settings):
    settings = make_settings(
        features=type(make_settings().features)(rag=True),
        llm=LLMConfig(provider="disabled"),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.rag.settings", settings)

    llm_stub = MagicMock()
    llm_stub.enabled = False
    monkeypatch.setattr(
        "app.routers.rag.get_llm_client", lambda: llm_stub
    )
    return settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _collect_sse_body(response: StreamingResponse) -> list[str]:
    """Drain a StreamingResponse into a list of decoded SSE frames.

    StreamingResponse.body_iterator yields whatever the upstream
    generator produced — in our case, already-formatted SSE frames
    (strings). We concatenate, then split on the blank-line frame
    delimiter to recover individual frames in order.
    """
    buffer = ""
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            buffer += chunk.decode("utf-8")
        else:
            buffer += chunk
    # Split on the spec's blank-line delimiter. Empty trailing frame
    # is normal (the final "\n\n" produces one) so drop it.
    frames = [f for f in buffer.split("\n\n") if f.strip()]
    return frames


def _parse_sse_frame(frame: str) -> tuple[str, dict]:
    """Parse a single SSE frame into (event_name, data_dict)."""
    event_name = ""
    data_line = ""
    for line in frame.split("\n"):
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_line = line[len("data:"):].strip()
    data = json.loads(data_line) if data_line else {}
    return event_name, data


def _stream_factory(events: list[AnswerEvent]):
    """Return an async generator function yielding the given events.

    Used to stub ``stream_answer`` without pulling in the full
    retrieval + LLM pipeline. The factory shape matches the real
    signature so ``monkeypatch.setattr`` can drop it straight in.
    """
    async def _gen(**kwargs):
        for event in events:
            yield event

    return _gen


# ---------------------------------------------------------------------------
# Feature / LLM / query gating
# ---------------------------------------------------------------------------


class TestAskEndpointGating:
    """Gating runs before the stream opens and raises HTTPException."""

    @pytest.mark.asyncio
    async def test_returns_400_when_rag_disabled(self, rag_disabled):
        body = AskRequest(query="a valid question")

        with pytest.raises(HTTPException) as exc_info:
            await ask_endpoint(body=body, access_token="tok")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_returns_400_when_llm_disabled(self, llm_disabled):
        body = AskRequest(query="a valid question")

        with pytest.raises(HTTPException) as exc_info:
            await ask_endpoint(body=body, access_token="tok")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_returns_400_for_short_query(self, rag_enabled):
        body = AskRequest(query="ab")

        with pytest.raises(HTTPException) as exc_info:
            await ask_endpoint(body=body, access_token="tok")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_returns_400_for_whitespace_only_query(self, rag_enabled):
        body = AskRequest(query="   ")

        with pytest.raises(HTTPException) as exc_info:
            await ask_endpoint(body=body, access_token="tok")

        assert exc_info.value.status_code == 400

    def test_query_over_1000_chars_rejected_by_pydantic(self):
        with pytest.raises(Exception):
            AskRequest(query="x" * 1001)


# ---------------------------------------------------------------------------
# Happy path: StreamingResponse shape + SSE frames
# ---------------------------------------------------------------------------


class TestAskEndpointStreaming:
    @pytest.mark.asyncio
    async def test_returns_streaming_response(self, rag_enabled, monkeypatch):
        monkeypatch.setattr(
            "app.routers.rag.stream_answer",
            _stream_factory([
                AnswerEvent(kind="keywords", data={"keywords": "kw"}),
                AnswerEvent(kind="sources", data={"sources": []}),
                AnswerEvent(kind="answer_chunk", data={"delta": "hi"}),
                AnswerEvent(kind="citations", data={"citations": []}),
                AnswerEvent(kind="done", data={}),
            ]),
        )

        body = AskRequest(query="what is the topic")
        result = await ask_endpoint(body=body, access_token="tok")

        assert isinstance(result, StreamingResponse)
        assert result.media_type == "text/event-stream"
        # The SSE hygiene headers must be set so reverse proxies do
        # not buffer the response.
        assert result.headers.get("cache-control") == "no-cache"
        assert result.headers.get("x-accel-buffering") == "no"
        # Drain the body so the generator's finally block runs, which
        # releases the in-flight concurrency-cap slot. Without this a
        # follow-up test that shares the event loop would see a
        # permanently leaked slot.
        await _collect_sse_body(result)

    @pytest.mark.asyncio
    async def test_sse_frame_format(self, rag_enabled, monkeypatch):
        """Each emitted frame must be parseable as a standard SSE event."""
        monkeypatch.setattr(
            "app.routers.rag.stream_answer",
            _stream_factory([
                AnswerEvent(kind="keywords", data={"keywords": "京都 紅葉"}),
                AnswerEvent(
                    kind="sources",
                    data={"sources": [{"file_id": "f1", "filename": "a.mp4"}]},
                ),
                AnswerEvent(kind="answer_chunk", data={"delta": "京都の"}),
                AnswerEvent(kind="answer_chunk", data={"delta": "紅葉は"}),
                AnswerEvent(kind="citations", data={"citations": []}),
                AnswerEvent(
                    kind="done",
                    data={"retrieved_count": 1, "took_ms": 42},
                ),
            ]),
        )

        body = AskRequest(query="京都の紅葉について")
        result = await ask_endpoint(body=body, access_token="tok")

        frames = await _collect_sse_body(result)
        parsed = [_parse_sse_frame(f) for f in frames]
        kinds = [name for name, _ in parsed]

        assert kinds == [
            "keywords",
            "sources",
            "answer_chunk",
            "answer_chunk",
            "citations",
            "done",
        ]

        # Each event's data survives the JSON round-trip.
        assert parsed[0][1] == {"keywords": "京都 紅葉"}
        assert parsed[2][1] == {"delta": "京都の"}
        assert parsed[3][1] == {"delta": "紅葉は"}
        assert parsed[5][1]["retrieved_count"] == 1
        assert parsed[5][1]["took_ms"] == 42

    @pytest.mark.asyncio
    async def test_mid_stream_exception_emits_done_with_error(
        self, rag_enabled, monkeypatch
    ):
        """An exception raised mid-generator must not crash the response."""

        async def _broken_stream(**kwargs):
            yield AnswerEvent(kind="keywords", data={"keywords": "kw"})
            raise RuntimeError("transport exploded")

        monkeypatch.setattr(
            "app.routers.rag.stream_answer", _broken_stream
        )

        body = AskRequest(query="will this blow up?")
        result = await ask_endpoint(body=body, access_token="tok")

        frames = await _collect_sse_body(result)
        parsed = [_parse_sse_frame(f) for f in frames]
        kinds = [name for name, _ in parsed]

        # The partial stream up to the crash is preserved, then a
        # terminal ``done`` frame carries an ``error`` field.
        assert "keywords" in kinds
        assert kinds[-1] == "done"
        assert parsed[-1][1].get("error") == "Answer generation failed"

    @pytest.mark.asyncio
    async def test_forwards_access_token_to_service(
        self, rag_enabled, monkeypatch
    ):
        captured: dict = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            yield AnswerEvent(kind="done", data={})

        monkeypatch.setattr("app.routers.rag.stream_answer", _capture)

        body = AskRequest(query="a valid query")
        result = await ask_endpoint(
            body=body, access_token="my-secret-token"
        )
        # Drain the body so the generator actually runs.
        await _collect_sse_body(result)

        assert captured.get("hv_token") == "my-secret-token"

    @pytest.mark.asyncio
    async def test_forwards_optional_params(
        self, rag_enabled, monkeypatch
    ):
        captured: dict = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            yield AnswerEvent(kind="done", data={})

        monkeypatch.setattr("app.routers.rag.stream_answer", _capture)

        body = AskRequest(
            query="a valid query",
            top_k=7,
            file_type="video",
            drive="Videos",
        )
        # X-HV-Drive is the source of truth; body.drive must agree. The
        # router resolves the header via Depends, so when calling the
        # handler directly we pass ``drive`` explicitly.
        result = await ask_endpoint(
            body=body, access_token=None, drive="Videos",
        )
        await _collect_sse_body(result)

        assert captured.get("top_k") == 7
        assert captured.get("file_type") == "video"
        assert captured.get("drive") == "Videos"
        assert captured.get("hv_token") is None


# ---------------------------------------------------------------------------
# Concurrency cap (rate limiting)
# ---------------------------------------------------------------------------


class TestAskEndpointConcurrencyCap:
    """/ask must return 503 when the in-flight concurrency slot pool is full.

    The semaphore is a process-global guard against cloud-LLM cost
    blowouts and local-LLM resource exhaustion. This test exercises the
    rejection path by forcibly exhausting the semaphore before issuing
    a request, without relying on racing two real coroutines (which is
    flaky under pytest-asyncio's deterministic event loop scheduler).
    """

    @pytest.mark.asyncio
    async def test_returns_503_when_concurrency_cap_reached(
        self, rag_enabled, monkeypatch
    ):
        import asyncio as _asyncio

        from app.routers import rag as rag_module

        # Reset the lazy semaphore so we know its exact capacity, then
        # drain every slot to simulate a full pool.
        rag_module._ask_semaphore = None
        rag_module._ask_semaphore_loop = None
        sem = rag_module._get_ask_semaphore()
        for _ in range(rag_module._MAX_CONCURRENT_ASK):
            await sem.acquire()

        try:
            body = AskRequest(query="a valid question")
            with pytest.raises(HTTPException) as exc_info:
                await ask_endpoint(body=body, access_token="tok")

            assert exc_info.value.status_code == 503
            assert "concurrent" in str(exc_info.value.detail).lower()
        finally:
            for _ in range(rag_module._MAX_CONCURRENT_ASK):
                sem.release()

    @pytest.mark.asyncio
    async def test_releases_slot_after_normal_stream(
        self, rag_enabled, monkeypatch
    ):
        """A successful stream must release its slot in finally."""
        from app.routers import rag as rag_module

        rag_module._ask_semaphore = None
        rag_module._ask_semaphore_loop = None

        monkeypatch.setattr(
            "app.routers.rag.stream_answer",
            _stream_factory([AnswerEvent(kind="done", data={})]),
        )

        body = AskRequest(query="a valid question")

        # Run the full pool size worth of requests serially; if the
        # slot is not released, the 4th would 503.
        for _ in range(rag_module._MAX_CONCURRENT_ASK + 1):
            result = await ask_endpoint(body=body, access_token="tok")
            await _collect_sse_body(result)

        sem = rag_module._get_ask_semaphore()
        # All slots should be back in the pool.
        assert sem._value == rag_module._MAX_CONCURRENT_ASK


# ---------------------------------------------------------------------------
# AskRequest schema (unchanged, but kept local to avoid cross-file drift)
# ---------------------------------------------------------------------------


class TestAskRequestSchema:
    def test_defaults(self):
        body = AskRequest(query="valid query")
        assert body.query == "valid query"
        assert body.top_k is None
        assert body.file_type is None
        assert body.drive is None

    def test_top_k_must_be_in_range(self):
        with pytest.raises(Exception):
            AskRequest(query="q", top_k=0)
        with pytest.raises(Exception):
            AskRequest(query="q", top_k=21)

    def test_empty_query_rejected(self):
        with pytest.raises(Exception):
            AskRequest(query="")
