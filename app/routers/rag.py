"""RAG (question answering) endpoint.

Single public route: ``POST /ask``. Responds with
``text/event-stream`` (Server-Sent Events) so the frontend can render
the answer as it's generated. The heavy lifting lives in
``app.rag.service.stream_answer``; this module exists purely to:

1. Enforce feature + LLM + query-length gating.
2. Adapt the service's ``AnswerEvent`` stream to the SSE wire format.
3. Fail closed on misconfiguration (feature disabled / LLM disabled)
   with a normal JSON 4xx rather than an empty SSE stream.

Gating layers (all must pass):

1. ``features.rag`` must be True (config toggle).
2. ``LLMClient.enabled`` must be True (provider configured).
3. ``body.query.strip()`` must be at least 3 characters.

The Pydantic ``AskRequest`` model already enforces ``1 <= len(query)
<= 1000`` and ``1 <= top_k <= 20``; the strict >=3 post-strip check
lives here so it runs after whitespace normalization.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.dependencies import get_llm_client
from app.drive_context import require_drive
from app.rag.service import (
    AnswerEvent,
    find_files,
    stream_answer,
)
from app.schemas import AskRequest, FindRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])


# Media type for SSE. Kept as a module constant so tests can assert
# on it without re-typing the string literal.
_SSE_MEDIA_TYPE = "text/event-stream"


# Concurrency cap for in-flight /ask requests. LLM answer generation is
# the most expensive operation the addon performs (wall time + tokens +,
# if a cloud provider is configured, real money), so we bound the fan-out
# at a small number. 3 concurrent streams is plenty for a single-family
# LAN deployment and keeps the failure mode graceful: the 4th caller gets
# a 503 "busy, retry" instead of piling onto an overloaded upstream.
#
# The semaphore is lazily instantiated inside the endpoint so it binds
# to the running event loop (module-level construction would bind to the
# import-time loop, which breaks under pytest-asyncio's fresh-loop-per-
# test fixture). ``None`` here is the sentinel for "not yet created".
_MAX_CONCURRENT_ASK = 3
_ask_semaphore: "asyncio.Semaphore | None" = None
_ask_semaphore_loop: "asyncio.AbstractEventLoop | None" = None


def _get_ask_semaphore() -> "asyncio.Semaphore":
    """Return the process-global /ask concurrency semaphore.

    Lazy initialization keeps the object bound to the event loop that
    actually serves requests rather than whatever loop happened to be
    active at import time. If the running loop changes (pytest-asyncio
    tears down and rebuilds its loop between tests), we recreate the
    semaphore to avoid "attached to a different loop" runtime errors.
    """
    global _ask_semaphore, _ask_semaphore_loop
    current_loop = asyncio.get_running_loop()
    if _ask_semaphore is None or _ask_semaphore_loop is not current_loop:
        _ask_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_ASK)
        _ask_semaphore_loop = current_loop
    return _ask_semaphore


# HTTP headers that keep SSE responses flowing through reverse proxies.
# ``X-Accel-Buffering: no`` disables nginx's default 8KB write buffer,
# without which clients see a long stall followed by a flood of
# already-stale tokens. ``Cache-Control: no-cache`` is part of the SSE
# spec and prevents CDNs / browsers from caching partial streams.
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def _require_rag_enabled() -> None:
    """Raise 4xx/5xx if the RAG feature or LLM is not available.

    * 400 when the feature is explicitly disabled in config.
    * 400 when the LLM provider is configured as "disabled".
    * 503 when the dependency container isn't initialized yet
      (startup race: the router is mounted before the lifespan
      handler finishes). This is a transient condition, not a
      permanent misconfiguration, so the distinct status code
      makes it easier to diagnose in logs vs. an everything-500.
    """
    if not settings.features.rag:
        raise HTTPException(
            status_code=400, detail="RAG feature is disabled"
        )
    try:
        client = get_llm_client()
    except RuntimeError:
        # Startup race: dependency injection not yet populated.
        raise HTTPException(
            status_code=503, detail="LLM client not initialized yet"
        )
    if not client.enabled:
        raise HTTPException(
            status_code=400, detail="LLM is not enabled"
        )


def _format_sse_event(event: AnswerEvent) -> str:
    """Render an ``AnswerEvent`` as a Server-Sent Events frame.

    The SSE wire format is:

        event: <kind>\\n
        data: <json>\\n
        \\n

    The blank line is the record terminator — without it the browser
    EventSource parser buffers the frame indefinitely. We JSON-encode
    the data payload even when it's an empty dict so clients can
    unconditionally ``JSON.parse`` the ``data:`` line instead of
    handling "maybe JSON, maybe blank" edge cases.

    Multi-line data would need per-line ``data:`` prefixes to comply
    with the spec, but JSON is always emitted on a single line by
    ``json.dumps`` with ``ensure_ascii=False`` so this is a non-issue.
    """
    payload = json.dumps(event.data, ensure_ascii=False)
    return f"event: {event.kind}\ndata: {payload}\n\n"


async def _sse_stream(
    query: str,
    access_token: str | None,
    top_k: int | None,
    file_type: str | None,
    drive: str | None,
    viewer_id: str | None,
    semaphore: "asyncio.Semaphore",
) -> AsyncIterator[str]:
    """Adapt ``stream_answer`` to the text/event-stream wire format.

    Wraps the entire generator in a try/except so any exception raised
    *mid-stream* is converted to a final ``event: error`` frame and
    then a ``done`` marker, rather than killing the ASGI task
    mid-response. A partial stream with an error frame is better UX
    than a hung connection: the client can show "generation failed"
    and offer a retry button without timing out.

    Owns the lifetime of the concurrency-cap semaphore slot: releases
    it in a finally block so client disconnects (CancelledError) and
    mid-stream failures both return the slot to the pool. The caller
    must have already acquired the slot before the generator starts.
    """
    try:
        async for event in stream_answer(
            query=query,
            lit_token=access_token,
            top_k=top_k,
            file_type=file_type,
            drive=drive,
            viewer_id=viewer_id,
        ):
            yield _format_sse_event(event)
    except asyncio.CancelledError:
        # Client disconnect or task cancellation. Re-raise so the
        # underlying async generators in stream_answer run their
        # finally blocks (closing upstream LLM streams, releasing the
        # rate-limit slot, etc). Swallowing CancelledError here would
        # starve those cleanup paths and leave the LLM connection
        # dangling until GC.
        raise
    except Exception as e:
        logger.error("RAG stream failed mid-response: %s", type(e).__name__)
        error_event = AnswerEvent(
            kind="done",
            data={"error": "Answer generation failed"},
        )
        yield _format_sse_event(error_event)
    finally:
        # Always release the slot, even on CancelledError. This runs
        # during generator finalization which Starlette triggers on
        # both normal end-of-stream and client disconnect.
        semaphore.release()


@router.post("/ask")
async def ask_endpoint(
    body: AskRequest,
    access_token: Annotated[str | None, Cookie()] = None,
    drive: str = Depends(require_drive),
    viewer_id: Annotated[
        str | None, Header(alias="X-Lit-Viewer-Id")
    ] = None,
) -> StreamingResponse:
    """Answer a natural-language question using retrieval-augmented generation.

    Returns a ``text/event-stream`` response with the following events:

    * ``keywords`` — the LLM-extracted keyword string used for search.
    * ``sources`` — the access-filtered list of retrieved files.
    * ``answer_chunk`` — one per LLM token chunk (may be many).
    * ``citations`` — final anti-hallucination-filtered citation list.
    * ``done`` — terminal marker. On mid-stream failure this event
      carries an ``error`` field.

    Security notes:

    * The caller's ``access_token`` cookie is forwarded to the retriever
      so drive access control runs BEFORE file content reaches the LLM.
      The host's Generic Addon Proxy passes browser cookies through to
      the intelligence service verbatim (see
      ``backend/app/routers/addon_proxy.py``), so we read the cookie
      directly from the request. The parameter name matches the cookie
      key used by ``get_unlocked_groups`` in ``backend/app/auth.py``.
    * The service layer drops citations referencing file_ids that were
      not in the retrieved set (anti-hallucination).
    * Query length is clamped to 1000 characters and >= 3 non-whitespace
      characters to deter DoS-by-giant-prompt.

    Personal-history scope (spec
    ``2026-04-26-intelligence-ask-personal-history-query.md``): the
    addon proxy injects ``X-Lit-Viewer-Id`` from the ``lit_viewer``
    cookie. Clients cannot spoof another viewer because the proxy
    *replaces* whatever the client sent. ``None`` here means "no
    profile" — the service runs the legacy viewer-agnostic path.
    """
    _require_rag_enabled()

    # Post-strip length check. Pydantic's min_length=1 only rejects the
    # empty string, but a 2-char query gives the LLM nothing to work
    # with and would waste an API call. "   " strips to "" which would
    # pass min_length=1 if it weren't for this check.
    if len(body.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query too short")

    # If the body redundantly specifies a drive, it must agree with the
    # X-Lit-Drive header. The header is the source of truth (set by the
    # host's Generic Addon Proxy from the URL); a mismatch indicates a
    # spoofed body and is rejected outright.
    if body.drive and body.drive != drive:
        raise HTTPException(status_code=403, detail="Drive mismatch")

    # Acquire a concurrency slot BEFORE opening the SSE stream. If the
    # semaphore is full, fail fast with 503 so the caller sees a real
    # HTTP error instead of an opaque "stream never produced anything"
    # timeout. A 1ms wait keeps the acquire non-blocking in practice
    # while still yielding to the scheduler — critical under contention
    # so we do not bounce-and-retry clients that raced the slot.
    semaphore = _get_ask_semaphore()
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Too many concurrent Ask requests, please retry shortly",
        )

    # From this point the slot is owned by the generator's finally
    # block; any failure BEFORE the StreamingResponse is constructed
    # must release the slot manually.
    try:
        generator = _sse_stream(
            query=body.query,
            access_token=access_token,
            top_k=body.top_k,
            file_type=body.file_type,
            drive=drive,
            viewer_id=viewer_id,
            semaphore=semaphore,
        )
        return StreamingResponse(
            generator,
            media_type=_SSE_MEDIA_TYPE,
            headers=_SSE_HEADERS,
        )
    except BaseException:
        semaphore.release()
        raise


@router.post("/find")
async def find_endpoint(
    body: FindRequest,
    access_token: Annotated[str | None, Cookie()] = None,
    drive: str = Depends(require_drive),
    viewer_id: Annotated[
        str | None, Header(alias="X-Lit-Viewer-Id")
    ] = None,
) -> dict:
    """File-listing sibling of /ask (spec
    ``2026-04-30-intelligence-find-mode.md``).

    Returns a single-shot JSON payload (NOT SSE) with the structured
    decomposed query plus the retrieve hits. Stage E (LLM answer
    generation) is deliberately skipped — the whole point of Find mode
    is "no hallucination, just files" — so the hit text comes verbatim
    from the retriever's segments.

    Gating mirrors /ask: ``features.rag`` must be on, the LLM provider
    must be enabled (Stages A and C use it), and the question must be
    >= 3 non-whitespace chars after strip. Drive header is required —
    we read it via ``require_drive`` for FastAPI dispatch and re-check
    here so direct-call tests with ``drive=""`` produce the expected
    400.

    Viewer-id is opt-in: a personal-scope question with no
    ``X-Lit-Viewer-Id`` header degrades gracefully (Stage B is skipped
    inside the service, no 4xx).
    """
    _require_rag_enabled()

    # Drive non-empty check. ``require_drive`` already raises 400 on a
    # missing header, but tests invoke the handler directly with
    # ``drive=""`` to simulate that path — re-validate so both code
    # paths produce the same status code.
    if not drive:
        raise HTTPException(status_code=400, detail="Drive context required")

    if len(body.question.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query too short")

    # Share the /ask semaphore. Find calls Stages A and C (decompose +
    # category expand) which are the same LLM provider as /ask, so the
    # operator's "max concurrent LLM calls" budget applies uniformly. A
    # 1ms wait keeps acquire non-blocking while still yielding — same
    # pattern as ask_endpoint.
    semaphore = _get_ask_semaphore()
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Too many concurrent requests, please retry shortly",
        )

    try:
        return await find_files(
            question=body.question,
            drive=drive,
            viewer_id=viewer_id,
            overrides=body.overrides,
            limit=body.limit,
        )
    finally:
        semaphore.release()
