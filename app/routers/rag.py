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
import math
import threading
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from app import llm
from app.config import settings
from app.credentials import CallerCredential
from app.file_matching import (
    FilterUnavailable,
    SearchFilter,
    fetch_matching_file_ids,
    scope_member_ids,
)
from app.llm import ImageRejected
from app.llm_routing import Resolved, profile_can_see
from app.routers.llm_gate import require_llm
from app.drive_context import require_drive
from app.rag import file_question
from app.rag.service import (
    AnswerEvent,
    stream_answer,
)
from app.rag.prompt import (
    GET_FRAMES_TOOL,
    build_file_question_system_prompt,
    build_file_question_user_prompt,
    format_position,
)
from app.schemas import AskRequest, FileQuestionRequest

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


async def _require_rag_enabled(drive: str, requested: str | None = None) -> Resolved:
    """Raise 4xx/5xx unless RAG is on and an LLM serves this drive.

    * 400 when the feature is explicitly disabled in config.
    * 400 when no LLM profile serves the drive.
    * 503 when the drive's cloud policy could not be looked up.
    """
    if not settings.features.rag:
        raise HTTPException(
            status_code=400, detail="RAG feature is disabled"
        )
    return await require_llm(drive, "rag", requested)


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


def _caller_credential(
    request: Request | None, access_token: str | None
) -> CallerCredential:
    if request is not None:
        return CallerCredential.from_request(request)
    if access_token:
        return CallerCredential(cookie=f"access_token={access_token}")
    return CallerCredential()


async def _sse_stream(
    query: str,
    credential: CallerCredential | None,
    top_k: int | None,
    file_type: str | None,
    drive: str | None,
    viewer_id: str | None,
    semaphore: "asyncio.Semaphore",
    resolved: Resolved,
    scope: dict | None = None,
    scope_file_ids: list[str] | None = None,
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
            credential=credential,
            top_k=top_k,
            file_type=file_type,
            drive=drive,
            viewer_id=viewer_id,
            resolved=resolved,
            scope=scope,
            scope_file_ids=scope_file_ids,
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
    request: Request = None,
    access_token: str | None = None,
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
    resolved = await _require_rag_enabled(drive, body.profile)

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

    # Resolve the scope before a concurrency slot is taken: a slow core
    # call must not hold one, and a failure must answer 503 rather than
    # open a stream. The scope and a search filter with the same values
    # name the same files (they share ``SearchFilter``'s validator), and
    # resolution is per request with no cache.
    scope_payload: dict | None = None
    scope_file_ids: list[str] | None = None
    if body.scope is not None:
        scope_payload = body.scope.as_event_payload()
        conditions = {
            "folders": body.scope.folders,
            "tags": body.scope.tags,
            "kinds": body.scope.kinds,
        }
        try:
            without_collections = await fetch_matching_file_ids(
                drive, SearchFilter(**conditions)
            )
            with_collections = None
            if body.scope.collections:
                with_collections = await fetch_matching_file_ids(
                    drive,
                    SearchFilter(
                        collections=body.scope.collections, **conditions
                    ),
                )
        except FilterUnavailable:
            logger.warning("ask scope resolution failed for drive %s", drive)
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "scope_unavailable",
                    "message": "Scope unavailable",
                },
            )
        scope_file_ids = scope_member_ids(
            without_collections, with_collections, body.scope.file_ids
        )

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
        credential = _caller_credential(request, access_token)
        generator = _sse_stream(
            query=body.query,
            credential=credential,
            top_k=body.top_k,
            file_type=body.file_type,
            drive=drive,
            viewer_id=viewer_id,
            semaphore=semaphore,
            resolved=resolved,
            scope=scope_payload,
            scope_file_ids=scope_file_ids,
        )
        return StreamingResponse(
            generator,
            media_type=_SSE_MEDIA_TYPE,
            headers=_SSE_HEADERS,
        )
    except BaseException:
        semaphore.release()
        raise


@router.get("/files/{file_id}/ask")
async def file_ask_availability(
    file_id: str,
    drive: str = Depends(require_drive),
) -> dict[str, bool]:
    """Say whether this drive may ask about one file (SPEC-ADDON-053 1).

    An active indexed file of the request drive with at least one text
    chunk is askable. Anything else — another drive, not indexed, no
    chunk — is ``available: false``, not an error: the tab simply has no
    button, and reporting a failure there would look like a broken page.
    """
    return {
        "available": await asyncio.to_thread(
            file_question.available, file_id, drive
        )
    }


@router.post("/files/{file_id}/ask")
async def file_ask_endpoint(
    file_id: str,
    body: FileQuestionRequest,
    drive: str = Depends(require_drive),
) -> StreamingResponse:
    """Answer a question about one file from that file's indexed text.

    SPEC-ADDON-053. Gates are ``/ask``'s: ``features.rag``, the routed
    LLM (honouring ``llm_cloud``), and the shared Ask semaphore, so a file
    question in flight counts against the same limit. The material is
    read before a slot is taken, so a file that cannot be asked about
    neither waits for one nor sends anything to a model.

    Streams ``answer_chunk``, then ``citations``, then ``meta``, then
    ``done``; a failure after the stream started ends with ``done``
    carrying ``error``. A picture may attach one image (SPEC-ADDON-071), and
    a video whose profile can see up to eight of its frames, with the video's
    passages when it has any and from the frames alone when it has none
    (SPEC-ADDON-108, -114); such a question may also ask for up to twelve more
    frames of the video at times of its choosing (SPEC-ADDON-116). A picture or
    a video with no text chunk whose profile cannot see is refused with 409
    before the slot.
    """
    resolved = await _require_rag_enabled(drive, body.profile)

    max_chars = settings.rag.file_question_max_chars
    history = file_question.trim_history(body.history, max_chars)
    budget = max_chars - sum(
        len(turn.question) + len(turn.answer) for turn in history
    )
    ranking_text = (
        body.history[-1].question + "\n" + body.question
        if body.history
        else body.question
    )

    material = await asyncio.to_thread(
        file_question.load_material,
        file_id=file_id,
        drive=drive,
        question=body.question,
        budget=budget,
        ranking_text=ranking_text,
    )
    if material is None:
        raise HTTPException(status_code=404, detail="File not found")

    if (
        file_question.is_picture(material)
        and not material.chunks
        and not profile_can_see(resolved)
    ):
        raise HTTPException(status_code=409, detail="picture_needs_vision")

    if (
        file_question.is_text_less_video(material)
        and not profile_can_see(resolved)
    ):
        raise HTTPException(status_code=409, detail="video_needs_vision")

    semaphore = _get_ask_semaphore()
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Too many concurrent Ask requests, please retry shortly",
        )

    frames_deadline = (
        asyncio.get_running_loop().time() + file_question.FRAME_WAIT_SECONDS
    )

    try:
        return StreamingResponse(
            _file_question_stream(
                file_id=file_id,
                question=body.question,
                material=material,
                resolved=resolved,
                semaphore=semaphore,
                position_seconds=body.position_seconds,
                history=history,
                frames_deadline=frames_deadline,
                slot_time=(
                    frames_deadline - file_question.FRAME_WAIT_SECONDS
                ),
            ),
            media_type=_SSE_MEDIA_TYPE,
            headers=_SSE_HEADERS,
        )
    except BaseException:
        semaphore.release()
        raise


async def _file_question_stream(
    file_id: str,
    question: str,
    material: file_question.Material,
    resolved: Resolved,
    semaphore: "asyncio.Semaphore",
    position_seconds: float | None = None,
    history: list[file_question.HistoryTurn] | None = None,
    frames_deadline: float | None = None,
    slot_time: float | None = None,
) -> AsyncIterator[str]:
    """Adapt the routed model's answer to the file-question SSE wire format.

    Owns the semaphore slot: it is released on the end of the stream, on a
    mid-stream failure and on a client disconnect. A video whose profile can
    see is answered from a set of its frames, with its passages when it has
    any; every other file keeps the text path.
    """
    try:
        passages = (
            material.chunks
            if profile_can_see(resolved)
            and file_question.is_text_video(material)
            else ()
        )
        stream = (
            _frames_question_stream(
                file_id=file_id,
                question=question,
                material=material,
                resolved=resolved,
                position_seconds=position_seconds,
                history=history,
                deadline=frames_deadline,
                passages=passages,
                slot_time=slot_time,
            )
            if passages or file_question.is_text_less_video(material)
            else _text_question_stream(
                file_id=file_id,
                question=question,
                material=material,
                resolved=resolved,
                position_seconds=position_seconds,
                history=history,
            )
        )
        async for event in stream:
            yield event
    finally:
        semaphore.release()


async def _text_question_stream(
    file_id: str,
    question: str,
    material: file_question.Material,
    resolved: Resolved,
    position_seconds: float | None = None,
    history: list[file_question.HistoryTurn] | None = None,
) -> AsyncIterator[str]:
    """Answer from the file's text, with the picture or position image of 071.

    Image preparation and the text-only retry run under the slot the caller
    holds. The upstream stream is closed explicitly rather than left to
    garbage collection, so a disconnected tab tears the provider request down
    at once.
    """
    passages = [chunk.text for chunk in material.chunks]
    position = (
        position_seconds
        if material.file_type in ("video", "audio")
        else None
    )
    can_see = profile_can_see(resolved)
    image_applies = file_question.is_picture(material) or (
        material.file_type == "video" and position_seconds is not None
    )

    image_status = "none"
    image: file_question.ImageMaterial | None = None

    def meta_event() -> AnswerEvent:
        data: dict = {"partial": material.partial}
        if image_applies or can_see:
            data["image"] = image_status
        return AnswerEvent(kind="meta", data=data)

    def text_stream():
        return resolved.client.generate_stream(
            build_file_question_system_prompt(
                resolved.profile.config.output_language,
                has_passages=bool(passages),
                has_history=bool(history),
            ),
            build_file_question_user_prompt(
                question, passages, position_seconds=position,
                history=history,
            ),
            max_tokens_override=settings.rag.max_tokens,
        )

    stream = None
    answer: list[str] = []
    try:
        if image_applies and can_see:
            try:
                image = await asyncio.to_thread(
                    file_question.load_image, material, position_seconds,
                )
                image_status = "attached"
            except file_question.ImageUnavailable as exc:
                image_status = "unavailable"
                logger.warning(
                    "file question image failed: %s (file %s)",
                    exc.failure,
                    file_id,
                )

        if image_status == "unavailable" and not passages:
            yield _format_sse_event(meta_event())
            yield _format_sse_event(
                AnswerEvent(kind="done", data={"error": "image_unusable"})
            )
            return

        if image_status == "attached" and image is not None:
            stream = resolved.client.generate_stream_with_image(
                image.data,
                image.mime_type,
                build_file_question_system_prompt(
                    resolved.profile.config.output_language,
                    has_image=True,
                    has_passages=bool(passages),
                    has_history=bool(history),
                ),
                build_file_question_user_prompt(
                    question,
                    passages,
                    has_image=True,
                    position_seconds=position,
                    history=history,
                ),
                max_tokens_override=settings.rag.max_tokens,
            )
            try:
                async for delta in stream:
                    answer = [*answer, delta]
                    yield _format_sse_event(
                        AnswerEvent(kind="answer_chunk", data={"delta": delta})
                    )
            except ImageRejected:
                logger.warning(
                    "file question image failed: %s (file %s)",
                    file_question.IMAGE_FAILURE_REJECTED,
                    file_id,
                )
                image_status = "unavailable"
                try:
                    await stream.aclose()
                finally:
                    stream = None
                if not passages:
                    yield _format_sse_event(meta_event())
                    yield _format_sse_event(
                        AnswerEvent(
                            kind="done", data={"error": "image_unusable"},
                        )
                    )
                    return
                stream = text_stream()
                async for delta in stream:
                    answer = [*answer, delta]
                    yield _format_sse_event(
                        AnswerEvent(kind="answer_chunk", data={"delta": delta})
                    )
        else:
            stream = text_stream()
            async for delta in stream:
                answer = [*answer, delta]
                yield _format_sse_event(
                    AnswerEvent(kind="answer_chunk", data={"delta": delta})
                )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - the stream must end, not die
        logger.warning(
            "file question failed: %s (file %s)",
            type(exc).__name__,
            file_id,
        )
        yield _format_sse_event(meta_event())
        yield _format_sse_event(
            AnswerEvent(
                kind="done", data={"error": "Answer generation failed"},
            )
        )
    else:
        if not "".join(answer).strip():
            logger.warning(
                "file question failed: %s (file %s)",
                "EmptyAnswer",
                file_id,
            )
            yield _format_sse_event(meta_event())
            yield _format_sse_event(
                AnswerEvent(
                    kind="done", data={"error": "Answer generation failed"},
                )
            )
            return
        yield _format_sse_event(
            AnswerEvent(
                kind="citations",
                data=file_question.citations(
                    "".join(answer), material.chunks
                ),
            )
        )
        yield _format_sse_event(meta_event())
        yield _format_sse_event(AnswerEvent(kind="done", data={}))
    finally:
        if stream is not None:
            await stream.aclose()


#: One SSE comment frame, written while the frames path has nothing to send.
_FRAMES_KEEPALIVE = ": keepalive\n\n"
#: How often the frames path writes one before the first answer token.
_KEEPALIVE_SECONDS = 5.0

#: How long a video's question may keep offering ``get_frames``, from the slot.
_FRAME_TOOL_WINDOW_SECONDS = 120.0
#: How long a video's question may run in all, from the slot.
_FRAME_TOOL_DEADLINE_SECONDS = 300.0
#: How many frame requests one question serves.
_FRAME_REQUESTS_PER_QUESTION = 3
#: How many frames one frame request may add.
_FRAMES_PER_REQUEST = 4
#: The name a frame request must call.
_FRAME_TOOL_NAME = GET_FRAMES_TOOL["function"]["name"]


class _Deadline(Exception):
    """The question's total deadline passed while work was in flight."""


@dataclass(frozen=True)
class _Rejected:
    """One time a frame request asked for and did not get."""

    value: object
    reason: str
    #: The time it was asked at, when that time could be read.
    time: float | None = None
    #: The frame it duplicates, when it duplicates one sent before.
    number: int | None = None


@dataclass(frozen=True)
class _CallPlan:
    """What one tool call of a reply gets, before any frame is cut."""

    call: "llm.FrameToolCall"
    #: Whether this call spends one of the question's three frame requests.
    spends: bool
    #: The whole result text when nothing is cut at all.
    content: str = ""
    #: The times to cut, in time order.
    used: tuple[float, ...] = ()
    rejected: tuple[_Rejected, ...] = ()


def _file_meta(material: file_question.Material, image_status: str) -> AnswerEvent:
    """The ``meta`` event of a video's answer."""
    return AnswerEvent(
        kind="meta",
        data={"partial": material.partial, "image": image_status},
    )


async def _keepalive_answer(
    stream, answer: list[str],
) -> AsyncIterator[str]:
    """One model stream as answer frames, with a comment on each silent interval."""
    async for delta in _answer_with_keepalive(stream):
        if delta is None:
            yield _FRAMES_KEEPALIVE
            continue
        answer.append(delta)
        yield _format_sse_event(
            AnswerEvent(kind="answer_chunk", data={"delta": delta})
        )


async def _text_only_answer(
    file_id: str,
    question: str,
    material: file_question.Material,
    resolved: Resolved,
    passages: tuple[file_question.SelectedChunk, ...],
    history: list[file_question.HistoryTurn] | None,
    position_seconds: float | None,
    image_status: str,
) -> AsyncIterator[str]:
    """Answer a text video from its passages alone, then its meta and close.

    ``passages`` are the ones the request with frames carried, and the stream
    is kept alive until its first token as the frames path is. ``image_status``
    says why no frame came with it: ``"none"`` when none was tried and
    ``"unavailable"`` when one was.
    """
    stream = None
    answer: list[str] = []
    try:
        stream = resolved.client.generate_stream(
            build_file_question_system_prompt(
                resolved.profile.config.output_language,
                has_passages=True,
                has_history=bool(history),
            ),
            build_file_question_user_prompt(
                question,
                [chunk.text for chunk in passages],
                position_seconds=position_seconds,
                history=history,
            ),
            max_tokens_override=settings.rag.max_tokens,
        )
        async for event in _keepalive_answer(stream, answer):
            yield event
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - the stream must end, not die
        logger.warning(
            "file question failed: %s (file %s)",
            type(exc).__name__,
            file_id,
        )
        yield _format_sse_event(_file_meta(material, image_status))
        yield _format_sse_event(
            AnswerEvent(
                kind="done", data={"error": "Answer generation failed"},
            )
        )
        return
    finally:
        if stream is not None:
            await stream.aclose()
    if not "".join(answer).strip():
        logger.warning(
            "file question failed: %s (file %s)",
            "EmptyAnswer",
            file_id,
        )
        yield _format_sse_event(_file_meta(material, image_status))
        yield _format_sse_event(
            AnswerEvent(
                kind="done", data={"error": "Answer generation failed"},
            )
        )
        return
    yield _format_sse_event(
        AnswerEvent(
            kind="citations",
            data=file_question.citations("".join(answer), passages),
        )
    )
    yield _format_sse_event(_file_meta(material, image_status))
    yield _format_sse_event(AnswerEvent(kind="done", data={}))


async def _todays_frames_answer(
    *,
    file_id: str,
    question: str,
    material: file_question.Material,
    resolved: Resolved,
    passages: tuple[file_question.SelectedChunk, ...],
    history: list[file_question.HistoryTurn] | None,
    position: float | None,
    frames: list[file_question.Frame],
) -> AsyncIterator[str]:
    """The request this path sent before the frame tool, and how it is handled.

    One refusal of the frames falls back to the passages alone when there are
    any, and ends the answer otherwise; every other failure ends it. This is
    the copy a question falls back to, so it stays exactly as it was.
    """
    first_number = len(passages) + 1
    frames_to_send = [
        (
            f"[{first_number + index}] {format_position(frame.time)}",
            frame.data,
            frame.mime_type,
        )
        for index, frame in enumerate(frames)
    ]
    stream = None
    answer: list[str] = []
    image_status = "attached"
    try:
        stream = resolved.client.generate_stream_with_images(
            frames_to_send,
            build_file_question_system_prompt(
                resolved.profile.config.output_language,
                has_image=True,
                has_passages=bool(passages),
                has_history=bool(history),
                frames=True,
            ),
            build_file_question_user_prompt(
                question,
                [chunk.text for chunk in passages],
                has_image=True,
                frames=True,
                position_seconds=position,
                history=history,
            ),
            max_tokens_override=settings.rag.max_tokens,
        )
        async for event in _keepalive_answer(stream, answer):
            yield event
    except ImageRejected:
        logger.warning(
            "file question frames failed: %s (file %s)",
            "frames_rejected",
            file_id,
        )
        if not passages:
            yield _format_sse_event(_file_meta(material, "unavailable"))
            yield _format_sse_event(
                AnswerEvent(kind="done", data={"error": "frames_rejected"})
            )
            return
        image_status = "unavailable"
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - the stream must end, not die
        logger.warning(
            "file question failed: %s (file %s)",
            type(exc).__name__,
            file_id,
        )
        yield _format_sse_event(_file_meta(material, image_status))
        yield _format_sse_event(
            AnswerEvent(
                kind="done", data={"error": "Answer generation failed"},
            )
        )
        return
    finally:
        if stream is not None:
            await stream.aclose()

    if image_status == "unavailable":
        async for event in _text_only_answer(
            file_id=file_id,
            question=question,
            material=material,
            resolved=resolved,
            passages=passages,
            history=history,
            position_seconds=position,
            image_status="unavailable",
        ):
            yield event
        return

    if not "".join(answer).strip():
        logger.warning(
            "file question failed: %s (file %s)",
            "EmptyAnswer",
            file_id,
        )
        yield _format_sse_event(_file_meta(material, "attached"))
        yield _format_sse_event(
            AnswerEvent(
                kind="done", data={"error": "Answer generation failed"},
            )
        )
        return
    yield _format_sse_event(
        AnswerEvent(
            kind="citations",
            data=file_question.citations(
                "".join(answer),
                passages
                + tuple(
                    file_question.SelectedChunk(
                        text="", position={"seconds": frame.time},
                    )
                    for frame in frames
                ),
            ),
        )
    )
    yield _format_sse_event(_file_meta(material, "attached"))
    yield _format_sse_event(AnswerEvent(kind="done", data={}))


async def _frames_question_stream(
    file_id: str,
    question: str,
    material: file_question.Material,
    resolved: Resolved,
    position_seconds: float | None = None,
    history: list[file_question.HistoryTurn] | None = None,
    deadline: float | None = None,
    passages: tuple[file_question.SelectedChunk, ...] = (),
    slot_time: float | None = None,
) -> AsyncIterator[str]:
    """Answer a video from a set of its frames, and from its passages when it has any.

    The frames are chosen and prepared under the caller's slot, from the
    moment it was taken. Until the first token the stream writes an SSE
    comment, so no proxy hop waits on frame preparation in silence. A frame
    that was dropped is simply not sent; the answer is from the frames that
    were. A question with at least one frame may ask for more of them for as
    long as its window allows; with no frame to send, a video with passages is
    answered from them alone, in one request kept alive the same way.
    """
    position = (
        position_seconds
        if position_seconds is not None and position_seconds > 0
        else None
    )
    loop = asyncio.get_running_loop()
    if slot_time is None:
        slot_time = loop.time()
    if deadline is None:
        deadline = slot_time + file_question.FRAME_WAIT_SECONDS

    cancelled = threading.Event()
    prepare = asyncio.create_task(
        file_question.prepare_frames(
            material, position, deadline=deadline, cancelled=cancelled,
        )
    )
    try:
        while not prepare.done():
            done, _ = await asyncio.wait(
                {prepare}, timeout=_KEEPALIVE_SECONDS,
            )
            if not done:
                yield _FRAMES_KEEPALIVE
        frames, dropped, contents = prepare.result()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - the stream must end, not die
        logger.warning(
            "file question frames failed: %s (file %s)",
            type(exc).__name__,
            file_id,
        )
        frames, dropped, contents = [], 0, ()
    finally:
        if not prepare.done():
            cancelled.set()
            prepare.cancel()
            try:
                await prepare
            except asyncio.CancelledError:
                pass

    if not frames:
        if not passages:
            logger.warning("frames_unusable (file %s)", file_id)
            yield _format_sse_event(_file_meta(material, "unavailable"))
            yield _format_sse_event(
                AnswerEvent(kind="done", data={"error": "frames_unusable"})
            )
            return
        image_status = "none"
        if dropped:
            logger.warning("frames_unusable (file %s)", file_id)
            image_status = "unavailable"
        async for event in _text_only_answer(
            file_id=file_id,
            question=question,
            material=material,
            resolved=resolved,
            passages=passages,
            history=history,
            position_seconds=position,
            image_status=image_status,
        ):
            yield event
        return

    async for event in _frame_tool_answer(
        file_id=file_id,
        question=question,
        material=material,
        resolved=resolved,
        passages=passages,
        history=history,
        position=position,
        frames=frames,
        contents=contents,
        total_deadline=slot_time + _FRAME_TOOL_DEADLINE_SECONDS,
        window_end=slot_time + _FRAME_TOOL_WINDOW_SECONDS,
    ):
        yield event


async def _frame_tool_answer(
    *,
    file_id: str,
    question: str,
    material: file_question.Material,
    resolved: Resolved,
    passages: tuple[file_question.SelectedChunk, ...],
    history: list[file_question.HistoryTurn] | None,
    position: float | None,
    frames: list[file_question.Frame],
    contents: tuple[str, ...],
    total_deadline: float,
    window_end: float,
) -> AsyncIterator[str]:
    """Offer the frame tool while the question may, then answer.

    A tool request is read to its end before any of its text is written, so a
    reply that asks for frames never reaches the viewer as an answer. Once
    three requests are served or the window has passed, the next request is
    final and its answer streams. Any failure before the first written chunk
    sends today's request once; nothing is started after the total deadline.
    """
    loop = asyncio.get_running_loop()
    first_number = len(passages) + 1
    initial_frames = [
        (
            f"[{first_number + index}] {format_position(frame.time)}",
            frame.data,
            frame.mime_type,
        )
        for index, frame in enumerate(frames)
    ]
    positions = [frame.time for frame in frames]
    sent = [
        (frame.time, first_number + index)
        for index, frame in enumerate(frames)
    ]
    duration = material.duration
    system_prompt = build_file_question_system_prompt(
        resolved.profile.config.output_language,
        has_image=True,
        has_passages=bool(passages),
        has_history=bool(history),
        frames=True,
        tool=True,
    )
    user_prompt = build_file_question_user_prompt(
        question,
        [chunk.text for chunk in passages],
        has_image=True,
        frames=True,
        position_seconds=position,
        history=history,
        duration=(
            duration
            if duration is not None and math.isfinite(duration)
            else None
        ),
        contents=contents,
    )
    rounds: list[llm.FrameRound] = []
    answer: list[str] = []
    next_number = first_number + len(frames)
    served = 0
    requested = 0
    cancelled = threading.Event()

    def closing(text: str) -> list[AnswerEvent]:
        return [
            AnswerEvent(
                kind="citations",
                data=file_question.citations(
                    text,
                    passages
                    + tuple(
                        file_question.SelectedChunk(
                            text="", position={"seconds": time},
                        )
                        for time in positions
                    ),
                ),
            ),
            _file_meta(material, "attached"),
            AnswerEvent(kind="done", data={}),
        ]

    def failed() -> list[AnswerEvent]:
        return [
            _file_meta(material, "attached"),
            AnswerEvent(
                kind="done", data={"error": "Answer generation failed"},
            ),
        ]

    async def fallback() -> AsyncIterator[str]:
        yield _FRAMES_KEEPALIVE
        async for event in _fallback_answer(
            file_id=file_id,
            question=question,
            material=material,
            resolved=resolved,
            passages=passages,
            history=history,
            position=position,
            frames=frames,
            total_deadline=total_deadline,
        ):
            yield event

    try:
        while True:
            # Each phase's own keep-alive timer starts afresh, so phases that
            # each end just under the interval would add up to a silence.
            yield _FRAMES_KEEPALIVE
            if loop.time() >= total_deadline:
                logger.warning("frame_tool_deadline (file %s)", file_id)
                for event in failed():
                    yield _format_sse_event(event)
                return

            if (
                served >= _FRAME_REQUESTS_PER_QUESTION
                or loop.time() >= window_end
            ):
                stream = None
                remaining = total_deadline - loop.time()
                if remaining <= 0:
                    logger.warning("frame_tool_deadline (file %s)", file_id)
                    for event in failed():
                        yield _format_sse_event(event)
                    return
                try:
                    stream = resolved.client.stream_with_frames(
                        system_prompt,
                        user_prompt,
                        initial_frames,
                        rounds,
                        tools=[GET_FRAMES_TOOL],
                        tool_choice="none",
                    )
                    async with asyncio.timeout(remaining):
                        async for delta in _answer_with_keepalive(stream):
                            if delta is None:
                                yield _FRAMES_KEEPALIVE
                                continue
                            answer.append(delta)
                            yield _format_sse_event(
                                AnswerEvent(
                                    kind="answer_chunk",
                                    data={"delta": delta},
                                )
                            )
                except TimeoutError:
                    logger.warning("frame_tool_deadline (file %s)", file_id)
                    for event in failed():
                        yield _format_sse_event(event)
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - reported, not raised
                    if answer:
                        logger.warning(
                            "file question failed: %s (file %s)",
                            type(exc).__name__,
                            file_id,
                        )
                        for event in failed():
                            yield _format_sse_event(event)
                        return
                    async for event in fallback():
                        yield event
                    return
                finally:
                    if stream is not None:
                        await stream.aclose()
                if not "".join(answer).strip():
                    async for event in fallback():
                        yield event
                    return
                for event in closing("".join(answer)):
                    yield _format_sse_event(event)
                return

            reply_task = asyncio.create_task(
                resolved.client.reply_with_frames(
                    system_prompt,
                    user_prompt,
                    initial_frames,
                    rounds,
                    tools=[GET_FRAMES_TOOL],
                    tool_choice="auto",
                )
            )
            reply: llm.FrameReply | None = None
            failure: Exception | None = None
            try:
                async for event in _until_done(
                    reply_task, deadline=total_deadline,
                ):
                    yield event
                reply = reply_task.result()
            except _Deadline:
                logger.warning("frame_tool_deadline (file %s)", file_id)
                for event in failed():
                    yield _format_sse_event(event)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the fallback answers
                failure = exc
            finally:
                if not reply_task.done():
                    reply_task.cancel()
                    try:
                        await reply_task
                    except BaseException:  # noqa: BLE001 - cancelled, not read
                        pass

            if failure is not None or reply is None or not (
                reply.text.strip() or reply.calls
            ):
                if failure is not None:
                    logger.warning(
                        "file question frames failed: %s (file %s)",
                        type(failure).__name__,
                        file_id,
                    )
                async for event in fallback():
                    yield event
                return

            if not reply.calls:
                yield _format_sse_event(
                    AnswerEvent(kind="answer_chunk", data={"delta": reply.text})
                )
                for event in closing(reply.text):
                    yield _format_sse_event(event)
                return

            plans = _frame_plan(
                reply.calls,
                sent=sent,
                position=position,
                duration=duration,
                served=served,
            )
            wanted = sorted({time for plan in plans for time in plan.used})
            yield _FRAMES_KEEPALIVE
            cut_task = asyncio.create_task(
                file_question.cut_frames(
                    material, wanted, deadline=total_deadline,
                    cancelled=cancelled,
                )
            )
            cut: list[file_question.Frame] = []
            try:
                async for event in _until_done(
                    cut_task, deadline=total_deadline,
                ):
                    yield event
                cut = cut_task.result()
            except _Deadline:
                logger.warning("frame_tool_deadline (file %s)", file_id)
                for event in failed():
                    yield _format_sse_event(event)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the fallback answers
                logger.warning(
                    "file question frames failed: %s (file %s)",
                    type(exc).__name__,
                    file_id,
                )
                async for event in fallback():
                    yield event
                return
            finally:
                if not cut_task.done():
                    cut_task.cancel()
                    try:
                        await cut_task
                    except BaseException:  # noqa: BLE001 - cancelled, not read
                        pass

            by_time = {frame.time: frame for frame in cut}
            cut_per_plan: list[list[file_question.Frame]] = []
            round_frames: list[tuple[str, bytes, str]] = []
            for plan in plans:
                if plan.spends:
                    served += 1
                frames_here = [
                    by_time[time] for time in plan.used if time in by_time
                ]
                cut_per_plan = [*cut_per_plan, frames_here]
                for frame in frames_here:
                    round_frames = [
                        *round_frames,
                        (
                            f"[{next_number}] {format_position(frame.time)}",
                            frame.data,
                            frame.mime_type,
                        ),
                    ]
                    positions = [*positions, frame.time]
                    sent = [*sent, (frame.time, next_number)]
                    next_number += 1
                    requested += 1
            results = []
            for plan, frames_here in zip(plans, cut_per_plan):
                if plan.content:
                    results = [
                        *results,
                        llm.FrameToolResult(
                            call=plan.call, content=plan.content,
                        ),
                    ]
                    continue
                notes = [
                    _rejection_note(rejected) for rejected in plan.rejected
                ]
                notes = [
                    *notes,
                    *[
                        f"{format_position(time)}: not available"
                        for time in plan.used
                        if time not in by_time
                    ],
                ]
                results = [
                    *results,
                    llm.FrameToolResult(
                        call=plan.call,
                        content=_result_text(len(frames_here), notes),
                    ),
                ]
            rounds = [
                *rounds,
                llm.FrameRound(
                    text=reply.text,
                    results=tuple(results),
                    frames=tuple(round_frames),
                ),
            ]
    finally:
        cancelled.set()
        if served:
            logger.info(
                "frame requests: %d calls, %d frames (file %s)",
                served,
                requested,
                file_id,
            )


async def _fallback_answer(
    *,
    file_id: str,
    question: str,
    material: file_question.Material,
    resolved: Resolved,
    passages: tuple[file_question.SelectedChunk, ...],
    history: list[file_question.HistoryTurn] | None,
    position: float | None,
    frames: list[file_question.Frame],
    total_deadline: float,
) -> AsyncIterator[str]:
    """Send today's request once, unless the total deadline has passed.

    A request already in flight when the deadline passes is stopped, and the
    answer ends as a failure after whatever it had written.
    """
    logger.warning("frame_tool_fallback (file %s)", file_id)
    loop = asyncio.get_running_loop()
    remaining = total_deadline - loop.time()
    if remaining <= 0:
        logger.warning("frame_tool_deadline (file %s)", file_id)
        yield _format_sse_event(_file_meta(material, "attached"))
        yield _format_sse_event(
            AnswerEvent(
                kind="done", data={"error": "Answer generation failed"},
            )
        )
        return
    try:
        async with asyncio.timeout(remaining):
            async for event in _todays_frames_answer(
                file_id=file_id,
                question=question,
                material=material,
                resolved=resolved,
                passages=passages,
                history=history,
                position=position,
                frames=frames,
            ):
                yield event
    except TimeoutError:
        logger.warning("frame_tool_deadline (file %s)", file_id)
        yield _format_sse_event(_file_meta(material, "attached"))
        yield _format_sse_event(
            AnswerEvent(
                kind="done", data={"error": "Answer generation failed"},
            )
        )


async def _until_done(task: "asyncio.Task", *, deadline: float) -> AsyncIterator[str]:
    """Write a comment on each silent interval until ``task`` finishes.

    Raises :class:`_Deadline` rather than waiting past the question's deadline.
    """
    loop = asyncio.get_running_loop()
    while not task.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise _Deadline
        done, _ = await asyncio.wait(
            {task}, timeout=min(_KEEPALIVE_SECONDS, remaining),
        )
        if not done:
            yield _FRAMES_KEEPALIVE


def _call_times(call: "llm.FrameToolCall") -> list | None:
    """The ``times`` list of a ``get_frames`` call, or ``None`` for another call."""
    if call.name != _FRAME_TOOL_NAME:
        return None
    arguments = call.arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
    if not isinstance(arguments, dict):
        return None
    times = arguments.get("times")
    return times if isinstance(times, list) else None


def _frame_plan(
    calls: Sequence["llm.FrameToolCall"],
    *,
    sent: Sequence[tuple[float, int]],
    position: float | None,
    duration: float | None,
    served: int,
) -> list[_CallPlan]:
    """Decide, for every call of one reply, what it gets before anything is cut."""
    plans: list[_CallPlan] = []
    taken = served
    used_here: list[float] = []
    for call in calls:
        if taken >= _FRAME_REQUESTS_PER_QUESTION:
            plans = [
                *plans,
                _CallPlan(call=call, spends=False, content=_limit_text()),
            ]
            continue
        taken += 1
        times = _call_times(call)
        if times is None:
            plans = [
                *plans,
                _CallPlan(
                    call=call, spends=True, content=_bad_call_text(call.name),
                ),
            ]
            continue
        used: list[float] = []
        rejected: list[_Rejected] = []
        for value in times:
            seconds = file_question.read_time(value)
            if seconds is None:
                rejected = [
                    *rejected, _Rejected(value=value, reason="unreadable"),
                ]
                continue
            time = file_question.frame_time(seconds)
            if duration is not None and time > duration:
                rejected = [
                    *rejected,
                    _Rejected(value=value, reason="past the end", time=time),
                ]
                continue
            if position is not None and (
                abs(time - position) < file_question.FRAME_MIN_GAP_SECONDS
            ):
                rejected = [
                    *rejected,
                    _Rejected(
                        value=value, reason="at the position", time=time,
                    ),
                ]
                continue
            duplicate = _duplicate(time, sent, used_here)
            if duplicate is not None:
                rejected = [*rejected, duplicate]
                continue
            if len(used) >= _FRAMES_PER_REQUEST:
                rejected = [
                    *rejected,
                    _Rejected(value=value, reason="over the limit", time=time),
                ]
                continue
            used = [*used, time]
            used_here = [*used_here, time]
        plans = [
            *plans,
            _CallPlan(
                call=call,
                spends=True,
                used=tuple(sorted(used)),
                rejected=tuple(rejected),
            ),
        ]
    return plans


def _duplicate(
    time: float,
    sent: Sequence[tuple[float, int]],
    used_here: Sequence[float],
) -> _Rejected | None:
    """The rejection a time that duplicates one already sent or used gets."""
    for sent_time, number in sent:
        if abs(time - sent_time) < file_question.FRAME_MIN_GAP_SECONDS:
            return _Rejected(
                value=time, reason="already sent", time=time, number=number,
            )
    for other in used_here:
        if abs(time - other) < file_question.FRAME_MIN_GAP_SECONDS:
            return _Rejected(value=time, reason="used earlier", time=time)
    return None


#: What each reason a time is not used reads as in a tool result.
_REJECTION_REASONS = {
    "unreadable": "not a time I can read",
    "past the end": "past the end of the video",
    "at the position": "at the position the viewer is at",
    "over the limit": (
        f"over the limit of {_FRAMES_PER_REQUEST} times for one call"
    ),
    # A frame of this reply has no number yet, so none is named: a tool result
    # may name only the frame an unused time duplicates among those sent.
    "used earlier": "within a second of an earlier time of this request",
}


def _rejection_note(rejected: _Rejected) -> str:
    """One line of a tool result: a time that was not used, and why."""
    shown = (
        format_position(rejected.time)
        if rejected.time is not None
        else _shown_value(rejected.value)
    )
    if rejected.reason == "already sent":
        return f"{shown}: already sent as [{rejected.number}]"
    return f"{shown}: {_REJECTION_REASONS[rejected.reason]}"


def _result_text(frames: int, notes: Sequence[str]) -> str:
    """A tool result: how many frames follow, then a line per time that does not."""
    head = (
        f"{frames} frame{'s' if frames != 1 else ''} follow."
        if frames
        else "No frame follows."
    )
    return "\n".join([head, *[f"- {note}" for note in notes]])


def _shown_value(value: object) -> str:
    """A time a model gave, written back to it without inventing one."""
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= 40 else text[:40] + "…"


def _limit_text() -> str:
    return (
        "No frame follows: this question has already been served its "
        f"{_FRAME_REQUESTS_PER_QUESTION} frame requests."
    )


def _bad_call_text(name: str) -> str:
    called = f"{name} cannot" if name else "this call cannot"
    return (
        f"No frame follows: {called} be served. Only {_FRAME_TOOL_NAME} takes a "
        "JSON object with a list of times named times, at most "
        f"{_FRAMES_PER_REQUEST} of them."
    )


async def _answer_with_keepalive(stream) -> AsyncIterator[str | None]:
    """Yield the stream's deltas, and ``None`` on each silent interval.

    The model's own stream is consumed by a task so that waiting for its
    first token does not block the keep-alive the frames path owes the proxy.
    """
    queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for delta in stream:
                queue.put_nowait(("delta", delta))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported to the reader
            queue.put_nowait(("error", exc))
            return
        queue.put_nowait(("end", None))

    task = asyncio.create_task(pump())
    try:
        first = True
        while True:
            try:
                kind, value = await asyncio.wait_for(
                    queue.get(), timeout=_KEEPALIVE_SECONDS if first else None,
                )
            except asyncio.TimeoutError:
                yield None
                continue
            if kind == "delta":
                first = False
                yield value
            elif kind == "error":
                raise value
            else:
                return
    finally:
        task.cancel()
        # The caller closes the provider stream once this returns, so the pump
        # must finish unwinding first; a cancel aimed at this task still wins.
        try:
            await task
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                raise
