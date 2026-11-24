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
    GET_PAGE_IMAGES_TOOL,
    GET_TEXT_SECTION_TOOL,
    GET_TEXT_TOOL,
    GET_TRANSCRIPT_TOOL,
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
    any; a PDF, or an EPUB whose material was cut, from its document tools
    when its profile can see; every other file keeps the text path.
    """
    try:
        passages = (
            material.chunks
            if profile_can_see(resolved)
            and file_question.is_text_video(material)
            else ()
        )
        if file_question.uses_document_tools(
            material, can_see=profile_can_see(resolved),
        ):
            stream = _document_tool_answer(
                file_id=file_id,
                question=question,
                material=material,
                resolved=resolved,
                passages=material.chunks,
                history=history,
                slot_time=slot_time,
            )
        elif passages or file_question.is_text_less_video(material):
            stream = _frames_question_stream(
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
        else:
            stream = _text_question_stream(
                file_id=file_id,
                question=question,
                material=material,
                resolved=resolved,
                position_seconds=position_seconds,
                history=history,
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
                filename=material.filename,
                folder=material.folder,
                type_label=material.type_label,
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
                    filename=material.filename,
                    folder=material.folder,
                    type_label=material.type_label,
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
#: The name a transcript request must call.
_TRANSCRIPT_TOOL_NAME = GET_TRANSCRIPT_TOOL["function"]["name"]
#: How many characters one transcript result may hold.
_TRANSCRIPT_RESULT_CHARS = 3000
#: What a transcript result says when its range holds no row.
_TRANSCRIPT_NOTHING = "Nothing is said or shown between {start} and {end}."
#: The note a transcript result ends with when a row does not fit.
_TRANSCRIPT_STOP = (
    "Text from {time} on was not sent: the limit of one call was reached."
)
#: The longest the stop note can be: its clock time is at most 8 characters.
_TRANSCRIPT_STOP_CHARS = len(_TRANSCRIPT_STOP.format(time="0" * 8))

#: The name a text request must call.
_GET_TEXT_TOOL_NAME = GET_TEXT_TOOL["function"]["name"]
#: The name a page request must call.
_GET_PAGE_IMAGES_TOOL_NAME = GET_PAGE_IMAGES_TOOL["function"]["name"]
#: How many characters one text result may hold.
_DOCUMENT_TEXT_CHARS = 3000
#: The note a text result ends with when a row does not fit.
_GET_TEXT_CONTINUES = (
    "The text continues: call get_text with after {number} for more."
)
#: The longest that note can be: its number is at most three digits.
_GET_TEXT_CONTINUES_CHARS = len(_GET_TEXT_CONTINUES.format(number=999))
#: What a text result says when the file has no row from the named place on.
_GET_TEXT_NOTHING = "Nothing is written from {place} on."
#: What a text result says when the walk started after the file's last row.
_GET_TEXT_ENDS = "The file ends after [{number}]."
#: What a text result ends with when the walk wrote the file's last row.
_GET_TEXT_FILE_ENDS = "The file ends here."
#: What a page request with bad arguments gets.
_PAGE_IMAGES_BAD_ARGUMENTS = (
    "No page image follows: get_page_images takes a JSON object with pages, a "
    "list of page numbers, or passages, a list of numbers of passages you were "
    "given; at most 4 pages in all."
)


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
    #: Whether this call spends one of the question's three tool calls.
    spends: bool
    #: What it is: ``"frames"``, ``"transcript"`` or neither.
    kind: str = "frames"
    #: The whole result text when nothing is cut or read at all.
    content: str = ""
    #: A served transcript request's range.
    span: tuple[float, float] | None = None
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
                filename=material.filename,
                folder=material.folder,
                type_label=material.type_label,
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
                filename=material.filename,
                folder=material.folder,
                type_label=material.type_label,
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
    three tool calls are served or the window has passed, the next request is
    final and its answer streams. Any failure before the first written chunk
    sends today's request once; nothing is started after the total deadline.
    A material that was cut also offers ``get_transcript``, which reads the
    text of a time range the passages left out.
    """
    loop = asyncio.get_running_loop()
    cut = material.partial and bool(material.rows)
    tools = (
        [GET_FRAMES_TOOL, GET_TRANSCRIPT_TOOL] if cut else [GET_FRAMES_TOOL]
    )
    first_number = len(passages) + 1
    initial_frames = [
        (
            f"[{first_number + index}] {format_position(frame.time)}",
            frame.data,
            frame.mime_type,
        )
        for index, frame in enumerate(frames)
    ]
    sent = [
        (frame.time, first_number + index)
        for index, frame in enumerate(frames)
    ]
    # What each number was sent with, filled in the order the numbers run.
    positions: dict[int, dict | None] = {
        index: chunk.position
        for index, chunk in enumerate(passages, start=1)
    }
    positions.update({
        first_number + index: {"seconds": frame.time}
        for index, frame in enumerate(frames)
    })
    # Each sent row's number and whether it was sent with its filtered text.
    sent_rows: dict[int, tuple[int, str]] = {
        row.order: (row.number, "filtered")
        for row in material.rows
        if row.number is not None
    }
    # The removed parts of a frame-text row already sent.
    sent_parts: set[str] = set()
    duration = material.duration
    system_prompt = build_file_question_system_prompt(
        resolved.profile.config.output_language,
        has_image=True,
        has_passages=bool(passages),
        has_history=bool(history),
        frames=True,
        tool=True,
        cut=cut,
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
        filename=material.filename,
        folder=material.folder,
        type_label=material.type_label,
    )
    rounds: list[llm.FrameRound] = []
    answer: list[str] = []
    next_number = first_number + len(frames)
    served = 0
    frame_calls = 0
    transcript_calls = 0
    requested = 0
    passages_sent = 0
    cancelled = threading.Event()

    def closing(text: str) -> list[AnswerEvent]:
        return [
            AnswerEvent(
                kind="citations",
                data=file_question.citations_from(text, positions),
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
                        tools=tools,
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
                    tools=tools,
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

            plans = _tool_plan(
                reply.calls,
                sent=sent,
                position=position,
                duration=duration,
                served=served,
                cut=cut,
            )
            wanted = sorted({time for plan in plans for time in plan.used})
            yield _FRAMES_KEEPALIVE
            cut_task = asyncio.create_task(
                file_question.cut_frames(
                    material, wanted, deadline=total_deadline,
                    cancelled=cancelled,
                )
            )
            cut_frames: list[file_question.Frame] = []
            try:
                async for event in _until_done(
                    cut_task, deadline=total_deadline,
                ):
                    yield event
                cut_frames = cut_task.result()
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

            by_time = {frame.time: frame for frame in cut_frames}
            # A reply's requested passages take their numbers before its
            # frames, whatever order the calls came in.
            transcript_results: dict[int, str] = {}
            try:
                for index, plan in enumerate(plans):
                    if plan.content or plan.kind != "transcript":
                        continue
                    span = plan.span
                    if span is None:
                        continue
                    content, next_number, added = _serve_transcript(
                        material,
                        span[0],
                        span[1],
                        sent_rows=sent_rows,
                        sent_parts=sent_parts,
                        next_number=next_number,
                    )
                    transcript_results[index] = content
                    for number, seconds in added:
                        positions[number] = {"seconds": seconds}
                        passages_sent += 1
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
            cut_per_plan: list[list[file_question.Frame]] = []
            round_frames: list[tuple[str, bytes, str]] = []
            for plan in plans:
                if plan.spends:
                    served += 1
                    if cut:
                        if plan.call.name == _FRAME_TOOL_NAME:
                            frame_calls += 1
                        elif plan.call.name == _TRANSCRIPT_TOOL_NAME:
                            transcript_calls += 1
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
                    positions[next_number] = {"seconds": frame.time}
                    sent = [*sent, (frame.time, next_number)]
                    next_number += 1
                    requested += 1
            results = []
            for index, (plan, frames_here) in enumerate(
                zip(plans, cut_per_plan)
            ):
                if index in transcript_results:
                    results = [
                        *results,
                        llm.FrameToolResult(
                            call=plan.call,
                            content=transcript_results[index],
                        ),
                    ]
                    continue
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
        if cut:
            if served:
                logger.info(
                    "tool requests: %d frame calls, %d frames, %d transcript "
                    "calls, %d passages (file %s)",
                    frame_calls,
                    requested,
                    transcript_calls,
                    passages_sent,
                    file_id,
                )
        elif served:
            logger.info(
                "frame requests: %d calls, %d frames (file %s)",
                served,
                requested,
                file_id,
            )


async def _document_tool_answer(
    *,
    file_id: str,
    question: str,
    material: file_question.Material,
    resolved: Resolved,
    passages: tuple[file_question.SelectedChunk, ...],
    history: list[file_question.HistoryTurn] | None,
    slot_time: float | None = None,
) -> AsyncIterator[str]:
    """Offer the document's tools while the question may, then answer.

    A tool request is read to its end before any of its text is written, so a
    reply that asks for more of the file never reaches the viewer as an
    answer. Once three tool calls are served or the window has passed, the
    next request is final and its answer streams. Any failure, refusal or
    blank reply before the first written chunk sends today's text request
    once; nothing is started after the total deadline.
    """
    loop = asyncio.get_running_loop()
    if slot_time is None:
        slot_time = loop.time()
    total_deadline = slot_time + _FRAME_TOOL_DEADLINE_SECONDS
    window_end = slot_time + _FRAME_TOOL_WINDOW_SECONDS
    pdf = material.mime_type == file_question.PDF_MIME
    cut = material.partial
    unit = "page" if pdf else "section"
    tools = _document_tools(pdf=pdf, cut=cut)
    offered = [tool["function"]["name"] for tool in tools]
    text_offered = _GET_TEXT_TOOL_NAME in offered
    pages_offered = _GET_PAGE_IMAGES_TOOL_NAME in offered
    system_prompt = build_file_question_system_prompt(
        resolved.profile.config.output_language,
        has_passages=bool(passages),
        has_history=bool(history),
        cut=cut,
        document=True,
        get_text=text_offered,
        page_images=pages_offered,
        unit=unit,
    )
    user_prompt = build_file_question_user_prompt(
        question,
        [chunk.text for chunk in passages],
        history=history,
        filename=material.filename,
        folder=material.folder,
        type_label=material.type_label,
        document=True,
        page_count=material.page_count if pdf else None,
        sections=material.sections,
        contents=file_question.document_contents(material) if cut else (),
    )
    # What each number was sent with, filled in the order the numbers run.
    positions: dict[int, dict | None] = {
        index: chunk.position
        for index, chunk in enumerate(passages, start=1)
    }
    # Each sent row's file order and the number it was sent with.
    sent_rows: dict[int, int] = {
        row.order: row.number
        for row in material.document_rows
        if row.number is not None
    }
    pages_taken: set[int] = set()
    next_number = len(passages) + 1
    rounds: list[llm.FrameRound] = []
    answer: list[str] = []
    served = 0
    text_calls = 0
    page_calls = 0
    passages_sent = 0
    pages_sent = 0

    def image_status() -> str:
        if pages_sent:
            return "attached"
        return "unavailable" if pages_taken else "none"

    def closing(text: str) -> list[AnswerEvent]:
        return [
            AnswerEvent(
                kind="citations",
                data=file_question.citations_from(text, positions),
            ),
            _document_meta(material, image_status()),
            AnswerEvent(kind="done", data={}),
        ]

    def failed() -> list[AnswerEvent]:
        return [
            _document_meta(material, image_status()),
            AnswerEvent(
                kind="done", data={"error": "Answer generation failed"},
            ),
        ]

    def deadline() -> list[AnswerEvent]:
        logger.warning("document_tool_deadline (file %s)", file_id)
        return failed()

    async def fallback() -> AsyncIterator[str]:
        yield _FRAMES_KEEPALIVE
        async for event in _document_fallback_answer(
            file_id=file_id,
            question=question,
            material=material,
            resolved=resolved,
            history=history,
            total_deadline=total_deadline,
            image_status="unavailable" if pages_taken else "none",
        ):
            yield event

    try:
        while True:
            yield _FRAMES_KEEPALIVE
            if loop.time() >= total_deadline:
                for event in deadline():
                    yield _format_sse_event(event)
                return

            if (
                served >= _FRAME_REQUESTS_PER_QUESTION
                or loop.time() >= window_end
            ):
                stream = None
                remaining = total_deadline - loop.time()
                if remaining <= 0:
                    for event in deadline():
                        yield _format_sse_event(event)
                    return
                try:
                    stream = resolved.client.stream_with_frames(
                        system_prompt,
                        user_prompt,
                        [],
                        rounds,
                        tools=tools,
                        tool_choice="none",
                    )
                    async with asyncio.timeout(remaining):
                        # A final answer that is only whitespace must reach the
                        # fallback with nothing of its own written first, so its
                        # leading whitespace is held until it shows something.
                        started = False
                        held: list[str] = []
                        async for delta in _answer_with_keepalive(stream):
                            if delta is None:
                                yield _FRAMES_KEEPALIVE
                                continue
                            if not started:
                                held = [*held, delta]
                                if not "".join(held).strip():
                                    continue
                                started = True
                                delta = "".join(held)
                                held = []
                            answer.append(delta)
                            yield _format_sse_event(
                                AnswerEvent(
                                    kind="answer_chunk",
                                    data={"delta": delta},
                                )
                            )
                except TimeoutError:
                    for event in deadline():
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
                    logger.warning(
                        "file question failed: %s (file %s)",
                        "EmptyAnswer",
                        file_id,
                    )
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
                    [],
                    rounds,
                    tools=tools,
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
                for event in deadline():
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

            blank = (
                reply is not None
                and not reply.text.strip()
                and not reply.calls
            )
            if failure is not None or blank or reply is None:
                if failure is not None:
                    logger.warning(
                        "file question document tools failed: %s (file %s)",
                        type(failure).__name__,
                        file_id,
                    )
                elif blank:
                    logger.warning(
                        "file question failed: %s (file %s)",
                        "EmptyAnswer",
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

            # Calls spend the question's tool calls in the order they came in.
            plans: list[tuple["llm.FrameToolCall", str]] = []
            taken = served
            for call in reply.calls:
                if taken >= _FRAME_REQUESTS_PER_QUESTION:
                    plans = [*plans, (call, "limit")]
                    continue
                taken += 1
                if call.name == _GET_TEXT_TOOL_NAME and text_offered:
                    plans = [*plans, (call, "text")]
                elif call.name == _GET_PAGE_IMAGES_TOOL_NAME and pages_offered:
                    plans = [*plans, (call, "pages")]
                else:
                    plans = [*plans, (call, "other")]
            served = taken

            yield _FRAMES_KEEPALIVE
            contents: dict[int, str] = {}
            round_images: list[tuple[str, bytes, str]] = []
            try:
                # A reply's requested passages take their numbers before its
                # page images, whatever order the calls came in.
                for index, (call, kind) in enumerate(plans):
                    if kind != "text":
                        continue
                    text_calls += 1
                    content, next_number, added = _serve_text_request(
                        material,
                        call,
                        sent_rows=sent_rows,
                        next_number=next_number,
                    )
                    contents[index] = content
                    for number, position in added:
                        positions[number] = position
                        passages_sent += 1
                for index, (call, kind) in enumerate(plans):
                    if kind != "pages":
                        continue
                    page_calls += 1
                    content, images = _serve_page_request(
                        call, taken=pages_taken,
                    )
                    contents[index] = content
                    for page, image in images:
                        round_images = [
                            *round_images,
                            (f"[{next_number}] page {page}", image, "image/jpeg"),
                        ]
                        positions[next_number] = {"page": page}
                        next_number += 1
                        pages_sent += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the fallback answers
                logger.warning(
                    "file question document tools failed: %s (file %s)",
                    type(exc).__name__,
                    file_id,
                )
                async for event in fallback():
                    yield event
                return

            results: list[llm.FrameToolResult] = []
            for index, (call, kind) in enumerate(plans):
                if kind in ("text", "pages"):
                    content = contents[index]
                elif kind == "other":
                    content = _document_cannot_serve(call.name, offered)
                else:
                    content = _document_limit_text()
                results = [
                    *results, llm.FrameToolResult(call=call, content=content),
                ]
            rounds = [
                *rounds,
                llm.FrameRound(
                    text=reply.text,
                    results=tuple(results),
                    frames=tuple(round_images),
                ),
            ]
    finally:
        if served:
            logger.info(
                "document tool requests: %d text calls, %d passages, %d page "
                "calls, %d pages (file %s)",
                text_calls,
                passages_sent,
                page_calls,
                pages_sent,
                file_id,
            )


def _document_tools(*, pdf: bool, cut: bool) -> list[dict]:
    """The tools a document's question offers, in the order it names them."""
    tools: list[dict] = []
    if cut:
        tools = [*tools, GET_TEXT_TOOL if pdf else GET_TEXT_SECTION_TOOL]
    if pdf:
        tools = [*tools, GET_PAGE_IMAGES_TOOL]
    return tools


def _document_meta(
    material: file_question.Material, image_status: str,
) -> AnswerEvent:
    """The ``meta`` event of a document's answer."""
    return AnswerEvent(
        kind="meta",
        data={"partial": material.partial, "image": image_status},
    )


def _document_cannot_serve(name: str, offered: Sequence[str]) -> str:
    """What a call of a tool the question does not offer gets."""
    called = f"{name} cannot" if name else "this call cannot"
    return (
        f"Nothing follows: {called} be served. Only "
        f"{' and '.join(offered)} are available."
    )


def _document_limit_text() -> str:
    return (
        "Nothing follows: this question has already been served its "
        f"{_FRAME_REQUESTS_PER_QUESTION} tool calls."
    )


def _get_text_bad_arguments(material: file_question.Material) -> str:
    """What a text request with bad arguments gets."""
    if material.mime_type == file_question.PDF_MIME:
        return (
            "No text follows: get_text takes a JSON object with either page, "
            f"a page number from 1 to {material.page_count or 0}, or after, the "
            "number of a passage you were given."
        )
    first, last = material.sections or (0, 0)
    return (
        "No text follows: get_text takes a JSON object with either section, "
        f"a section number from {first} to {last}, or after, the number of a "
        "passage you were given."
    )


def _unit_in_range(material: file_question.Material, number: int) -> bool:
    """Whether a PDF page or EPUB section is inside the file's own range."""
    if material.mime_type == file_question.PDF_MIME:
        return 1 <= number <= (material.page_count or 0)
    if material.sections is None:
        return False
    return material.sections[0] <= number <= material.sections[1]


def _text_request_start(
    material: file_question.Material,
    arguments: object,
    *,
    unit: str,
    sent_rows: dict[int, int],
) -> tuple[int | None, str | None]:
    """The row a text request starts at, or the text that answers it instead."""
    rows = material.document_rows
    named: int | None = None
    after: int | None = None
    if isinstance(arguments, dict):
        values = [
            (key, value)
            for key, value in (
                (unit, arguments.get(unit)),
                ("after", arguments.get("after")),
            )
            if value is not None
        ]
        if len(values) == 1:
            key, value = values[0]
            if key == "after":
                after = file_question.read_integer(value)
            else:
                named = file_question.read_integer(value)
    if named is None and after is None:
        return None, _get_text_bad_arguments(material)
    if named is not None:
        if not _unit_in_range(material, named):
            return None, _get_text_bad_arguments(material)
        # Rows with no unit are passed over in this search.
        for position, row in enumerate(rows):
            if row.unit is not None and row.unit >= named:
                return position, None
        return None, _GET_TEXT_NOTHING.format(
            place=f"{unit} {named}",
        )
    orders = {
        row.order for row in rows if sent_rows.get(row.order) == after
    }
    if not orders:
        return None, _get_text_bad_arguments(material)
    for position, row in enumerate(rows):
        if row.order in orders:
            if position + 1 >= len(rows):
                return None, _GET_TEXT_ENDS.format(number=after)
            return position + 1, None
    return None, _get_text_bad_arguments(material)


def _unit_marker(
    rows: Sequence[file_question.DocumentRow],
    position: int,
    *,
    unit: str,
    first: bool,
) -> str | None:
    """The line a row of a text result carries before it, if it carries one."""
    row = rows[position]
    if row.unit is None:
        return None
    starts = position == 0 or rows[position - 1].unit != row.unit
    if first and not starts:
        return f"Continuing {unit} {row.unit}."
    if not starts:
        return None
    return f"{unit.capitalize()} {row.unit} begins."


def _serve_text_request(
    material: file_question.Material,
    call: "llm.FrameToolCall",
    *,
    sent_rows: dict[int, int],
    next_number: int,
) -> tuple[str, int, list[tuple[int, dict | None]]]:
    """One text result, the next number, and the passages it sent.

    ``sent_rows`` maps a row's file order to the number it was sent with, and
    grows with every row this result sends. A row that was already sent is
    named by its number instead of being sent again.
    """
    pdf = material.mime_type == file_question.PDF_MIME
    unit = "page" if pdf else "section"
    rows = material.document_rows
    start, content = _text_request_start(
        material, _call_arguments(call), unit=unit, sent_rows=sent_rows,
    )
    if content is not None:
        return content, next_number, []
    assert start is not None
    lines: list[str] = []
    added: list[tuple[int, dict | None]] = []
    last = 0
    stopped = False
    for position in range(start, len(rows)):
        row = rows[position]
        marker = _unit_marker(
            rows, position, unit=unit, first=position == start,
        )
        number = sent_rows.get(row.order)
        if number is not None:
            group = [f"already sent as [{number}]"]
        else:
            group = [f"[{next_number}]", row.text, ""]
        if marker is not None:
            group = [marker, *group]
        # A trailing empty line is dropped before a closing note is joined, so
        # it must not count against the result's length here.
        written = [*lines, *group]
        while written and written[-1] == "":
            written = written[:-1]
        # The walk's first row is always written, however long it is.
        if position != start and (
            len("\n".join(written)) + 1 + _GET_TEXT_CONTINUES_CHARS
            > _DOCUMENT_TEXT_CHARS
        ):
            stopped = True
            break
        lines = [*lines, *group]
        if number is None:
            added = [*added, (next_number, row.position)]
            sent_rows[row.order] = next_number
            last = next_number
            next_number += 1
        else:
            last = number
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if stopped:
        lines = [*lines, _GET_TEXT_CONTINUES.format(number=last)]
    else:
        lines = [*lines, _GET_TEXT_FILE_ENDS]
    return "\n".join(lines), next_number, added


def _page_images_text(images: int, notes: Sequence[str]) -> str:
    """A page request's result: how many images follow, then a line per note."""
    head = (
        f"{images} page image{'s' if images != 1 else ''} follow."
        if images
        else "No page image follows."
    )
    return "\n".join([head, *[f"- {note}" for note in notes]])


def _serve_page_request(
    call: "llm.FrameToolCall",
    *,
    taken: set[int],
) -> tuple[str, list[tuple[int, bytes]]]:
    """One page request's result text and the page images it sends.

    A page the request names is taken; no page image is read, so a taken page
    has none to send and is named as not available.
    """
    arguments = _call_arguments(call)
    if not isinstance(arguments, dict) or not any(
        isinstance(arguments.get(key), list) for key in ("pages", "passages")
    ):
        return _PAGE_IMAGES_BAD_ARGUMENTS, []
    notes: list[str] = []
    for key in ("pages", "passages"):
        for value in arguments.get(key) or []:
            page = file_question.read_integer(value)
            if page is None:
                notes = [
                    *notes, f"{_shown_value(value)}: not a number I can read",
                ]
                continue
            taken.add(page)
            notes = [*notes, f"page {page}: not available"]
    return _page_images_text(0, notes), []


async def _document_fallback_answer(
    *,
    file_id: str,
    question: str,
    material: file_question.Material,
    resolved: Resolved,
    history: list[file_question.HistoryTurn] | None,
    total_deadline: float,
    image_status: str,
) -> AsyncIterator[str]:
    """Send today's text request once, unless the total deadline has passed.

    It carries no requested passage and no page image, and its citations name
    the passages the question first sent.
    """
    logger.warning("document_tool_fallback (file %s)", file_id)
    loop = asyncio.get_running_loop()
    remaining = total_deadline - loop.time()
    if remaining <= 0:
        logger.warning("document_tool_deadline (file %s)", file_id)
        yield _format_sse_event(_document_meta(material, image_status))
        yield _format_sse_event(
            AnswerEvent(kind="done", data={"error": "Answer generation failed"})
        )
        return
    stream = None
    answer: list[str] = []
    try:
        stream = resolved.client.generate_stream(
            build_file_question_system_prompt(
                resolved.profile.config.output_language,
                has_passages=bool(material.chunks),
                has_history=bool(history),
            ),
            build_file_question_user_prompt(
                question,
                [chunk.text for chunk in material.chunks],
                history=history,
                filename=material.filename,
                folder=material.folder,
                type_label=material.type_label,
            ),
            max_tokens_override=settings.rag.max_tokens,
        )
        async with asyncio.timeout(remaining):
            async for event in _keepalive_answer(stream, answer):
                yield event
    except TimeoutError:
        logger.warning("document_tool_deadline (file %s)", file_id)
        yield _format_sse_event(_document_meta(material, image_status))
        yield _format_sse_event(
            AnswerEvent(kind="done", data={"error": "Answer generation failed"})
        )
        return
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - the stream must end, not die
        logger.warning(
            "file question failed: %s (file %s)",
            type(exc).__name__,
            file_id,
        )
        yield _format_sse_event(_document_meta(material, image_status))
        yield _format_sse_event(
            AnswerEvent(kind="done", data={"error": "Answer generation failed"})
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
        yield _format_sse_event(_document_meta(material, image_status))
        yield _format_sse_event(
            AnswerEvent(kind="done", data={"error": "Answer generation failed"})
        )
        return
    yield _format_sse_event(
        AnswerEvent(
            kind="citations",
            data=file_question.citations(
                "".join(answer), material.chunks,
            ),
        )
    )
    yield _format_sse_event(_document_meta(material, image_status))
    yield _format_sse_event(AnswerEvent(kind="done", data={}))


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


def _call_arguments(call: "llm.FrameToolCall") -> object:
    """A tool call's arguments as the JSON value they were written as."""
    arguments = call.arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except ValueError:
            return None
    return arguments


def _call_times(call: "llm.FrameToolCall") -> list | None:
    """The ``times`` list of a ``get_frames`` call, or ``None`` for another call."""
    if call.name != _FRAME_TOOL_NAME:
        return None
    arguments = _call_arguments(call)
    if not isinstance(arguments, dict):
        return None
    times = arguments.get("times")
    return times if isinstance(times, list) else None


def _transcript_span(call: "llm.FrameToolCall") -> tuple[float, float] | None:
    """The range of a ``get_transcript`` call, or ``None`` for bad arguments."""
    arguments = _call_arguments(call)
    if not isinstance(arguments, dict):
        return None
    start = file_question.read_time(arguments.get("start"))
    end = file_question.read_time(arguments.get("end"))
    if start is None or end is None or start >= end:
        return None
    return start, end


def _tool_plan(
    calls: Sequence["llm.FrameToolCall"],
    *,
    sent: Sequence[tuple[float, int]],
    position: float | None,
    duration: float | None,
    served: int,
    cut: bool = False,
) -> list[_CallPlan]:
    """Decide, for every call of one reply, what it gets before anything is cut.

    ``cut`` offers the transcript tool as well: a call that names it is served
    by reading the material where a frame call is served by cutting frames.
    """
    plans: list[_CallPlan] = []
    taken = served
    used_here: list[float] = []
    for call in calls:
        if taken >= _FRAME_REQUESTS_PER_QUESTION:
            plans = [
                *plans,
                _CallPlan(call=call, spends=False, content=_limit_text(cut)),
            ]
            continue
        taken += 1
        if cut and call.name == _TRANSCRIPT_TOOL_NAME:
            span = _transcript_span(call)
            plans = [
                *plans,
                _CallPlan(
                    call=call,
                    spends=True,
                    kind="transcript" if span is not None else "other",
                    content="" if span is not None else _transcript_bad_call_text(),
                    span=span,
                ),
            ]
            continue
        times = _call_times(call)
        if times is None:
            plans = [
                *plans,
                _CallPlan(
                    call=call,
                    spends=True,
                    kind=(
                        "frames" if call.name == _FRAME_TOOL_NAME else "other"
                    ),
                    content=_bad_call_text(call.name, cut),
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


def _limit_text(cut: bool = False) -> str:
    if cut:
        return (
            "No frame or text follows: this question has already been served "
            f"its {_FRAME_REQUESTS_PER_QUESTION} tool calls."
        )
    return (
        "No frame follows: this question has already been served its "
        f"{_FRAME_REQUESTS_PER_QUESTION} frame requests."
    )


def _bad_call_text(name: str, cut: bool = False) -> str:
    called = f"{name} cannot" if name else "this call cannot"
    if cut:
        return (
            f"No frame or text follows: {called} be served. Only "
            f"{_FRAME_TOOL_NAME} (a list of times named times, at most "
            f"{_FRAMES_PER_REQUEST}) and {_TRANSCRIPT_TOOL_NAME} (start and "
            "end) are available."
        )
    return (
        f"No frame follows: {called} be served. Only {_FRAME_TOOL_NAME} takes a "
        "JSON object with a list of times named times, at most "
        f"{_FRAMES_PER_REQUEST} of them."
    )


def _transcript_bad_call_text() -> str:
    """What a ``get_transcript`` call with bad arguments gets."""
    return (
        f"No text follows: {_TRANSCRIPT_TOOL_NAME} takes a JSON object with "
        "start and end, two times with start before end."
    )


def _row_overlaps(
    row: file_question.MaterialRow, start: float, end: float,
) -> bool:
    """Whether the row's span, or its start alone, overlaps ``[start, end)``."""
    row_end = row.start if row.end is None else row.end
    return row.start < end and row_end > start


def _time_order(row: file_question.MaterialRow) -> tuple[float, int, int]:
    """Time order: start, then a transcript row before frame text, then file order."""
    return (row.start, 0 if row.kind == "transcript" else 1, row.order)


def _stop_note(start: float) -> str:
    return _TRANSCRIPT_STOP.format(time=format_position(start))


def _serve_transcript(
    material: file_question.Material,
    start: float,
    end: float,
    *,
    sent_rows: dict[int, tuple[int, str]],
    sent_parts: set[str],
    next_number: int,
) -> tuple[str, int, list[tuple[int, float]]]:
    """One transcript result, the next number, and the passages it sent.

    ``sent_rows`` maps a row's file order to the number it was sent with and
    whether that was its filtered or its stored text; ``sent_parts`` holds the
    removed parts already sent. A row of the range's first segment is written
    with its stored text; a later one with its filtered text, and one the
    filter removed whole is skipped.
    """
    rows = sorted(
        (row for row in material.rows if _row_overlaps(row, start, end)),
        key=_time_order,
    )
    if not rows:
        return (
            _TRANSCRIPT_NOTHING.format(
                start=format_position(start), end=format_position(end),
            ),
            next_number,
            [],
        )
    frame_starts = [row.start for row in rows if row.kind == "frame_text"]
    first_start = min(frame_starts) if frame_starts else None
    lines: list[str] = []
    added: list[tuple[int, float]] = []
    for row in rows:
        first = row.kind == "frame_text" and row.start == first_start
        sent = sent_rows.get(row.order)
        supplement: list[str] = []
        if sent is not None:
            number, sent_as = sent
            group = [f"already sent as [{number}]"]
            if first and sent_as == "filtered":
                supplement = [
                    part for part in row.removed
                    if part.strip() not in sent_parts
                ]
            if supplement:
                group = [*group, f"[{next_number}]", "\n".join(supplement), ""]
        else:
            if row.kind == "frame_text" and not first and row.text is None:
                continue
            text = row.stored_text if first else row.text
            group = [f"[{next_number}]", "" if text is None else text, ""]
        candidate = "\n".join([*lines, *group])
        # The stop note is joined to the result with a newline of its own.
        if len(candidate) + 1 + _TRANSCRIPT_STOP_CHARS > _TRANSCRIPT_RESULT_CHARS:
            lines = [*lines, _stop_note(row.start)]
            break
        if sent is None:
            sent_rows[row.order] = (
                next_number, "stored" if first else "filtered",
            )
            added = [*added, (next_number, row.start)]
            next_number += 1
        elif supplement:
            sent_parts.update(part.strip() for part in supplement)
            added = [*added, (next_number, row.start)]
            next_number += 1
        lines = [*lines, *group]
    else:
        if lines and lines[-1] == "":
            lines = lines[:-1]
    return "\n".join(lines), next_number, added


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
