"""``get_file_chunks`` tool wrapper.

Two backends:

* ``type="transcript"`` — reads ``TranscriptChunk`` rows from the
  intelligence DB. Returns chunks in ``chunk_index`` order.
* ``type="text"`` — fetches ``GET /api/internal/files/{id}/content``
  from the core (gated by ``CORE_INTERNAL_SECRET``), then splits the
  body into pseudo-chunks of fixed character size so the LLM can ask
  for ranges symmetrically with transcript-backed files.

Two modes:

* ``summary`` (default) — return every chunk's ``chunk_id`` + location
  + 200-char preview. Lets the LLM scout cheaply before deciding
  what to drill into.
* ``full`` — return text for a specific range; ``range`` is required
  and capped at 50 chunks. Out-of-range or oversize asks are
  auto-truncated and a warning is included.

A per-call token cap is applied as a final defence: if the encoded
payload would exceed it, the chunks list is truncated and the
envelope is marked ``truncated=True`` with a warning describing how
much was dropped.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

from app.database import get_search_db_read
from app.models import TranscriptChunk
from app.rag.tools._access import ensure_access, is_valid_file_id
from app.rag.tools.budget import (
    DEFAULT_PER_CALL_TOKEN_CAP,
    estimate_payload_tokens,
    estimate_tokens,
)
from app.rag.tools.context import ToolContext, ToolResultEnvelope

logger = logging.getLogger(__name__)


_INTERNAL_API_BASE_URL_DEFAULT = "http://backend:8000/api/internal"
_INTERNAL_API_TIMEOUT_SECONDS = 10.0  # text content can be up to 10 MB

# Synthetic chunk size for ``type="text"`` (no transcript chunks
# exist). ~800 chars ≈ 200-400 tokens depending on script; the
# preview length matches the ``summary`` mode contract.
_TEXT_CHUNK_CHARS = 800

# ``full`` mode hard cap. Matches spec §2.2.
_FULL_MAX_CHUNKS = 50

# ``summary`` mode preview length per chunk. The spec §2.6 originally
# picked 200; in practice long-form transcripts (40+ chunks) blow the
# per-call cap and force the LLM to digest 8K+ tokens before its next
# move. 80 chars keeps a 40-chunk video summary under ~4K tokens
# (enough headroom for Qwen3:8b on a 32K context).
_SUMMARY_PREVIEW_CHARS = 80

# Hard cap on a single chunk's text. WhisperX transcript chunks have
# no upstream size bound, so an LLM repeatedly fetching a giant chunk
# could overrun the loop's cumulative-token budget before the
# per-call cap kicks in. 32 KB ≈ 8000 tokens of CJK text — still
# generous, just not unbounded.
_MAX_PER_CHUNK_TEXT_CHARS = 32_000


@dataclass(frozen=True)
class _ChunkRow:
    chunk_id: int
    location: str
    text: str


def _base_url() -> str:
    return os.environ.get(
        "HOMEVAULT_INTERNAL_API_URL", _INTERNAL_API_BASE_URL_DEFAULT
    )


def _internal_secret() -> str | None:
    return os.environ.get("CORE_INTERNAL_SECRET") or None


def _format_time(seconds: float) -> str:
    """``m:ss`` formatter shared with the legacy citation builder."""
    if seconds < 0:
        seconds = 0
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def _truncate_chunk_text(text: str) -> tuple[str, bool]:
    """Cap a single chunk's text at ``_MAX_PER_CHUNK_TEXT_CHARS`` chars.

    Returns the (possibly clipped) text plus a flag indicating whether
    truncation happened. Centralised so transcript / text-content
    paths share the same upper bound.
    """
    if len(text) <= _MAX_PER_CHUNK_TEXT_CHARS:
        return text, False
    return text[:_MAX_PER_CHUNK_TEXT_CHARS], True


def _load_transcript_chunks(file_id: str) -> tuple[list[_ChunkRow], bool]:
    """Return ordered transcript chunks for ``file_id``.

    Second element of the tuple is True when at least one chunk's
    text exceeded ``_MAX_PER_CHUNK_TEXT_CHARS`` and was truncated
    individually — the caller surfaces this in the envelope's warnings.
    """
    try:
        with get_search_db_read() as db:
            rows = (
                db.query(TranscriptChunk)
                .filter(TranscriptChunk.file_id == file_id)
                .order_by(TranscriptChunk.chunk_index.asc())
                .all()
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "get_file_chunks transcript fetch failed for %s: %s",
            file_id,
            type(exc).__name__,
        )
        return [], False
    any_truncated = False
    out: list[_ChunkRow] = []
    for r in rows:
        text, t = _truncate_chunk_text(r.text)
        if t:
            any_truncated = True
        out.append(
            _ChunkRow(
                chunk_id=r.chunk_index,
                location=_format_time(r.timestamp_start),
                text=text,
            )
        )
    return out, any_truncated


class _TextFetchError(Exception):
    """Raised when the host's text-content endpoint reports an
    operational failure (5xx, network, malformed). 404 / 415 / 413 are
    domain-level "no usable content" answers and return an empty list
    instead — they are not operational errors."""


async def _load_text_chunks(file_id: str) -> list[_ChunkRow]:
    secret = _internal_secret()
    headers: dict[str, str] = {}
    if secret:
        headers["x-internal-secret"] = secret

    url = f"{_base_url().rstrip('/')}/files/{file_id}/content"
    try:
        async with httpx.AsyncClient(
            timeout=_INTERNAL_API_TIMEOUT_SECONDS
        ) as client:
            response = await client.get(url, headers=headers)
            # 404 (not found / missing) / 415 (wrong mime) / 413 (too
            # large) are domain answers — the LLM should learn that
            # this file has no readable body and stop.
            if response.status_code in (404, 415, 413):
                return []
            # 401 / 403 reveal a CORE_INTERNAL_SECRET rotation problem;
            # fail-loud so the agentic loop's outer handler can abort
            # and the operator's monitoring picks it up.
            if response.status_code in (401, 403):
                raise _TextFetchError(
                    f"internal API auth failed ({response.status_code})"
                )
            response.raise_for_status()
            text = response.text
    except httpx.HTTPError as exc:
        # 5xx / network / timeout = operational. Surface upward so the
        # agentic loop treats it like a fail-loud tool error rather
        # than "this file is empty" (hako TtOsHILDUcbcyCghciY-9).
        raise _TextFetchError(
            f"internal API call failed: {type(exc).__name__}"
        ) from exc

    return _split_text_into_chunks(text)


def _split_text_into_chunks(text: str) -> list[_ChunkRow]:
    if not text:
        return []
    chunks: list[_ChunkRow] = []
    idx = 0
    pos = 0
    # Floor on chunk width — without it, a pathological file with a
    # newline every few chars produces thousands of micro-chunks that
    # bypass the token cap through cardinality.
    min_width = _TEXT_CHUNK_CHARS // 4
    while pos < len(text):
        end = min(pos + _TEXT_CHUNK_CHARS, len(text))
        # Prefer to break on a newline within the last 100 chars to
        # avoid mid-sentence cuts. Cheap, no NLP.
        if end < len(text):
            nl = text.rfind("\n", pos + _TEXT_CHUNK_CHARS - 100, end)
            if nl != -1 and (nl + 1) - pos >= min_width:
                end = nl + 1
        chunk_text, _ = _truncate_chunk_text(text[pos:end])
        chunks.append(
            _ChunkRow(
                chunk_id=idx,
                location=f"chunk {idx}",
                text=chunk_text,
            )
        )
        idx += 1
        pos = end
    return chunks


def _clip_range(
    chunks: list[_ChunkRow], rng: tuple[int, int]
) -> tuple[list[_ChunkRow], bool, str | None]:
    """Slice ``chunks`` to the inclusive ``[lo, hi]`` range.

    Auto-truncates oversize / out-of-bounds asks. Returns the slice
    plus a (truncated, warning) tuple so the envelope can carry the
    diagnostic back to the LLM.
    """
    if not chunks:
        return [], False, None
    lo_raw, hi_raw = rng
    lo = max(0, min(int(lo_raw), len(chunks) - 1))
    hi = max(lo, min(int(hi_raw), len(chunks) - 1))
    width = hi - lo + 1
    truncated = False
    warning: str | None = None
    if width > _FULL_MAX_CHUNKS:
        hi = lo + _FULL_MAX_CHUNKS - 1
        truncated = True
        warning = (
            f"range capped at {_FULL_MAX_CHUNKS} chunks; "
            f"requested {width}, returned {_FULL_MAX_CHUNKS}"
        )
    return chunks[lo : hi + 1], truncated, warning


def _summary_rows(chunks: list[_ChunkRow]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": c.chunk_id,
            "location": c.location,
            "preview": c.text[:_SUMMARY_PREVIEW_CHARS],
        }
        for c in chunks
    ]


def _full_rows(chunks: list[_ChunkRow]) -> list[dict[str, Any]]:
    return [
        {"chunk_id": c.chunk_id, "location": c.location, "text": c.text}
        for c in chunks
    ]


def _apply_token_cap(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool, str | None]:
    """Drop trailing rows until the payload fits the per-call cap.

    Returns the (possibly shortened) list plus a (truncated, warning)
    tuple. The warning is only set when at least one row was dropped.
    """
    if not rows:
        return rows, False, None
    if estimate_payload_tokens(rows) <= DEFAULT_PER_CALL_TOKEN_CAP:
        return rows, False, None

    kept: list[dict[str, Any]] = []
    cumulative = 0
    for row in rows:
        cost = estimate_tokens(row.get("text") or row.get("preview") or "")
        if kept and cumulative + cost > DEFAULT_PER_CALL_TOKEN_CAP:
            break
        kept.append(row)
        cumulative += cost
        if cumulative > DEFAULT_PER_CALL_TOKEN_CAP:
            # First row alone exceeds the cap; return it so the LLM
            # sees something rather than an empty list, but stop here.
            break

    dropped = len(rows) - len(kept)
    warning = (
        f"per-call token cap exceeded; "
        f"returned {len(kept)} of {len(rows)} chunks"
    )
    return kept, dropped > 0, warning


async def get_file_chunks(
    *,
    context: ToolContext,
    file_id: str,
    type: str,
    mode: str = "summary",
    range: tuple[int, int] | list[int] | None = None,
) -> ToolResultEnvelope:
    context.register_tool_call("get_file_chunks")

    if type not in ("transcript", "text"):
        return ToolResultEnvelope(
            payload={"error": "type must be 'transcript' or 'text'"},
            token_estimate=0,
            warning="invalid type",
        )
    if mode not in ("summary", "full"):
        return ToolResultEnvelope(
            payload={"error": "mode must be 'summary' or 'full'"},
            token_estimate=0,
            warning="invalid mode",
        )
    if not is_valid_file_id(file_id):
        return ToolResultEnvelope(
            payload={"error": "invalid file_id"},
            token_estimate=0,
            warning="invalid file_id",
        )

    # Access gate. Note that for type='text' the host's /content
    # endpoint trusts the shared secret only — without this check, the
    # LLM could exfiltrate any .md / .txt body on the box.
    allowed = await ensure_access([file_id], lit_token=context.lit_token)
    if file_id not in allowed:
        return ToolResultEnvelope(
            payload={"file_id": file_id, "error": "not_found"},
            token_estimate=0,
            warning="file not found",
        )

    per_chunk_truncated = False
    text_fetch_failed = False
    if type == "transcript":
        chunks, per_chunk_truncated = await asyncio.to_thread(
            _load_transcript_chunks, file_id
        )
    else:
        try:
            chunks = await _load_text_chunks(file_id)
        except _TextFetchError as exc:
            logger.warning("get_file_chunks text fetch error: %s", exc)
            chunks = []
            text_fetch_failed = True

    if not chunks:
        payload_empty: dict[str, Any] = {
            "file_id": file_id,
            "type": type,
            "mode": mode,
            "chunks": [],
            "total_chunks": 0,
        }
        if text_fetch_failed:
            payload_empty["error"] = "internal_api_failed"
        return ToolResultEnvelope(
            payload=payload_empty,
            token_estimate=estimate_payload_tokens(payload_empty),
            warning=(
                "text fetch failed; the loop should not retry this file"
                if text_fetch_failed
                else None
            ),
        )

    truncated = False
    warnings: list[str] = []
    if per_chunk_truncated:
        truncated = True
        warnings.append(
            f"one or more chunks exceeded the per-chunk text cap "
            f"({_MAX_PER_CHUNK_TEXT_CHARS} chars) and were clipped"
        )

    if mode == "summary":
        rows = _summary_rows(chunks)
    else:
        # ``full`` requires range — the schema marks it as required only
        # when mode='full', but enforcement is here because OpenAI does
        # not gate one field on another.
        if range is None:
            return ToolResultEnvelope(
                payload={
                    "error": "range required when mode='full'",
                    "total_chunks": len(chunks),
                },
                token_estimate=0,
                warning="range required when mode='full'",
            )
        try:
            rng = (int(range[0]), int(range[1]))
        except (TypeError, ValueError, IndexError):
            return ToolResultEnvelope(
                payload={"error": "range must be [start, end] integers"},
                token_estimate=0,
                warning="invalid range",
            )
        clipped, range_truncated, range_warning = _clip_range(chunks, rng)
        if range_truncated:
            truncated = True
            if range_warning:
                warnings.append(range_warning)
        rows = _full_rows(clipped)

    rows, cap_truncated, cap_warning = _apply_token_cap(rows)
    if cap_truncated:
        truncated = True
        if cap_warning:
            warnings.append(cap_warning)

    payload: dict[str, Any] = {
        "file_id": file_id,
        "type": type,
        "mode": mode,
        "chunks": rows,
        "total_chunks": len(chunks),
    }
    if truncated:
        payload["truncated"] = True
        payload["warnings"] = warnings

    context.register_file_ids([file_id])
    tokens = estimate_payload_tokens(payload)
    context.register_result_tokens(tokens)

    return ToolResultEnvelope(
        payload=payload,
        token_estimate=tokens,
        truncated=truncated,
        warning="; ".join(warnings) if warnings else None,
    )


__all__ = ["get_file_chunks"]
