"""``get_file_detail`` tool wrapper.

Stitches three sources into a single response:

1. ``GET /api/internal/files/{id}`` — basic file metadata (drive,
   filename, file_type, folder_path).
2. ``POST /api/internal/files/bulk`` (via ``file_hydrate``) — richer
   shape used by the frontend (tags, file_size, duration, etc.).
3. ``GET /api/internal/file_relations`` — summarised as ``{kind: count}``
   so the LLM can decide whether to follow up with
   ``get_related_files`` for a specific kind.

Plus two intelligence-DB lookups computed locally:

* ``has_transcript`` — ``True`` iff at least one ``TranscriptChunk``
  exists for ``file_id``.
* ``chunk_count`` — number of transcript chunks (legacy text chunk
  counts are not tracked separately yet; the spec accepts that this
  field reflects transcript-only chunks in Phase 1.B).

Tier 3 fields (auto_tags-suggested, detailed_summary) are NOT
returned. The schema docstring is explicit about this so future
edits do not accidentally re-introduce them.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from app.database import get_search_db_read
from app.file_hydrate import hydrate_files
from app.models import TranscriptChunk
from app.rag.tools._access import ensure_access, is_valid_file_id
from app.rag.tools.budget import estimate_payload_tokens
from app.rag.tools.context import ToolContext, ToolResultEnvelope

# Explicit allow-list of fields surfaced from the host's bulk-hydrate
# response. ``file_hydrate`` returns the FileResponse shape verbatim;
# without an explicit projection, a future host change that adds Tier 3
# fields (auto_tags_suggested / detailed_summary) would leak silently.
_HYDRATE_ALLOWED_FIELDS = frozenset(
    {"title", "tags", "mime_type", "file_size", "duration"}
)

logger = logging.getLogger(__name__)


_INTERNAL_API_BASE_URL_DEFAULT = "http://backend:8000/api/internal"
_INTERNAL_API_TIMEOUT_SECONDS = 5.0


def _base_url() -> str:
    return os.environ.get(
        "HOMEVAULT_INTERNAL_API_URL", _INTERNAL_API_BASE_URL_DEFAULT
    )


async def _fetch_basic_metadata(file_id: str) -> dict[str, Any] | None:
    url = f"{_base_url().rstrip('/')}/files/{file_id}"
    try:
        async with httpx.AsyncClient(
            timeout=_INTERNAL_API_TIMEOUT_SECONDS
        ) as client:
            response = await client.get(url)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        logger.warning(
            "get_file_detail: basic metadata fetch failed for %s: %s",
            file_id,
            type(exc).__name__,
        )
        return None
    if not isinstance(data, dict):
        return None
    return data


async def _fetch_relations_summary(file_id: str) -> dict[str, int]:
    url = f"{_base_url().rstrip('/')}/file_relations"
    try:
        async with httpx.AsyncClient(
            timeout=_INTERNAL_API_TIMEOUT_SECONDS
        ) as client:
            response = await client.get(url, params={"file_id": file_id})
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        logger.warning(
            "get_file_detail: file_relations fetch failed for %s: %s",
            file_id,
            type(exc).__name__,
        )
        return {}
    summary: dict[str, int] = {}
    if isinstance(data, list):
        for r in data:
            if not isinstance(r, dict):
                continue
            kind = r.get("kind")
            if isinstance(kind, str):
                summary[kind] = summary.get(kind, 0) + 1
    return summary


def _count_transcript_chunks(file_id: str) -> int:
    """Count transcript chunks for ``file_id`` in the intelligence DB.

    Sync helper — the caller wraps it with ``asyncio.to_thread`` to
    keep the event loop free, matching the pattern other RAG modules
    use for sync DB calls.
    """
    try:
        with get_search_db_read() as db:
            return (
                db.query(TranscriptChunk)
                .filter(TranscriptChunk.file_id == file_id)
                .count()
            )
    except Exception as exc:  # noqa: BLE001 — DB error must not crash the loop
        logger.warning(
            "get_file_detail: chunk count failed for %s: %s",
            file_id,
            type(exc).__name__,
        )
        return 0


async def get_file_detail(
    *,
    context: ToolContext,
    file_id: str,
) -> ToolResultEnvelope:
    """Return enriched file metadata for ``file_id``.

    Tier 3 fields (auto_tags-suggested, detailed_summary) are NOT
    returned. The schema lists every field explicitly so future edits
    don't accidentally re-introduce them.
    """
    import asyncio

    context.register_tool_call("get_file_detail")

    if not is_valid_file_id(file_id):
        return ToolResultEnvelope(
            payload={"error": "invalid file_id"},
            token_estimate=0,
            truncated=False,
            warning="invalid file_id",
        )

    # Access gate: untrusted LLM input cannot reach the host's Internal
    # API for a file the viewer cannot see. Mirrors the 404-not-403
    # existence-hiding rule (design-decisions.md).
    allowed = await ensure_access([file_id], lit_token=context.lit_token)
    if file_id not in allowed:
        return ToolResultEnvelope(
            payload={"file_id": file_id, "error": "not_found"},
            token_estimate=0,
            truncated=False,
            warning="file not found",
        )

    basic, hydrated, relations_summary, chunk_count = await asyncio.gather(
        _fetch_basic_metadata(file_id),
        hydrate_files([file_id]),
        _fetch_relations_summary(file_id),
        asyncio.to_thread(_count_transcript_chunks, file_id),
    )

    # Defence in depth: even if the access filter said yes, refuse to
    # surface a row whose drive does not match the request's drive.
    # ``context.drive`` may be None in tests or for a global-scope
    # caller — in that case we accept whatever the host returned.
    if (
        basic is not None
        and context.drive
        and basic.get("drive")
        and basic["drive"] != context.drive
    ):
        return ToolResultEnvelope(
            payload={"file_id": file_id, "error": "not_found"},
            token_estimate=0,
            truncated=False,
            warning="file not found",
        )

    if basic is None:
        payload_404: dict[str, Any] = {
            "file_id": file_id,
            "error": "not_found",
        }
        return ToolResultEnvelope(
            payload=payload_404,
            token_estimate=estimate_payload_tokens(payload_404),
            truncated=False,
            warning="file not found",
        )

    hyd_raw = hydrated.get(file_id) if isinstance(hydrated, dict) else None
    hyd: dict[str, Any] = (
        {k: v for k, v in hyd_raw.items() if k in _HYDRATE_ALLOWED_FIELDS}
        if isinstance(hyd_raw, dict)
        else {}
    )

    payload: dict[str, Any] = {
        "file_id": file_id,
        "title": hyd.get("title") or basic.get("filename"),
        "drive": basic.get("drive"),
        "mime": hyd.get("mime_type") or basic.get("mime_type"),
        "file_type": basic.get("file_type"),
        "folder_path": basic.get("folder_path"),
        "tags": list(hyd.get("tags") or []),
        "has_transcript": chunk_count > 0,
        "has_active_summary": False,  # knowledge addon check, not in Phase 1.B
        "chunk_count": chunk_count,
        "relations": relations_summary,
    }

    context.register_file_ids([file_id])
    tokens = estimate_payload_tokens(payload)
    context.register_result_tokens(tokens)

    return ToolResultEnvelope(
        payload=payload,
        token_estimate=tokens,
        truncated=False,
    )


__all__ = ["get_file_detail"]
