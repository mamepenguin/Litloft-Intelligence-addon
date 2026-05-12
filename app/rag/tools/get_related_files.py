"""``get_related_files`` tool wrapper.

Calls the core's ``GET /api/internal/file_relations`` and shapes the
response for the agentic loop. Returns both directions (incoming +
outgoing) flagged by ``direction`` so the LLM can tell A→B from B→A.

Per spec §2.2 the kind filter is optional; when omitted, every kind
is returned. ``file_relations`` itself enforces same-drive constraint
on creation, so we do not re-check it here.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from app.rag.tools._access import ensure_access, is_valid_file_id, is_valid_kind
from app.rag.tools.budget import estimate_payload_tokens
from app.rag.tools.context import ToolContext, ToolResultEnvelope


class _RelationsFetchError(Exception):
    """Operational error from the host's file_relations endpoint.

    Mirrors ``_TextFetchError`` semantics in ``get_file_chunks``: 5xx /
    network / auth failures should not collapse silently into "no
    relations" because the agentic loop would then re-try the same
    file or move on without realising the host is degraded.
    """

logger = logging.getLogger(__name__)


_INTERNAL_API_BASE_URL_DEFAULT = "http://backend:8000/api/internal"
_INTERNAL_API_TIMEOUT_SECONDS = 5.0


def _base_url() -> str:
    return os.environ.get(
        "HOMEVAULT_INTERNAL_API_URL", _INTERNAL_API_BASE_URL_DEFAULT
    )


async def get_related_files(
    *,
    context: ToolContext,
    file_id: str,
    kind: str | None = None,
) -> ToolResultEnvelope:
    context.register_tool_call("get_related_files")

    if not is_valid_file_id(file_id):
        return ToolResultEnvelope(
            payload={"file_id": file_id, "relations": [], "error": "invalid file_id"},
            token_estimate=0,
            truncated=False,
            warning="invalid file_id",
        )

    # Access gate on the **input** file_id.
    allowed_in = await ensure_access([file_id], lit_token=context.lit_token)
    if file_id not in allowed_in:
        return ToolResultEnvelope(
            payload={"file_id": file_id, "relations": [], "error": "not_found"},
            token_estimate=0,
            truncated=False,
            warning="file not found",
        )

    # Validate ``kind`` shape (host has no validator on the list path).
    if kind is not None:
        if not is_valid_kind(kind):
            return ToolResultEnvelope(
                payload={
                    "file_id": file_id,
                    "relations": [],
                    "error": "invalid kind",
                },
                token_estimate=0,
                truncated=False,
                warning="invalid kind",
            )

    params: dict[str, str] = {"file_id": file_id}
    if kind:
        params["kind"] = kind

    url = f"{_base_url().rstrip('/')}/file_relations"

    try:
        async with httpx.AsyncClient(
            timeout=_INTERNAL_API_TIMEOUT_SECONDS
        ) as client:
            response = await client.get(url, params=params)
            if response.status_code in (401, 403):
                raise _RelationsFetchError(
                    f"internal API auth failed ({response.status_code})"
                )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        logger.warning(
            "get_related_files: request failed for %s: %s",
            file_id,
            type(exc).__name__,
        )
        return ToolResultEnvelope(
            payload={
                "file_id": file_id,
                "relations": [],
                "error": "internal_api_failed",
            },
            token_estimate=0,
            truncated=False,
            warning="internal API call failed",
        )
    except _RelationsFetchError as exc:
        logger.warning("get_related_files: %s", exc)
        return ToolResultEnvelope(
            payload={
                "file_id": file_id,
                "relations": [],
                "error": "internal_api_failed",
            },
            token_estimate=0,
            truncated=False,
            warning=str(exc),
        )

    relations_raw: list[dict[str, Any]] = (
        data if isinstance(data, list) else []
    )
    candidate_ids: set[str] = set()
    raw_relations: list[dict[str, Any]] = []
    for r in relations_raw:
        if not isinstance(r, dict):
            continue
        a = r.get("file_id_a")
        b = r.get("file_id_b")
        k = r.get("kind")
        if not isinstance(a, str) or not isinstance(b, str) or not isinstance(k, str):
            continue
        if a == file_id:
            raw_relations.append(
                {"file_id": b, "kind": k, "direction": "outgoing"}
            )
            candidate_ids.add(b)
        elif b == file_id:
            raw_relations.append(
                {"file_id": a, "kind": k, "direction": "incoming"}
            )
            candidate_ids.add(a)

    # Access gate on the **output** IDs. ``file_relations`` is the only
    # endpoint that can legitimately point at IDs in another drive
    # (the same-drive constraint is enforced on create but not on read).
    # Without this filter the citation allow-list would leak cross-drive.
    allowed_out = await ensure_access(
        candidate_ids, lit_token=context.lit_token
    )
    relations = [r for r in raw_relations if r["file_id"] in allowed_out]

    payload = {"file_id": file_id, "relations": relations}
    context.register_file_ids(r["file_id"] for r in relations)
    tokens = estimate_payload_tokens(payload)
    context.register_result_tokens(tokens)

    return ToolResultEnvelope(
        payload=payload,
        token_estimate=tokens,
        truncated=False,
    )


__all__ = ["get_related_files"]
