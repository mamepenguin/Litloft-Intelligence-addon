"""``search_files`` tool wrapper.

Re-uses the existing hybrid retriever (``retrieve_candidates``) but
returns file-level rows instead of chunk-level segments. The agentic
loop will use this to discover candidates; deeper drill-down happens
through ``get_file_chunks`` / ``get_file_detail``.

Output shape (per spec §2.2):
    [{
      file_id,
      title,
      score,
      tags,
      has_transcript,
      has_active_summary,
      relations_summary  # {kind: count}, lazily empty in Phase 1.B
      folder_path,
      drive,
    }]
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.credentials import CallerCredential
from app.database import get_search_db_read
from app.models import TranscriptChunk
from app.rag.retriever import RetrievedFile, retrieve_with_keywords
from app.rag.tools.budget import estimate_payload_tokens
from app.rag.tools.context import ToolContext, ToolResultEnvelope

logger = logging.getLogger(__name__)


def _transcript_bearing_file_ids(file_ids: list[str]) -> set[str]:
    """Return the subset of ``file_ids`` that have TranscriptChunk rows.

    ``match_types`` alone is unreliable for the agentic loop: a file
    may have transcript chunks but be ranked solely on title / tags,
    leaving ``match_types`` without ``"transcript"`` and the LLM
    thinking the file is unreadable. One ``DISTINCT`` query against
    the intelligence DB beats N round-trips, and is cheaper than the
    follow-up ``get_file_chunks`` call we want to encourage.
    """
    if not file_ids:
        return set()
    try:
        with get_search_db_read() as db:
            rows = (
                db.query(TranscriptChunk.file_id)
                .filter(TranscriptChunk.file_id.in_(file_ids))
                .distinct()
                .all()
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "search_files: transcript existence check failed: %s",
            type(exc).__name__,
        )
        return set()
    return {row[0] for row in rows}


async def search_files(
    *,
    context: ToolContext,
    query: str,
    top_k: int = 10,
    credential: CallerCredential | None = None,
) -> ToolResultEnvelope:
    """Run hybrid retrieval and return up to ``top_k`` files.

    The drive is read from ``context.drive`` rather than letting the
    LLM choose, so a leaky LLM cannot ask for files outside the
    current request's drive.

    Access-control filtering happens inside ``retrieve_candidates``
    (it calls the host's filter-file-ids API). Files the caller
    cannot see never make it into the returned rows.
    """
    context.register_tool_call("search_files")

    if top_k <= 0:
        top_k = 1
    if top_k > 30:
        top_k = 30

    drive = context.drive
    token = credential if credential is not None else context.credential

    # Skip the LLM-driven query transform that ``retrieve_candidates``
    # would run upstream. The agentic loop's LLM has already chosen
    # this query; running another structured-transform call here just
    # doubles the LLM hops (and the json_object retry path is unreliable
    # on reasoning models like Qwen3 because thinking-mode output goes
    # to ``reasoning`` and leaves ``content`` empty). The raw query is
    # forwarded both as the keyword string and as the semantic query
    # so the hybrid retriever still ranks by embeddings.
    candidates: list[RetrievedFile] = await retrieve_with_keywords(
        keywords=query,
        top_k=top_k,
        credential=token,
        drive=drive,
        original_query=query,
    )

    # Single batched DB check so the LLM sees a reliable
    # ``has_transcript`` flag and learns to call get_file_chunks.
    transcript_ids = await asyncio.to_thread(
        _transcript_bearing_file_ids, [c.file_id for c in candidates]
    )

    rows: list[dict[str, Any]] = []
    for c in candidates:
        rows.append(
            {
                "file_id": c.file_id,
                "title": c.title or c.filename,
                "score": round(c.score, 4),
                "tags": [],  # populated by get_file_detail; kept lean here
                "has_transcript": c.file_id in transcript_ids,
                "has_active_summary": False,
                # ``relations_summary`` is computed by get_file_detail
                # against the core's file_relations endpoint; quoting
                # zero here keeps the LLM aware that it can ask for
                # detail without implying "no relations exist".
                "relations_summary": {},
                "folder_path": "",  # filled by get_file_detail
                "drive": c.drive,
                "mime_type": c.mime_type,
                "file_type": c.file_type,
            }
        )

    payload = {"files": rows}
    context.register_file_ids(r["file_id"] for r in rows)
    tokens = estimate_payload_tokens(payload)
    context.register_result_tokens(tokens)

    return ToolResultEnvelope(
        payload=payload,
        token_estimate=tokens,
        truncated=False,
    )


__all__ = ["search_files"]
