"""RAG retriever: hybrid search wrapper with access control.

Wraps the existing ``app.search.search()`` pipeline in **recall mode**
(channel-rebalanced RRF + relaxed cutoffs) and enforces drive-level
access control by calling the host's Internal API
(``/api/internal/filter-file-ids``) with the caller's credential.
Files the caller cannot see are dropped before they ever
reach the LLM context builder — this is the primary access gate;
the manifest's ``response_filter`` is a secondary belt-and-suspenders.

A natural-language query is always rewritten into keyword form by
``app.rag.query_transform.transform_query`` *before* hitting the
search index. This eliminates the failure mode where question-style
noise words ("共通点は？", "教えて") poison the FTS5 AND-joined query
and drop the true-positive file entirely. The streaming router uses
``retrieve_with_keywords`` directly so it can emit the ``keywords``
SSE event between transform and search.

Public surface:

* ``transform_query`` (re-exported): natural-language → keyword string.
* ``retrieve_candidates(query, ...)``: transform + retrieve + access.
* ``retrieve_with_keywords(keywords, ...)``: retrieve + access only.
"""

import asyncio
import logging
import os
from dataclasses import dataclass

import httpx

from app.credentials import CallerCredential
from app.database import get_search_db
from app.models import IndexedFile
from app.rag.query_transform import (  # re-export
    RequiredTerm,
    StructuredQuery,
    iter_required_fallback_subsets,
    transform_query,
    transform_query_structured,
)
from app.search import SearchResult, SegmentGroup, search

# re-export so callers can ``from app.rag.retriever import transform_query``
__all__ = [
    "RetrievedFile",
    "RequiredTerm",
    "StructuredQuery",
    "retrieve_candidates",
    "retrieve_with_keywords",
    "transform_query",
    "transform_query_structured",
]

logger = logging.getLogger(__name__)

# Base URL for the host's Internal API. Override with
# HOMEVAULT_INTERNAL_API_URL in docker-compose if the host listens on
# a different address. The default matches the compose service name.
_INTERNAL_API_BASE_URL_DEFAULT = "http://backend:8000/api/internal"

# Conservative timeout for the filter call. The Internal API is
# in-cluster and should respond in well under a second; anything
# longer than this is almost certainly a misconfiguration.
_INTERNAL_API_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class RetrievedFile:
    """A file returned by the retriever, enriched with IndexedFile info.

    Wraps a ``SearchResult`` from ``app.search`` with title/description
    pulled from the ``IndexedFile`` row so the prompt builder doesn't
    need a second round trip to the DB.
    """

    file_id: str
    drive: str
    filename: str
    file_type: str
    title: str | None
    description: str | None
    score: float
    match_types: tuple[str, ...]
    segments: tuple[SegmentGroup, ...]
    mime_type: str | None = None


def _internal_api_base_url() -> str:
    """Resolve the Internal API base URL from env or fall back to default."""
    return os.environ.get(
        "HOMEVAULT_INTERNAL_API_URL", _INTERNAL_API_BASE_URL_DEFAULT
    )


async def _filter_file_ids_via_internal_api(
    file_ids: list[str],
    credential: CallerCredential | None,
) -> set[str]:
    """Call the host's Internal API to filter file_ids by access.

    The caller's browser Cookie or non-browser Authorization: Bearer
    credential is forwarded unchanged to the host. The host remains the
    single source of truth for drive access.

    When ``credential`` is None the caller is unauthenticated — the host
    returns the union of fully-public drives, which is the intended
    behaviour for "全公開モード" (all drives public). We still call the
    API so unauthenticated users cannot see protected drives.

    Args:
        file_ids: The candidate file_ids from the search pipeline.
        credential: Optional caller credential to forward.

    Returns:
        The subset of ``file_ids`` the caller is allowed to see. On
        any error (network, non-200 response, unexpected shape) the
        function fails closed and returns an empty set.
    """
    if not file_ids:
        return set()

    url = f"{_internal_api_base_url()}/filter-file-ids"
    headers = credential.headers() if credential is not None else {}

    try:
        async with httpx.AsyncClient(
            timeout=_INTERNAL_API_TIMEOUT_SECONDS,
            headers=headers,
        ) as client:
            response = await client.post(
                url,
                json={"file_ids": file_ids},
            )
    except httpx.HTTPError as e:
        # Network / timeout / connection error. Fail closed — returning
        # an empty set drops every candidate so no data leaks past the
        # access check on transient failure.
        logger.error("Internal API filter call failed: %s", type(e).__name__)
        return set()

    if response.status_code != 200:
        logger.error(
            "Internal API filter returned status %d", response.status_code
        )
        return set()

    try:
        data = response.json()
    except ValueError:
        logger.error("Internal API filter returned non-JSON body")
        return set()

    # The host's internal router returns {"accessible": [...]}.
    # We accept the legacy "allowed" key too in case older hosts are
    # deployed, but log a deprecation warning so the mismatch is obvious.
    accessible = None
    if isinstance(data, dict):
        if isinstance(data.get("accessible"), list):
            accessible = data["accessible"]
        elif isinstance(data.get("allowed"), list):
            logger.warning(
                "Internal API returned legacy 'allowed' key; "
                "expected 'accessible'. Update host backend."
            )
            accessible = data["allowed"]

    if accessible is None:
        logger.error(
            "Internal API filter response missing 'accessible' list"
        )
        return set()

    return {fid for fid in accessible if isinstance(fid, str)}


def _get_indexed_files_meta(file_ids: list[str]) -> dict[str, dict]:
    """Fetch title/description for a list of file_ids.

    Returns a mapping ``file_id -> {"title": ..., "description": ...}``.
    Files that have no ``IndexedFile`` row are simply absent from the
    result — callers must handle missing keys gracefully.
    """
    if not file_ids:
        return {}

    with get_search_db() as session:
        rows = (
            session.query(IndexedFile)
            .filter(IndexedFile.file_id.in_(file_ids))
            .all()
        )
        return {
            row.file_id: {
                "file_id": row.file_id,
                "title": row.title or None,
                "description": row.description or None,
                "mime_type": row.mime_type or None,
            }
            for row in rows
        }


def _to_retrieved_file(
    result: SearchResult,
    meta: dict,
) -> RetrievedFile:
    """Merge a SearchResult and its IndexedFile metadata into a RetrievedFile."""
    title = meta.get("title") if meta else None
    description = meta.get("description") if meta else None
    mime_type = meta.get("mime_type") if meta else None
    return RetrievedFile(
        file_id=result.file_id,
        drive=result.drive,
        filename=result.filename,
        file_type=result.file_type,
        mime_type=mime_type,
        title=title,
        description=description,
        score=result.score,
        match_types=tuple(result.match_types),
        segments=tuple(result.segments),
    )


async def retrieve_with_keywords(
    keywords: str,
    top_k: int,
    credential: CallerCredential | None,
    file_type: str | None = None,
    drive: str | None = None,
    *,
    original_query: str | None = None,
    file_id_scope: list[str] | None = None,
    required: tuple[RequiredTerm, ...] | None = None,
    include_scene_clip: bool = False,
) -> list[RetrievedFile]:
    """Retrieve top-k RAG candidates for a **pre-transformed** keyword query.

    This is the entry point used by the streaming router, which needs
    to emit the transformed keywords to the client as an SSE event
    before running the retrieval. Callers with a natural-language
    question should use ``retrieve_candidates`` instead, which handles
    the transform step itself.

    Pipeline:

    1. Run hybrid search in **recall mode** via ``app.search.search()``,
       with the structured-transform ``required`` tuple applied as a
       hard filter when supplied.
    2. If the hard filter yielded zero results (Tier 1 fallback),
       re-run the search with ``required=None`` so the user gets at
       least the loose-recall ranking. The spec §3.5 calls this
       "Tier 3 demote required to semantic" — Tier 2 (drop one term
       at a time) is deferred to Phase 4.
    3. Filter file_ids via the Internal API (forwarding credential).
    4. Enrich the surviving results with IndexedFile title/description.

    Args:
        keywords: A search-friendly keyword string (already transformed
            from the natural-language question). Passed straight into
            the hybrid search index.
        top_k: Max number of files to pull from the search pipeline.
        credential: Optional caller credential to forward for access
            control (None = unauthenticated caller).
        file_type: Optional file type filter (video / audio / ...).
        drive: Optional drive name filter.
        required: Optional tuple of ``RequiredTerm`` from the structured
            transform. Drives the FTS hard filter when present; falls
            back to ``required=None`` re-search on zero results.

    Returns:
        A list of ``RetrievedFile`` preserving the original search order.
    """
    # app.search.search is a sync function — offload to a thread so we
    # don't block the event loop while it does DB / vector work.
    response = await asyncio.to_thread(
        search,
        keywords,
        limit=top_k,
        file_type=file_type,
        drive=drive,
        mode="recall",
        semantic_query=original_query,
        file_id_scope=file_id_scope,
        required=required,
        include_scene_clip=include_scene_clip,
    )

    results = list(response.results)

    # Phase 4: Tier 2 → Tier 3 fallback ladder. When the full required
    # tuple yields zero hits we step through subsets dropping one
    # term at a time (most-aliased first; ties broken by position so
    # the user's leading term is preserved). The ladder terminates
    # at the empty tuple which is equivalent to Tier 3 ("demote all
    # required to semantic"). Surfacing the fallback step to the
    # client (SSE event) is left to the streaming layer.
    if not results and required:
        for subset in iter_required_fallback_subsets(required):
            tier_label = (
                "Tier 3 (no required filter)"
                if not subset
                else f"Tier 2 with {len(subset)} required term(s)"
            )
            logger.info(
                "Required-keyword hard filter empty; retrying with %s",
                tier_label,
            )
            response = await asyncio.to_thread(
                search,
                keywords,
                limit=top_k,
                file_type=file_type,
                drive=drive,
                mode="recall",
                semantic_query=original_query,
                file_id_scope=file_id_scope,
                required=subset or None,
                include_scene_clip=include_scene_clip,
            )
            results = list(response.results)
            if results:
                break

    if not results:
        return []

    file_ids = [r.file_id for r in results]

    # Access filter (primary gate). Always call, even when token is
    # None — the host decides what a token-less caller can see.
    allowed_ids = await _filter_file_ids_via_internal_api(
        file_ids=file_ids,
        credential=credential,
    )

    allowed_results = [r for r in results if r.file_id in allowed_ids]
    if not allowed_results:
        return []

    meta_by_id = _get_indexed_files_meta(
        [r.file_id for r in allowed_results]
    )

    return [
        _to_retrieved_file(r, meta_by_id.get(r.file_id, {}))
        for r in allowed_results
    ]


async def retrieve_candidates(
    query: str,
    top_k: int,
    credential: CallerCredential | None,
    file_type: str | None = None,
    drive: str | None = None,
    *,
    transform_temperature: float | None = None,
    file_id_scope: list[str] | None = None,
    include_scene_clip: bool = False,
) -> list[RetrievedFile]:
    """Transform a natural-language question and retrieve RAG candidates.

    Convenience wrapper that runs the LLM structured query transform
    first, then hands off to ``retrieve_with_keywords``. Used by the
    non-streaming code paths (service layer, tests) where the keyword
    string does not need to be surfaced to the caller separately.

    The structured transform extracts ``required`` proper-noun terms
    that are forwarded as the FTS hard filter, so retrieval first
    shrinks to the must-include set and then ranks within it. On any
    transform failure ``StructuredQuery.passthrough`` is returned and
    the call degrades to the legacy loose retrieval (no hard filter,
    raw query as keywords).

    Args:
        query: The user's natural-language question.
        top_k: Max number of files to pull from the search pipeline.
        credential: Optional caller credential to forward for access
            control.
        file_type: Optional file type filter (video / audio / ...).
        drive: Optional drive name filter.

    Returns:
        A list of ``RetrievedFile`` preserving the original search order.
    """
    structured = await transform_query_structured(
        query, temperature=transform_temperature
    )
    keywords = structured.raw_keywords or query
    return await retrieve_with_keywords(
        keywords=keywords,
        top_k=top_k,
        credential=credential,
        original_query=query,
        file_type=file_type,
        drive=drive,
        file_id_scope=file_id_scope,
        required=structured.required or None,
        include_scene_clip=include_scene_clip,
    )
