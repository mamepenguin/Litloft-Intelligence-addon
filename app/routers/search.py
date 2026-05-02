"""Search endpoints: semantic search, compare, and debug."""

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from app.drive_context import require_drive
from app.file_hydrate import hydrate_files
from app.schemas import (
    CompareResponseModel,
    SearchResponseModel,
    SearchResultItem,
    SearchResultSegment,
    SearchResultSegmentMatch,
    SourceCountsModel,
)
from app.search import execute_search_compare, search as execute_search

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


async def _to_response_model(result: Any) -> SearchResponseModel:
    """Convert an internal SearchResponse to the API response model.

    Calls ``POST /api/internal/files/bulk`` once per response to enrich
    each result with FileItem-shaped metadata (favorite, tags, etc.) so
    the frontend can render the same ``FileCard`` as filename-match
    results. Hydration failures degrade silently — the IndexedFile
    snapshot fields on each result remain valid for basic display.
    """
    file_ids = [r.file_id for r in result.results]
    hydrate_map = await hydrate_files(file_ids)

    return SearchResponseModel(
        results=[
            SearchResultItem(
                file_id=r.file_id,
                drive=r.drive,
                filename=r.filename,
                file_type=r.file_type,
                score=round(r.score, 4),
                match_types=list(r.match_types),
                segments=[
                    SearchResultSegment(
                        time_range=(
                            list(s.time_range) if s.time_range else None
                        ),
                        matches=[
                            SearchResultSegmentMatch(
                                type=m.match_type,
                                text=m.text,
                                score=round(m.score, 4),
                                page=m.page,
                            )
                            for m in s.matches
                        ],
                    )
                    for s in r.segments
                ],
                file=hydrate_map.get(r.file_id),
            )
            for r in result.results
        ],
        total=result.total,
        indexed_files=result.indexed_files,
        service_version=result.service_version,
    )


@router.get("/debug/search")
async def debug_search_endpoint(
    q: str = Query(..., min_length=1, description="Search query"),
) -> dict:
    """Debug search: returns raw scores from each search system."""
    from app.debug import debug_search
    result = debug_search(q)
    return result.model_dump()


@router.get("/search", response_model=SearchResponseModel)
async def search_endpoint(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=20, ge=1, le=100, description="Max results"),
    type: str | None = Query(default=None, description="File type filter"),
    mode: Literal["precision", "recall"] = Query(
        default="precision",
        description=(
            "Ranking mode. 'precision' (default) is for the human search UI; "
            "'recall' is for admin comparison with the RAG / Ask pipeline."
        ),
    ),
    include_scene_clip: bool = Query(
        default=False,
        description=(
            "When true, scene-frame CLIP embeddings (``embedding_type=\"clip\"``) "
            "are included alongside the default representative-frame "
            "(``embedding_type=\"clip_thumbnail\"``). The frontend's "
            "'シーン検索' / 'Scene search' toggle drives this flag. "
            "Spec 2026-05-02-thumbnail-clip-default-shallow-search.md."
        ),
    ),
    drive: str = Depends(require_drive),
) -> SearchResponseModel:
    """Execute a semantic search query within the request's drive.

    Drive scope is set by the ``X-Lit-Drive`` header (forwarded by the
    Litloft Generic Addon Proxy from ``/drive/{drive}/...``). Results
    are constrained to that drive — cross-drive search is not exposed
    at all to keep drive-as-privacy-boundary intact.
    """
    try:
        result = execute_search(
            query=q, limit=limit, file_type=type, drive=drive, mode=mode,
            include_scene_clip=include_scene_clip,
        )
    except Exception as e:
        logger.error("Search failed: %s", e)
        raise HTTPException(status_code=500, detail="Search failed") from e

    return await _to_response_model(result)


@router.get("/search/compare", response_model=CompareResponseModel)
async def search_compare_endpoint(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=20, ge=1, le=100, description="Max results"),
    type: str | None = Query(default=None, description="File type filter"),
    drive: str = Depends(require_drive),
) -> CompareResponseModel:
    """Compare RRF vs cosine-similarity scoring side by side."""
    try:
        compare = execute_search_compare(
            query=q, limit=limit, file_type=type, drive=drive,
        )
    except Exception as e:
        logger.error("Compare search failed: %s", e)
        raise HTTPException(status_code=500, detail="Search failed") from e

    return CompareResponseModel(
        rrf=await _to_response_model(compare.rrf),
        cosine=await _to_response_model(compare.cosine),
        rrf_no_cutoff=await _to_response_model(compare.rrf_no_cutoff),
        cosine_no_cutoff=await _to_response_model(compare.cosine_no_cutoff),
        source_counts=SourceCountsModel(
            text_vector=compare.source_counts.text_vector,
            clip_vector=compare.source_counts.clip_vector,
            keyword=compare.source_counts.keyword,
            transcript_keyword=compare.source_counts.transcript_keyword,
            text_content_keyword=compare.source_counts.text_content_keyword,
        ),
    )
