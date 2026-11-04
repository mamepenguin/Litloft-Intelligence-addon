"""Search endpoints: semantic search, compare, and debug."""

import asyncio
import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from app.document_sections import EPUB_MIME
from app.drive_context import require_drive
from app.file_hydrate import hydrate_files
from app.file_matching import (
    FilterUnavailable,
    InvalidFilter,
    fetch_matching_file_ids,
    parse_search_filter,
)
from app import scene_segments
from app.schemas import (
    CompareResponseModel,
    SceneSegmentsResponse,
    SearchPassage as SearchPassageModel,
    SearchResponseModel,
    SearchResultItem,
    SearchResultSegment,
    SearchResultSegmentMatch,
    SourceCountsModel,
)
from app.workers.clip import embed_text_clip
from app.search import (
    empty_search_response,
    execute_search_compare,
    search as execute_search,
    search_images as execute_search_images,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


async def _to_response_model(
    result: Any,
    *,
    hydrate_map: dict[str, Any] | None = None,
) -> SearchResponseModel:
    """Convert an internal SearchResponse to the API response model.

    Calls ``POST /api/internal/files/bulk`` once per response to enrich
    each result with FileItem-shaped metadata (favorite, tags, etc.) so
    the frontend can render the same ``FileCard`` as filename-match
    results. Hydration failures degrade silently — the IndexedFile
    snapshot fields on each result remain valid for basic display.

    ``hydrate_map`` lets a caller that already hydrated its own batches
    reuse the answer instead of asking again.
    """
    if hydrate_map is None:
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
                            _segment_match(m, r.mime_type, dict(r.section_titles))
                            for m in s.matches
                        ],
                    )
                    for s in r.segments
                ],
                file=hydrate_map.get(r.file_id),
                passages=(
                    [
                        SearchPassageModel(
                            kind=p.kind,
                            start=p.start,
                            end=p.end,
                            page=p.page,
                            section=p.section,
                            section_title=p.section_title,
                            chunk_index=p.chunk_index,
                            text=p.text,
                            score=p.score,
                            keyword=p.keyword,
                            highlight_terms=list(p.highlight_terms),
                            frame_url=p.frame_url,
                        )
                        for p in r.passages
                    ]
                    if r.passages is not None
                    else None
                ),
            )
            for r in result.results
        ],
        total=result.total,
        indexed_files=result.indexed_files,
        service_version=result.service_version,
    )


_HYDRATE_BATCH_SIZE = 500


async def _hydrate_batches(file_ids: list[str]) -> dict[str, Any]:
    """Hydrate ids in batches, failing the whole answer when a batch does.

    ``hydrate_files`` reports a transport or protocol failure as an empty
    mapping. A continuation row without hydration is dropped by the trust
    filter, so this route must not answer with one.
    """
    hydrated: dict[str, Any] = {}
    for start in range(0, len(file_ids), _HYDRATE_BATCH_SIZE):
        batch = file_ids[start : start + _HYDRATE_BATCH_SIZE]
        part = await hydrate_files(batch)
        if not part:
            raise HTTPException(status_code=503, detail="Hydration unavailable")
        hydrated.update(part)
    return hydrated


async def _filter_scope(drive: str, raw: str | None) -> list[str] | None:
    """The file ids a ``filter`` names, or ``None`` when the request has none."""
    if raw is None:
        return None
    try:
        search_filter = parse_search_filter(raw)
    except InvalidFilter as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        return await fetch_matching_file_ids(drive, search_filter)
    except FilterUnavailable as exc:
        raise HTTPException(status_code=503, detail="Filter unavailable") from exc


def _segment_match(
    m: Any, mime_type: str | None, section_titles: dict[int, str],
) -> SearchResultSegmentMatch:
    if mime_type != EPUB_MIME:
        return SearchResultSegmentMatch(
            type=m.match_type, text=m.text, score=round(m.score, 4), page=m.page,
        )
    # An EPUB chunk keeps its section number in ``page``; on the wire it is
    # a section, never a page.
    return SearchResultSegmentMatch(
        type=m.match_type,
        text=m.text,
        score=round(m.score, 4),
        page=None,
        section=m.page,
        section_title=section_titles.get(m.page) if m.page is not None else None,
    )


@router.get("/search", response_model=SearchResponseModel)
async def search_endpoint(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=20, ge=1, le=1000, description="Max results"),
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
    filter: str | None = Query(
        default=None,
        description=(
            "JSON object of tag / kind / favourite / liked conditions, "
            "resolved to a file-id scope through the core's Internal API."
        ),
    ),
    passages: bool = Query(
        default=False,
        description=(
            "When true, each result also lists the places inside the file that "
            "its matches point at (speech, scene, page, section, picture text)."
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
    if filter is not None:
        if include_scene_clip:
            raise HTTPException(
                status_code=422, detail="filter cannot include scene clips"
            )
        if type is not None:
            raise HTTPException(status_code=422, detail="filter carries the kinds")
        if mode != "precision":
            raise HTTPException(
                status_code=422, detail="filter is not available in recall mode"
            )
    scope = await _filter_scope(drive, filter)
    if scope is not None and not scope:
        return await _to_response_model(
            await run_in_threadpool(empty_search_response)
        )
    try:
        result = await run_in_threadpool(
            execute_search,
            query=q, limit=limit, file_type=type, drive=drive, mode=mode,
            include_scene_clip=include_scene_clip, file_id_constraint=scope,
            passages=passages,
        )
    except Exception as e:
        logger.error("Search failed: %s", e)
        raise HTTPException(status_code=500, detail="Search failed") from e

    return await _to_response_model(result)


@router.get("/search/images", response_model=SearchResponseModel)
async def search_images_endpoint(
    q: str = Query(..., min_length=1, description="Search query"),
    type: str | None = Query(default=None, description="File type filter"),
    filter: str | None = Query(
        default=None,
        description=(
            "JSON object of tag / kind / favourite / liked conditions, "
            "resolved to a file-id scope through the core's Internal API."
        ),
    ),
    drive: str = Depends(require_drive),
) -> SearchResponseModel:
    """Rank the request drive's representative frames by similarity to ``q``."""
    if filter is not None and type is not None:
        raise HTTPException(status_code=422, detail="filter carries the kinds")
    scope = await _filter_scope(drive, filter)
    if scope is not None and not scope:
        return await _to_response_model(
            await run_in_threadpool(empty_search_response)
        )
    try:
        vector = await asyncio.to_thread(embed_text_clip, q)
    except Exception as e:
        logger.warning("Image search embedding failed (%s)", e.__class__.__name__)
        raise HTTPException(status_code=503, detail="Embedding unavailable") from e
    try:
        result = await run_in_threadpool(
            execute_search_images, vector, file_type=type, drive=drive,
            file_id_constraint=scope,
        )
    except Exception as e:
        logger.error("Search images failed: %s", e)
        raise HTTPException(status_code=500, detail="Search failed") from e

    hydrate_map = await _hydrate_batches([r.file_id for r in result.results])
    return await _to_response_model(result, hydrate_map=hydrate_map)


@router.get("/search/compare", response_model=CompareResponseModel)
async def search_compare_endpoint(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=20, ge=1, le=100, description="Max results"),
    type: str | None = Query(default=None, description="File type filter"),
    drive: str = Depends(require_drive),
) -> CompareResponseModel:
    """Compare RRF vs cosine-similarity scoring side by side."""
    try:
        compare = await run_in_threadpool(
            execute_search_compare,
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


@router.get("/search/scene-segments", response_model=SceneSegmentsResponse)
async def scene_segments_endpoint(
    q: str = Query(...),
    limit: int = Query(default=20, ge=1, le=50),
    drive: str = Depends(require_drive),
) -> SceneSegmentsResponse:
    query = q.strip()
    if not 1 <= len(query) <= 200:
        raise HTTPException(
            status_code=422, detail="q must be 1 to 200 characters after stripping"
        )
    try:
        frames = await asyncio.to_thread(scene_segments.count_frames, drive)
    except Exception as e:
        logger.error("Scene segment search failed: %s", e)
        raise HTTPException(status_code=500, detail="Search failed") from e
    if frames == 0:
        return SceneSegmentsResponse(query=query, segments=[], partial=False)
    try:
        vector = await asyncio.to_thread(embed_text_clip, query)
    except Exception as e:
        logger.warning("Scene segment embedding failed (%s)", type(e).__name__)
        raise HTTPException(status_code=503, detail="Embedding unavailable") from e
    try:
        return await asyncio.to_thread(
            scene_segments.find_segments, vector, query, drive, limit, frames,
        )
    except Exception as e:
        logger.error("Scene segment search failed: %s", e)
        raise HTTPException(status_code=500, detail="Search failed") from e
