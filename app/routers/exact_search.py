"""``GET /search/exact`` and ``GET /search/scenes`` for agents.

Drive scope comes from ``X-Lit-Drive``; every query filters on
``indexed_files.drive`` itself, so the manifest declares no response filter.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from app import exact_search
from app.config import settings
from app.drive_context import require_drive
from app.policy_client import is_feature_enabled
from app.schemas import ExactSearchResponse, SceneSearchResponse
from app.workers.embedder import embed_query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])

_EXACT_Q_MAX = 200
_SCENES_Q_MAX = 1000


def _stripped_query(q: str, max_length: int) -> str:
    stripped = q.strip()
    if not 1 <= len(stripped) <= max_length:
        raise HTTPException(
            status_code=422, detail=f"q must be 1 to {max_length} characters after stripping"
        )
    return stripped


def _parse_sources(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return exact_search.SOURCES
    names = raw.split(",")
    unknown = [n for n in names if n not in exact_search.SOURCES]
    if not raw or unknown:
        raise HTTPException(
            status_code=422,
            detail=f"sources must be a comma-separated subset of {', '.join(exact_search.SOURCES)}",
        )
    return tuple(s for s in exact_search.SOURCES if s in names)


async def _visual_index_open(drive: str) -> bool:
    if settings.features.video_visual_index == "false":
        return False
    try:
        return await is_feature_enabled(drive, "video_visual_index")
    except Exception:
        return False


@router.get("/search/exact", response_model=ExactSearchResponse)
async def search_exact(
    q: str = Query(...),
    sources: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    drive: str = Depends(require_drive),
) -> ExactSearchResponse:
    query = _stripped_query(q, _EXACT_Q_MAX)
    selected = _parse_sources(sources)
    include_scenes = "scene" in selected and await _visual_index_open(drive)
    try:
        total, segments = await asyncio.to_thread(
            exact_search.find_exact,
            query, selected, drive,
            include_scenes=include_scenes, limit=limit, offset=offset,
        )
    except Exception as exc:
        logger.exception("Exact search failed")
        raise HTTPException(status_code=500, detail="Search failed") from exc
    return ExactSearchResponse(
        query=query, sources=list(selected), total=total, offset=offset, segments=segments,
    )


@router.get("/search/scenes", response_model=SceneSearchResponse)
async def search_scenes(
    q: str = Query(...),
    limit: int = Query(default=20, ge=1, le=50),
    drive: str = Depends(require_drive),
) -> SceneSearchResponse:
    if not await _visual_index_open(drive):
        raise HTTPException(status_code=404, detail="Feature disabled for drive")
    query = _stripped_query(q, _SCENES_Q_MAX)
    try:
        vector = await asyncio.to_thread(embed_query, query)
    except Exception as exc:
        logger.warning("Scene search embedding failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Embedding unavailable") from exc
    try:
        results = await asyncio.to_thread(
            exact_search.rank_scenes, np.asarray(vector, dtype=np.float32), drive, limit,
        )
    except Exception as exc:
        logger.exception("Scene search failed")
        raise HTTPException(status_code=500, detail="Search failed") from exc
    return SceneSearchResponse(query=query, results=results)
