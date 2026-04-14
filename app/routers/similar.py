"""Similar files endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.drive_context import assert_file_in_drive, require_drive
from app.schemas import KeywordScore, SimilarFileItem, SimilarFilesResponse
from app.search import find_similar

logger = logging.getLogger(__name__)

router = APIRouter(tags=["similar"])


@router.get("/similar/{file_id}", response_model=SimilarFilesResponse)
async def similar_files_endpoint(
    file_id: str,
    limit: int = Query(default=6, ge=1, le=20),
    drive: str = Depends(require_drive),
) -> SimilarFilesResponse:
    """Find files similar to ``file_id`` within the request's drive."""
    try:
        search_result = find_similar(file_id=file_id, limit=limit, drive=drive)
    except Exception as e:
        logger.error("Similar files search failed: %s", e)
        raise HTTPException(status_code=500, detail="Similar search failed") from e

    return SimilarFilesResponse(
        results=[
            SimilarFileItem(
                file_id=r.file_id,
                drive=r.drive,
                filename=r.filename,
                file_type=r.file_type,
                mime_type=r.mime_type,
                score=round(r.score, 4),
                match_type=r.match_type,
                primary_score=round(r.primary_score, 4) if r.primary_score is not None else None,
                secondary_score=round(r.secondary_score, 4) if r.secondary_score is not None else None,
                shared_keywords=[
                    KeywordScore(**kw) for kw in r.shared_keywords
                ],
            )
            for r in search_result.results
        ],
        source_keywords=[
            KeywordScore(**kw) for kw in search_result.source_keywords
        ],
    )


@router.get("/debug/similar/{file_id}")
async def debug_similar_endpoint(
    file_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    drive: str = Depends(require_drive),
) -> dict:
    """Debug similar files: returns raw scores from each embedding type."""
    from app.search import (
        _find_similar_by_embedding,
        _select_embedding_types,
    )
    from app.database import get_search_db
    from app.models import IndexedFile

    with get_search_db() as session:
        source = (
            session.query(IndexedFile)
            .filter(
                IndexedFile.file_id == file_id,
                IndexedFile.active.is_(True),
            )
            .first()
        )
        if not source:
            raise HTTPException(status_code=404, detail="File not indexed")
        assert_file_in_drive(source.drive, drive)

        source_info = {
            "file_id": source.file_id,
            "filename": source.filename,
            "file_type": source.file_type,
            "drive": source.drive,
        }

    primary_type, fallback_type = _select_embedding_types(source_info["file_type"])

    all_types = ["clip", "metadata", "whisper", "text_content"]
    results_by_type: dict[str, list[dict]] = {}

    for etype in all_types:
        raw = _find_similar_by_embedding(file_id, etype, limit, drive)
        results_by_type[etype] = [
            {
                "file_id": r["file_id"],
                "filename": r["filename"],
                "file_type": r["file_type"],
                "score": round(r["score"], 6),
            }
            for r in raw
        ]

    return {
        "source": source_info,
        "primary_type": primary_type,
        "fallback_type": fallback_type,
        "results_by_type": results_by_type,
    }
