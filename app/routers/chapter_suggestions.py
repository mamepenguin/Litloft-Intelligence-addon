"""Suggest → approve/dismiss routes for LLM-derived media chapters."""

from __future__ import annotations

import json
import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text as sql_text

from app.config import settings
from app.database import get_search_db
from app.dependencies import get_chapter_suggestions_worker, get_llm_client
from app.drive_context import require_drive
from app.routers.files import _get_indexed_file_or_404
from app.schemas import ChapterSuggestionsResponse, MessageResponse
from app.workers.chapter_suggestions import (
    is_chapter_suggestions_enabled,
    normalise_chapter_candidates,
    promote_chapters_to_core,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chapter-suggestions"])


async def _require_allowed(file_id: str, drive: str) -> dict:
    if settings.features.chapter_suggestions == "false":
        raise HTTPException(status_code=404, detail="Feature not available")
    indexed = _get_indexed_file_or_404(file_id, drive)
    if indexed["file_type"] not in ("video", "audio"):
        raise HTTPException(status_code=404, detail="File not eligible")
    if not await is_chapter_suggestions_enabled(drive):
        raise HTTPException(status_code=404, detail="Feature not available")
    return indexed


@router.get(
    "/files/{file_id}/chapter-suggestions",
    response_model=ChapterSuggestionsResponse,
)
async def get_chapter_suggestions(
    file_id: str,
    drive: str = Depends(require_drive),
) -> ChapterSuggestionsResponse:
    if settings.features.chapter_suggestions == "false":
        return ChapterSuggestionsResponse(enabled=False, available=False)
    await _require_allowed(file_id, drive)
    with get_search_db() as session:
        row = session.execute(sql_text(
            "SELECT file_id, chapters_json, model, status, created_at "
            "FROM suggested_chapters WHERE file_id=:fid"
        ), {"fid": file_id}).fetchone()
    if row is None:
        return ChapterSuggestionsResponse(available=False)
    try:
        chapters = normalise_chapter_candidates(json.loads(row[1]))
    except (json.JSONDecodeError, TypeError):
        chapters = []
    return ChapterSuggestionsResponse(
        available=bool(chapters),
        file_id=row[0],
        chapters=chapters,
        model=row[2],
        status=row[3],
        created_at=row[4],
    )


@router.post(
    "/files/{file_id}/chapter-suggestions/generate",
    response_model=MessageResponse,
)
async def generate_chapter_suggestions(
    file_id: str,
    drive: str = Depends(require_drive),
) -> MessageResponse:
    if settings.features.chapter_suggestions == "false":
        raise HTTPException(status_code=400, detail="Chapter suggestions disabled")
    await _require_allowed(file_id, drive)
    if not get_llm_client().enabled:
        raise HTTPException(status_code=400, detail="LLM is not enabled")
    with get_search_db() as session:
        has_transcript = session.execute(sql_text(
            "SELECT 1 FROM transcript_chunks WHERE file_id=:fid LIMIT 1"
        ), {"fid": file_id}).first()
    if has_transcript is None:
        raise HTTPException(status_code=409, detail="Transcript is not available")
    queued = await get_chapter_suggestions_worker().enqueue(file_id, force=True)
    return MessageResponse(
        status="accepted" if queued else "already_queued",
        message="Generation queued" if queued else "Generation already queued",
    )


@router.post(
    "/files/{file_id}/chapter-suggestions/dismiss",
    response_model=MessageResponse,
)
async def dismiss_chapter_suggestions(
    file_id: str,
    drive: str = Depends(require_drive),
) -> MessageResponse:
    await _require_allowed(file_id, drive)
    with get_search_db() as session:
        row = session.execute(sql_text(
            "SELECT status FROM suggested_chapters WHERE file_id=:fid"
        ), {"fid": file_id}).fetchone()
        if row is None:
            raise HTTPException(
                status_code=404,
                detail="No chapter suggestions found",
            )
        if row[0] != "pending":
            raise HTTPException(
                status_code=409,
                detail="Only pending chapter suggestions can be dismissed",
            )
        session.execute(sql_text(
            "UPDATE suggested_chapters SET status='dismissed' "
            "WHERE file_id=:fid AND status='pending'"
        ), {"fid": file_id})
    return MessageResponse(status="ok", message="Chapter suggestions dismissed")


@router.post(
    "/files/{file_id}/chapter-suggestions/approve",
    response_model=MessageResponse,
)
async def approve_chapter_suggestions(
    file_id: str,
    drive: str = Depends(require_drive),
) -> MessageResponse:
    await _require_allowed(file_id, drive)
    with get_search_db() as session:
        row = session.execute(sql_text(
            "SELECT chapters_json, created_at, status FROM suggested_chapters "
            "WHERE file_id=:fid"
        ), {"fid": file_id}).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="No chapter suggestions found")
    if row[2] != "pending":
        raise HTTPException(
            status_code=409,
            detail="Only pending chapter suggestions can be approved",
        )
    try:
        chapters = normalise_chapter_candidates(json.loads(row[0]))
    except (json.JSONDecodeError, TypeError):
        chapters = []
    if not chapters:
        raise HTTPException(status_code=422, detail="Cannot approve empty chapters")
    try:
        await promote_chapters_to_core(file_id, chapters)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        detail = "Core rejected chapter promotion"
        if status == 422:
            raise HTTPException(status_code=422, detail=detail) from exc
        raise HTTPException(status_code=502, detail=detail) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Core unavailable") from exc

    # Approval state changes only after core confirms its durable write.
    with get_search_db() as session:
        result = session.execute(sql_text(
            "UPDATE suggested_chapters SET status='accepted' "
            "WHERE file_id=:fid AND created_at=:created_at "
            "AND status='pending'"
        ), {"fid": file_id, "created_at": row[1]})
    if result.rowcount != 1:
        raise HTTPException(
            status_code=409,
            detail="Chapter suggestions changed during approval",
        )
    return MessageResponse(status="ok", message="Chapter suggestions approved")
