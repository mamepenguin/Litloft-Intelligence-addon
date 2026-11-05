"""``GET /files/{file_id}/find``: every occurrence of a term inside one file.

Drive scope comes from ``X-Lit-Drive``; the file lookup and every source
query filter on that drive and on an active file, so the manifest declares
no response filter.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app import file_find
from app.drive_context import require_drive
from app.routers.exact_search import _EXACT_Q_MAX, _stripped_query, _visual_index_open
from app.schemas import FileFindResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


@router.get("/files/{file_id}/find", response_model=FileFindResponse)
async def find_file(
    file_id: str,
    q: str = Query(...),
    drive: str = Depends(require_drive),
) -> FileFindResponse:
    query = _stripped_query(q, _EXACT_Q_MAX)
    include_scenes = await _visual_index_open(drive)
    try:
        result = await asyncio.to_thread(
            file_find.find_in_file,
            file_id, drive, query, include_scenes=include_scenes,
        )
    except Exception as exc:
        logger.exception("File find failed")
        raise HTTPException(status_code=500, detail="Search failed") from exc
    return FileFindResponse(**result)
