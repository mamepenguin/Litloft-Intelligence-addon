"""Related-passages endpoint.

Serves the passage-level counterpart to ``/similar/{file_id}``: that one
answers which *files* relate to this one, this one answers which
*passages* relate to which. See ``app.passages``.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, Query, Request

from app.credentials import CallerCredential
from app.drive_context import assert_file_in_drive, require_drive
from app.passages import find_related_passages
from app.schemas import (
    PassageRef,
    RelatedPassageItem,
    RelatedPassagesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["passages"])


def _is_indexed_here(file_id: str, drive: str) -> bool:
    """Whether this addon holds an index row for the file, in this drive.

    A file the addon has never indexed is not an error: core discovers
    files before the addon reaches them, and the honest answer during
    that window is the same as for an indexed file with no matches —
    nothing. Raising instead would make the section show an error every
    time a viewer opened something newly added.

    A file that *is* indexed but belongs to another drive is different,
    and gets the drive boundary's 404.
    """
    from app.database import get_search_db_read
    from app.models import IndexedFile

    with get_search_db_read() as db:
        indexed = (
            db.query(IndexedFile)
            .filter(IndexedFile.file_id == file_id, IndexedFile.active.is_(True))
            .first()
        )
        if not indexed:
            return False
        assert_file_in_drive(indexed.drive, drive)
        return True


@router.get(
    "/files/{file_id}/related-passages",
    response_model=RelatedPassagesResponse,
)
async def related_passages_endpoint(
    file_id: str,
    request: Request,
    limit: int = Query(default=5, ge=1, le=20),
    drive: str = Depends(require_drive),
) -> RelatedPassagesResponse:
    """Passages of this file paired with passages of verified files.

    An empty ``results`` is an ordinary answer: the file may not be
    indexed yet, or nothing vouched for may resemble it. Only the drive
    boundary produces an error.
    """
    if not await asyncio.to_thread(_is_indexed_here, file_id, drive):
        return RelatedPassagesResponse(results=[])

    pairs = await find_related_passages(
        file_id=file_id,
        drive=drive,
        credential=CallerCredential.from_request(request),
        limit=limit,
    )

    return RelatedPassagesResponse(
        results=[
            RelatedPassageItem(
                source=PassageRef(
                    text=pair.text,
                    page=pair.page,
                    timestamp=pair.timestamp,
                ),
                match=PassageRef(
                    text=pair.other_text,
                    page=pair.other_page,
                    timestamp=pair.other_timestamp,
                ),
                file_id=pair.other_file_id,
                drive=pair.other_drive,
                filename=pair.other_filename,
                score=round(pair.score, 4),
                overlap=pair.overlap,
            )
            for pair in pairs
        ]
    )
