"""``GET /files/{file_id}/document-text``: a document's text, unit by unit."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app import document_text
from app.database import get_search_db_read
from app.document_sections import load_section_titles
from app.drive_context import require_drive
from app.models import IndexedFile
from app.schemas import DocumentTextCursor, DocumentTextResponse, DocumentTextUnit

logger = logging.getLogger(__name__)

router = APIRouter(tags=["files"])

_MAX_CHARS = 200_000


def _lookup(file_id: str, drive: str) -> tuple[str, str]:
    with get_search_db_read() as db:
        row = (
            db.query(IndexedFile.mime_type, IndexedFile.file_path)
            .filter(
                IndexedFile.file_id == file_id,
                IndexedFile.drive == drive,
                IndexedFile.active.is_(True),
            )
            .first()
        )
    if row is None:
        raise HTTPException(status_code=404, detail="File not found")
    return row[0], row[1]


def _section_titles(file_id: str, numbers: list[int]) -> dict[int, str]:
    with get_search_db_read() as db:
        titles = load_section_titles(db, [(file_id, n) for n in numbers])
    return {page: title for (_fid, page), title in titles.items()}


@router.get("/files/{file_id}/document-text", response_model=DocumentTextResponse)
async def get_document_text(
    file_id: str,
    start: int = Query(default=1, ge=1),
    count: int | None = Query(default=None, ge=1),
    offset: int = Query(default=0, ge=0),
    max_chars: int = Query(default=50_000, ge=1, le=_MAX_CHARS),
    drive: str = Depends(require_drive),
) -> DocumentTextResponse:
    mime, file_path = await asyncio.to_thread(_lookup, file_id, drive)
    kind = document_text.KINDS.get(mime)
    if kind is None:
        raise HTTPException(
            status_code=415,
            detail="Readable types: " + ", ".join(sorted(document_text.KINDS)),
        )
    wanted = kind.default_count if count is None else count
    if wanted > kind.max_count:
        raise HTTPException(
            status_code=422,
            detail=f"count must be at most {kind.max_count} for a {kind.unit}",
        )
    end = start + wanted - 1
    try:
        extracted = await asyncio.to_thread(
            document_text.extract_units, mime, file_path, start, end,
        )
    except document_text.Busy as exc:
        raise HTTPException(status_code=429, detail="Document reader busy") from exc
    except Exception as exc:
        logger.warning(
            "document-text: %s could not be read (%s)", file_id, type(exc).__name__,
        )
        raise HTTPException(status_code=422, detail="Document could not be read") from exc

    page = document_text.page_units(extracted, start, end, offset, max_chars)
    titles = (
        await asyncio.to_thread(_section_titles, file_id, [u.number for u in page.units])
        if mime == document_text.EPUB and page.units
        else {}
    )
    return DocumentTextResponse(
        file_id=file_id,
        unit=kind.unit,
        unit_count=extracted.unit_count,
        units=[
            DocumentTextUnit(
                number=u.number,
                title=titles.get(u.number, u.title),
                text=u.text,
                truncated=u.truncated,
                extractor=u.extractor,
            )
            for u in page.units
        ],
        next=(
            None if page.next_start is None
            else DocumentTextCursor(start=page.next_start, offset=page.next_offset)
        ),
    )
