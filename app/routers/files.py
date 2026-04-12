"""File inspection and suggested tags endpoints."""

import logging
import subprocess
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import text as sql_text

from app.config import settings
from app.dependencies import get_auto_tags_worker
from app.schemas import (
    BatchSuggestedTagsRequest,
    BatchSuggestedTagsResponse,
    ClipTimestampItem,
    ClipTimestampsResponse,
    IndexDetailEmbeddingItem,
    IndexDetailType,
    IndexDetailsResponse,
    MessageResponse,
    SuggestedTagsResponse,
    TranscriptChunkResponse,
    TranscriptResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["files"])

_EMBEDDING_TYPES = ("metadata", "clip", "whisper", "text_content", "blip_caption")
_ITEMS_PER_TYPE = 50


def _get_indexed_file_or_404(file_id: str) -> dict[str, Any]:
    """Get an indexed file by ID or raise 404."""
    from app.database import get_search_db
    from app.models import IndexedFile

    with get_search_db() as db:
        indexed = db.query(IndexedFile).filter(
            IndexedFile.file_id == file_id,
            IndexedFile.active.is_(True),
        ).first()
        if not indexed:
            raise HTTPException(status_code=404, detail="File not indexed")
        # Detach from session by copying attributes
        return {
            "file_id": indexed.file_id,
            "drive": indexed.drive,
            "filename": indexed.filename,
            "file_path": indexed.file_path,
            "file_type": indexed.file_type,
            "metadata_indexed": indexed.metadata_indexed,
            "clip_indexed": indexed.clip_indexed,
            "whisper_indexed": indexed.whisper_indexed,
            "text_indexed": indexed.text_indexed,
            "indexed_at": indexed.indexed_at.isoformat() if indexed.indexed_at else "",
        }


@router.get("/files/{file_id}/transcript", response_model=TranscriptResponse)
async def get_transcript(file_id: str) -> TranscriptResponse:
    """Get Whisper transcript chunks for a file."""
    from app.database import get_search_db
    from app.models import TranscriptChunk

    indexed = _get_indexed_file_or_404(file_id)

    with get_search_db() as db:
        chunks = (
            db.query(TranscriptChunk)
            .filter(TranscriptChunk.file_id == file_id)
            .order_by(TranscriptChunk.chunk_index)
            .all()
        )

    if not chunks:
        raise HTTPException(status_code=404, detail="No transcript available")

    language = chunks[0].language if chunks else ""

    return TranscriptResponse(
        file_id=file_id,
        drive=indexed["drive"],
        language=language,
        chunks=[
            TranscriptChunkResponse(
                index=c.chunk_index,
                text=c.text,
                start=c.timestamp_start,
                end=c.timestamp_end,
            )
            for c in chunks
        ],
    )


@router.get("/files/{file_id}/index-details", response_model=IndexDetailsResponse)
async def get_index_details(file_id: str) -> IndexDetailsResponse:
    """Get detailed indexing status and embedding info for a file."""
    from sqlalchemy import func
    from app.database import get_search_db
    from app.models import Embedding

    indexed = _get_indexed_file_or_404(file_id)

    embeddings_by_type: dict[str, IndexDetailType] = {}

    with get_search_db() as db:
        for etype in _EMBEDDING_TYPES:
            count = (
                db.query(func.count(Embedding.id))
                .filter(
                    Embedding.file_id == file_id,
                    Embedding.embedding_type == etype,
                )
                .scalar()
            ) or 0

            items_query = (
                db.query(Embedding)
                .filter(
                    Embedding.file_id == file_id,
                    Embedding.embedding_type == etype,
                )
                .order_by(Embedding.timestamp_start.asc().nullsfirst())
                .limit(_ITEMS_PER_TYPE)
                .all()
            )

            embeddings_by_type[etype] = IndexDetailType(
                count=count,
                items=[
                    IndexDetailEmbeddingItem(
                        content_preview=e.content_preview,
                        start=e.timestamp_start,
                        end=e.timestamp_end,
                    )
                    for e in items_query
                ],
            )

    return IndexDetailsResponse(
        file_id=file_id,
        drive=indexed["drive"],
        filename=indexed["filename"],
        status={
            "metadata": indexed["metadata_indexed"],
            "clip": indexed["clip_indexed"],
            "whisper": indexed["whisper_indexed"],
            "text": indexed["text_indexed"],
        },
        indexed_at=indexed["indexed_at"],
        embeddings=embeddings_by_type,
    )


@router.get("/files/{file_id}/clip-timestamps", response_model=ClipTimestampsResponse)
async def get_clip_timestamps(file_id: str) -> ClipTimestampsResponse:
    """Get CLIP frame extraction timestamps for a file."""
    from app.database import get_search_db
    from app.models import Embedding

    indexed = _get_indexed_file_or_404(file_id)

    with get_search_db() as db:
        clips = (
            db.query(Embedding)
            .filter(
                Embedding.file_id == file_id,
                Embedding.embedding_type == "clip",
            )
            .order_by(Embedding.timestamp_start.asc())
            .all()
        )

    return ClipTimestampsResponse(
        file_id=file_id,
        drive=indexed["drive"],
        timestamps=[
            ClipTimestampItem(
                start=c.timestamp_start or 0.0,
                content_preview=c.content_preview,
            )
            for c in clips
        ],
    )


@router.get("/files/{file_id}/frame")
async def get_frame(
    file_id: str,
    t: float = Query(..., ge=0, description="Timestamp in seconds"),
) -> Response:
    """Extract a single video frame at the given timestamp using ffmpeg."""
    from app.config import resolve_file_path, validate_file_path

    indexed = _get_indexed_file_or_404(file_id)

    if indexed["file_type"] != "video":
        raise HTTPException(status_code=400, detail="Not a video file")

    abs_path = resolve_file_path(indexed["drive"], indexed["file_path"])
    if not abs_path or not validate_file_path(abs_path):
        raise HTTPException(status_code=404, detail="Video file not accessible")

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-ss", str(t),
                "-i", abs_path,
                "-frames:v", "1",
                "-vf", "scale=480:-1",
                "-f", "image2pipe",
                "-vcodec", "mjpeg",
                "-q:v", "3",
                "-",
            ],
            capture_output=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Frame extraction timed out")

    if result.returncode != 0 or not result.stdout:
        logger.warning("Frame extraction failed for %s at %.1fs", file_id, t)
        raise HTTPException(status_code=500, detail="Frame extraction failed")

    return Response(
        content=result.stdout,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


# --- Suggested tags endpoints ---


@router.get("/files/{file_id}/suggested-tags", response_model=SuggestedTagsResponse)
async def get_suggested_tags(file_id: str) -> SuggestedTagsResponse:
    """Get suggested tags for a file."""
    import json as json_mod
    from app.database import get_search_db

    if settings.features.auto_tags == "false":
        return SuggestedTagsResponse(available=False)

    with get_search_db() as session:
        row = session.execute(
            sql_text(
                "SELECT file_id, tags, model, status, created_at "
                "FROM suggested_tags WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchone()

    if row is None:
        return SuggestedTagsResponse(available=False)

    try:
        tags = json_mod.loads(row[1])
    except (json_mod.JSONDecodeError, TypeError):
        tags = []

    return SuggestedTagsResponse(
        available=True,
        file_id=row[0],
        tags=tags,
        model=row[2],
        status=row[3],
        created_at=row[4],
    )


@router.post("/files/{file_id}/suggested-tags/dismiss", response_model=MessageResponse)
async def dismiss_suggested_tags(file_id: str) -> MessageResponse:
    """Dismiss suggested tags for a file."""
    from app.database import get_search_db

    with get_search_db() as session:
        result = session.execute(
            sql_text(
                "UPDATE suggested_tags SET status = 'dismissed' "
                "WHERE file_id = :fid"
            ),
            {"fid": file_id},
        )

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="No suggested tags found")

    return MessageResponse(status="ok", message="Suggested tags dismissed")


@router.post("/files/{file_id}/suggested-tags/regenerate", response_model=MessageResponse)
async def regenerate_suggested_tags(file_id: str) -> MessageResponse:
    """Delete existing suggested tags and re-queue for auto-tagging."""
    from app.database import get_search_db

    if settings.features.auto_tags == "false":
        raise HTTPException(status_code=400, detail="Auto-tags feature is disabled")

    auto_tags_worker = get_auto_tags_worker()

    # Delete existing entry
    with get_search_db() as session:
        session.execute(
            sql_text("DELETE FROM suggested_tags WHERE file_id = :fid"),
            {"fid": file_id},
        )

    # Re-queue
    await auto_tags_worker.enqueue(file_id)

    return MessageResponse(
        status="accepted", message="Regeneration queued"
    )


@router.post("/batch/suggested-tags", response_model=BatchSuggestedTagsResponse)
async def batch_suggested_tags(body: BatchSuggestedTagsRequest) -> BatchSuggestedTagsResponse:
    """Queue auto-tagging for a batch of files. Skips files that already have suggestions."""
    from app.database import get_search_db

    if settings.features.auto_tags == "false":
        raise HTTPException(status_code=400, detail="Auto-tags feature is disabled")

    auto_tags_worker = get_auto_tags_worker()

    # Find which files already have suggested tags
    with get_search_db() as session:
        existing: set[str] = set()
        if body.file_ids:
            placeholders = ",".join(f":id{i}" for i in range(len(body.file_ids)))
            params = {f"id{i}": fid for i, fid in enumerate(body.file_ids)}
            for row in session.execute(
                sql_text(f"SELECT file_id FROM suggested_tags WHERE file_id IN ({placeholders})"),
                params,
            ).fetchall():
                existing.add(row[0])

    queued = 0
    skipped = 0
    for file_id in body.file_ids:
        if file_id in existing:
            skipped += 1
        else:
            await auto_tags_worker.enqueue(file_id)
            queued += 1

    return BatchSuggestedTagsResponse(queued=queued, skipped=skipped)
