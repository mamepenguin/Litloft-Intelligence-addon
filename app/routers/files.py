"""File inspection and suggested tags endpoints."""

import logging
import subprocess
from typing import Annotated, Any
from urllib.parse import unquote

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import text as sql_text

from app.config import settings
from app.dependencies import get_auto_tags_worker
from app.drive_context import assert_file_in_drive, require_drive
from app.schemas import (
    BatchSuggestedTagsRequest,
    BatchSuggestedTagsResponse,
    ChunkExcerptResponse,
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


def _get_indexed_file_or_404(file_id: str, drive: str) -> dict[str, Any]:
    """Get an indexed file by ID or raise 404.

    Returns 404 both for unknown file_ids and for files that belong to a
    drive other than the request's. Treating both as 404 keeps the API
    from leaking which file_ids exist outside the current drive.
    """
    from app.database import get_search_db
    from app.models import IndexedFile

    with get_search_db() as db:
        indexed = db.query(IndexedFile).filter(
            IndexedFile.file_id == file_id,
            IndexedFile.active.is_(True),
        ).first()
        if not indexed:
            raise HTTPException(status_code=404, detail="File not indexed")
        assert_file_in_drive(indexed.drive, drive)
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
async def get_transcript(
    file_id: str,
    drive: str = Depends(require_drive),
) -> TranscriptResponse:
    """Get Whisper transcript chunks for a file."""
    from app.database import get_search_db
    from app.models import TranscriptChunk

    indexed = _get_indexed_file_or_404(file_id, drive)

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
                text_refined_at=c.text_refined_at,
            )
            for c in chunks
        ],
    )


@router.get("/files/{file_id}/subtitles.vtt")
async def get_subtitles_vtt(
    file_id: str,
    x_hv_drive: Annotated[str | None, Header(alias="X-HV-Drive")] = None,
) -> Response:
    """Return Whisper-derived subtitles as a WebVTT document.

    Drive context is optional here because the route is loaded via
    ``<track src>`` which can't carry custom headers (mirrors the
    ``/files/{id}/frame`` pattern). The host's ``file_access``
    pre_check still verifies drive access for the specific file; when
    the header *is* present we additionally enforce the strict
    current-drive match for non-track consumers.

    Built on demand from the ``transcript_words`` table. The file must
    have been indexed after the word-level rollout; older files lack
    words and yield 404 until they are re-indexed.
    """
    from app.database import get_search_db
    from app.models import IndexedFile, TranscriptWord
    from app.subtitle_builder import build_vtt

    drive = unquote(x_hv_drive) if x_hv_drive else None
    if drive is None:
        with get_search_db() as db:
            row = (
                db.query(IndexedFile)
                .filter(
                    IndexedFile.file_id == file_id,
                    IndexedFile.active.is_(True),
                )
                .first()
            )
        if not row:
            raise HTTPException(status_code=404, detail="File not indexed")
    else:
        _get_indexed_file_or_404(file_id, drive)

    with get_search_db() as db:
        rows = (
            db.query(TranscriptWord)
            .filter(TranscriptWord.file_id == file_id)
            .order_by(TranscriptWord.timestamp_start, TranscriptWord.id)
            .all()
        )

    if not rows:
        raise HTTPException(status_code=404, detail="No word-level transcript available")

    language = rows[0].language or ""
    words = [
        {
            "text": r.text,
            "timestamp_start": r.timestamp_start,
            "timestamp_end": r.timestamp_end,
        }
        for r in rows
    ]

    vtt = build_vtt(words, language=language)
    return Response(
        content=vtt,
        media_type="text/vtt; charset=utf-8",
        headers={"Cache-Control": "private, max-age=300"},
    )


@router.get("/files/{file_id}/index-details", response_model=IndexDetailsResponse)
async def get_index_details(
    file_id: str,
    drive: str = Depends(require_drive),
) -> IndexDetailsResponse:
    """Get detailed indexing status and embedding info for a file."""
    from sqlalchemy import func
    from app.database import get_search_db
    from app.models import Embedding

    indexed = _get_indexed_file_or_404(file_id, drive)

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
async def get_clip_timestamps(
    file_id: str,
    drive: str = Depends(require_drive),
) -> ClipTimestampsResponse:
    """Get CLIP frame extraction timestamps for a file."""
    from app.database import get_search_db
    from app.models import Embedding

    indexed = _get_indexed_file_or_404(file_id, drive)

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


_EXCERPT_CONTEXT_CHARS = 100


def _parse_chunk_id(chunk_id: str) -> tuple[str, int]:
    """Split ``"transcript:5"`` / ``"document:12"`` into (source, index).

    Raises ``HTTPException(400)`` when the format is malformed — unknown
    prefix, missing colon, or non-integer index. Keeping the parse
    failure as 400 (rather than 404) lets the frontend distinguish a
    client bug from a legitimately missing chunk.
    """
    if not chunk_id or ":" not in chunk_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid chunk_id format; expected '<source>:<index>'",
        )
    source, _, raw_index = chunk_id.partition(":")
    if source not in {"transcript", "document"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid chunk_id source; expected 'transcript' or 'document'",
        )
    try:
        index = int(raw_index)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid chunk_id index; expected an integer",
        ) from None
    if index < 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid chunk_id index; must be non-negative",
        )
    return source, index


def _compose_excerpt(
    target: str,
    prev: str | None,
    nxt: str | None,
    context_chars: int = _EXCERPT_CONTEXT_CHARS,
) -> str:
    """Build the ±``context_chars`` excerpt around ``target``.

    Takes the tail of ``prev`` and head of ``nxt`` (up to ``context_chars``
    each) and joins them with ``" … "`` when the neighbour text was
    actually truncated. Empty / missing neighbours are skipped silently
    so edge chunks don't advertise phantom context.
    """
    parts: list[str] = []
    if prev:
        tail = prev[-context_chars:]
        if len(prev) > context_chars:
            parts = [*parts, f"… {tail}"]
        else:
            parts = [*parts, tail]
    parts = [*parts, target]
    if nxt:
        head = nxt[:context_chars]
        if len(nxt) > context_chars:
            parts = [*parts, f"{head} …"]
        else:
            parts = [*parts, head]
    return " ".join(parts)


@router.get(
    "/files/{file_id}/chunks/{chunk_id}/excerpt",
    response_model=ChunkExcerptResponse,
)
async def get_chunk_excerpt(
    file_id: str,
    chunk_id: str,
    drive: str = Depends(require_drive),
) -> ChunkExcerptResponse:
    """Return the text of a cited chunk with ±100 chars of context.

    Citations in the detailed-summary response store ``chunk_id`` as a
    prefixed identifier (``transcript:{idx}`` / ``document:{idx}``).
    This endpoint resolves that id back to the underlying text — the
    chunk's own body plus a short excerpt from the neighbour chunks —
    so the UI can render a preview without refetching the whole
    transcript or document.

    Behaviour:

    * 400 when ``chunk_id`` is malformed (bad prefix / non-integer index).
    * 404 when the file is not indexed, lives in another drive, the
      detailed-summaries feature is disabled for this drive, or the
      specific chunk row does not exist.
    * Transcript chunks return ``start_time`` / ``end_time``; ``page``
      is null.
    * Document chunks return ``page`` (when the extractor provided one);
      ``start_time`` / ``end_time`` are null.
    """
    # Per-drive policy: when detailed_summaries is off the citations
    # UX is unreachable and this endpoint serves no legitimate caller.
    # Surface it as 404 (not 400) to match the host's addon_feature
    # pre_check so disabled drives behave identically end-to-end.
    if settings.features.detailed_summaries == "false":
        raise HTTPException(status_code=404, detail="Not found")

    source, chunk_index = _parse_chunk_id(chunk_id)

    # Confirms file exists and belongs to the caller's drive. Raises
    # 404 otherwise — never leaks the existence of files in other
    # drives.
    _get_indexed_file_or_404(file_id, drive)

    from app.database import get_search_db
    from app.models import TranscriptChunk

    if source == "transcript":
        with get_search_db() as db:
            target_row = (
                db.query(TranscriptChunk)
                .filter(
                    TranscriptChunk.file_id == file_id,
                    TranscriptChunk.chunk_index == chunk_index,
                )
                .first()
            )
            if target_row is None:
                raise HTTPException(status_code=404, detail="Chunk not found")

            prev_row = (
                db.query(TranscriptChunk)
                .filter(
                    TranscriptChunk.file_id == file_id,
                    TranscriptChunk.chunk_index == chunk_index - 1,
                )
                .first()
            )
            next_row = (
                db.query(TranscriptChunk)
                .filter(
                    TranscriptChunk.file_id == file_id,
                    TranscriptChunk.chunk_index == chunk_index + 1,
                )
                .first()
            )

            text = _compose_excerpt(
                target_row.text,
                prev_row.text if prev_row else None,
                next_row.text if next_row else None,
            )
            start = target_row.timestamp_start
            end = target_row.timestamp_end

        return ChunkExcerptResponse(
            chunk_id=chunk_id,
            file_id=file_id,
            text=text,
            start_time=start,
            end_time=end,
            page=None,
        )

    # source == "document": fts_text_content is an FTS5 virtual table
    # whose ``chunk_index`` and ``page`` columns are stored as TEXT, so
    # numeric comparisons must CAST to INTEGER (see hako memo
    # EAiVExR4vGgOym5aAv_Up).
    with get_search_db() as db:
        target = db.execute(
            sql_text(
                "SELECT text, page FROM fts_text_content "
                "WHERE file_id = :fid "
                "AND CAST(chunk_index AS INTEGER) = :idx"
            ),
            {"fid": file_id, "idx": chunk_index},
        ).fetchone()
        if target is None:
            raise HTTPException(status_code=404, detail="Chunk not found")

        prev_row = db.execute(
            sql_text(
                "SELECT text FROM fts_text_content "
                "WHERE file_id = :fid "
                "AND CAST(chunk_index AS INTEGER) = :idx"
            ),
            {"fid": file_id, "idx": chunk_index - 1},
        ).fetchone()
        next_row = db.execute(
            sql_text(
                "SELECT text FROM fts_text_content "
                "WHERE file_id = :fid "
                "AND CAST(chunk_index AS INTEGER) = :idx"
            ),
            {"fid": file_id, "idx": chunk_index + 1},
        ).fetchone()

    target_text = target[0] or ""
    page_raw = target[1]
    page: int | None
    try:
        page = int(page_raw) if page_raw not in (None, "") else None
    except (TypeError, ValueError):
        page = None

    text = _compose_excerpt(
        target_text,
        prev_row[0] if prev_row else None,
        next_row[0] if next_row else None,
    )

    return ChunkExcerptResponse(
        chunk_id=chunk_id,
        file_id=file_id,
        text=text,
        start_time=None,
        end_time=None,
        page=page,
    )


@router.get("/files/{file_id}/frame")
async def get_frame(
    file_id: str,
    t: float = Query(..., ge=0, description="Timestamp in seconds"),
    x_hv_drive: Annotated[str | None, Header(alias="X-HV-Drive")] = None,
) -> Response:
    """Extract a single video frame at the given timestamp using ffmpeg.

    Drive context is optional here because the route is loaded via
    ``<img src>`` which can't carry custom headers. The host's
    ``file_access`` pre_check still verifies the caller can read the
    file's drive; when the header *is* present we additionally enforce
    the strict current-drive match (defence in depth for non-image
    consumers).
    """
    from app.config import resolve_file_path, validate_file_path

    drive = unquote(x_hv_drive) if x_hv_drive else None
    if drive is None:
        # Look up the indexed file without the drive assertion. The
        # host already authorised the caller for this specific file.
        from app.database import get_search_db
        from app.models import IndexedFile
        with get_search_db() as db:
            row = (
                db.query(IndexedFile)
                .filter(
                    IndexedFile.file_id == file_id,
                    IndexedFile.active.is_(True),
                )
                .first()
            )
        if not row:
            raise HTTPException(status_code=404, detail="File not indexed")
        indexed = {
            "file_id": row.file_id,
            "drive": row.drive,
            "filename": row.filename,
            "file_path": row.file_path,
            "file_type": row.file_type,
        }
    else:
        indexed = _get_indexed_file_or_404(file_id, drive)

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
async def get_suggested_tags(
    file_id: str,
    drive: str = Depends(require_drive),
) -> SuggestedTagsResponse:
    """Get suggested tags for a file."""
    import json as json_mod
    from app.database import get_search_db

    if settings.features.auto_tags == "false":
        return SuggestedTagsResponse(available=False)

    # Confirm the target file lives in the request drive before reading
    # its suggestions; otherwise a malicious caller could probe other
    # drives' file_ids.
    _get_indexed_file_or_404(file_id, drive)

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
async def dismiss_suggested_tags(
    file_id: str,
    drive: str = Depends(require_drive),
) -> MessageResponse:
    """Dismiss suggested tags for a file."""
    from app.database import get_search_db

    _get_indexed_file_or_404(file_id, drive)

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
async def regenerate_suggested_tags(
    file_id: str,
    drive: str = Depends(require_drive),
) -> MessageResponse:
    """Delete existing suggested tags and re-queue for auto-tagging."""
    from app.database import get_search_db

    if settings.features.auto_tags == "false":
        raise HTTPException(status_code=400, detail="Auto-tags feature is disabled")

    _get_indexed_file_or_404(file_id, drive)
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
async def batch_suggested_tags(
    body: BatchSuggestedTagsRequest,
    drive: str = Depends(require_drive),
) -> BatchSuggestedTagsResponse:
    """Queue auto-tagging for a batch of files in the current drive.

    Files that don't belong to the request drive are silently dropped
    (counted as skipped) so other-drive file_ids cannot be probed and
    cross-drive LLM cost cannot be incurred from a single drive context.
    """
    from app.database import get_search_db
    from app.models import IndexedFile

    if settings.features.auto_tags == "false":
        raise HTTPException(status_code=400, detail="Auto-tags feature is disabled")

    auto_tags_worker = get_auto_tags_worker()

    in_drive: set[str] = set()
    existing: set[str] = set()
    if body.file_ids:
        with get_search_db() as session:
            in_drive = {
                row.file_id
                for row in session.query(IndexedFile.file_id)
                .filter(
                    IndexedFile.file_id.in_(body.file_ids),
                    IndexedFile.drive == drive,
                    IndexedFile.active.is_(True),
                )
                .all()
            }
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
        if file_id not in in_drive or file_id in existing:
            skipped += 1
        else:
            await auto_tags_worker.enqueue(file_id)
            queued += 1

    return BatchSuggestedTagsResponse(queued=queued, skipped=skipped)
