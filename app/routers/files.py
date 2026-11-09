"""File inspection and suggested tags endpoints."""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from urllib.parse import unquote

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from sqlalchemy import text as sql_text

from app.config import settings
from app.idf_backfill import mark_idf_terms_stale
from app.dependencies import get_auto_tags_worker, get_index_manager
from app.document_sections import EPUB_MIME, load_section_titles
from app.drive_context import assert_file_in_drive, require_drive
from app.workers.ocr import OCR_IMAGE_TYPES
from app.workers.clip import VIDEO_TYPES, request_storyboard_retry
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
    PdfMarkdownResponse,
    ProcessingFile,
    ProcessingRequest,
    ProcessingResponse,
    ProcessingStage,
    ProviderStats,
    PictureTextResponse,
    PictureTextUpdateRequest,
    ReindexResponse,
    SuggestedTagsResponse,
    TranscriptChunkResponse,
    TranscriptResponse,
    UtteranceCorrectionRequest,
)
from app import extraction

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
    from app.database import get_search_db_read
    from app.models import IndexedFile

    with get_search_db_read() as db:
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
            "mime_type": indexed.mime_type,
            "metadata_indexed": indexed.metadata_indexed,
            "clip_indexed": indexed.clip_indexed,
            "whisper_indexed": indexed.whisper_indexed,
            "text_indexed": indexed.text_indexed,
            "ocr_indexed": indexed.ocr_indexed,
            "frame_ocr_indexed": indexed.frame_ocr_indexed,
            "page_ocr_indexed": indexed.page_ocr_indexed,
            "indexed_at": indexed.indexed_at.isoformat() if indexed.indexed_at else "",
            "transcript_corrected": indexed.transcript_corrected,
            "picture_text_corrected": indexed.picture_text_corrected,
            "transcript_redo": indexed.transcript_redo,
            "picture_text_redo": indexed.picture_text_redo,
            "transcript_redo_ended": indexed.transcript_redo_ended,
            "picture_text_redo_ended": indexed.picture_text_redo_ended,
        }


_SEARCH_I18N_KEY = "intelligence.processing.search"
_TEXT_I18N_KEY = "intelligence.processing.text"
_TRANSCRIPT_I18N_KEY = "intelligence.processing.transcript"


def _search_stage(row: Any) -> ProcessingStage:
    """Required, ``done`` only for an active row of this drive that is indexed."""
    done = row is not None and row.active and row.metadata_indexed
    return ProcessingStage(
        key="search",
        i18n_key=_SEARCH_I18N_KEY,
        state="done" if done else "pending",
        required=True,
    )


def _text_stage(row: Any) -> ProcessingStage:
    """Informational: this addon records no terminal "will not run" for text."""
    return ProcessingStage(
        key="text",
        i18n_key=_TEXT_I18N_KEY,
        state="done" if row.text_indexed else "pending",
        required=False,
    )


def _transcript_stage(row: Any, latest_job: Any) -> ProcessingStage:
    """Follow the newest transcription attempt and the whisper flag.

    A failure is only reported when nothing will retry it: the row is
    already ``whisper_indexed`` (a ``.loft`` whose temporary audio was
    consumed) or the provider called it fatal. Transient, rate-limit,
    circuit-breaker and container-restart failures stay ``pending``.
    """
    if latest_job is not None and latest_job.status == "running":
        state = "running"
    elif (
        latest_job is not None
        and latest_job.status == "failed"
        and (row.whisper_indexed or latest_job.error_class == "FatalError")
    ):
        state = "failed"
    elif row.whisper_indexed:
        state = "done"
    else:
        state = "pending"
    return ProcessingStage(
        key="transcript",
        i18n_key=_TRANSCRIPT_I18N_KEY,
        state=state,
        required=False,
    )


@router.post("/files/processing", response_model=ProcessingResponse)
def get_files_processing(
    body: ProcessingRequest,
    drive: str = Depends(require_drive),
) -> ProcessingResponse:
    """Report the ingest stages of the requested files of the caller's drive.

    SPEC-CORE-031 step 3 / SPEC-ADDON-057 I5. The core calls this on the
    addon's proxy target directly, which is why the path is absent from
    ``manifest.json``'s browser-facing route table.

    An id with no row yet still answers with ``search`` pending: an
    upload reaches the drawer before its first webhook creates the row.
    A row of another drive is left out entirely, so the answer never
    reveals a file outside the request drive.
    """
    from app.database import get_search_db_read
    from app.indexer import TEXT_MIMES
    from app.models import IndexedFile, JobRecord
    from app.workers.whisper import LOFT_MIME, TRANSCRIBABLE_TYPES

    requested = list(dict.fromkeys(body.file_ids))

    with get_search_db_read() as db:
        rows = {
            row.file_id: row
            for row in db.query(IndexedFile)
            .filter(IndexedFile.file_id.in_(requested))
            .all()
        }

        jobs = (
            db.query(JobRecord)
            .filter(
                JobRecord.file_id.in_(requested),
                JobRecord.job_kind == "transcription",
            )
            .order_by(JobRecord.id)
            .all()
        )
        # Ascending id: the last row per file is its newest attempt.
        latest_jobs = {job.file_id: job for job in jobs}

        files: dict[str, ProcessingFile] = {}
        for file_id in requested:
            row = rows.get(file_id)
            if row is not None and row.drive != drive:
                continue

            stages = [_search_stage(row)]
            if row is not None and row.mime_type in TEXT_MIMES:
                stages.append(_text_stage(row))
            if row is not None and (
                row.mime_type in TRANSCRIBABLE_TYPES
                or row.mime_type == LOFT_MIME
            ):
                stages.append(_transcript_stage(row, latest_jobs.get(file_id)))

            files[file_id] = ProcessingFile(stages=stages)

    return ProcessingResponse(files=files)


@router.get("/files/{file_id}/transcript", response_model=TranscriptResponse)
def get_transcript(
    file_id: str,
    drive: str = Depends(require_drive),
) -> TranscriptResponse:
    """Get Whisper transcript chunks for a file."""
    from app.database import get_search_db_read
    from app.models import FollowupItem, KeptOriginal, TranscriptChunk

    indexed = _get_indexed_file_or_404(file_id, drive)

    with get_search_db_read() as db:
        chunks = (
            db.query(TranscriptChunk)
            .filter(TranscriptChunk.file_id == file_id)
            .order_by(TranscriptChunk.chunk_index)
            .all()
        )
        has_original = (
            db.query(KeptOriginal)
            .filter_by(file_id=file_id, kind=extraction.KIND_TRANSCRIPT)
            .first()
            is not None
        )
        followups_pending = (
            db.query(FollowupItem.id).filter_by(file_id=file_id).first()
            is not None
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
                corrected_at=c.corrected_at,
                words_pending=bool(c.words_pending),
            )
            for c in chunks
        ],
        has_original=has_original,
        corrected=bool(indexed["transcript_corrected"]),
        busy=extraction.busy(file_id),
        followups_pending=followups_pending,
    )


def _transcript_answer(file_id: str, drive: str) -> TranscriptResponse:
    """The transcript GET shape, re-read after a write."""
    from app.database import get_search_db_read
    from app.models import FollowupItem, KeptOriginal, TranscriptChunk

    indexed = _get_indexed_file_or_404(file_id, drive)
    with get_search_db_read() as db:
        chunks = (
            db.query(TranscriptChunk)
            .filter(TranscriptChunk.file_id == file_id)
            .order_by(TranscriptChunk.chunk_index)
            .all()
        )
        has_original = (
            db.query(KeptOriginal)
            .filter_by(file_id=file_id, kind=extraction.KIND_TRANSCRIPT)
            .first()
            is not None
        )
        followups_pending = (
            db.query(FollowupItem.id).filter_by(file_id=file_id).first()
            is not None
        )
    return TranscriptResponse(
        file_id=file_id,
        drive=indexed["drive"],
        language=chunks[0].language if chunks else "",
        chunks=[
            TranscriptChunkResponse(
                index=c.chunk_index,
                text=c.text,
                start=c.timestamp_start,
                end=c.timestamp_end,
                text_refined_at=c.text_refined_at,
                corrected_at=c.corrected_at,
                words_pending=bool(c.words_pending),
            )
            for c in chunks
        ],
        has_original=has_original,
        corrected=bool(indexed["transcript_corrected"]),
        busy=extraction.busy(file_id),
        followups_pending=followups_pending,
    )


def _coded(status_code: int, code: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code})


@router.put(
    "/files/{file_id}/transcript/utterances/{index}",
    response_model=TranscriptChunkResponse,
)
def correct_utterance(
    file_id: str,
    index: int,
    body: UtteranceCorrectionRequest,
    drive: str = Depends(require_drive),
) -> TranscriptChunkResponse:
    """Save one corrected utterance. SPEC-ADDON-070 A3-A7."""
    from app.database import get_search_db
    from app.models import Embedding, TranscriptChunk, TranscriptWord

    _get_indexed_file_or_404(file_id, drive)
    text = body.text.strip()
    expected_text = body.expected_text.strip()

    def _apply() -> tuple[dict[str, Any], bool]:
        with get_search_db() as session:
            chunks = (
                session.query(TranscriptChunk)
                .filter_by(file_id=file_id)
                .order_by(TranscriptChunk.chunk_index)
                .all()
            )
            if not chunks:
                raise HTTPException(status_code=404, detail="No transcript available")
            target = next(
                (c for c in chunks if c.chunk_index == index), None
            )
            if extraction.busy(file_id):
                raise _coded(409, "busy")
            if (
                target is None
                or target.text != expected_text
                or target.timestamp_start != body.expected_start
            ):
                raise _coded(409, "changed")
            if "\n" in text or "\r" in text:
                raise _coded(422, "line_break")
            if len(text) > 2000:
                raise _coded(422, "too_long")
            if not text and len(chunks) == 1:
                raise _coded(422, "last_utterance")

            answer = {
                "index": target.chunk_index,
                "text": text,
                "start": target.timestamp_start,
                "end": target.timestamp_end,
                "corrected_at": None,
                "words_pending": False,
            }
            if text == target.text:
                return answer, False

            extraction.store_original_if_absent(
                session, file_id, extraction.KIND_TRANSCRIPT, "correction"
            )
            extraction.mark_corrected(
                session, file_id, extraction.KIND_TRANSCRIPT
            )

            # A zero-duration word sits on the boundary two utterances
            # share; it is not the text of either, so saving one leaves
            # it for the neighbour. The start one already belongs to the
            # previous utterance.
            prev_end: float | None = None
            for chunk in chunks:
                if chunk.timestamp_start >= target.timestamp_start:
                    break
                if prev_end is None or chunk.timestamp_end > prev_end:
                    prev_end = chunk.timestamp_end
            next_start: float | None = None
            for chunk in chunks:
                if chunk.timestamp_start <= target.timestamp_start:
                    continue
                if next_start is None or chunk.timestamp_start < next_start:
                    next_start = chunk.timestamp_start

            def _shared_boundary(word: Any) -> bool:
                if word.timestamp_start != word.timestamp_end:
                    return False
                at = word.timestamp_start
                return (
                    at == target.timestamp_start and prev_end == at
                ) or (
                    at == target.timestamp_end and next_start == at
                )

            words = (
                session.query(TranscriptWord)
                .filter(
                    TranscriptWord.file_id == file_id,
                    TranscriptWord.timestamp_start >= target.timestamp_start,
                    TranscriptWord.timestamp_end <= target.timestamp_end,
                )
                .all()
            )
            for word in words:
                if _shared_boundary(word):
                    continue
                session.delete(word)

            now = datetime.now(UTC)
            if text:
                target.text = text
                target.corrected_at = now
                target.words_pending = True
                target.words_pending_attempts = 0
                answer["corrected_at"] = now
                answer["words_pending"] = True
            else:
                session.delete(target)

            session.flush()
            from app.database import upsert_fts_transcripts

            rows = (
                session.query(TranscriptChunk)
                .filter_by(file_id=file_id)
                .order_by(TranscriptChunk.chunk_index)
                .all()
            )
            upsert_fts_transcripts(
                session,
                file_id,
                [
                    {"chunk_index": c.chunk_index, "text": c.text}
                    for c in rows
                    if (c.text or "").strip()
                ],
            )

            embeddings = (
                session.query(Embedding)
                .filter(
                    Embedding.file_id == file_id,
                    Embedding.embedding_type.in_(("whisper", "tfidf_keywords")),
                )
                .all()
            )
            for emb in embeddings:
                if (
                    emb.embedding_type == "whisper"
                    and (
                        emb.timestamp_start != target.timestamp_start
                        or emb.timestamp_end != target.timestamp_end
                    )
                ):
                    continue
                session.execute(
                    sql_text("DELETE FROM vec_text WHERE embedding_id = :id"),
                    {"id": emb.id},
                )
                session.delete(emb)

            file_row = extraction.get_file(session, file_id)
            if file_row is not None:
                file_row.tfidf_keywords_indexed = False
            mark_idf_terms_stale(session, file_id)
            extraction.queue_item(
                session,
                file_id,
                extraction.KIND_TRANSCRIPT,
                "utterance" if text else "keyword",
                chunk_index=target.chunk_index if text else None,
                expected_text=text if text else None,
            )
            return answer, True

    answer, _ = _apply()
    return TranscriptChunkResponse(
        index=answer["index"],
        text=answer["text"],
        start=answer["start"],
        end=answer["end"],
        corrected_at=answer["corrected_at"],
        words_pending=answer["words_pending"],
    )


@router.post(
    "/files/{file_id}/transcript/restore",
    response_model=TranscriptResponse,
)
async def restore_transcript_route(
    file_id: str,
    drive: str = Depends(require_drive),
) -> TranscriptResponse:
    """Put the kept original transcript back. SPEC-ADDON-068 C2-C5."""
    from app.database import get_search_db
    from app.addon_events import post_addon_event

    indexed = await asyncio.to_thread(_get_indexed_file_or_404, file_id, drive)

    def _apply() -> None:
        if extraction.busy(file_id):
            raise _coded(409, "busy")
        extraction.begin_restore(file_id)
        try:
            with get_search_db() as session:
                if not extraction.restore_transcript(session, file_id):
                    raise HTTPException(status_code=404, detail="No original available")
        finally:
            extraction.end_restore(file_id)

    await asyncio.to_thread(_apply)
    await post_addon_event(
        "intelligence.transcript.updated",
        {
            "file_id": file_id,
            "drive": indexed["drive"],
            "reason": "restored",
        },
    )
    return await asyncio.to_thread(_transcript_answer, file_id, drive)


# --- Picture text (SPEC-ADDON-067) -----------------------------------------


def _is_picture(indexed: dict[str, Any]) -> bool:
    return (
        indexed["file_type"] == "image"
        or indexed["mime_type"] in OCR_IMAGE_TYPES
    )


def _rebuild_picture_text(file_id: str, indexed: dict[str, Any]) -> str | None:
    """The text a picture read before the stored-text change has.

    Rebuilt from ``fts_text_content`` with the same merge find-in-file
    uses. The first correction materialises it as the picture text and
    keeps it as the original.
    """
    from app.database import get_search_db_read
    from app.file_find import _merge_chunks, _replay_limit

    with get_search_db_read() as session:
        rows = session.execute(
            sql_text(
                "SELECT CAST(chunk_index AS INTEGER) AS ci, text "
                "FROM fts_text_content WHERE file_id = :fid ORDER BY ci"
            ),
            {"fid": file_id},
        ).fetchall()
    if not rows:
        return None
    limit = _replay_limit(indexed["mime_type"], indexed["file_type"])
    return _merge_chunks([r[1] for r in rows], limit)


def _picture_answer(file_id: str, drive: str) -> PictureTextResponse:
    from app.database import get_search_db_read
    from app.models import PictureText

    indexed = _get_indexed_file_or_404(file_id, drive)
    if not _is_picture(indexed):
        raise HTTPException(status_code=404, detail="Not a picture")
    with get_search_db_read() as session:
        row = session.query(PictureText).filter_by(file_id=file_id).first()
    text = row.text if row is not None else None
    if text is None:
        text = _rebuild_picture_text(file_id, indexed)
    has_original = False
    with get_search_db_read() as session:
        has_original = extraction.has_original(
            session, file_id, extraction.KIND_PICTURE
        )
    if text is None or (not text and not has_original):
        raise HTTPException(status_code=404, detail="No picture text available")
    return PictureTextResponse(
        file_id=file_id,
        text=text,
        corrected=bool(indexed["picture_text_corrected"]),
        has_original=has_original,
        busy=extraction.busy(file_id),
    )


@router.get("/files/{file_id}/picture-text", response_model=PictureTextResponse)
def get_picture_text_route(
    file_id: str,
    drive: str = Depends(require_drive),
) -> PictureTextResponse:
    """The stored text of one picture. SPEC-ADDON-067 B1/B8/D."""
    return _picture_answer(file_id, drive)


@router.put("/files/{file_id}/picture-text", response_model=PictureTextResponse)
def update_picture_text_route(
    file_id: str,
    body: PictureTextUpdateRequest,
    drive: str = Depends(require_drive),
) -> PictureTextResponse:
    """Save a corrected picture text. SPEC-ADDON-067 B3-B7."""
    from app.database import get_search_db

    indexed = _get_indexed_file_or_404(file_id, drive)
    if not _is_picture(indexed):
        raise HTTPException(status_code=404, detail="Not a picture")
    text = _normalise_picture_text(body.text)
    expected_text = _normalise_picture_text(body.expected_text)

    def _apply() -> None:
        if extraction.busy(file_id):
            raise _coded(409, "busy")
        from app.database import get_search_db_read

        with get_search_db_read() as read_session:
            stored = extraction.get_picture_text(read_session, file_id)
        if stored is None:
            stored = _rebuild_picture_text(file_id, indexed)
        with get_search_db() as session:
            file_row = extraction.get_file(session, file_id)
            if (
                file_row is None
                or (not stored and not file_row.picture_text_corrected)
            ):
                raise HTTPException(status_code=404, detail="No picture text available")
            if stored != expected_text:
                raise _coded(409, "changed")
            if len(text) > 100_000:
                raise _coded(422, "too_long")
            if text == stored:
                return
            if extraction.get_picture_text(session, file_id) is None:
                extraction.set_picture_text(session, file_id, stored or "")
            extraction.store_original_if_absent(
                session, file_id, extraction.KIND_PICTURE, "correction"
            )
            extraction.mark_corrected(session, file_id, extraction.KIND_PICTURE)
            extraction.set_picture_text(session, file_id, text)
            from app.extractors.base import ContentExtractor

            chunks = ContentExtractor.chunk_text(text)
            from app.database import upsert_fts_text_content
            from app.workers.metadata import _remove_embeddings

            _remove_embeddings(session, file_id, "text_content")
            upsert_fts_text_content(
                session,
                file_id,
                [
                    {"chunk_index": index, "page": None, "text": chunk}
                    for index, chunk in enumerate(chunks)
                ],
            )
            mark_idf_terms_stale(session, file_id)
            extraction.queue_item(
                session,
                file_id,
                extraction.KIND_PICTURE,
                "picture",
                expected_text=text,
            )

    _apply()
    return _picture_answer(file_id, drive)


@router.post(
    "/files/{file_id}/picture-text/restore",
    response_model=PictureTextResponse,
)
async def restore_picture_text_route(
    file_id: str,
    drive: str = Depends(require_drive),
) -> PictureTextResponse:
    """Put the kept original picture text back. SPEC-ADDON-068 C2-C5."""
    from app.database import get_search_db
    from app.addon_events import post_addon_event

    indexed = await asyncio.to_thread(_get_indexed_file_or_404, file_id, drive)

    def _apply() -> None:
        if extraction.busy(file_id):
            raise _coded(409, "busy")
        extraction.begin_restore(file_id)
        try:
            with get_search_db() as session:
                if not extraction.restore_picture_text(session, file_id):
                    raise HTTPException(status_code=404, detail="No original available")
        finally:
            extraction.end_restore(file_id)

    await asyncio.to_thread(_apply)
    await post_addon_event(
        "intelligence.picture_text.updated",
        {
            "file_id": file_id,
            "drive": indexed["drive"],
            "reason": "restored",
        },
    )
    return await asyncio.to_thread(_picture_answer, file_id, drive)


def _normalise_picture_text(text: str) -> str:
    """Strip trailing space on each line and blank lines at the ends."""
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)



@router.get("/files/{file_id}/subtitles.vtt")
async def get_subtitles_vtt(
    file_id: str,
    x_hv_drive: Annotated[str | None, Header(alias="X-Lit-Drive")] = None,
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
    from app.database import get_search_db_read
    from app.models import IndexedFile, KeptOriginal, TranscriptChunk, TranscriptWord
    from app.subtitle_builder import build_vtt_with_fallback

    def _build_vtt() -> Response:
        drive = unquote(x_hv_drive) if x_hv_drive else None
        if drive is None:
            with get_search_db_read() as db:
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

        with get_search_db_read() as db:
            rows = (
                db.query(TranscriptWord)
                .filter(TranscriptWord.file_id == file_id)
                .order_by(TranscriptWord.timestamp_start, TranscriptWord.id)
                .all()
            )
            chunks = (
                db.query(TranscriptChunk)
                .filter(TranscriptChunk.file_id == file_id)
                .order_by(TranscriptChunk.chunk_index)
                .all()
            )
            original = (
                db.query(KeptOriginal)
                .filter_by(file_id=file_id, kind=extraction.KIND_TRANSCRIPT)
                .first()
            )

        original_words = 0
        if original is not None:
            _, snapshot_words = extraction.transcript_snapshot(original)
            original_words = len(snapshot_words)

        if not rows and original_words == 0:
            raise HTTPException(status_code=404, detail="No word-level transcript available")

        language = (
            rows[0].language
            if rows
            else (chunks[0].language if chunks else "")
        )
        words = [
            {
                "text": r.text,
                "timestamp_start": r.timestamp_start,
                "timestamp_end": r.timestamp_end,
            }
            for r in rows
        ]
        def _chunk_has_words(chunk: Any) -> bool:
            for word in rows:
                if not (
                    word.timestamp_start >= chunk.timestamp_start
                    and word.timestamp_end <= chunk.timestamp_end
                ):
                    continue
                if (
                    word.timestamp_start == word.timestamp_end
                    and word.timestamp_start
                    in (chunk.timestamp_start, chunk.timestamp_end)
                ):
                    continue
                return True
            return False

        fallback = [
            {
                "start": chunk.timestamp_start,
                "end": chunk.timestamp_end,
                "text": chunk.text,
            }
            for chunk in chunks
            if chunk.corrected_at is not None
            and not _chunk_has_words(chunk)
            and chunk.text.strip()
        ]

        vtt = build_vtt_with_fallback(words, fallback, language=language)
        return Response(
            content=vtt,
            media_type="text/vtt; charset=utf-8",
            headers={"Cache-Control": "private, no-cache"},
        )

    return await asyncio.to_thread(_build_vtt)


# --- Per-file × per-task reindex (spec 2026-05-24-intelligence-reindex-controls) ---

# Spec §2.1 fixes the task vocabulary; the handler validates the
# body against this set before touching any DB row so a malformed
# payload can never reset a flag partially.
_REINDEX_ALLOWED_TASKS: tuple[str, ...] = (
    "metadata", "clip", "whisper", "text", "ocr", "frame_ocr", "page_ocr",
)

# Spec task name → ``IndexedFile`` column name. The frontend speaks
# spec names; the DB column for the text-content leg is
# ``text_indexed`` (no ``_content`` suffix), so we keep the mapping
# explicit rather than string-concatenating.
_REINDEX_FLAG_COLUMN: dict[str, str] = {
    "metadata": "metadata_indexed",
    "clip": "clip_indexed",
    "whisper": "whisper_indexed",
    "text": "text_indexed",
    "ocr": "ocr_indexed",
    "frame_ocr": "frame_ocr_indexed",
    "page_ocr": "page_ocr_indexed",
}

# (file_id, task) pairs a reindex request has passed the ``is_queued``
# gate for but not yet enqueued. Touched only on the event loop.
_reindex_in_flight: set[tuple[str, str]] = set()


@router.post("/files/{file_id}/reindex")
async def reindex_file(
    file_id: str,
    body: dict[str, Any] = Body(...),
    drive: str = Depends(require_drive),
    manager: Any = Depends(get_index_manager),
):
    """Reset selected ``*_indexed`` flags for one file and re-enqueue.

    Spec ``2026-05-24-intelligence-reindex-controls.md`` §2.1. The
    legacy ``POST /queue/reindex`` global reset has been removed; this
    endpoint is the targeted replacement.

    Request body shape::

        {"tasks": ["whisper", "clip", ...]}

    Behaviour:

    * Unknown task names → 422 (``detail`` enumerates the allowed set)
      *before* any DB mutation, so a rejected payload can never leave a
      flag half-reset.
    * Empty ``tasks`` array → 422 (Pydantic-style validation surfaced
      as ``HTTPException`` for parity with the admin endpoints).
    * Unknown / soft-deleted / cross-drive ``file_id`` → 404 (never
      leak the existence of files in other drives — see the project's
      drive-boundary rule).
    * Tasks already sitting in the index queue are short-circuited: the
      response is 202 with ``status='already_queued'`` and **no** flag
      flip + **no** new enqueue. Mixed batches still progress the
      fresh tasks (queued ones drop out of ``tasks_reset``).
    """
    # ---- Layer 1: validate the request payload before any mutation ----
    raw_tasks = body.get("tasks") if isinstance(body, dict) else None
    if not isinstance(raw_tasks, list) or len(raw_tasks) == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "Request body must include a non-empty 'tasks' array. "
                f"Allowed: {', '.join(_REINDEX_ALLOWED_TASKS)}"
            ),
        )
    unknown = [t for t in raw_tasks if t not in _REINDEX_ALLOWED_TASKS]
    if unknown:
        # Surface the first unknown name explicitly so the operator can
        # spot a typo at a glance; the full allow-list is included so
        # they don't have to grep source.
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown task: {unknown[0]!r}. "
                f"Allowed: {', '.join(_REINDEX_ALLOWED_TASKS)}"
            ),
        )

    # Dedup while preserving order (a caller asking for whisper twice
    # in one payload would otherwise double-enqueue).
    seen: set[str] = set()
    tasks: list[str] = []
    for t in raw_tasks:
        if t not in seen:
            seen.add(t)
            tasks.append(t)

    # ---- Layer 2: drive boundary + soft-delete check ----
    # 404 (not cross-drive / soft-deleted / unknown — keeps the
    # drive boundary opaque to the caller.
    indexed = await asyncio.to_thread(
        _get_indexed_file_or_404, file_id, drive
    )

    # OCR has no meaning for a document, and "false" refuses the request
    # rather than silently ignoring it.
    if "ocr" in tasks and (
        settings.features.image_ocr == "false"
        or indexed["mime_type"] not in OCR_IMAGE_TYPES
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Task 'ocr' applies only to pictures and is unavailable "
                "while features.image_ocr is false"
            ),
        )

    if "frame_ocr" in tasks and (
        settings.features.video_frame_ocr == "false"
        or indexed["mime_type"] not in VIDEO_TYPES
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Task 'frame_ocr' applies only to videos and is unavailable "
                "while features.video_frame_ocr is false"
            ),
        )

    if "page_ocr" in tasks and (
        settings.features.scanned_pdf_ocr == "false"
        or indexed["mime_type"] != "application/pdf"
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Task 'page_ocr' applies only to PDFs and is unavailable "
                "while features.scanned_pdf_ocr is false"
            ),
        )

    # ---- Layer 2b: a corrected file needs its explicit confirmation ----
    overwrite = body.get("overwrite_corrections", False)
    if not isinstance(overwrite, bool):
        raise HTTPException(
            status_code=422,
            detail="overwrite_corrections must be a boolean",
        )
    corrected_kinds: list[str] = []
    if "whisper" in tasks and indexed["transcript_corrected"]:
        corrected_kinds.append(extraction.KIND_TRANSCRIPT)
    if "ocr" in tasks and indexed["picture_text_corrected"]:
        corrected_kinds.append(extraction.KIND_PICTURE)
    if corrected_kinds and not overwrite:
        raise HTTPException(
            status_code=409, detail={"code": "corrected"}
        )
    if corrected_kinds:
        if await asyncio.to_thread(extraction.busy, file_id):
            raise HTTPException(status_code=409, detail={"code": "busy"})

        def _set_redo_marks() -> None:
            from app.database import get_search_db

            with get_search_db() as session:
                for kind in corrected_kinds:
                    extraction.set_redo(session, file_id, kind)

        await asyncio.to_thread(_set_redo_marks)

    if "clip" in tasks:
        request_storyboard_retry(file_id)

    # ---- Layer 3: split tasks into already-queued vs fresh ----
    if manager is None or not hasattr(manager, "is_queued"):
        manager = get_index_manager()
    fresh: list[str] = []
    already_queued: list[str] = []
    for t in tasks:
        if manager.is_queued(file_id, t) or (file_id, t) in _reindex_in_flight:
            already_queued.append(t)
        else:
            fresh.append(t)

    # All tasks were already queued — short-circuit per spec §2.1.
    if fresh == []:
        return JSONResponse(
            status_code=202,
            content={
                "status": "already_queued",
                "file_id": file_id,
                "tasks_reset": [],
            },
        )

    # ---- Layer 4: flip flags and enqueue ----
    from app.database import get_search_db
    from app.models import IndexedFile

    def _reset_flags() -> None:
        with get_search_db() as session:
            row = (
                session.query(IndexedFile)
                .filter(
                    IndexedFile.file_id == file_id,
                    IndexedFile.active.is_(True),
                )
                .first()
            )
            # Re-fetch under the write session — between the lookup above
            # and now the row could have been soft-deleted; surface as 404.
            if row is None:
                raise HTTPException(status_code=404, detail="File not indexed")
            for t in fresh:
                setattr(row, _REINDEX_FLAG_COLUMN[t], False)
                if t == "frame_ocr":
                    row.frame_ocr_read = False

    # ``is_queued`` lives on the loop and cannot join the flag flip's
    # transaction, so the claim keeps a concurrent request that checks
    # during the offloaded write from passing the gate too.
    claimed = [(file_id, t) for t in fresh]
    _reindex_in_flight.update(claimed)
    try:
        await asyncio.to_thread(_reset_flags)
        for t in fresh:
            await manager.enqueue_task_for_file(file_id, t)
    finally:
        _reindex_in_flight.difference_update(claimed)

    return ReindexResponse(
        status="accepted",
        file_id=file_id,
        tasks_reset=fresh,
    )


@router.get("/files/{file_id}/index-details", response_model=IndexDetailsResponse)
async def get_index_details(
    file_id: str,
    drive: str = Depends(require_drive),
) -> IndexDetailsResponse:
    """Get detailed indexing status and embedding info for a file."""
    from sqlalchemy import func
    from app.database import get_search_db_read
    from app.models import Embedding

    indexed = await asyncio.to_thread(_get_indexed_file_or_404, file_id, drive)

    def _read_details() -> tuple[dict[str, IndexDetailType], dict[str, ProviderStats]]:
        embeddings_by_type: dict[str, IndexDetailType] = {}

        with get_search_db_read() as db:
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

            return embeddings_by_type, _compute_provider_stats(db, file_id)

    embeddings_by_type, provider_stats = await asyncio.to_thread(_read_details)

    return IndexDetailsResponse(
        file_id=file_id,
        drive=indexed["drive"],
        filename=indexed["filename"],
        image_ocr=settings.features.image_ocr,
        video_frame_ocr=settings.features.video_frame_ocr,
        scanned_pdf_ocr=settings.features.scanned_pdf_ocr,
        status={
            "metadata": indexed["metadata_indexed"],
            "clip": indexed["clip_indexed"],
            "whisper": indexed["whisper_indexed"],
            "text": indexed["text_indexed"],
            "ocr": indexed["ocr_indexed"],
            "frame_ocr": indexed["frame_ocr_indexed"],
            "page_ocr": indexed["page_ocr_indexed"],
        },
        indexed_at=indexed["indexed_at"],
        embeddings=embeddings_by_type,
        provider_stats=provider_stats,
        corrected={
            "transcript": bool(indexed["transcript_corrected"]),
            "picture_text": bool(indexed["picture_text_corrected"]),
        },
        redo_ended_without_result={
            "transcript": bool(indexed["transcript_redo_ended"]),
            "picture_text": bool(indexed["picture_text_redo_ended"]),
        },
    )


# Phase 1C: ``provider_stats`` aggregate window. 7 days matches the
# spec; older rows stay in job_records for historical inspection but
# the dashboard only surfaces "what's broken right now".
_PROVIDER_STATS_WINDOW_DAYS = 7


def _compute_provider_stats(db, file_id: str) -> dict[str, ProviderStats]:
    """Aggregate transcription JobRecord rows for the file (last 7 days).

    Returns a ``{provider_name: ProviderStats}`` mapping.

    Args:
        db: Active SQLAlchemy session bound to the search DB.
        file_id: The file whose history we want.
    """
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import func

    from app.models import JobRecord

    cutoff = datetime.now(UTC) - timedelta(days=_PROVIDER_STATS_WINDOW_DAYS)

    rows = (
        db.query(
            JobRecord.provider,
            JobRecord.status,
            func.count(JobRecord.id).label("count"),
        )
        .filter(
            JobRecord.file_id == file_id,
            JobRecord.job_kind == "transcription",
            JobRecord.attempted_at >= cutoff,
            JobRecord.provider.isnot(None),
        )
        .group_by(JobRecord.provider, JobRecord.status)
        .all()
    )

    stats: dict[str, dict[str, int]] = {}
    for provider, status, count in rows:
        bucket = stats.setdefault(provider, {"calls": 0, "failures": 0})
        bucket["calls"] += int(count)
        if status == "failed":
            bucket["failures"] += int(count)

    # Resolve last_error per provider via a single per-provider query
    # (cheap because each call does ``ORDER BY attempted_at DESC LIMIT 1``).
    out: dict[str, ProviderStats] = {}
    for provider, counts in stats.items():
        last_error_row = (
            db.query(JobRecord.error_class)
            .filter(
                JobRecord.file_id == file_id,
                JobRecord.job_kind == "transcription",
                JobRecord.provider == provider,
                JobRecord.status == "failed",
                JobRecord.attempted_at >= cutoff,
            )
            .order_by(JobRecord.attempted_at.desc())
            .first()
        )
        out[provider] = ProviderStats(
            calls=counts["calls"],
            failures=counts["failures"],
            last_error=last_error_row[0] if last_error_row else None,
        )
    return out


@router.get("/files/{file_id}/clip-timestamps", response_model=ClipTimestampsResponse)
def get_clip_timestamps(
    file_id: str,
    drive: str = Depends(require_drive),
) -> ClipTimestampsResponse:
    """Get CLIP frame extraction timestamps for a file."""
    from app.database import get_search_db_read
    from app.models import Embedding

    indexed = _get_indexed_file_or_404(file_id, drive)

    with get_search_db_read() as db:
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
) -> tuple[str, str, str]:
    """Build the ±``context_chars`` excerpt around ``target``.

    Returns ``(prefix, target, suffix)`` so the UI can render the
    target chunk as a visual highlight and the surrounding neighbour
    text as muted context. The prefix trails the target with a single
    space separator and the suffix leads with one — both already baked
    into the returned strings, so concatenation reproduces the old
    single-line layout. A leading / trailing ``"… "`` marker is
    appended only when the neighbour was actually truncated at
    ``context_chars``. Empty / missing neighbours yield empty strings
    rather than phantom context.
    """
    if prev:
        tail = prev[-context_chars:]
        prefix = f"… {tail} " if len(prev) > context_chars else f"{tail} "
    else:
        prefix = ""
    if nxt:
        head = nxt[:context_chars]
        suffix = f" {head} …" if len(nxt) > context_chars else f" {head}"
    else:
        suffix = ""
    return prefix, target, suffix


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

    def _build_excerpt() -> ChunkExcerptResponse:
        # Confirms file exists and belongs to the caller's drive. Raises
        # 404 otherwise — never leaks the existence of files in other
        # drives.
        indexed = _get_indexed_file_or_404(file_id, drive)
        is_book = indexed["mime_type"] == EPUB_MIME

        from app.database import get_search_db_read
        from app.models import TranscriptChunk

        if source == "transcript":
            with get_search_db_read() as db:
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
                        TranscriptChunk.chunk_index < chunk_index,
                    )
                    .order_by(TranscriptChunk.chunk_index.desc())
                    .first()
                )
                next_row = (
                    db.query(TranscriptChunk)
                    .filter(
                        TranscriptChunk.file_id == file_id,
                        TranscriptChunk.chunk_index > chunk_index,
                    )
                    .order_by(TranscriptChunk.chunk_index.asc())
                    .first()
                )

                prefix, target_text, suffix = _compose_excerpt(
                    target_row.text,
                    prev_row.text if prev_row else None,
                    next_row.text if next_row else None,
                )
                start = target_row.timestamp_start
                end = target_row.timestamp_end

            return ChunkExcerptResponse(
                chunk_id=chunk_id,
                file_id=file_id,
                prefix=prefix,
                target=target_text,
                suffix=suffix,
                start_time=start,
                end_time=end,
                page=None,
            )

        # source == "document": fts_text_content is an FTS5 virtual table
        # whose ``chunk_index`` and ``page`` columns are stored as TEXT, so
        # numeric comparisons must CAST to INTEGER (see hako memo
        # EAiVExR4vGgOym5aAv_Up).
        with get_search_db_read() as db:
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

            title = (
                load_section_titles(db, [(file_id, page)]).get((file_id, page))
                if is_book and page is not None
                else None
            )

        prefix, target_out, suffix = _compose_excerpt(
            target_text,
            prev_row[0] if prev_row else None,
            next_row[0] if next_row else None,
        )

        return ChunkExcerptResponse(
            chunk_id=chunk_id,
            file_id=file_id,
            prefix=prefix,
            target=target_out,
            suffix=suffix,
            start_time=None,
            end_time=None,
            page=None if is_book else page,
            section=page if is_book else None,
            section_title=title,
        )

    return await asyncio.to_thread(_build_excerpt)


@router.get(
    "/files/{file_id}/pdf-markdown",
    response_model=PdfMarkdownResponse,
)
async def get_pdf_markdown(
    file_id: str,
    drive: str = Depends(require_drive),
) -> PdfMarkdownResponse:
    """Return the persisted Markdown body for an indexed PDF.

    Markdown is produced by PyMuPDF4LLM during text-content indexing
    and stored 1:1 with the indexed file. Callers must already be
    authorised for the file's drive — ``_get_indexed_file_or_404``
    surfaces both unknown ``file_id`` and cross-drive access as 404 to
    avoid leaking which file_ids exist outside the current drive
    (matches the project-wide ``drive_boundary`` rule).

    Returns 404 when:

    * the file is not indexed (or lives in another drive),
    * the file is not a PDF (no ``pdf_markdown`` row will exist),
    * the PDF was extracted via the fitz fallback path (no row written
      because Markdown is not available for that path).

    The host's ``file_access`` pre_check provides the first layer of
    drive enforcement; the handler-side ``_get_indexed_file_or_404``
    is the second layer of defence in depth.
    """
    from app.database import get_search_db_read
    from app.models import PdfMarkdown

    def _read_markdown() -> PdfMarkdownResponse:
        _get_indexed_file_or_404(file_id, drive)

        with get_search_db_read() as db:
            row = (
                db.query(PdfMarkdown)
                .filter(PdfMarkdown.file_id == file_id)
                .first()
            )
            if row is None:
                raise HTTPException(status_code=404, detail="PDF Markdown not available")
            return PdfMarkdownResponse(
                file_id=row.file_id,
                markdown=row.markdown,
                page_count=row.page_count,
                extractor=row.extractor,
                generated_at=row.generated_at,
            )

    return await asyncio.to_thread(_read_markdown)


# --- CLIP frame extraction with persistent disk cache ---
#
# The cache implementation (path safety, atomic writes, purge) lives in
# app.frame_cache, shared with the video-visual worker. Thin aliases are
# kept here for existing internal callers / test monkeypatches.

from app.frame_cache import (  # noqa: E402
    extract_frame_to_cache as _extract_frame_to_cache,
    frame_cache_path as _frame_cache_path,
    purge_frame_cache,
    tile_cache_path as _tile_cache_path,
)

_IMAGE_CACHE_HEADERS = {"Cache-Control": "public, max-age=86400, immutable"}


def _image_route_indexed_file(
    file_id: str, x_hv_drive: str | None,
) -> dict[str, Any]:
    """The file an ``<img>``-loaded route serves, with the drive header optional.

    Without the header the host's ``file_access`` pre_check has already
    authorised the caller for this file; with it, the strict current-drive
    match runs as for every other route.
    """
    drive = unquote(x_hv_drive) if x_hv_drive else None
    if drive is not None:
        return _get_indexed_file_or_404(file_id, drive)

    from app.database import get_search_db_read
    from app.models import IndexedFile

    with get_search_db_read() as db:
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
    return {
        "file_id": row.file_id,
        "drive": row.drive,
        "filename": row.filename,
        "file_path": row.file_path,
        "file_type": row.file_type,
        "mime_type": row.mime_type,
    }


def _extract_loft_frame(
    indexed: dict[str, Any], cache_path: Any, t: float, size: str,
) -> None:
    """Cache the frame of a YouTube ``.loft``; any other ``.loft`` has no source."""
    from app.youtube_source import (
        YOUTUBE_PROVIDER, LoftNotAccessible, LoftUnreadable, ensure_youtube_frame,
        read_loft,
    )

    file_id = indexed["file_id"]
    try:
        source = read_loft(indexed["file_path"], indexed["drive"])
    except LoftNotAccessible:
        raise HTTPException(status_code=404, detail="Video file not accessible")
    except LoftUnreadable:
        source = None
    if source is None or source.provider != YOUTUBE_PROVIDER:
        logger.warning("youtube frame failed: file=%s class=not_youtube", file_id)
        raise HTTPException(status_code=400, detail="No frame source")
    ensure_youtube_frame(file_id, source.url, cache_path, t, size)


@router.get("/files/{file_id}/frame")
def get_frame(
    file_id: str,
    t: float = Query(..., ge=0, description="Timestamp in seconds"),
    size: Literal["small", "large"] = Query(default="small"),
    x_hv_drive: Annotated[str | None, Header(alias="X-Lit-Drive")] = None,
) -> Response:
    """Serve a single cached video frame, extracting via ffmpeg on miss.

    Drive context is optional here because the route is loaded via
    ``<img src>`` which can't carry custom headers. The host's
    ``file_access`` pre_check still verifies the caller can read the
    file's drive; when the header *is* present we additionally enforce
    the strict current-drive match (defence in depth for non-image
    consumers).

    Cache lives under ``intelligence_data_dir/frames/{file_id}/{ms}.webp``
    and is reused across requests / users / browser sessions. Cache
    misses spawn ffmpeg; hits short-circuit to a static FileResponse.
    """
    from app.config import resolve_file_path, resolve_stored_path, validate_file_path
    from app.workers.whisper import LOFT_MIME

    indexed = _image_route_indexed_file(file_id, x_hv_drive)

    if indexed["file_type"] != "video":
        raise HTTPException(status_code=400, detail="Not a video file")

    size_args = ("large",) if size == "large" else ()
    cache_path = _frame_cache_path(file_id, t, *size_args)
    cache_headers = _IMAGE_CACHE_HEADERS

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return FileResponse(
            path=str(cache_path),
            media_type="image/webp",
            headers=cache_headers,
        )

    if indexed["mime_type"] == LOFT_MIME:
        _extract_loft_frame(indexed, cache_path, t, size)
        return FileResponse(
            path=str(cache_path),
            media_type="image/webp",
            headers=cache_headers,
        )

    abs_path = resolve_stored_path(
        resolve_file_path(indexed["drive"], indexed["file_path"])
    )
    if not abs_path or not validate_file_path(abs_path):
        raise HTTPException(status_code=404, detail="Video file not accessible")

    _extract_frame_to_cache(abs_path, cache_path, t, *size_args)

    return FileResponse(
        path=str(cache_path),
        media_type="image/webp",
        headers=cache_headers,
    )


@router.get("/files/{file_id}/scene-tile")
def get_scene_tile(
    file_id: str,
    t: float = Query(..., ge=0, description="Timestamp in seconds"),
    x_hv_drive: Annotated[str | None, Header(alias="X-Lit-Drive")] = None,
) -> Response:
    """Serve the storyboard tile stored for a file's scene at ``t``.

    Reads one file under ``frames/{file_id}/tiles/``: no YouTube lookup, no
    ffmpeg, and nothing is written. The drive header is optional for the same
    reason as the frame route.
    """
    from app.workers.whisper import LOFT_MIME

    indexed = _image_route_indexed_file(file_id, x_hv_drive)

    if indexed["file_type"] != "video":
        raise HTTPException(status_code=400, detail="Not a video file")

    # Only a `.loft` reference has storyboard tiles; a native video's pictures
    # come from the frame route and must not be served here.
    if indexed["mime_type"] != LOFT_MIME:
        raise HTTPException(status_code=404, detail="Scene tile not found")

    tile_path = _tile_cache_path(file_id, t)
    if not tile_path.exists() or tile_path.stat().st_size == 0:
        raise HTTPException(status_code=404, detail="Scene tile not found")

    return FileResponse(
        path=str(tile_path),
        media_type="image/webp",
        headers=_IMAGE_CACHE_HEADERS,
    )


# --- Suggested tags endpoints ---


@router.get("/files/{file_id}/suggested-tags", response_model=SuggestedTagsResponse)
def get_suggested_tags(
    file_id: str,
    drive: str = Depends(require_drive),
) -> SuggestedTagsResponse:
    """Get suggested tags for a file."""
    import json as json_mod
    from app.database import get_search_db_read

    if settings.features.auto_tags == "false":
        return SuggestedTagsResponse(available=False)

    # Confirm the target file lives in the request drive before reading
    # its suggestions; otherwise a malicious caller could probe other
    # drives' file_ids.
    _get_indexed_file_or_404(file_id, drive)

    with get_search_db_read() as session:
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
def dismiss_suggested_tags(
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

    await asyncio.to_thread(_get_indexed_file_or_404, file_id, drive)
    auto_tags_worker = get_auto_tags_worker()

    def _delete_existing() -> None:
        with get_search_db() as session:
            session.execute(
                sql_text("DELETE FROM suggested_tags WHERE file_id = :fid"),
                {"fid": file_id},
            )

    await asyncio.to_thread(_delete_existing)

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

    def _read_eligibility() -> tuple[set[str], set[str]]:
        existing: set[str] = set()
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
        return in_drive, existing

    in_drive: set[str] = set()
    existing: set[str] = set()
    if body.file_ids:
        in_drive, existing = await asyncio.to_thread(_read_eligibility)

    queued = 0
    skipped = 0
    for file_id in body.file_ids:
        if file_id not in in_drive or file_id in existing:
            skipped += 1
        else:
            await auto_tags_worker.enqueue(file_id)
            queued += 1

    return BatchSuggestedTagsResponse(queued=queued, skipped=skipped)
