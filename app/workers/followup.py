"""The one background worker that finishes an extraction change.

SPEC-ADDON-070/067/068 queue work here instead of doing it inside the
request: an utterance's embedding, keyword vector and realignment, a
picture's embeddings, a restore's re-embedding, and the rebuild of a
corrected file's search rows. Items are stored in ``followup_items`` so
a restart resumes them; one item is taken at a time across all files.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text as sql_text

from app import database, extraction
from app.addon_events import post_addon_event_sync
from app.models import (
    Embedding,
    FollowupItem,
    IndexedFile,
    PictureText,
    TranscriptChunk,
    TranscriptWord,
)
from app.workers import aligner

logger = logging.getLogger(__name__)

RETRY_SECONDS = 600
MAX_ALIGN_ATTEMPTS = 3
_IDLE_SECONDS = 10


class FollowupWorker:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="followup_worker")
            logger.info("Follow-up worker started")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def wake(self) -> None:
        self._wake.set()

    async def _run(self) -> None:
        try:
            await asyncio.to_thread(resume_items)
            await asyncio.to_thread(startup_sweep)
        except Exception:
            logger.exception("follow-up startup failed; continuing")

        while True:
            due = await asyncio.to_thread(next_due_item_id)
            if due is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=_IDLE_SECONDS)
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                with aligner.job_scope():
                    await asyncio.to_thread(process_item, due)
            except Exception:
                logger.exception("follow-up item %s failed", due)


def resume_items() -> None:
    """Retry every stored item at every start."""
    with database.get_search_db() as session:
        session.query(FollowupItem).update(
            {FollowupItem.last_attempt_at: None},
            synchronize_session=False,
        )


def startup_sweep() -> None:
    """Queue the rebuilds the start itself owes."""
    rebuilt = database.text_embedding_model_rebuilt
    with database.get_search_db() as session:
        rows = session.query(IndexedFile).filter(
            IndexedFile.active.is_(True)
        ).all()
        for row in rows:
            if (
                row.transcript_corrected and not row.whisper_indexed
            ):
                extraction.queue_item(
                    session,
                    row.file_id,
                    extraction.KIND_TRANSCRIPT,
                    "rebuild",
                )
            if row.picture_text_corrected and not row.ocr_indexed:
                extraction.queue_item(
                    session,
                    row.file_id,
                    extraction.KIND_PICTURE,
                    "rebuild",
                )
            if (
                rebuilt
                and not row.picture_text_corrected
                and not row.ocr_indexed
                and extraction.get_picture_text(session, row.file_id)
            ):
                extraction.queue_item(
                    session,
                    row.file_id,
                    extraction.KIND_PICTURE,
                    "rebuild",
                )
    if rebuilt:
        database.text_embedding_model_rebuilt = False


def next_due_item_id() -> int | None:
    with database.get_search_db() as session:
        cutoff = datetime.now(UTC) - timedelta(seconds=RETRY_SECONDS)
        row = (
            session.query(FollowupItem.id)
            .filter(
                (FollowupItem.last_attempt_at.is_(None))
                | (FollowupItem.last_attempt_at <= cutoff)
            )
            .order_by(FollowupItem.id)
            .first()
        )
        return int(row[0]) if row else None


def _drop(item_id: int) -> None:
    with database.get_search_db() as session:
        session.query(FollowupItem).filter_by(id=item_id).delete()


def _park(item_id: int) -> None:
    """Leave an item for a trashed/missing file without blocking the queue."""
    with database.get_search_db() as session:
        item = session.get(FollowupItem, item_id)
        if item is not None:
            item.last_attempt_at = datetime.now(UTC)


def _stale(item: FollowupItem, file: IndexedFile, session: Any) -> bool:
    if item.generation != extraction.generation(
        session, item.file_id, item.kind
    ):
        return True
    if item.item_type == "utterance":
        chunk = (
            session.query(TranscriptChunk)
            .filter_by(file_id=item.file_id, chunk_index=item.chunk_index)
            .first()
        )
        if chunk is None or chunk.text != item.expected_text:
            return True
    elif item.item_type == "picture":
        if extraction.get_picture_text(session, item.file_id) != item.expected_text:
            return True
    return False


def process_item(item_id: int) -> None:
    with database.get_search_db_read() as session:
        item = session.get(FollowupItem, item_id)
        if item is None:
            return
        file_row = extraction.get_file(session, item.file_id)
        if file_row is None:
            _drop(item_id)
            return
        if not file_row.active:
            _park(item_id)
            return
        if _stale(item, file_row, session):
            _drop(item_id)
            return
        data = {
            "id": item.id,
            "file_id": item.file_id,
            "kind": item.kind,
            "item_type": item.item_type,
            "chunk_index": item.chunk_index,
            "expected_text": item.expected_text,
            "attempts": item.attempts,
            "drive": file_row.drive,
            "file_path": file_row.file_path,
            "mime_type": file_row.mime_type,
        }

    try:
        if data["item_type"] == "utterance":
            _process_utterance(data)
        elif data["item_type"] == "keyword":
            _recompute_keywords(data["file_id"])
        elif data["item_type"] == "picture":
            _process_picture(data)
        elif data["item_type"] == "restore":
            _process_restore(data)
        elif data["item_type"] == "rebuild":
            _process_rebuild(data)
    except Exception:
        logger.exception("follow-up work failed for %s", data["file_id"])
        _record_failure(data)
        return
    _drop(item_id)


def _record_failure(data: dict[str, Any]) -> None:
    with database.get_search_db() as session:
        item = session.get(FollowupItem, data["id"])
        if item is not None:
            item.attempts = int(item.attempts or 0) + 1
            item.last_attempt_at = datetime.now(UTC)


def _mark_done_after_failure(item_id: int) -> None:
    _drop(item_id)


def _process_utterance(data: dict[str, Any]) -> None:
    from app.workers import whisper

    text = data["expected_text"]
    chunk_start: float | None = None
    chunk_end: float | None = None
    language = ""
    with database.get_search_db_read() as session:
        chunk = (
            session.query(TranscriptChunk)
            .filter_by(file_id=data["file_id"], chunk_index=data["chunk_index"])
            .first()
        )
        if chunk is not None:
            text = chunk.text
            chunk_start = chunk.timestamp_start
            chunk_end = chunk.timestamp_end
            language = chunk.language
        chunk_texts = [
            row[0]
            for row in session.query(TranscriptChunk.text)
            .filter_by(file_id=data["file_id"])
            .order_by(TranscriptChunk.chunk_index)
            .all()
            if (row[0] or "").strip()
        ]

    if chunk_start is not None and text is not None:
        _embed_utterance(data["file_id"], data["chunk_index"], text, chunk_start, chunk_end)
    # A realignment that will be retried raises below; the keyword vector
    # does not depend on the alignment and must not wait for it. The event
    # does depend on the realignment having ended, so it stays after it.
    _recompute_keywords(data["file_id"], chunk_texts)
    if chunk_start is not None and text is not None:
        _realign_utterance(data, text, chunk_start, chunk_end, language)
    post_addon_event_sync(
        "intelligence.transcript.updated",
        {
            "file_id": data["file_id"],
            "drive": data["drive"],
            "reason": "corrected",
        },
    )


def _embed_utterance(
    file_id: str,
    chunk_index: int,
    text: str,
    start: float,
    end: float,
) -> None:
    if not text.strip():
        return
    from app.workers.embedder import embed_passages

    vector = embed_passages([text])[0]
    with database.get_search_db() as session:
        chunk = (
            session.query(TranscriptChunk)
            .filter_by(file_id=file_id, chunk_index=chunk_index)
            .first()
        )
        if chunk is None or chunk.text != text:
            return
        existing = (
            session.query(Embedding)
            .filter(
                Embedding.file_id == file_id,
                Embedding.embedding_type == "whisper",
                Embedding.timestamp_start == start,
                Embedding.timestamp_end == end,
            )
            .all()
        )
        for emb in existing:
            session.execute(
                sql_text("DELETE FROM vec_text WHERE embedding_id = :id"),
                {"id": emb.id},
            )
            session.delete(emb)
        from app.workers.metadata import _store_embedding

        _store_embedding(
            session=session,
            embedding_id=f"wh_{file_id}_{chunk_index}_{uuid.uuid4().hex[:8]}",
            file_id=file_id,
            embedding_type="whisper",
            vector_table="vec_text",
            vector=vector,
            content_preview=text[:200],
            timestamp_start=start,
            timestamp_end=end,
            chunk_index=chunk_index,
        )


def _realign_utterance(
    data: dict[str, Any],
    text: str,
    start: float,
    end: float,
    language: str,
) -> None:
    if (data["mime_type"] or "").endswith("loft+json"):
        _clear_pending(data["file_id"], data["chunk_index"])
        return
    outcome, units = aligner.align_span(
        data["file_path"], start, end, text, language
    )
    if outcome in ("load_failed", "unreachable"):
        retry = False
        with database.get_search_db() as session:
            chunk = (
                session.query(TranscriptChunk)
                .filter_by(file_id=data["file_id"], chunk_index=data["chunk_index"])
                .first()
            )
            if chunk is None or chunk.text != text:
                return
            chunk.words_pending_attempts = int(
                chunk.words_pending_attempts or 0
            ) + 1
            if chunk.words_pending_attempts < MAX_ALIGN_ATTEMPTS:
                # Persist the attempt before the retry: the raise below
                # must not roll the counter back.
                retry = True
            else:
                chunk.words_pending = False
                chunk.words_pending_attempts = 0
        if retry:
            raise RuntimeError(outcome)
        return
    if outcome != "aligned" or not units:
        _clear_pending(data["file_id"], data["chunk_index"])
        return
    with database.get_search_db() as session:
        chunk = (
            session.query(TranscriptChunk)
            .filter_by(file_id=data["file_id"], chunk_index=data["chunk_index"])
            .first()
        )
        if chunk is None or chunk.text != text:
            return
        _delete_own_words(session, data["file_id"], chunk)
        for unit in units:
            session.add(TranscriptWord(
                file_id=data["file_id"],
                text=unit["text"],
                language=language,
                timestamp_start=unit["timestamp_start"],
                timestamp_end=unit["timestamp_end"],
                speaker_id=chunk.speaker_id,
            ))
        chunk.words_pending = False
        chunk.words_pending_attempts = 0


def _delete_own_words(session: Any, file_id: str, chunk: TranscriptChunk) -> None:
    """Delete the utterance's words, keeping the neighbour's boundary word."""
    earlier = (
        session.query(TranscriptChunk.timestamp_end)
        .filter(
            TranscriptChunk.file_id == file_id,
            TranscriptChunk.timestamp_start < chunk.timestamp_start,
        )
        .all()
    )
    prev_end = max((row[0] for row in earlier), default=None)
    words = (
        session.query(TranscriptWord)
        .filter(
            TranscriptWord.file_id == file_id,
            TranscriptWord.timestamp_start >= chunk.timestamp_start,
            TranscriptWord.timestamp_end <= chunk.timestamp_end,
        )
        .all()
    )
    for word in words:
        if (
            word.timestamp_start == word.timestamp_end == chunk.timestamp_start
            and prev_end == chunk.timestamp_start
        ):
            continue
        session.delete(word)


def _clear_pending(file_id: str, chunk_index: int) -> None:
    with database.get_search_db() as session:
        chunk = (
            session.query(TranscriptChunk)
            .filter_by(file_id=file_id, chunk_index=chunk_index)
            .first()
        )
        if chunk is not None:
            chunk.words_pending = False
            chunk.words_pending_attempts = 0
            _delete_own_words(session, file_id, chunk)


def _recompute_keywords(file_id: str, texts: list[str] | None = None) -> None:
    from app.workers import whisper

    if texts is None:
        with database.get_search_db_read() as session:
            texts = [
                row[0]
                for row in session.query(TranscriptChunk.text)
                .filter_by(file_id=file_id)
                .order_by(TranscriptChunk.chunk_index)
                .all()
                if (row[0] or "").strip()
            ]
    try:
        whisper._index_tfidf_keywords(file_id, texts)
    except Exception:
        logger.exception("follow-up keyword rebuild failed for %s", file_id)


def _process_picture(data: dict[str, Any]) -> None:
    from app.extractors.base import ContentExtractor
    from app.workers.embedder import embed_passages
    from app.workers.metadata import _remove_embeddings, _store_embedding
    from app.database import upsert_fts_text_content

    with database.get_search_db_read() as session:
        text = extraction.get_picture_text(session, data["file_id"]) or ""
    chunks = ContentExtractor.chunk_text(text)
    vectors = list(embed_passages(chunks)) if chunks else []
    with database.get_search_db() as session:
        if extraction.get_picture_text(session, data["file_id"]) != text:
            return
        _remove_embeddings(session, data["file_id"], "text_content")
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            _store_embedding(
                session=session,
                embedding_id=f"ocr_{data['file_id']}_{index}_{uuid.uuid4().hex[:8]}",
                file_id=data["file_id"],
                embedding_type="text_content",
                vector_table="vec_text",
                vector=vector,
                content_preview=chunk[:200],
                page=None,
                chunk_index=index,
            )
        upsert_fts_text_content(
            session,
            data["file_id"],
            [
                {"chunk_index": index, "page": None, "text": chunk}
                for index, chunk in enumerate(chunks)
            ],
        )
    post_addon_event_sync(
        "intelligence.picture_text.updated",
        {
            "file_id": data["file_id"],
            "drive": data["drive"],
            "reason": "corrected",
        },
    )


def _process_restore(data: dict[str, Any]) -> None:
    if data["kind"] == extraction.KIND_TRANSCRIPT:
        from app.workers import whisper

        with database.get_search_db_read() as session:
            texts = [
                row[0]
                for row in session.query(TranscriptChunk.text)
                .filter_by(file_id=data["file_id"])
                .order_by(TranscriptChunk.chunk_index)
                .all()
                if (row[0] or "").strip()
            ]
        _reembed_transcript(data["file_id"])
        whisper._index_tfidf_keywords(data["file_id"], texts)
    else:
        _reembed_picture(data["file_id"])


def _process_rebuild(data: dict[str, Any]) -> None:
    if data["kind"] == extraction.KIND_TRANSCRIPT:
        _rebuild_transcript(data)
    else:
        _rebuild_picture(data)


def _rebuild_transcript(data: dict[str, Any]) -> None:
    from app.database import upsert_fts_transcripts
    from app.workers import whisper

    file_id = data["file_id"]
    with database.get_search_db_read() as session:
        rows = (
            session.query(TranscriptChunk)
            .filter_by(file_id=file_id)
            .order_by(TranscriptChunk.chunk_index)
            .all()
        )
        existing = {
            (row.timestamp_start, row.timestamp_end)
            for row in session.query(Embedding)
            .filter_by(file_id=file_id, embedding_type="whisper")
            .all()
        }
        has_keywords = (
            session.query(Embedding.id)
            .filter_by(file_id=file_id, embedding_type="tfidf_keywords")
            .first()
            is not None
        )
        snapshots = [
            {
                "chunk_index": row.chunk_index,
                "text": row.text,
                "start": row.timestamp_start,
                "end": row.timestamp_end,
            }
            for row in rows
            if (row.text or "").strip()
        ]
    _embed_missing(file_id, snapshots, existing)
    with database.get_search_db() as session:
        upsert_fts_transcripts(
            session,
            file_id,
            [
                {"chunk_index": snap["chunk_index"], "text": snap["text"]}
                for snap in snapshots
            ],
        )
        row = extraction.get_file(session, file_id)
        if row is not None:
            row.whisper_indexed = True
    if not has_keywords:
        whisper._index_tfidf_keywords(
            file_id, [snap["text"] for snap in snapshots]
        )
    post_addon_event_sync(
        "intelligence.transcript.updated",
        {"file_id": file_id, "drive": data["drive"], "reason": "rebuilt"},
    )
    if data["drive"]:
        from app.processing_events import emit_processing_updated

        emit_processing_updated(data["drive"], [file_id])


def _rebuild_picture(data: dict[str, Any]) -> None:
    from app.database import upsert_fts_text_content
    from app.extractors.base import ContentExtractor

    file_id = data["file_id"]
    with database.get_search_db_read() as session:
        text = extraction.get_picture_text(session, file_id) or ""
        has_rows = (
            session.query(Embedding.id)
            .filter_by(file_id=file_id, embedding_type="text_content")
            .first()
            is not None
        )
    chunks = ContentExtractor.chunk_text(text)
    if not has_rows and chunks:
        _write_picture_embeddings(file_id, chunks)
    with database.get_search_db() as session:
        upsert_fts_text_content(
            session,
            file_id,
            [
                {"chunk_index": index, "page": None, "text": chunk}
                for index, chunk in enumerate(chunks)
            ],
        )
        row = extraction.get_file(session, file_id)
        if row is not None:
            row.ocr_indexed = True
    post_addon_event_sync(
        "intelligence.picture_text.updated",
        {"file_id": file_id, "drive": data["drive"], "reason": "rebuilt"},
    )
    if data["drive"]:
        from app.processing_events import emit_processing_updated

        emit_processing_updated(data["drive"], [file_id])


def _embed_missing(
    file_id: str, snapshots: list[dict[str, Any]], existing: set[tuple]
) -> None:
    from app.workers.embedder import embed_passages
    from app.workers.metadata import _store_embedding

    missing = [
        snap
        for snap in snapshots
        if (snap["start"], snap["end"]) not in existing
    ]
    if not missing:
        return
    vectors = embed_passages([snap["text"] for snap in missing])
    with database.get_search_db() as session:
        for snap, vector in zip(missing, vectors):
            chunk = (
                session.query(TranscriptChunk)
                .filter_by(file_id=file_id, chunk_index=snap["chunk_index"])
                .first()
            )
            if chunk is None or chunk.text != snap["text"]:
                continue
            _store_embedding(
                session=session,
                embedding_id=(
                    f"wh_{file_id}_{snap['chunk_index']}_"
                    f"{uuid.uuid4().hex[:8]}"
                ),
                file_id=file_id,
                embedding_type="whisper",
                vector_table="vec_text",
                vector=vector,
                content_preview=snap["text"][:200],
                timestamp_start=snap["start"],
                timestamp_end=snap["end"],
                chunk_index=snap["chunk_index"],
            )


def _reembed_transcript(file_id: str) -> None:
    with database.get_search_db_read() as session:
        snapshots = [
            {
                "chunk_index": row.chunk_index,
                "text": row.text,
                "start": row.timestamp_start,
                "end": row.timestamp_end,
            }
            for row in session.query(TranscriptChunk)
            .filter_by(file_id=file_id)
            .all()
            if (row.text or "").strip()
        ]
    _embed_missing(file_id, snapshots, set())


def _reembed_picture(file_id: str) -> None:
    from app.extractors.base import ContentExtractor

    with database.get_search_db_read() as session:
        text = extraction.get_picture_text(session, file_id) or ""
    chunks = ContentExtractor.chunk_text(text)
    if chunks:
        _write_picture_embeddings(file_id, chunks)


def _write_picture_embeddings(file_id: str, chunks: list[str]) -> None:
    from app.workers.embedder import embed_passages
    from app.workers.metadata import _remove_embeddings, _store_embedding

    vectors = list(embed_passages(chunks))
    with database.get_search_db() as session:
        _remove_embeddings(session, file_id, "text_content")
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            _store_embedding(
                session=session,
                embedding_id=f"ocr_{file_id}_{index}_{uuid.uuid4().hex[:8]}",
                file_id=file_id,
                embedding_type="text_content",
                vector_table="vec_text",
                vector=vector,
                content_preview=chunk[:200],
                page=None,
                chunk_index=index,
            )


__all__ = [
    "FollowupWorker",
    "next_due_item_id",
    "process_item",
    "resume_items",
    "startup_sweep",
]
