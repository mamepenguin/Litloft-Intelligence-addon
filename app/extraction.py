"""State shared by the extraction-correction routes and workers.

SPEC-ADDON-065/067/068 keep one machine-produced original per file and
kind, protect a corrected file from automatic re-processing, and queue
the embedding and alignment a correction still needs. The rules about
those marks live here so a route, a worker and the follow-up queue read
the same definition of "corrected", "busy" and "stale".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable

from sqlalchemy import text as sql_text

from app import database
from app.idf_backfill import mark_idf_terms_stale
from app.models import (
    FollowupItem,
    IndexedFile,
    KeptOriginal,
    PictureText,
    TranscriptChunk,
    TranscriptWord,
)

logger = logging.getLogger(__name__)

KIND_TRANSCRIPT = "transcript"
KIND_PICTURE = "picture_text"
KINDS = (KIND_TRANSCRIPT, KIND_PICTURE)

_FLAG_COLUMN = {
    KIND_TRANSCRIPT: "transcript_corrected",
    KIND_PICTURE: "picture_text_corrected",
}
_REDO_COLUMN = {
    KIND_TRANSCRIPT: "transcript_redo",
    KIND_PICTURE: "picture_text_redo",
}
_ENDED_COLUMN = {
    KIND_TRANSCRIPT: "transcript_redo_ended",
    KIND_PICTURE: "picture_text_redo_ended",
}
_GENERATION_COLUMN = {
    KIND_TRANSCRIPT: "transcript_generation",
    KIND_PICTURE: "picture_text_generation",
}

#: A restore is in progress in this process; a Refine job's claim lives
#: in ``workers.refine._active_refine_files``, which ``start_refine_job``
#: fills before it returns.
_restore_in_progress: set[str] = set()


def get_file(session: Any, file_id: str) -> IndexedFile | None:
    return session.query(IndexedFile).filter_by(file_id=file_id).first()


def is_corrected(session: Any, file_id: str, kind: str) -> bool:
    row = get_file(session, file_id)
    return bool(
        isinstance(row, IndexedFile)
        and getattr(row, _FLAG_COLUMN[kind]) is True
    )


def mark_corrected(session: Any, file_id: str, kind: str) -> None:
    row = get_file(session, file_id)
    if row is not None:
        setattr(row, _FLAG_COLUMN[kind], True)
        setattr(row, _ENDED_COLUMN[kind], False)


def set_redo(session: Any, file_id: str, kind: str) -> None:
    row = get_file(session, file_id)
    if row is not None:
        setattr(row, _REDO_COLUMN[kind], True)


def clear_redo(session: Any, file_id: str, kind: str) -> None:
    row = get_file(session, file_id)
    if row is not None:
        setattr(row, _REDO_COLUMN[kind], False)


def mark_redo_ended(session: Any, file_id: str, kind: str) -> None:
    row = get_file(session, file_id)
    if row is not None:
        setattr(row, _ENDED_COLUMN[kind], True)


def generation(session: Any, file_id: str, kind: str) -> int:
    row = get_file(session, file_id)
    return int(getattr(row, _GENERATION_COLUMN[kind]) or 0) if row else 0


def bump_generation(session: Any, file_id: str, kind: str) -> int:
    row = get_file(session, file_id)
    if row is None:
        return 0
    value = int(getattr(row, _GENERATION_COLUMN[kind]) or 0) + 1
    setattr(row, _GENERATION_COLUMN[kind], value)
    return value


def clear_corrections(session: Any, file_id: str, kind: str) -> None:
    """The state a stored machine result leaves: no marks, no original."""
    row = get_file(session, file_id)
    if row is not None:
        setattr(row, _FLAG_COLUMN[kind], False)
        setattr(row, _REDO_COLUMN[kind], False)
        setattr(row, _ENDED_COLUMN[kind], False)
    delete_original(session, file_id, kind)


def has_original(session: Any, file_id: str, kind: str) -> bool:
    return (
        session.query(KeptOriginal)
        .filter_by(file_id=file_id, kind=kind)
        .first()
        is not None
    )


def get_original(session: Any, file_id: str, kind: str) -> KeptOriginal | None:
    return (
        session.query(KeptOriginal)
        .filter_by(file_id=file_id, kind=kind)
        .first()
    )


def delete_original(session: Any, file_id: str, kind: str) -> None:
    session.query(KeptOriginal).filter_by(
        file_id=file_id, kind=kind
    ).delete()


def _json_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _from_json_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def snapshot_transcript(session: Any, file_id: str) -> tuple[str, str]:
    chunks = (
        session.query(TranscriptChunk)
        .filter_by(file_id=file_id)
        .order_by(TranscriptChunk.chunk_index)
        .all()
    )
    words = (
        session.query(TranscriptWord)
        .filter_by(file_id=file_id)
        .order_by(TranscriptWord.timestamp_start, TranscriptWord.id)
        .all()
    )
    return (
        json.dumps([
            {
                "chunk_index": c.chunk_index,
                "text": c.text,
                "language": c.language,
                "timestamp_start": c.timestamp_start,
                "timestamp_end": c.timestamp_end,
                "speaker_id": c.speaker_id,
                "text_refined_at": _json_datetime(c.text_refined_at),
                "refined_model": c.refined_model,
            }
            for c in chunks
        ]),
        json.dumps([
            {
                "text": w.text,
                "language": w.language,
                "timestamp_start": w.timestamp_start,
                "timestamp_end": w.timestamp_end,
                "speaker_id": w.speaker_id,
            }
            for w in words
        ]),
    )


def store_original_if_absent(
    session: Any, file_id: str, kind: str, trigger: str
) -> bool:
    """Store the state as it is now, once per file and kind."""
    if has_original(session, file_id, kind):
        return False
    row = KeptOriginal(file_id=file_id, kind=kind, trigger=trigger)
    if kind == KIND_TRANSCRIPT:
        row.chunks_json, row.words_json = snapshot_transcript(session, file_id)
    else:
        row.text = get_picture_text(session, file_id) or ""
    session.add(row)
    logger.info(
        "Stored the original %s for %s (trigger=%s)", kind, file_id, trigger
    )
    return True


def transcript_snapshot(original: KeptOriginal) -> tuple[list[dict], list[dict]]:
    chunks = json.loads(original.chunks_json or "[]")
    words = json.loads(original.words_json or "[]")
    return chunks, words


@dataclass(frozen=True)
class PendingItem:
    id: int
    file_id: str
    kind: str
    item_type: str
    chunk_index: int | None
    generation: int
    expected_text: str | None
    attempts: int


def queue_item(
    session: Any,
    file_id: str,
    kind: str,
    item_type: str,
    *,
    chunk_index: int | None = None,
    expected_text: str | None = None,
) -> FollowupItem | None:
    """Add one pending item, without duplicating the singleton kinds."""
    if item_type == "rebuild":
        existing = (
            session.query(FollowupItem)
            .filter_by(file_id=file_id, kind=kind, item_type="rebuild")
            .first()
        )
        if existing is not None:
            return None
    elif item_type == "keyword":
        existing = (
            session.query(FollowupItem)
            .filter_by(file_id=file_id, item_type="keyword")
            .first()
        )
        if existing is not None:
            return None
    item = FollowupItem(
        file_id=file_id,
        kind=kind,
        item_type=item_type,
        chunk_index=chunk_index,
        generation=generation(session, file_id, kind),
        expected_text=expected_text,
    )
    session.add(item)
    return item


def pending_items(session: Any, file_id: str) -> list[FollowupItem]:
    return (
        session.query(FollowupItem)
        .filter_by(file_id=file_id)
        .order_by(FollowupItem.id)
        .all()
    )


def has_pending(session: Any, file_id: str) -> bool:
    return isinstance(
        session.query(FollowupItem).filter_by(file_id=file_id).first(),
        FollowupItem,
    )


def drop_items(session: Any, file_id: str, kind: str | None = None) -> None:
    query = session.query(FollowupItem).filter_by(file_id=file_id)
    if kind is not None:
        query = query.filter_by(kind=kind)
    query.delete()


def get_picture_text(session: Any, file_id: str) -> str | None:
    row = session.query(PictureText).filter_by(file_id=file_id).first()
    return row.text if row is not None else None


def set_picture_text(session: Any, file_id: str, text: str) -> None:
    row = session.query(PictureText).filter_by(file_id=file_id).first()
    if row is None:
        session.add(PictureText(file_id=file_id, text=text))
    else:
        row.text = text
        row.updated_at = datetime.now(UTC)


# --- Busy -------------------------------------------------------------------


def is_refine_claimed(file_id: str) -> bool:
    try:
        from app.workers.refine import _active_refine_files

        return file_id in _active_refine_files
    except Exception:
        return False


def begin_restore(file_id: str) -> None:
    _restore_in_progress.add(file_id)


def end_restore(file_id: str) -> None:
    _restore_in_progress.discard(file_id)


def busy(file_id: str) -> bool:
    """A Refine claim, a confirmed redo mark, or a restore in progress."""
    if file_id in _restore_in_progress or is_refine_claimed(file_id):
        return True
    with database.get_search_db_read() as session:
        row = get_file(session, file_id)
        if not isinstance(row, IndexedFile):
            return False
        return bool(
            row.transcript_redo is True or row.picture_text_redo is True
        )


def busy_in_session(session: Any, file_id: str) -> bool:
    """``busy`` for a caller that already holds a session."""
    if file_id in _restore_in_progress or is_refine_claimed(file_id):
        return True
    row = get_file(session, file_id)
    if not isinstance(row, IndexedFile):
        return False
    return bool(
        row.transcript_redo is True or row.picture_text_redo is True
    )


def clear_redo_marks_on_start() -> int:
    """Every confirmed run left by the previous process ended there.

    The correction stays; the file becomes protected again and a rebuild
    supplies the search rows the interrupted run would have written.
    """
    cleared = 0
    with database.get_search_db() as session:
        rows = (
            session.query(IndexedFile)
            .filter(
                (IndexedFile.transcript_redo.is_(True))
                | (IndexedFile.picture_text_redo.is_(True))
            )
            .all()
        )
        for row in rows:
            for kind in KINDS:
                if getattr(row, _REDO_COLUMN[kind]):
                    clear_redo(session, row.file_id, kind)
                    mark_redo_ended(session, row.file_id, kind)
                    queue_item(session, row.file_id, kind, "rebuild")
                    cleared += 1
    if cleared:
        logger.info("Cleared %d confirmed redo mark(s) at startup", cleared)
    return cleared


# --- Restore ----------------------------------------------------------------


def restore_transcript(session: Any, file_id: str) -> bool:
    original = get_original(session, file_id, KIND_TRANSCRIPT)
    if original is None:
        return False
    chunks, words = transcript_snapshot(original)
    session.query(TranscriptChunk).filter_by(file_id=file_id).delete()
    session.query(TranscriptWord).filter_by(file_id=file_id).delete()
    session.flush()
    for chunk in chunks:
        session.add(TranscriptChunk(
            file_id=file_id,
            chunk_index=chunk["chunk_index"],
            text=chunk["text"],
            language=chunk["language"],
            timestamp_start=chunk["timestamp_start"],
            timestamp_end=chunk["timestamp_end"],
            speaker_id=chunk.get("speaker_id"),
            text_refined_at=_from_json_datetime(chunk.get("text_refined_at")),
            refined_model=chunk.get("refined_model"),
            corrected_at=None,
            words_pending=False,
            words_pending_attempts=0,
        ))
    for word in words:
        session.add(TranscriptWord(
            file_id=file_id,
            text=word["text"],
            language=word["language"],
            timestamp_start=word["timestamp_start"],
            timestamp_end=word["timestamp_end"],
            speaker_id=word.get("speaker_id"),
        ))
    session.flush()

    from app.database import upsert_fts_transcripts

    upsert_fts_transcripts(
        session,
        file_id,
        [
            {"chunk_index": chunk["chunk_index"], "text": chunk["text"]}
            for chunk in chunks
            if (chunk["text"] or "").strip()
        ],
    )
    _remove_embeddings(session, file_id, ("whisper", "tfidf_keywords"))
    row = get_file(session, file_id)
    if row is not None:
        row.tfidf_keywords_indexed = False
    mark_idf_terms_stale(session, file_id)
    clear_corrections(session, file_id, KIND_TRANSCRIPT)
    bump_generation(session, file_id, KIND_TRANSCRIPT)
    drop_items(session, file_id, KIND_TRANSCRIPT)
    queue_item(session, file_id, KIND_TRANSCRIPT, "restore")
    return True


def restore_picture_text(session: Any, file_id: str) -> bool:
    original = get_original(session, file_id, KIND_PICTURE)
    if original is None:
        return False
    text = original.text or ""
    set_picture_text(session, file_id, text)
    _rewrite_picture_rows(session, file_id, text)
    row = get_file(session, file_id)
    if row is not None:
        row.tfidf_keywords_indexed = False
    mark_idf_terms_stale(session, file_id)
    clear_corrections(session, file_id, KIND_PICTURE)
    bump_generation(session, file_id, KIND_PICTURE)
    drop_items(session, file_id, KIND_PICTURE)
    queue_item(session, file_id, KIND_PICTURE, "restore")
    return True


def _remove_embeddings(
    session: Any, file_id: str, kinds: Iterable[str]
) -> None:
    from app.models import Embedding

    rows = (
        session.query(Embedding)
        .filter(
            Embedding.file_id == file_id,
            Embedding.embedding_type.in_(tuple(kinds)),
        )
        .all()
    )
    for emb in rows:
        session.execute(
            sql_text("DELETE FROM vec_text WHERE embedding_id = :id"),
            {"id": emb.id},
        )
        session.delete(emb)
    session.flush()


def _rewrite_picture_rows(session: Any, file_id: str, text: str) -> None:
    from app.database import upsert_fts_text_content
    from app.extractors.base import ContentExtractor

    chunks = ContentExtractor.chunk_text(text)
    _remove_embeddings(session, file_id, ("text_content",))
    upsert_fts_text_content(
        session,
        file_id,
        [
            {"chunk_index": index, "page": None, "text": chunk}
            for index, chunk in enumerate(chunks)
        ],
    )


__all__ = [
    "KIND_PICTURE",
    "KIND_TRANSCRIPT",
    "KINDS",
    "begin_restore",
    "bump_generation",
    "busy",
    "busy_in_session",
    "clear_corrections",
    "clear_redo",
    "clear_redo_marks_on_start",
    "delete_original",
    "drop_items",
    "end_restore",
    "generation",
    "get_file",
    "get_original",
    "get_picture_text",
    "has_original",
    "has_pending",
    "is_corrected",
    "mark_corrected",
    "mark_redo_ended",
    "pending_items",
    "queue_item",
    "restore_picture_text",
    "restore_transcript",
    "set_picture_text",
    "set_redo",
    "snapshot_transcript",
    "store_original_if_absent",
    "transcript_snapshot",
]
