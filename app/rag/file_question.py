"""The material a question about one file is answered from (SPEC-ADDON-053).

A file's text is its indexed chunks, read from ``search.db``; nothing is
extracted at question time, which is why a file becomes askable only once
indexing has run. Every chunk is sent when they fit
``rag.file_question_max_chars``. Otherwise the chunks are ranked by the
cosine similarity of their stored embedding to the ranking text, taken in
rank order while they fit, and sent in file order with ``partial`` set.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from sqlalchemy import text as sql_text

import app.config as config
from app import llm_routing
from app.config import settings
from app.database import get_search_db_read, get_search_engine
from app.document_sections import EPUB_MIME
from app.frame_cache import extract_frame_to_cache, frame_cache_path
from app.workers.embedder import embed_query
from app.workers.vision import _load_image_bytes, _preprocess_image
from app.workers.whisper import LOFT_MIME
from app.youtube_source import YOUTUBE_PROVIDER, ensure_youtube_frame, read_loft

logger = logging.getLogger(__name__)

PDF_MIME = "application/pdf"

#: Why an image could not be attached; the route logs the class, never
#: the question or the bytes.
IMAGE_FAILURE_PICTURE = "picture_unreadable"
IMAGE_FAILURE_FRAME = "frame_failed"
IMAGE_FAILURE_REJECTED = "image_rejected"

# vec0 takes its ids as one parameter each, so they go in batches.
_VECTOR_BATCH = 400

# ``[n]`` markers in the answer, up to a three-digit chunk number.
_CITATION_MARKER = re.compile(r"\[(\d{1,3})\]")


@dataclass(frozen=True)
class SelectedChunk:
    """One chunk as sent: its text, and where inside the file it sits."""

    text: str
    position: dict[str, Any] | None


@dataclass(frozen=True)
class Material:
    """The chunks a question is answered from, in file order."""

    chunks: tuple[SelectedChunk, ...]
    #: True when only some of the file's chunks fit the budget.
    partial: bool
    file_id: str
    file_type: str
    mime_type: str
    file_path: str | None
    duration: float | None


@dataclass(frozen=True)
class HistoryTurn:
    """One earlier turn as the model receives it, markers removed."""

    question: str
    answer: str


@dataclass(frozen=True)
class ImageMaterial:
    """One image as the provider receives it."""

    data: bytes
    mime_type: str


class ImageUnavailable(Exception):
    """An image that applies to the question could not be prepared."""

    def __init__(self, failure: str) -> None:
        super().__init__(failure)
        self.failure = failure


@dataclass(frozen=True)
class _Row:
    text: str
    position: dict[str, Any] | None
    embedding_id: str | None
    #: Position in the file, which is also the order the chunks are sent in.
    order: int


def _is_picture(file_type: str, mime_type: str) -> bool:
    return file_type == "image" or mime_type.startswith("image/")


def is_picture(material: Material) -> bool:
    return _is_picture(material.file_type, material.mime_type)


def available(file_id: str, drive: str) -> bool:
    """Whether this drive may ask about this file (SPEC-ADDON-053 I6).

    An active ``indexed_files`` row of the request drive with at least one
    text chunk. A picture with no text is askable when some profile can
    see (SPEC-ADDON-074); every other kind still needs a chunk. Anything
    else is ``False`` rather than an error, so the tab simply has no
    button.
    """
    with get_search_db_read() as session:
        indexed = session.execute(
            sql_text(
                "SELECT file_type, mime_type FROM indexed_files "
                "WHERE file_id = :fid AND drive = :drive AND active = 1"
            ),
            {"fid": file_id, "drive": drive},
        ).first()
        if indexed is None:
            return False
        chunks = session.execute(
            sql_text(
                "SELECT EXISTS(SELECT 1 FROM transcript_chunks "
                "WHERE file_id = :fid) "
                "OR EXISTS(SELECT 1 FROM fts_text_content "
                "WHERE file_id = :fid)"
            ),
            {"fid": file_id},
        ).scalar()
    if chunks:
        return True
    if not _is_picture(indexed[0] or "", indexed[1] or ""):
        return False
    return llm_routing.any_profile_can_see(settings)


def trim_history(
    turns: Sequence[Any], max_chars: int,
) -> list[HistoryTurn]:
    """The earlier turns the model gets, oldest first, within half the limit.

    ``[n]`` markers numbered an earlier request's passages, so they are
    removed from every answer before it is counted or sent. Turns are taken
    newest first; the first one that does not fit ends the block, so a kept
    turn is never older than a dropped one. The newest turn is kept cut to
    the budget when it does not fit on its own, question first.
    """
    budget = max_chars // 2
    kept: list[HistoryTurn] = []
    used = 0
    for turn in reversed(turns):
        question = turn.question
        answer = _without_markers(turn.answer)
        size = len(question) + len(answer)
        if not kept:
            if size > budget:
                question = question[:budget]
                answer = answer[:budget - len(question)]
                kept = [HistoryTurn(question, answer)]
                break
            kept = [HistoryTurn(question, answer)]
            used = size
            continue
        if used + size > budget:
            break
        kept = [*kept, HistoryTurn(question, answer)]
        used += size
    kept.reverse()
    return kept


def _without_markers(text: str) -> str:
    while True:
        stripped = _CITATION_MARKER.sub("", text)
        if stripped == text:
            return text
        text = stripped


def load_material(
    file_id: str, drive: str, question: str, *,
    budget: int | None = None,
    ranking_text: str | None = None,
) -> Material | None:
    """The chunks to send, or ``None`` when this file cannot be asked about.

    A picture is material even with no chunk; every other kind needs one.
    ``budget`` is what a history block leaves of
    ``rag.file_question_max_chars``; ``ranking_text`` is what chunks are
    ranked against when not all of them fit.
    """
    if budget is None:
        budget = settings.rag.file_question_max_chars
    if ranking_text is None:
        ranking_text = question
    with get_search_db_read() as session:
        indexed = session.execute(
            sql_text(
                "SELECT mime_type, file_type, file_path, duration "
                "FROM indexed_files "
                "WHERE file_id = :fid AND drive = :drive AND active = 1"
            ),
            {"fid": file_id, "drive": drive},
        ).first()
        if indexed is None:
            return None
        mime = indexed[0] or ""
        file_type = indexed[1] or ""
        file_path = indexed[2]
        duration = indexed[3]
        transcripts = session.execute(
            sql_text(
                "SELECT chunk_index, text, timestamp_start, timestamp_end "
                "FROM transcript_chunks WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchall()
        documents = session.execute(
            sql_text(
                "SELECT chunk_index, page, text FROM fts_text_content "
                "WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchall()
        if not transcripts and not documents:
            if not _is_picture(file_type, mime):
                return None
            return Material(
                chunks=(),
                partial=False,
                file_id=file_id,
                file_type=file_type,
                mime_type=mime,
                file_path=file_path,
                duration=duration,
            )
        section_titles = {
            int(page): title
            for page, title in session.execute(
                sql_text(
                    "SELECT page, title FROM document_sections "
                    "WHERE file_id = :fid"
                ),
                {"fid": file_id},
            ).fetchall()
        }
        whisper_embeddings = session.execute(
            sql_text(
                "SELECT id, timestamp_start, timestamp_end FROM embeddings "
                "WHERE file_id = :fid AND embedding_type = 'whisper' "
                "AND vector_table = 'vec_text'"
            ),
            {"fid": file_id},
        ).fetchall()
        text_embeddings = session.execute(
            sql_text(
                "SELECT id, chunk_index FROM embeddings "
                "WHERE file_id = :fid AND embedding_type = 'text_content' "
                "AND vector_table = 'vec_text'"
            ),
            {"fid": file_id},
        ).fetchall()
        frame_starts = (
            {
                int(row[0]): row[1]
                for row in session.execute(
                    sql_text(
                        "SELECT chunk_index, start_time FROM video_frame_texts "
                        "WHERE file_id = :fid"
                    ),
                    {"fid": file_id},
                ).fetchall()
            }
            if file_type == "video"
            else {}
        )

    by_time = {
        (round(row[1], 3), round(row[2], 3)): row[0]
        for row in whisper_embeddings
        if row[1] is not None and row[2] is not None
    }
    by_chunk = {
        int(row[1]): row[0] for row in text_embeddings if row[1] is not None
    }

    rows: list[_Row] = []
    # Transcript chunks first, by timestamp; a file with both kinds uses both.
    for _chunk_index, text, start, end in sorted(
        transcripts, key=lambda row: (row[2], row[0]),
    ):
        if not text:
            continue
        rows = [
            *rows,
            _Row(
                text=text,
                position={"seconds": start},
                embedding_id=by_time.get((round(start, 3), round(end, 3))),
                order=len(rows),
            ),
        ]
    # The FTS column is untyped, so "10" would sort before "2" as text.
    for chunk_index, page, text in sorted(
        documents, key=lambda row: _chunk_order(row[0]),
    ):
        if not text:
            continue
        index = _as_int(chunk_index)
        if file_type == "video" and index is not None and index in frame_starts:
            position = {"seconds": frame_starts[index]}
        else:
            position = _document_position(mime, page, section_titles)
        rows = [
            *rows,
            _Row(
                text=text,
                position=position,
                embedding_id=by_chunk.get(index),
                order=len(rows),
            ),
        ]

    if not rows:
        return None

    material = dict(
        file_id=file_id,
        file_type=file_type,
        mime_type=mime,
        file_path=file_path,
        duration=duration,
    )
    if sum(len(row.text) for row in rows) <= budget:
        return Material(chunks=_selected(rows), partial=False, **material)

    return Material(
        chunks=_selected(_fit(_rank(rows, ranking_text), budget)),
        partial=True,
        **material,
    )


def load_image(
    material: Material, position_seconds: float | None,
) -> ImageMaterial:
    """Prepare the one image the question may carry.

    A picture is re-encoded through the description worker's own helpers;
    a video is one ``large`` frame at the millisecond key its cache uses.
    Raises :class:`ImageUnavailable` with the class the route logs.
    """
    if is_picture(material):
        loaded = _load_image_bytes(material.file_id)
        if loaded is None:
            raise ImageUnavailable(IMAGE_FAILURE_PICTURE)
        raw, mime_type = loaded
        processed = _preprocess_image(raw, mime_type)
        if processed is None:
            raise ImageUnavailable(IMAGE_FAILURE_PICTURE)
        data, out_mime = processed
        return ImageMaterial(data, out_mime)
    if material.file_type == "video" and position_seconds is not None:
        return _frame_image(material, position_seconds)
    raise ImageUnavailable(IMAGE_FAILURE_FRAME)


def _frame_image(material: Material, position_seconds: float) -> ImageMaterial:
    if material.duration is not None and position_seconds > material.duration:
        raise ImageUnavailable(IMAGE_FAILURE_FRAME)

    ms = int(round(position_seconds * 1000))
    timestamp = ms / 1000.0
    cache_path = frame_cache_path(material.file_id, timestamp, "large")
    if not (cache_path.exists() and cache_path.stat().st_size > 0):
        if material.mime_type == LOFT_MIME:
            _youtube_frame_to_cache(material, cache_path, timestamp)
        else:
            _native_frame_to_cache(material, cache_path, timestamp)
    if not (cache_path.exists() and cache_path.stat().st_size > 0):
        raise ImageUnavailable(IMAGE_FAILURE_FRAME)
    try:
        return ImageMaterial(cache_path.read_bytes(), "image/webp")
    except OSError as exc:
        raise ImageUnavailable(IMAGE_FAILURE_FRAME) from exc


def _youtube_frame_to_cache(
    material: Material, cache_path: Any, timestamp: float,
) -> None:
    try:
        source = read_loft(material.file_path)
        if source is None or source.provider != YOUTUBE_PROVIDER:
            raise ImageUnavailable(IMAGE_FAILURE_FRAME)
        ensure_youtube_frame(
            material.file_id, source.url, cache_path, timestamp, "large",
        )
    except ImageUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - the frame route's result decides
        logger.warning(
            "file question frame extraction failed: %s", type(exc).__name__,
        )
        raise ImageUnavailable(IMAGE_FAILURE_FRAME) from exc


def _native_frame_to_cache(
    material: Material, cache_path: Any, timestamp: float,
) -> None:
    if not material.file_path or not config.validate_file_path(
        material.file_path
    ):
        raise ImageUnavailable(IMAGE_FAILURE_FRAME)
    try:
        extract_frame_to_cache(
            material.file_path, cache_path, timestamp, size="large",
        )
    except Exception as exc:  # noqa: BLE001 - ffmpeg's result decides
        logger.warning(
            "file question frame extraction failed: %s",
            type(exc).__name__,
        )
        raise ImageUnavailable(IMAGE_FAILURE_FRAME) from exc


def citations(
    answer: str, chunks: tuple[SelectedChunk, ...],
) -> list[dict[str, Any]]:
    """Every ``[n]`` the answer names, with that chunk's position.

    A number that was not sent is dropped (SPEC-ADDON-053 I3). Each number
    appears once, in the order the answer first names it.
    """
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for match in _CITATION_MARKER.finditer(answer):
        number = int(match.group(1))
        if number in seen or not 1 <= number <= len(chunks):
            continue
        seen = {*seen, number}
        out = [*out, {"n": number, "position": chunks[number - 1].position}]
    return out


def _selected(rows: list[_Row]) -> tuple[SelectedChunk, ...]:
    return tuple(
        SelectedChunk(text=row.text, position=row.position) for row in rows
    )


def _rank(rows: list[_Row], question: str) -> list[_Row]:
    """The rows by descending similarity to the question, then file order.

    A chunk whose embedding row is missing ranks after every embedded one.
    ``embed_query`` raising means the question cannot be embedded, and file
    order is used instead.
    """
    try:
        query_vector = embed_query(question)
    except Exception as exc:  # noqa: BLE001 - file order is a valid answer
        logger.warning(
            "file question embedding failed: %s", type(exc).__name__,
        )
        return rows

    try:
        vectors = _vectors(
            [row.embedding_id for row in rows if row.embedding_id]
        )
    except Exception as exc:  # noqa: BLE001 - file order is a valid answer
        logger.warning(
            "file question vector scan failed: %s", type(exc).__name__,
        )
        return rows

    scored: list[tuple[float, _Row]] = []
    unembedded: list[_Row] = []
    for row in rows:
        vector = vectors.get(row.embedding_id) if row.embedding_id else None
        if vector is None:
            unembedded = [*unembedded, row]
        else:
            scored = [*scored, (_cosine(query_vector, vector), row)]
    scored.sort(key=lambda item: (-item[0], item[1].order))
    return [*[row for _score, row in scored], *unembedded]


def _fit(rows: list[_Row], budget: int) -> list[_Row]:
    """Greedily take rows while they fit, then restore file order.

    An overflow is skipped rather than ending the fill. When not even the
    best-ranked row fits, it is cut to the budget and sent alone.
    """
    taken: list[_Row] = []
    used = 0
    for row in rows:
        if used + len(row.text) > budget:
            continue
        taken = [*taken, row]
        used += len(row.text)
    if not taken and rows:
        taken = [replace(rows[0], text=rows[0].text[:budget])]
    taken.sort(key=lambda row: row.order)
    return taken


def _vectors(embedding_ids: list[str]) -> dict[str, np.ndarray]:
    """The stored ``vec_text`` vectors of ``embedding_ids``."""
    wanted = sorted(set(embedding_ids))
    out: dict[str, np.ndarray] = {}
    engine = get_search_engine()
    for start in range(0, len(wanted), _VECTOR_BATCH):
        batch = wanted[start:start + _VECTOR_BATCH]
        placeholders = ", ".join(f":id{i}" for i in range(len(batch)))
        with engine.connect() as conn:
            rows = conn.execute(
                sql_text(
                    "SELECT embedding_id, vector FROM vec_text "
                    f"WHERE embedding_id IN ({placeholders})"
                ),
                {f"id{i}": value for i, value in enumerate(batch)},
            ).fetchall()
        for embedding_id, blob in rows:
            if blob:
                out[embedding_id] = np.frombuffer(blob, dtype=np.float32)
    return out


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right)) / denominator


def _document_position(
    mime: str, page_raw: Any, section_titles: dict[int, str],
) -> dict[str, Any] | None:
    """Where a document chunk sits, in the vocabulary its viewer moves by.

    A PDF's page is the page number; an EPUB's is the spine section number
    ``?section=`` takes, with the stored section title. Every other kind
    has no position a viewer can move to.
    """
    page = _as_int(page_raw)
    if page is None:
        return None
    if mime == PDF_MIME:
        return {"page": page}
    if mime == EPUB_MIME:
        return {"section": page, "title": section_titles.get(page)}
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _chunk_order(value: Any) -> int:
    """A sortable chunk index; an unreadable one sorts first, as text would."""
    number = _as_int(value)
    return -1 if number is None else number
