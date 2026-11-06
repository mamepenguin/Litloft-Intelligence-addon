"""The material a question about one file is answered from (SPEC-ADDON-053).

A file's text is its indexed chunks, read from ``search.db``; nothing is
extracted at question time, which is why a file becomes askable only once
indexing has run. Every chunk is sent when they fit
``rag.file_question_max_chars``. Otherwise the chunks are ranked by the
cosine similarity of their stored embedding to the question, taken in rank
order while they fit, and sent in file order with ``partial`` set.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from sqlalchemy import text as sql_text

from app.config import settings
from app.database import get_search_db_read, get_search_engine
from app.document_sections import EPUB_MIME
from app.workers.embedder import embed_query

logger = logging.getLogger(__name__)

PDF_MIME = "application/pdf"

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


@dataclass(frozen=True)
class _Row:
    text: str
    position: dict[str, Any] | None
    embedding_id: str | None
    #: Position in the file, which is also the order the chunks are sent in.
    order: int


def available(file_id: str, drive: str) -> bool:
    """Whether this drive may ask about this file (SPEC-ADDON-053 I6).

    An active ``indexed_files`` row of the request drive with at least one
    text chunk. Everything else is ``False`` rather than an error, so the
    tab simply has no button.
    """
    with get_search_db_read() as session:
        indexed = session.execute(
            sql_text(
                "SELECT 1 FROM indexed_files "
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
    return bool(chunks)


def load_material(
    file_id: str, drive: str, question: str,
) -> Material | None:
    """The chunks to send, or ``None`` when this file cannot be asked about."""
    budget = settings.rag.file_question_max_chars
    with get_search_db_read() as session:
        indexed = session.execute(
            sql_text(
                "SELECT mime_type, file_type FROM indexed_files "
                "WHERE file_id = :fid AND drive = :drive AND active = 1"
            ),
            {"fid": file_id, "drive": drive},
        ).first()
        if indexed is None:
            return None
        mime = indexed[0] or ""
        file_type = indexed[1] or ""
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
            return None
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

    if sum(len(row.text) for row in rows) <= budget:
        return Material(_selected(rows), False)

    return Material(_selected(_fit(_rank(rows, question), budget)), True)


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
