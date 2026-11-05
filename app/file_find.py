"""Every occurrence of a term inside one indexed file.

The search DB stores text chunks that replay part of the previous chunk.
This module rebuilds each unit (a page, a section, a transcript chunk, a
scene, or the whole text) by merging that replay once, then scans the unit
for non-overlapping occurrences. Matching folds ASCII letters only, as
``exact_search`` does.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text as sql_text

from app.config import settings
from app.database import get_search_db_read
from app.exact_search import (
    SNIPPET_CONTEXT_CHARS,
    _ascii_lower,
    _int_or_none,
    _snippet,
)
from app.search import text_content_kind

ROW_CAP = 2000
_OVERLAP_MIN = 20
#: ``ContentExtractor.chunk_text``'s default, which Office extraction and
#: picture OCR use because they pass no overlap.
_CHUNK_TEXT_OVERLAP = 200
_OFFICE_MIMES = frozenset({
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
})

_MEDIA = 0
_DOCUMENT = 1
_TEXT = 2

_STAGE_BY_KIND = {
    "video": "whisper_indexed",
    "audio": "whisper_indexed",
    "image": "ocr_indexed",
}
_STAGE_FOR_OTHER_KINDS = "text_indexed"


def _stage_done(indexed) -> bool:
    stage = _STAGE_BY_KIND.get(indexed.file_type, _STAGE_FOR_OTHER_KINDS)
    return bool(getattr(indexed, stage))


def _replay_limit(mime_type: str, file_type: str) -> int:
    if file_type == "image" or mime_type in _OFFICE_MIMES:
        return _CHUNK_TEXT_OVERLAP
    return settings.indexing.text_chunking.overlap


def _overlap(text: str, chunk: str, limit: int) -> int:
    limit = min(limit, len(text), len(chunk))
    for size in range(limit, _OVERLAP_MIN - 1, -1):
        if text.endswith(chunk[:size]):
            return size
    return 0


def _merge_chunks(chunks: list[str], limit: int) -> str:
    text = ""
    for chunk in chunks:
        if not text:
            text = chunk
            continue
        overlap = _overlap(text, chunk, limit)
        if overlap:
            text += chunk[overlap:]
        else:
            text += "\n" + chunk
    return text


def _positions(text: str, needle: str):
    folded = _ascii_lower(text)
    at = folded.find(needle)
    ordinal = 0
    while at != -1:
        ordinal += 1
        yield at, ordinal
        at = folded.find(needle, at + len(needle))


def _row(
    text: str,
    at: int,
    needle: str,
    ordinal: int,
    *,
    kind: str,
    start: float | None = None,
    end: float | None = None,
    page: int | None = None,
    section: int | None = None,
    section_title: str | None = None,
) -> dict[str, Any]:
    snippet_start = max(0, at - SNIPPET_CONTEXT_CHARS)
    return {
        "kind": kind,
        "start": start,
        "end": end,
        "page": page,
        "section": section,
        "section_title": section_title,
        "ordinal": ordinal,
        "snippet": _snippet(text, at, len(needle)),
        "match_start": at - snippet_start + (1 if snippet_start else 0),
        "match_length": len(needle),
    }


def _text_occurrences(
    db, file_id: str, needle: str, mime_type: str, file_type: str,
) -> list[tuple[tuple, dict[str, Any]]]:
    rows = db.execute(
        sql_text(
            "SELECT CAST(chunk_index AS INTEGER) AS ci, page, text "
            "FROM fts_text_content WHERE file_id = :file_id ORDER BY ci"
        ),
        {"file_id": file_id},
    ).fetchall()
    units: dict[int | None, list[str]] = {}
    for r in rows:
        units.setdefault(_int_or_none(r.page), []).append(r.text)

    titles: dict[int, str] = {}
    if units:
        titles = {
            int(r.page): r.title
            for r in db.execute(
                sql_text(
                    "SELECT page, title FROM document_sections WHERE file_id = :file_id"
                ),
                {"file_id": file_id},
            ).fetchall()
        }

    limit = _replay_limit(mime_type, file_type)
    out: list[tuple[tuple, dict[str, Any]]] = []
    for page, chunks in units.items():
        text = _merge_chunks(chunks, limit)
        kind = text_content_kind(mime_type, page, file_type)
        location: dict[str, Any] = {}
        if kind == "section":
            location = {
                "section": page,
                "section_title": titles.get(page) if page is not None else None,
            }
            family = _DOCUMENT
        elif kind == "page":
            location = {"page": page}
            family = _DOCUMENT
        else:
            family = _TEXT
        for at, ordinal in _positions(text, needle):
            if family == _DOCUMENT:
                sort_key = (family, page, ordinal)
            else:
                sort_key = (family, ordinal)
            out.append((sort_key, _row(text, at, needle, ordinal, kind=kind, **location)))
    return out


def _transcript_occurrences(
    db, file_id: str, needle: str,
) -> list[tuple[tuple, dict[str, Any]]]:
    rows = db.execute(
        sql_text(
            "SELECT CAST(t.chunk_index AS INTEGER) AS ci, t.text, "
            "c.timestamp_start, c.timestamp_end "
            "FROM fts_transcripts t "
            "LEFT JOIN transcript_chunks c ON c.file_id = t.file_id "
            "AND c.chunk_index = CAST(t.chunk_index AS INTEGER) "
            "WHERE t.file_id = :file_id ORDER BY ci"
        ),
        {"file_id": file_id},
    ).fetchall()
    out = []
    for r in rows:
        if r.timestamp_start is None:
            continue
        for at, ordinal in _positions(r.text, needle):
            out.append((
                (_MEDIA, r.timestamp_start, 0, ordinal),
                _row(
                    r.text, at, needle, ordinal,
                    kind="utterance", start=r.timestamp_start, end=r.timestamp_end,
                ),
            ))
    return out


def _scene_occurrences(
    db, file_id: str, needle: str,
) -> list[tuple[tuple, dict[str, Any]]]:
    rows = db.execute(
        sql_text(
            "SELECT s.start_time, s.end_time, s.visible_text, s.ordering "
            "FROM video_visual_scenes s "
            "JOIN video_visual_runs r ON r.id = s.run_id AND r.is_active = 1 "
            "WHERE r.file_id = :file_id ORDER BY s.ordering"
        ),
        {"file_id": file_id},
    ).fetchall()
    out = []
    for r in rows:
        if not r.visible_text:
            continue
        for at, ordinal in _positions(r.visible_text, needle):
            out.append((
                (_MEDIA, r.start_time, 1, ordinal),
                _row(
                    r.visible_text, at, needle, ordinal,
                    kind="scene", start=r.start_time, end=r.end_time,
                ),
            ))
    return out


def find_in_file(
    file_id: str, drive: str, q: str, *, include_scenes: bool,
) -> dict[str, Any]:
    """The requested file's occurrences of ``q``, capped at ``ROW_CAP`` rows."""
    needle = _ascii_lower(q)
    with get_search_db_read() as db:
        indexed = db.execute(
            sql_text(
                "SELECT file_type, mime_type, text_indexed, whisper_indexed, ocr_indexed "
                "FROM indexed_files "
                "WHERE file_id = :file_id AND drive = :drive AND active = 1"
            ),
            {"file_id": file_id, "drive": drive},
        ).fetchone()
        if indexed is None:
            return {"query": q, "indexed": False, "total": 0, "truncated": False, "rows": []}

        has_visual_run = False
        if include_scenes:
            has_visual_run = db.execute(
                sql_text(
                    "SELECT 1 FROM video_visual_runs "
                    "WHERE file_id = :file_id AND is_active = 1 LIMIT 1"
                ),
                {"file_id": file_id},
            ).fetchone() is not None

        occurrences = _text_occurrences(
            db, file_id, needle, indexed.mime_type, indexed.file_type,
        )
        occurrences += _transcript_occurrences(db, file_id, needle)
        if include_scenes:
            occurrences += _scene_occurrences(db, file_id, needle)

    occurrences.sort(key=lambda item: item[0])
    total = len(occurrences)
    rows = [row for _key, row in occurrences[:ROW_CAP]]
    is_indexed = _stage_done(indexed) or has_visual_run
    return {
        "query": q,
        "indexed": is_indexed,
        "total": total,
        "truncated": total > ROW_CAP,
        "rows": rows,
    }
