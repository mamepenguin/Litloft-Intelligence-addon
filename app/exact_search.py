"""Exact-substring search over indexed text, and nearest-scene ranking.

Both read the search DB only. Matching is ``instr(lower(text), lower(q))``:
SQLite's ``lower()`` folds ASCII letters only, so the Python side folds the
same way (``_ascii_lower``) when it counts occurrences and places snippets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import bindparam
from sqlalchemy import text as sql_text

from app.database import get_search_db_read
from app.document_sections import EPUB_MIME

SOURCES = ("text", "transcript", "scene")
SNIPPET_CONTEXT_CHARS = 80
SCENE_FIELDS = ("scene_label", "visible_text")

_ASCII_LOWER = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"
)


def _ascii_lower(value: str) -> str:
    return value.translate(_ASCII_LOWER)


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _snippet(text: str, at: int, length: int) -> str:
    start = max(0, at - SNIPPET_CONTEXT_CHARS)
    end = min(len(text), at + length + SNIPPET_CONTEXT_CHARS)
    head = "…" if start > 0 else ""
    tail = "…" if end < len(text) else ""
    return f"{head}{text[start:end]}{tail}"


@dataclass(frozen=True)
class _Hit:
    sort_key: tuple
    segment: dict[str, Any]


def _segment(
    *, file_id: str, filename: str, source: str, text: str, needle: str, **location: Any
) -> dict[str, Any]:
    folded = _ascii_lower(text)
    at = folded.find(needle)
    return {
        "file_id": file_id,
        "filename": filename,
        "source": source,
        "chunk_index": None,
        "page": None,
        "section": None,
        "section_title": None,
        "start_time": None,
        "end_time": None,
        "field": None,
        **location,
        "match_count": folded.count(needle),
        "snippet": _snippet(text, at, len(needle)),
    }


_MATCH = "instr(lower({col}), lower(:q)) > 0"
_IN_DRIVE = "f.drive = :drive AND f.active = 1"


def _text_hits(db, q: str, needle: str, drive: str) -> list[_Hit]:
    rows = db.execute(
        sql_text(
            "SELECT f.file_id, f.filename, f.mime_type, t.chunk_index, t.page, t.text "
            "FROM fts_text_content t JOIN indexed_files f ON f.file_id = t.file_id "
            f"WHERE {_IN_DRIVE} AND {_MATCH.format(col='t.text')}"
        ),
        {"q": q, "drive": drive},
    ).fetchall()
    epub_pages = {
        (r.file_id, _int_or_none(r.page)) for r in rows if r.mime_type == EPUB_MIME
    }
    titles = _section_titles(db, epub_pages)
    hits = []
    for r in rows:
        index = int(r.chunk_index)
        page = _int_or_none(r.page)
        if r.mime_type == EPUB_MIME:
            location = {"section": page, "section_title": titles.get((r.file_id, page))}
        else:
            location = {"page": page}
        hits.append(_Hit(
            (r.filename, r.file_id, 0, index),
            _segment(
                file_id=r.file_id, filename=r.filename, source="text", text=r.text,
                needle=needle, chunk_index=index, **location,
            ),
        ))
    return hits


def _section_titles(db, pairs: set[tuple[str, int | None]]) -> dict[tuple[str, int], str]:
    file_ids = sorted({fid for fid, page in pairs if page is not None})
    if not file_ids:
        return {}
    rows = db.execute(
        sql_text(
            "SELECT file_id, page, title FROM document_sections WHERE file_id IN :ids"
        ).bindparams(bindparam("ids", expanding=True)),
        {"ids": file_ids},
    ).fetchall()
    return {(r.file_id, r.page): r.title for r in rows}


def _transcript_hits(db, q: str, needle: str, drive: str) -> list[_Hit]:
    rows = db.execute(
        sql_text(
            "SELECT f.file_id, f.filename, t.chunk_index, t.text, "
            "c.timestamp_start, c.timestamp_end "
            "FROM fts_transcripts t JOIN indexed_files f ON f.file_id = t.file_id "
            "LEFT JOIN transcript_chunks c ON c.file_id = t.file_id "
            "AND c.chunk_index = CAST(t.chunk_index AS INTEGER) "
            f"WHERE {_IN_DRIVE} AND {_MATCH.format(col='t.text')}"
        ),
        {"q": q, "drive": drive},
    ).fetchall()
    return [
        _Hit(
            (r.filename, r.file_id, 1, int(r.chunk_index)),
            _segment(
                file_id=r.file_id, filename=r.filename, source="transcript", text=r.text,
                needle=needle, chunk_index=int(r.chunk_index),
                start_time=r.timestamp_start, end_time=r.timestamp_end,
            ),
        )
        for r in rows
    ]


def _scene_hits(db, q: str, needle: str, drive: str) -> list[_Hit]:
    rows = db.execute(
        sql_text(
            "SELECT f.file_id, f.filename, s.start_time, s.end_time, s.ordering, "
            "s.scene_label, s.visible_text "
            "FROM video_visual_scenes s "
            "JOIN video_visual_runs r ON r.id = s.run_id AND r.is_active = 1 "
            "JOIN indexed_files f ON f.file_id = r.file_id "
            f"WHERE {_IN_DRIVE} AND ("
            f"{_MATCH.format(col='s.scene_label')} OR {_MATCH.format(col='s.visible_text')})"
        ),
        {"q": q, "drive": drive},
    ).fetchall()
    hits = []
    for r in rows:
        for rank, field in enumerate(SCENE_FIELDS):
            value = getattr(r, field)
            if not value or needle not in _ascii_lower(value):
                continue
            hits.append(_Hit(
                (r.filename, r.file_id, 2, r.start_time, r.ordering, rank),
                _segment(
                    file_id=r.file_id, filename=r.filename, source="scene", text=value,
                    needle=needle, start_time=r.start_time, end_time=r.end_time,
                    field=field,
                ),
            ))
    return hits


def find_exact(
    q: str, sources: tuple[str, ...], drive: str, *, include_scenes: bool,
    limit: int, offset: int,
) -> tuple[int, list[dict[str, Any]]]:
    """Every matching segment of the drive, in the stated order, paged."""
    needle = _ascii_lower(q)
    hits: list[_Hit] = []
    with get_search_db_read() as db:
        if "text" in sources:
            hits.extend(_text_hits(db, q, needle, drive))
        if "transcript" in sources:
            hits.extend(_transcript_hits(db, q, needle, drive))
        if "scene" in sources and include_scenes:
            hits.extend(_scene_hits(db, q, needle, drive))
    hits.sort(key=lambda h: h.sort_key)
    return len(hits), [h.segment for h in hits[offset:offset + limit]]


def _scene_id(embedding_id: str) -> int | None:
    parts = embedding_id.split("_")
    if len(parts) < 3 or parts[0] != "vvs" or not parts[1].isdigit():
        return None
    return int(parts[1])


def rank_scenes(query_vector: np.ndarray, drive: str, limit: int) -> list[dict[str, Any]]:
    """The drive's active-run scenes nearest to ``query_vector`` by cosine."""
    with get_search_db_read() as db:
        embeddings = db.execute(
            sql_text(
                "SELECT e.id, e.file_id FROM embeddings e "
                "JOIN indexed_files f ON f.file_id = e.file_id "
                f"WHERE e.embedding_type = 'video_visual_scene' AND {_IN_DRIVE}"
            ),
            {"drive": drive},
        ).fetchall()
        by_scene: dict[int, list[str]] = {}
        owner: dict[str, str] = {}
        for e in embeddings:
            scene_id = _scene_id(e.id)
            if scene_id is not None:
                by_scene.setdefault(scene_id, []).append(e.id)
                owner[e.id] = e.file_id
        if not by_scene:
            return []
        scenes = db.execute(
            sql_text(
                "SELECT s.id, r.file_id, f.filename, s.start_time, s.end_time, "
                "s.scene_label, s.visible_text "
                "FROM video_visual_scenes s "
                "JOIN video_visual_runs r ON r.id = s.run_id AND r.is_active = 1 "
                "JOIN indexed_files f ON f.file_id = r.file_id "
                f"WHERE s.id IN :ids AND {_IN_DRIVE}"
            ).bindparams(bindparam("ids", expanding=True)),
            {"ids": sorted(by_scene), "drive": drive},
        ).fetchall()
        wanted = [
            emb_id
            for s in scenes
            for emb_id in by_scene[s.id]
            if owner[emb_id] == s.file_id
        ]
        if not wanted:
            return []
        vectors = db.execute(
            sql_text(
                "SELECT embedding_id, vector FROM vec_text WHERE embedding_id IN :ids"
            ).bindparams(bindparam("ids", expanding=True)),
            {"ids": wanted},
        ).fetchall()

    vector_of = {
        row.embedding_id: np.frombuffer(row.vector, dtype=np.float32)
        for row in vectors
        if row.vector
    }
    query_norm = float(np.linalg.norm(query_vector))
    results = []
    for s in scenes:
        scores = []
        for emb_id in by_scene[s.id]:
            vec = vector_of.get(emb_id)
            if vec is None or owner[emb_id] != s.file_id:
                continue
            denom = query_norm * float(np.linalg.norm(vec))
            if denom == 0:
                continue
            scores.append(float(np.dot(query_vector, vec)) / denom)
        if not scores:
            continue
        results.append({
            "file_id": s.file_id,
            "filename": s.filename,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "scene_label": s.scene_label,
            "visible_text": s.visible_text,
            "score": max(scores),
        })
    results.sort(key=lambda r: (-r["score"], r["file_id"], r["start_time"]))
    return [{**r, "score": round(r["score"], 4)} for r in results[:limit]]
