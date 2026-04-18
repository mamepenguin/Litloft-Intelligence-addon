"""Citation linker for detailed_summary segments.

Given a ``detailed_summary`` Markdown document and the file it was
generated from, compute the best-matching source chunks for each
segment and persist them as ``detailed_summary_citations`` rows.

Design:

* Parsing happens in :mod:`app.summary_parser` — this module only
  consumes ``Segment`` objects.
* Embeddings are produced by the shared ``text_embedding`` model
  (:mod:`app.workers.embedder`), the same one used to index
  transcripts and documents. Using the same model keeps the cosine
  space consistent.
* Top-k candidates per segment are pulled from ``vec_text`` using
  sqlite-vec's KNN operator, filtered to the current ``file_id`` and
  the transcript / text_content embedding types. CLIP (image) vectors
  are excluded because the detailed_summary is text-only.
* A segment with top-1 score below ``summaries.citation_threshold``
  still gets a row with ``has_citation = False`` so the UI can render
  a "no strong source" warning.

The writer replaces-in-place: it wipes all existing citations for
``file_id`` and writes the freshly computed set in one transaction.
This keeps the ``UNIQUE (file_id, section_path)`` invariant simple —
regeneration / edit / revert always start from a clean slate.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np
from sqlalchemy import text as sql_text

from app.config import settings
from app.database import get_search_db
from app.summary_parser import Segment, parse_segments

logger = logging.getLogger(__name__)

# Embedding id formats used by the indexer. We parse these to recover
# a human-readable chunk identifier for ``citation_chunk_ids``.
#
# Transcript (Whisper) rows:      wh_{file_id}_{chunk_index}_{hash}
# Text-content (document) rows:   txt_{file_id}_{chunk_index}_{hash}
_WHISPER_RE = re.compile(r"^wh_[^_]+_(\d+)_")
_TEXT_CONTENT_RE = re.compile(r"^txt_[^_]+_(\d+)_")


def _make_chunk_id(embedding_id: str) -> str | None:
    """Derive a UI-friendly chunk identifier from a ``vec_text`` row.

    Transcripts become ``transcript:{chunk_index}`` and document chunks
    become ``document:{chunk_index}``. The prefix lets the frontend
    choose an appropriate jump target (seek-to-timestamp vs
    scroll-to-chunk). Returns ``None`` for embedding ids that don't
    match either known format.
    """
    match = _WHISPER_RE.match(embedding_id)
    if match:
        return f"transcript:{match.group(1)}"
    match = _TEXT_CONTENT_RE.match(embedding_id)
    if match:
        return f"document:{match.group(1)}"
    return None


def _query_top_chunks(
    file_id: str, query_vector: np.ndarray, top_k: int
) -> list[tuple[str, float]]:
    """Return ``[(chunk_id, score), ...]`` for the top-K source chunks.

    Uses sqlite-vec's KNN operator on ``vec_text`` joined against
    ``embeddings`` to restrict results to ``file_id`` and supported
    embedding types (``whisper`` / ``text_content``).

    Scores are returned as cosine similarity (higher = better). The
    virtual table exposes ``distance`` which for normalised vectors is
    ``1 - cos_sim``; we convert back for caller convenience so the
    threshold comparison reads naturally.
    """
    if top_k <= 0:
        return []

    try:
        vec_bytes = np.asarray(query_vector, dtype=np.float32).tobytes()
    except (TypeError, ValueError) as e:
        logger.warning("Invalid query vector for %s: %s", file_id, e)
        return []

    # We over-fetch because the KNN scan is global; filtering by
    # file_id + embedding_type on the join side trims the result down
    # to the per-file candidates. 200 keeps the scan fast even on
    # databases with millions of chunks.
    knn_k = min(200, max(top_k * 20, 40))

    try:
        with get_search_db() as session:
            rows = session.execute(
                sql_text(
                    "SELECT v.embedding_id, v.distance "
                    "FROM vec_text v "
                    "JOIN embeddings e "
                    "ON CAST(e.id AS TEXT) = v.embedding_id "
                    "WHERE v.vector MATCH :vec AND k = :k "
                    "AND e.file_id = :fid "
                    "AND e.embedding_type IN ('whisper', 'text_content') "
                    "ORDER BY v.distance"
                ),
                {"vec": vec_bytes, "k": knn_k, "fid": file_id},
            ).fetchall()
    except Exception as e:  # noqa: BLE001 — fail soft, don't break worker
        logger.warning(
            "Citation KNN query failed for %s: %s", file_id, e
        )
        return []

    results: list[tuple[str, float]] = []
    seen: set[str] = set()
    for embedding_id, distance in rows:
        chunk_id = _make_chunk_id(embedding_id or "")
        if chunk_id is None or chunk_id in seen:
            continue
        # Guard against NULL distance (shouldn't happen but fail soft).
        try:
            dist = float(distance)
        except (TypeError, ValueError):
            continue
        score = max(0.0, 1.0 - dist)
        seen = {*seen, chunk_id}
        results = [*results, (chunk_id, score)]
        if len(results) >= top_k:
            break
    return results


def _embed_segment(segment: Segment) -> np.ndarray | None:
    """Embed one segment's text using the shared passage encoder.

    Imports ``embed_passages`` lazily so tests that don't exercise the
    real embedder (most of the suite) can stub it out before the first
    citation call. Returns ``None`` on embedding failure so we can
    persist a ``has_citation = False`` row instead of crashing.
    """
    text = segment.segment_text.strip()
    if not text:
        return None
    try:
        from app.workers.embedder import embed_passages

        vectors = embed_passages([text])
    except Exception as e:  # noqa: BLE001 — fail soft
        logger.warning(
            "Citation embedding failed for %s: %s", segment.section_path, e
        )
        return None
    if vectors is None or len(vectors) == 0:
        return None
    return np.asarray(vectors[0], dtype=np.float32)


def compute_citations(
    file_id: str, detailed_summary: str
) -> list[dict]:
    """Compute (but do not persist) citations for a detailed_summary.

    Split into its own function so the worker path can persist via
    :func:`write_citations` while tests can assert on the raw output.

    Each returned dict has the keys used by ``write_citations``:

    * ``section_path``         — from the parser
    * ``segment_type``         — from the parser
    * ``segment_text``         — trimmed copy of the raw segment
    * ``citation_chunk_ids``   — list of chunk ids whose score passes threshold
    * ``top_score``            — the top-1 cosine similarity (0.0 if no chunks)
    * ``has_citation``         — True iff top-1 >= threshold
    """
    threshold = settings.summaries.citation_threshold
    top_k = settings.summaries.citation_top_k

    segments = parse_segments(detailed_summary)
    results: list[dict] = []

    for seg in segments:
        vector = _embed_segment(seg)
        if vector is None:
            results.append(
                {
                    "section_path": seg.section_path,
                    "segment_type": seg.segment_type,
                    "segment_text": seg.segment_text,
                    "citation_chunk_ids": [],
                    "top_score": 0.0,
                    "has_citation": False,
                }
            )
            continue

        candidates = _query_top_chunks(file_id, vector, top_k)
        if not candidates:
            results.append(
                {
                    "section_path": seg.section_path,
                    "segment_type": seg.segment_type,
                    "segment_text": seg.segment_text,
                    "citation_chunk_ids": [],
                    "top_score": 0.0,
                    "has_citation": False,
                }
            )
            continue

        top_score = candidates[0][1]
        has_citation = top_score >= threshold
        # Only persist chunks whose individual score clears the bar.
        # This keeps the UI promise ("shown citations are strong") but
        # still records top_score so the ⚠ marker can fire even when
        # the chunk list is empty.
        passing = [cid for cid, score in candidates if score >= threshold]
        results.append(
            {
                "section_path": seg.section_path,
                "segment_type": seg.segment_type,
                "segment_text": seg.segment_text,
                "citation_chunk_ids": passing,
                "top_score": top_score,
                "has_citation": has_citation,
            }
        )
    return results


def write_citations(file_id: str, citations: list[dict]) -> tuple[int, int]:
    """Replace all stored citations for ``file_id``.

    Returns ``(citation_count, no_citation_count)`` for use in the
    ``citations_ready`` WebSocket payload. ``citation_count`` is the
    number of segments with ``has_citation = True``; the "no" count
    is the complement (segments the LLM produced without a strong
    source anchor).

    Safe to call with an empty ``citations`` list: the existing rows
    are still wiped, leaving the file in a "no citations computed"
    state. The API endpoint renders this identically to "no detailed
    summary" so the UI stays quiet.
    """
    with_count = sum(1 for c in citations if c["has_citation"])
    without_count = len(citations) - with_count

    with get_search_db() as session:
        session.execute(
            sql_text(
                "DELETE FROM detailed_summary_citations "
                "WHERE file_id = :fid"
            ),
            {"fid": file_id},
        )
        for cit in citations:
            session.execute(
                sql_text(
                    "INSERT INTO detailed_summary_citations "
                    "(file_id, section_path, segment_type, segment_text, "
                    "citation_chunk_ids, top_score, has_citation) "
                    "VALUES (:fid, :section_path, :segment_type, "
                    ":segment_text, :citation_chunk_ids, :top_score, "
                    ":has_citation)"
                ),
                {
                    "fid": file_id,
                    "section_path": cit["section_path"],
                    "segment_type": cit["segment_type"],
                    "segment_text": cit["segment_text"],
                    "citation_chunk_ids": json.dumps(
                        cit["citation_chunk_ids"]
                    ),
                    "top_score": float(cit["top_score"]),
                    "has_citation": bool(cit["has_citation"]),
                },
            )
    return (with_count, without_count)


def calculate_and_store(
    file_id: str, detailed_summary: str
) -> tuple[int, int]:
    """Compute citations for ``file_id`` and persist them.

    Convenience wrapper for workers: equivalent to
    ``write_citations(file_id, compute_citations(file_id, summary))``.
    Returns ``(citation_count, no_citation_count)`` so the caller can
    emit the ``citations_ready`` WebSocket event without a second
    database round trip.
    """
    citations = compute_citations(file_id, detailed_summary)
    return write_citations(file_id, citations)


def get_citations(file_id: str) -> list[dict]:
    """Fetch all citations for ``file_id`` in section order.

    Rows are returned as plain dicts with the fields shaped for the
    API response (JSON array of chunk ids is decoded; booleans are
    coerced to native Python bools). Returns an empty list when no
    citations have been computed yet.
    """
    with get_search_db() as session:
        rows = session.execute(
            sql_text(
                "SELECT section_path, segment_type, segment_text, "
                "citation_chunk_ids, top_score, has_citation "
                "FROM detailed_summary_citations "
                "WHERE file_id = :fid "
                "ORDER BY id"
            ),
            {"fid": file_id},
        ).fetchall()

    results: list[dict] = []
    for row in rows:
        try:
            chunk_ids = json.loads(row[3]) if row[3] else []
        except (TypeError, ValueError):
            chunk_ids = []
        results.append(
            {
                "section_path": row[0],
                "segment_type": row[1],
                "segment_text": row[2],
                "chunk_ids": chunk_ids,
                "top_score": float(row[4]),
                "has_citation": bool(row[5]),
            }
        )
    return results


def delete_citations(file_id: str) -> int:
    """Delete all citations for ``file_id``. Returns row count deleted."""
    with get_search_db() as session:
        result = session.execute(
            sql_text(
                "DELETE FROM detailed_summary_citations "
                "WHERE file_id = :fid"
            ),
            {"fid": file_id},
        )
        return int(result.rowcount or 0)
