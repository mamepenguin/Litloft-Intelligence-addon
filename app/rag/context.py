"""RAG context builder: turn retrieved files into LLM-ready snippets.

Given a set of ``RetrievedFile`` (from ``app.rag.retriever``), produce
compact ``FileContext`` objects that carry only the text the LLM
actually needs to answer the query. Two public functions:

* ``build_file_context`` — per-file extraction driven by segment matches.
* ``assemble_contexts``  — multi-file aggregator that enforces the
  configured total-context budget by dropping the lowest-scoring files
  when the cap would be exceeded.

The fetch helpers (``_fetch_transcript_chunks_around``,
``_fetch_document_chunks_around``, ``_fetch_blip_caption``) are defined
as module-level functions so tests can monkeypatch them without
touching the real search database.
"""

import logging
import re
from dataclasses import dataclass

import numpy as np
from sqlalchemy import text as sql_text

from app.config import RagConfig
from app.database import get_search_db
from app.models import Embedding, TranscriptChunk
from app.rag.keyword_filter import filter_keywords
from app.rag.retriever import RetrievedFile
from app.search import MatchInfo, SegmentGroup
from app.text_utils import trim_to_sentence_boundary
from app.workers.embedder import embed_query
from app.workers.whisper import HVLINK_MIME

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextSnippet:
    """A single excerpt from a file relevant to the query."""

    source: str  # "transcript" | "text_content" | "clip" | "metadata"
    text: str
    location: str | None  # e.g. "0:45", "page 3"


@dataclass(frozen=True)
class FileContext:
    """All context extracted from a single retrieved file."""

    file_id: str
    filename: str
    drive: str
    file_type: str
    title: str | None
    description: str | None
    snippets: tuple[ContextSnippet, ...]
    total_chars: int


# ---------------------------------------------------------------------------
# Fetch helpers (tests monkeypatch these)
# ---------------------------------------------------------------------------


def _fetch_transcript_chunks_around(
    file_id: str,
    start: float,
    end: float,
    window_seconds: float,
) -> list[tuple[str, float, float]]:
    """Fetch transcript chunks overlapping a ``[start, end]`` window.

    Expands the query range by ``window_seconds`` on each side so that
    surrounding context (the sentence before and after the matching
    segment) is available to the LLM.

    Returns:
        List of ``(text, timestamp_start, timestamp_end)`` tuples in
        chronological order.
    """
    lo = max(0.0, start - window_seconds)
    hi = end + window_seconds

    with get_search_db() as session:
        rows = (
            session.query(TranscriptChunk)
            .filter(
                TranscriptChunk.file_id == file_id,
                TranscriptChunk.timestamp_end >= lo,
                TranscriptChunk.timestamp_start <= hi,
            )
            .order_by(TranscriptChunk.timestamp_start)
            .all()
        )
        return [
            (row.text, row.timestamp_start, row.timestamp_end)
            for row in rows
            if row.text
        ]


def _fetch_document_chunks_around(
    file_id: str,
    chunk_index: int,
) -> list[tuple[int, str]]:
    """Fetch document FTS chunks around a given chunk index.

    Returns the chunk itself plus its immediate neighbors
    (chunk_index - 1 and chunk_index + 1) so the LLM sees enough
    surrounding prose to understand the match.

    Notes:
        The ``chunk_index`` column in ``fts_text_content`` is a string
        (FTS5 stores everything as TEXT), so numeric comparisons must
        cast to INTEGER — otherwise "10" would sort before "2". See
        hako memo ``EAiVExR4vGgOym5aAv_Up``.

    Returns:
        List of ``(chunk_index, text)`` tuples in numeric order.
    """
    lo = max(0, chunk_index - 1)
    hi = chunk_index + 1

    with get_search_db() as session:
        rows = session.execute(
            sql_text(
                "SELECT chunk_index, text FROM fts_text_content "
                "WHERE file_id = :fid "
                "AND CAST(chunk_index AS INTEGER) BETWEEN :lo AND :hi "
                "ORDER BY CAST(chunk_index AS INTEGER)"
            ),
            {"fid": file_id, "lo": lo, "hi": hi},
        ).fetchall()
        return [
            (int(row[0]), row[1])
            for row in rows
            if row[1]
        ]


_TEXT_EMBEDDING_ID_CHUNK_RE = re.compile(r"^txt_[^_]+_(\d+)_")
_WHISPER_EMBEDDING_ID_CHUNK_RE = re.compile(r"^wh_[^_]+_(\d+)_")


def _fetch_document_chunks_by_vector(
    file_id: str,
    query_vector: np.ndarray,
    top_n: int,
) -> list[int]:
    """Return the top-N chunk indices of a document by vector similarity.

    Queries ``vec_text`` for the ``top_n`` nearest ``text_content``
    embeddings that belong to ``file_id`` and extracts the chunk index
    encoded in each embedding id (format ``txt_{file_id}_{idx}_{hash}``).

    This rescues the "semantically close file, keyword-poor chunk" case
    where the file is retrieved via vector similarity but the specific
    passage that answers the query does not overlap the transformed
    keyword tokens. Without this, ``_collect_document_snippets`` only
    sees FTS-matched chunks and the LLM context can miss the actual
    answer while the file itself is clearly on-topic.

    Args:
        file_id: The file to fetch chunks from.
        query_vector: L2-normalized query embedding (same model as
            the passages, same shape as ``vec_text.vector``).
        top_n: Max number of chunk indices to return.

    Returns:
        Chunk indices in vector-similarity order (nearest first).
        Duplicates are removed while preserving first-seen order.
        Empty list on any error or when ``top_n <= 0``.
    """
    if top_n <= 0:
        return []

    try:
        vec_bytes = np.asarray(query_vector, dtype=np.float32).tobytes()
    except (TypeError, ValueError) as e:
        logger.warning("Invalid query vector for %s: %s", file_id, e)
        return []

    # sqlite-vec's KNN operator ranks globally; we pull more than top_n
    # to tolerate the case where another file dominates the top of the
    # distance ordering. k is capped at 4096 by sqlite-vec; 200 is ample
    # for a single-file rescue and keeps the scan fast.
    knn_k = min(200, max(top_n * 20, 40))

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
                    "AND e.embedding_type = 'text_content' "
                    "ORDER BY v.distance"
                ),
                {"vec": vec_bytes, "k": knn_k, "fid": file_id},
            ).fetchall()
    except Exception as e:  # noqa: BLE001 - fail soft, don't break RAG
        logger.warning(
            "Vector chunk lookup failed for %s: %s", file_id, e
        )
        return []

    indices: list[int] = []
    seen: set[int] = set()
    for emb_id, _distance in rows:
        match = _TEXT_EMBEDDING_ID_CHUNK_RE.match(emb_id or "")
        if not match:
            continue
        idx = int(match.group(1))
        if idx in seen:
            continue
        seen = {*seen, idx}
        indices = [*indices, idx]
        if len(indices) >= top_n:
            break
    return indices


def _fetch_document_chunks_by_keyword_or(
    file_id: str,
    keywords: str,
    top_n: int,
) -> list[int]:
    """Return chunk indices that literally contain any of the keywords.

    FTS5 OR search within the target file — complements the vector
    pass by catching chunks where the answer phrasing literally
    contains a query token even though the chunk's embedding is far
    from the query embedding. This is common for terse summary
    passages (e.g. a one-line enumeration of 5 agreed points) whose
    vector doesn't match the wordy question but whose text contains
    the question's literal anchor words.

    Each whitespace-separated keyword is quoted as an FTS5 phrase and
    joined with ``OR`` so any single keyword hitting is enough. Chunk
    indices are returned in FTS rank order, deduped.
    """
    if top_n <= 0:
        return []

    terms = [t.strip() for t in (keywords or "").split() if t.strip()]
    if not terms:
        return []

    quoted = [f'"{t.replace(chr(34), "")}"' for t in terms if t.replace(chr(34), "")]
    if not quoted:
        return []
    fts_query = " OR ".join(quoted)

    try:
        with get_search_db() as session:
            rows = session.execute(
                sql_text(
                    "SELECT chunk_index "
                    "FROM fts_text_content "
                    "WHERE fts_text_content MATCH :q "
                    "AND file_id = :fid "
                    "ORDER BY rank "
                    "LIMIT :lim"
                ),
                {"q": fts_query, "fid": file_id, "lim": top_n * 3},
            ).fetchall()
    except Exception as e:  # noqa: BLE001 - fail soft
        logger.warning(
            "Keyword chunk lookup failed for %s: %s", file_id, e
        )
        return []

    indices: list[int] = []
    seen: set[int] = set()
    for row in rows:
        try:
            idx = int(row[0])
        except (TypeError, ValueError):
            continue
        if idx in seen:
            continue
        seen = {*seen, idx}
        indices = [*indices, idx]
        if len(indices) >= top_n:
            break
    return indices


def _fetch_transcript_chunks_by_vector(
    file_id: str,
    query_vector: np.ndarray,
    top_n: int,
) -> list[tuple[str, float, float]]:
    """Return top-N transcript chunks by vector similarity.

    Queries ``vec_text`` for the ``top_n`` nearest ``whisper``
    embeddings belonging to ``file_id``, extracts the chunk index
    from each embedding id (format ``wh_{file_id}_{idx}_{hash}``),
    then fetches the actual text and timestamps from
    ``TranscriptChunk``.

    Returns:
        List of ``(text, timestamp_start, timestamp_end)`` tuples
        in vector-similarity order (nearest first).
    """
    if top_n <= 0:
        return []

    try:
        vec_bytes = np.asarray(query_vector, dtype=np.float32).tobytes()
    except (TypeError, ValueError) as e:
        logger.warning("Invalid query vector for %s: %s", file_id, e)
        return []

    knn_k = min(200, max(top_n * 20, 40))

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
                    "AND e.embedding_type = 'whisper' "
                    "ORDER BY v.distance"
                ),
                {"vec": vec_bytes, "k": knn_k, "fid": file_id},
            ).fetchall()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Transcript vector chunk lookup failed for %s: %s",
            file_id, e,
        )
        return []

    chunk_indices: list[int] = []
    seen: set[int] = set()
    for emb_id, _distance in rows:
        match = _WHISPER_EMBEDDING_ID_CHUNK_RE.match(emb_id or "")
        if not match:
            continue
        idx = int(match.group(1))
        if idx in seen:
            continue
        seen = {*seen, idx}
        chunk_indices = [*chunk_indices, idx]
        if len(chunk_indices) >= top_n:
            break

    if not chunk_indices:
        return []

    with get_search_db() as session:
        chunks = (
            session.query(TranscriptChunk)
            .filter(
                TranscriptChunk.file_id == file_id,
                TranscriptChunk.chunk_index.in_(chunk_indices),
            )
            .all()
        )
        by_idx = {c.chunk_index: c for c in chunks}

    return [
        (by_idx[idx].text, by_idx[idx].timestamp_start, by_idx[idx].timestamp_end)
        for idx in chunk_indices
        if idx in by_idx and by_idx[idx].text
    ]


def _fetch_transcript_chunks_by_keyword_or(
    file_id: str,
    keywords: str,
    top_n: int,
) -> list[tuple[str, float, float]]:
    """Return transcript chunks that literally contain any keyword.

    FTS5 OR search on ``fts_transcripts`` within the target file,
    then fetches timestamps from ``TranscriptChunk``.

    Returns:
        List of ``(text, timestamp_start, timestamp_end)`` tuples
        in FTS rank order.
    """
    if top_n <= 0:
        return []

    terms = [t.strip() for t in (keywords or "").split() if t.strip()]
    if not terms:
        return []

    quoted = [
        f'"{t.replace(chr(34), "")}"' for t in terms if t.replace(chr(34), "")
    ]
    if not quoted:
        return []
    fts_query = " OR ".join(quoted)

    try:
        with get_search_db() as session:
            rows = session.execute(
                sql_text(
                    "SELECT chunk_index "
                    "FROM fts_transcripts "
                    "WHERE fts_transcripts MATCH :q "
                    "AND file_id = :fid "
                    "ORDER BY rank "
                    "LIMIT :lim"
                ),
                {"q": fts_query, "fid": file_id, "lim": top_n * 3},
            ).fetchall()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Transcript keyword chunk lookup failed for %s: %s",
            file_id, e,
        )
        return []

    chunk_indices: list[int] = []
    seen: set[int] = set()
    for row in rows:
        try:
            idx = int(row[0])
        except (TypeError, ValueError):
            continue
        if idx in seen:
            continue
        seen = {*seen, idx}
        chunk_indices = [*chunk_indices, idx]
        if len(chunk_indices) >= top_n:
            break

    if not chunk_indices:
        return []

    with get_search_db() as session:
        chunks = (
            session.query(TranscriptChunk)
            .filter(
                TranscriptChunk.file_id == file_id,
                TranscriptChunk.chunk_index.in_(chunk_indices),
            )
            .all()
        )
        by_idx = {c.chunk_index: c for c in chunks}

    return [
        (by_idx[idx].text, by_idx[idx].timestamp_start, by_idx[idx].timestamp_end)
        for idx in chunk_indices
        if idx in by_idx and by_idx[idx].text
    ]


def _fetch_long_summary(file_id: str) -> str | None:
    """Fetch the AI-generated long summary for a file, if available.

    Only returns summaries with ``status='generated'`` (not hidden).
    Returns None when no summary exists — the caller falls back to
    chunk-only context.
    """
    try:
        with get_search_db() as session:
            row = session.execute(
                sql_text(
                    "SELECT long_summary FROM file_summaries "
                    "WHERE file_id = :fid AND status = 'generated'"
                ),
                {"fid": file_id},
            ).fetchone()
            return row[0] if row and row[0] else None
    except Exception as e:  # noqa: BLE001
        logger.warning("Summary lookup failed for %s: %s", file_id, e)
        return None


def _fetch_blip_caption(file_id: str) -> str | None:
    """Fetch the BLIP caption for an image file, if any.

    BLIP captions are stored in the ``embeddings`` table with
    ``embedding_type='blip_caption'`` and the caption text in the
    ``content_preview`` column (see ``app.workers.clip``).

    Returns:
        The caption string, or None if no BLIP row exists.
    """
    with get_search_db() as session:
        rows = (
            session.query(Embedding)
            .filter(
                Embedding.file_id == file_id,
                Embedding.embedding_type == "blip_caption",
            )
            .all()
        )
        if not rows:
            return None
        captions = [r.content_preview for r in rows if r.content_preview]
        if not captions:
            return None
        return " ".join(captions)


# ---------------------------------------------------------------------------
# Per-file context building
# ---------------------------------------------------------------------------


def _format_timestamp(seconds: float) -> str:
    """Format a timestamp as "m:ss" for human-readable locations."""
    total = max(0, int(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _build_metadata_snippets(
    candidate: RetrievedFile,
) -> list[ContextSnippet]:
    """Fallback metadata-only snippets (filename / title / description)."""
    parts: list[str] = []
    if candidate.filename:
        parts = [*parts, f"filename: {candidate.filename}"]
    if candidate.title:
        parts = [*parts, f"title: {candidate.title}"]
    if candidate.description:
        parts = [*parts, f"description: {candidate.description}"]
    if not parts:
        return []
    return [
        ContextSnippet(
            source="metadata",
            text=" | ".join(parts),
            location=None,
        )
    ]


def _cap_snippets(
    snippets: list[ContextSnippet],
    max_chars: int,
) -> tuple[list[ContextSnippet], int]:
    """Enforce the per-file character budget, trimming the last snippet.

    Snippets are kept in order until the running total would exceed
    ``max_chars``. The snippet that crosses the boundary is included
    but clipped and run through ``trim_to_sentence_boundary`` so it
    still ends cleanly.

    Returns:
        ``(kept_snippets, total_chars)`` — total_chars is the exact
        sum of the kept snippet text lengths.
    """
    if max_chars <= 0 or not snippets:
        return (snippets, sum(len(s.text) for s in snippets))

    kept: list[ContextSnippet] = []
    running = 0
    for snippet in snippets:
        remaining = max_chars - running
        if remaining <= 0:
            break
        text = snippet.text
        if len(text) <= remaining:
            kept = [*kept, snippet]
            running += len(text)
            continue
        # This snippet would overflow: clip + sentence-boundary trim.
        clipped = trim_to_sentence_boundary(text[:remaining])
        if clipped:
            kept = [
                *kept,
                ContextSnippet(
                    source=snippet.source,
                    text=clipped,
                    location=snippet.location,
                ),
            ]
            running += len(clipped)
        break

    return (kept, running)


def _overlaps_any(
    start: float,
    end: float,
    covered: list[tuple[float, float]],
) -> bool:
    """Check if a time range overlaps any of the already-covered ranges."""
    return any(s <= end and start <= e for s, e in covered)


def _collect_transcript_snippets(
    candidate: RetrievedFile,
    rag_config: RagConfig,
    query_vector: np.ndarray | None = None,
    keywords: str | None = None,
) -> list[ContextSnippet]:
    """Build transcript snippets for a video / audio file.

    Combines three chunk-selection strategies (mirroring
    ``_collect_document_snippets``):

    1. **Keyword-OR chunks** (listed first): literal keyword matches
       via ``fts_transcripts``, without window expansion. Listed first
       so they win the per-file character budget — terse answer
       passages (e.g. "卒業の理由は膝と腰が…") often sit far from
       the search-matched timestamp and would be crowded out by
       verbose window context if appended last.
    2. **Vector-similar chunks**: nearest embeddings in ``vec_text``
       (``embedding_type='whisper'``), without window expansion.
    3. **Time-window segments** (historical path): each search-matched
       segment expanded by ``transcript_window_seconds`` on each side.

    Strategies 1 and 2 rescue the case where a relevant passage sits
    far from any search-matched timestamp — e.g. the topic is
    introduced at the start but detailed in the middle of the video.

    Deduplication is timestamp-based: chunks are skipped if their time
    range overlaps an already-covered range from a prior strategy.
    """
    snippets: list[ContextSnippet] = []
    # Track covered time ranges for deduplication.
    covered_ranges: list[tuple[float, float]] = []

    # Helper to add non-overlapping chunks from keyword / vector passes.
    def _add_extra_chunks(
        extra: list[tuple[str, float, float]],
        source_tag: str,
    ) -> None:
        nonlocal snippets, covered_ranges
        for text, ts_start, ts_end in extra:
            if not text:
                continue
            if _overlaps_any(ts_start, ts_end, covered_ranges):
                continue
            snippets = [
                *snippets,
                ContextSnippet(
                    source=source_tag,
                    text=text,
                    location=_format_timestamp(ts_start),
                ),
            ]
            covered_ranges = [*covered_ranges, (ts_start, ts_end)]

    top_n = rag_config.transcript_vector_top_n

    # (1) Keyword-OR chunks first (literal anchor words). Terse answer
    # passages rank low in vector space against a wordy question but
    # contain the question's literal anchor words — surface them
    # before window context so they win the per-file char budget.
    if keywords and top_n > 0:
        cleaned = filter_keywords(keywords)
        if cleaned:
            kw_chunks = _fetch_transcript_chunks_by_keyword_or(
                candidate.file_id, cleaned, top_n,
            )
            _add_extra_chunks(kw_chunks, "transcript_keyword")

    # (2) Vector-similar chunks (vocabulary-mismatch rescue).
    if query_vector is not None and top_n > 0:
        vec_chunks = _fetch_transcript_chunks_by_vector(
            candidate.file_id, query_vector, top_n,
        )
        _add_extra_chunks(vec_chunks, "transcript_vector")

    # (3) Time-window around each search-matched segment (historical
    # path). Listed last so keyword/vector results that directly
    # answer the query are not crowded out by verbose surrounding
    # context. Window chunks that overlap keyword/vector ranges are
    # still included — their time ranges won't collide because window
    # dedup checks against the per-chunk ranges above, not the
    # expanded ±window range.
    for segment in candidate.segments:
        if segment.time_range is None:
            continue
        start, end = segment.time_range
        chunks = _fetch_transcript_chunks_around(
            candidate.file_id,
            start,
            end,
            rag_config.transcript_window_seconds,
        )
        if not chunks:
            continue
        text = " ".join(c[0] for c in chunks if c[0])
        if not text:
            continue
        location = _format_timestamp(start)
        snippets = [
            *snippets,
            ContextSnippet(
                source="transcript",
                text=text,
                location=location,
            ),
        ]
        # Record the actual covered range (with window expansion).
        window = rag_config.transcript_window_seconds
        covered_ranges = [
            *covered_ranges,
            (max(0.0, start - window), end + window),
        ]

    return snippets


def _collect_document_snippets(
    candidate: RetrievedFile,
    rag_config: RagConfig,
    query_vector: np.ndarray | None = None,
    keywords: str | None = None,
) -> list[ContextSnippet]:
    """Build text_content snippets for a document file.

    Combines two chunk-selection strategies:

    1. **Vector-similar chunks** (preferred, listed first): when
       ``query_vector`` is provided and ``document_vector_top_n > 0``,
       pull the top-N nearest ``text_content`` chunks of this file
       from ``vec_text``. This handles the vocabulary-mismatch case
       where the file was retrieved via semantic similarity but no
       literal keyword landed on the answer passage.
    2. **Keyword-match chunks**: the historical path — chunks that
       FTS matched, plus their immediate neighbors.

    Vector-selected chunks appear before keyword-selected ones so the
    per-file character budget (enforced downstream in ``_cap_snippets``)
    favours semantically-relevant text over keyword-incidental text.
    Duplicate chunk indices are dropped on the way in.
    """
    snippets: list[ContextSnippet] = []
    seen_indices: set[int] = set()

    def _emit(chunk_idx: int, source_tag: str, expand: bool) -> None:
        nonlocal snippets, seen_indices
        if chunk_idx in seen_indices:
            return
        seen_indices = {*seen_indices, chunk_idx}
        if expand:
            chunks = _fetch_document_chunks_around(
                candidate.file_id, chunk_idx
            )
            if not chunks:
                return
            text = "\n\n".join(c[1] for c in chunks if c[1])
        else:
            # Vector-selected chunks don't need ±1 neighbor padding —
            # vector similarity already picked the right passage, and
            # padding every hit would crowd the per-file budget so
            # only the first-ranked chunk fits. Emit the single chunk
            # text directly so top_n distinct chunks can coexist.
            chunks = _fetch_document_chunks_around(
                candidate.file_id, chunk_idx
            )
            text = next(
                (c[1] for c in chunks if c[0] == chunk_idx and c[1]), ""
            )
        if not text:
            return
        snippets = [
            *snippets,
            ContextSnippet(
                source=source_tag,
                text=text,
                location=f"chunk {chunk_idx}",
            ),
        ]

    # (1a) Literal keyword-OR chunks first. Terse summary passages
    # (e.g. a one-line "5 点で合意" enumeration) rank low in vector
    # space against a wordy question but contain the question's
    # literal anchor words — surface them before vector candidates
    # so they win the per-file char budget. Run against
    # ``filter_keywords(keywords)`` so question-word noise does not
    # dilute the OR set.
    if keywords and rag_config.document_vector_top_n > 0:
        cleaned = filter_keywords(keywords)
        if cleaned:
            for idx in _fetch_document_chunks_by_keyword_or(
                candidate.file_id,
                cleaned,
                rag_config.document_vector_top_n,
            ):
                _emit(idx, "text_content_keyword", expand=False)

    # (1b) Vector-similar chunks next. No ±1 expansion — keep each
    # hit compact so multiple chunks fit under the per-file budget.
    if query_vector is not None and rag_config.document_vector_top_n > 0:
        for idx in _fetch_document_chunks_by_vector(
            candidate.file_id,
            query_vector,
            rag_config.document_vector_top_n,
        ):
            _emit(idx, "text_content_vector", expand=False)

    # (2) Keyword-match chunks with ±1 neighbor padding (historical
    # path). Neighbor padding still matters here because the keyword
    # match may land on a short fragment that needs surrounding prose
    # for the LLM to interpret.
    for segment in candidate.segments:
        for match in segment.matches:
            chunk_idx = match.page if match.page is not None else 0
            _emit(chunk_idx, "text_content", expand=True)

    return snippets


def _collect_image_snippets(
    candidate: RetrievedFile,
) -> list[ContextSnippet]:
    """Build snippets for an image: BLIP caption + metadata."""
    snippets: list[ContextSnippet] = []

    caption = _fetch_blip_caption(candidate.file_id)
    if caption:
        snippets = [
            *snippets,
            ContextSnippet(
                source="clip",
                text=caption,
                location=None,
            ),
        ]

    # Always append metadata for images — the filename / description
    # often carries useful context the model cannot infer from BLIP.
    snippets = [*snippets, *_build_metadata_snippets(candidate)]
    return snippets


def build_file_context(
    candidate: RetrievedFile,
    rag_config: RagConfig,
    query_vector: np.ndarray | None = None,
    keywords: str | None = None,
) -> FileContext:
    """Build a compact context excerpt for a single retrieved file.

    Strategy by file type:

    * ``video`` / ``audio``: transcript chunks around each timestamped
      segment, with a ``transcript_window_seconds`` window.
    * ``document``: FTS text-content chunks around each matched chunk
      index (the match and its immediate neighbors).
    * ``image``: BLIP caption (if present) plus filename + description.
    * other: filename + title + description only.

    The combined snippet text is capped at
    ``rag_config.max_context_chars_per_file``; overflow is trimmed to
    a sentence boundary so the last snippet still ends cleanly.
    """
    file_type = candidate.file_type
    # HvLink files carry host file_type="other" from MIME heuristics but
    # have VTT-derived TranscriptChunks — route them through the transcript
    # path so the LLM sees the subtitles (mirrors summaries._classify_file_type).
    is_hvlink = candidate.mime_type == HVLINK_MIME

    if file_type in ("video", "audio") or is_hvlink:
        snippets = _collect_transcript_snippets(
            candidate,
            rag_config,
            query_vector=query_vector,
            keywords=keywords,
        )
    elif file_type in ("document", "text"):
        snippets = _collect_document_snippets(
            candidate,
            rag_config,
            query_vector=query_vector,
            keywords=keywords,
        )
    elif file_type == "image":
        snippets = _collect_image_snippets(candidate)
    else:
        snippets = []

    # Fallback: if no type-specific snippets were produced, use metadata.
    if not snippets:
        snippets = _build_metadata_snippets(candidate)

    # Prepend AI summary when available. The summary gives the LLM a
    # bird's-eye view of the file, which is essential for "summarize
    # this video/document" queries where chunk-level excerpts cannot
    # cover the full content under the per-file budget. Listed first
    # so it always survives _cap_snippets trimming.
    if file_type in ("video", "audio", "document", "text") or is_hvlink:
        summary = _fetch_long_summary(candidate.file_id)
        if summary:
            snippets = [
                ContextSnippet(
                    source="summary",
                    text=summary,
                    location=None,
                ),
                *snippets,
            ]

    kept, total_chars = _cap_snippets(
        snippets, rag_config.max_context_chars_per_file
    )

    return FileContext(
        file_id=candidate.file_id,
        filename=candidate.filename,
        drive=candidate.drive,
        file_type=candidate.file_type,
        title=candidate.title,
        description=candidate.description,
        snippets=tuple(kept),
        total_chars=total_chars,
    )


# ---------------------------------------------------------------------------
# Multi-file assembly with total budget enforcement
# ---------------------------------------------------------------------------


def assemble_contexts(
    candidates: list[RetrievedFile],
    rag_config: RagConfig,
    query: str | None = None,
    keywords: str | None = None,
) -> list[FileContext]:
    """Build contexts for all candidates, enforcing the total budget.

    Candidates are processed in score-descending order. When adding a
    file's context would push the cumulative character total past
    ``max_total_context_chars``, the remaining lower-scored files are
    dropped. Files with zero snippets are filtered out of the result.

    When ``query`` is provided and any document file has
    ``document_vector_top_n > 0`` configured, the query is embedded
    once here and the resulting vector is threaded into document
    context builders so chunk selection can use semantic similarity
    in addition to FTS keyword matches. Embedding failures fall back
    to keyword-only behaviour; the helper never raises.
    """
    if not candidates:
        return []

    # Embed the query once (rather than per-file) when it's worth
    # doing — i.e., the vector pass is enabled and at least one
    # candidate is a document or transcript file.
    query_vector: np.ndarray | None = None
    _needs_vector = (
        rag_config.document_vector_top_n > 0
        and any(c.file_type in ("document", "text") for c in candidates)
    ) or (
        rag_config.transcript_vector_top_n > 0
        and any(
            c.file_type in ("video", "audio") or c.mime_type == HVLINK_MIME
            for c in candidates
        )
    )
    if query and _needs_vector:
        try:
            query_vector = embed_query(query)
        except Exception as e:  # noqa: BLE001 - fail soft, don't break RAG
            logger.warning(
                "Query embedding failed in context builder: %s", e
            )
            query_vector = None

    # Sort by score desc — highest-scored files get the budget first.
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)

    results: list[FileContext] = []
    running_total = 0
    budget = rag_config.max_total_context_chars

    # Use transformed keywords when available; otherwise fall back
    # to the raw query so the keyword-OR FTS pass still has tokens
    # to work with (filter_keywords strips question words inside).
    effective_keywords = keywords or query

    for candidate in ordered:
        ctx = build_file_context(
            candidate,
            rag_config,
            query_vector=query_vector,
            keywords=effective_keywords,
        )
        if not ctx.snippets:
            continue
        if running_total + ctx.total_chars > budget:
            # This context would overflow the shared budget — drop it
            # and every lower-scored candidate. Keeping partial files
            # would force a second round of trimming and make total_chars
            # harder to reason about.
            break
        results = [*results, ctx]
        running_total += ctx.total_chars

    return results
