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
from dataclasses import dataclass

from sqlalchemy import text as sql_text

from app.config import RagConfig
from app.database import get_search_db
from app.models import Embedding, TranscriptChunk
from app.rag.retriever import RetrievedFile
from app.search import MatchInfo, SegmentGroup
from app.text_utils import trim_to_sentence_boundary

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


def _collect_transcript_snippets(
    candidate: RetrievedFile,
    rag_config: RagConfig,
) -> list[ContextSnippet]:
    """Build transcript snippets for a video / audio file."""
    snippets: list[ContextSnippet] = []
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
    return snippets


def _collect_document_snippets(
    candidate: RetrievedFile,
) -> list[ContextSnippet]:
    """Build text_content snippets for a document file."""
    snippets: list[ContextSnippet] = []
    seen_indices: set[int] = set()
    for segment in candidate.segments:
        for match in segment.matches:
            # The `page` field on MatchInfo carries the chunk index for
            # document results (see app.search). Fall back to 0 when
            # unset so we still fetch something.
            chunk_idx = match.page if match.page is not None else 0
            if chunk_idx in seen_indices:
                continue
            seen_indices = {*seen_indices, chunk_idx}
            chunks = _fetch_document_chunks_around(
                candidate.file_id, chunk_idx
            )
            if not chunks:
                continue
            text = "\n\n".join(c[1] for c in chunks if c[1])
            if not text:
                continue
            snippets = [
                *snippets,
                ContextSnippet(
                    source="text_content",
                    text=text,
                    location=f"chunk {chunk_idx}",
                ),
            ]
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

    if file_type in ("video", "audio"):
        snippets = _collect_transcript_snippets(candidate, rag_config)
    elif file_type in ("document", "text"):
        snippets = _collect_document_snippets(candidate)
    elif file_type == "image":
        snippets = _collect_image_snippets(candidate)
    else:
        snippets = []

    # Fallback: if no type-specific snippets were produced, use metadata.
    if not snippets:
        snippets = _build_metadata_snippets(candidate)

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
) -> list[FileContext]:
    """Build contexts for all candidates, enforcing the total budget.

    Candidates are processed in score-descending order. When adding a
    file's context would push the cumulative character total past
    ``max_total_context_chars``, the remaining lower-scored files are
    dropped. Files with zero snippets are filtered out of the result.
    """
    if not candidates:
        return []

    # Sort by score desc — highest-scored files get the budget first.
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)

    results: list[FileContext] = []
    running_total = 0
    budget = rag_config.max_total_context_chars

    for candidate in ordered:
        ctx = build_file_context(candidate, rag_config)
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
