"""RAG orchestration layer.

``answer_question`` is the single public entry point. It runs the
full pipeline:

1. ``retrieve_candidates`` — hybrid search + Internal API access filter.
2. ``assemble_contexts``   — build per-file snippets under budgets.
3. LLM call                — system + user prompt via generate_json.
4. ``parse_answer``        — validate shape, drop hallucinated file_ids.
5. Assemble ``AnswerResponse`` with citations + sources + timing.

The function short-circuits when retrieval is empty (no LLM call)
and gracefully degrades when the LLM returns unparseable output
(answer=None but sources populated so the UI can still show them).
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.dependencies import get_llm_client
from app.rag.context import assemble_contexts
from app.rag.parser import Citation, parse_answer
from app.rag.prompt import build_system_prompt, build_user_prompt
from app.rag.retriever import RetrievedFile, retrieve_candidates

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnswerResponse:
    """The full RAG answer payload returned to the router.

    ``citations`` and ``sources`` use plain dicts instead of Pydantic
    models so this module stays import-cheap for tests that stub the
    entire service behind an ``AsyncMock``. The router layer converts
    these dicts to ``AnswerResponseModel`` on the way out.

    ``retrieved_count`` reflects how many files reached the LLM context
    builder — i.e. *after* the Internal API access filter dropped files
    the caller cannot see. If the raw hybrid search returned 10 files
    but the caller only had access to 3, ``retrieved_count`` is 3.
    """

    query: str
    answer: str | None
    citations: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    retrieved_count: int
    took_ms: int


def _segment_location_for(
    file_id: str,
    candidates: list[RetrievedFile],
) -> str | None:
    """Best-effort timestamp/page label for a citation's source file.

    Uses the first segment of the matching retrieved file to produce
    a human-readable location. For video/audio this is ``m:ss``; for
    documents we use the chunk/page number from ``MatchInfo.page``.
    """
    for candidate in candidates:
        if candidate.file_id != file_id:
            continue
        for segment in candidate.segments:
            if segment.time_range is not None:
                seconds = int(max(0.0, segment.time_range[0]))
                return f"{seconds // 60}:{seconds % 60:02d}"
            for match in segment.matches:
                if match.page is not None:
                    return f"page {match.page}"
        return None
    return None


def _to_citation_dict(
    citation: Citation,
    candidates: list[RetrievedFile],
) -> dict[str, Any]:
    """Convert a parsed Citation into the router-ready dict shape.

    Enriches the raw LLM fields with the drive / filename / file_type
    from the retriever so the frontend can render a citation card
    without a second lookup.
    """
    source_file: RetrievedFile | None = None
    for candidate in candidates:
        if candidate.file_id == citation.file_id:
            source_file = candidate
            break

    if source_file is None:
        # Defensive: parser already dropped unknown file_ids, but guard
        # anyway so a race between parser and response assembly cannot
        # produce a KeyError. Return a minimal dict with empty strings.
        return {
            "file_id": citation.file_id,
            "drive": "",
            "filename": "",
            "file_type": "",
            "quote": citation.quote,
            "relevance": citation.relevance,
            "segment_location": None,
        }

    return {
        "file_id": citation.file_id,
        "drive": source_file.drive,
        "filename": source_file.filename,
        "file_type": source_file.file_type,
        "quote": citation.quote,
        "relevance": citation.relevance,
        "segment_location": _segment_location_for(
            citation.file_id, candidates
        ),
    }


def _to_source_dict(candidate: RetrievedFile) -> dict[str, Any]:
    """Convert a RetrievedFile into the router-ready source dict shape."""
    return {
        "file_id": candidate.file_id,
        "drive": candidate.drive,
        "filename": candidate.filename,
        "file_type": candidate.file_type,
        "score": candidate.score,
        "match_types": list(candidate.match_types),
    }


async def answer_question(
    query: str,
    hv_token: str | None,
    top_k: int | None = None,
    file_type: str | None = None,
    drive: str | None = None,
) -> AnswerResponse:
    """Run the full RAG pipeline and return an ``AnswerResponse``.

    The function never raises on LLM failure — it returns an answer
    with ``answer=None`` but populated ``sources`` so the caller can
    at least show the user which files were considered.
    """
    rag_config = settings.rag
    effective_top_k = top_k if top_k is not None else rag_config.top_k

    start = time.monotonic()

    # Stage 1: retrieve + access filter.
    candidates = await retrieve_candidates(
        query=query,
        top_k=effective_top_k,
        hv_token=hv_token,
        file_type=file_type,
        drive=drive,
    )

    if not candidates:
        return AnswerResponse(
            query=query,
            answer=None,
            citations=[],
            sources=[],
            retrieved_count=0,
            took_ms=int((time.monotonic() - start) * 1000),
        )

    # Stage 2: build per-file contexts under budget.
    contexts = assemble_contexts(candidates, rag_config)

    # Stage 3: LLM call.
    llm = get_llm_client()
    system_prompt = build_system_prompt(settings.llm.output_language)
    user_prompt = build_user_prompt(query, contexts)

    raw = await llm.generate_json(
        system_prompt,
        user_prompt,
        max_tokens_override=rag_config.max_tokens,
    )

    # Stage 4: parse + validate citations against the retrieved set.
    allowed = frozenset(c.file_id for c in candidates)
    parsed = parse_answer(raw, allowed)

    sources = [_to_source_dict(c) for c in candidates]

    if parsed is None:
        # LLM returned unparseable output. Still surface the retrieval
        # so the UI can offer a retry / "we found these files" view.
        return AnswerResponse(
            query=query,
            answer=None,
            citations=[],
            sources=sources,
            retrieved_count=len(candidates),
            took_ms=int((time.monotonic() - start) * 1000),
        )

    citations = [
        _to_citation_dict(c, candidates) for c in parsed.citations
    ]

    return AnswerResponse(
        query=query,
        answer=parsed.answer,
        citations=citations,
        sources=sources,
        retrieved_count=len(candidates),
        took_ms=int((time.monotonic() - start) * 1000),
    )
