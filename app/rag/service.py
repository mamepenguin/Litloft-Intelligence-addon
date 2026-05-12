"""RAG orchestration layer.

Two public entry points:

* ``answer_question`` — legacy non-streaming path. Runs the full
  pipeline and returns an ``AnswerResponse`` in one shot. Still used
  by internal tests and tooling.
* ``stream_answer`` — SSE streaming path used by ``POST /ask``. Yields
  typed ``AnswerEvent`` instances (``keywords`` → zero or more
  ``answer_chunk`` → ``citations`` → ``done``). The router layer is
  responsible for encoding these events to ``text/event-stream``.

Both paths share the same retrieval + context + anti-hallucination
stages. The only differences are that ``stream_answer``:

1. Runs ``transform_query`` separately so it can emit a ``keywords``
   event to the client before the search even starts (fast feedback).
2. Uses ``LLMClient.generate_stream`` to pipe answer tokens to the
   client as they arrive.
3. Buffers the full answer text in parallel with streaming so the
   terminal citations event can parse+validate against the full JSON
   payload the LLM returned, not a mid-stream fragment.
"""

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from app.config import settings
from app.dependencies import get_llm_client
from app.rag.agentic import (
    AgenticAnswer,
    agentic_capability_supported,
    compute_token_budget,
    get_agentic_model_entry,
    run_agentic_loop,
)
from app.rag.agentic_types import AgenticTelemetry
from app.rag.answer_stream import (
    AnswerSanitizer,
    AnswerStreamExtractor,
    CitationStreamExtractor,
)
from app.rag.category_expander import expand_category
from app.rag.clue_generator import fetch_long_summaries, generate_clues
from app.rag.coarse_retriever import ShortlistResult, coarse_retrieve
from app.rag.context import assemble_contexts
from app.rag.history_client import fetch_viewer_history
from app.rag.parser import Citation, _parse_citation, parse_answer
from app.rag.prompt import build_system_prompt, build_user_prompt
from app.rag.query_decomposer import DecomposedQuery, TimeRange, decompose_query
from app.rag.query_transform import (
    RequiredTerm,
    transform_query,
    transform_query_structured,
)
from app.rag.retriever import (
    RetrievedFile,
    _filter_file_ids_via_internal_api,
    retrieve_candidates,
    retrieve_with_keywords,
)

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
    # Populated only when the agentic loop ran (Phase 1.C). Legacy
    # single-turn answers leave this as ``None`` so the eval harness can
    # tell "ran legacy" apart from "ran agentic with no tool calls".
    agentic_telemetry: AgenticTelemetry | None = None


_LOCATION_MARKER_RE = re.compile(r"^\d+:\d{2,}$|^page\s+\d+$|^chunk\s+\d+$", re.IGNORECASE)


def _is_location_marker(loc: str) -> bool:
    """Return True when ``loc`` is a recognised short location marker.

    Accepts the formats the system prompt teaches the LLM to emit:
    ``m:ss`` (video timestamp), ``page N`` (document), ``chunk N``
    (vector-only snippet).  Any longer string (a verbatim sentence from
    the transcript) returns False so the caller falls through to the
    candidate-segment timestamp lookup instead.
    """
    return bool(_LOCATION_MARKER_RE.match(loc.strip()))


def _segment_location_for(
    file_id: str,
    candidates: list[RetrievedFile],
    quote: str = "",
    contexts: list | None = None,
) -> str | None:
    """Best-effort timestamp/page label for a citation's source file.

    Drills down to the individual MatchInfo whose text best overlaps
    the LLM-provided ``quote``. This matters for long videos where the
    indexer aggregates many transcript chunks into one wide
    SegmentGroup (e.g. [95s, 478s]); using the group's start would
    point to the segment opening rather than the cited moment.

    Output: ``m:ss`` for time-based matches, ``page N`` for document
    matches. Falls back to the first match in the first segment when
    quote is empty or has no overlap with any match text.

    The ``MatchInfo`` path (candidate.segments) is preferred because
    it carries fine-grained per-match timestamps — for long videos the
    enclosing ``SegmentGroup`` covers many seconds, and the group's
    start is a much worse location than the individual match's
    ``timestamp_start``. The ``contexts`` snippet path is used only as
    a fallback when the candidate has no matches (e.g. vector-only
    document chunks have no ``MatchInfo`` entry and their location
    lives on the assembled snippet as ``"chunk N"``).
    """
    for candidate in candidates:
        if candidate.file_id != file_id:
            continue
        if not candidate.segments:
            break

        best_match = _pick_match_for_quote(candidate.segments, quote)
        if best_match is None:
            seg0 = candidate.segments[0]
            if seg0.time_range is not None:
                seconds = int(max(0.0, seg0.time_range[0]))
                return f"{seconds // 60}:{seconds % 60:02d}"
            break

        if best_match.timestamp_start is not None:
            seconds = int(max(0.0, best_match.timestamp_start))
            return f"{seconds // 60}:{seconds % 60:02d}"
        if best_match.page is not None:
            return f"page {best_match.page}"
        break

    # Fallback: vector-selected document chunks have no MatchInfo entry.
    # Locate the snippet whose body contains the quote and return its
    # "chunk N" label.
    if contexts and quote and quote.strip():
        snippet_location = _snippet_location_for_quote(
            file_id, contexts, quote
        )
        if snippet_location is not None:
            return snippet_location

    return None


def _snippet_location_for_quote(
    file_id: str,
    contexts: list,
    quote: str,
) -> str | None:
    """Find the snippet whose text best overlaps the LLM-provided quote.

    Walks the snippets of the ``FileContext`` matching ``file_id`` and
    returns the ``location`` of the snippet with the largest overlap
    with ``quote``. Returns ``None`` when the file has no contexts or
    no snippet overlaps the quote at all — the caller then falls back
    to the candidate-segments path.
    """
    for ctx in contexts:
        if getattr(ctx, "file_id", None) != file_id:
            continue
        best_score = 0
        best_location: str | None = None
        for snippet in getattr(ctx, "snippets", ()):
            text = getattr(snippet, "text", "") or ""
            if not text:
                continue
            score = _quote_overlap_score(quote, text)
            if score > best_score:
                best_score = score
                best_location = getattr(snippet, "location", None)
        # Require a real overlap (length * 1000 term) before trusting
        # the location — pure character-set overlap is too noisy for
        # long snippets and would pin unrelated quotes to arbitrary
        # chunks.
        if best_score >= 3_000 and best_location:
            return best_location
        return None
    return None


def _pick_match_for_quote(segments, quote: str):
    """Return the single MatchInfo whose text best overlaps ``quote``.

    Iterates every MatchInfo across every segment and scores by
    substring containment of the quote in the match text (with
    char-overlap tiebreak). Returns the first match by retriever order
    when ``quote`` is empty or no overlap is found, or ``None`` when
    the candidate has no matches at all.
    """
    first_match = None
    for segment in segments:
        for match in segment.matches:
            if first_match is None:
                first_match = match
            if not match.text:
                continue

    if not quote or not quote.strip():
        return first_match

    quote_norm = quote.strip()
    best = first_match
    best_score = -1
    for segment in segments:
        for match in segment.matches:
            if not match.text:
                continue
            score = _quote_overlap_score(quote_norm, match.text)
            if score > best_score:
                best_score = score
                best = match
    return best


def _quote_overlap_score(quote: str, text: str) -> int:
    """Substring-containment score; tiebreaker = shared char count.

    Walks decreasing-length windows of the quote until one is found in
    the text. The returned score is dominated by window length (×1000)
    so a 10-char substring always beats a 5-char one regardless of how
    many random characters happen to overlap.
    """
    max_window = min(len(quote), 64)
    for size in range(max_window, 2, -1):
        for start in range(0, len(quote) - size + 1):
            window = quote[start : start + size]
            if window in text:
                return size * 1000 + len(set(quote) & set(text))
    return len(set(quote) & set(text))


def _quote_from_contexts(
    file_id: str,
    contexts: list | None,
    location: str = "",
    max_chars: int = 200,
) -> str:
    """Pull a representative quote from the already-retrieved contexts.

    The LLM no longer generates quote strings (see build_system_prompt
    rationale). Instead, we display the snippet text as the quote.

    When ``location`` is provided (e.g. "0:45", "page 3"), we look for
    the snippet whose ``location`` field matches and return its text —
    this lets the LLM cite the same file at multiple points and each
    citation gets the correct excerpt. Falls back to the first snippet
    when no location is given or none matches.
    """
    if not contexts:
        return ""
    for ctx in contexts:
        if getattr(ctx, "file_id", None) != file_id:
            continue
        snippets = getattr(ctx, "snippets", None)
        if not snippets:
            return ""

        chosen = snippets[0]
        if location:
            for snippet in snippets:
                snippet_loc = snippet.location or ""
                # Accept exact match or substring match so
                # "0:45" matches a snippet tagged "0:45" even when
                # the location happens to be embedded in a longer
                # label like "transcript @ 0:45".
                if snippet_loc == location or (
                    snippet_loc and location in snippet_loc
                ):
                    chosen = snippet
                    break

        text = chosen.text.strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        return text
    return ""


def _to_citation_dict(
    citation: Citation,
    candidates: list[RetrievedFile],
    contexts: list | None = None,
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

    # Quote is populated from the actual context snippet when the LLM
    # didn't provide one (the new default) — this is both faster and
    # safer than letting the model generate quotes freely. The LLM's
    # optional ``location`` lets us pick the right snippet when the
    # same file is cited at multiple points.
    quote = citation.quote or _quote_from_contexts(
        citation.file_id, contexts, location=citation.location
    )

    # Prefer the LLM-provided location when it's a recognised marker
    # format (m:ss / page N). Local LLMs routinely ignore the format
    # instruction and emit a verbatim sentence instead; in that case we
    # fall through to the candidate-segment lookup so the frontend gets
    # a seekable timestamp rather than a ?highlight= anchor.
    raw_loc = citation.location or ""
    if raw_loc and _is_location_marker(raw_loc):
        segment_location = raw_loc
    else:
        segment_location = _segment_location_for(
            citation.file_id, candidates, quote, contexts=contexts
        ) or (raw_loc if raw_loc else None)

    if source_file is None:
        # Defensive: parser already dropped unknown file_ids, but guard
        # anyway so a race between parser and response assembly cannot
        # produce a KeyError. Return a minimal dict with empty strings.
        return {
            "file_id": citation.file_id,
            "drive": "",
            "filename": "",
            "file_type": "",
            "quote": quote,
            "relevance": citation.relevance,
            "segment_location": segment_location or None,
        }

    return {
        "file_id": citation.file_id,
        "drive": source_file.drive,
        "filename": source_file.filename,
        "file_type": source_file.file_type,
        "quote": quote,
        "relevance": citation.relevance,
        "segment_location": segment_location,
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


# Standard IR Reciprocal Rank Fusion constant. 60 is the canonical
# value across the literature (Cormack/Clarke 2009) and is what the
# legacy single-mode RRF combiner in ``app.search`` already uses.
# Promoting this to a module-level constant makes it greppable when
# tuning multi-query expansion weights against eval baselines.
_RRF_K_DEFAULT = 60


def _rrf_merge_candidates(
    per_clue: list[list[RetrievedFile]],
    *,
    top_k: int,
    rrf_k: int = _RRF_K_DEFAULT,
    weights: list[float] | None = None,
) -> list[RetrievedFile]:
    """Reciprocal Rank Fusion across per-clue candidate lists.

    Each clue's ranked list contributes ``weight / (rrf_k + rank)`` to
    a file's combined score. The standard IR constant ``rrf_k=60``
    gives a smooth blend without letting top-1 of any single clue
    dominate.

    ``weights`` (optional, same length as ``per_clue``) lets callers
    bias the merge toward higher-priority lists — used by Find's
    category-expansion path so the LLM's leading terms ("料理" /
    "レシピ") outweigh the trailing tail ("ディナー" / "盛り付け")
    that drift further from the user's intent. Default ``None``
    treats all lists equally (legacy behaviour).

    File metadata (title, segments, etc.) is taken from the first
    list that surfaced the file — clues are run with the same
    scope/filters so the metadata is functionally identical across
    lists, and picking the first encountered keeps the merge
    deterministic without paying for a full segment-level merge that
    the LLM context builder would just down-trim later anyway.
    """
    if weights is not None and len(weights) != len(per_clue):
        raise ValueError(
            "weights length must match per_clue length "
            f"(got {len(weights)} vs {len(per_clue)})"
        )

    # Local accumulator dicts — the project rule mandates immutability,
    # but rebuilding either dict per row is O(n²) and the rule's
    # rationale is shared-state safety, which doesn't apply to a
    # function-scoped accumulator that never escapes. Keep the
    # iterative form, but document why so future readers don't
    # "fix" it into a comprehension that pessimises the merge.
    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievedFile] = {}
    for idx, results in enumerate(per_clue):
        weight = weights[idx] if weights is not None else 1.0
        for rank, candidate in enumerate(results, start=1):
            scores[candidate.file_id] = (
                scores.get(candidate.file_id, 0.0)
                + weight / (rrf_k + rank)
            )
            if candidate.file_id not in first_seen:
                first_seen[candidate.file_id] = candidate

    ordered = sorted(
        first_seen.values(),
        key=lambda c: scores[c.file_id],
        reverse=True,
    )
    return ordered[:top_k]


def _category_expansion_weights(n: int) -> list[float]:
    """Linear-decay weights for ``n`` category-expansion terms.

    ``expand_category`` returns terms in LLM-emitted order with the
    user's original ``semantic_query`` always at index 0 (see
    ``category_expander.py``). The order is "most-relevant first" by
    construction, so per-term RRF should weigh head terms more than
    tail terms — otherwise the broad expansion ("ディナー" /
    "盛り付け") accumulates rank-30 noise from unrelated files and
    can outvote a head term ("レシピ") that strongly surfaces the
    real cooking content.

    Decay: ``weight_i = 1.0 - 0.5 * (i / (n - 1))`` for ``n >= 2``.
    Head term weight 1.0, tail term weight 0.5. Smooth enough not to
    erase tail terms entirely (we still want their signal when the
    head is ambiguous), aggressive enough to prevent tail-driven
    rank inversions for queries like "料理に関する動画" where the
    user clearly cares about the head concept.
    """
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    return [1.0 - 0.5 * (i / (n - 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# Personal-history pre-scope (spec §4.2 Stages A + B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PersonalHistoryResolution:
    """Resolved Stage A + B output for one Ask request.

    Three observable shapes:

    * ``decomposed is None`` — feature off, no viewer, or empty query.
      The caller should run the legacy retrieval path verbatim.
    * ``file_ids is None`` — Stage A produced a structured query but
      Stage B was bypassed (no personal signal, or scope=none).
      The caller may still surface ``decomposed`` over SSE for UI
      transparency but should not narrow retrieval.
    * ``file_ids is not None`` — Stage B ran. The empty-list case is
      governed by ``fallback_when_empty``: ``"strict"`` short-circuits
      to "該当なし"; ``"graceful"`` drops the filter and runs legacy
      retrieval. The caller decides which branch to take.
    """

    decomposed: DecomposedQuery | None
    file_ids: list[str] | None
    short_circuit: bool = False  # strict mode + empty file_ids


def _drop_required_overlapping_with_expansion(
    required: tuple[RequiredTerm, ...] | None,
    semantic_query: str,
) -> tuple[RequiredTerm, ...] | None:
    """Drop ``required`` terms whose canonical is contained in
    ``semantic_query`` (case-insensitive substring match).

    Architecture rationale: ``category_expansion`` exists precisely
    because the broad ``semantic_query`` token is not a discriminating
    anchor — multi-query fan-out replaces hard filtering with softer
    retrieval over surface variants. Keeping that same token as a
    ``required`` hard filter while expanding it short-circuits the
    expansion: every per-term retrieve still requires the literal
    ``semantic_query`` to appear, so files that match a synonym
    ("レシピ" / "弁当") but not the original ("料理") get filtered out,
    while files that mention the original token in passing survive —
    the opposite of what the user wants.

    Proper-noun anchors that are *not* in ``semantic_query`` (e.g.
    "東福寺" in "東福寺の通天橋") survive because they don't overlap.
    Callers must invoke this only when expansion is actually being
    applied (``len(category_expansion) >= 2``); the single-keyword
    path keeps ``required`` intact so proper-noun-only queries still
    benefit from the hard filter.
    """
    if not required:
        return None
    sq = semantic_query.strip().lower()
    if not sq:
        return required
    survivors = tuple(
        t for t in required if t.canonical.strip().lower() not in sq
    )
    return survivors or None


async def _resolve_category_expansion(
    decomposed: DecomposedQuery | None,
) -> list[str]:
    """Expand the decomposed semantic_query when Stage C is enabled.

    Returns a list of surface forms suitable for multi-query
    retrieval. Empty list means "Stage C did not contribute" — caller
    should fall back to the single-keyword path.

    Skipped (returns ``[]``) when:
    * the feature is disabled in config,
    * Stage A produced no decomposition (None),
    * the decomposed semantic_query is empty (the user asked something
      like "今月観てない動画" with no concept), or
    * the LLM expansion collapsed to ``[semantic_query]`` — in that
      case multi-query gives no benefit over the single-keyword path
      that the caller will already run.
    """
    cfg = settings.rag.category_expansion
    if not cfg.enabled or decomposed is None:
        return []
    if not decomposed.semantic_query.strip():
        return []
    terms = await expand_category(
        decomposed.semantic_query, max_terms=cfg.max_terms
    )
    # Single-element fallback (e.g. LLM disabled) is the same as the
    # legacy keyword path; emitting a ``category_expanded`` event for
    # it would lie about Stage C having added value.
    if len(terms) <= 1:
        return []
    return terms


async def _resolve_personal_history(
    *,
    query: str,
    viewer_id: str | None,
    drive: str | None,
) -> _PersonalHistoryResolution:
    """Run Stages A + B and return the file_id_scope for the retriever.

    Returns ``_PersonalHistoryResolution`` describing what the streaming
    /non-streaming caller should do with the result. The ``file_ids``
    field is what eventually gets passed to ``retrieve_with_keywords``
    as ``file_id_scope`` — when present and non-empty.
    """
    cfg = settings.rag.personal_history
    if not cfg.enabled or not viewer_id or not drive:
        return _PersonalHistoryResolution(decomposed=None, file_ids=None)

    decomposed = await decompose_query(
        query, max_lookback_days=cfg.max_lookback_days
    )
    if not decomposed.has_personal_signal:
        # Stage A succeeded but the user did not ask anything personal.
        # Surface the decomposition (callers may want the file_type_hint
        # / time_range echoed in SSE) but skip the history fetch.
        return _PersonalHistoryResolution(
            decomposed=decomposed, file_ids=None
        )

    file_ids = await fetch_viewer_history(
        viewer_id=viewer_id,
        drive=drive,
        after=decomposed.time_range.after,
        before=decomposed.time_range.before,
        kind=decomposed.personal_scope,  # type: ignore[arg-type]
    )

    if not file_ids:
        if cfg.fallback_when_empty == "strict":
            return _PersonalHistoryResolution(
                decomposed=decomposed,
                file_ids=[],
                short_circuit=True,
            )
        # Graceful: drop the filter so the user still gets *some*
        # answer rather than a brittle "該当なし". The decomposition
        # is still surfaced over SSE so the UI can hint that the
        # personal narrowing was attempted but yielded nothing.
        return _PersonalHistoryResolution(
            decomposed=decomposed, file_ids=None
        )

    return _PersonalHistoryResolution(
        decomposed=decomposed, file_ids=file_ids
    )


async def _run_hierarchical_retrieval(
    *,
    query: str,
    keywords: str,
    drive: str | None,
    lit_token: str | None,
    file_type: str | None,
    top_k: int,
    original_query: str | None = None,
    pre_scope_file_ids: list[str] | None = None,
    required: tuple[RequiredTerm, ...] | None = None,
) -> tuple[
    list[RetrievedFile],
    "ShortlistResult | None",
    "tuple[str, ...] | None",
]:
    """Run Stage 1 (coarse) + Stage 2 (clue) + Stage 3 (fine) retrieval.

    Returns ``(candidates, shortlist, clues)``. ``shortlist`` and
    ``clues`` are ``None`` whenever the hierarchical pipeline was
    bypassed (config off, drive missing, small drive, low confidence,
    empty shortlist, or the entire shortlist is access-blocked for the
    caller). The streaming path uses these to decide whether to emit
    the corresponding SSE events — bypassed paths must NOT emit them
    because doing so would lie about how the retrieval ran AND, more
    critically, would leak file_ids the caller is not authorised to see
    (a protected drive that's locked must not appear in any response —
    see ``design-decisions.md`` "保護ドライブが locked の場合は API
    応答から完全除外する").

    The returned ``ShortlistResult`` (when not None) carries the
    **access-filtered** file_ids — the same set that was forwarded to
    ``retrieve_with_keywords`` as ``file_id_scope``. ``clues`` contains
    the actual keyword strings that were run; on clue generation
    failure this collapses to a single-entry tuple holding the
    legacy keyword query so downstream callers / observability stay
    consistent.

    Clue dispatch (spec §4.2 Stage 2 / §7.4): once the shortlist is
    confirmed and access-filtered, generate up to
    ``cfg.clue_count`` independent search queries from the user's
    natural-language question + the shortlist's AI summaries, run each
    clue concurrently against ``retrieve_with_keywords`` under the
    same shortlist scope, and merge the per-clue ranked lists with
    Reciprocal Rank Fusion. ``clue_count <= 1`` collapses gracefully
    to a single scoped retrieve identical to Phase 2 behaviour.

    The merge fallback (spec §7.4): when the scoped retrieval returns
    fewer than two candidates AND ``fallback_full_search`` is set, we
    also run an unscoped pass and union the results, preserving the
    scoped order first. This protects pinpoint factual queries whose
    answer chunk lives in a file the AI summary did not foreground.

    Composition with personal_history (spec
    `2026-04-29-intelligence-ask-hierarchical-personal-history-composition.md`):
    when ``pre_scope_file_ids`` is set, the coarse shortlist is
    intersected with the personal-history file_ids (preserving the
    coarse rank order — coarse carries the score signal, pre_scope is
    just a set membership filter). The small-drive guard is skipped
    because pre_scope is already much smaller than the drive corpus.
    The ``coarse_score_threshold`` check is re-evaluated against the
    *post-intersection* top score: a drive whose strongest summary
    matches were never viewed by the user is still allowed to bypass
    on low confidence (the rationale is "無関係なファイルしか先週観て
    ないなら narrow しない方が良い" — better to widen than narrow on
    a weak intersected match). On bypass (empty intersection / low
    confidence / access-filter empty), the fallback retrieve is scoped
    to ``pre_scope_file_ids`` rather than unscoped — losing the user's
    personal scope just because the summary-narrowing didn't help
    would defeat the point of the composition. Note the bypass path
    runs only the single-keyword pass; Stage C ``category_expansion``
    would have run on the personal-history-only branch but is skipped
    here for now (see follow-up M2 in the spec).

    ``pre_scope_file_ids`` is trusted to be drive-bounded by the
    caller (in production, the host's ``viewer-history`` Internal API
    enforces this; ``retrieve_with_keywords`` re-applies the access
    filter unconditionally as defense-in-depth).
    """
    cfg = settings.rag.hierarchical
    semantic = original_query if original_query is not None else query

    # Fallback scope used by every bypass branch. Composed callers
    # carry their personal-history scope through; legacy callers fall
    # back to drive-wide retrieval as before.
    bypass_scope: list[str] | None = pre_scope_file_ids

    if not cfg.enabled or not drive:
        # Hierarchical disabled or drive missing — strict legacy path.
        candidates = await retrieve_with_keywords(
            keywords=keywords,
            top_k=top_k,
            lit_token=lit_token,
            file_type=file_type,
            drive=drive,
            original_query=semantic,
            file_id_scope=bypass_scope,
            required=required,
        )
        return candidates, None, None

    shortlist = await coarse_retrieve(
        query=query,
        drive=drive,
        top_k=cfg.coarse_top_k,
    )

    # Project the coarse shortlist onto the personal-history scope when
    # composing. The intersection preserves coarse rank order — coarse
    # carries the score signal, pre_scope is just set membership.
    if pre_scope_file_ids is not None:
        scope_set = set(pre_scope_file_ids)
        intersected_pairs = [
            (fid, score)
            for fid, score in zip(shortlist.file_ids, shortlist.scores)
            if fid in scope_set
        ]
        intersected_ids = tuple(fid for fid, _ in intersected_pairs)
        intersected_scores = tuple(score for _, score in intersected_pairs)
        shortlist = ShortlistResult(
            file_ids=intersected_ids,
            scores=intersected_scores,
            top_score=intersected_scores[0] if intersected_scores else 0.0,
            drive_file_count=shortlist.drive_file_count,
        )

    # Bypass conditions — small drive, low confidence, empty shortlist.
    # Each is logged at DEBUG so an operator running with verbose
    # logging can see exactly why scoping was skipped.
    bypass_reason: str | None = None
    # ``min_drive_files_for_shortlist`` protects against meaningless
    # shortlists on tiny corpora. When pre_scope is set, the corpus IS
    # the pre_scope (already much smaller than the drive) — the guard
    # would always fire and defeat composition, so skip it.
    if (
        pre_scope_file_ids is None
        and shortlist.drive_file_count < cfg.min_drive_files_for_shortlist
    ):
        bypass_reason = "small_drive"
    elif not shortlist.file_ids:
        bypass_reason = (
            "empty_intersection"
            if pre_scope_file_ids is not None
            else "empty_shortlist"
        )
    elif shortlist.top_score < cfg.coarse_score_threshold:
        bypass_reason = "low_confidence"

    if bypass_reason is not None:
        logger.debug(
            "hierarchical retrieval bypass: reason=%s drive=%s "
            "drive_file_count=%d top_score=%.4f shortlist_size=%d "
            "pre_scope_size=%s",
            bypass_reason, drive, shortlist.drive_file_count,
            shortlist.top_score, len(shortlist.file_ids),
            "n/a" if pre_scope_file_ids is None else len(pre_scope_file_ids),
        )
        candidates = await retrieve_with_keywords(
            keywords=keywords,
            top_k=top_k,
            lit_token=lit_token,
            file_type=file_type,
            drive=drive,
            original_query=semantic,
            file_id_scope=bypass_scope,
            required=required,
        )
        return candidates, None, None

    # Access-filter the shortlist BEFORE forwarding it anywhere. The
    # coarse retriever is drive-scoped but does not consult the host's
    # per-drive locked/unlocked state — a caller without the unlock
    # cookie for a protected drive must not see the file_ids those
    # rows belong to, and must not have the scoped retrieval pass them
    # through (downstream ``retrieve_with_keywords`` already filters,
    # but the SSE event the streaming path emits is built from the
    # ShortlistResult and must reflect the same gate).
    accessible = await _filter_file_ids_via_internal_api(
        list(shortlist.file_ids), lit_token
    )
    if not accessible:
        # Whole shortlist is inaccessible (locked protected drive,
        # transient internal-API failure that fails closed, etc.).
        # Treat as a bypass: run the legacy path scoped to
        # ``bypass_scope`` (the pre_scope on composed callers, drive-
        # wide otherwise) and emit no ``shortlist`` event so the
        # streaming caller doesn't leak any file_ids.
        logger.debug(
            "hierarchical retrieval bypass: reason=access_filter_empty "
            "drive=%s shortlist_size=%d",
            drive, len(shortlist.file_ids),
        )
        candidates = await retrieve_with_keywords(
            keywords=keywords,
            top_k=top_k,
            lit_token=lit_token,
            file_type=file_type,
            drive=drive,
            original_query=semantic,
            file_id_scope=bypass_scope,
            required=required,
        )
        return candidates, None, None

    # Project the access-filtered ids back onto the original ordering so
    # the score / top_score / cosine ranking the SSE event reports stays
    # consistent with what the coarse retriever ranked. Files dropped by
    # the access filter are simply skipped — their ranks collapse.
    filtered_pairs = [
        (fid, score)
        for fid, score in zip(shortlist.file_ids, shortlist.scores)
        if fid in accessible
    ]
    filtered_ids = tuple(fid for fid, _ in filtered_pairs)
    filtered_scores = tuple(score for _, score in filtered_pairs)
    filtered_shortlist = ShortlistResult(
        file_ids=filtered_ids,
        scores=filtered_scores,
        top_score=filtered_scores[0] if filtered_scores else 0.0,
        drive_file_count=shortlist.drive_file_count,
    )

    # Stage 2: clue generation. Pull the AI summaries for the
    # access-filtered shortlist and ask the LLM to expand the query
    # into ``cfg.clue_count`` independent search queries that each
    # match the shortlist's domain vocabulary. ``generate_clues``
    # always returns at least one entry — on any failure (LLM down,
    # parse error, empty array, missing summaries) it collapses to
    # ``[keywords]`` so the downstream loop runs the equivalent of the
    # legacy single-keyword path without any branching here.
    clue_count = max(1, cfg.clue_count)
    summary_map = fetch_long_summaries(list(filtered_shortlist.file_ids))
    summaries = [
        summary_map[fid]
        for fid in filtered_shortlist.file_ids
        if fid in summary_map
    ]
    clues = await generate_clues(
        query=semantic,
        summaries=summaries,
        clue_count=clue_count,
        fallback_keywords=keywords,
    )

    # Stage 3: run each clue concurrently against the scoped retriever.
    # The scope is identical across clues (same access-filtered
    # shortlist) so RRF can merge purely by file_id rank without
    # worrying about scope-induced ordering bias.
    per_clue_results = await asyncio.gather(
        *[
            retrieve_with_keywords(
                keywords=clue,
                top_k=top_k,
                lit_token=lit_token,
                file_type=file_type,
                drive=drive,
                original_query=semantic,
                file_id_scope=list(filtered_shortlist.file_ids),
                required=required,
            )
            for clue in clues
        ]
    )

    if len(clues) == 1:
        # Fast path: single-clue is identical to legacy Phase 2
        # behaviour, no merge cost.
        scoped = per_clue_results[0]
    else:
        scoped = _rrf_merge_candidates(
            list(per_clue_results), top_k=top_k
        )

    clue_tuple = tuple(clues)

    if cfg.fallback_full_search and len(scoped) < 2:
        # Pinpoint-fact fallback: scoped pass (across all clues)
        # produced almost nothing. Run a wider pass against the legacy
        # single-keyword query — the merged scoped path's steer still
        # wins ties because it's prepended. Composed callers keep the
        # personal-history scope (``pre_scope_file_ids``) so the
        # fallback widens *within* the user's "先週観た" set rather
        # than escaping it; non-composed callers run drive-wide as
        # before.
        unscoped = await retrieve_with_keywords(
            keywords=keywords,
            top_k=top_k,
            lit_token=lit_token,
            file_type=file_type,
            drive=drive,
            original_query=semantic,
            file_id_scope=pre_scope_file_ids,
            required=required,
        )
        seen: set[str] = set()
        merged: list[RetrievedFile] = []
        for cand in [*scoped, *unscoped]:
            if cand.file_id in seen:
                continue
            seen.add(cand.file_id)
            merged = [*merged, cand]
        return merged, filtered_shortlist, clue_tuple

    return scoped, filtered_shortlist, clue_tuple


def _language_instruction() -> str:
    """Mirror the legacy ``build_system_prompt`` language postfix.

    Centralised here so the agentic system prompt picks up the same
    operator-configured language directive (``llm.output_language``)
    without re-importing the legacy prompt builder, which would
    create a circular import (``service`` → ``prompt`` → ``service``).
    """
    lang = (settings.llm.output_language or "auto").strip().lower()
    if lang == "auto" or not lang:
        return ""
    mapping = {
        "ja": "- Always answer in Japanese.\n",
        "en": "- Always answer in English.\n",
    }
    return mapping.get(lang, f"- Always answer in {lang}.\n")


async def _maybe_run_agentic(
    *,
    query: str,
    lit_token: str | None,
    drive: str | None,
    viewer_id: str | None,
    temperature: float | None,
    force_legacy_rag: bool,
    start: float,
) -> AnswerResponse | None:
    """Return an ``AnswerResponse`` if the agentic loop ran, else None.

    Falls through to legacy when any of:
    * the eval harness passed ``force_legacy_rag=True``
    * ``llm.agentic_mode == "off"``
    * the active model is not on the operator's allowlist
    * the LLM client is disabled (no provider configured)
    * the LLM client lacks the tools-API entry point (Ollama-native)
    """
    if force_legacy_rag:
        return None
    llm_config = settings.llm
    if not agentic_capability_supported(llm_config.model, llm_config):
        return None
    llm_client = get_llm_client()
    if llm_client is None or not getattr(llm_client, "enabled", False):
        return None
    # The OpenAI-compatible client owns ``chat_with_tools``; the
    # Ollama-native client does not. Fall back to legacy for the
    # latter so we never call a method that does not exist.
    if not hasattr(llm_client, "chat_with_tools"):
        return None

    entry = get_agentic_model_entry(llm_config.model, llm_config)
    context_window = entry.context_window if entry is not None else 32768
    budget = compute_token_budget(context_window)

    answer: AgenticAnswer = await run_agentic_loop(
        query=query,
        llm_client=llm_client,
        drive=drive,
        viewer_id=viewer_id,
        lit_token=lit_token,
        max_total_tokens=budget,
        language_instruction=_language_instruction(),
        temperature=temperature,
    )

    # ``sources`` is left empty: the agentic loop's tool transcript is
    # the authoritative record. The eval harness reads
    # ``agentic_telemetry`` for tool_call_count and route_correctness.
    return AnswerResponse(
        query=query,
        answer=answer.answer,
        citations=list(answer.citations),
        sources=[],
        retrieved_count=len(answer.telemetry.tool_calls),
        took_ms=int((time.monotonic() - start) * 1000),
        agentic_telemetry=answer.telemetry,
    )


async def answer_question(
    query: str,
    lit_token: str | None,
    top_k: int | None = None,
    file_type: str | None = None,
    drive: str | None = None,
    *,
    viewer_id: str | None = None,
    temperature: float | None = None,
    force_legacy_rag: bool = False,
) -> AnswerResponse:
    """Run the full RAG pipeline and return an ``AnswerResponse``.

    The function never raises on LLM failure — it returns an answer
    with ``answer=None`` but populated ``sources`` so the caller can
    at least show the user which files were considered.

    ``force_legacy_rag`` is an internal flag used by the eval harness
    to force the legacy single-turn pipeline even when the agentic
    loop (Phase 1.C) would otherwise activate. Production callers
    never set it; it is not exposed through the HTTP surface.
    """
    rag_config = settings.rag
    effective_top_k = top_k if top_k is not None else rag_config.top_k

    start = time.monotonic()

    # Phase 1.C agentic branch — gated by (a) the operator's
    # ``agentic_mode`` config, (b) the active model being on the
    # ``agentic_models`` allowlist, and (c) the eval-only
    # ``force_legacy_rag`` flag. Anything missing falls through to
    # the legacy single-turn pipeline below; this keeps the rollback
    # path one config edit away.
    agentic_response = await _maybe_run_agentic(
        query=query,
        lit_token=lit_token,
        drive=drive,
        viewer_id=viewer_id,
        temperature=temperature,
        force_legacy_rag=force_legacy_rag,
        start=start,
    )
    if agentic_response is not None:
        return agentic_response

    # Stages A + B: optional personal-history pre-scope. The legacy
    # callers that did not pass ``viewer_id`` get a no-op resolution
    # and the rest of this function behaves exactly as before.
    history = await _resolve_personal_history(
        query=query, viewer_id=viewer_id, drive=drive
    )
    if history.short_circuit:
        return AnswerResponse(
            query=query,
            answer=None,
            citations=[],
            sources=[],
            retrieved_count=0,
            took_ms=int((time.monotonic() - start) * 1000),
        )

    # Stage 1: retrieve + access filter.
    #
    # Routing matrix:
    # * personal-history file_ids set + hierarchical enabled + drive →
    #   composed pipeline: ``Stage B file_ids ∩ shortlist``. Coarse
    #   retrieval narrows the personal scope by summary topicality,
    #   then Stage 2 clue generation supersedes Stage C category
    #   expansion (clues are stronger because they're generated from
    #   the actual shortlist summaries). Spec
    #   ``2026-04-29-intelligence-ask-hierarchical-personal-history-composition.md``.
    # * personal-history file_ids set (hierarchical off) → pass them
    #   straight in as ``file_id_scope`` plus optional Stage C
    #   category expansion. The history scope is already drive-bounded
    #   by the host's join, so the security invariant holds.
    # * hierarchical enabled + drive given → hierarchical helper.
    # * fallback → legacy single-stage retrieval.
    if history.file_ids and settings.rag.hierarchical.enabled and drive:
        structured = await transform_query_structured(
            query, temperature=temperature
        )
        keywords = structured.raw_keywords or query
        required = structured.required or None
        candidates, _shortlist, _clues = await _run_hierarchical_retrieval(
            query=query,
            keywords=keywords,
            drive=drive,
            lit_token=lit_token,
            file_type=file_type,
            top_k=effective_top_k,
            original_query=query,
            pre_scope_file_ids=history.file_ids,
            required=required,
        )
    elif history.file_ids:
        structured = await transform_query_structured(
            query, temperature=temperature
        )
        keywords = structured.raw_keywords or query
        required = structured.required or None
        category_terms = await _resolve_category_expansion(history.decomposed)
        if category_terms:
            effective_required = _drop_required_overlapping_with_expansion(
                required,
                history.decomposed.semantic_query
                if history.decomposed
                else "",
            )
            per_term_results = await asyncio.gather(
                *[
                    retrieve_with_keywords(
                        keywords=term,
                        top_k=effective_top_k,
                        lit_token=lit_token,
                        file_type=file_type,
                        drive=drive,
                        original_query=term,
                        file_id_scope=history.file_ids,
                        required=effective_required,
                    )
                    for term in category_terms
                ]
            )
            candidates = _rrf_merge_candidates(
                list(per_term_results),
                top_k=effective_top_k,
                weights=_category_expansion_weights(len(category_terms)),
            )
        else:
            candidates = await retrieve_with_keywords(
                keywords=keywords,
                top_k=effective_top_k,
                lit_token=lit_token,
                file_type=file_type,
                drive=drive,
                original_query=query,
                file_id_scope=history.file_ids,
                required=required,
            )
    elif settings.rag.hierarchical.enabled and drive:
        structured = await transform_query_structured(
            query, temperature=temperature
        )
        keywords = structured.raw_keywords or query
        required = structured.required or None
        candidates, _shortlist, _clues = await _run_hierarchical_retrieval(
            query=query,
            keywords=keywords,
            drive=drive,
            lit_token=lit_token,
            file_type=file_type,
            top_k=effective_top_k,
            original_query=query,
            required=required,
        )
    else:
        candidates = await retrieve_candidates(
            query=query,
            top_k=effective_top_k,
            lit_token=lit_token,
            file_type=file_type,
            drive=drive,
            transform_temperature=temperature,
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
    contexts = assemble_contexts(candidates, rag_config, query=query)

    # Stage 3: LLM call.
    llm = get_llm_client()
    system_prompt = build_system_prompt(settings.llm.output_language)
    user_prompt = build_user_prompt(query, contexts)

    raw = await llm.generate_json(
        system_prompt,
        user_prompt,
        max_tokens_override=rag_config.max_tokens,
        temperature=temperature,
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
        _to_citation_dict(c, candidates, contexts=contexts) for c in parsed.citations
    ]

    return AnswerResponse(
        query=query,
        answer=parsed.answer,
        citations=citations,
        sources=sources,
        retrieved_count=len(candidates),
        took_ms=int((time.monotonic() - start) * 1000),
    )


# ---------------------------------------------------------------------------
# Streaming path
# ---------------------------------------------------------------------------


# Event kinds emitted over the SSE stream. Kept as a Literal so the
# router layer can exhaustively switch on event.kind without string
# typos sneaking past review.
AnswerEventKind = Literal[
    "keywords",
    "query_decomposed",
    "history_filter",
    "category_expanded",
    "shortlist",
    "clues",
    "answer_chunk",
    "citation",
    "citations",
    "sources",
    "done",
]


@dataclass(frozen=True)
class AnswerEvent:
    """A single event in the streaming RAG answer pipeline.

    ``kind`` determines what ``data`` contains:

    * ``keywords`` → ``{"keywords": str}`` — emitted once, right after
      the LLM query transform completes.
    * ``answer_chunk`` → ``{"delta": str}`` — emitted many times,
      one per token chunk from the LLM stream.
    * ``sources`` → ``{"sources": [...]}`` — emitted once, after the
      retrieval step so the UI can show "I looked at these files"
      even before the answer finishes generating.
    * ``citation`` → ``{"citation": dict, "index": int}`` — emitted
      zero or more times, once per citation as its closing ``}``
      arrives from the LLM stream. The index is 1-based and reflects
      the order the citations appeared in the model output. This is
      progressive: the UI should render a card per event as they
      arrive. The dict shape matches a terminal-list element so
      existing card renderers can be reused.
    * ``citations`` → ``{"citations": [...]}`` — emitted once, after
      the LLM stream ends and the answer JSON has been parsed and
      anti-hallucination-filtered. Kept for backwards compatibility:
      clients that ignore the progressive ``citation`` events still
      get the full validated list here.
    * ``done`` → ``{}`` — terminal marker. The router closes the
      response after sending this.

    The frozen dataclass keeps immutability guarantees without dragging
    Pydantic into the streaming hot path.
    """

    kind: AnswerEventKind
    data: dict[str, Any]


def _empty_done_event(
    *,
    extra: dict[str, Any] | None = None,
) -> AnswerEvent:
    """Construct a terminal ``done`` event with optional metadata."""
    payload: dict[str, Any] = extra or {}
    return AnswerEvent(kind="done", data=payload)


def _decomposed_to_event_payload(decomposed: DecomposedQuery) -> dict[str, Any]:
    """Serialise a ``DecomposedQuery`` for the ``query_decomposed`` SSE event.

    Naive ISO is used to mirror the host's Internal API contract — see
    ``history_client._format_naive_iso``.
    """
    tr = decomposed.time_range
    return {
        "time_range": {
            "label": tr.label,
            "after": tr.after.isoformat() if tr.after else None,
            "before": tr.before.isoformat() if tr.before else None,
        },
        "personal_scope": decomposed.personal_scope,
        "file_type_hint": decomposed.file_type_hint,
        "semantic_query": decomposed.semantic_query,
    }


async def stream_answer(
    query: str,
    lit_token: str | None,
    top_k: int | None = None,
    file_type: str | None = None,
    drive: str | None = None,
    *,
    viewer_id: str | None = None,
) -> AsyncIterator[AnswerEvent]:
    """Run the RAG pipeline and yield SSE-ready events.

    The generator is safe to consume concurrently with request
    cancellation: if the client disconnects, the ``async for`` in the
    router stops iterating and the LLM stream is dropped as the object
    is garbage-collected. We deliberately do **not** cache partial
    answers — Phase 1 is stateless by design (see redesign spec).

    Event ordering (always):

    ``keywords`` → optional ``shortlist`` → optional ``clues`` →
    ``sources`` → 0..N ``answer_chunk`` → 0..N ``citation`` →
    terminal ``citations`` → ``done``

    ``shortlist`` and ``clues`` only fire when the hierarchical RAG
    pipeline actually runs (config on, drive set, shortlist confident,
    at least one shortlist file accessible to the caller). Bypassed
    paths skip both — see ``_run_hierarchical_retrieval`` for the full
    bypass matrix.

    The progressive ``citation`` events are emitted as each citation's
    closing ``}`` arrives from the LLM; the terminal ``citations``
    event always carries the full validated list for older clients.

    On retrieval-empty the pipeline short-circuits: ``keywords`` +
    empty ``sources`` + empty ``citations`` + ``done`` with no
    ``answer_chunk`` or ``citation`` events. On LLM failure the
    pipeline emits as many ``answer_chunk`` events as it managed to
    receive, then tries to parse whatever it buffered; an unparseable
    buffer produces an empty ``citations`` payload (consistent with
    ``answer_question``).
    """
    rag_config = settings.rag
    effective_top_k = top_k if top_k is not None else rag_config.top_k

    start = time.monotonic()

    # Stages A + B: personal-history pre-scope. Resolved before the
    # keyword transform so the SSE event ordering is
    # ``query_decomposed`` → ``history_filter`` → ``keywords`` → ...
    # This matches the conceptual flow ("we noticed you said '先週観た',
    # we found N files, now we'll search them").
    history = await _resolve_personal_history(
        query=query, viewer_id=viewer_id, drive=drive
    )
    if history.decomposed is not None:
        yield AnswerEvent(
            kind="query_decomposed",
            data=_decomposed_to_event_payload(history.decomposed),
        )
    if history.file_ids is not None:
        yield AnswerEvent(
            kind="history_filter",
            data={
                "drive": drive,
                "kind": (
                    history.decomposed.personal_scope
                    if history.decomposed is not None
                    else "viewed"
                ),
                "matched_file_count": len(history.file_ids),
            },
        )
    if history.short_circuit:
        # ``fallback_when_empty="strict"`` + empty Stage B result.
        # No keywords / sources to emit — collapse straight to a
        # citation-less ``done`` so the UI surfaces "該当なし".
        yield AnswerEvent(kind="citations", data={"citations": []})
        yield _empty_done_event(
            extra={
                "retrieved_count": 0,
                "took_ms": int((time.monotonic() - start) * 1000),
            }
        )
        return

    # Stage 0: LLM keyword transform.
    # This is the earliest point we can give the user visible feedback
    # ("searching for: ...") which matters a lot when the downstream
    # retrieval + LLM latency is 2-5 seconds on a home LAN.
    # The structured form drives the required-keyword hard filter
    # (Phase 2+ of 2026-04-30-required-semantic-hybrid-retrieval). The
    # SSE event still carries the flat keyword string for backwards
    # compatibility — surfacing required vs semantic to the UI is a
    # follow-up (Phase 5).
    structured = await transform_query_structured(query)
    keywords = structured.raw_keywords or query
    required = structured.required or None
    yield AnswerEvent(kind="keywords", data={"keywords": keywords})

    # Stage 1+3: retrieval. Routing mirrors ``answer_question``:
    # * history.file_ids + hierarchical + drive → composed (Stage B
    #   ∩ shortlist). clue generation supersedes Stage C category
    #   expansion. Spec
    #   ``2026-04-29-intelligence-ask-hierarchical-personal-history-composition.md``.
    # * history.file_ids only → personal-history scope + optional Stage C.
    # * hierarchical + drive → hierarchical helper.
    # * else → legacy.
    # ``original_query`` carries the raw natural-language question
    # through so vector channels get full semantic context while FTS
    # only sees the noise-free keywords.
    if history.file_ids and settings.rag.hierarchical.enabled and drive:
        candidates, shortlist, clues = await _run_hierarchical_retrieval(
            query=query,
            keywords=keywords,
            drive=drive,
            lit_token=lit_token,
            file_type=file_type,
            top_k=effective_top_k,
            original_query=query,
            pre_scope_file_ids=history.file_ids,
            required=required,
        )
    elif history.file_ids:
        # Stage C: bilingual surface-form expansion of the decomposed
        # semantic_query (e.g. "SF" → ["SF", "science fiction",
        # "宇宙船", ...]). Empty list ⇒ skip multi-query.
        category_terms = await _resolve_category_expansion(history.decomposed)
        if category_terms:
            yield AnswerEvent(
                kind="category_expanded",
                data={
                    "semantic_query": (
                        history.decomposed.semantic_query
                        if history.decomposed
                        else ""
                    ),
                    "expanded": list(category_terms),
                },
            )
            effective_required = _drop_required_overlapping_with_expansion(
                required,
                history.decomposed.semantic_query
                if history.decomposed
                else "",
            )
            per_term_results = await asyncio.gather(
                *[
                    retrieve_with_keywords(
                        keywords=term,
                        top_k=effective_top_k,
                        lit_token=lit_token,
                        file_type=file_type,
                        drive=drive,
                        original_query=term,
                        file_id_scope=history.file_ids,
                        required=effective_required,
                    )
                    for term in category_terms
                ]
            )
            candidates = _rrf_merge_candidates(
                list(per_term_results),
                top_k=effective_top_k,
                weights=_category_expansion_weights(len(category_terms)),
            )
        else:
            candidates = await retrieve_with_keywords(
                keywords=keywords,
                top_k=effective_top_k,
                lit_token=lit_token,
                file_type=file_type,
                drive=drive,
                original_query=query,
                file_id_scope=history.file_ids,
                required=required,
            )
        shortlist = None
        clues = None
    else:
        candidates, shortlist, clues = await _run_hierarchical_retrieval(
            query=query,
            keywords=keywords,
            drive=drive,
            lit_token=lit_token,
            file_type=file_type,
            top_k=effective_top_k,
            original_query=query,
            required=required,
        )

    # Emit the shortlist + clues events ONLY when the hierarchical path
    # actually ran (non-None) — bypassed paths must not lie about what
    # scoping took place. Both sit between ``keywords`` and ``sources``
    # so the UI can show "narrowed to N files, searching for X / Y / Z"
    # before the chunk results land. Order is shortlist → clues since
    # clues are generated *from* the shortlist's summaries.
    if shortlist is not None:
        yield AnswerEvent(
            kind="shortlist",
            data={
                "file_ids": list(shortlist.file_ids),
                "drive_file_count": shortlist.drive_file_count,
                "top_score": shortlist.top_score,
            },
        )
    if clues is not None:
        yield AnswerEvent(
            kind="clues",
            data={"clues": list(clues)},
        )

    sources = [_to_source_dict(c) for c in candidates]
    yield AnswerEvent(kind="sources", data={"sources": sources})

    if not candidates:
        yield AnswerEvent(kind="citations", data={"citations": []})
        yield _empty_done_event(
            extra={
                "retrieved_count": 0,
                "took_ms": int((time.monotonic() - start) * 1000),
            }
        )
        return

    # Stage 2: build per-file contexts under budget.
    contexts = assemble_contexts(candidates, rag_config, query=query)

    # Stage 3: stream the LLM answer.
    llm = get_llm_client()
    system_prompt = build_system_prompt(settings.llm.output_language)
    user_prompt = build_user_prompt(query, contexts)

    buffered: list[str] = []
    # The LLM is asked for a JSON object `{"answer": "...", "citations": [...]}`,
    # so forwarding raw provider chunks would make the UI display JSON
    # syntax instead of the answer. ``AnswerStreamExtractor`` parses
    # the stream on the fly and yields only the decoded characters of
    # the ``answer`` field value, while ``buffered`` keeps the full
    # raw payload for the post-stream terminal citations parse.
    #
    # ``CitationStreamExtractor`` is fed the same raw chunks and
    # returns completed citation dicts one at a time as each closing
    # ``}`` arrives. This lets us yield progressive ``citation``
    # events to the UI instead of making it wait until the whole
    # JSON closes. Each raw citation still runs through the
    # ``_parse_citation`` hallucination filter before it's emitted —
    # the progressive path is NOT an escape hatch around the
    # allowed-file-id gate.
    extractor = AnswerStreamExtractor()
    # Some local LLMs ignore the system prompt's "no inline source
    # references" rule and emit ``[file_id: ...]`` / ``[location: ...]``
    # fragments next to the prose. The sanitizer strips them on the fly
    # so the rendered answer stays clean while the dedicated citations
    # panel is unaffected.
    sanitizer = AnswerSanitizer()
    citation_extractor = CitationStreamExtractor()
    allowed = frozenset(c.file_id for c in candidates)
    emitted_keys: set[tuple[str, str]] = set()
    progressive_index = 0
    # Explicit aiter/aclose lifecycle so a client disconnect mid-stream
    # deterministically tears down the upstream LLM connection instead
    # of waiting on async-generator GC. Without this, a cancelled SSE
    # request could leave the OpenAI-compatible HTTP stream open until
    # the event loop runs its periodic GC pass, wasting provider quota.
    llm_stream = llm.generate_stream(
        system_prompt,
        user_prompt,
        max_tokens_override=rag_config.max_tokens,
    )
    try:
        async for delta in llm_stream:
            buffered = [*buffered, delta]
            extracted = extractor.feed(delta)
            if extracted:
                cleaned = sanitizer.feed(extracted)
                if cleaned:
                    yield AnswerEvent(
                        kind="answer_chunk", data={"delta": cleaned}
                    )
            for raw in citation_extractor.feed(delta):
                event = _build_progressive_citation_event(
                    raw,
                    allowed,
                    candidates,
                    contexts,
                    emitted_keys,
                    progressive_index,
                )
                if event is not None:
                    progressive_index += 1
                    yield event
        # Flush any content still held back by the extractor — happens
        # when the LLM truncated the answer string or emitted short
        # prose that never crossed the mode-decision threshold.
        tail = extractor.finalize()
        if tail:
            cleaned_tail = sanitizer.feed(tail)
            if cleaned_tail:
                yield AnswerEvent(
                    kind="answer_chunk", data={"delta": cleaned_tail}
                )
        sanitizer_tail = sanitizer.finalize()
        if sanitizer_tail:
            yield AnswerEvent(
                kind="answer_chunk", data={"delta": sanitizer_tail}
            )
        # CitationStreamExtractor.finalize is a no-op for completed
        # streams but may return late objects in future tolerant-parse
        # modes. Apply the same hallucination filter + dedup gate.
        for raw in citation_extractor.finalize():
            event = _build_progressive_citation_event(
                raw,
                allowed,
                candidates,
                contexts,
                emitted_keys,
                progressive_index,
            )
            if event is not None:
                progressive_index += 1
                yield event
    finally:
        aclose = getattr(llm_stream, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass

    # Stage 4: parse + validate citations against the retrieved set.
    # The LLM is instructed to return a single JSON object; we parse
    # the full buffered output here, post-stream, so the terminal
    # ``citations`` event only contains file_ids that survived the
    # hallucination filter. The parser uses the same code path as the
    # non-streaming pipeline for consistency.
    full_text = "".join(buffered)
    parsed = _parse_streamed_answer(full_text, candidates)

    if parsed is None:
        citations: list[dict[str, Any]] = []
    else:
        citations = [
            _to_citation_dict(c, candidates, contexts=contexts) for c in parsed.citations
        ]

    yield AnswerEvent(kind="citations", data={"citations": citations})
    yield _empty_done_event(
        extra={
            "retrieved_count": len(candidates),
            "took_ms": int((time.monotonic() - start) * 1000),
        }
    )


def _build_progressive_citation_event(
    raw: dict,
    allowed_file_ids: frozenset[str],
    candidates: list[RetrievedFile],
    contexts: list,
    emitted_keys: set[tuple[str, str]],
    current_index: int,
) -> "AnswerEvent | None":
    """Validate a raw streamed citation dict and wrap it in an event.

    Runs the same hallucination filter and ``(file_id, location)``
    dedup gate as the non-streaming parser so the progressive path
    is not an escape hatch around any security-critical check. Returns
    None when the citation fails validation or was already emitted.

    ``current_index`` is the index this citation *would* receive
    (1-based, so callers pass the current running count and increment
    on a non-None return). Keeping the counter outside this helper
    means filtered/duplicate citations don't create gaps in the
    published numbering.
    """
    parsed = _parse_citation(raw, allowed_file_ids)
    if parsed is None:
        return None
    key = (parsed.file_id, parsed.location)
    if key in emitted_keys:
        return None
    emitted_keys.add(key)
    cit_dict = _to_citation_dict(parsed, candidates, contexts=contexts)
    return AnswerEvent(
        kind="citation",
        data={"citation": cit_dict, "index": current_index + 1},
    )


def _parse_streamed_answer(
    full_text: str,
    candidates: list[RetrievedFile],
):
    """Parse the LLM's full buffered JSON output and validate citations.

    Mirrors the relevant portion of ``answer_question``'s Stage 4 but
    operates on a text blob (since ``generate_stream`` returns raw
    strings) instead of a pre-parsed dict. Returns the parsed answer
    object, or None if the buffer is empty / unparseable.
    """
    if not full_text.strip():
        return None

    # Try direct JSON parse first; fall back to regex object extraction
    # for models that wrap output in prose or code fences.
    import re

    raw_json: dict | list | None = None
    try:
        raw_json = json.loads(full_text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", full_text, re.DOTALL)
        if match:
            try:
                raw_json = json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError):
                raw_json = None

    if raw_json is None:
        logger.debug("Streamed answer buffer was not valid JSON")
        return None

    allowed = frozenset(c.file_id for c in candidates)
    return parse_answer(raw_json, allowed)


# ---------------------------------------------------------------------------
# Find mode (file-listing sibling of Ask) — spec
# ``2026-04-30-intelligence-find-mode.md``
# ---------------------------------------------------------------------------


# Default page size mirrors spec §3.2 example. The pydantic ``FindRequest``
# clamps to [1, 20] but the service-level default lets callers omit the
# argument entirely (e.g. internal tooling, eval harness).
_FIND_DEFAULT_LIMIT = 20

# File-type strings the LLM emits that must NOT be forwarded as a literal
# filter — they translate to "no hint", which means "do not filter".
_FIND_NO_FILE_TYPE = frozenset({"none", "", None})


def _build_decomposed_from_overrides(
    raw_query: str, overrides: dict[str, Any]
) -> DecomposedQuery:
    """Turn a chip-edited ``overrides`` dict into a ``DecomposedQuery``.

    The frontend posts overrides as plain-string axis values
    (``time_range="none"``, ``personal_scope="viewed"``, ...). We resolve
    the symbolic time-range label here so downstream stages receive the
    same shape as the LLM-decomposed path.
    """
    from app.rag.query_decomposer import _resolve_time_range  # local import

    time_label = overrides.get("time_range") or "none"
    if not isinstance(time_label, str):
        time_label = "none"
    personal_scope = overrides.get("personal_scope") or "none"
    if not isinstance(personal_scope, str):
        personal_scope = "none"
    file_type_hint = overrides.get("file_type_hint") or "none"
    if not isinstance(file_type_hint, str):
        file_type_hint = "none"
    semantic_query = overrides.get("semantic_query") or ""
    if not isinstance(semantic_query, str):
        semantic_query = ""

    cfg_lookback = settings.rag.personal_history.max_lookback_days
    try:
        from datetime import UTC, datetime
        time_range = _resolve_time_range(
            time_label,
            now=datetime.now(UTC),
            max_lookback_days=cfg_lookback,
        )
    except Exception:  # noqa: BLE001 — defensive: never hard-fail Find
        time_range = TimeRange.empty()

    return DecomposedQuery(
        raw_query=raw_query,
        time_range=time_range,
        personal_scope=personal_scope,
        file_type_hint=file_type_hint,
        semantic_query=semantic_query,
    )


def _decomposed_to_dict(
    decomposed: DecomposedQuery, category_expansion: list[str]
) -> dict[str, Any]:
    """Render a ``DecomposedQuery`` as the spec §3.2 ``decomposed`` block.

    All five keys are always present so the frontend can render its chip
    layout without conditional existence checks. ``time_range`` is a
    nested dict (kind / value / after / before); the other axes collapse
    to plain strings.
    """
    tr = decomposed.time_range
    return {
        "time_range": {
            "kind": "relative" if tr.label not in ("none", "") else "none",
            "value": tr.label or "none",
            "after": tr.after.isoformat() if tr.after else None,
            "before": tr.before.isoformat() if tr.before else None,
        },
        "personal_scope": decomposed.personal_scope,
        "file_type_hint": decomposed.file_type_hint,
        "semantic_query": decomposed.semantic_query,
        "category_expansion": list(category_expansion),
    }


def _segment_to_hit(retrieved: Any) -> dict[str, Any]:
    """Build the spec §3.2 ``hit`` block from a retrieved file.

    Picks the first segment as the representative hit. Segment shape is
    duck-typed so the function works with both real ``RetrievedFile``
    (segments are ``SegmentGroup`` instances) and the test mocks (which
    expose ``text`` / ``start_seconds`` / ``end_seconds`` / ``kind``
    directly on a ``MagicMock``).
    """
    segments = getattr(retrieved, "segments", None) or ()
    if not segments:
        return {
            "kind": "transcript",
            "location": None,
            "text": "",
        }
    seg = segments[0]
    text = getattr(seg, "text", None)
    if not isinstance(text, str):
        # SegmentGroup carries ``matches[].text`` instead of a flat
        # ``text`` attribute; pull the first match's text as a fallback.
        matches = getattr(seg, "matches", None) or ()
        text = matches[0].text if matches and getattr(matches[0], "text", None) else ""

    kind = getattr(seg, "kind", None)
    if not isinstance(kind, str):
        match_types = getattr(retrieved, "match_types", None) or ()
        kind = match_types[0] if match_types else "transcript"

    location: dict[str, Any] = {}
    start = getattr(seg, "start_seconds", None)
    end = getattr(seg, "end_seconds", None)
    if isinstance(start, (int, float)):
        location["start_seconds"] = float(start)
    if isinstance(end, (int, float)):
        location["end_seconds"] = float(end)
    # SegmentGroup exposes ``time_range`` as a tuple instead of named
    # attributes. Use it as a fallback when the explicit fields are
    # missing — keeps real retrieve output usable without changes.
    if not location:
        time_range = getattr(seg, "time_range", None)
        if isinstance(time_range, tuple) and len(time_range) >= 2:
            location = {
                "start_seconds": float(time_range[0]),
                "end_seconds": float(time_range[1]),
            }

    # Normalise empty location to null so the wire format matches the TS
    # contract (``location: {...} | null``). An empty dict is uninformative
    # and a future consumer using ``if (hit.location !== null)`` would be
    # surprised.
    return {
        "kind": kind,
        "location": location if location else None,
        "text": text,
    }


def _build_result_item(
    retrieved: Any, viewed_at_by_id: dict[str, str]
) -> dict[str, Any]:
    """Render one retrieved file as a spec §3.2 ``results[]`` element."""
    file_id = getattr(retrieved, "file_id", "")
    score = getattr(retrieved, "score", 0.0)
    name = getattr(retrieved, "filename", None) or getattr(retrieved, "title", "") or ""
    file_type = getattr(retrieved, "file_type", "") or ""

    return {
        "file_id": file_id,
        "score": float(score) if isinstance(score, (int, float)) else 0.0,
        "hit": _segment_to_hit(retrieved),
        "file": {
            "name": name,
            "file_type": file_type,
            "thumbnail_url": f"/api/files/{file_id}/thumbnail",
            "viewed_at": viewed_at_by_id.get(file_id),
        },
    }


def _empty_find_response(
    decomposed: DecomposedQuery,
    limit: int,
    category_expansion: list[str] | None = None,
) -> dict[str, Any]:
    """Build a zero-result response with the decomposed block populated.

    ``category_expansion`` reflects whatever Stage C produced (or ``[]``
    on skip / failure) so the chip layout stays accurate even when no
    files matched.
    """
    return {
        "decomposed": _decomposed_to_dict(decomposed, category_expansion or []),
        "results": [],
        "total": 0,
        "limit": limit,
    }


async def find_files(
    question: str,
    drive: str,
    viewer_id: str | None = None,
    overrides: dict[str, Any] | None = None,
    limit: int = _FIND_DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Run the Find pipeline (Stage A-D) and return spec §3.2 JSON.

    Find mode is the file-listing sibling of Ask: same Stages A-D, no
    Stage E (LLM answer generation). On any per-stage failure the
    pipeline degrades gracefully — empty history scope, raw semantic
    fallback, empty retrieve — but never raises.

    Args:
        question: Raw user question (already length-validated).
        drive: Canonical drive name.
        viewer_id: Optional viewer hash. ``None`` skips Stage B
            entirely (graceful: spec §13.A).
        overrides: Chip-edited structured query. When present, Stage A
            (LLM decompose) is skipped and the dict is treated as the
            source of truth.
        limit: Page size cap on the returned ``results`` list.

    Returns:
        A dict with ``decomposed`` / ``results`` / ``total`` / ``limit``
        keys per spec §3.2.
    """
    effective_limit = max(1, int(limit))

    # Stage A: decompose (or use overrides verbatim).
    if overrides:
        decomposed = _build_decomposed_from_overrides(question, overrides)
    else:
        try:
            decomposed = await decompose_query(question)
        except Exception as exc:  # noqa: BLE001 — graceful fallback
            logger.debug("Find Stage A decompose failed: %s", type(exc).__name__)
            decomposed = DecomposedQuery.passthrough(question)

    # Stage B: viewer history filter. Only runs when the user asked
    # something personal AND we have a viewer_id to filter by. Missing
    # viewer_id is a graceful skip — spec §13.A explicitly forbids 4xx
    # here so opt-in profile state stays optional.
    file_id_scope: list[str] | None = None
    viewed_at_by_id: dict[str, str] = {}
    if (
        decomposed.personal_scope in ("viewed", "not_viewed")
        and viewer_id is not None
    ):
        try:
            file_id_scope = await fetch_viewer_history(
                viewer_id=viewer_id,
                drive=drive,
                after=decomposed.time_range.after,
                before=decomposed.time_range.before,
                kind=decomposed.personal_scope,  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001 — graceful fallback
            logger.debug(
                "Find Stage B history failed: %s", type(exc).__name__
            )
            file_id_scope = None

    # Stage C: category expansion. Shared with Ask via the
    # ``_resolve_category_expansion`` helper so the two pipelines stay
    # in lockstep — the helper handles config gating, empty
    # semantic_query, and the "single-element collapse → []" rule
    # (single term adds no multi-query value, so the caller falls
    # back to the legacy single-keyword path). Failures degrade to
    # ``[]`` to keep Find's no-raise contract.
    try:
        category_expansion = await _resolve_category_expansion(decomposed)
    except Exception as exc:  # noqa: BLE001 — graceful fallback
        logger.debug("Find Stage C expand failed: %s", type(exc).__name__)
        category_expansion = []

    # Stage D: scoped retrieval. ``file_type=None`` when the hint is
    # "none" / empty — passing the literal "none" would filter to
    # nothing.
    file_type_hint = decomposed.file_type_hint
    file_type_filter = (
        None if file_type_hint in _FIND_NO_FILE_TYPE else file_type_hint
    )

    # Stage D bonus: structured query transform for the required-keyword
    # hard filter. Independent from ``decomposed`` (which carries
    # time / personal_scope / file_type) — the structured transform
    # extracts proper-noun-like ``required`` terms that anchor the
    # retrieve. Failures collapse to ``required=None`` so Find still
    # works without the hard filter.
    try:
        structured = await transform_query_structured(question)
        required = structured.required or None
    except Exception as exc:  # noqa: BLE001 — graceful fallback
        logger.debug(
            "Find Stage D structured transform failed: %s",
            type(exc).__name__,
        )
        required = None

    # Top-K headroom for rerank: pull twice the user-visible limit so
    # the response cap doesn't starve downstream filtering.
    retrieve_top_k = max(effective_limit * 2, effective_limit)

    # Multi-query RRF when Stage C produced 2+ expansion terms. Mirrors
    # the Ask path (``answer_question`` / ``stream_answer``) — using a
    # single term collapses retrieval onto one embedding's noise, which
    # surfaces unrelated files (e.g. "料理" alone pulling in "スマホ料金
    # プラン"). The per-term RRF dilutes single-term noise so files that
    # match multiple expansions ("レシピ" + "調理" + "食材") rise.
    #
    # When expanding, drop any ``required`` term that overlaps with
    # ``semantic_query`` — the expansion is the multi-query mechanism
    # for that very token, so keeping it as a hard filter would defeat
    # the expansion (see ``_drop_required_overlapping_with_expansion``).
    try:
        if len(category_expansion) >= 2:
            effective_required = _drop_required_overlapping_with_expansion(
                required, decomposed.semantic_query
            )
            # Per-term expansion must diversify BOTH the FTS keyword
            # channel and the vector channel — pass the term as
            # ``original_query`` so the vector embedding is computed
            # per-term too. With the legacy ``original_query=question``
            # every per-term retrieve fed the same "料理に関する動画"
            # embedding into the vector channel; the per-term FTS hits
            # got diversified but the vector channel returned the
            # same noisy nearest-neighbours four times, which RRF
            # then accumulated to push unrelated files (ポケモンGO,
            # スマホ料金プラン etc.) above the actual cooking content.
            per_term_results = await asyncio.gather(
                *[
                    retrieve_with_keywords(
                        keywords=term,
                        top_k=retrieve_top_k,
                        lit_token=None,
                        file_type=file_type_filter,
                        drive=drive,
                        original_query=term,
                        file_id_scope=file_id_scope,
                        required=effective_required,
                    )
                    for term in category_expansion
                ]
            )
            retrieved = _rrf_merge_candidates(
                list(per_term_results),
                top_k=retrieve_top_k,
                weights=_category_expansion_weights(len(category_expansion)),
            )
        else:
            retrieved = await retrieve_with_keywords(
                keywords=decomposed.semantic_query or question,
                top_k=retrieve_top_k,
                lit_token=None,
                file_type=file_type_filter,
                drive=drive,
                original_query=question,
                file_id_scope=file_id_scope,
                required=required,
            )
    except Exception as exc:  # noqa: BLE001 — graceful fallback
        logger.debug("Find Stage D retrieve failed: %s", type(exc).__name__)
        retrieved = []

    if not retrieved:
        return _empty_find_response(
            decomposed, effective_limit, category_expansion
        )

    # Format results. ``viewed_at`` is populated from the history fetch
    # when available; absent otherwise (frontend renders the missing
    # case as "視聴履歴なし").
    items = [
        _build_result_item(r, viewed_at_by_id)
        for r in retrieved[:effective_limit]
    ]

    return {
        "decomposed": _decomposed_to_dict(decomposed, category_expansion),
        "results": items,
        "total": len(retrieved),
        "limit": effective_limit,
    }
