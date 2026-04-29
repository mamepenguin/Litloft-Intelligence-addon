"""Stage 2 clue generation: multi-query expansion under shortlist context.

Once Stage 1 has narrowed the drive to a shortlist of file candidates,
their AI summaries describe the domain vocabulary actually present in
that subset. Feeding those summaries to a small LLM call produces 2-3
independent search queries (clues) that the legacy single-keyword path
could not synthesise — covering paraphrase / synonym ranges that the
user's original phrasing missed.

Each clue is run through Stage 3 (``retrieve_with_keywords``) under the
same shortlist scope; the resulting per-clue candidate lists are
RRF-merged into a single ranked list (the merger lives in
``app.rag.service`` since clue dispatch is the service-layer concern).

Two failure modes both fall back to a single-element list containing
the caller's pre-transformed keyword query:

* LLM disabled or throws — graceful degradation, retrieval still runs.
* JSON parse fails / clues array is empty / all entries are blank.

The fallback is the same string the legacy path would have used, so a
broken clue generator never *worsens* recall — it only loses the
multi-query expansion benefit.
"""

import logging

from sqlalchemy import text as sql_text

from app.database import get_search_db
from app.dependencies import get_llm_client
from app.prompt_loader import render
from app.rag.keyword_filter import filter_keywords

logger = logging.getLogger(__name__)


# Same reasoning as ``query_transform``: small local models occasionally
# emit short reasoning prose before the JSON. 256 absorbs that without
# meaningfully changing latency — generation itself is bounded by the
# answer LLM, not this short rewrite call.
_CLUE_GEN_MAX_TOKENS = 256


# System prompt is rendered per-call via prompt_loader because clue_count
# varies. The template lives at prompts/rag/clue_generator_system.jinja2.


def fetch_long_summaries(file_ids: list[str]) -> dict[str, str]:
    """Look up summary text rows for ``file_ids`` (text or visual).

    For transcribable / textual files (video / audio / document) this
    returns the AI ``long_summary`` (status='generated' only — hidden
    rows respect the user's opt-out). For images this returns the
    AI ``visual_description`` (status='success' only). The two fields
    are mutually exclusive in practice (the summaries worker and the
    vision worker target disjoint context_types) so this function
    folds them into a single ``file_id -> text`` map without preferring
    one over the other.

    The function name retains the legacy ``long_summaries`` wording
    because callers (``_run_hierarchical_retrieval``) treat both
    text-summary and visual-description as the same Phase 3 input
    signal; renaming it would just mean threading the rename through
    every call site for no semantic change.

    Files without a usable row in ``file_summaries`` are simply absent
    from the result — callers handle the missing-key case (an empty
    map triggers the fallback keyword path in ``generate_clues``).
    """
    if not file_ids:
        return {}

    # Use named placeholders rather than IN :ids expanding because the
    # search DB session occasionally runs against legacy SQLAlchemy
    # versions in tests; the explicit parameter list keeps this stable.
    placeholders = ",".join(f":id{i}" for i in range(len(file_ids)))
    params: dict[str, str] = {f"id{i}": fid for i, fid in enumerate(file_ids)}

    with get_search_db() as session:
        rows = session.execute(
            sql_text(
                "SELECT file_id, "
                "  CASE WHEN status = 'generated' THEN long_summary END, "
                "  CASE WHEN visual_description_status = 'success' "
                "    THEN visual_description END "
                "FROM file_summaries "
                f"WHERE file_id IN ({placeholders})"
            ),
            params,
        ).fetchall()

    # Whitespace-only summaries propagated downstream would only feed
    # the LLM noise and trigger the fallback path anyway, so prune
    # them at the fetch layer for a single source of truth. Prefer the
    # text summary (``long_summary``) over the visual description when
    # — by some future schema drift — both happen to be populated;
    # text summaries are richer for clue-generation prompting.
    out: dict[str, str] = {}
    for fid, long_s, vis_s in rows:
        text = (long_s or "").strip() or (vis_s or "").strip()
        if text:
            out[fid] = text
    return out


async def generate_clues(
    query: str,
    summaries: list[str],
    *,
    clue_count: int,
    fallback_keywords: str,
    temperature: float | None = None,
) -> list[str]:
    """Generate up to ``clue_count`` retrieval queries from ``query`` + ``summaries``.

    Args:
        query: The user's natural-language question. Used verbatim so
            proper nouns survive into the LLM prompt.
        summaries: ``long_summary`` strings for the files in the Stage 1
            shortlist, in shortlist score order. May be empty — that
            triggers the keyword fallback (the LLM has no domain
            context to expand around).
        clue_count: Target number of independent clues. Honoured as a
            hard upper bound (the LLM may return fewer; ``[]`` triggers
            fallback).
        fallback_keywords: The pre-transformed keyword string from
            ``query_transform``. Returned as a single-element list on
            *any* failure so the caller's downstream retrieve still has
            something usable.
        temperature: Optional override for the LLM call. Defaults to
            the provider's default — clue generation benefits slightly
            from low temperature for consistency.

    Returns:
        A list of 1..``clue_count`` non-empty keyword strings. Always
        contains at least one entry; on failure the single entry is
        ``fallback_keywords`` so the caller can blindly iterate without
        special-casing the empty-list path.
    """
    stripped = query.strip()
    if not stripped:
        return [fallback_keywords]

    if clue_count < 1:
        return [fallback_keywords]

    if not summaries:
        # No Stage 1 context to expand around. Single-clue path is the
        # legacy behaviour — preserve it.
        return [fallback_keywords]

    try:
        llm = get_llm_client()
    except RuntimeError:
        return [fallback_keywords]

    if not llm.enabled:
        return [fallback_keywords]

    # Build the candidate-summaries block. Numbering keeps the LLM from
    # bleeding two summaries into one mental cluster, which is a real
    # failure mode on small local models. A trailing newline between
    # entries also helps tokenizers split cleanly.
    summary_block = "\n\n".join(
        f"[{i + 1}] {s.strip()}" for i, s in enumerate(summaries) if s.strip()
    )
    if not summary_block:
        # All summaries were whitespace/empty — same as no summaries.
        return [fallback_keywords]

    system_prompt = render(
        "rag/clue_generator_system.jinja2",
        clue_count=clue_count,
    )
    user_prompt = (
        f"<user_question>\n{stripped}\n</user_question>\n"
        f"<candidate_summaries>\n{summary_block}\n</candidate_summaries>"
    )

    raw = await llm.generate_json(
        system_prompt,
        user_prompt,
        max_tokens_override=_CLUE_GEN_MAX_TOKENS,
        temperature=temperature,
    )

    if not isinstance(raw, dict):
        logger.debug("Clue generation returned non-dict, falling back to keywords")
        return [fallback_keywords]

    raw_clues = raw.get("clues")
    if not isinstance(raw_clues, list):
        logger.debug("Clue generation missing 'clues' list, falling back to keywords")
        return [fallback_keywords]

    cleaned: list[str] = []
    for entry in raw_clues:
        if not isinstance(entry, str):
            continue
        # Same blocklist that protects ``transform_query`` from local
        # models leaking question / file-type words into the keyword
        # string. If filtering empties this clue we drop it — but only
        # this clue, not the whole result.
        candidate = filter_keywords(entry.strip())
        if candidate:
            cleaned = [*cleaned, candidate]
        if len(cleaned) >= clue_count:
            break

    if not cleaned:
        logger.debug("All generated clues were unusable, falling back to keywords")
        return [fallback_keywords]

    return cleaned
