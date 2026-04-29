"""Stage C: semantic category expansion for the personal-history Ask path.

Spec: ``2026-04-26-intelligence-ask-personal-history-query.md`` §4.2.
Genre / category words like "SF" or "ホラー" rarely appear verbatim
in transcripts. The retriever's vector channel can sometimes bridge
the gap, but FTS5 / keyword channels need the surface forms. So we
ask a small LLM to rewrite "SF" into a bag of bilingual surface
words ("science fiction" / "宇宙船" / "ロボット" / ...) and dispatch
each as an independent multi-query retrieval pass.

Failure modes
-------------
On any failure (LLM disabled, parse error, all expansions blocked by
the keyword filter, empty list) we fall back to the single-element
list ``[semantic_query]`` so the caller's downstream loop can iterate
without branching. This mirrors ``clue_generator.generate_clues`` —
a broken expander never *worsens* recall, only loses the multi-query
expansion benefit.
"""

from __future__ import annotations

import logging

from app.dependencies import get_llm_client
from app.prompt_loader import render
from app.rag.keyword_filter import filter_keywords

logger = logging.getLogger(__name__)


# Same reasoning as ``query_transform`` / ``clue_generator``: small
# local models occasionally emit short reasoning prose before the
# JSON. 256 absorbs that without meaningfully changing latency — the
# answer LLM is the dominant wall-clock contributor.
_EXPAND_MAX_TOKENS = 256


# System prompt is rendered per-call via prompt_loader because max_terms
# varies. The template lives at prompts/rag/category_expander_system.jinja2.


async def expand_category(
    semantic_query: str,
    *,
    max_terms: int = 8,
    temperature: float | None = None,
) -> list[str]:
    """Return up to ``max_terms`` surface forms for a category word.

    Args:
        semantic_query: The decomposer's ``semantic_query`` field —
            the residual concept after stripping time / scope / file
            type ("SF", "ホラー", "勉強用", etc.). Empty strings short
            circuit to ``[]`` because the caller's multi-query path
            should skip Stage C entirely on no-concept queries.
        max_terms: Hard cap on the LLM-emitted list. Spec default 8.
        temperature: Optional LLM temperature override.

    Returns:
        A list of 1..``max_terms`` non-empty keyword strings. Always
        contains at least one entry on a non-empty input — the original
        ``semantic_query`` is the safe fallback. Empty input returns
        an empty list (caller decides what to do).
    """
    stripped = semantic_query.strip()
    if not stripped:
        # Personal-history "今月観てない動画" — no concept to expand,
        # caller skips Stage C and lets retrieval run on the keyword
        # path alone.
        return []

    if max_terms < 1:
        return [stripped]

    try:
        llm = get_llm_client()
    except RuntimeError:
        return [stripped]

    if not llm.enabled:
        return [stripped]

    system_prompt = render(
        "rag/category_expander_system.jinja2",
        max_terms=max_terms,
    )
    user_prompt = f"<category>\n{stripped}\n</category>"

    raw = await llm.generate_json(
        system_prompt,
        user_prompt,
        max_tokens_override=_EXPAND_MAX_TOKENS,
        temperature=temperature,
    )

    if not isinstance(raw, dict):
        logger.debug("Category expansion returned non-dict, using raw query")
        return [stripped]

    terms = raw.get("terms")
    if not isinstance(terms, list):
        logger.debug("Category expansion missing 'terms' list, using raw query")
        return [stripped]

    cleaned: list[str] = []
    seen: set[str] = set()
    for entry in terms:
        if not isinstance(entry, str):
            continue
        # Reuse the keyword filter so file-type / question words the LLM
        # might leak into expansions ("動画" / "what") get stripped.
        candidate = filter_keywords(entry.strip())
        if not candidate:
            continue
        # Case-insensitive dedup keeps "SF"+"sf"+"Sf" from filling the
        # budget with near-duplicates. Keep the first form so the order
        # the LLM emitted (typically most-relevant first) is preserved.
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned = [*cleaned, candidate]
        if len(cleaned) >= max_terms:
            break

    if not cleaned:
        logger.debug("All expansion terms unusable, falling back to raw query")
        return [stripped]

    # Always include the raw query if the LLM didn't already. Some
    # callers fan out per-term retrieval — losing the original word
    # could miss hits that match it verbatim.
    if stripped.lower() not in seen:
        cleaned = [stripped, *cleaned[: max_terms - 1]]

    return cleaned
