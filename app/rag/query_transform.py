"""Natural-language query → search-keyword transformation for RAG.

Runs a single short LLM call that rewrites the user's question into
2–3 retrieval keywords. The transform is intentionally narrow:

* Noise words (question words, verbs, file-type words) are dropped so
  they do not poison the FTS5 AND-joined query. "おでかけ子ザメ 共通点は？"
  becomes "おでかけ子ザメ".
* Proper nouns stay in their original language — translation loses
  information in a Japanese-first corpus with a Japanese-first CLIP.
* Only a single ``{"keywords": "..."}`` field is emitted. The LLM
  does not return the list of dropped words; exposing them would
  just be user-education noise.

On any failure (LLM disabled, network error, parse failure, empty
keywords) the function falls back to the original natural-language
query. Graceful degradation matters because RAG answers are more
useful with ugly keywords than a hard failure.
"""

import logging

from app.dependencies import get_llm_client
from app.prompt_loader import render
from app.rag.keyword_filter import filter_keywords

logger = logging.getLogger(__name__)


# Max tokens for the keyword extraction response. A three-keyword
# JSON payload fits in ~40 tokens, but small local models (notably
# gemma4:e2b under ollama) emit short reasoning before the JSON and
# get silently truncated to an empty body at 64. 256 accommodates that
# preamble without meaningfully increasing end-to-end latency — the
# generate step itself is bounded by rag.max_tokens (1024+).
_QUERY_TRANSFORM_MAX_TOKENS = 256


_SYSTEM_PROMPT = render("rag/query_transform_system.jinja2")


async def transform_query(
    natural_query: str,
    *,
    temperature: float | None = None,
) -> str:
    """Rewrite a natural-language RAG question as a search-keyword string.

    Args:
        natural_query: The raw user input (a question, typically in
            Japanese). Must already be length-validated by the caller.

    Returns:
        A keyword string suitable for passing to ``app.search.search``.
        Falls back to the original ``natural_query`` on any error so
        the caller can always attempt retrieval.
    """
    stripped = natural_query.strip()
    if not stripped:
        return natural_query

    try:
        llm = get_llm_client()
    except RuntimeError:
        # Dependency container not initialized. Fall through to raw
        # query — the caller (RAG pipeline) will surface a clearer
        # error via its own gating.
        return natural_query

    if not llm.enabled:
        return natural_query

    # Wrap user input in an explicit delimiter block so the LLM treats
    # it as data, not as instructions. The system prompt tells the model
    # to ignore any directives inside the tag. This is a mitigation (not
    # a guarantee) against prompt injection — keyword extraction is a
    # narrow task with a structured JSON output gate, so the blast
    # radius of a successful injection is limited to poisoning the
    # retrieval (which the attacker already controls via their query).
    user_prompt = f"<user_question>\n{stripped}\n</user_question>"
    raw = await llm.generate_json(
        _SYSTEM_PROMPT,
        user_prompt,
        max_tokens_override=_QUERY_TRANSFORM_MAX_TOKENS,
        temperature=temperature,
    )

    if isinstance(raw, dict):
        keywords = raw.get("keywords")
        if isinstance(keywords, str):
            cleaned = keywords.strip()
            if cleaned:
                # Server-side blocklist safety net: weak local LLMs
                # (gemma4:e2b raw fallback) leak file-type / question
                # words back into the keyword string even when the
                # system prompt forbids them. Strip those tokens so FTS
                # AND-joins are not poisoned. If filtering empties the
                # result the LLM extraction was useless anyway, so fall
                # through to the raw-query graceful path below.
                filtered = filter_keywords(cleaned)
                if filtered:
                    return filtered

    logger.debug(
        "Query transform returned unusable output, falling back to raw query"
    )
    return natural_query
