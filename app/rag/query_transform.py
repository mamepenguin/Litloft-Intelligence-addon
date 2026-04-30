"""Natural-language query → keyword transformation for RAG retrieval.

Two surfaces:

* ``transform_query(natural_query) -> str`` — legacy flat-string output
  used by callers that still feed the keyword string directly into the
  FTS path.
* ``transform_query_structured(natural_query) -> StructuredQuery`` —
  the Phase 1 surface for the required-keyword hard filter spec
  (``2026-04-30-required-semantic-hybrid-retrieval.md``). The LLM
  classifies each token as a *required* proper-noun-like term or a
  *semantic* concept, and Python re-checks the script and unions in
  mechanical aliases (hira↔kata, NFKD/casefold) so common variants
  survive even when the LLM omits them.

Both surfaces share a single LLM call. ``transform_query`` is implemented
as a thin wrapper around ``transform_query_structured`` so the two
cannot drift, but it preserves the existing graceful-degradation
contract: any failure mode (LLM disabled, parse failure, schema
mismatch, empty output) returns the raw query string unchanged.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass

from app.dependencies import get_llm_client
from app.prompt_loader import render
from app.rag.keyword_filter import filter_keywords

logger = logging.getLogger(__name__)


# Max tokens for the structured response. The Phase 1 schema can carry
# multiple required terms with several aliases each, so we raise the
# 256-token cap that the legacy flat-keywords prompt used. 512 still
# fits comfortably below ``rag.max_tokens`` (1024+) and accommodates
# the short reasoning preamble that small local models emit before
# the JSON body.
_QUERY_TRANSFORM_MAX_TOKENS = 512


_SYSTEM_PROMPT = render("rag/query_transform_system.jinja2")


# --- StructuredQuery types ------------------------------------------------


_VALID_SCRIPTS = frozenset(
    {
        "han",
        "hira",
        "kata",
        "japanese-mix",
        "latin",
        "cyrillic",
        "hangul",
        "other",
    }
)


@dataclass(frozen=True)
class RequiredTerm:
    """A required keyword group with its alias variants.

    The ``script`` field is the *Python-detected* script of the
    canonical, not whatever the LLM said. We re-detect server-side
    because weak local LLMs occasionally mis-classify (e.g. labelling
    "Python" as ``han``). ``aliases`` always contains ``canonical``
    and the mechanical expansions for ``script``; LLM-supplied
    aliases that fall in the same script are unioned in.
    """

    canonical: str
    script: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class StructuredQuery:
    """Query split into hard-filter required terms and semantic terms.

    ``raw_keywords`` is a flat-string projection (canonicals + semantic
    joined by spaces) used by callers that have not yet been migrated
    to the structured retriever. New callers should consume ``required``
    and ``semantic`` directly.
    """

    required: tuple[RequiredTerm, ...]
    semantic: tuple[str, ...]
    raw_keywords: str

    @classmethod
    def passthrough(cls, raw_query: str) -> "StructuredQuery":
        """No-signal form returned on every failure path."""
        return cls(
            required=(),
            semantic=(raw_query,),
            raw_keywords=raw_query,
        )


# --- Script detection -----------------------------------------------------


def _classify_codepoint(cp: int) -> str:
    """Return the script bucket for a single Unicode code point.

    Buckets used internally by ``detect_script``. The aggregator
    folds han/hira/kata into ``"japanese-mix"`` when more than one
    Japanese sub-bucket is present in the input.
    """
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
        return "han"
    if 0x3040 <= cp <= 0x309F:
        return "hira"
    if 0x30A0 <= cp <= 0x30FF:
        return "kata"
    if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
        return "latin"
    # Latin Supplement / Extended-A / Extended-B — diacritics, etc.
    if 0x00C0 <= cp <= 0x024F:
        return "latin"
    if 0x0400 <= cp <= 0x04FF:
        return "cyrillic"
    if 0xAC00 <= cp <= 0xD7AF:
        return "hangul"
    return "other"


def detect_script(text: str) -> str:
    """Classify a string by dominant Unicode script.

    Han/hira/kata co-occurrence collapses to ``"japanese-mix"`` so
    per-script alias rules can apply Japanese-wide kana folding. For
    other scripts the majority bucket wins; ties favor the first
    non-``other`` bucket encountered in the input.

    Returns one of: ``han``, ``hira``, ``kata``, ``japanese-mix``,
    ``latin``, ``cyrillic``, ``hangul``, ``other``.
    """
    if not text:
        return "other"

    counts: dict[str, int] = {}
    for ch in text:
        bucket = _classify_codepoint(ord(ch))
        counts[bucket] = counts.get(bucket, 0) + 1

    japanese_buckets = [b for b in ("han", "hira", "kata") if counts.get(b, 0) > 0]
    if len(japanese_buckets) >= 2:
        return "japanese-mix"

    # Drop "other" (whitespace/punctuation) before picking the winner.
    significant = {b: c for b, c in counts.items() if b != "other"}
    if not significant:
        return "other"

    # Pick the script with the highest count; deterministic on ties via
    # bucket name sort (rare in practice).
    return max(significant.items(), key=lambda kv: (kv[1], kv[0]))[0]


# --- Mechanical alias expansion -------------------------------------------


def _hira_to_kata(s: str) -> str:
    out: list[str] = []
    for ch in s:
        cp = ord(ch)
        if 0x3041 <= cp <= 0x3096:
            out.append(chr(cp + 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _kata_to_hira(s: str) -> str:
    out: list[str] = []
    for ch in s:
        cp = ord(ch)
        if 0x30A1 <= cp <= 0x30F6:
            out.append(chr(cp - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _strip_diacritics(s: str) -> str:
    """NFKD-decompose then drop combining marks. Keeps base letters."""
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def expand_aliases_mechanical(canonical: str, script: str) -> tuple[str, ...]:
    """Return the deduplicated mechanical aliases for a canonical term.

    Per-script rules:

    * ``hira`` / ``kata`` / ``japanese-mix`` — apply both hira→kata
      and kata→hira folding to the entire string. Han characters pass
      through unchanged. This catches the common "transcripts spell a
      character name in katakana" failure mode without needing LLM
      help.
    * ``han`` — identity only. Yomi (reading) variants are not
      mechanically derivable from kanji; if the LLM supplies them
      they will be unioned in by the caller.
    * ``latin`` — case-fold, NFKD diacritic strip, hyphen/underscore
      removal and replacement with space. These are safe substring
      transforms for FTS.
    * other scripts — identity only. Cyrillic / Hangul / "other" can
      grow rules later as evidence accrues.

    Always preserves ``canonical`` as the first alias.
    """
    seen: dict[str, None] = {canonical: None}

    def _add(value: str) -> None:
        if value and value not in seen:
            seen[value] = None

    if script in ("hira", "kata", "japanese-mix"):
        _add(_hira_to_kata(canonical))
        _add(_kata_to_hira(canonical))
    elif script == "latin":
        # Case folding produces the lowercase form via Unicode rules.
        _add(canonical.casefold())
        # Diacritic strip, both with original case and lowercase.
        stripped = _strip_diacritics(canonical)
        _add(stripped)
        _add(stripped.casefold())
        # Hyphen/underscore → space, and joined-no-separator form.
        for sep in ("-", "_"):
            if sep in canonical:
                _add(canonical.replace(sep, " "))
                _add(canonical.replace(sep, ""))

    return tuple(seen.keys())


# --- Structured transform -------------------------------------------------


def _coerce_str_list(value: object) -> tuple[str, ...]:
    """Pull a list of non-empty strings out of an LLM JSON field."""
    if not isinstance(value, list):
        return ()
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                out.append(stripped)
    return tuple(out)


def _build_required_term(
    raw: object,
    *,
    llm_alias_pool: list[str] | None = None,
) -> RequiredTerm | None:
    """Validate a single required-term entry from the LLM response.

    Returns None when the entry is malformed (missing/empty canonical,
    wrong types). The caller filters Nones out before constructing the
    final tuple. ``llm_alias_pool`` is unused but reserved for future
    cross-term alias dedup work.
    """
    if not isinstance(raw, dict):
        return None
    canonical_raw = raw.get("canonical")
    if not isinstance(canonical_raw, str):
        return None
    canonical = canonical_raw.strip()
    if not canonical:
        return None

    # Python re-detects script. The LLM's claim is logged but ignored.
    detected = detect_script(canonical)
    llm_script = raw.get("script")
    if isinstance(llm_script, str) and llm_script.strip().lower() not in _VALID_SCRIPTS:
        logger.debug(
            "LLM emitted unknown script label %r for %r; using %r",
            llm_script, canonical, detected,
        )
    elif (
        isinstance(llm_script, str)
        and llm_script.strip().lower() != detected
    ):
        logger.debug(
            "LLM script %r mismatched detected %r for %r; using detected",
            llm_script, detected, canonical,
        )

    # Mechanical aliases come first; LLM aliases are unioned on top so
    # the deduplication preserves the mechanical baseline.
    seen: dict[str, None] = {}
    for alias in expand_aliases_mechanical(canonical, detected):
        seen[alias] = None
    for alias in _coerce_str_list(raw.get("aliases")):
        seen[alias] = None

    return RequiredTerm(
        canonical=canonical,
        script=detected,
        aliases=tuple(seen.keys()),
    )


def _structured_from_legacy_keywords(keywords: str) -> StructuredQuery:
    """Map a pre-Phase-1 ``{"keywords": "a b c"}`` payload to StructuredQuery.

    Used as a transitional fallback when the LLM happens to follow the
    older prompt schema. Every word becomes a semantic term; nothing
    is hard-filtered. Intentionally conservative — we'd rather lose
    precision than drop true positives during the schema migration.
    """
    cleaned = filter_keywords(keywords.strip())
    if not cleaned:
        return StructuredQuery.passthrough(keywords.strip() or keywords)
    parts = tuple(t for t in cleaned.split() if t)
    return StructuredQuery(
        required=(),
        semantic=parts,
        raw_keywords=cleaned,
    )


def _build_raw_keywords(
    required: tuple[RequiredTerm, ...],
    semantic: tuple[str, ...],
) -> str:
    """Project the structured form back into a flat keyword string.

    Required canonicals are listed first (so the FTS path still gives
    them the implicit AND weight when consumed by an unmigrated caller)
    followed by semantic terms. Order within each group is preserved
    from the LLM output.
    """
    parts = [t.canonical for t in required]
    parts.extend(semantic)
    return " ".join(p for p in parts if p)


async def transform_query_structured(
    natural_query: str,
    *,
    temperature: float | None = None,
) -> StructuredQuery:
    """Rewrite a natural-language question as a StructuredQuery.

    See module docstring for the contract. On any failure mode this
    function returns ``StructuredQuery.passthrough(natural_query)`` so
    the caller never has to handle ``None`` or exceptions.

    Args:
        natural_query: The raw user question, length-validated by
            the caller.
        temperature: Optional override for the LLM temperature.

    Returns:
        A StructuredQuery. ``required`` is empty whenever the LLM was
        unavailable, returned a malformed shape, or classified the
        whole query as semantic.
    """
    stripped = natural_query.strip()
    if not stripped:
        return StructuredQuery.passthrough(natural_query)

    try:
        llm = get_llm_client()
    except RuntimeError:
        return StructuredQuery.passthrough(natural_query)

    if not llm.enabled:
        return StructuredQuery.passthrough(natural_query)

    user_prompt = f"<user_question>\n{stripped}\n</user_question>"
    raw = await llm.generate_json(
        _SYSTEM_PROMPT,
        user_prompt,
        max_tokens_override=_QUERY_TRANSFORM_MAX_TOKENS,
        temperature=temperature,
    )

    if not isinstance(raw, dict):
        logger.debug("Structured query transform: non-dict response, passthrough")
        return StructuredQuery.passthrough(natural_query)

    # Transitional path: if the LLM still emits the legacy flat-keywords
    # schema, demote everything to semantic rather than failing.
    if (
        "keywords" in raw
        and "required" not in raw
        and "semantic" not in raw
        and isinstance(raw["keywords"], str)
    ):
        return _structured_from_legacy_keywords(raw["keywords"])

    raw_required = raw.get("required")
    if not isinstance(raw_required, list):
        raw_required = []
    required_terms: list[RequiredTerm] = []
    for entry in raw_required:
        term = _build_required_term(entry)
        if term is not None:
            required_terms.append(term)

    semantic = _coerce_str_list(raw.get("semantic"))

    # Whole-response empty check: if the LLM produced no usable signal
    # at all, fall back to passthrough so the retriever has something
    # to embed instead of an empty string.
    if not required_terms and not semantic:
        return StructuredQuery.passthrough(natural_query)

    required_tuple = tuple(required_terms)
    raw_keywords = _build_raw_keywords(required_tuple, semantic)
    return StructuredQuery(
        required=required_tuple,
        semantic=semantic,
        raw_keywords=raw_keywords,
    )


# --- Legacy flat-string surface ------------------------------------------


# --- Phase 4 fallback ladder --------------------------------------------


def iter_required_fallback_subsets(
    required: tuple[RequiredTerm, ...],
) -> list[tuple[RequiredTerm, ...]]:
    """Generate the Tier 2/3 fallback ladder for a required-term tuple.

    Phase 4 of the required-keyword hard filter spec
    (``2026-04-30-required-semantic-hybrid-retrieval``). When the full
    ``required`` tuple yields zero hits, the retriever steps through
    progressively-relaxed subsets before falling back to Tier 3
    (no required filter at all).

    The drop order uses alias count as a "genericness" proxy: a term
    with many alias variants tends to be a common-noun-like word the
    LLM elaborated, while a term with one alias is a tight proper
    noun the user intends to anchor the query on. Dropping
    most-aliased first keeps the most distinctive term the longest.
    Ties are broken by position — later-positioned terms drop first
    so the user's leading concept is preserved. This matches the
    structured-prompt convention of "list the most central term
    first".

    The returned list does NOT include the full input tuple — the
    caller has already tried that and only invokes this ladder on
    zero results. The terminal step is the empty tuple ``()`` which
    represents Tier 3 ("demote all required to semantic").

    Args:
        required: The Tier 1 required tuple that already yielded
            zero results.

    Returns:
        Ordered ladder of subsets from N-1 → 0 terms. Empty input
        returns ``[()]`` (only the Tier 3 terminal).
    """
    n = len(required)
    if n <= 1:
        return [()]

    # Order indices by (alias_count desc, position desc) so the term
    # we want to drop FIRST sorts to the front of ``drop_order``.
    drop_order = sorted(
        range(n),
        key=lambda i: (-len(required[i].aliases), -i),
    )

    ladder: list[tuple[RequiredTerm, ...]] = []
    dropped_indices: set[int] = set()
    for idx in drop_order:
        dropped_indices.add(idx)
        survivors = tuple(
            required[i] for i in range(n) if i not in dropped_indices
        )
        ladder.append(survivors)
    return ladder


async def transform_query(
    natural_query: str,
    *,
    temperature: float | None = None,
) -> str:
    """Rewrite a natural-language question as a flat keyword string.

    Backwards-compatible wrapper around ``transform_query_structured``.
    Required canonicals are joined with semantic terms by spaces and
    passed through ``filter_keywords`` so the legacy blocklist still
    catches leaked question / file-type words. On any failure path the
    raw natural query is returned unchanged.
    """
    stripped = natural_query.strip()
    if not stripped:
        return natural_query

    structured = await transform_query_structured(
        natural_query, temperature=temperature
    )

    # ``passthrough`` form (semantic == (raw_query,), required empty)
    # collapses to the raw query string here so the legacy contract
    # ("on any failure return the original query") holds.
    if not structured.required and structured.semantic == (stripped,):
        return natural_query

    candidate = structured.raw_keywords.strip()
    if not candidate:
        return natural_query

    filtered = filter_keywords(candidate)
    if filtered:
        return filtered

    logger.debug(
        "Query transform: filtered output empty, falling back to raw query"
    )
    return natural_query
