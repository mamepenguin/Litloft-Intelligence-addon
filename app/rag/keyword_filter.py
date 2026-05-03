"""Server-side blocklist filter for transformed search keywords.

The Stage 1 LLM ``transform_query`` is *asked* to drop file-type words
("写真", "動画") and question words ("何", "共通点"). When it complies,
this filter is a no-op. When it doesn't (gemma4:e2b raw fallback,
malformed JSON, etc.), the leaked tokens poison the FTS5 AND-query and
suppress legitimate hits — case 005 (cross-modal) is the canonical
failure mode.

This module applies the same blocklist that the eval harness measures
violations against (``app/blocklists/*.txt``). After this filter, eval
reports should show ``must_exclude_violations: 0`` for every case
because the LLM's misbehaviour is silently corrected before retrieval.

Normalisation matches ``app.evals.text_match`` (NFKC + casefold +
hira→kata + plain substring) so a blocklist entry like "写真" catches
the LLM's "写真" or "ｼｬｼﾝ" or anything that normalises to the same
form. Tokens are split on whitespace; if any normalised substring of
a token matches a normalised blocklist entry, the whole token is
dropped.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# Shared dir with text_match.py; defined here to avoid app/rag depending
# on app/evals (eval is dev-time only and shouldn't be in the runtime
# import graph).
_BLOCKLIST_DIR = Path(__file__).resolve().parent.parent / "blocklists"


def _hira_to_kata(s: str) -> str:
    out: list[str] = []
    for ch in s:
        cp = ord(ch)
        if 0x3041 <= cp <= 0x3096:
            out.append(chr(cp + 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _normalize(text: str) -> str:
    if not text:
        return ""
    import unicodedata

    s = unicodedata.normalize("NFKC", text)
    s = s.casefold()
    return _hira_to_kata(s)


def _parse_blocklist(text: str) -> tuple[str, ...]:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(stripped)
    return tuple(out)


@lru_cache(maxsize=4)
def _load_blocklist(name: str) -> tuple[str, ...]:
    path = _BLOCKLIST_DIR / f"{name}.txt"
    if not path.exists():
        return ()
    return _parse_blocklist(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _blocked_normals() -> tuple[str, ...]:
    """Cached union of normalised blocklist entries from both files."""
    raw = _load_blocklist("question_words") + _load_blocklist("file_type_words")
    return tuple(_normalize(w) for w in raw if w)


def is_blocked(term: str) -> bool:
    """Return True iff ``term`` normalises to a blocklist substring.

    Single-token check used by ``query_transform._build_required_term``
    to drop file-type / question words that the LLM mis-classified as
    a ``required`` canonical. Same normalisation as
    ``filter_keywords`` so a blocklist entry like ``video`` matches
    ``Video`` / ``ＶＩＤＥＯ`` etc.
    """
    if not term:
        return True
    blocked = _blocked_normals()
    if not blocked:
        return False
    norm = _normalize(term)
    if not norm:
        return True
    return any(b and b in norm for b in blocked)


def filter_keywords(keywords: str) -> str:
    """Drop tokens that match any blocklist entry; preserve order otherwise.

    Tokens are split on whitespace. A token is dropped iff any blocklist
    entry (after the same NFKC + casefold + hira→kata normalisation) is
    a substring of the normalised token — this catches "写真集" when the
    blocklist contains "写真", which is the right behaviour for FTS
    poisoning (the substring is what FTS will tokenise on).

    Returns the rejoined keyword string. If filtering empties the
    result, returns the empty string — callers should fall back to the
    raw query (matching ``transform_query``'s existing graceful path).
    """
    if not keywords or not keywords.strip():
        return ""
    blocked = _blocked_normals()
    if not blocked:
        return keywords

    kept: list[str] = []
    for token in keywords.split():
        norm = _normalize(token)
        if not norm:
            continue
        if any(b and b in norm for b in blocked):
            continue
        kept.append(token)
    return " ".join(kept)
