"""Text normalisation + substring matching for the eval harness (Phase F).

Design per spec §"マッチング正規化規約":

1. NFKC normalisation (folds full-width ASCII, compatibility forms).
2. ASCII lowercasing (casefold on Latin only; CJK is untouched).
3. Hiragana → Katakana fold (unify kana without breaking non-Japanese text).
4. Plain substring match (no token boundary check; Japanese has no spaces).

Global blocklists live in ``app/evals/blocklists/*.txt`` and are loaded once
per run. Each line is one entry; ``#`` starts a comment, blank lines are
skipped. The blocklist hash is embedded into reports so diffing two reports
reveals whether the blocklist itself changed between runs.
"""

from __future__ import annotations

import hashlib
import unicodedata
from functools import lru_cache
from pathlib import Path

BLOCKLIST_DIR = Path(__file__).resolve().parent / "blocklists"


def _hira_to_kata(s: str) -> str:
    """Fold hiragana codepoints to katakana (U+3041..U+3096 → +0x60)."""
    out: list[str] = []
    for ch in s:
        cp = ord(ch)
        if 0x3041 <= cp <= 0x3096:
            out.append(chr(cp + 0x60))
        else:
            out.append(ch)
    return "".join(out)


def normalize(text: str) -> str:
    """NFKC + ASCII casefold + hira→kata, for substring comparison."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = s.casefold()
    s = _hira_to_kata(s)
    return s


def contains(haystack: str, needle: str) -> bool:
    """Substring match after normalising both sides."""
    if not needle:
        return False
    return normalize(needle) in normalize(haystack)


def coverage(needles: tuple[str, ...], haystack: str) -> float:
    """Fraction of ``needles`` that substring-match in ``haystack``."""
    if not needles:
        return 1.0
    h = normalize(haystack)
    hits = sum(1 for n in needles if n and normalize(n) in h)
    return hits / len(needles)


def violation_count(needles: tuple[str, ...], haystack: str) -> int:
    """Count of forbidden substrings present (each needle counted once)."""
    if not needles:
        return 0
    h = normalize(haystack)
    return sum(1 for n in needles if n and normalize(n) in h)


# --------------------------------------------------------------------------- #
# Blocklist loading
# --------------------------------------------------------------------------- #


def _parse_blocklist(text: str) -> tuple[str, ...]:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(stripped)
    return tuple(out)


@lru_cache(maxsize=8)
def load_blocklist(name: str) -> tuple[str, ...]:
    """Load a blocklist file from ``blocklists/<name>.txt``.

    Results are cached per-process; the runner re-imports per run so
    stale cache is not a concern. Returns () if the file is absent.
    """
    path = BLOCKLIST_DIR / f"{name}.txt"
    if not path.exists():
        return ()
    return _parse_blocklist(path.read_text(encoding="utf-8"))


def blocklist_sha256(name: str) -> str:
    """sha256 of the raw file contents (empty string → empty hash input)."""
    path = BLOCKLIST_DIR / f"{name}.txt"
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def global_exclude_terms() -> tuple[str, ...]:
    """Union of every bundled blocklist (question + file-type words)."""
    return load_blocklist("question_words") + load_blocklist("file_type_words")
