"""Language-aware text normalization for fair WER/CER scoring.

Phase 2C: provider outputs vary in punctuation style, character set
(katakana vs hiragana vs hankaku), and surface-level whitespace. We
collapse those superficial differences before metric calculation so
WER/CER reflect actual transcription quality, not formatting choices.

Supported languages today: ``ja``, ``en``. Other language tags raise
``ValueError`` so the caller can record the case as failed without
running through a half-applicable normalizer (R0 spec M3 expects the
runner to catch this).
"""

from __future__ import annotations

import re
import unicodedata

import jaconv

# Punctuation / whitespace stripped before WER/CER. Keeping this list
# small but covering the most common style differences between
# providers (Whisper-family loves "、。", AssemblyAI / Deepgram favour
# ASCII commas, Gemini sometimes adds quotes).
_PUNCT_RE = re.compile(
    r"[、。，．！？「」『』（）()\[\]{}…――‐\-—–·,.;:!?\"'`~_]+"
)
_WS_RE = re.compile(r"\s+")

_SUPPORTED_LANGUAGES = ("ja", "en")


def normalize(text: str, language: str) -> str:
    """Collapse stylistic differences in ``text`` for fair scoring.

    Steps:
    1. NFKC unicode normalization (full-width → half-width, compose).
    2. ASCII lowercase (matters for English).
    3. Japanese only: katakana → hiragana via :mod:`jaconv` so
       providers that prefer one script over the other still match.
    4. Strip punctuation and collapse whitespace.

    Other CJK languages would need their own folding rule, so we
    refuse them up-front instead of silently producing wrong scores.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"normalize() expects str, got {type(text).__name__}"
        )
    code = (language or "").lower().split("-")[0]
    if code not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported eval language {language!r}. "
            f"Supported: {_SUPPORTED_LANGUAGES!r}."
        )

    out = unicodedata.normalize("NFKC", text)
    out = out.lower()
    if code == "ja":
        out = jaconv.kata2hira(out)
    out = _PUNCT_RE.sub(" ", out)
    out = _WS_RE.sub(" ", out)
    return out.strip()
