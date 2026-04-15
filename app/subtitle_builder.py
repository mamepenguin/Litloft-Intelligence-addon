"""Pack Whisper word-level timestamps into WebVTT cues.

Transcript chunks (10–30 s) are optimal for embedding but unreadable as
subtitles; subtitle cues need 1–5 s duration and a bounded character
width per line. This module derives the latter from the former by
repacking word rows, honouring sentence boundaries, silence gaps, and
east-asian display width.

Reference: docs/superpowers/specs/2026-04-15-whisper-word-level-subtitles.md
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

_PUNCT_BREAK = frozenset(".。!?！？…")
_PUNCT_SOFT = frozenset(",、;:：；")


@dataclass(frozen=True)
class CueConfig:
    max_duration: float = 5.0
    max_width: int = 42  # display columns (2 for CJK, 1 for ASCII)
    min_duration: float = 0.6
    silence_gap: float = 0.6  # gap between cues that forces a break


def _display_width(text: str) -> int:
    """Return the approximate display width of ``text`` in monospace columns.

    Follows the Unicode East Asian Width property: F (Fullwidth), W (Wide)
    count as 2, everything else as 1. Combining marks are treated as 0.
    Matches typical subtitle-reader assumptions (Netflix / JACP).
    """
    width = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        if unicodedata.east_asian_width(ch) in ("F", "W"):
            width += 2
        else:
            width += 1
    return width


def _join_for_language(tokens: list[str], language: str) -> str:
    lang = (language or "").lower()
    if lang.startswith(("ja", "zh", "ko", "th")):
        return "".join(tokens)
    return " ".join(tokens)


def _format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - hours * 3600 - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def build_cues(
    words: list[dict],
    language: str = "",
    config: CueConfig | None = None,
) -> list[dict]:
    """Group consecutive words into displayable subtitle cues.

    Each input word dict must carry ``text``, ``timestamp_start``, and
    ``timestamp_end``. The output dicts carry ``start``, ``end``, and
    ``text`` with a single ``\\n`` between pseudo-lines when the cue
    exceeds half the max width (keeps two-line subtitles balanced).
    """
    if not words:
        return []
    cfg = config or CueConfig()

    cues: list[dict] = []
    current: list[dict] = []
    cue_start = float(words[0]["timestamp_start"])

    def flush(end_time: float) -> None:
        if not current:
            return
        tokens = [w["text"].strip() for w in current if w["text"].strip()]
        if not tokens:
            return
        text = _join_for_language(tokens, language)
        cues.append({"start": cue_start, "end": end_time, "text": text})

    for i, word in enumerate(words):
        current.append(word)
        word_end = float(word["timestamp_end"])
        duration = word_end - cue_start
        tokens = [w["text"].strip() for w in current if w["text"].strip()]
        current_text = _join_for_language(tokens, language)
        current_width = _display_width(current_text)

        next_start = (
            float(words[i + 1]["timestamp_start"])
            if i + 1 < len(words)
            else word_end
        )
        gap = max(0.0, next_start - word_end)
        word_text = word["text"].strip()
        ends_hard = bool(word_text) and word_text[-1] in _PUNCT_BREAK
        ends_soft = bool(word_text) and word_text[-1] in _PUNCT_SOFT

        should_flush = False
        if duration >= cfg.max_duration:
            should_flush = True
        elif current_width >= cfg.max_width:
            should_flush = True
        elif duration >= cfg.min_duration and (ends_hard or gap >= cfg.silence_gap):
            should_flush = True
        elif duration >= cfg.min_duration and ends_soft and current_width >= cfg.max_width * 0.75:
            should_flush = True

        if should_flush:
            flush(word_end)
            current = []
            if i + 1 < len(words):
                cue_start = float(words[i + 1]["timestamp_start"])

    if current:
        flush(float(current[-1]["timestamp_end"]))

    return [_balance_two_lines(c, cfg.max_width // 2) for c in cues]


def _balance_two_lines(cue: dict, soft_width: int) -> dict:
    """Break long single-line cues into two balanced lines.

    Only triggers when the cue exceeds the soft width. Breaks on space
    for languages that use spaces; for CJK, breaks at punctuation or
    midpoint fallback. Subtle visual polish — pure display concern.
    """
    text: str = cue["text"]
    if _display_width(text) <= soft_width:
        return cue

    if " " in text:
        words = text.split(" ")
        midpoint_width = _display_width(text) // 2
        first: list[str] = []
        running = 0
        for w in words:
            w_width = _display_width(w) + (1 if first else 0)
            if running + w_width > midpoint_width and first:
                break
            first.append(w)
            running += w_width
        second = words[len(first):]
        if first and second:
            return {**cue, "text": " ".join(first) + "\n" + " ".join(second)}
        return cue

    for i, ch in enumerate(text):
        if ch in _PUNCT_SOFT and _display_width(text[: i + 1]) >= soft_width // 2:
            return {**cue, "text": text[: i + 1] + "\n" + text[i + 1 :].lstrip()}

    mid = len(text) // 2
    return {**cue, "text": text[:mid] + "\n" + text[mid:]}


def _sanitise_cue_text(text: str) -> str:
    """Strip characters that would break WebVTT cue framing.

    ``-->`` would be re-interpreted as a timestamp separator and blank
    lines end the cue; neither is a realistic Whisper output but we
    defend against future model quirks or adversarial transcripts.
    Single ``\\n`` is preserved (we emit it for balanced two-line cues).
    """
    sanitised = text.replace("-->", "→")
    # Collapse any sequence of blank lines that would terminate the cue.
    lines = [ln for ln in sanitised.splitlines() if ln.strip() or ln == ""]
    return "\n".join(ln for ln in lines if ln.strip())


def to_webvtt(cues: list[dict], language: str = "") -> str:
    """Serialise cue dicts into a WebVTT document.

    Emits a ``Language:`` header when ``language`` is provided so user
    agents can pick the right font-shaping rules.
    """
    header = "WEBVTT"
    if language:
        header += f"\nLanguage: {language}"
    lines = [header, ""]
    for i, cue in enumerate(cues, start=1):
        text = _sanitise_cue_text(cue["text"])
        if not text:
            continue
        lines.append(str(i))
        lines.append(
            f"{_format_timestamp(cue['start'])} --> {_format_timestamp(cue['end'])}"
        )
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def build_vtt(words: list[dict], language: str = "") -> str:
    """Shortcut: pack words and serialise to WebVTT in one call."""
    return to_webvtt(build_cues(words, language=language), language=language)
