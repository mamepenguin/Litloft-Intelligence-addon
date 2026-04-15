"""Pack Whisper word-level timestamps into WebVTT cues.

Transcript chunks (10–30 s) are optimal for embedding but unreadable as
subtitles; subtitle cues need 1–5 s duration and a bounded character
width per line. This module derives the latter from the former by
repacking word rows, honouring sentence boundaries, silence gaps, and
east-asian display width.

Reference: docs/superpowers/specs/2026-04-15-whisper-word-level-subtitles.md
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Module-level janome tokenizer. Lazily constructed on first JA cue build,
# then reused for subsequent calls — construction is the slow part (~100ms),
# tokenization of a 12-minute transcript runs in <50ms. ``None`` means the
# tokenizer isn't available (import failure) and callers should fall back
# to the character-level behaviour.
_ja_tokenizer: Any | None = None
_ja_tokenizer_tried: bool = False


def _get_ja_tokenizer() -> Any | None:
    global _ja_tokenizer, _ja_tokenizer_tried
    if _ja_tokenizer_tried:
        return _ja_tokenizer
    _ja_tokenizer_tried = True
    try:
        from janome.tokenizer import Tokenizer

        _ja_tokenizer = Tokenizer()
    except Exception as e:
        logger.info("janome unavailable, falling back to char-level cues: %s", e)
        _ja_tokenizer = None
    return _ja_tokenizer

_PUNCT_BREAK = frozenset(".。!?！？…")
_PUNCT_SOFT = frozenset(",、;:：；")
# Characters that must never appear at the start of a cue line. Small kana
# and the prolonged-sound mark bind to the preceding mora; leading
# punctuation is also visually awkward. Used to rewind width-capped flushes
# to a linguistically safe position.
_NO_BREAK_BEFORE = frozenset(
    "ーゝゞ々"
    "ぁぃぅぇぉっゃゅょゎ"
    "ァィゥェォッャュョヮヵヶ"
    "、。！？!?.,;:：；…"
    "」』）)】〉》"
)


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


def _should_attach_to_previous(pos: str) -> bool:
    """Heuristic bunsetsu post-merge: decide whether a token glues to its
    predecessor to form a phrasal unit.

    Janome returns morphemes; subtitle readers want phrasal chunks. These
    categories bind to the preceding content word in practice:

    - 助詞 (particles: は/が/を/に/で/と/も/…): postpositions, always bind
    - 助動詞 (auxiliary verbs: です/ます/た/ない/…): verb endings, bind
    - 接尾 (suffixes: さん/的/化/性/…): bind by definition
    - 記号 (punctuation, brackets): visually awkward to start a line with

    Everything else (名詞/動詞/形容詞/副詞/接続詞/連体詞/感動詞) is a
    phrase head and starts a new bunsetsu.
    """
    if not pos:
        return False
    head = pos.split(",", 1)[0]
    if head in ("助詞", "助動詞", "記号"):
        return True
    # "接尾" can appear at any position in the feature tuple.
    return "接尾" in pos


def _regroup_ja_with_janome(words: list[dict]) -> list[dict]:
    """Re-group char/subword-level timestamps along morpheme boundaries.

    Japanese transcripts from Whisper ship as subword tokens ("ド",
    "ラクエ", …) and the refine path emits pure characters. Neither grid
    aligns with linguistic word boundaries, so width-capped cue flushes
    land mid-word even with the safe-break rewind. Tokenising the joined
    text with janome and re-emitting one word per morpheme fixes this at
    the root — downstream cue packing then only ever splits between
    complete words.

    Timestamps are interpolated proportionally across each input word's
    characters (Whisper tokens don't carry per-char timing) and the
    resulting janome token spans the start/end of its first/last char.
    """
    if not words:
        return words
    tokenizer = _get_ja_tokenizer()
    if tokenizer is None:
        return words

    # Build a char-indexed timeline. Each entry carries the char itself
    # and its interpolated [start, end] window.
    char_times: list[tuple[str, float, float]] = []
    for w in words:
        text = (w.get("text") or "")
        if not text:
            continue
        try:
            ts = float(w["timestamp_start"])
            te = float(w["timestamp_end"])
        except (KeyError, TypeError, ValueError):
            continue
        n = len(text)
        if n <= 0 or te < ts:
            continue
        span = te - ts
        for ci, ch in enumerate(text):
            char_times.append((
                ch,
                ts + span * ci / n,
                ts + span * (ci + 1) / n,
            ))

    if not char_times:
        return words

    joined = "".join(c[0] for c in char_times)
    try:
        # POS-tagged output so the bunsetsu post-merge below can tell
        # particles / auxiliaries from content words.
        tagged = [
            (tok.surface, tok.part_of_speech)
            for tok in tokenizer.tokenize(joined)
            if tok.surface
        ]
    except Exception as e:
        logger.warning("janome tokenize failed, falling back: %s", e)
        return words

    result: list[dict] = []
    cursor = 0
    total = len(char_times)
    for surface, pos in tagged:
        tok_len = len(surface)
        if tok_len <= 0 or cursor >= total:
            break
        # Clamp to available chars in case of mismatched lengths (shouldn't
        # happen for wakati but be defensive).
        end_pos = min(cursor + tok_len, total)
        if end_pos <= cursor:
            break
        start_ts = char_times[cursor][1]
        end_ts = char_times[end_pos - 1][2]
        # Bunsetsu-style merge: particles, auxiliaries, suffixes and
        # punctuation glue to the preceding content word. Timestamps
        # stay exact because adjacent morphemes share char-timeline
        # boundaries (end_prev == start_curr).
        if result and _should_attach_to_previous(pos):
            prev = result[-1]
            prev["text"] = prev["text"] + surface
            prev["timestamp_end"] = end_ts
        else:
            result.append({
                "text": surface,
                "timestamp_start": start_ts,
                "timestamp_end": end_ts,
            })
        cursor = end_pos

    return result or words


def _first_char(word: dict) -> str:
    text = (word.get("text") or "").strip()
    return text[0] if text else ""


def _can_break_before(word: dict) -> bool:
    """Return False when splitting a cue just before ``word`` would leave
    a dangling prolonged-sound mark, small kana, or closing punctuation.
    """
    ch = _first_char(word)
    return bool(ch) and ch not in _NO_BREAK_BEFORE


def _safe_break_between(prev: dict, nxt: dict) -> bool:
    """Check a cue boundary between two adjacent word tokens.

    Rejects leading-prolongation / small-kana cases and also refuses to
    split a run of consecutive katakana, which is almost always a single
    loanword (e.g. ``ドラクエ``) even though Whisper may emit it as two
    subword tokens.
    """
    if not _can_break_before(nxt):
        return False
    prev_text = (prev.get("text") or "").strip()
    nxt_text = (nxt.get("text") or "").strip()
    if prev_text and nxt_text and _is_katakana(prev_text[-1]) and _is_katakana(nxt_text[0]):
        return False
    return True


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

    When a width cap forces a mid-sentence flush we rewind to the most
    recent "safe" break position (after punctuation, after a silence
    gap, after an ASCII space) so CJK cues don't split mid-word between
    a kana and its prolonged-sound mark.
    """
    if not words:
        return []
    cfg = config or CueConfig()

    # Japanese: re-group subword / per-char timestamps into morpheme-level
    # words via janome so cue breaks land on real word boundaries.
    if (language or "").lower().startswith("ja"):
        words = _regroup_ja_with_janome(words)
        if not words:
            return []

    cues: list[dict] = []
    current: list[dict] = []
    # Indices into ``current`` where splitting is linguistically safe
    # (i.e. current[:idx] is a complete phrase; current[idx:] carries
    # forward). Populated on punctuation / silence-gap boundaries.
    safe_breaks: list[int] = []
    cue_start = float(words[0]["timestamp_start"])

    def emit(upto: int, end_time: float) -> None:
        """Flush ``current[:upto]`` as a cue ending at ``end_time``."""
        nonlocal current, safe_breaks, cue_start
        if upto <= 0 or not current:
            return
        head = current[:upto]
        tail = current[upto:]
        tokens = [w["text"].strip() for w in head if w["text"].strip()]
        if tokens:
            text = _join_for_language(tokens, language)
            cues.append({"start": cue_start, "end": end_time, "text": text})
        current = tail
        safe_breaks = [b - upto for b in safe_breaks if b > upto]
        if current:
            cue_start = float(current[0]["timestamp_start"])

    def flush_with_rewind(fallback_end: float) -> None:
        """Flush the accumulated cue, preferring a recorded safe break."""
        if not current:
            return
        # Walk safe breaks from newest to oldest; use the first one that
        # keeps the cue above half the max width (avoids trailing tiny
        # fragments) and whose carry-forward first word is breakable.
        half_width = cfg.max_width / 2
        for cand in reversed(safe_breaks):
            if cand <= 0 or cand >= len(current):
                continue
            head = current[:cand]
            head_tokens = [w["text"].strip() for w in head if w["text"].strip()]
            head_text = _join_for_language(head_tokens, language)
            if _display_width(head_text) < half_width:
                continue
            if not _safe_break_between(current[cand - 1], current[cand]):
                continue
            emit(cand, float(current[cand - 1]["timestamp_end"]))
            return
        # No recorded candidate worked — scan backwards for any position
        # whose successor is breakable, then fall back to hard flush.
        for cand in range(len(current) - 1, 0, -1):
            if _safe_break_between(current[cand - 1], current[cand]):
                emit(cand, float(current[cand - 1]["timestamp_end"]))
                return
        emit(len(current), fallback_end)

    for i, word in enumerate(words):
        # When the previous iteration flushed the entire cue, cue_start
        # is stale — reset it before we start building the next cue.
        if not current:
            cue_start = float(word["timestamp_start"])
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
        has_space = " " in word_text  # ASCII/EN segmentation hint

        # Record a safe-break candidate AFTER this word whenever the
        # boundary is linguistically clean.
        if ends_hard or ends_soft or gap >= cfg.silence_gap or has_space:
            if i + 1 < len(words) and _safe_break_between(word, words[i + 1]):
                safe_breaks.append(len(current))

        hard_boundary = (
            duration >= cfg.min_duration
            and (ends_hard or gap >= cfg.silence_gap)
        )
        soft_boundary = (
            duration >= cfg.min_duration
            and ends_soft
            and current_width >= cfg.max_width * 0.75
        )

        if duration >= cfg.max_duration or current_width >= cfg.max_width:
            flush_with_rewind(word_end)
        elif hard_boundary or soft_boundary:
            emit(len(current), word_end)

    if current:
        emit(len(current), float(current[-1]["timestamp_end"]))

    return [_balance_two_lines(c, cfg.max_width // 2) for c in cues]


def _is_katakana(ch: str) -> bool:
    return bool(ch) and "\u30a0" <= ch <= "\u30ff"


def _janome_break_position(text: str, target: int) -> int | None:
    """Pick the janome token boundary closest to ``target`` char index.

    Returns ``None`` when the tokenizer is unavailable or the best
    boundary degenerates to 0 / len(text) (no useful split).
    """
    tokenizer = _get_ja_tokenizer()
    if tokenizer is None:
        return None
    try:
        tagged = [
            (tok.surface, tok.part_of_speech)
            for tok in tokenizer.tokenize(text)
            if tok.surface
        ]
    except Exception:
        return None
    if len(tagged) < 2:
        return None
    # Bunsetsu-aware boundaries: skip positions where the next token glues
    # to its predecessor (particle / auxiliary / suffix / punctuation).
    boundaries: list[int] = []
    pos = 0
    for i, (surface, token_pos) in enumerate(tagged[:-1]):
        pos += len(surface)
        next_pos = tagged[i + 1][1]
        if _should_attach_to_previous(next_pos):
            continue
        boundaries.append(pos)
    if not boundaries:
        return None
    # Pick the boundary nearest the target; ties go to the later one so
    # the first line doesn't run short.
    best = min(boundaries, key=lambda b: (abs(b - target), -b))
    return best


def _adjust_cjk_break(text: str, idx: int) -> int:
    """Nudge a midpoint break index outward to a safer CJK position.

    Avoids splitting in the middle of a katakana run (usually a single
    loanword) and never places the break right before a prolonged-sound
    mark or small kana. Returns the original index when no better
    position exists within the search window.
    """
    if idx <= 0 or idx >= len(text):
        return idx
    limit = min(len(text), max(4, len(text) // 4))

    def _bad(pos: int) -> bool:
        if pos <= 0 or pos >= len(text):
            return True
        if text[pos] in _NO_BREAK_BEFORE:
            return True
        # Don't break between two consecutive katakana (same loanword).
        if _is_katakana(text[pos - 1]) and _is_katakana(text[pos]):
            return True
        return False

    if not _bad(idx):
        return idx
    for step in range(1, limit + 1):
        if idx + step < len(text) and not _bad(idx + step):
            return idx + step
        if idx - step > 0 and not _bad(idx - step):
            return idx - step
    return idx


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

    target = len(text) // 2
    mid = _janome_break_position(text, target)
    if mid is None or not (0 < mid < len(text)):
        mid = _adjust_cjk_break(text, target)
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
