"""Tell a decimal point or digit-group comma from sentence punctuation."""

from __future__ import annotations

_SEPARATORS = frozenset(".,")
_DIGITS = frozenset("0123456789")


def is_digit_separator(prev_text: str | None, text: str, next_text: str | None) -> bool:
    """True when ``text`` ends in ``.``/``,`` sitting between two digits.

    The separator is the last character of ``text``; the digit before it
    is in ``text`` itself, or the last character of ``prev_text`` when
    ``text`` is the separator alone. ``next_text`` must start with a digit.
    Rows are compared stripped of surrounding whitespace.
    """
    current = text.strip()
    following = (next_text or "").strip()
    if not current or current[-1] not in _SEPARATORS:
        return False
    if not following or following[0] not in _DIGITS:
        return False
    before = current[:-1] or (prev_text or "").strip()
    return bool(before) and before[-1] in _DIGITS


def splits_number(text: str, pos: int) -> bool:
    """True when a line break before ``text[pos]`` would cut a number in two.

    A number is a run of digits with single ``.``/``,`` between digits; a
    space after such a separator (``3. 3km``, rows joined by a space) stays
    inside it, as ``is_digit_separator`` treats those rows.
    """
    if pos <= 0 or pos >= len(text):
        return False
    before, ch = text[:pos], text[pos]
    if ch in _SEPARATORS:
        return before[-1] in _DIGITS and text[pos + 1 : pos + 2] in _DIGITS
    if ch not in _DIGITS:
        return False
    if before[-1] in _DIGITS:
        return True
    if before[-1] == " ":
        before = before[:-1]
    return len(before) >= 2 and before[-1] in _SEPARATORS and before[-2] in _DIGITS
