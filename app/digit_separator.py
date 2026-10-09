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
