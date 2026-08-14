"""Safe prompt instructions derived from the configured output language."""

from __future__ import annotations

import re


_BCP47_TAG = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")


def configured_language_requirement(
    output_language: str | None,
    *,
    auto_requirement: str,
) -> str:
    """Build a model-facing requirement without hard-coding a language.

    Explicit values are interpreted as IETF BCP 47 language tags. ``auto``,
    blank, and invalid values use the feature-specific inference rule instead.
    Validation prevents an operator-controlled config value from becoming an
    arbitrary prompt fragment.
    """
    language_tag = (output_language or "auto").strip()
    if language_tag.lower() == "auto" or not _BCP47_TAG.fullmatch(language_tag):
        return auto_requirement
    return (
        "Use the language identified by the user-configured IETF BCP 47 "
        f'language tag "{language_tag}". '
        "Do not choose a different output language."
    )


__all__ = ["configured_language_requirement"]
