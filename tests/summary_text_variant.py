"""The document (text) system prompts, derived from the transcript goldens.

SPEC-ADDON-175, SPEC-ADDON-176: a document's prompt is the transcript prompt
with a fixed set of substitutions and nothing else, so the expectation is
built by applying those substitutions to the golden rather than kept as a
second golden that could drift with it.
"""

from __future__ import annotations

from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent / "golden_prompts"

LANGS = ("ja", "en", "auto", "fr")

GOLDEN_STEM = {
    "_build_system_prompt": "summaries_short_long_system",
    "_build_detailed_system_prompt": "summaries_detailed_system",
}

_TEXT_PROPER_NOUNS = [
    "## Handling proper nouns",
    "",
    "This content is text, so proper nouns are written as they appear in it.",
    "",
    "- When the content spells one entity in more than one way, use the spelling"
    " in the filename, title or description; otherwise use the most frequent"
    " spelling in the content",
    "- Do not substitute a different spelling, character or reading by guesswork",
]

_SHORT_LONG_NAME_RULE = (
    "- In both short and long, use a person's name when it appears in the"
    " content, filename, title or description (a character, a first-person"
    " narrator, an author, an interviewee); use a role such as the narrator,"
    " the author or the speaker only when the content gives no name"
)

_DETAILED_NAME_RULE = (
    "- Use a person's name when it appears in the content, filename, title or"
    " description (a character, a first-person narrator, an author, an"
    " interviewee). Use a role such as the narrator or the author only when the"
    " content gives no name"
)

_DETAILED_SPANS = (
    (
        "Maintain the narrator's tone and energy but write from an observer"
        " (third-person) perspective",
        "Maintain the source's tone and energy but write from an observer"
        " (third-person) perspective; a first-person narrator is referred to by"
        " name when the content gives one",
    ),
    ("the narrator speculates that ...", "the author speculates that ..."),
    ("the narrator evaluates ... positively", "the author evaluates ... positively"),
    ("the narrator describes it as ...", "the author describes it as ..."),
)


def golden(build: str, lang: str) -> str:
    return (GOLDEN_DIR / f"{GOLDEN_STEM[build]}_{lang}.txt").read_bytes().decode(
        "utf-8"
    )


def _one_index(lines: list[str], match) -> int:
    hits = [i for i, line in enumerate(lines) if match(line)]
    assert len(hits) == 1, f"expected one matching line, found {len(hits)}"
    return hits[0]


def _replace_line(lines: list[str], prefix: str, new: str) -> list[str]:
    i = _one_index(lines, lambda line: line.startswith(prefix))
    return lines[:i] + [new] + lines[i + 1 :]


def text_variant(build: str, transcript_prompt: str) -> str:
    """Apply the step-5 substitutions for ``build`` to its transcript prompt."""
    lines = transcript_prompt.split("\n")

    start = _one_index(lines, lambda line: line == "## Handling proper nouns")
    care = _one_index(lines, lambda line: line == "### Handle with care")
    assert start < care
    end = care + 1
    while end < len(lines) and lines[end].startswith("- "):
        end += 1
    assert end > care + 1 and lines[end] == ""
    lines = lines[:start] + _TEXT_PROPER_NOUNS + lines[end:]

    culinary = _one_index(
        lines,
        lambda line: line.startswith(
            "- For culinary names, technique names, and operation names that"
            " ASR tends to distort"
        ),
    )
    lines = lines[:culinary] + lines[culinary + 1 :]

    if build == "_build_system_prompt":
        lines = _replace_line(
            lines,
            "- In both short and long, write a person's name only if",
            _SHORT_LONG_NAME_RULE,
        )
        return "\n".join(lines)

    lines = _replace_line(
        lines,
        "- Write a person's name only if it appears in the filename",
        _DETAILED_NAME_RULE,
    )
    out = "\n".join(lines)
    for old, new in _DETAILED_SPANS:
        assert out.count(old) == 1, f"expected one {old!r}, found {out.count(old)}"
        out = out.replace(old, new)
    return out
