"""Markdown parser for detailed_summary citation segmentation.

The detailed_summary output has four canonical ``##`` sections (intro /
bullets / key-points table / proper-noun annotation) plus any sub-sections
the LLM chooses. For citation linking we split the document into
"segments" — either a single bullet (``-``), a single table row (``|``),
or a paragraph (a blank-line-separated run of plain text).

The parser is intentionally lightweight (regex + line split) so we stay
off heavyweight dependencies like ``markdown-it``. Nested bullets are
flattened to one segment per line; their tree structure is not
preserved. Tables are exploded into one segment per *body* row
(the header and separator rows are skipped).

``section_path`` is a human-readable locator used as the primary key
for ``detailed_summary_citations``. Examples:

* ``"全体像/0"``                — first paragraph in the intro section
* ``"主要な章/場面/3"``          — fourth bullet in that section
* ``"重要ポイントまとめ/row/2"`` — third body row of the table

The parser is pure: it never touches a DB or a network. All callers
pass in a string and get back a list of dicts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ``## 見出し`` — leading whitespace tolerated so snippets that got
# indented by the LLM still parse. We deliberately match exactly two
# hashes (level-2 heading) so ``###`` subsections merge into the
# parent section's paragraph run.
_HEADING_RE = re.compile(r"^\s{0,3}##\s+(?P<title>.+?)\s*$")

# ``- item`` or ``* item`` with any indent. The indent is preserved so
# nested bullets get their own segments instead of merging into the
# parent line.
_BULLET_RE = re.compile(r"^\s*[-*]\s+(?P<text>.+?)\s*$")

# ``| col | col |`` — table row. GFM separator rows (``|---|---|``) are
# skipped separately via ``_IS_SEPARATOR_RE``.
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")
_IS_SEPARATOR_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")


@dataclass(frozen=True)
class Segment:
    """One citation-linkable piece of the detailed summary.

    ``section_path`` uniquely identifies the segment within a given
    detailed_summary. ``segment_type`` is either ``"bullet"`` or
    ``"paragraph"`` (table rows are classified as bullets because the
    UI and citation lookup both treat a single row as one atomic unit).
    """

    section_path: str
    segment_type: str  # "bullet" | "paragraph"
    segment_text: str


def parse_segments(markdown: str) -> list[Segment]:
    """Split a detailed_summary Markdown string into citation segments.

    The parser walks line-by-line and tracks the current ``##`` section
    title. Inside each section:

    * Consecutive plain-text lines (no leading ``-``, ``*``, or ``|``)
      accumulate into a paragraph; the paragraph closes on the next
      blank line, bullet, table row, or heading.
    * Each bullet line becomes a standalone ``bullet`` segment.
    * Each table *body* row becomes a standalone ``bullet`` segment
      (the header + separator are excluded).

    ``section_path`` uses the section title plus a per-section counter:

    * Paragraphs:   ``"<section>/<index>"``
    * Bullets:      ``"<section>/<index>"``
    * Table rows:   ``"<section>/row/<index>"``

    Indices restart at 0 within each section. Paragraphs and bullets
    share the same counter so their ordering is preserved in the path.

    Args:
        markdown: Raw detailed_summary text, as stored in
            ``file_summaries.detailed_summary``.

    Returns:
        List of ``Segment`` objects in source order. Empty list if the
        input has no recognisable content.
    """
    if not markdown:
        return []

    segments: list[Segment] = []
    current_section = ""
    # Separate counter for plain segments (paragraph/bullet) vs table rows
    # so the two namespaces don't collide in section_path.
    plain_idx = 0
    row_idx = 0
    # Buffer for an in-progress paragraph. ``None`` means no paragraph
    # is currently open.
    paragraph_buf: list[str] | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph_buf, plain_idx
        if paragraph_buf is None:
            return
        text = " ".join(line.strip() for line in paragraph_buf).strip()
        paragraph_buf = None
        if not text:
            return
        segments.append(
            Segment(
                section_path=f"{current_section}/{plain_idx}",
                segment_type="paragraph",
                segment_text=text,
            )
        )
        plain_idx += 1

    # Table state: once we open a table we need to skip the header line
    # (the first row) and the separator line, then treat following rows
    # as body rows. GFM requires header + separator but we tolerate
    # malformed tables (no separator) by only skipping the separator
    # when it actually matches _IS_SEPARATOR_RE.
    in_table = False
    table_header_consumed = False

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush_paragraph()
            in_table = False
            table_header_consumed = False
            current_section = heading_match.group("title").strip()
            plain_idx = 0
            row_idx = 0
            continue

        # Blank line: terminates paragraph and table.
        if not line.strip():
            flush_paragraph()
            in_table = False
            table_header_consumed = False
            continue

        if _TABLE_ROW_RE.match(line):
            flush_paragraph()
            if _IS_SEPARATOR_RE.match(line):
                # Separator row: mark that the header has been consumed
                # and skip. Subsequent rows are body.
                table_header_consumed = True
                continue
            if not in_table:
                # First row of a new table — treat as header, skip.
                in_table = True
                table_header_consumed = False
                continue
            if not table_header_consumed:
                # Malformed table without a separator row — still skip
                # the implicit second row? No: once we've seen the
                # header, any further row is body. Fall through.
                table_header_consumed = True
            row_text = _flatten_table_row(line)
            if row_text:
                segments.append(
                    Segment(
                        section_path=f"{current_section}/row/{row_idx}",
                        segment_type="bullet",
                        segment_text=row_text,
                    )
                )
                row_idx += 1
            continue

        # Non-table line terminates any in-progress table.
        in_table = False
        table_header_consumed = False

        bullet_match = _BULLET_RE.match(line)
        if bullet_match:
            flush_paragraph()
            bullet_text = bullet_match.group("text").strip()
            if bullet_text:
                segments.append(
                    Segment(
                        section_path=f"{current_section}/{plain_idx}",
                        segment_type="bullet",
                        segment_text=bullet_text,
                    )
                )
                plain_idx += 1
            continue

        # Plain text line — accumulate into the current paragraph.
        if paragraph_buf is None:
            paragraph_buf = [line]
        else:
            paragraph_buf.append(line)

    # EOF: flush any trailing paragraph.
    flush_paragraph()
    return segments


def _flatten_table_row(line: str) -> str:
    """Concatenate the cells of a ``| a | b | c |`` row into one string.

    The leading/trailing ``|`` are stripped and each cell is trimmed;
    cells are joined with `` | `` so the output is still humanly
    readable when it becomes a citation ``segment_text``.
    """
    stripped = line.strip()
    # Drop leading/trailing pipes so split() doesn't produce empty
    # first/last cells.
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in stripped.split("|")]
    cells = [c for c in cells if c]
    return " | ".join(cells)


# ---------------------------------------------------------------------------
# Section replacement (Phase 2: edit)
# ---------------------------------------------------------------------------


def replace_section_body(
    markdown: str, section_heading: str, new_body: str
) -> str:
    """Replace the body of a ``## section_heading`` while preserving layout.

    The body is everything between the target ``##`` line and the next
    ``##`` line (or end-of-string). The heading itself is kept intact so
    the heading-counting logic in ``parse_segments`` stays stable.

    * Trailing whitespace on ``new_body`` is stripped; a single blank
      line is inserted between the heading and the body for readability.
    * If the section is not found, ``ValueError`` is raised — callers
      should catch this and surface a 400.
    * If there are multiple sections with the same heading (shouldn't
      happen in our prompt output but defensive), only the first is
      replaced.

    Args:
        markdown: Full detailed_summary text.
        section_heading: Exact heading text *without* the ``## `` prefix
            (e.g. ``"全体像"``).
        new_body: User-edited body text. Leading/trailing whitespace is
            trimmed.

    Returns:
        New markdown string with the section body replaced.

    Raises:
        ValueError: If the heading is not found.
    """
    if not markdown:
        raise ValueError(f"Section not found: {section_heading}")

    target = section_heading.strip()
    lines = markdown.splitlines()

    start_idx: int | None = None
    for i, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match and match.group("title").strip() == target:
            start_idx = i
            break

    if start_idx is None:
        raise ValueError(f"Section not found: {section_heading}")

    # Scan forward for the next ``##`` (body ends there).
    end_idx = len(lines)
    for j in range(start_idx + 1, len(lines)):
        if _HEADING_RE.match(lines[j]):
            end_idx = j
            break

    # Preserve the original heading line; replace lines between it and
    # the next heading with the new body. A single blank line separates
    # the heading from the body so the rendered markdown stays tidy.
    heading_line = lines[start_idx]
    body = new_body.strip("\n")
    body_lines = body.splitlines() if body else [""]
    # Build the replacement block. Leading blank line so readers see a
    # visual break between heading and body. Trailing blank line so the
    # next section's heading doesn't butt up against user content.
    replacement: list[str] = [heading_line, "", *body_lines]
    if end_idx < len(lines):
        replacement = [*replacement, ""]

    new_lines = [*lines[:start_idx], *replacement, *lines[end_idx:]]
    return "\n".join(new_lines)
