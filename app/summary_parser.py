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
# parent section's paragraph run. Keeping parsing H2-only preserves
# the ``plain_idx`` numbering scheme that backs existing
# ``section_path`` citations, so H3-granularity edits don't
# invalidate older rows.
_HEADING_RE = re.compile(r"^\s{0,3}##\s+(?P<title>.+?)\s*$")

# ``### 見出し`` — used only by the splice helper to locate an H3
# boundary inside an H2 body. Matches exactly three hashes so
# subsections of subsections (``####``) don't accidentally terminate
# the splice range.
_H3_HEADING_RE = re.compile(r"^\s{0,3}###\s+(?P<title>.+?)\s*$")

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

    ``cells`` is populated only for table rows — each tuple entry is
    one body-row cell in source order. Citation embedding pools one
    vector per cell so header/value asymmetry (e.g. "保存期間 | 3 日")
    doesn't let the header noun dominate the " | "-joined text
    embedding. For paragraphs, bullets, and single-cell rows the
    field is ``None`` and callers use ``segment_text`` directly.

    ``ancestor_headings`` captures the section context above the
    segment: ``(h2,)`` when no H3 is active, ``(h2, h3)`` when the
    parser is inside an H3 subsection. Citation linking uses this to
    anchor the segment's search to the transcript region where that
    heading is discussed — necessary when the content chunks
    themselves don't mention the topic name (e.g. the speaker says
    "保存方法は冷蔵庫で" without repeating "カレーの" because the
    dish was established a few turns earlier). Empty tuple when the
    segment was emitted outside any heading (the parser tolerates
    malformed documents).
    """

    section_path: str
    segment_type: str  # "bullet" | "paragraph"
    segment_text: str
    cells: tuple[str, ...] | None = None
    ancestor_headings: tuple[str, ...] = ()


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
    current_h3 = ""
    # Separate counter for plain segments (paragraph/bullet) vs table rows
    # so the two namespaces don't collide in section_path.
    plain_idx = 0
    row_idx = 0
    # Buffer for an in-progress paragraph. ``None`` means no paragraph
    # is currently open.
    paragraph_buf: list[str] | None = None

    def _current_ancestors() -> tuple[str, ...]:
        """H2 only when no H3 is active, else (H2, H3)."""
        if not current_section:
            return ()
        if current_h3:
            return (current_section, current_h3)
        return (current_section,)

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
                ancestor_headings=_current_ancestors(),
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
            # Reset the nested H3 context whenever we enter a new H2 so
            # subsequent segments don't inherit the previous H2's
            # trailing H3 as their heading context.
            current_h3 = ""
            plain_idx = 0
            row_idx = 0
            continue

        # H3 subheading: structural marker, not a claim. Terminate any
        # in-progress paragraph / table but emit no segment and leave
        # ``plain_idx`` untouched so downstream bullets keep the
        # H2-scoped counter the frontend parser also uses. Tracking the
        # title itself is new — citation linking uses it to anchor
        # searches to the transcript region where that subsection is
        # discussed.
        h3_match = _H3_HEADING_RE.match(line)
        if h3_match:
            flush_paragraph()
            in_table = False
            table_header_consumed = False
            current_h3 = h3_match.group("title").strip()
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
            row_text, cells = _flatten_table_row(line)
            if row_text:
                segments.append(
                    Segment(
                        section_path=f"{current_section}/row/{row_idx}",
                        segment_type="bullet",
                        segment_text=row_text,
                        cells=cells,
                        ancestor_headings=_current_ancestors(),
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
                        ancestor_headings=_current_ancestors(),
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


def _flatten_table_row(line: str) -> tuple[str, tuple[str, ...]]:
    """Return ``(joined_text, cells_tuple)`` for a ``| a | b | c |`` row.

    The leading/trailing ``|`` are stripped and each cell is trimmed;
    cells are joined with `` | `` so ``segment_text`` stays humanly
    readable. The cell tuple is returned alongside so downstream
    callers (notably citation embedding) can pool per-cell vectors
    instead of letting the header cell dominate the joined string.
    Empty cells are dropped from both the joined text and the tuple.
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
    return " | ".join(cells), tuple(cells)


# ---------------------------------------------------------------------------
# Section splice (Phase 2: edit)
# ---------------------------------------------------------------------------


def splice_section(
    markdown: str,
    h2_heading: str,
    h3_heading: str | None,
    new_content: str,
) -> str:
    """Replace a heading-anchored range with an arbitrary Markdown fragment.

    The splice range is inclusive of the heading line itself so the
    user can rename it. ``new_content`` is treated as opaque Markdown:
    it may add / drop / rearrange ``##`` or ``###`` lines and the
    document is always re-parsed downstream for citation alignment,
    so structural changes propagate naturally.

    Boundary rules:

    * ``h3_heading is None`` (H2-grained edit): splice from the matched
      ``##`` line to the line before the next ``##`` (or EOF).
    * ``h3_heading`` set (H3-grained edit): locate the ``##`` first;
      within its body locate the matched ``###`` line; splice from
      that ``###`` to the line before the next ``###`` *or* ``##``
      (whichever comes first), or EOF.

    The only validation is anchor-existence — if the heading can't be
    located, raise ``ValueError`` so the caller surfaces a 409 Conflict
    (optimistic lock: "the document changed under you, reload"). The
    fragment itself is never validated: users may accidentally delete
    the heading line, write junk, or splice in an entirely new section
    hierarchy. All of those are accepted and reflected on reload.

    Args:
        markdown: Full detailed_summary text.
        h2_heading: Exact H2 heading text without the ``## `` prefix
            (e.g. ``"全体像"``).
        h3_heading: Exact H3 heading text without the ``### `` prefix,
            or ``None`` for H2-grained edits.
        new_content: User-edited fragment. May include its own heading
            line(s).

    Returns:
        New markdown string.

    Raises:
        ValueError: If the H2 anchor (or, when given, the H3 anchor
            inside it) is not found.
    """
    if not markdown:
        raise ValueError(f"Section not found: {h2_heading}")

    lines = markdown.splitlines()
    h2_start = _find_h2_start(lines, h2_heading)
    if h2_start is None:
        raise ValueError(f"Section not found: {h2_heading}")
    h2_end = _find_h2_end(lines, h2_start)

    if h3_heading is None:
        start, end = h2_start, h2_end
    else:
        h3_start = _find_h3_start(lines, h2_start + 1, h2_end, h3_heading)
        if h3_start is None:
            raise ValueError(
                f"Subsection not found: {h2_heading}/{h3_heading}"
            )
        end = _find_h3_end(lines, h3_start + 1, h2_end)
        start = h3_start

    body = new_content.strip("\n")
    body_lines = body.splitlines() if body else []

    # Preserve a blank-line separator before the following lines when
    # the fragment doesn't already end with one — keeps adjacent ``##``
    # / ``###`` headings from butting up against user content after the
    # splice.
    if end < len(lines) and body_lines and body_lines[-1].strip() != "":
        body_lines = [*body_lines, ""]

    return "\n".join([*lines[:start], *body_lines, *lines[end:]])


def _find_h2_start(lines: list[str], target: str) -> int | None:
    """Return the index of the first ``## target`` line, or ``None``."""
    target = target.strip()
    for i, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match and match.group("title").strip() == target:
            return i
    return None


def _find_h2_end(lines: list[str], h2_start: int) -> int:
    """Return the index of the next ``##`` line after ``h2_start``, or len."""
    for j in range(h2_start + 1, len(lines)):
        if _HEADING_RE.match(lines[j]):
            return j
    return len(lines)


def _find_h3_start(
    lines: list[str], start: int, end: int, target: str
) -> int | None:
    """Return the index of the first ``### target`` line within ``[start, end)``."""
    target = target.strip()
    for i in range(start, end):
        match = _H3_HEADING_RE.match(lines[i])
        if match and match.group("title").strip() == target:
            return i
    return None


def _find_h3_end(
    lines: list[str], search_from: int, h2_end: int
) -> int:
    """Return the index of the next ``###`` (or ``h2_end``) boundary."""
    for j in range(search_from, h2_end):
        if _H3_HEADING_RE.match(lines[j]):
            return j
    return h2_end
