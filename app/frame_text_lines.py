"""Frame-text segments, their lines, and the lines a later segment repeats.

Frame OCR opens a new segment whenever anything on screen changes, and every
segment stores everything on screen. A line that stays up is therefore in
every segment of its run. The three readers that show frame text ask this
module which of a segment's lines an earlier segment put on screen within
:data:`OCR_REPEAT_WINDOW_SECONDS` of it.

The segment's text is the merge the find route already builds, so a line cut
by a chunk boundary is whole here and can be recognised in every reader.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

#: How long a line may be off screen and still count as the same line.
OCR_REPEAT_WINDOW_SECONDS = 60.0

_OVERLAP_MIN = 20
#: The chunk overlap picture OCR stores, which ``file_find`` merges with.
_MERGE_OVERLAP = 200


def _overlap(text: str, chunk: str, limit: int) -> int:
    limit = min(limit, len(text), len(chunk))
    for size in range(limit, _OVERLAP_MIN - 1, -1):
        if text.endswith(chunk[:size]):
            return size
    return 0


def _merge_chunks(chunks: list[str], limit: int) -> str:
    text = ""
    for chunk in chunks:
        if not text:
            text = chunk
            continue
        overlap = _overlap(text, chunk, limit)
        if overlap:
            text += chunk[overlap:]
        else:
            text += "\n" + chunk
    return text


def _lines(text: str) -> frozenset[str]:
    return frozenset(
        line for line in (part.strip() for part in text.split("\n")) if line
    )


@dataclass(frozen=True)
class FrameSegment:
    """One frame-OCR read: its chunks, merged text, lines and repeated lines."""

    start: float
    end: float
    #: Every ``(chunk_index, stored text)`` of the segment, in index order.
    chunks: tuple[tuple[int, str], ...]
    text: str
    #: The merged text's stripped ``\n``-parts, empty ones dropped.
    lines: frozenset[str]
    #: The lines an earlier segment put on screen within the window.
    repeated: frozenset[str]


def frame_segments(rows: Iterable[Any]) -> list[FrameSegment]:
    """Group ``video_frame_texts`` rows into segments and mark repeated lines.

    A row supplies ``chunk_index``, ``start_time``, ``end_time`` and ``text``.
    Segments are ordered by ``start_time``. Equal starts are not earlier than
    each other, so a line repeated between them is new on both sides.
    """
    grouped: dict[tuple[float, float], list[tuple[int, str]]] = {}
    for row in sorted(rows, key=lambda row: int(row.chunk_index)):
        key = (float(row.start_time), float(row.end_time))
        grouped.setdefault(key, []).append((int(row.chunk_index), row.text or ""))
    segments: list[FrameSegment] = []
    for (start, end), chunks in grouped.items():
        text = _merge_chunks([chunk for _index, chunk in chunks], _MERGE_OVERLAP)
        segments.append(FrameSegment(
            start=start,
            end=end,
            chunks=tuple(chunks),
            text=text,
            lines=_lines(text),
            repeated=frozenset(),
        ))

    ordered = sorted(segments, key=lambda segment: segment.start)
    latest_end: dict[str, float] = {}
    out: list[FrameSegment] = []
    index = 0
    while index < len(ordered):
        start = ordered[index].start
        stop = index
        while stop < len(ordered) and ordered[stop].start == start:
            stop += 1
        group = ordered[index:stop]
        for segment in group:
            repeated = frozenset(
                line for line in segment.lines
                if line in latest_end
                and segment.start - latest_end[line] <= OCR_REPEAT_WINDOW_SECONDS
            )
            out.append(replace(segment, repeated=repeated))
        for segment in group:
            for line in segment.lines:
                latest_end[line] = max(latest_end.get(line, segment.end), segment.end)
        index = stop
    return out


def line_at(text: str, at: int) -> str | None:
    """The stripped line whose own characters contain ``at``.

    A newline belongs to no line, so an occurrence that starts on one is not
    in a repeated line.
    """
    start = 0
    for part in text.split("\n"):
        end = start + len(part)
        if start <= at < end:
            return part.strip()
        start = end + 1
    return None
