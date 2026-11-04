"""The small PDFs and the page-by-page expectations for SPEC-ADDON-060.

Each page is A4 (595x842). A line drawn at ``y=30`` sits in the top 50 points and
one at ``y=810`` or lower in the bottom 50; body lines start at ``y=72``. The
bands are the ones ``PAGE_MARGINS`` states, not this docstring.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from tests.document_fixtures import make_pdf_lines

PDF = "application/pdf"

TOP_Y = 30
BOTTOM_Y = 810
BODY_Y = 72

# A line whose box starts at 46.4: it overlaps the top band and PyMuPDF4LLM
# still keeps it, so the recovery has to recognise it as already present.
LIBRARY_KEPT = "KEPT BY LIBRARY"
LIBRARY_KEPT_Y = 55
LIBRARY_KEPT_PAGE = 8


class PageRules(NamedTuple):
    page: int
    body: str
    top: tuple[str, ...]
    bottom: tuple[str, ...]
    dropped: tuple[str, ...]

    @property
    def kept(self) -> tuple[str, ...]:
        return self.top + self.bottom


BANDS_RULES: tuple[PageRules, ...] = (
    PageRules(
        1, "Alpha body paragraph one.",
        ("Section One Title",), ("Footnote one source",), (),
    ),
    PageRules(
        2, "Bravo body.",
        ("RUNNING HEADER",), ("Source: survey 2023",), ("12",),
    ),
    PageRules(
        3, "Charlie body.", (), (),
        ("RUNNING HEADER", "\u00b7 219 \u00b7", "Source: survey 2024"),
    ),
    PageRules(
        4, "Delta body.", (), ("ii",),
        ("RUNNING HEADER", "Source: survey 2025"),
    ),
    PageRules(5, "Echo body.", ("SHORT HEADER",), (), ()),
    PageRules(6, "Foxtrot body.", ("SHORT HEADER",), (), ()),
    PageRules(
        7, "Golf body.",
        ("ABOVE EDGE", "LEFT EDGE", "RIGHT EDGE", "BELOW EDGE"),
        ("BOTTOM ABOVE", "BOTTOM LEFT", "BOTTOM RIGHT"),
        (),
    ),
    PageRules(8, "Hotel body.", (), (), ()),
    PageRules(9, "India body line one.", (), (), ()),
)

BAND_MARKS: tuple[str, ...] = tuple(
    dict.fromkeys(
        mark
        for row in BANDS_RULES
        for mark in (*row.top, *row.bottom, *row.dropped)
    )
)


def bands_pages() -> list[list[tuple[str, int, int]]]:
    return [
        [
            ("Alpha body paragraph one.", 72, BODY_Y),
            ("Alpha second body line.", 72, 88),
            ("Section One Title", 72, TOP_Y),
            ("Footnote one source", 72, BOTTOM_Y),
        ],
        [
            ("Bravo body.", 72, BODY_Y),
            ("RUNNING HEADER", 72, TOP_Y),
            ("12", 290, BOTTOM_Y),
            ("Source: survey 2023", 72, 820),
        ],
        [
            ("Charlie body.", 72, BODY_Y),
            ("RUNNING HEADER", 72, TOP_Y),
            ("\u00b7 219 \u00b7", 250, BOTTOM_Y),
            ("Source: survey 2024", 72, 820),
        ],
        [
            ("Delta body.", 72, BODY_Y),
            ("RUNNING HEADER", 72, TOP_Y),
            ("ii", 290, BOTTOM_Y),
            ("Source: survey 2025", 72, 820),
        ],
        [
            ("Echo body.", 72, BODY_Y),
            ("SHORT HEADER", 72, TOP_Y),
        ],
        [
            ("Foxtrot body.", 72, BODY_Y),
            ("SHORT HEADER", 72, TOP_Y),
        ],
        [
            ("Golf body.", 72, BODY_Y),
            ("ABOVE EDGE", 72, 20),
            ("LEFT EDGE", 72, 35),
            ("RIGHT EDGE", 350, 35),
            ("BELOW EDGE", 72, 45),
            ("BOTTOM ABOVE", 72, 805),
            ("BOTTOM LEFT", 72, 815),
            ("BOTTOM RIGHT", 350, 815),
        ],
        [
            ("Hotel body.", 72, BODY_Y),
            (LIBRARY_KEPT, 72, LIBRARY_KEPT_Y),
        ],
        [
            ("India body line one.", 72, BODY_Y),
            ("India body line two.", 72, 88),
        ],
    ]


def bands_pdf(path: Path) -> Path:
    return make_pdf_lines(path, bands_pages())


def body_only_pdf(path: Path, pages: int = 2) -> Path:
    return make_pdf_lines(path, [
        [(f"Body line of page {number}.", 72, BODY_Y)] for number in range(1, pages + 1)
    ])
