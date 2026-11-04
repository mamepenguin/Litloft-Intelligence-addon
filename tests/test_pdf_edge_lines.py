"""SPEC-ADDON-060 I1, I2, I3, I8 and I10 through ``PdfExtractor.extract``."""

from __future__ import annotations

import re

import pymupdf
import pymupdf4llm
import pytest

import app.config as config
from app.extractors.base import ContentExtractor
from app.extractors.pdf import (
    EXTRACTOR_PRIMARY,
    PAGE_SEPARATOR,
    PYMUPDF_LOCK,
    PdfExtractor,
)
from tests import pdf_footnote_api as api
from tests.pdf_footnote_fixtures import (
    BANDS_RULES,
    LIBRARY_KEPT,
    LIBRARY_KEPT_PAGE,
    bands_pdf,
    body_only_pdf,
)

PAGE_IDS = [f"page-{row.page}" for row in BANDS_RULES]


@pytest.fixture(autouse=True)
def allow_tmp_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "validate_file_path", lambda _path: True)


def _library_pages(path) -> list[dict]:
    return pymupdf4llm.to_markdown(
        str(path), page_chunks=True, show_progress=False
    )


def _page_parts(result) -> list[str]:
    return result.markdown.split(PAGE_SEPARATOR)


def _chunk_texts(result, page: int) -> str:
    return "".join(chunk.text for chunk in result.chunks if chunk.page == page)


@pytest.mark.parametrize("page_count", [1, 3], ids=["one-page", "three-pages"])
def test_spec_addon_060_i1_a_body_only_page_reads_and_chunks_as_the_library_page(
    tmp_path, page_count,
):
    path = body_only_pdf(tmp_path / "body.pdf", page_count)

    result = PdfExtractor().extract(str(path))

    library = _library_pages(path)
    chunking = config.settings.indexing.text_chunking
    expected = [
        (
            number,
            chunk,
        )
        for number, page in enumerate(library, start=1)
        for chunk in ContentExtractor.chunk_text(
            page["text"].strip(),
            max_size=chunking.max_chunk_size,
            overlap=chunking.overlap,
        )
    ]
    assert result.extractor == EXTRACTOR_PRIMARY
    assert _page_parts(result) == [page["text"] for page in library]
    assert [(chunk.page, chunk.text) for chunk in result.chunks] == expected


def test_spec_addon_060_i1_a_page_with_no_kept_line_is_byte_identical_to_the_library_page(
    tmp_path,
):
    path = bands_pdf(tmp_path / "bands.pdf")

    result = PdfExtractor().extract(str(path))

    library = _library_pages(path)
    parts = _page_parts(result)
    for row in (row for row in BANDS_RULES if not row.kept):
        assert parts[row.page - 1] == library[row.page - 1]["text"], row.page


def test_spec_addon_060_i1_extraction_converts_with_the_band_the_recovery_reads(
    tmp_path, monkeypatch,
):
    path = bands_pdf(tmp_path / "bands.pdf")
    seen: list[object] = []
    real = pymupdf4llm.to_markdown

    def spy(doc, *args, **kwargs):
        seen.append(kwargs.get("margins"))
        return real(doc, *args, **kwargs)

    monkeypatch.setattr(pymupdf4llm, "to_markdown", spy)

    PdfExtractor().extract(str(path))

    assert api.page_margins() == (0, 50, 0, 50)
    assert seen and set(seen) == {api.page_margins()}


@pytest.mark.parametrize("row", BANDS_RULES, ids=PAGE_IDS)
def test_spec_addon_060_i2_a_band_line_lands_on_its_own_page_in_reading_order(
    tmp_path, row,
):
    path = bands_pdf(tmp_path / "bands.pdf")

    result = PdfExtractor().extract(str(path))

    part = _page_parts(result)[row.page - 1]
    first = 0
    for mark in row.kept:
        at = part.find(mark, first)
        assert at >= 0, f"{mark!r} missing from page {row.page}"
        first = at + len(mark)
    if row.top:
        assert part.find(row.body) > max(part.find(mark) for mark in row.top)
        assert part.startswith("\n".join(row.top) + "\n\n")
    if row.bottom:
        assert part.find(row.body) < min(part.find(mark) for mark in row.bottom)
        assert re.search(r"\n\n+" + re.escape("\n".join(row.bottom)), part)
        assert part.rstrip().endswith(row.bottom[-1])
    for mark in row.kept:
        chunks = [
            chunk for chunk in result.chunks
            if mark in chunk.text and chunk.page == row.page
        ]
        assert chunks, mark
        assert all(chunk.metadata == f"page {row.page}" for chunk in chunks)


@pytest.mark.parametrize(
    "page,mark",
    [
        pytest.param(2, "12", id="bare-page-number"),
        pytest.param(3, "\u00b7 219 \u00b7", id="dash-wrapped-page-number"),
        pytest.param(3, "RUNNING HEADER", id="repeated-header-page-3"),
        pytest.param(4, "RUNNING HEADER", id="repeated-header-page-4"),
        pytest.param(3, "Source: survey 2024", id="repeated-key-different-digits-page-3"),
        pytest.param(4, "Source: survey 2025", id="repeated-key-different-digits-page-4"),
    ],
)
def test_spec_addon_060_i3_a_dropped_band_line_reaches_no_chunk_or_markdown(
    tmp_path, page, mark,
):
    path = bands_pdf(tmp_path / "bands.pdf")

    result = PdfExtractor().extract(str(path))

    assert mark not in _page_parts(result)[page - 1]
    assert mark not in _chunk_texts(result, page)


def test_spec_addon_060_i3_a_key_on_two_pages_keeps_the_line_on_both(tmp_path):
    path = bands_pdf(tmp_path / "bands.pdf")

    result = PdfExtractor().extract(str(path))

    for page in (5, 6):
        assert "SHORT HEADER" in _page_parts(result)[page - 1]
        assert "SHORT HEADER" in _chunk_texts(result, page)
    assert "RUNNING HEADER" in _page_parts(result)[1]


def test_spec_addon_060_i3_a_repeated_roman_numeral_is_not_a_page_number_rule(
    tmp_path,
):
    path = bands_pdf(tmp_path / "bands.pdf")

    result = PdfExtractor().extract(str(path))

    assert "ii" in _page_parts(result)[3]
    assert "ii" in _chunk_texts(result, 4)


def test_spec_addon_060_i10_a_band_line_the_library_kept_appears_once(tmp_path):
    path = bands_pdf(tmp_path / "bands.pdf")

    result = PdfExtractor().extract(str(path))

    part = _page_parts(result)[LIBRARY_KEPT_PAGE - 1]
    assert part == _library_pages(path)[LIBRARY_KEPT_PAGE - 1]["text"]
    assert part.count(LIBRARY_KEPT) == 1
    assert result.markdown.count(LIBRARY_KEPT) == 1
    assert sum(chunk.text.count(LIBRARY_KEPT) for chunk in result.chunks) == 1


def test_spec_addon_060_i8_extraction_holds_the_lock_for_the_edge_reads(
    tmp_path, monkeypatch,
):
    path = bands_pdf(tmp_path / "bands.pdf")
    locked: list[bool] = []
    real = pymupdf.Page.get_text

    def spy(self, *args, **kwargs):
        if args[:1] == ("dict",) and "flags" not in kwargs:
            locked.append(PYMUPDF_LOCK.locked())
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_text", spy)

    result = PdfExtractor().extract(str(path))

    assert result.extractor == EXTRACTOR_PRIMARY
    assert locked, "no edge-line read reached the text layer"
    assert all(locked)
