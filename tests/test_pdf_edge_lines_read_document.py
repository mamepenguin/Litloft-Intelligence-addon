"""SPEC-ADDON-060 I2, I3, I4, I5 and I8 through ``GET /files/{id}/document-text``."""

from __future__ import annotations

import logging

import fitz
import pymupdf
import pymupdf4llm
import pytest

from app.extractors.pdf import (
    MAX_PAGES,
    PYMUPDF_LOCK,
    PdfExtractor,
)
from tests import pdf_footnote_api as api_words
from tests.document_fixtures import make_pdf_lines
from tests.document_text_api import api, reader, search_index, seed  # noqa: F401
from tests.pdf_footnote_fixtures import (
    BAND_MARKS,
    BANDS_RULES,
    PDF,
    bands_pdf,
)

PAGE_IDS = [f"page-{row.page}" for row in BANDS_RULES]


def _single_page_library(path, page: int) -> str:
    pages = pymupdf4llm.to_markdown(
        str(path), pages=[page - 1], page_chunks=True, show_progress=False
    )
    return pages[0]["text"]


def _fail_edge_read_for_page(monkeypatch, number: int) -> None:
    real = pymupdf.Page.get_text

    def spy(self, *args, **kwargs):
        if (
            args[:1] == ("dict",)
            and "flags" not in kwargs
            and self.number == number - 1
        ):
            raise RuntimeError("simulated edge read failure")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_text", spy)


def _warned(caplog, name: str) -> bool:
    return any(
        record.levelno == logging.WARNING and name in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize("row", BANDS_RULES, ids=PAGE_IDS)
def test_spec_addon_060_i4_read_document_recovers_the_lines_indexing_recovers(
    reader, tmp_path, row,
):
    path = bands_pdf(tmp_path / "bands.pdf")
    reader.register("f1", path, PDF)
    indexed = PdfExtractor().extract(str(path))

    unit = reader.ok("f1", start=row.page, count=1)["units"][0]

    indexed_text = "".join(
        chunk.text for chunk in indexed.chunks if chunk.page == row.page
    )
    assert [mark for mark in row.kept if mark in unit["text"]] == list(row.kept)
    assert [mark for mark in row.kept if mark in indexed_text] == list(row.kept)
    for mark in row.dropped:
        assert mark not in unit["text"], mark
        assert mark not in indexed_text, mark


def test_spec_addon_060_i4_the_recovered_lines_agree_before_containment(reader, tmp_path):
    path = bands_pdf(tmp_path / "bands.pdf")
    reader.register("f1", path, PDF)
    indexed = PdfExtractor().extract(str(path))
    library = pymupdf4llm.to_markdown(
        str(path), page_chunks=True, show_progress=False
    )

    for row in BANDS_RULES:
        unit = reader.ok("f1", start=row.page, count=1)["units"][0]["text"]
        indexed_text = "".join(
            chunk.text for chunk in indexed.chunks if chunk.page == row.page
        )
        from_indexing = {
            mark
            for mark in BAND_MARKS
            if mark in indexed_text and mark not in library[row.page - 1]["text"]
        }
        from_reading = {
            mark
            for mark in BAND_MARKS
            if mark in unit and mark not in _single_page_library(path, row.page)
        }
        assert from_reading == from_indexing == set(row.kept), row.page


def test_spec_addon_060_i4_a_page_past_the_indexing_cap_gets_no_recovered_lines(
    reader, tmp_path,
):
    pages = [
        [(f"Body line of page {number}.", 72, 72)]
        for number in range(1, MAX_PAGES + 3)
    ]
    pages[MAX_PAGES - 1].append(("EDGE AT THE CAP", 72, 30))
    pages[MAX_PAGES].append(("EDGE PAST THE CAP", 72, 30))
    path = make_pdf_lines(tmp_path / "long.pdf", pages)
    reader.register("f1", path, PDF)

    body = reader.ok("f1", start=MAX_PAGES, count=3, max_chars=200_000)

    units = {unit["number"]: unit for unit in body["units"]}
    assert body["unit_count"] == MAX_PAGES + 2
    assert "EDGE AT THE CAP" in units[MAX_PAGES]["text"]
    assert "EDGE PAST THE CAP" not in units[MAX_PAGES + 1]["text"]
    assert units[MAX_PAGES + 1]["text"].strip() == _single_page_library(
        path, MAX_PAGES + 1
    ).strip()


def test_spec_addon_060_i4_a_page_that_fell_back_gets_no_recovered_lines(
    reader, tmp_path, monkeypatch,
):
    path = bands_pdf(tmp_path / "bands.pdf")
    reader.register("f1", path, PDF)
    real = pymupdf4llm.to_markdown

    def spy(doc, *args, **kwargs):
        if kwargs.get("pages") == [1]:
            raise RuntimeError("simulated conversion failure")
        return real(doc, *args, **kwargs)

    monkeypatch.setattr(pymupdf4llm, "to_markdown", spy)

    unit = reader.ok("f1", start=2, count=1)["units"][0]

    with fitz.open(str(path)) as doc:
        plain = doc[1].get_text("text").strip()
    assert unit["extractor"] == "fitz_fallback"
    assert unit["text"] == plain


def test_spec_addon_060_i5_a_page_whose_edge_read_raises_keeps_the_library_text(
    reader, tmp_path, monkeypatch, caplog,
):
    path = bands_pdf(tmp_path / "bands.pdf")
    reader.register("f1", path, PDF)
    _fail_edge_read_for_page(monkeypatch, 2)

    with caplog.at_level(logging.WARNING):
        body = reader.ok("f1", count=9, max_chars=200_000)

    units = {unit["number"]: unit for unit in body["units"]}
    assert "Section One Title" in units[1]["text"]
    assert "RUNNING HEADER" not in units[2]["text"]
    assert units[2]["text"].strip() == _single_page_library(path, 2).strip()
    assert _warned(caplog, "bands.pdf")


def test_spec_addon_060_i5_an_edge_read_that_fails_for_the_document_serves_the_library_text(
    reader, tmp_path, monkeypatch, caplog,
):
    path = bands_pdf(tmp_path / "bands.pdf")
    reader.register("f1", path, PDF)
    monkeypatch.setattr(
        pymupdf4llm, "to_markdown",
        lambda doc, *args, **kwargs: [{"text": "Stub body of page one.\n"}],
    )

    def broken(*_args, **_kwargs):
        raise RuntimeError("simulated document walk failure")

    for name in ("__getitem__", "__iter__", "__len__", "load_page"):
        monkeypatch.setattr(pymupdf.Document, name, broken, raising=False)

    with caplog.at_level(logging.WARNING):
        body = reader.ok("f1", count=3, max_chars=200_000)

    assert [unit["text"] for unit in body["units"]] == [
        "Stub body of page one.",
    ] * 3
    assert _warned(caplog, "bands.pdf")


def test_spec_addon_060_i1_read_document_converts_with_the_recovery_band(
    reader, tmp_path, monkeypatch,
):
    path = bands_pdf(tmp_path / "bands.pdf")
    reader.register("f1", path, PDF)
    seen: list[object] = []
    real = pymupdf4llm.to_markdown

    def spy(doc, *args, **kwargs):
        seen.append(kwargs.get("margins"))
        return real(doc, *args, **kwargs)

    monkeypatch.setattr(pymupdf4llm, "to_markdown", spy)

    reader.ok("f1", count=2)

    assert api_words.page_margins() == (0, 50, 0, 50)
    assert seen and set(seen) == {api_words.page_margins()}


def test_spec_addon_060_i8_read_document_holds_the_lock_for_the_edge_reads(
    reader, tmp_path, monkeypatch,
):
    path = bands_pdf(tmp_path / "bands.pdf")
    reader.register("f1", path, PDF)
    locked: list[bool] = []
    real = pymupdf.Page.get_text

    def spy(self, *args, **kwargs):
        if args[:1] == ("dict",) and "flags" not in kwargs:
            locked.append(PYMUPDF_LOCK.locked())
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_text", spy)

    reader.ok("f1", count=2)

    assert locked, "no edge-line read reached the text layer"
    assert all(locked)
