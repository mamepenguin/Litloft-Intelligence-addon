"""SPEC-ADDON-060 I5: an edge read that fails leaves the library Markdown alone."""

from __future__ import annotations

import logging

import fitz
import pymupdf
import pymupdf4llm
import pytest

import app.config as config
from app.extractors.base import ContentExtractor
from app.extractors.pdf import (
    EXTRACTOR_FALLBACK,
    EXTRACTOR_PRIMARY,
    PAGE_SEPARATOR,
    PdfExtractor,
)
from tests.document_fixtures import make_pdf_lines


@pytest.fixture(autouse=True)
def allow_tmp_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "validate_file_path", lambda _path: True)


def _stub_conversion(monkeypatch, texts: list[str]) -> None:
    monkeypatch.setattr(
        pymupdf4llm,
        "to_markdown",
        lambda doc, *args, **kwargs: [{"text": text} for text in texts],
    )


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


class _UnwalkableDoc:
    def __init__(self, page_count: int):
        self.page_count = page_count
        self.needs_pass = False

    def _fail(self, *_args, **_kwargs):
        raise RuntimeError("simulated document walk failure")

    __getitem__ = _fail
    __iter__ = _fail
    __len__ = _fail
    load_page = _fail

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False


def _warns(caplog, *parts: str) -> bool:
    return any(
        record.levelno == logging.WARNING
        and all(part in record.getMessage() for part in parts)
        for record in caplog.records
    )


def test_spec_addon_060_i5_a_page_whose_edge_read_raises_keeps_the_library_markdown(
    tmp_path, monkeypatch, caplog,
):
    path = make_pdf_lines(tmp_path / "edges.pdf", [
        [("EDGE ONE", 72, 30), ("Body one.", 72, 72)],
        [("EDGE TWO", 72, 30), ("Body two.", 72, 72)],
        [("EDGE THREE", 72, 30), ("Body three.", 72, 72)],
    ])
    _stub_conversion(monkeypatch, ["Body one.\n", "Body two.\n", "Body three.\n"])
    _fail_edge_read_for_page(monkeypatch, 2)

    with caplog.at_level(logging.WARNING):
        result = PdfExtractor().extract(str(path))

    parts = result.markdown.split(PAGE_SEPARATOR)
    assert result.extractor == EXTRACTOR_PRIMARY
    assert parts == ["Body one.\n", "Body two.\n", "Body three.\n"]
    by_page = {chunk.page: chunk.text for chunk in result.chunks}
    assert "EDGE ONE" in by_page[1]
    assert by_page[2] == "Body two."
    assert "EDGE THREE" in by_page[3]
    assert _warns(caplog, "edges.pdf", "2")


def test_spec_addon_060_i5_an_edge_read_that_fails_for_the_document_keeps_the_library_markdown(
    tmp_path, monkeypatch, caplog,
):
    path = make_pdf_lines(tmp_path / "edges.pdf", [
        [("EDGE ONE", 72, 30), ("Body one.", 72, 72)],
        [("EDGE TWO", 72, 30), ("Body two.", 72, 72)],
    ])
    _stub_conversion(monkeypatch, ["Body one.\n", "Body two.\n"])
    monkeypatch.setattr(fitz, "open", lambda *_a, **_k: _UnwalkableDoc(2))
    monkeypatch.setattr(pymupdf, "open", lambda *_a, **_k: _UnwalkableDoc(2))

    with caplog.at_level(logging.WARNING):
        result = PdfExtractor().extract(str(path))

    assert result.extractor == EXTRACTOR_PRIMARY
    assert result.markdown.split(PAGE_SEPARATOR) == ["Body one.\n", "Body two.\n"]
    assert "EDGE ONE" not in result.markdown
    assert _warns(caplog, "edges.pdf")


def test_spec_addon_060_i5_the_fitz_fallback_output_is_unchanged(tmp_path, monkeypatch):
    path = make_pdf_lines(tmp_path / "edges.pdf", [
        [("EDGE ONE", 72, 30), ("Body one.", 72, 72), ("EDGE FOOT", 72, 810)],
        [("EDGE TWO", 72, 30), ("Body two.", 72, 72)],
    ])

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated parse failure")

    monkeypatch.setattr(pymupdf4llm, "to_markdown", boom)

    result = PdfExtractor().extract(str(path))

    chunking = config.settings.indexing.text_chunking
    expected: list[tuple[int, str]] = []
    with fitz.open(str(path)) as doc:
        for index in range(doc.page_count):
            plain = (doc[index].get_text("text") or "").strip()
            expected.extend(
                (index + 1, chunk)
                for chunk in ContentExtractor.chunk_text(
                    plain,
                    max_size=chunking.max_chunk_size,
                    overlap=chunking.overlap,
                )
            )
    assert result.extractor == EXTRACTOR_FALLBACK
    assert result.markdown is None
    assert [(chunk.page, chunk.text) for chunk in result.chunks] == expected
