"""Unit tests for ``PdfExtractor`` (Phase 2 of the PyMuPDF4LLM rollout).

The PDF extractor has two paths:

- **Primary**: PyMuPDF4LLM page-chunked Markdown → embedding chunks +
  full Markdown.
- **Fallback**: PyMuPDF (fitz) raw text per page when PyMuPDF4LLM is
  unavailable or raises. Markdown is ``None`` on this path.

These tests stub both libraries so the suite stays self-contained and
does not require real PDF fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Heavy ML deps are stubbed by ``conftest.py``; we only need to ensure
# the extractor module's imports resolve.

import app.config as config  # noqa: E402
from app.extractors.base import ExtractionResult  # noqa: E402
from app.extractors.pdf import (  # noqa: E402
    EXTRACTOR_FALLBACK,
    EXTRACTOR_PRIMARY,
    MAX_PDF_SIZE,
    PAGE_SEPARATOR,
    PdfExtractor,
)


@pytest.fixture()
def fake_pdf(tmp_path: Path) -> Path:
    """Create a non-empty file with a ``.pdf`` extension for size checks."""
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4\n%fake\n")
    return path


@pytest.fixture(autouse=True)
def allow_tmp_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Bypass the production ``allowed_base_dirs`` check for fixture files."""
    monkeypatch.setattr(
        config, "validate_file_path", lambda _path: True
    )


def _install_pymupdf4llm(
    monkeypatch: pytest.MonkeyPatch,
    to_markdown: object,
) -> MagicMock:
    """Register a fake ``pymupdf4llm`` module with the given callable."""
    fake = MagicMock()
    fake.to_markdown = to_markdown
    monkeypatch.setitem(sys.modules, "pymupdf4llm", fake)
    return fake


def _install_pymupdf4llm_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``import pymupdf4llm`` to raise ``ImportError``."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pymupdf4llm":
            raise ImportError("simulated missing dependency")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def _install_fitz(
    monkeypatch: pytest.MonkeyPatch,
    pages: list[str],
) -> MagicMock:
    """Register a fake ``fitz`` module that yields the given page texts."""
    fake_pages = []
    for text in pages:
        page = MagicMock()
        page.get_text.return_value = text
        fake_pages = [*fake_pages, page]

    doc = MagicMock()
    doc.page_count = len(fake_pages)
    doc.__getitem__.side_effect = lambda i: fake_pages[i]
    doc.close = MagicMock()

    fake = MagicMock()
    fake.open.return_value = doc
    monkeypatch.setitem(sys.modules, "fitz", fake)
    return fake


def test_pymupdf4llm_success_returns_markdown_and_chunks(
    monkeypatch: pytest.MonkeyPatch,
    fake_pdf: Path,
) -> None:
    page_chunks = [
        {"text": "# Title\n\nFirst page body."},
        {"text": "## Section\n\nSecond page body."},
    ]
    _install_pymupdf4llm(monkeypatch, MagicMock(return_value=page_chunks))

    result = PdfExtractor().extract(str(fake_pdf))

    assert isinstance(result, ExtractionResult)
    assert result.extractor == EXTRACTOR_PRIMARY
    assert result.markdown is not None
    assert result.markdown == (
        "# Title\n\nFirst page body."
        + PAGE_SEPARATOR
        + "## Section\n\nSecond page body."
    )
    assert len(result.chunks) >= 2
    pages = [c.page for c in result.chunks]
    assert 1 in pages and 2 in pages


def test_pymupdf4llm_chunks_carry_page_numbers(
    monkeypatch: pytest.MonkeyPatch,
    fake_pdf: Path,
) -> None:
    page_chunks = [
        {"text": "Alpha content on page one."},
        {"text": "Bravo content on page two."},
        {"text": "Charlie content on page three."},
    ]
    _install_pymupdf4llm(monkeypatch, MagicMock(return_value=page_chunks))

    result = PdfExtractor().extract(str(fake_pdf))

    by_page: dict[int, list[str]] = {}
    for chunk in result.chunks:
        assert chunk.page is not None
        assert chunk.metadata == f"page {chunk.page}"
        by_page.setdefault(chunk.page, []).append(chunk.text)
    assert set(by_page.keys()) == {1, 2, 3}
    assert any("Alpha" in t for t in by_page[1])
    assert any("Bravo" in t for t in by_page[2])
    assert any("Charlie" in t for t in by_page[3])


def test_pymupdf4llm_import_error_falls_back_to_fitz(
    monkeypatch: pytest.MonkeyPatch,
    fake_pdf: Path,
) -> None:
    _install_pymupdf4llm_missing(monkeypatch)
    _install_fitz(monkeypatch, ["fitz extracted page one text"])

    result = PdfExtractor().extract(str(fake_pdf))

    assert result.extractor == EXTRACTOR_FALLBACK
    assert result.markdown is None
    assert len(result.chunks) >= 1
    assert all(c.page == 1 for c in result.chunks)
    assert any("fitz extracted" in c.text for c in result.chunks)


def test_pymupdf4llm_index_error_retries_without_pages_arg(
    monkeypatch: pytest.MonkeyPatch,
    fake_pdf: Path,
) -> None:
    """PDFs shorter than MAX_PAGES make to_markdown raise IndexError;
    the extractor must retry without ``pages=`` and stay on the
    primary path (no fitz fallback).
    """
    fake_chunks = [{"text": "Short PDF body."}]

    call_log: list[dict] = []

    def _to_markdown(_path, **kwargs):
        call_log.append(kwargs)
        if "pages" in kwargs:
            raise IndexError("page 116 not in document")
        return fake_chunks

    _install_pymupdf4llm(monkeypatch, _to_markdown)

    result = PdfExtractor().extract(str(fake_pdf))

    assert result.extractor == EXTRACTOR_PRIMARY
    assert result.markdown is not None
    assert "Short PDF body." in result.markdown
    assert len(call_log) == 2
    assert "pages" in call_log[0]
    assert "pages" not in call_log[1]


def test_pymupdf4llm_runtime_error_falls_back_to_fitz(
    monkeypatch: pytest.MonkeyPatch,
    fake_pdf: Path,
) -> None:
    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated parse failure")

    _install_pymupdf4llm(monkeypatch, _boom)
    _install_fitz(monkeypatch, ["fallback page text"])

    result = PdfExtractor().extract(str(fake_pdf))

    assert result.extractor == EXTRACTOR_FALLBACK
    assert result.markdown is None
    assert any("fallback" in c.text for c in result.chunks)


def test_oversize_pdf_returns_empty_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    big = tmp_path / "big.pdf"
    # Sparse file — fast to create, the extractor only uses ``stat``.
    with big.open("wb") as fh:
        fh.seek(MAX_PDF_SIZE + 1)
        fh.write(b"\0")

    sentinel = MagicMock(side_effect=AssertionError("must not be called"))
    _install_pymupdf4llm(monkeypatch, sentinel)

    result = PdfExtractor().extract(str(big))

    assert result == ExtractionResult(
        chunks=[], markdown=None, extractor=EXTRACTOR_FALLBACK
    )
    sentinel.assert_not_called()


def test_missing_file_returns_empty_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ghost = tmp_path / "does_not_exist.pdf"
    sentinel = MagicMock(side_effect=AssertionError("must not be called"))
    _install_pymupdf4llm(monkeypatch, sentinel)

    result = PdfExtractor().extract(str(ghost))

    assert result.chunks == []
    assert result.markdown is None
    assert result.extractor == EXTRACTOR_FALLBACK
    sentinel.assert_not_called()


def test_can_handle_recognises_pdf_extension(tmp_path: Path) -> None:
    extractor = PdfExtractor()
    assert extractor.can_handle(str(tmp_path / "x.pdf")) is True
    assert extractor.can_handle(str(tmp_path / "x.PDF")) is True
    assert extractor.can_handle(str(tmp_path / "x.txt")) is False


def test_pymupdf4llm_empty_page_list_returns_empty_markdown(
    monkeypatch: pytest.MonkeyPatch,
    fake_pdf: Path,
) -> None:
    _install_pymupdf4llm(monkeypatch, MagicMock(return_value=[]))

    result = PdfExtractor().extract(str(fake_pdf))

    assert result.extractor == EXTRACTOR_PRIMARY
    assert result.markdown == ""
    assert result.chunks == []


def test_pymupdf4llm_skips_blank_pages(
    monkeypatch: pytest.MonkeyPatch,
    fake_pdf: Path,
) -> None:
    page_chunks = [
        {"text": "   \n  "},
        {"text": "Real content here."},
        {"text": ""},
    ]
    _install_pymupdf4llm(monkeypatch, MagicMock(return_value=page_chunks))

    result = PdfExtractor().extract(str(fake_pdf))

    assert result.extractor == EXTRACTOR_PRIMARY
    pages = {c.page for c in result.chunks}
    assert pages == {2}
