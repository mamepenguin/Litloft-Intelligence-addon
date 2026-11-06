"""A scanned PDF page's stored OCR text is what ``document-text`` returns.

SPEC-ADDON-069 B3/B7, I5: the page test is "the text layer is empty after
stripping whitespace"; a page that passes it and has stored OCR text takes that
text in both of the reader's paths, and a page with any text layer keeps its
own text.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from tests.document_fixtures import PDF, make_pdf
from tests.document_text_api import api, reader, search_index, seed  # noqa: F401


def make_image_only_pdf(path: Path) -> Path:
    """One page holding a drawn picture and no text layer at all."""
    import fitz

    # Draw on one page and rasterise it, so the inserted picture carries no
    # text layer and the test does not depend on a stubbed Pillow.
    source = fitz.open()
    drawn = source.new_page(width=595, height=842)
    drawn.draw_rect(
        fitz.Rect(40, 60, 555, 240), color=(0, 0, 0), fill=(0.8, 0.8, 0.8),
        width=4,
    )
    drawn.insert_text((80, 150), "SCANNED IMAGE", fontsize=24)
    pixmap = drawn.get_pixmap(dpi=150)
    source.close()

    # A fresh page whose only content is that picture: PyMuPDF adds no text.
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(40, 60, 555, 240), pixmap=pixmap)
    document.save(str(path))
    document.close()
    return path


def make_whitespace_only_pdf(path: Path) -> Path:
    """One page with a text layer of spaces only, plus a drawn picture."""
    import fitz

    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((72, 72), "   ", fontsize=11)
    page.draw_rect(
        fitz.Rect(40, 160, 555, 340), color=(0, 0, 0),
        fill=(0.8, 0.8, 0.8), width=4,
    )
    document.save(str(path))
    document.close()
    return path


def _page_texts(body: dict) -> list[str]:
    return [unit["text"] for unit in body["units"]]


def _store_page_text(seed, file_id: str, page: int, text: str) -> None:
    seed.index.sql(
        "INSERT INTO pdf_page_ocr_texts(file_id, page, text) "
        "VALUES(:f, :p, :t)",
        f=file_id, p=page, t=text,
    )


def test_spec_addon_069_b7_document_text_returns_the_stored_ocr_for_an_empty_page(
    reader, tmp_path,
):
    """SPEC-ADDON-069 B7: the unit text is what indexing built for the page."""
    path = make_image_only_pdf(tmp_path / "scan.pdf")
    reader.register("f1", path, PDF)
    _store_page_text(reader.seed, "f1", 1, "STORED SCAN TEXT")

    body = reader.ok("f1")

    assert _page_texts(body) == ["STORED SCAN TEXT"]
    assert body["units"][0]["extractor"] == "pymupdf4llm"


def test_spec_addon_069_b7_the_fitz_fallback_returns_it_too(
    reader, tmp_path, monkeypatch,
):
    """SPEC-ADDON-069 B7: both paths of the reader use the stored text."""
    import pymupdf4llm

    path = make_image_only_pdf(tmp_path / "scan.pdf")
    reader.register("f1", path, PDF)
    _store_page_text(reader.seed, "f1", 1, "STORED SCAN TEXT")
    monkeypatch.setattr(
        pymupdf4llm,
        "to_markdown",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated conversion failure")
        ),
    )

    body = reader.ok("f1")

    assert body["units"][0]["extractor"] == "fitz_fallback"
    assert _page_texts(body) == ["STORED SCAN TEXT"]


def test_spec_addon_069_i5_a_page_with_a_text_layer_keeps_its_own_text(
    reader, tmp_path,
):
    """SPEC-ADDON-069 B3/I5: any text layer, even a page number, is not read."""
    path = make_pdf(tmp_path / "text.pdf", ["ORIGINAL PAGE TEXT"])
    reader.register("f1", path, PDF)
    _store_page_text(reader.seed, "f1", 1, "STORED SCAN TEXT")

    body = reader.ok("f1")

    assert "ORIGINAL PAGE TEXT" in " ".join(_page_texts(body))
    assert "STORED SCAN TEXT" not in " ".join(_page_texts(body))


def test_spec_addon_069_b3_a_whitespace_only_text_layer_is_scanned(
    reader, tmp_path,
):
    """SPEC-ADDON-069 B3: stripping whitespace decides, not raw length."""
    path = make_whitespace_only_pdf(tmp_path / "spaces.pdf")
    reader.register("f1", path, PDF)
    _store_page_text(reader.seed, "f1", 1, "STORED SCAN TEXT")

    body = reader.ok("f1")

    assert _page_texts(body) == ["STORED SCAN TEXT"]
