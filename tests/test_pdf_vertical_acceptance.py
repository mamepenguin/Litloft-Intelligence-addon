from __future__ import annotations

import fitz
import pymupdf4llm
import pytest

import app.config as config
from app.extractors.pdf import PAGE_SEPARATOR, PdfExtractor
from tests.document_text_api import api, reader, search_index, seed  # noqa: F401


@pytest.fixture(autouse=True)
def allow_tmp_paths(monkeypatch):
    monkeypatch.setattr(config, "validate_file_path", lambda _path: True)


@pytest.fixture
def vertical_pdf(tmp_path):
    path = tmp_path / "vertical.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((100, 100), "VERTICALWORD", rotate=270)
    page.insert_text((200, 100), "tiny")
    page = doc.new_page()
    page.insert_text((72, 100), "Horizontal paragraph with plenty of ordinary body text.")
    page.insert_text((72, 120), "Another horizontal paragraph with more searchable words.")
    page.insert_text((350, 100), "MIXEDLABEL", rotate=270)
    doc.save(path)
    doc.close()
    return path


def test_spec_addon_115_i1_i2_vertical_pages_reach_index_and_reader(
    vertical_pdf, reader,
):
    reader.register("vertical", vertical_pdf, "application/pdf")
    result = PdfExtractor().extract(str(vertical_pdf))
    parts = result.markdown.split(PAGE_SEPARATOR)
    units = reader.ok("vertical", start=1, count=2, max_chars=200_000)["units"]

    for number, word in [(1, "VERTICALWORD"), (2, "MIXEDLABEL")]:
        assert word in parts[number - 1]
        assert any(word in c.text and c.page == number for c in result.chunks)
        assert units[number - 1]["number"] == number
        assert word in units[number - 1]["text"]
    assert "tiny" in parts[0]
    assert "Horizontal paragraph" in parts[1]


def test_spec_addon_115_i2_dominant_page_uses_plain_page_text(
    vertical_pdf, monkeypatch, reader,
):
    reader.register("vertical", vertical_pdf, "application/pdf")
    real = pymupdf4llm.to_markdown

    def marked(*args, **kwargs):
        pages = real(*args, **kwargs)
        for page in pages:
            page["text"] = "MARKDOWNONLY\n" + page["text"]
        return pages

    monkeypatch.setattr(pymupdf4llm, "to_markdown", marked)
    result = PdfExtractor().extract(str(vertical_pdf))
    parts = result.markdown.split(PAGE_SEPARATOR)
    units = reader.ok("vertical", start=1, count=2, max_chars=200_000)["units"]

    assert "MARKDOWNONLY" not in parts[0]
    assert "MARKDOWNONLY" not in units[0]["text"]
    assert "VERTICALWORD" in parts[0] and "tiny" in parts[0]
    assert "MARKDOWNONLY" in parts[1]
    assert "MARKDOWNONLY" in units[1]["text"]
    assert parts[1].index("MARKDOWNONLY") < parts[1].index("MIXEDLABEL")


def test_spec_addon_115_i7_reader_recovers_a_requested_page_past_index_cap(
    tmp_path, reader,
):
    path = tmp_path / "long.pdf"
    doc = fitz.open()
    for _ in range(500):
        doc.new_page()
    page = doc.new_page()
    page.insert_text((100, 100), "LATEVERTICAL", rotate=270)
    doc.save(path)
    doc.close()
    reader.register("long", path, "application/pdf")

    unit = reader.ok("long", start=501, count=1)["units"][0]

    assert unit["number"] == 501
    assert "LATEVERTICAL" in unit["text"]


def _rows(index, statement):
    from sqlalchemy import text

    with index.engine.connect() as conn:
        return [tuple(row) for row in conn.execute(text(statement))]


def _embeddings(monkeypatch, index):
    import numpy as np
    from app.workers import metadata

    monkeypatch.setattr(
        metadata, "embed_passages",
        lambda passages: [np.zeros(index.dim("vec_text"), dtype=np.float32)
                          for _ in passages],
    )


def test_spec_addon_115_i5_parser_failure_keeps_index_and_sections_and_read_fails(
    vertical_pdf, reader, search_index, monkeypatch, caplog,
):
    from pdfminer.pdfparser import PDFParser
    from app.workers import metadata

    reader.register("vertical", vertical_pdf, "application/pdf")
    _embeddings(monkeypatch, search_index)
    assert metadata.index_text_content("vertical") is True
    reader.seed.section("vertical", 1, "Existing section")
    search_index.sql("UPDATE indexed_files SET text_indexed = 0 WHERE file_id = 'vertical'")
    queries = [
        "SELECT * FROM fts_text_content WHERE file_id = 'vertical'",
        "SELECT * FROM embeddings WHERE file_id = 'vertical'",
        "SELECT * FROM pdf_markdown WHERE file_id = 'vertical'",
        "SELECT * FROM document_sections WHERE file_id = 'vertical'",
    ]
    before = [_rows(search_index, query) for query in queries]
    assert all(before)

    def fail(*_args, **_kwargs):
        raise RuntimeError("simulated parser failure")

    monkeypatch.setattr(PDFParser, "__init__", fail)
    with caplog.at_level("WARNING"):
        assert metadata.index_text_content("vertical") is False
        response = reader.get("vertical", start=1, count=1)

    assert response.status_code == 422
    assert [_rows(search_index, query) for query in queries] == before
    assert _rows(search_index, "SELECT text_indexed FROM indexed_files WHERE file_id = 'vertical'") == [(0,)]
    assert any("vertical" in record.getMessage() and "1" in record.getMessage()
               for record in caplog.records)
    assert all("VERTICALWORD" not in record.getMessage() for record in caplog.records)


def test_spec_addon_115_i4_reindex_and_reader_keep_stored_ocr(
    tmp_path, reader, search_index, monkeypatch,
):
    from app.workers import metadata
    from tests.test_scanned_pdf_text import make_image_only_pdf

    path = make_image_only_pdf(tmp_path / "scan.pdf")
    reader.register("scan", path, "application/pdf")
    search_index.sql(
        "INSERT INTO pdf_page_ocr_texts(file_id, page, text) "
        "VALUES ('scan', 1, 'STORED OCR WORDS')"
    )
    before = _rows(search_index, "SELECT * FROM pdf_page_ocr_texts WHERE file_id = 'scan'")
    _embeddings(monkeypatch, search_index)

    assert metadata.index_text_content("scan") is True
    unit = reader.ok("scan", start=1, count=1)["units"][0]

    assert unit["text"] == "STORED OCR WORDS"
    assert _rows(search_index, "SELECT * FROM pdf_page_ocr_texts WHERE file_id = 'scan'") == before
    assert any("STORED OCR WORDS" in row[0] for row in _rows(
        search_index, "SELECT text FROM fts_text_content WHERE file_id = 'scan'"
    ))


def test_spec_addon_115_i5_empty_vertical_extraction_keeps_index_and_read_fails(
    vertical_pdf, reader, search_index, monkeypatch,
):
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.layout import LTPage
    from app.workers import metadata

    reader.register("vertical", vertical_pdf, "application/pdf")
    _embeddings(monkeypatch, search_index)
    assert metadata.index_text_content("vertical") is True
    reader.seed.section("vertical", 1, "Existing section")
    search_index.sql("UPDATE indexed_files SET text_indexed = 0 WHERE file_id = 'vertical'")
    queries = [
        "SELECT * FROM fts_text_content WHERE file_id = 'vertical'",
        "SELECT * FROM embeddings WHERE file_id = 'vertical'",
        "SELECT * FROM pdf_markdown WHERE file_id = 'vertical'",
        "SELECT * FROM document_sections WHERE file_id = 'vertical'",
    ]
    before = [_rows(search_index, query) for query in queries]
    assert all(before)
    monkeypatch.setattr(
        PDFPageAggregator, "get_result", lambda _self: LTPage(1, (0, 0, 595, 842))
    )

    assert metadata.index_text_content("vertical") is False
    response = reader.get("vertical", start=1, count=1)

    assert response.status_code == 422
    assert [_rows(search_index, query) for query in queries] == before
    assert _rows(
        search_index, "SELECT text_indexed FROM indexed_files WHERE file_id = 'vertical'"
    ) == [(0,)]


def test_spec_addon_115_i7_late_page_does_not_read_unneeded_earlier_edges(
    tmp_path, reader, monkeypatch,
):
    from app.extractors import pdf

    path = tmp_path / "late.pdf"
    doc = fitz.open()
    for _ in range(500):
        doc.new_page()
    doc.new_page().insert_text((100, 100), "LATEVERTICAL", rotate=270)
    doc.save(path)
    doc.close()
    reader.register("late", path, "application/pdf")
    calls = []
    read_edges = pdf.read_edge_lines

    def record_edges(*args):
        calls.append(True)
        return read_edges(*args)

    monkeypatch.setattr(pdf, "read_edge_lines", record_edges)
    unit = reader.ok("late", start=501, count=1)["units"][0]
    assert "LATEVERTICAL" in unit["text"]
    assert calls == []


def test_spec_addon_115_i5_reopen_failure_keeps_existing_index(
    vertical_pdf, reader, search_index, monkeypatch,
):
    from app.workers import metadata

    reader.register("vertical", vertical_pdf, "application/pdf")
    _embeddings(monkeypatch, search_index)
    assert metadata.index_text_content("vertical") is True
    search_index.sql("UPDATE indexed_files SET text_indexed = 0 WHERE file_id = 'vertical'")
    queries = [
        "SELECT * FROM fts_text_content WHERE file_id = 'vertical'",
        "SELECT * FROM embeddings WHERE file_id = 'vertical'",
        "SELECT * FROM pdf_markdown WHERE file_id = 'vertical'",
    ]
    before = [_rows(search_index, query) for query in queries]
    monkeypatch.setattr(
        pymupdf4llm, "to_markdown", lambda *_a, **_k: [{"text": "partial"}]
    )

    def fail_open(*_args, **_kwargs):
        raise OSError("file became unreadable")

    monkeypatch.setattr(fitz, "open", fail_open)
    assert metadata.index_text_content("vertical") is False
    assert [_rows(search_index, query) for query in queries] == before
    assert _rows(
        search_index, "SELECT text_indexed FROM indexed_files WHERE file_id = 'vertical'"
    ) == [(0,)]


def test_spec_addon_115_i5_orientation_failure_keeps_existing_index(
    vertical_pdf, reader, search_index, monkeypatch,
):
    from app.workers import metadata

    reader.register("vertical", vertical_pdf, "application/pdf")
    _embeddings(monkeypatch, search_index)
    assert metadata.index_text_content("vertical") is True
    search_index.sql("UPDATE indexed_files SET text_indexed = 0 WHERE file_id = 'vertical'")
    queries = [
        "SELECT * FROM fts_text_content WHERE file_id = 'vertical'",
        "SELECT * FROM embeddings WHERE file_id = 'vertical'",
        "SELECT * FROM pdf_markdown WHERE file_id = 'vertical'",
    ]
    before = [_rows(search_index, query) for query in queries]
    monkeypatch.setattr(
        pymupdf4llm, "to_markdown", lambda *_a, **_k: [{"text": "partial"}]
    )
    original = fitz.Page.get_text
    orientation_flags = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES

    def fail_orientation(page, *args, **kwargs):
        if args[:1] == ("dict",) and kwargs.get("flags") == orientation_flags:
            raise OSError("orientation unavailable")
        return original(page, *args, **kwargs)

    monkeypatch.setattr(fitz.Page, "get_text", fail_orientation)
    assert metadata.index_text_content("vertical") is False
    assert reader.get("vertical", start=1, count=1).status_code == 422
    assert [_rows(search_index, query) for query in queries] == before
    assert _rows(
        search_index, "SELECT text_indexed FROM indexed_files WHERE file_id = 'vertical'"
    ) == [(0,)]


def test_spec_addon_115_fallback_recovers_vertical_text_without_document_name(
    vertical_pdf, monkeypatch,
):
    real_open = fitz.open

    class PageWithoutJoinedText:
        def __init__(self, page):
            self.page = page

        def get_text(self, mode, **kwargs):
            text = self.page.get_text(mode, **kwargs)
            if mode == "text" and self.page.number == 0:
                return "\n".join("VERTICALWORD")
            return text

    class NamelessDocument:
        name = None

        def __init__(self, path):
            self.doc = real_open(path)
            self.page_count = self.doc.page_count

        def __getitem__(self, number):
            return PageWithoutJoinedText(self.doc[number])

        def close(self):
            self.doc.close()

    def fail_conversion(*_args, **_kwargs):
        raise RuntimeError("conversion unavailable")

    monkeypatch.setattr(pymupdf4llm, "to_markdown", fail_conversion)
    monkeypatch.setattr(fitz, "open", NamelessDocument)
    result = PdfExtractor().extract(str(vertical_pdf))

    assert any(
        chunk.page == 1 and "VERTICALWORD" in chunk.text
        for chunk in result.chunks
    )


def test_spec_addon_115_i10_recovers_nested_pdf_form_text(tmp_path, reader):
    source = fitz.open()
    source_page = source.new_page(width=400, height=500)
    source_page.insert_text((300, 70), "NESTEDVERTICALWORD", rotate=270)
    source_page.insert_text((40, 300), "NESTEDHORIZONTALWORD")
    path = tmp_path / "nested.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=400, height=500)
        page.show_pdf_page(page.rect, source, 0)
        page.insert_text((40, 40), "H")
        page = doc.new_page(width=400, height=500)
        page.show_pdf_page(page.rect, source, 0)
        page.insert_text((40, 40), "A horizontal line with more text than the vertical label.")
        page.insert_text((40, 60), "Another horizontal line with additional searchable words.")
        doc.save(path)
    source.close()
    reader.register("nested", path, "application/pdf")

    indexed = PdfExtractor().extract(str(path))
    units = reader.ok("nested", start=1, count=2)["units"]
    for number in (1, 2):
        assert any(
            chunk.page == number and "NESTEDVERTICALWORD" in chunk.text
            for chunk in indexed.chunks
        )
        assert "NESTEDVERTICALWORD" in units[number - 1]["text"]
        assert "NESTEDHORIZONTALWORD" in units[number - 1]["text"]
    assert "horizontal line" in units[1]["text"]


def test_spec_addon_115_i10_vertical_dominant_form_page(tmp_path, reader):
    source = fitz.open()
    source_page = source.new_page(width=400, height=500)
    source_page.insert_text((300, 70), "NESTEDVERTICALWORD", rotate=270)
    source_page.insert_text((40, 300), "CHARTLABEL")
    path = tmp_path / "dominant-form.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=400, height=500)
        page.show_pdf_page(page.rect, source, 0)
        page.insert_text((40, 40), "H")
        doc.save(path)
    source.close()
    reader.register("dominant", path, "application/pdf")

    indexed = PdfExtractor().extract(str(path))
    unit = reader.ok("dominant", start=1, count=1)["units"][0]
    assert any(
        chunk.page == 1 and "NESTEDVERTICALWORD" in chunk.text
        for chunk in indexed.chunks
    )
    assert "NESTEDVERTICALWORD" in unit["text"]
    assert "CHARTLABEL" in unit["text"]
    assert unit["text"].splitlines() == ["NESTEDVERTICALWORD", "CHARTLABEL", "H"]
