"""SPEC-ADDON-036: what a unit is, and its text, title and number, for each of
the five formats ``GET /files/{file_id}/document-text`` reads."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import app.config as config
from tests.document_fixtures import (
    DOCX,
    EPUB,
    PDF,
    PPTX,
    XLSX,
    make_docx,
    make_pdf,
    make_pptx,
    make_xlsx,
)
from tests.document_text_api import api, reader, search_index, seed  # noqa: F401
from tests.epub_fixtures import Item, make_epub, nav_doc, xhtml

UNIT_KEYS = {"number", "title", "text", "truncated", "extractor"}
_MARK = re.compile(r"MARK-[A-Za-z0-9]+")


def _indexed(path: Path):
    config_validate = config.validate_file_path
    config.validate_file_path = lambda _p: True
    try:
        if path.suffix == ".pdf":
            from app.extractors.pdf import PdfExtractor

            return PdfExtractor().extract(str(path))
        if path.suffix == ".epub":
            from app.extractors.epub import EpubExtractor

            return EpubExtractor().extract(str(path))
        from app.extractors.office import OfficeExtractor

        return OfficeExtractor().extract(str(path))
    finally:
        config.validate_file_path = config_validate


def _without_heading_line(chunk_text: str) -> str:
    return chunk_text.split("\n", 1)[1].strip() if "\n" in chunk_text else ""


def _marks_by_number(units: list[dict]) -> set[tuple[int, str]]:
    return {(u["number"], m) for u in units for m in _MARK.findall(u["text"])}


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


class TestPdf:
    def test_spec_addon_036_every_page_is_a_numbered_unit(self, reader, tmp_path):
        path = make_pdf(tmp_path / "a.pdf", ["MARK-p1 one", "MARK-p2 two", "MARK-p3 three"])
        reader.register("f1", path, PDF)

        body = reader.ok("f1")

        assert body["file_id"] == "f1"
        assert body["unit"] == "page"
        assert body["unit_count"] == 3
        assert body["next"] is None
        assert [u["number"] for u in body["units"]] == [1, 2, 3]
        for unit in body["units"]:
            assert set(unit) == UNIT_KEYS
            assert unit["title"] is None
            assert unit["truncated"] is False
            assert unit["extractor"] == "pymupdf4llm"
        assert _marks_by_number(body["units"]) == {(1, "MARK-p1"), (2, "MARK-p2"), (3, "MARK-p3")}

    def test_spec_addon_036_i3_i4_page_text_and_number_match_what_indexing_built(
        self, reader, tmp_path,
    ):
        from app.extractors.pdf import PAGE_SEPARATOR

        path = make_pdf(
            tmp_path / "a.pdf",
            ["MARK-p1 first line\nsecond line", "MARK-p2 only line", "MARK-p3 x\ny\nz"],
        )
        reader.register("f1", path, PDF)
        indexed = _indexed(path)

        body = reader.ok("f1")

        indexed_pages = indexed.markdown.split(PAGE_SEPARATOR)
        assert [u["text"].strip() for u in body["units"]] == [p.strip() for p in indexed_pages]
        searched = {(c.page, m) for c in indexed.chunks for m in _MARK.findall(c.text)}
        assert _marks_by_number(body["units"]) == searched

    def test_spec_addon_036_default_count_is_five_pages(self, reader, tmp_path):
        path = make_pdf(tmp_path / "a.pdf", [f"MARK-p{i}" for i in range(1, 8)])
        reader.register("f1", path, PDF)

        body = reader.ok("f1")

        assert body["unit_count"] == 7
        assert [u["number"] for u in body["units"]] == [1, 2, 3, 4, 5]
        assert body["next"] == {"start": 6, "offset": 0}

    def test_spec_addon_036_pages_past_the_indexing_cap_of_500_are_read(self, reader, tmp_path):
        path = make_pdf(tmp_path / "a.pdf", [f"MARK-p{i}" for i in range(1, 503)])
        reader.register("f1", path, PDF)

        body = reader.ok("f1", start=501, count=2)

        assert body["unit_count"] == 502
        assert _marks_by_number(body["units"]) == {(501, "MARK-p501"), (502, "MARK-p502")}
        assert body["next"] is None

    def test_spec_addon_036_a_page_pymupdf4llm_fails_on_is_read_as_plain_text(
        self, reader, tmp_path, pdf_conversions,
    ):
        import fitz

        path = make_pdf(tmp_path / "a.pdf", ["MARK-p1 a", "MARK-p2 b\nsecond", "MARK-p3 c"])
        reader.register("f1", path, PDF)
        pdf_conversions.fail_on = "MARK-p2"

        body = reader.ok("f1")

        assert [(u["number"], u["extractor"]) for u in body["units"]] == [
            (1, "pymupdf4llm"), (2, "fitz_fallback"), (3, "pymupdf4llm"),
        ]
        with fitz.open(str(path)) as doc:
            plain = doc[1].get_text()
        assert body["units"][1]["text"].strip() == plain.strip()

    def test_spec_addon_036_i5_only_the_requested_pages_are_converted_each_on_its_own(
        self, reader, tmp_path, pdf_conversions,
    ):
        path = make_pdf(tmp_path / "a.pdf", [f"MARK-p{i}" for i in range(1, 7)])
        reader.register("f1", path, PDF)

        body = reader.ok("f1", start=2, count=2)

        assert [u["number"] for u in body["units"]] == [2, 3]
        assert pdf_conversions.calls, "pymupdf4llm.to_markdown was not called"
        assert all(len(marks) == 1 for marks in pdf_conversions.calls)
        assert sorted(m for marks in pdf_conversions.calls for m in marks) == [
            "MARK-p2", "MARK-p3",
        ]

    def test_spec_addon_036_i5_a_ten_page_request_converts_at_most_ten_pages(
        self, reader, tmp_path, pdf_conversions,
    ):
        path = make_pdf(tmp_path / "a.pdf", [f"MARK-p{i}" for i in range(1, 16)])
        reader.register("f1", path, PDF)

        body = reader.ok("f1", count=10, max_chars=200_000)

        assert [u["number"] for u in body["units"]] == list(range(1, 11))
        assert len(pdf_conversions.calls) <= 10
        assert sum(len(m) for m in pdf_conversions.calls) <= 10


class _Conversions:
    def __init__(self):
        self.calls: list[list[str]] = []
        self.fail_on: str | None = None


@pytest.fixture()
def pdf_conversions(monkeypatch):
    """Records which pages each ``pymupdf4llm.to_markdown`` call converts, by the
    marks on those pages, and raises for a page carrying ``fail_on``."""
    import fitz
    import pymupdf4llm

    record = _Conversions()
    real = pymupdf4llm.to_markdown

    def spy(doc, *args, **kwargs):
        opened = doc if hasattr(doc, "page_count") else fitz.open(str(doc))
        pages = kwargs.get("pages")
        indices = list(pages) if pages is not None else list(range(opened.page_count))
        marks = [m for i in indices for m in _MARK.findall(opened[i].get_text())]
        record.calls.append(marks)
        if record.fail_on is not None and record.fail_on in marks:
            raise RuntimeError("simulated conversion failure")
        return real(doc, *args, **kwargs)

    monkeypatch.setattr(pymupdf4llm, "to_markdown", spy)
    return record


# ---------------------------------------------------------------------------
# docx
# ---------------------------------------------------------------------------


class TestDocx:
    def test_spec_addon_036_the_whole_document_is_one_unit(self, reader, tmp_path):
        path = make_docx(
            tmp_path / "a.docx",
            heading="Chapter",
            paragraphs=("Body text.", "Second."),
            table=(("a", "b"), ("c", "d")),
        )
        reader.register("f1", path, DOCX)

        body = reader.ok("f1")

        assert body["unit"] == "document"
        assert body["unit_count"] == 1
        assert body["next"] is None
        [unit] = body["units"]
        assert set(unit) == UNIT_KEYS
        assert (unit["number"], unit["title"], unit["truncated"], unit["extractor"]) == (
            1, None, False, "office",
        )
        assert [line for line in unit["text"].splitlines() if line.strip()] == [
            "# Chapter", "Body text.", "Second.", "a | b", "c | d",
        ]

    def test_spec_addon_036_i3_docx_text_matches_what_indexing_built(self, reader, tmp_path):
        path = make_docx(
            tmp_path / "a.docx",
            heading="Intro",
            paragraphs=("One.", "Two."),
            table=(("x", "y", "z"),),
        )
        reader.register("f1", path, DOCX)
        [chunk] = _indexed(path).chunks

        [unit] = reader.ok("f1")["units"]

        assert unit["text"].strip() == chunk.text.strip()

    def test_spec_addon_036_an_empty_document_is_one_empty_unit(self, reader, tmp_path):
        reader.register("f1", make_docx(tmp_path / "e.docx"), DOCX)

        body = reader.ok("f1")

        assert body["unit_count"] == 1
        assert body["units"] == [{
            "number": 1, "title": None, "text": "", "truncated": False, "extractor": "office",
        }]
        assert body["next"] is None


# ---------------------------------------------------------------------------
# xlsx
# ---------------------------------------------------------------------------


class TestXlsx:
    def test_spec_addon_036_every_sheet_in_workbook_order_including_empty_and_chartsheets(
        self, reader, tmp_path,
    ):
        path = make_xlsx(
            tmp_path / "a.xlsx",
            [
                ("Data", [["alpha", "beta"], ["gamma", "delta"]]),
                ("Chart", None),
                ("Empty", []),
                ("More", [["x"]]),
            ],
        )
        reader.register("f1", path, XLSX)

        body = reader.ok("f1")

        assert body["unit"] == "sheet"
        assert body["unit_count"] == 4
        assert body["next"] is None
        assert [(u["number"], u["title"]) for u in body["units"]] == [
            (1, "Data"), (2, "Chart"), (3, "Empty"), (4, "More"),
        ]
        assert [u["text"].splitlines() for u in body["units"]] == [
            ["alpha beta", "gamma delta"], [], [], ["x"],
        ]
        assert [u["text"] for u in body["units"]][1:3] == ["", ""]
        assert {u["extractor"] for u in body["units"]} == {"office"}
        assert {u["truncated"] for u in body["units"]} == {False}

    def test_spec_addon_036_i3_sheet_text_matches_indexing_less_its_heading_line(
        self, reader, tmp_path,
    ):
        path = make_xlsx(
            tmp_path / "a.xlsx",
            [("Data", [["a", None, "b"], [None, None], ["c"]]), ("Two", [["d", "e"]])],
        )
        reader.register("f1", path, XLSX)
        indexed = [_without_heading_line(c.text) for c in _indexed(path).chunks]

        body = reader.ok("f1")

        assert [u["text"].strip() for u in body["units"]] == indexed

    @pytest.mark.parametrize(
        "rows,lines,truncated",
        [
            pytest.param([[f"r{i}"] for i in range(1000)], 1000, False, id="1000-rows"),
            pytest.param([[f"r{i}"] for i in range(1001)], 1000, True, id="1001-rows"),
            pytest.param(
                [row for i in range(1000) for row in ([f"r{i}"], [None])], 1000, False,
                id="1000-non-empty-among-empty-rows",
            ),
        ],
    )
    def test_spec_addon_036_a_sheet_keeps_1000_non_empty_rows_and_flags_more(
        self, reader, tmp_path, rows, lines, truncated,
    ):
        reader.register("f1", make_xlsx(tmp_path / "a.xlsx", [("Data", rows)]), XLSX)

        [unit] = reader.ok("f1", max_chars=200_000)["units"]

        assert [line for line in unit["text"].splitlines() if line.strip()] == [
            f"r{i}" for i in range(lines)
        ]
        assert unit["truncated"] is truncated

    def test_spec_addon_036_default_count_is_five_sheets(self, reader, tmp_path):
        path = make_xlsx(tmp_path / "a.xlsx", [(f"S{i}", [[f"v{i}"]]) for i in range(1, 7)])
        reader.register("f1", path, XLSX)

        body = reader.ok("f1")

        assert [u["number"] for u in body["units"]] == [1, 2, 3, 4, 5]
        assert body["next"] == {"start": 6, "offset": 0}


# ---------------------------------------------------------------------------
# pptx
# ---------------------------------------------------------------------------


class TestPptx:
    def test_spec_addon_036_every_slide_is_a_unit_titled_when_it_has_a_title(
        self, reader, tmp_path,
    ):
        path = make_pptx(
            tmp_path / "a.pptx",
            [(None, ["loose", "two"]), ("Intro", ["first", "second"])],
        )
        reader.register("f1", path, PPTX)

        body = reader.ok("f1")

        assert body["unit"] == "slide"
        assert body["unit_count"] == 2
        assert body["next"] is None
        assert [(u["number"], u["title"]) for u in body["units"]] == [(1, None), (2, "Intro")]
        assert body["units"][0]["text"].splitlines() == ["loose", "two"]
        assert body["units"][1]["text"].strip() == "Intro\nfirst\nsecond"
        assert {u["extractor"] for u in body["units"]} == {"office"}

    def test_spec_addon_036_i3_untitled_slide_text_matches_indexing_less_its_heading_line(
        self, reader, tmp_path,
    ):
        path = make_pptx(tmp_path / "a.pptx", [(None, ["one", "two", "three"])])
        reader.register("f1", path, PPTX)
        [chunk] = _indexed(path).chunks

        [unit] = reader.ok("f1")["units"]

        assert unit["text"].strip() == _without_heading_line(chunk.text)

    def test_spec_addon_036_i3_titled_slide_text_keeps_the_title_less_the_heading_line(
        self, reader, tmp_path,
    ):
        path = make_pptx(tmp_path / "a.pptx", [("Intro", ["first", "second"])])
        reader.register("f1", path, PPTX)
        [chunk] = _indexed(path).chunks

        [unit] = reader.ok("f1")["units"]

        assert unit["title"] == "Intro"
        assert unit["text"].strip() == _without_heading_line(chunk.text) == "Intro\nfirst\nsecond"

    def test_spec_addon_036_default_count_is_fifty_slides(self, reader, tmp_path):
        path = make_pptx(tmp_path / "a.pptx", [(None, [f"s{i}"]) for i in range(1, 52)])
        reader.register("f1", path, PPTX)

        body = reader.ok("f1")

        assert body["unit_count"] == 51
        assert [u["number"] for u in body["units"]] == list(range(1, 51))
        assert body["next"] == {"start": 51, "offset": 0}


# ---------------------------------------------------------------------------
# EPUB
# ---------------------------------------------------------------------------


def _section(mark: str) -> str:
    return xhtml(f"<p>{mark}</p>")


def _padded(mark: str, size: int) -> str:
    # A comment is decompressed and parsed but leaves no text behind.
    return xhtml(f"<p>{mark}</p><!--{'x' * size}-->")


class TestEpub:
    def test_spec_addon_036_i4_every_spine_item_is_a_unit_numbered_as_search_numbers_it(
        self, reader, tmp_path,
    ):
        path = make_epub(
            tmp_path / "b.epub",
            [
                Item("cover", "cover.xhtml", _section("MARK-cover")),
                Item("nav", "nav.xhtml", nav_doc([("c1.xhtml", "One")]), properties="nav"),
                Item("img", "p.svg", '<svg xmlns="http://www.w3.org/2000/svg"><text>MARK-svg</text></svg>',
                     media_type="image/svg+xml"),
                Item("c1", "c1.xhtml", _section("MARK-one")),
                Item("c2", "c2.xhtml", _section("MARK-two")),
            ],
            spine=["cover", "nav", "img", "c1", "c2"],
        )
        reader.register("f1", path, EPUB)
        indexed = _indexed(path)

        body = reader.ok("f1")

        assert body["unit"] == "section"
        assert body["unit_count"] == 5 == indexed.page_count
        assert [u["number"] for u in body["units"]] == [1, 2, 3, 4, 5]
        searched = {(c.page, m) for c in indexed.chunks for m in _MARK.findall(c.text)}
        assert _marks_by_number(body["units"]) == searched
        assert [u["text"] for u in body["units"]][1:3] == ["", ""]
        assert {u["extractor"] for u in body["units"]} == {"epub"}

    def test_spec_addon_036_i4_a_section_title_is_the_stored_one_else_none(
        self, reader, seed, tmp_path,
    ):
        path = make_epub(
            tmp_path / "b.epub",
            [
                Item("nav", "nav.xhtml",
                     nav_doc([("c1.xhtml", "FromToc1"), ("c2.xhtml", "FromToc2")]),
                     properties="nav"),
                Item("c1", "c1.xhtml", _section("MARK-one")),
                Item("c2", "c2.xhtml", _section("MARK-two")),
                Item("c3", "c3.xhtml", _section("MARK-three")),
            ],
            spine=["c1", "c2", "c3"],
        )
        reader.register("f1", path, EPUB)
        seed.section("f1", 1, "Stored one")

        body = reader.ok("f1")

        assert [(u["number"], u["title"]) for u in body["units"]] == [
            (1, "Stored one"), (2, None), (3, None),
        ]

    @pytest.mark.parametrize(
        "body,expected",
        [
            pytest.param(
                "<p><ruby>吾輩<rp>（</rp><rt>わがはい</rt><rp>）</rp></ruby>は"
                "<ruby>猫<rt>ねこ</rt></ruby>である。</p>",
                "吾輩は猫である。",
                id="ruby-dropped",
            ),
            pytest.param(
                '<p>A claim<a href="notes.xhtml#n1" epub:type="noteref">1</a> stands.</p>',
                "A claim1 stands.",
                id="links-removed",
            ),
        ],
    )
    def test_spec_addon_036_i3_section_text_is_the_markdown_indexing_builds(
        self, reader, tmp_path, body, expected,
    ):
        path = make_epub(tmp_path / "b.epub", [Item("c1", "text/c1.xhtml", xhtml(body))])
        reader.register("f1", path, EPUB)
        [chunk] = _indexed(path).chunks

        [unit] = reader.ok("f1")["units"]

        assert unit["text"].strip() == chunk.text.strip() == expected

    @pytest.mark.parametrize(
        "first,extra",
        [
            pytest.param(Item("c1", "c1.xhtml", None), {}, id="missing-from-archive"),
            pytest.param(
                Item("c1", "c1.xhtml", "<svg xmlns='http://www.w3.org/2000/svg'>MARK-one</svg>",
                     media_type="image/svg+xml"),
                {}, id="not-html",
            ),
            pytest.param(Item("c1", "c1.xhtml", "<<not xml>> <p>MARK-one</p>"), {}, id="unparsable"),
            pytest.param(
                Item("c1", "c1.xhtml", _padded("MARK-one", 5 * 1024 * 1024 + 1024)),
                {}, id="over-5mb",
            ),
            pytest.param(
                Item("c1", "c1.xhtml", _section("MARK-one")),
                {"META-INF/encryption.xml": (
                    '<?xml version="1.0"?>'
                    '<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                    '<enc:EncryptedData xmlns:enc="http://www.w3.org/2001/04/xmlenc#">'
                    '<enc:CipherData><enc:CipherReference URI="OEBPS/c1.xhtml"/>'
                    "</enc:CipherData></enc:EncryptedData></encryption>"
                )},
                id="encrypted",
            ),
        ],
    )
    def test_spec_addon_036_an_unreadable_section_is_empty_and_the_rest_reads(
        self, reader, tmp_path, first, extra,
    ):
        path = make_epub(
            tmp_path / "b.epub",
            [first, Item("c2", "c2.xhtml", _section("MARK-two"))],
            spine=["c1", "c2"],
            extra=extra,
        )
        reader.register("f1", path, EPUB)

        body = reader.ok("f1")

        assert body["unit_count"] == 2
        assert [(u["number"], u["text"]) for u in body["units"]][0] == (1, "")
        assert _marks_by_number(body["units"]) == {(2, "MARK-two")}

    def test_spec_addon_036_sections_past_the_indexing_cap_of_2000_are_read(
        self, reader, tmp_path,
    ):
        path = make_epub(
            tmp_path / "b.epub",
            [Item(f"c{i}", f"c{i}.xhtml", _section(f"MARK-s{i}")) for i in range(1, 2002)],
        )
        reader.register("f1", path, EPUB)

        body = reader.ok("f1", start=2000, count=2)

        assert body["unit_count"] == 2001
        assert _marks_by_number(body["units"]) == {(2000, "MARK-s2000"), (2001, "MARK-s2001")}

    def test_spec_addon_036_text_past_the_indexing_cap_of_2m_characters_is_read(
        self, reader, tmp_path,
    ):
        words = "<p>" + "word " * 2_000 + "</p>"
        path = make_epub(
            tmp_path / "b.epub",
            [
                Item("c1", "c1.xhtml", xhtml(words * 220)),
                Item("c2", "c2.xhtml", _section("MARK-two")),
            ],
        )
        reader.register("f1", path, EPUB)

        body = reader.ok("f1", start=1, count=2, offset=2_150_000, max_chars=200_000)

        assert _marks_by_number(body["units"]) == {(2, "MARK-two")}

    def test_spec_addon_036_a_request_stops_before_the_section_that_passes_20mb(
        self, reader, tmp_path,
    ):
        size = 4 * 1024 * 1024 + 512 * 1024
        path = make_epub(
            tmp_path / "b.epub",
            [Item(f"c{i}", f"c{i}.xhtml", _padded(f"MARK-s{i}", size)) for i in range(1, 7)],
        )
        reader.register("f1", path, EPUB)

        body = reader.ok("f1")

        assert [u["number"] for u in body["units"]] == [1, 2, 3, 4]
        assert _marks_by_number(body["units"]) == {(i, f"MARK-s{i}") for i in range(1, 5)}
        assert body["next"] == {"start": 5, "offset": 0}

        rest = reader.ok("f1", **body["next"])

        assert [u["number"] for u in rest["units"]][0] == 5
        assert (5, "MARK-s5") in _marks_by_number(rest["units"])

    def test_spec_addon_036_default_count_is_ten_sections(self, reader, tmp_path):
        path = make_epub(
            tmp_path / "b.epub",
            [Item(f"c{i}", f"c{i}.xhtml", _section(f"MARK-s{i}")) for i in range(1, 12)],
        )
        reader.register("f1", path, EPUB)

        body = reader.ok("f1")

        assert [u["number"] for u in body["units"]] == list(range(1, 11))
        assert body["next"] == {"start": 11, "offset": 0}

    def test_spec_addon_036_i5_only_sections_in_the_requested_range_are_read(
        self, reader, tmp_path, monkeypatch,
    ):
        import zipfile

        path = make_epub(
            tmp_path / "b.epub",
            [Item(f"c{i}", f"c{i}.xhtml", _section(f"MARK-s{i}")) for i in range(1, 6)],
        )
        reader.register("f1", path, EPUB)
        opened: list[str] = []
        real_open = zipfile.ZipFile.open

        def spy(self, name, *args, **kwargs):
            opened.append(name if isinstance(name, str) else name.filename)
            return real_open(self, name, *args, **kwargs)

        monkeypatch.setattr(zipfile.ZipFile, "open", spy)

        body = reader.ok("f1", start=2, count=2)

        assert [u["number"] for u in body["units"]] == [2, 3]
        sections_read = {n for n in opened if re.fullmatch(r"OEBPS/c\d\.xhtml", n)}
        assert sections_read == {"OEBPS/c2.xhtml", "OEBPS/c3.xhtml"}

    def test_spec_addon_036_a_book_with_no_spine_has_no_units(self, reader, tmp_path):
        path = make_epub(tmp_path / "b.epub", [Item("c1", "c1.xhtml", _section("MARK-one"))], spine=[])
        reader.register("f1", path, EPUB)

        body = reader.ok("f1")

        assert (body["unit_count"], body["units"], body["next"]) == (0, [], None)
