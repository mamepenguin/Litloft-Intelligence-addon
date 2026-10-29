"""SPEC-ADDON-036 I9: indexing's ``extract()`` output for PDF and Office files,
read through the real libraries. Indexing does not read a workbook that holds
a chartsheet, and that failure is part of the output kept."""

from __future__ import annotations

from pathlib import Path

import pytest

import app.config as config
from tests.document_fixtures import make_docx, make_pdf, make_pptx, make_xlsx


@pytest.fixture(autouse=True)
def allow_tmp_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "validate_file_path", lambda _path: True)


def _extract(path: Path):
    if path.suffix == ".pdf":
        from app.extractors.pdf import PdfExtractor

        return PdfExtractor().extract(str(path))
    from app.extractors.office import OfficeExtractor

    return OfficeExtractor().extract(str(path))


CASES = [
    pytest.param(
        lambda t: make_pdf(t / "g.pdf", ["Page one line\nsecond line", "Page two"]),
        (
            [(1, "Page one line\nsecond line\n\n\n-----"), (2, "Page two\n\n\n-----")],
            "Page one line\nsecond line\n\n\n-----\n\n\n\n---\n\nPage two\n\n\n-----\n\n",
            "pymupdf4llm", 2, (),
        ),
        id="pdf",
    ),
    pytest.param(
        lambda t: make_docx(
            t / "g.docx", heading="Chapter", paragraphs=("Body text.", "Second."),
            table=(("a", "b"), ("c", "d")),
        ),
        ([(None, "# Chapter\nBody text.\nSecond.\n\na | b\nc | d")], None, "office", None, ()),
        id="docx",
    ),
    pytest.param(
        lambda t: make_xlsx(
            t / "g.xlsx",
            [("Data", [["alpha", "beta"], ["gamma", None, "delta"]]), ("Empty", []), ("More", [["x"]])],
        ),
        (
            [(None, "## Sheet: Data\n\nalpha beta\ngamma delta"), (None, "## Sheet: More\n\nx")],
            None, "office", None, (),
        ),
        id="xlsx",
    ),
    pytest.param(
        lambda t: make_xlsx(t / "gc.xlsx", [("Data", [["alpha", "beta"]]), ("Chart", None)]),
        ([], None, "office_error", None, ()),
        id="xlsx-with-chartsheet",
    ),
    pytest.param(
        lambda t: make_pptx(t / "g.pptx", [("Intro", ["first", "second"]), (None, ["loose"])]),
        (
            [(None, "## Slide 1: Intro\n\nIntro\nfirst\nsecond"), (None, "## Slide 2\n\nloose")],
            None, "office", None, (),
        ),
        id="pptx",
    ),
]


@pytest.mark.parametrize("build,expected", CASES)
def test_spec_addon_036_i9_extract_output_is_unchanged(tmp_path, build, expected) -> None:
    result = _extract(build(tmp_path))

    assert (
        [(c.page, c.text) for c in result.chunks],
        result.markdown,
        result.extractor,
        result.page_count,
        result.section_titles,
    ) == expected
