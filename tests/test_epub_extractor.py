"""Unit tests for ``EpubExtractor``."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import app.config as config
import app.extractors.epub as epub_module
from app.extractors.base import ExtractionResult
from app.extractors.epub import EpubExtractor
from tests.epub_fixtures import (
    OPF_NS,
    Item,
    container_xml,
    make_epub,
    nav_doc,
    ncx_doc,
    xhtml,
)

_MARK_RE = re.compile(r"MARK-[A-Za-z0-9]+")


@pytest.fixture(autouse=True)
def allow_tmp_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "validate_file_path", lambda _path: True)


def _extract(path: Path) -> ExtractionResult:
    return EpubExtractor().extract(str(path))


def _marks(result: ExtractionResult) -> set[tuple[int | None, str]]:
    return {
        (chunk.page, mark)
        for chunk in result.chunks
        for mark in _MARK_RE.findall(chunk.text)
    }


def _section(mark: str) -> str:
    return xhtml(f"<p>{mark}</p>")


# ---------------------------------------------------------------------------
# Numbering
# ---------------------------------------------------------------------------


def test_can_handle_epub_only() -> None:
    extractor = EpubExtractor()
    assert extractor.can_handle("book.epub") is True
    assert extractor.can_handle("BOOK.EPUB") is True
    assert extractor.can_handle("book.pdf") is False
    assert extractor.can_handle("book.xhtml") is False
    assert extractor.can_handle("book.zip") is False


def test_validate_file_path_false_returns_empty(tmp_path, monkeypatch) -> None:
    path = make_epub(tmp_path / "b.epub", [Item("c1", "c1.xhtml", _section("MARK-one"))])
    monkeypatch.setattr(config, "validate_file_path", lambda _path: False)

    result = _extract(path)

    assert result.chunks == []
    assert result.section_titles == ()
    assert result.page_count is None


def test_dangling_idref_is_not_counted(tmp_path) -> None:
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("c1", "c1.xhtml", _section("MARK-one")),
            Item("c2", "c2.xhtml", _section("MARK-two")),
        ],
        spine=["ghost", "c1", "c2"],
    )

    result = _extract(path)

    assert _marks(result) == {(1, "MARK-one"), (2, "MARK-two")}
    assert result.page_count == 2
    assert result.extractor == "epub"
    assert result.markdown is None


def test_numbering_matches_foliate_rule(tmp_path) -> None:
    manifest = (
        '<item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" '
        'properties="nav"/>'
        '<item id="img" href="p.jpg" media-type="image/jpeg"/>'
        '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="c2" href="c2.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="c2" href="c2dup.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="c3" href="c3.xhtml" media-type="application/xhtml+xml"/>'
    )
    spine = (
        '<itemref idref="cover"/>'
        '<itemref idref="nav"/>'
        '<itemref idref="img"/>'
        '<itemref idref="c1"/>'
        '<itemref idref="dangling"/>'
        '<x:itemref xmlns:x="urn:other" idref="c3"/>'
        '<group><itemref idref="c3"/></group>'
        '<itemref idref="c2"/>'
        '<itemref idref="c1"/>'
        '<itemref idref="c3" linear="no"/>'
    )
    opf = (
        f'<?xml version="1.0"?><package xmlns="{OPF_NS}" version="3.0">'
        f"<manifest>{manifest}</manifest>"
        f"<spine>{spine}</spine>"
        '<spine><itemref idref="c1"/></spine>'
        '<wrapper><spine><itemref idref="c1"/></spine></wrapper>'
        "</package>"
    )
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("cover", "cover.xhtml", _section("MARK-cover")),
            Item("nav", "nav.xhtml", nav_doc([("c1.xhtml", "MARK-nav")])),
            Item("img", "p.jpg", b"\xff\xd8\xff", media_type="image/jpeg"),
            Item("c1", "c1.xhtml", _section("MARK-one")),
            Item("c2", "c2.xhtml", _section("MARK-two")),
            Item("c2dup", "c2dup.xhtml", _section("MARK-twodup")),
            Item("c3", "c3.xhtml", _section("MARK-three")),
        ],
        opf=opf,
    )

    result = _extract(path)

    assert _marks(result) == {
        (1, "MARK-cover"),
        (4, "MARK-one"),
        (5, "MARK-two"),
        (6, "MARK-one"),
        (7, "MARK-three"),
    }
    assert result.page_count == 7


def test_numbering_without_opf_namespace_matches_local_names(tmp_path) -> None:
    opf = (
        '<?xml version="1.0"?><package version="2.0">'
        '<manifest>'
        '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="c2" href="c2.xhtml" media-type="application/xhtml+xml"/>'
        "</manifest>"
        '<spine><itemref idref="c1"/><x:itemref xmlns:x="urn:other" idref="c2"/>'
        "</spine></package>"
    )
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("c1", "c1.xhtml", _section("MARK-one")),
            Item("c2", "c2.xhtml", _section("MARK-two")),
        ],
        opf=opf,
    )

    result = _extract(path)

    assert _marks(result) == {(1, "MARK-one"), (2, "MARK-two")}


def test_hrefs_resolve_against_the_opf_directory(tmp_path) -> None:
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("c1", "text/c1.xhtml", _section("MARK-one")),
            Item("c2", "text/c2.xhtml", _section("MARK-two")),
        ],
        opf_dir="item/standard",
    )

    result = _extract(path)

    assert _marks(result) == {(1, "MARK-one"), (2, "MARK-two")}


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------


def _heading(level: int, text: str, mark: str = "") -> str:
    return xhtml(f"<h{level}>{text}</h{level}><p>body {mark}</p>")


def test_nav_titles_first_entry_wins(tmp_path) -> None:
    nav = nav_doc(
        [
            ("../text/c1.xhtml", "Chapter One"),
            ("../text/c1.xhtml#s2", "Section 1.2"),
            ("../text/c2.xhtml#top", "Chapter Two"),
            ("../text/missing.xhtml", "Not in spine"),
        ],
        extra_navs=(
            '<nav epub:type="landmarks"><ol>'
            '<li><a href="../text/c2.xhtml">Landmark</a></li></ol></nav>'
        ),
    )
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("nav", "nav/toc.xhtml", nav, properties="nav"),
            Item("c1", "text/c1.xhtml", _section("MARK-one")),
            Item("c2", "text/c2.xhtml", _section("MARK-two")),
        ],
        spine=["c1", "c2"],
    )

    assert _extract(path).section_titles == ((1, "Chapter One"), (2, "Chapter Two"))


@pytest.mark.parametrize("spine_toc", ["ncx", None])
def test_ncx_used_when_no_nav(tmp_path, spine_toc) -> None:
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("ncx", "toc.ncx", ncx_doc([("c1.xhtml", "One"), ("c2.xhtml#x", "Two")]),
                 media_type="application/x-dtbncx+xml"),
            Item("c1", "c1.xhtml", _section("MARK-one")),
            Item("c2", "c2.xhtml", _section("MARK-two")),
        ],
        spine=["c1", "c2"],
        spine_toc=spine_toc,
    )

    assert _extract(path).section_titles == ((1, "One"), (2, "Two"))


def test_ncx_used_when_nav_has_no_toc(tmp_path) -> None:
    nav = xhtml('<nav epub:type="landmarks"><ol><li><a href="c1.xhtml">L</a></li></ol></nav>')
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("nav", "nav.xhtml", nav, properties="nav"),
            Item("ncx", "toc.ncx", ncx_doc([("c1.xhtml", "From NCX")]),
                 media_type="application/x-dtbncx+xml"),
            Item("c1", "c1.xhtml", _section("MARK-one")),
        ],
        spine=["c1"],
        spine_toc="ncx",
    )

    assert _extract(path).section_titles == ((1, "From NCX"),)


def test_untitled_section_inherits_preceding_toc_title(tmp_path) -> None:
    nav = nav_doc([("c1.xhtml", "One"), ("c3.xhtml", "Three")])
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("nav", "nav.xhtml", nav, properties="nav"),
            *(Item(f"c{i}", f"c{i}.xhtml", _heading(1, f"Heading {i}")) for i in range(1, 5)),
        ],
        spine=["c1", "c2", "c3", "c4"],
    )

    assert _extract(path).section_titles == (
        (1, "One"), (2, "One"), (3, "Three"), (4, "Three"),
    )


def test_section_before_first_toc_entry_uses_heading(tmp_path) -> None:
    nav = nav_doc([("c3.xhtml", "One")])
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("nav", "nav.xhtml", nav, properties="nav"),
            Item("c1", "c1.xhtml", _section("MARK-cover")),
            Item("c2", "c2.xhtml", xhtml("<p>intro</p><h3>Title Page</h3><h1>Later</h1>")),
            Item("c3", "c3.xhtml", _heading(1, "Chapter heading")),
            Item("c4", "c4.xhtml", _heading(1, "Ignored heading")),
        ],
        spine=["c1", "c2", "c3", "c4"],
    )

    assert _extract(path).section_titles == (
        (2, "Title Page"), (3, "One"), (4, "One"),
    )


def test_no_toc_no_heading_no_title_row(tmp_path) -> None:
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("c1", "c1.xhtml", _section("MARK-one")),
            Item("c2", "c2.xhtml", _section("MARK-two")),
        ],
    )

    result = _extract(path)

    assert result.section_titles == ()
    assert _marks(result) == {(1, "MARK-one"), (2, "MARK-two")}


def test_title_whitespace_collapsed_and_capped_at_200(tmp_path) -> None:
    label = "  Long \n\t title  " + "x" * 300
    nav = nav_doc([("c1.xhtml", label), ("c2.xhtml", " \n ")])
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("nav", "nav.xhtml", nav, properties="nav"),
            Item("c1", "c1.xhtml", _section("MARK-one")),
            Item("c2", "c2.xhtml", _section("MARK-two")),
        ],
        spine=["c1", "c2"],
    )

    expected = "Long title " + "x" * 189
    assert _extract(path).section_titles == ((1, expected), (2, expected))


def test_title_drops_ruby_readings(tmp_path) -> None:
    ruby = "<ruby>漢字<rp>(</rp><rt>かんじ</rt><rp>)</rp></ruby>"
    nav = nav_doc([("c2.xhtml", f"{ruby}の章")])
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("nav", "nav.xhtml", nav, properties="nav"),
            Item("c1", "c1.xhtml", _heading(2, f"前書き{ruby}")),
            Item("c2", "c2.xhtml", _section("MARK-two")),
        ],
        spine=["c1", "c2"],
    )

    assert _extract(path).section_titles == ((1, "前書き漢字"), (2, "漢字の章"))


# ---------------------------------------------------------------------------
# Body text
# ---------------------------------------------------------------------------


def _chunk_texts(result: ExtractionResult) -> list[tuple[int | None, str]]:
    return [(c.page, c.text) for c in result.chunks]


def test_ruby_readings_dropped_base_kept(tmp_path) -> None:
    body = (
        "<p><ruby>吾輩<rp>（</rp><rt>わがはい</rt><rp>）</rp></ruby>は"
        "<ruby>猫<rt>ねこ</rt></ruby>である。</p>"
    )
    path = make_epub(tmp_path / "b.epub", [Item("c1", "c1.xhtml", xhtml(body))])

    assert _chunk_texts(_extract(path)) == [(1, "吾輩は猫である。")]


def test_note_links_do_not_put_paths_in_text(tmp_path) -> None:
    body = '<p>A claim<a href="notes.xhtml#n1" epub:type="noteref">1</a> stands.</p>'
    path = make_epub(tmp_path / "b.epub", [Item("c1", "text/c1.xhtml", xhtml(body))])

    assert _chunk_texts(_extract(path)) == [(1, "A claim1 stands.")]


def test_script_style_stripped(tmp_path) -> None:
    body = (
        "<style>.x { color: red }</style><script>var secret = 1;</script>"
        "<noscript>fallback</noscript><p>Visible text.</p>"
    )
    path = make_epub(tmp_path / "b.epub", [Item("c1", "c1.xhtml", xhtml(body))])

    assert _chunk_texts(_extract(path)) == [(1, "Visible text.")]


def test_nav_document_not_indexed(tmp_path) -> None:
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("nav", "nav.xhtml", nav_doc([("c1.xhtml", "MARK-navlabel")]),
                 properties="nav"),
            Item("c1", "c1.xhtml", _section("MARK-one")),
        ],
        spine=["nav", "c1"],
    )

    result = _extract(path)

    assert _marks(result) == {(2, "MARK-one")}
    assert result.page_count == 2


def test_image_spine_item_counted_not_indexed(tmp_path) -> None:
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("img", "cover.svg", '<svg xmlns="http://www.w3.org/2000/svg">'
                 "<text>MARK-svg</text></svg>", media_type="image/svg+xml"),
            Item("c1", "c1.xhtml", _section("MARK-one")),
        ],
        spine=["img", "c1"],
    )

    result = _extract(path)

    assert _marks(result) == {(2, "MARK-one")}
    assert result.page_count == 2


def test_every_chunk_page_is_its_section_number(tmp_path) -> None:
    items = [
        Item(f"c{n}", f"c{n}.xhtml", xhtml("".join(
            f"<p>Word{n} sentence number {k}.</p>" for k in range(60)
        )))
        for n in (1, 2, 3)
    ]
    path = make_epub(tmp_path / "b.epub", items)

    result = _extract(path)

    for chunk in result.chunks:
        words = set(re.findall(r"Word(\d)", chunk.text))
        assert words == {str(chunk.page)}
    assert {c.page for c in result.chunks} == {1, 2, 3}


def test_chunk_size_follows_config(tmp_path, monkeypatch) -> None:
    from dataclasses import replace

    from app.config import TextChunkingConfig

    settings = config.settings
    monkeypatch.setattr(
        config,
        "settings",
        replace(
            settings,
            indexing=replace(
                settings.indexing,
                text_chunking=TextChunkingConfig(max_chunk_size=20, overlap=0),
            ),
        ),
    )
    path = make_epub(
        tmp_path / "b.epub",
        [Item("c1", "c1.xhtml", xhtml("<p>One two three four five six seven eight</p>"))],
    )

    assert _chunk_texts(_extract(path)) == [
        (1, "One two three four"),
        (1, "five six seven"),
        (1, "eight"),
    ]
