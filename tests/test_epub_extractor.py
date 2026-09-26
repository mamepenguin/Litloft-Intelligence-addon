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
    opf_xml,
    xhtml,
    XHTML_DOCTYPE,
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


def test_nav_preferred_over_ncx(tmp_path) -> None:
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("nav", "nav.xhtml", nav_doc([("c1.xhtml", "From nav")]), properties="nav"),
            Item("ncx", "toc.ncx", ncx_doc([("c1.xhtml", "From NCX")]),
                 media_type="application/x-dtbncx+xml"),
            Item("c1", "c1.xhtml", _section("MARK-one")),
        ],
        spine=["c1"],
        spine_toc="ncx",
    )

    assert _extract(path).section_titles == ((1, "From nav"),)


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


# ---------------------------------------------------------------------------
# Untrusted input
# ---------------------------------------------------------------------------

_LAUGHS = (
    '<!DOCTYPE lolz [<!ENTITY lol "lol">'
    '<!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
    '<!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">'
    '<!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">]>'
)


def _two_sections(first: str | bytes, second: str | bytes | None = None) -> list[Item]:
    return [
        Item("c1", "c1.xhtml", first),
        Item("c2", "c2.xhtml", second if second is not None else _section("MARK-two")),
    ]


def _empty(result: ExtractionResult) -> bool:
    return (
        result.chunks == []
        and result.section_titles == ()
        and result.page_count is None
        and result.extractor == "epub"
    )


def test_caps_are_declared() -> None:
    assert (
        epub_module.XML_MAX_BYTES,
        epub_module.SECTION_MAX_BYTES,
        epub_module.TEXT_MAX_CHARS,
        epub_module.SECTION_MAX,
        epub_module.TITLE_MAX,
    ) == (1024 * 1024, 5 * 1024 * 1024, 2_000_000, 2_000, 200)


def test_entity_declaration_in_section_is_refused(tmp_path) -> None:
    first = (
        '<?xml version="1.0"?><!DOCTYPE html [<!ENTITY x "MARK-expanded">]>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>&x; MARK-one</p>'
        "</body></html>"
    )
    path = make_epub(tmp_path / "b.epub", _two_sections(first))

    result = _extract(path)

    assert _marks(result) == {(2, "MARK-two")}
    assert result.page_count == 2


def test_entity_declaration_in_nav_falls_back_to_ncx(tmp_path) -> None:
    nav = (
        '<?xml version="1.0"?><!DOCTYPE html [<!ENTITY t "Expanded">]>'
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops"><body>'
        '<nav epub:type="toc"><ol><li><a href="c1.xhtml">&t;</a></li></ol></nav>'
        "</body></html>"
    )
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("nav", "nav.xhtml", nav, properties="nav"),
            Item("ncx", "toc.ncx", ncx_doc([("c1.xhtml", "From NCX")]),
                 media_type="application/x-dtbncx+xml"),
            Item("c1", "c1.xhtml", _section("MARK-one")),
        ],
        spine=["c1"],
    )

    assert _extract(path).section_titles == ((1, "From NCX"),)


@pytest.mark.parametrize("doctype", ["<!DOCTYPE html>", XHTML_DOCTYPE])
def test_bare_doctype_section_indexed(tmp_path, doctype) -> None:
    path = make_epub(
        tmp_path / "b.epub",
        _two_sections(xhtml("<p>MARK-one</p>", doctype=doctype)),
    )

    assert _marks(_extract(path)) == {(1, "MARK-one"), (2, "MARK-two")}


def test_external_dtd_is_never_fetched(tmp_path) -> None:
    dtd = tmp_path / "evil.dtd"
    dtd.write_text('<!ENTITY leak "MARK-leaked">', encoding="utf-8")
    doctype = f'<!DOCTYPE html SYSTEM "{dtd.as_uri()}">'
    ncx = ncx_doc([("c1.xhtml", "Title")]).replace(
        '<!DOCTYPE ncx PUBLIC "-//NISO//DTD ncx 2005-1//EN" '
        '"http://www.daisy.org/z3986/2005/ncx-2005-1.dtd">',
        f'<!DOCTYPE ncx SYSTEM "{dtd.as_uri()}">',
    ).replace("<text>Title</text>", "<text>&leak;Title</text>")
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("ncx", "toc.ncx", ncx, media_type="application/x-dtbncx+xml"),
            Item("c1", "c1.xhtml", xhtml("<p>&leak; MARK-one</p>", doctype=doctype)),
        ],
        spine=["c1"],
        spine_toc="ncx",
    )

    result = _extract(path)

    assert _marks(result) == {(1, "MARK-one")}
    assert result.section_titles == ((1, "Title"),)


def test_billion_laughs_in_container_returns_empty(tmp_path) -> None:
    container = (
        f'<?xml version="1.0"?>{_LAUGHS}'
        '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles>'
        "<x>&lol3;</x></container>"
    )
    path = make_epub(tmp_path / "b.epub", _two_sections(_section("MARK-one")),
                     container=container)

    assert _empty(_extract(path))


def test_opf_with_doctype_is_refused(tmp_path) -> None:
    items = _two_sections(_section("MARK-one"))
    opf = opf_xml(items, ["c1", "c2"]).replace(
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE package>',
    )
    path = make_epub(tmp_path / "b.epub", items, opf=opf)

    assert _empty(_extract(path))


def test_reads_are_bounded_on_decompressed_bytes(tmp_path, monkeypatch) -> None:
    import zipfile

    monkeypatch.setattr(epub_module, "SECTION_MAX_BYTES", 4096)
    monkeypatch.setattr(epub_module, "XML_MAX_BYTES", 2048)
    huge = xhtml("<p>MARK-one</p>" + "<p>padding</p>" * 100_000)
    path = make_epub(tmp_path / "b.epub", _two_sections(huge))

    requested: list[int] = []
    original_read = zipfile.ZipExtFile.read

    def spy_read(self, n=-1):
        requested.append(n)
        return original_read(self, n)

    monkeypatch.setattr(zipfile.ZipExtFile, "read", spy_read)

    result = _extract(path)

    assert _marks(result) == {(2, "MARK-two")}
    assert requested
    assert all(0 < n <= 4096 + 1 for n in requested)


def test_member_over_cap_skipped_rest_indexed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(epub_module, "SECTION_MAX_BYTES", 2000)
    first = xhtml("<p>MARK-one</p>" + "<p>x</p>" * 300)
    third = xhtml("<p>MARK-three</p>")
    path = make_epub(
        tmp_path / "b.epub",
        [*_two_sections(first), Item("c3", "c3.xhtml", third)],
    )

    result = _extract(path)

    assert _marks(result) == {(2, "MARK-two"), (3, "MARK-three")}
    assert result.page_count == 3


def test_toc_over_cap_skipped_rest_indexed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(epub_module, "XML_MAX_BYTES", 3000)
    nav = nav_doc([("c1.xhtml", "One")] + [("c1.xhtml", "pad" * 20)] * 100)
    path = make_epub(
        tmp_path / "b.epub",
        [Item("nav", "nav.xhtml", nav, properties="nav"), *_two_sections(_section("MARK-one"))],
        spine=["c1", "c2"],
    )

    result = _extract(path)

    assert _marks(result) == {(1, "MARK-one"), (2, "MARK-two")}
    assert result.section_titles == ()


def test_book_text_cap_stops_and_keeps_extracted(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(epub_module, "TEXT_MAX_CHARS", 25)
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("c1", "c1.xhtml", xhtml("<p>" + "a" * 10 + "</p>")),
            Item("c2", "c2.xhtml", xhtml("<p>" + "b" * 20 + "</p>")),
            Item("c3", "c3.xhtml", xhtml("<p>" + "c" * 5 + "</p>")),
        ],
    )

    result = _extract(path)

    assert _chunk_texts(result) == [(1, "a" * 10), (2, "b" * 15)]
    assert result.page_count == 3


def test_section_cap_2000(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(epub_module, "SECTION_MAX", 2)
    nav = nav_doc([("c1.xhtml", "One"), ("c2.xhtml", "Two"), ("c3.xhtml", "Three")])
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("nav", "nav.xhtml", nav, properties="nav"),
            *_two_sections(_section("MARK-one")),
            Item("c3", "c3.xhtml", _section("MARK-three")),
        ],
        spine=["c1", "c2", "c3"],
    )

    result = _extract(path)

    assert _marks(result) == {(1, "MARK-one"), (2, "MARK-two")}
    assert result.section_titles == ((1, "One"), (2, "Two"))


def _encryption_xml(*uris: str) -> str:
    refs = "".join(
        '<enc:EncryptedData xmlns:enc="http://www.w3.org/2001/04/xmlenc#">'
        f'<enc:CipherData><enc:CipherReference URI="{uri}"/></enc:CipherData>'
        "</enc:EncryptedData>"
        for uri in uris
    )
    return (
        '<?xml version="1.0"?>'
        '<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        f"{refs}</encryption>"
    )


def test_encryption_xml_listed_section_skipped(tmp_path) -> None:
    nav = nav_doc([("c1.xhtml", "One"), ("c2.xhtml", "Two")])
    path = make_epub(
        tmp_path / "b.epub",
        [Item("nav", "nav.xhtml", nav, properties="nav"), *_two_sections(_section("MARK-one"))],
        spine=["c1", "c2"],
        extra={"META-INF/encryption.xml": _encryption_xml("OEBPS/c1.xhtml", "OEBPS/font.otf")},
    )

    result = _extract(path)

    assert _marks(result) == {(2, "MARK-two")}
    assert result.section_titles == ((1, "One"), (2, "Two"))


def test_encryption_xml_listed_nav_is_not_read(tmp_path) -> None:
    nav = nav_doc([("c1.xhtml", "One")])
    path = make_epub(
        tmp_path / "b.epub",
        [Item("nav", "nav.xhtml", nav, properties="nav"), *_two_sections(_section("MARK-one"))],
        spine=["c1", "c2"],
        extra={"META-INF/encryption.xml": _encryption_xml("OEBPS/nav.xhtml")},
    )

    result = _extract(path)

    assert _marks(result) == {(1, "MARK-one"), (2, "MARK-two")}
    assert result.section_titles == ()


def test_unparseable_encryption_xml_skips_all_sections(tmp_path) -> None:
    nav = nav_doc([("c1.xhtml", "One")])
    path = make_epub(
        tmp_path / "b.epub",
        [Item("nav", "nav.xhtml", nav, properties="nav"), *_two_sections(_section("MARK-one"))],
        spine=["c1", "c2"],
        extra={"META-INF/encryption.xml": "<encryption><unclosed"},
    )

    result = _extract(path)

    assert result.chunks == []
    assert result.section_titles == ()
    assert result.page_count == 2


def test_encryption_xml_over_cap_skips_all_sections(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(epub_module, "XML_MAX_BYTES", 3000)
    path = make_epub(
        tmp_path / "b.epub",
        _two_sections(_section("MARK-one")),
        extra={"META-INF/encryption.xml": _encryption_xml(*(f"x{i}.otf" for i in range(100)))},
    )

    result = _extract(path)

    assert result.chunks == []
    assert result.page_count == 2


def test_corrupt_zip_returns_empty(tmp_path) -> None:
    path = tmp_path / "b.epub"
    path.write_bytes(b"PK\x03\x04 this is not a zip archive" * 10)

    assert _empty(_extract(path))


def test_truncated_zip_returns_empty(tmp_path) -> None:
    path = make_epub(tmp_path / "b.epub", _two_sections(_section("MARK-one")))
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])

    assert _empty(_extract(path))


def test_zip_encrypted_member_skipped_rest_indexed(tmp_path) -> None:
    path = make_epub(
        tmp_path / "b.epub",
        _two_sections(_section("MARK-one")),
        encrypted_members=("OEBPS/c1.xhtml",),
    )

    result = _extract(path)

    assert _marks(result) == {(2, "MARK-two")}
    assert result.page_count == 2


def test_missing_container_returns_empty(tmp_path) -> None:
    import zipfile

    path = tmp_path / "b.epub"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("OEBPS/content.opf", "<package/>")
        zf.writestr("OEBPS/c1.xhtml", _section("MARK-one"))

    assert _empty(_extract(path))


@pytest.mark.parametrize(
    "container",
    [
        container_xml("OEBPS/elsewhere.opf"),
        container_xml("../content.opf"),
        container_xml("OEBPS/content.opf").replace(
            "application/oebps-package+xml", "application/xml",
        ),
    ],
)
def test_container_without_a_readable_opf_returns_empty(tmp_path, container) -> None:
    path = make_epub(
        tmp_path / "b.epub",
        _two_sections(_section("MARK-one")),
        container=container,
    )

    assert _empty(_extract(path))


def test_malformed_after_root_still_indexed(tmp_path) -> None:
    first = (
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        "<p>MARK-one&nbsp;and a stray & here</p><p>unclosed</body></html>"
    )
    path = make_epub(tmp_path / "b.epub", _two_sections(first))

    result = _extract(path)

    assert _marks(result) == {(1, "MARK-one"), (2, "MARK-two")}


def test_malformed_before_root_skips_only_that_section(tmp_path) -> None:
    path = make_epub(
        tmp_path / "b.epub",
        _two_sections("<<not xml>> <p>MARK-one</p>"),
    )

    result = _extract(path)

    assert _marks(result) == {(2, "MARK-two")}
    assert result.page_count == 2


def test_href_traversal_ignored(tmp_path) -> None:
    import zipfile

    items = [
        Item("evil", "../../etc/passwd", None),
        Item("abs", "/etc/passwd", None),
        Item("c1", "c1.xhtml", _section("MARK-one")),
    ]
    path = make_epub(tmp_path / "b.epub", items, spine=["evil", "abs", "c1"])
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr("../etc/passwd", _section("MARK-evil"))
        zf.writestr("etc/passwd", _section("MARK-evil"))

    result = _extract(path)

    assert _marks(result) == {(3, "MARK-one")}
    assert result.page_count == 3


def test_cp932_member_names_resolved(tmp_path) -> None:
    nav = nav_doc([("本文/第一章.xhtml", "第一章")])
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("nav", "目次.xhtml", nav, properties="nav"),
            Item("c1", "本文/第一章.xhtml", _section("MARK-one")),
        ],
        spine=["c1"],
        cp932_names=True,
    )

    result = _extract(path)

    assert _marks(result) == {(1, "MARK-one")}
    assert result.section_titles == ((1, "第一章"),)


def test_mutated_books_never_raise(tmp_path) -> None:
    import random

    nav = nav_doc([("c1.xhtml", "One"), ("c2.xhtml", "Two")])
    base = make_epub(
        tmp_path / "base.epub",
        [
            Item("nav", "nav.xhtml", nav, properties="nav"),
            Item("ncx", "toc.ncx", ncx_doc([("c1.xhtml", "One")]),
                 media_type="application/x-dtbncx+xml"),
            *_two_sections(xhtml("<h1>Head</h1><p>MARK-one " + "text " * 200 + "</p>")),
        ],
        spine=["c1", "c2"],
    ).read_bytes()
    rng = random.Random(20260926)
    path = tmp_path / "mutated.epub"

    for i in range(200):
        data = bytearray(base)
        if i % 2:
            data = data[: rng.randrange(len(data))]
        else:
            for _ in range(rng.randint(1, 8)):
                data[rng.randrange(len(data))] = rng.randrange(256)
        path.write_bytes(bytes(data))

        result = _extract(path)

        assert isinstance(result, ExtractionResult)
        pages = {c.page for c in result.chunks}
        assert pages <= set(range(1, (result.page_count or 0) + 1))
