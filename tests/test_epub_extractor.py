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


def test_validate_file_path_false_returns_empty(tmp_path, monkeypatch) -> None:
    path = make_epub(tmp_path / "b.epub", [Item("c1", "c1.xhtml", _section("MARK-one"))])
    monkeypatch.setattr(config, "validate_file_path", lambda _path: False)

    result = _extract(path)

    assert result.chunks == []
    assert result.section_titles == ()
    assert result.page_count is None


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


@pytest.mark.parametrize(
    ("opf_dir", "href", "member"),
    [
        ("item/standard", "text/c1.xhtml", "item/standard/text/c1.xhtml"),
        ("OEBPS", "chapter%201.xhtml", "OEBPS/chapter 1.xhtml"),
    ],
)
def test_hrefs_resolve_against_the_opf_directory(tmp_path, opf_dir, href, member) -> None:
    path = make_epub(
        tmp_path / "b.epub",
        [
            Item("nav", "nav.xhtml", nav_doc([(href, "Title")]), properties="nav"),
            Item("c1", href, None),
        ],
        spine=["c1"],
        opf_dir=opf_dir,
        extra={member: _section("MARK-one")},
    )

    result = _extract(path)

    assert _marks(result) == {(1, "MARK-one")}
    assert result.section_titles == ((1, "Title"),)


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------


def _heading(level: int, text: str, mark: str = "") -> str:
    return xhtml(f"<h{level}>{text}</h{level}><p>body {mark}</p>")


def _ncx_item(entries: list[tuple[str, str]]) -> Item:
    return Item("ncx", "toc.ncx", ncx_doc(entries), media_type="application/x-dtbncx+xml")


def _nav_item(nav: str) -> Item:
    return Item("nav", "nav.xhtml", nav, properties="nav")


_LONG_LABEL = "  Long \n\t title  " + "x" * 300
_CAPPED = "Long title " + "x" * 189


@pytest.mark.parametrize(
    ("items", "spine", "spine_toc", "expected"),
    [
        pytest.param(
            [
                Item("nav", "nav/toc.xhtml", nav_doc(
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
                ), properties="nav"),
                Item("c1", "text/c1.xhtml", _section("MARK-one")),
                Item("c2", "text/c2.xhtml", _section("MARK-two")),
            ],
            ["c1", "c2"], None,
            ((1, "Chapter One"), (2, "Chapter Two")),
            id="first-toc-entry-per-file-wins",
        ),
        pytest.param(
            [
                _nav_item(nav_doc([("c2.xhtml", "Two"), ("c1.xhtml", "One")])),
                Item("c1", "c1.xhtml", _section("MARK-one")),
                Item("c2", "c2.xhtml", _section("MARK-two")),
                Item("c1again", "c1.xhtml", None),
            ],
            ["c1", "c2", "c1again"], None,
            ((1, "One"), (2, "Two"), (3, "Two")),
            id="repeated-file-maps-to-first-occurrence",
        ),
        *(
            pytest.param(
                [
                    _ncx_item([("c1.xhtml", "One"), ("c2.xhtml#x", "Two")]),
                    Item("c1", "c1.xhtml", _section("MARK-one")),
                    Item("c2", "c2.xhtml", _section("MARK-two")),
                ],
                ["c1", "c2"], spine_toc,
                ((1, "One"), (2, "Two")),
                id=f"ncx-without-nav-found-by-{how}",
            )
            for spine_toc, how in (("ncx", "spine-toc"), (None, "media-type"))
        ),
        pytest.param(
            [
                _nav_item(nav_doc([("c1.xhtml", "From nav")])),
                _ncx_item([("c1.xhtml", "From NCX")]),
                Item("c1", "c1.xhtml", _section("MARK-one")),
            ],
            ["c1"], "ncx",
            ((1, "From nav"),),
            id="nav-preferred-over-ncx",
        ),
        pytest.param(
            [
                _nav_item(xhtml(
                    '<nav epub:type="landmarks"><ol><li><a href="c1.xhtml">L</a></li></ol></nav>'
                )),
                _ncx_item([("c1.xhtml", "From NCX")]),
                Item("c1", "c1.xhtml", _section("MARK-one")),
            ],
            ["c1"], "ncx",
            ((1, "From NCX"),),
            id="ncx-when-nav-has-no-toc",
        ),
        pytest.param(
            [
                _nav_item(nav_doc([("c1.xhtml", "One"), ("c3.xhtml", "Three")])),
                *(Item(f"c{i}", f"c{i}.xhtml", _heading(1, f"Heading {i}")) for i in range(1, 5)),
            ],
            ["c1", "c2", "c3", "c4"], None,
            ((1, "One"), (2, "One"), (3, "Three"), (4, "Three")),
            id="untitled-section-inherits-preceding-toc-title",
        ),
        pytest.param(
            [
                _nav_item(nav_doc([("c3.xhtml", "One")])),
                Item("c1", "c1.xhtml", _section("MARK-cover")),
                Item("c2", "c2.xhtml", xhtml("<p>intro</p><h3>Title Page</h3><h1>Later</h1>")),
                Item("c3", "c3.xhtml", _heading(1, "Chapter heading")),
                Item("c4", "c4.xhtml", _heading(1, "Ignored heading")),
            ],
            ["c1", "c2", "c3", "c4"], None,
            ((2, "Title Page"), (3, "One"), (4, "One")),
            id="before-first-toc-entry-first-heading-else-none",
        ),
        pytest.param(
            [
                _nav_item(nav_doc([("c1.xhtml", _LONG_LABEL), ("c2.xhtml", " \n ")])),
                Item("c1", "c1.xhtml", _section("MARK-one")),
                Item("c2", "c2.xhtml", _section("MARK-two")),
            ],
            ["c1", "c2"], None,
            ((1, _CAPPED), (2, _CAPPED)),
            id="whitespace-collapsed-capped-blank-label-ignored",
        ),
    ],
)
def test_section_titles(tmp_path, items, spine, spine_toc, expected) -> None:
    path = make_epub(tmp_path / "b.epub", items, spine=spine, spine_toc=spine_toc)

    assert _extract(path).section_titles == expected


_RUBY = "<ruby>漢字<rp>(</rp><rt>かんじ</rt><rp>)</rp></ruby>"


@pytest.mark.parametrize(
    "toc",
    [
        Item("nav", "nav.xhtml", nav_doc([("c2.xhtml", f"{_RUBY}の章")]), properties="nav"),
        Item("ncx", "toc.ncx", ncx_doc([("c2.xhtml", "@RUBY@の章")]).replace("@RUBY@", _RUBY),
             media_type="application/x-dtbncx+xml"),
    ],
    ids=["nav", "ncx"],
)
def test_title_drops_ruby_readings(tmp_path, toc) -> None:
    path = make_epub(
        tmp_path / "b.epub",
        [
            toc,
            Item("c1", "c1.xhtml", _heading(2, f"前書き{_RUBY}")),
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
        epub_module.BOOK_MAX_BYTES,
    ) == (1024 * 1024, 5 * 1024 * 1024, 2_000_000, 2_000, 200, 20 * 1024 * 1024)


def _expanding_ncx_entries(data: bytes, base_dir: str) -> list[tuple[str | None, str | None]]:
    # Stands in for an NCX parser that expands internal entities, so the
    # test does not depend on bs4 happening to drop an internal subset.
    from lxml import etree

    root = etree.fromstring(data, etree.XMLParser(resolve_entities=True, no_network=True))
    return [
        (
            epub_module._resolve(base_dir, point.xpath("string(*[local-name()='content']/@src)")),
            point.xpath("normalize-space(*[local-name()='navLabel'])") or None,
        )
        for point in root.xpath("//*[local-name()='navPoint']")
    ]


_ENTITY_NAV = (
    '<?xml version="1.0"?><!DOCTYPE html [<!ENTITY t "Expanded">]>'
    '<html xmlns="http://www.w3.org/1999/xhtml" '
    'xmlns:epub="http://www.idpf.org/2007/ops"><body>'
    '<nav epub:type="toc"><ol><li><a href="c1.xhtml">&t;</a></li></ol></nav>'
    "</body></html>"
)
_ENTITY_NCX = (
    '<?xml version="1.0"?><!DOCTYPE ncx [<!ENTITY t "Expanded">]>'
    '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap>'
    '<navPoint id="p1"><navLabel><text>&t;</text></navLabel>'
    '<content src="c1.xhtml"/></navPoint></navMap></ncx>'
)


@pytest.mark.parametrize(
    ("toc_items", "expected"),
    [
        (
            [
                Item("nav", "nav.xhtml", _ENTITY_NAV, properties="nav"),
                Item("ncx", "toc.ncx", ncx_doc([("c1.xhtml", "From NCX")]),
                     media_type="application/x-dtbncx+xml"),
            ],
            ((1, "From NCX"),),
        ),
        (
            [Item("ncx", "toc.ncx", _ENTITY_NCX, media_type="application/x-dtbncx+xml")],
            ((1, "Heading"),),
        ),
    ],
)
def test_toc_declaring_an_entity_gives_no_title(
    tmp_path, monkeypatch, toc_items, expected,
) -> None:
    monkeypatch.setattr(epub_module, "_ncx_entries", _expanding_ncx_entries)
    path = make_epub(
        tmp_path / "b.epub",
        [*toc_items, Item("c1", "c1.xhtml", xhtml("<h1>Heading</h1><p>MARK-one</p>"))],
        spine=["c1"],
    )

    assert _extract(path).section_titles == expected


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


# Large next to the container, package and nav, which the byte budget also pays for.
_REPEATED_MEMBER = xhtml("<h1>Head</h1><p>MARK-body</p>" + "<div><span></span></div>" * 500)
_MEMBER_BYTES = len(_REPEATED_MEMBER.encode())


def _break_crc(path: Path, name: str) -> None:
    data = bytearray(path.read_bytes())
    wanted = name.encode()
    offset = data.find(b"PK\x01\x02")
    while offset != -1:
        length = int.from_bytes(data[offset + 28:offset + 30], "little")
        if bytes(data[offset + 46:offset + 46 + length]) == wanted:
            data[offset + 16] ^= 0xFF
        offset = data.find(b"PK\x01\x02", offset + 4)
    path.write_bytes(bytes(data))


@pytest.mark.parametrize(
    ("caps", "with_toc", "bad_crc", "expected_reads", "expected_pages"),
    [
        ({"BOOK_MAX_BYTES": 3 * _MEMBER_BYTES}, True, False, 3, {1, 2, 3}),
        ({"TEXT_MAX_CHARS": 5}, True, False, 1, {1}),
        (
            {"SECTION_MAX_BYTES": _MEMBER_BYTES - 1, "BOOK_MAX_BYTES": 2 * _MEMBER_BYTES},
            True, False, 2, set(),
        ),
        ({"TEXT_MAX_CHARS": 5, "BOOK_MAX_BYTES": 3 * _MEMBER_BYTES}, False, False, 3, {1}),
        (
            {"SECTION_MAX_BYTES": _MEMBER_BYTES, "BOOK_MAX_BYTES": 3 * _MEMBER_BYTES},
            True, True, 3, set(),
        ),
    ],
)
def test_book_budget_stops_further_reads(
    tmp_path, monkeypatch, caps, with_toc, bad_crc, expected_reads, expected_pages,
) -> None:
    import zipfile

    for name, value in caps.items():
        monkeypatch.setattr(epub_module, name, value)
    nav = nav_doc([("s.xhtml", "Only title")])
    toc = [Item("nav", "nav.xhtml", nav, properties="nav")] if with_toc else []
    path = make_epub(
        tmp_path / "b.epub",
        [
            *toc,
            Item("s0", "s.xhtml", _REPEATED_MEMBER),
            *(Item(f"s{i}", "s.xhtml", None) for i in range(1, 10)),
        ],
        spine=[f"s{i}" for i in range(10)],
    )
    if bad_crc:
        _break_crc(path, "OEBPS/s.xhtml")
    opened: list[str] = []
    original_open = zipfile.ZipFile.open

    def spy_open(self, name, *args, **kwargs):
        opened.append(name if isinstance(name, str) else name.filename)
        return original_open(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", spy_open)

    result = _extract(path)

    assert opened.count("OEBPS/s.xhtml") == expected_reads
    assert {c.page for c in result.chunks} == expected_pages


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


@pytest.mark.parametrize(
    ("encryption", "caps", "expected_marks", "expected_titles"),
    [
        pytest.param(
            _encryption_xml("OEBPS/c1.xhtml", "OEBPS/font.otf"), {},
            {(2, "MARK-two")}, ((1, "One"), (2, "Two")),
            id="listed-section-skipped",
        ),
        pytest.param(
            _encryption_xml("OEBPS/nav.xhtml"), {},
            {(1, "MARK-one"), (2, "MARK-two")}, (),
            id="listed-nav-not-read",
        ),
        pytest.param(
            "<encryption><unclosed", {}, set(), (),
            id="unparseable-skips-everything",
        ),
        pytest.param(
            _encryption_xml(*(f"x{i}.otf" for i in range(100))), {"XML_MAX_BYTES": 3000},
            set(), (),
            id="over-cap-skips-everything",
        ),
    ],
)
def test_encryption_xml_decides_what_is_read(
    tmp_path, monkeypatch, encryption, caps, expected_marks, expected_titles,
) -> None:
    for name, value in caps.items():
        monkeypatch.setattr(epub_module, name, value)
    nav = nav_doc([("c1.xhtml", "One"), ("c2.xhtml", "Two")])
    path = make_epub(
        tmp_path / "b.epub",
        [Item("nav", "nav.xhtml", nav, properties="nav"), *_two_sections(_section("MARK-one"))],
        spine=["c1", "c2"],
        extra={"META-INF/encryption.xml": encryption},
    )

    result = _extract(path)

    assert _marks(result) == expected_marks
    assert result.section_titles == expected_titles
    assert result.page_count == 2


def _corrupt_zip(tmp_path: Path) -> Path:
    path = tmp_path / "b.epub"
    path.write_bytes(b"PK\x03\x04 this is not a zip archive" * 10)
    return path


def _truncated_zip(tmp_path: Path) -> Path:
    path = make_epub(tmp_path / "b.epub", _two_sections(_section("MARK-one")))
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    return path


def _no_container(tmp_path: Path) -> Path:
    import zipfile

    path = tmp_path / "b.epub"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("OEBPS/content.opf", "<package/>")
        zf.writestr("OEBPS/c1.xhtml", _section("MARK-one"))
    return path


def _with_container(container: str):
    def build(tmp_path: Path) -> Path:
        return make_epub(
            tmp_path / "b.epub", _two_sections(_section("MARK-one")), container=container,
        )
    return build


def _opf_with_doctype(tmp_path: Path) -> Path:
    items = _two_sections(_section("MARK-one"))
    opf = opf_xml(items, ["c1", "c2"]).replace(
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE package>',
    )
    return make_epub(tmp_path / "b.epub", items, opf=opf)


_LAUGHS_CONTAINER = (
    f'<?xml version="1.0"?>{_LAUGHS}'
    '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
    '<rootfiles><rootfile full-path="OEBPS/content.opf" '
    'media-type="application/oebps-package+xml"/></rootfiles>'
    "<x>&lol3;</x></container>"
)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(_corrupt_zip, id="corrupt-zip"),
        pytest.param(_truncated_zip, id="truncated-zip"),
        pytest.param(_no_container, id="no-container"),
        pytest.param(_with_container(container_xml("OEBPS/elsewhere.opf")), id="opf-missing"),
        pytest.param(_with_container(container_xml("../content.opf")), id="opf-outside-book"),
        pytest.param(
            _with_container(container_xml("OEBPS/content.opf").replace(
                "application/oebps-package+xml", "application/xml",
            )),
            id="no-package-rootfile",
        ),
        pytest.param(_with_container(_LAUGHS_CONTAINER), id="billion-laughs-container"),
        pytest.param(_opf_with_doctype, id="opf-with-doctype"),
    ],
)
def test_unopenable_book_returns_empty(tmp_path, build) -> None:
    assert _empty(_extract(build(tmp_path)))


@pytest.mark.parametrize(
    ("first", "caps", "encrypted", "expected"),
    [
        pytest.param(
            '<?xml version="1.0"?><!DOCTYPE html [<!ENTITY x "MARK-expanded">]>'
            '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>&x; MARK-one</p>'
            "</body></html>",
            {}, (), {(2, "MARK-two")},
            id="entity-declaration-refused",
        ),
        pytest.param(
            "<<not xml>> <p>MARK-one</p>", {}, (), {(2, "MARK-two")},
            id="malformed-before-root",
        ),
        pytest.param(
            _section("MARK-one"), {}, ("OEBPS/c1.xhtml",), {(2, "MARK-two")},
            id="zip-encrypted-member",
        ),
        pytest.param(
            xhtml("<p>MARK-one</p>" + "<p>x</p>" * 300), {"SECTION_MAX_BYTES": 2000}, (),
            {(2, "MARK-two")},
            id="member-over-cap",
        ),
        pytest.param(
            '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
            "<p>MARK-one&nbsp;and a stray & here</p><p>unclosed</body></html>",
            {}, (), {(1, "MARK-one"), (2, "MARK-two")},
            id="malformed-after-root-still-indexed",
        ),
    ],
)
def test_failing_section_loses_only_itself(
    tmp_path, monkeypatch, first, caps, encrypted, expected,
) -> None:
    for name, value in caps.items():
        monkeypatch.setattr(epub_module, name, value)
    path = make_epub(tmp_path / "b.epub", _two_sections(first), encrypted_members=encrypted)

    result = _extract(path)

    assert _marks(result) == expected
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
