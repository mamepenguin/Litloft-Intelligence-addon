"""EPUB content extractor.

Sections are numbered the way foliate-js builds ``book.sections``: the
``itemref``s that are direct children of the package's first ``spine``,
minus those whose ``idref`` has no manifest item, 1-based. Every chunk
carries its section number in ``page``.
"""

from __future__ import annotations

import io
import logging
import posixpath
import warnings
import zipfile
from dataclasses import dataclass, replace
from urllib.parse import unquote
from xml.etree import ElementTree
from xml.parsers import expat

import app.config as config
from app.extractors.base import ContentExtractor, ExtractionResult, TextChunk
from app.extractors.html import _pick_parser, html_to_markdown
from app.utils.zip_names import decode_zip_filename

logger = logging.getLogger(__name__)

EXTRACTOR_NAME = "epub"

XML_MAX_BYTES = 1024 * 1024
SECTION_MAX_BYTES = 5 * 1024 * 1024
TEXT_MAX_CHARS = 2_000_000
SECTION_MAX = 2_000
BOOK_MAX_BYTES = 20 * 1024 * 1024
TITLE_MAX = 200

_ENCRYPTION_PATH = "META-INF/encryption.xml"
_OPF_MEDIA_TYPE = "application/oebps-package+xml"
_OPF_NS = "http://www.idpf.org/2007/opf"
_NCX_MEDIA_TYPE = "application/x-dtbncx+xml"
_HTML_MEDIA_TYPES = frozenset({"application/xhtml+xml", "text/html"})
_RUBY_ANNOTATION_TAGS = ("rt", "rp")
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")


@dataclass(frozen=True)
class _Section:
    number: int
    path: str | None
    media_type: str
    is_nav: bool

    @property
    def is_html(self) -> bool:
        return self.media_type in _HTML_MEDIA_TYPES


@dataclass(frozen=True)
class _Book:
    sections: tuple[_Section, ...]
    nav_path: str | None
    ncx_path: str | None
    # None when encryption.xml exists but cannot be read: nothing is readable.
    encrypted: frozenset[str] | None = frozenset()

    def readable(self, path: str | None) -> bool:
        return path is not None and self.encrypted is not None and path not in self.encrypted


class _Refused(Exception):
    pass


class _ByteBudget:
    """Decompressed bytes one book may still read, across all of its members."""

    def __init__(self, total: int) -> None:
        self.left = total

    @property
    def spent(self) -> bool:
        return self.left <= 0


class EpubExtractor(ContentExtractor):
    """Extracts text content from EPUB books, one numbered section at a time."""

    supported_extensions: list[str] = [".epub"]

    def extract(self, file_path: str) -> ExtractionResult:
        empty = ExtractionResult(chunks=[], markdown=None, extractor=EXTRACTOR_NAME)
        if not config.validate_file_path(file_path):
            return empty
        try:
            with zipfile.ZipFile(file_path) as zf:
                entries = {decode_zip_filename(info): info for info in zf.infolist()}
                budget = _ByteBudget(BOOK_MAX_BYTES)
                spine = _read_book(zf, entries, budget)
                if spine is None:
                    return empty
                book = replace(
                    spine,
                    sections=spine.sections[:SECTION_MAX],
                    encrypted=_encrypted_paths(zf, entries, budget),
                )
                toc_titles = _toc_titles(zf, entries, book, budget)
                chunks, headings = _extract_sections(zf, entries, book, toc_titles, budget)
        except Exception as e:
            logger.warning("EPUB could not be opened %s: %s", file_path, e)
            return empty
        return ExtractionResult(
            chunks=chunks,
            markdown=None,
            extractor=EXTRACTOR_NAME,
            page_count=len(spine.sections),
            section_titles=_assign_titles(book, toc_titles, headings),
        )


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _namespace(tag: str) -> str | None:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") else None


def _read_member(
    zf: zipfile.ZipFile, info: zipfile.ZipInfo, cap: int, budget: _ByteBudget,
) -> bytes | None:
    with zf.open(info) as f:
        data = f.read(cap + 1)
    budget.left -= len(data)
    return None if len(data) > cap else data


def _refuse_declarations(*_args) -> None:
    raise _Refused("DTD declarations are not accepted")


def _parse_strict(data: bytes) -> tuple[ElementTree.Element, frozenset[str]]:
    """Parse XML that must carry no DOCTYPE at all.

    Returns the root and the namespace URIs declared on the root element.
    """
    probe = expat.ParserCreate()
    probe.StartDoctypeDeclHandler = _refuse_declarations
    probe.EntityDeclHandler = _refuse_declarations
    probe.Parse(data, True)

    root_namespaces: set[str] = set()
    root_started = False
    events = ElementTree.iterparse(io.BytesIO(data), events=("start-ns", "start"))
    for event, value in events:
        if event == "start":
            root_started = True
        elif not root_started:
            root_namespaces.add(value[1])
    return events.root, frozenset(root_namespaces)


def _refuse_entities(data: bytes) -> None:
    """Raise unless ``data`` declares no entity and its root element starts.

    A DOCTYPE is allowed. A well-formedness error after the root has started
    (an undefined ``&nbsp;``, a stray ``&``) leaves nothing to expand, so the
    lenient HTML parse that follows is still safe.
    """
    probe = expat.ParserCreate()
    probe.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    probe.EntityDeclHandler = _refuse_declarations
    root_started: list[bool] = []
    probe.StartElementHandler = lambda *_args: root_started.append(True)
    try:
        probe.Parse(data, True)
    except expat.ExpatError as e:
        if not root_started:
            raise _Refused(f"malformed before the root element: {e}") from e


def _soup(data: bytes, parser: str):
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(data, parser)
    for tag in soup.find_all(_RUBY_ANNOTATION_TAGS):
        tag.decompose()
    return soup


def _clean_title(text: str) -> str | None:
    title = " ".join(text.split())[:TITLE_MAX].rstrip()
    return title or None


def _resolve(base_dir: str, href: str) -> str | None:
    path = unquote(href.split("#", 1)[0])
    if not path or path.startswith("/") or "\\" in path or ":" in path:
        return None
    resolved = posixpath.normpath(posixpath.join(base_dir, path))
    if resolved == ".." or resolved.startswith("../") or resolved.startswith("/"):
        return None
    return resolved


def _opf_path(container: ElementTree.Element) -> str | None:
    for el in container.iter():
        if _local(el.tag) == "rootfile" and el.get("media-type") == _OPF_MEDIA_TYPE:
            return _resolve("", el.get("full-path", ""))
    return None


def _children(
    parent: ElementTree.Element, name: str, namespace: str | None,
) -> list[ElementTree.Element]:
    return [
        el for el in parent
        if isinstance(el.tag, str)
        and _local(el.tag) == name
        and (namespace is None or _namespace(el.tag) == namespace)
    ]


def _read_xml_member(
    zf: zipfile.ZipFile,
    entries: dict[str, zipfile.ZipInfo],
    path: str | None,
    budget: _ByteBudget,
) -> bytes | None:
    info = entries.get(path) if path else None
    return None if info is None else _read_member(zf, info, XML_MAX_BYTES, budget)


def _read_book(
    zf: zipfile.ZipFile, entries: dict[str, zipfile.ZipInfo], budget: _ByteBudget,
) -> _Book | None:
    container_data = _read_xml_member(zf, entries, "META-INF/container.xml", budget)
    if container_data is None:
        return None
    opf_path = _opf_path(_parse_strict(container_data)[0])
    opf_data = _read_xml_member(zf, entries, opf_path, budget)
    if opf_path is None or opf_data is None:
        return None
    package, root_namespaces = _parse_strict(opf_data)

    # foliate filters by the OPF namespace only when the package declares it.
    namespace = (
        _OPF_NS
        if _OPF_NS in root_namespaces or _namespace(package.tag) == _OPF_NS
        else None
    )
    opf_dir = posixpath.dirname(opf_path)

    manifest_els = _children(package, "manifest", namespace)
    manifest = _children(manifest_els[0], "item", namespace) if manifest_els else []
    items: dict[str, ElementTree.Element] = {}
    for item in manifest:
        item_id = item.get("id")
        if item_id is not None and item_id not in items:
            items = {**items, item_id: item}

    spine_els = _children(package, "spine", namespace)
    spine = spine_els[0] if spine_els else None
    itemrefs = _children(spine, "itemref", namespace) if spine is not None else []
    matched = [items[ref.get("idref")] for ref in itemrefs if ref.get("idref") in items]

    nav_item = next(
        (i for i in manifest if "nav" in (i.get("properties") or "").split()), None,
    )
    toc_id = spine.get("toc") if spine is not None else None
    ncx_item = items.get(toc_id) if toc_id else None
    if ncx_item is None:
        ncx_item = next(
            (i for i in manifest if i.get("media-type") == _NCX_MEDIA_TYPE), None,
        )

    def href_of(item: ElementTree.Element | None) -> str | None:
        return None if item is None else _resolve(opf_dir, item.get("href", ""))

    return _Book(
        sections=tuple(
            _Section(
                number=number,
                path=href_of(item),
                media_type=(item.get("media-type") or "").strip().lower(),
                is_nav="nav" in (item.get("properties") or "").split(),
            )
            for number, item in enumerate(matched, start=1)
        ),
        nav_path=href_of(nav_item),
        ncx_path=href_of(ncx_item),
    )


def _encrypted_paths(
    zf: zipfile.ZipFile, entries: dict[str, zipfile.ZipInfo], budget: _ByteBudget,
) -> frozenset[str] | None:
    if _ENCRYPTION_PATH not in entries:
        return frozenset()
    data = _read_xml_member(zf, entries, _ENCRYPTION_PATH, budget)
    if data is None:
        return None
    try:
        root, _ = _parse_strict(data)
    except Exception as e:
        logger.warning("EPUB encryption.xml unreadable, skipping every section: %s", e)
        return None
    return frozenset(
        path
        for el in root.iter()
        if isinstance(el.tag, str) and _local(el.tag) == "CipherReference"
        for path in (_resolve("", el.get("URI", "")),)
        if path is not None
    )


def _nav_entries(data: bytes, base_dir: str) -> list[tuple[str | None, str | None]]:
    soup = _soup(data, _pick_parser())
    toc = next(
        (
            nav for nav in soup.find_all("nav")
            if "toc" in (nav.get("epub:type") or "").split()
        ),
        None,
    )
    if toc is None:
        return []
    return [
        (_resolve(base_dir, a["href"]), _clean_title(a.get_text()))
        for a in toc.find_all("a", href=True)
    ]


def _ncx_entries(data: bytes, base_dir: str) -> list[tuple[str | None, str | None]]:
    soup = _soup(data, "xml")
    entries: list[tuple[str | None, str | None]] = []
    for point in soup.find_all("navPoint"):
        label = point.find("navLabel", recursive=False)
        content = point.find("content", recursive=False)
        if content is None or not content.get("src"):
            continue
        entries = [
            *entries,
            (
                _resolve(base_dir, content["src"]),
                _clean_title(label.get_text()) if label is not None else None,
            ),
        ]
    return entries


def _toc_titles(
    zf: zipfile.ZipFile,
    entries: dict[str, zipfile.ZipInfo],
    book: _Book,
    budget: _ByteBudget,
) -> dict[int, str]:
    toc: list[tuple[str | None, str | None]] = []
    for path, parse in ((book.nav_path, _nav_entries), (book.ncx_path, _ncx_entries)):
        if not book.readable(path):
            continue
        data = _read_xml_member(zf, entries, path, budget)
        if data is None:
            continue
        try:
            _refuse_entities(data)
            toc = [(p, t) for p, t in parse(data, posixpath.dirname(path)) if p and t]
        except Exception as e:
            logger.warning("EPUB table of contents %s skipped: %s", path, e)
            continue
        if toc:
            break

    first_section_of: dict[str, int] = {}
    for section in book.sections:
        if section.path and section.path not in first_section_of:
            first_section_of = {**first_section_of, section.path: section.number}

    titles: dict[int, str] = {}
    for path, title in toc:
        number = first_section_of.get(path)
        if number is not None and number not in titles:
            titles = {**titles, number: title}
    return titles


def _first_heading(data: bytes) -> str | None:
    heading = _soup(data, _pick_parser()).find(_HEADING_TAGS)
    return None if heading is None else _clean_title(heading.get_text())


def _extract_sections(
    zf: zipfile.ZipFile,
    entries: dict[str, zipfile.ZipInfo],
    book: _Book,
    toc_titles: dict[int, str],
    budget: _ByteBudget,
) -> tuple[list[TextChunk], dict[int, str]]:
    chunk_config = config.settings.indexing.text_chunking
    first_toc_section = min(toc_titles, default=None)
    chars_left = TEXT_MAX_CHARS
    chunks: list[TextChunk] = []
    headings: dict[int, str] = {}
    for section in book.sections:
        if budget.spent:
            break
        wants_heading = first_toc_section is None or section.number < first_toc_section
        wants_text = not section.is_nav and chars_left > 0
        info = entries.get(section.path) if book.readable(section.path) else None
        if info is None or not section.is_html or not (wants_heading or wants_text):
            continue
        data = _read_member(zf, info, SECTION_MAX_BYTES, budget)
        if data is None:
            continue
        try:
            _refuse_entities(data)
            heading = _first_heading(data) if wants_heading else None
            if heading:
                headings = {**headings, section.number: heading}
            if not wants_text:
                continue
            markdown = html_to_markdown(
                data, ignore_links=True, drop_tags=_RUBY_ANNOTATION_TAGS,
            ).strip()[:chars_left]
            chars_left -= len(markdown)
            pieces = ContentExtractor.chunk_text(
                markdown,
                max_size=chunk_config.max_chunk_size,
                overlap=chunk_config.overlap,
            )
        except Exception as e:
            logger.warning("EPUB section %d skipped: %s", section.number, e)
            continue
        chunks = [*chunks, *(TextChunk(text=p, page=section.number) for p in pieces)]
    return chunks, headings


def _assign_titles(
    book: _Book, toc_titles: dict[int, str], headings: dict[int, str],
) -> tuple[tuple[int, str], ...]:
    titles: tuple[tuple[int, str], ...] = ()
    current: str | None = None
    for section in book.sections:
        current = toc_titles.get(section.number, current)
        title = current if current is not None else headings.get(section.number)
        if title:
            titles = (*titles, (section.number, title))
    return titles
