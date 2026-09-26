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
import zipfile
from dataclasses import dataclass
from urllib.parse import unquote
from xml.etree import ElementTree
from xml.parsers import expat

import app.config as config
from app.extractors.base import ContentExtractor, ExtractionResult, TextChunk
from app.extractors.html import html_to_markdown
from app.utils.zip_names import decode_zip_filename

logger = logging.getLogger(__name__)

EXTRACTOR_NAME = "epub"

XML_MAX_BYTES = 1024 * 1024
SECTION_MAX_BYTES = 5 * 1024 * 1024

_OPF_MEDIA_TYPE = "application/oebps-package+xml"
_OPF_NS = "http://www.idpf.org/2007/opf"
_HTML_MEDIA_TYPES = frozenset({"application/xhtml+xml", "text/html"})


@dataclass(frozen=True)
class _Section:
    number: int
    path: str | None
    media_type: str
    is_nav: bool


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
                sections = _read_spine(zf, entries)
                chunks = _extract_sections(zf, entries, sections)
        except Exception as e:
            logger.warning("EPUB could not be opened %s: %s", file_path, e)
            return empty
        return ExtractionResult(
            chunks=chunks,
            markdown=None,
            extractor=EXTRACTOR_NAME,
            page_count=len(sections),
        )


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _namespace(tag: str) -> str | None:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") else None


def _read_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, cap: int) -> bytes | None:
    with zf.open(info) as f:
        data = f.read(cap + 1)
    return None if len(data) > cap else data


def _refuse_declarations(*_args) -> None:
    raise ValueError("DTD declarations are not accepted")


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


def _read_spine(
    zf: zipfile.ZipFile, entries: dict[str, zipfile.ZipInfo],
) -> list[_Section]:
    container_info = entries.get("META-INF/container.xml")
    if container_info is None:
        return []
    container_data = _read_member(zf, container_info, XML_MAX_BYTES)
    if container_data is None:
        return []
    container, _ = _parse_strict(container_data)
    opf_path = _opf_path(container)
    if opf_path is None or opf_path not in entries:
        return []
    opf_data = _read_member(zf, entries[opf_path], XML_MAX_BYTES)
    if opf_data is None:
        return []
    package, root_namespaces = _parse_strict(opf_data)

    # foliate filters by the OPF namespace only when the package declares it.
    namespace = (
        _OPF_NS
        if _OPF_NS in root_namespaces or _namespace(package.tag) == _OPF_NS
        else None
    )
    opf_dir = posixpath.dirname(opf_path)

    manifest_els = _children(package, "manifest", namespace)
    items: dict[str, ElementTree.Element] = {}
    for item in _children(manifest_els[0], "item", namespace) if manifest_els else []:
        item_id = item.get("id")
        if item_id is not None and item_id not in items:
            items = {**items, item_id: item}

    spine_els = _children(package, "spine", namespace)
    itemrefs = _children(spine_els[0], "itemref", namespace) if spine_els else []
    matched = [items[ref.get("idref")] for ref in itemrefs if ref.get("idref") in items]
    return [
        _Section(
            number=number,
            path=_resolve(opf_dir, item.get("href", "")),
            media_type=(item.get("media-type") or "").strip().lower(),
            is_nav="nav" in (item.get("properties") or "").split(),
        )
        for number, item in enumerate(matched, start=1)
    ]


def _extract_sections(
    zf: zipfile.ZipFile,
    entries: dict[str, zipfile.ZipInfo],
    sections: list[_Section],
) -> list[TextChunk]:
    chunk_config = config.settings.indexing.text_chunking
    chunks: list[TextChunk] = []
    for section in sections:
        if section.is_nav or section.media_type not in _HTML_MEDIA_TYPES:
            continue
        info = entries.get(section.path) if section.path else None
        if info is None:
            continue
        try:
            data = _read_member(zf, info, SECTION_MAX_BYTES)
            if data is None:
                continue
            markdown = html_to_markdown(data, ignore_links=True, drop_tags=("rt", "rp"))
            pieces = ContentExtractor.chunk_text(
                markdown,
                max_size=chunk_config.max_chunk_size,
                overlap=chunk_config.overlap,
            )
        except Exception as e:
            logger.warning("EPUB section %d skipped: %s", section.number, e)
            continue
        chunks = [*chunks, *(TextChunk(text=p, page=section.number) for p in pieces)]
    return chunks

