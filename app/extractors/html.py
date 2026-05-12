"""HTML / XHTML content extractor.

Strips ``<script>`` / ``<style>`` / ``<noscript>`` so AI-artifact code
doesn't pollute embeddings (hako ``9QglGpttXKGHbgRhVFLqs``), converts
the remainder to Markdown via ``html2text``, then splits into chunks
tagged with the nearest ``section: <heading>`` metadata (PDF=page,
docx=section, xlsx=sheet, pptx=slide, html=section — hako
``j3gabKrsO1TrD17nnStwT``).

Phase 1 returns chunks only; ``ExtractionResult.markdown`` is always
``None``. Promotion to a ``html_markdown`` table for detailed-summary
input is Phase 2 (spec ``2026-05-12-html-indexing.md`` §9).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import app.config as config
from app.extractors.base import ContentExtractor, ExtractionResult, TextChunk

logger = logging.getLogger(__name__)

EXTRACTOR_NAME = "html"
MAX_HTML_BYTES = 25 * 1024 * 1024  # 25MB — covers AI artifacts + plotly/D3 reports

_STRIP_TAGS = ("script", "style", "noscript")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class HtmlExtractor(ContentExtractor):
    """Extracts text content from HTML / XHTML files."""

    supported_extensions: list[str] = [".html", ".htm", ".xhtml"]

    def extract(self, file_path: str) -> ExtractionResult:
        empty = ExtractionResult(chunks=[], markdown=None, extractor=EXTRACTOR_NAME)

        if not config.validate_file_path(file_path):
            return empty
        path = Path(file_path)
        if not path.exists():
            return empty

        size = path.stat().st_size
        if size > MAX_HTML_BYTES:
            logger.warning(
                "Skipping HTML (size %d > %d): %s", size, MAX_HTML_BYTES, file_path,
            )
            return empty

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("Failed to read HTML %s: %s", file_path, e)
            return empty

        if not raw.strip():
            return empty

        try:
            markdown = _html_to_markdown(raw)
        except Exception as e:
            logger.warning("HTML parse failed for %s: %s", file_path, e)
            return ExtractionResult(chunks=[], markdown=None, extractor="html_error")

        if not markdown.strip():
            return empty

        chunk_config = config.settings.indexing.text_chunking
        chunks = _split_with_sections(
            markdown,
            max_size=chunk_config.max_chunk_size,
            overlap=chunk_config.overlap,
        )
        return ExtractionResult(chunks=chunks, markdown=None, extractor=EXTRACTOR_NAME)


def _html_to_markdown(html: str) -> str:
    """Strip script/style/noscript and convert HTML body to Markdown."""
    from bs4 import BeautifulSoup
    import html2text

    parser = _pick_parser()
    soup = BeautifulSoup(html, parser)

    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    converter = html2text.HTML2Text()
    converter.body_width = 0  # Don't wrap; chunker handles size
    converter.ignore_images = True  # Image alts add little signal
    converter.ignore_links = False  # Anchor text retains topical meaning
    return converter.handle(str(soup))


def _pick_parser() -> str:
    """Prefer lxml when available; fall back to the stdlib parser."""
    try:
        import lxml  # noqa: F401
        return "lxml"
    except ImportError:
        return "html.parser"


def _split_with_sections(
    markdown: str,
    max_size: int,
    overlap: int,
) -> list[TextChunk]:
    """Split Markdown into chunks, tagging each with its nearest heading.

    Walks the document by heading boundaries: for every ``# Heading``
    block, the body until the next heading is chunked via
    ``ContentExtractor.chunk_text`` and each piece is tagged
    ``section: <heading>``. Pre-heading content gets an empty metadata.
    """
    sections = _split_by_headings(markdown)
    chunks: list[TextChunk] = []
    for heading, body in sections:
        body = body.strip()
        if not body:
            continue
        metadata = f"section: {heading}" if heading else ""
        for piece in ContentExtractor.chunk_text(
            body, max_size=max_size, overlap=overlap,
        ):
            chunks = [*chunks, TextChunk(text=piece, page=None, metadata=metadata)]
    return chunks


def _split_by_headings(markdown: str) -> list[tuple[str, str]]:
    """Yield ``(heading, body)`` pairs. The first entry's heading is ``""``."""
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        sections.append((current_heading, "\n".join(current_lines)))
        current_lines = []

    for line in markdown.splitlines():
        match = _HEADING_RE.match(line)
        if match is not None:
            flush()
            current_heading = match.group(2).strip()
            continue
        current_lines.append(line)

    flush()
    return sections
