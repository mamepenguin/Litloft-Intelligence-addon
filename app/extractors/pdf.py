"""PDF content extractor.

Primary path: PyMuPDF4LLM converts each page to Markdown so that headings,
lists, and tables survive into the embedding chunks. Fallback: PyMuPDF
(fitz) raw text extraction when PyMuPDF4LLM is unavailable or raises.
"""

import logging
import re
import threading
import unicodedata
from pathlib import Path
from typing import NamedTuple

from app.config import settings
from app.extractors.base import ContentExtractor, ExtractionResult, TextChunk

logger = logging.getLogger(__name__)

# Maximum file size to process (200MB).
# OCR-scanner PDFs commonly exceed 50MB because of embedded image
# layers; 200MB covers typical scanned books/manuals while still
# rejecting pathologically huge files.
MAX_PDF_SIZE = 200 * 1024 * 1024

# Maximum pages to process
MAX_PAGES = 500

# Joiner placed between per-page Markdown blocks so consumers can split
# the full document back into pages if needed.
PAGE_SEPARATOR = "\n\n---\n\n"

# (left, top, right, bottom) in points. Passed to PyMuPDF4LLM explicitly so
# the band the page-edge recovery reads is the band the conversion cut,
# whatever the library's default becomes.
PAGE_MARGINS = (0, 50, 0, 50)

# Bumped when the rules below change: the startup check re-extracts every
# PDF once for a new value.
PDF_TEXT_RULES_VERSION = "1"

# Characters the containment check ignores besides whitespace, because the
# library writes Markdown and the recovered line is plain text.
_MARKDOWN_MARKUP = frozenset("*_`#|>-")
_DIGITS = re.compile(r"\d+")


class PageEdges(NamedTuple):
    """Band lines a page recovered, before the containment check."""

    top: tuple[str, ...]
    bottom: tuple[str, ...]


class _BandLine(NamedTuple):
    text: str
    top: bool
    box: tuple[float, float]

# PyMuPDF does not support use from several threads at once: a conversion that
# overlaps another fails over to plain text or returns wrong text. Every
# PyMuPDF call in the addon holds this lock.
PYMUPDF_LOCK = threading.Lock()

EXTRACTOR_PRIMARY = "pymupdf4llm"
EXTRACTOR_FALLBACK = "fitz_fallback"


class PdfExtractor(ContentExtractor):
    """Extracts text content from PDF files.

    Uses PyMuPDF4LLM for Markdown-aware extraction with a PyMuPDF (fitz)
    raw-text fallback. The Markdown rendering is returned alongside the
    embedding chunks via ``ExtractionResult``.
    """

    supported_extensions: list[str] = [".pdf"]

    def extract(self, file_path: str) -> ExtractionResult:
        """Extract chunks (and optional Markdown) from a PDF file.

        Args:
            file_path: Absolute path to the PDF file.

        Returns:
            ExtractionResult. ``markdown`` is non-None only on the
            primary PyMuPDF4LLM path; ``extractor`` records which path
            produced the result.
        """
        path = _validate_pdf_path(file_path)
        if path is None:
            return _empty_fallback()

        with PYMUPDF_LOCK:
            primary = _extract_with_pymupdf4llm(path)
            if primary is not None:
                return primary

            return _extract_with_fitz_fallback(path)


def _validate_pdf_path(file_path: str) -> Path | None:
    """Return the Path if the file is safe and within size limits."""
    from app.config import validate_file_path

    if not validate_file_path(file_path):
        return None
    path = Path(file_path)
    if not path.exists():
        return None
    if path.stat().st_size > MAX_PDF_SIZE:
        return None
    return path


def _empty_fallback() -> ExtractionResult:
    """Empty result tagged as fallback (used for size cap / missing file)."""
    return ExtractionResult(
        chunks=[], markdown=None, extractor=EXTRACTOR_FALLBACK
    )


def _extract_with_pymupdf4llm(path: Path) -> ExtractionResult | None:
    """Try the PyMuPDF4LLM Markdown path; return None to signal fallback.

    First tries with ``pages=range(MAX_PAGES)`` so the page cap is
    enforced inside the library, bounding peak memory for hostile /
    pathological PDFs. PyMuPDF4LLM raises ``IndexError`` when the PDF
    has fewer pages than the requested range (most real PDFs), so we
    retry without ``pages=`` to let the library walk every page —
    output is truncated to ``MAX_PAGES`` afterwards anyway.
    """
    try:
        import pymupdf4llm
    except ImportError as exc:
        logger.warning(
            "pymupdf4llm unavailable, falling back to fitz: %s", exc
        )
        return None
    except Exception as exc:
        logger.warning(
            "pymupdf4llm import failed, falling back to fitz: %s", exc
        )
        return None

    try:
        page_chunks = pymupdf4llm.to_markdown(
            str(path),
            pages=list(range(MAX_PAGES)),
            page_chunks=True,
            margins=PAGE_MARGINS,
        )
    except IndexError:
        # PDF has fewer than MAX_PAGES pages — common case. Retry
        # without an explicit page list so the library extracts all.
        try:
            page_chunks = pymupdf4llm.to_markdown(
                str(path), page_chunks=True, margins=PAGE_MARGINS
            )
        except Exception as exc:
            logger.warning(
                "pymupdf4llm.to_markdown retry raised on %s, "
                "falling back to fitz: %s",
                path.name, exc,
            )
            return None
    except Exception as exc:
        logger.warning(
            "pymupdf4llm.to_markdown raised on %s, falling back to fitz: %s",
            path.name,
            exc,
        )
        return None

    if not page_chunks:
        return ExtractionResult(
            chunks=[],
            markdown="",
            extractor=EXTRACTOR_PRIMARY,
            page_count=0,
        )

    limited = list(page_chunks)[:MAX_PAGES]
    edges = _safe_edge_lines(path)
    chunks = _build_chunks_from_pages(limited, edges)
    full_markdown = _join_page_markdown(limited, edges)
    return ExtractionResult(
        chunks=chunks,
        markdown=full_markdown,
        extractor=EXTRACTOR_PRIMARY,
        page_count=len(limited),
    )


def _safe_edge_lines(path: Path) -> dict[int, PageEdges]:
    """Read the page edges, returning nothing rather than failing extraction."""
    import fitz

    try:
        with fitz.open(str(path)) as doc:
            return read_edge_lines(doc, path.name)
    except Exception as exc:
        logger.warning(
            "PDF edge lines unavailable for %s (%s); using the library text",
            path.name,
            type(exc).__name__,
        )
        return {}


def read_edge_lines(doc: object, file_name: str) -> dict[int, PageEdges]:
    """Return the text overlapping each page's top and bottom bands.

    Every line overlapping the top or bottom ``PAGE_MARGINS`` band is read
    from the text layer, then filtered document-wide: lines with no letter
    are dropped (page numbers), and a line whose key repeats on three or
    more pages keeps only its first occurrence (running headers and
    footers). The result is keyed by 1-based page number, up to
    ``MAX_PAGES``.

    A page whose text layer raises is skipped with a warning; a document
    that cannot be walked raises to the caller.
    """
    page_count = min(getattr(doc, "page_count", 0) or 0, MAX_PAGES)
    by_page: dict[int, list[_BandLine]] = {}
    for index in range(page_count):
        lines = _page_band_lines(doc[index], index + 1, file_name)
        if lines:
            by_page[index + 1] = lines
    return _drop_repeated(_drop_letterless(by_page))


def _page_band_lines(page: object, number: int, file_name: str) -> list[_BandLine]:
    """The lines of one page that overlap a margin band, in reading order."""
    try:
        data = page.get_text("dict")
    except Exception as exc:
        logger.warning(
            "PDF edge lines: page %d of %s unreadable (%s)",
            number,
            file_name,
            type(exc).__name__,
        )
        return []

    top_limit = float(PAGE_MARGINS[1])
    bottom_limit = float(page.rect.height) - float(PAGE_MARGINS[3])
    found: list[_BandLine] = []
    for block in data.get("blocks", ()):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", ()):
            box = line.get("bbox")
            if not box or len(box) < 4:
                continue
            text = "".join(
                span.get("text", "") for span in line.get("spans", ())
            ).strip()
            if not text:
                continue
            at_top = box[1] < top_limit
            at_bottom = box[3] > bottom_limit
            if not at_top and not at_bottom:
                continue
            found.append(_BandLine(text, at_top, (box[1], box[0])))
    found.sort(key=lambda line: line.box)
    return found


def _drop_letterless(by_page: dict[int, list[_BandLine]]) -> dict[int, list[_BandLine]]:
    kept: dict[int, list[_BandLine]] = {}
    for number, lines in by_page.items():
        kept_lines = [line for line in lines if _has_letter(line.text)]
        if kept_lines:
            kept[number] = kept_lines
    return kept


def _drop_repeated(by_page: dict[int, list[_BandLine]]) -> dict[int, PageEdges]:
    """Keep only the first occurrence of a key found on three or more pages."""
    pages_by_key: dict[str, set[int]] = {}
    for number, lines in by_page.items():
        for line in lines:
            pages_by_key.setdefault(_line_key(line.text), set()).add(number)
    repeated = {
        key for key, pages in pages_by_key.items() if len(pages) >= 3
    }

    seen: set[str] = set()
    out: dict[int, PageEdges] = {}
    for number in sorted(by_page):
        top: list[str] = []
        bottom: list[str] = []
        for line in by_page[number]:
            key = _line_key(line.text)
            if key in repeated:
                if key in seen:
                    continue
                seen.add(key)
            (top if line.top else bottom).append(line.text)
        if top or bottom:
            out[number] = PageEdges(tuple(top), tuple(bottom))
    return out


def _has_letter(text: str) -> bool:
    return any(unicodedata.category(ch).startswith("L") for ch in text)


def _line_key(text: str) -> str:
    """Identity of a line across pages, ignoring its digits' values."""
    plain = "".join(unicodedata.normalize("NFKC", text).split())
    return _DIGITS.sub("\x00", plain)


def page_text_with_edges(markdown: str, edges: PageEdges | None) -> str:
    """Put a page's recovered band lines around its Markdown.

    Top-band lines go before the Markdown and bottom-band lines after it,
    each separated by a blank line. A line the Markdown already contains is
    dropped. A page with no kept line is returned unchanged.
    """
    if edges is None:
        return markdown
    normalised = _normalise(markdown)
    top = tuple(t for t in edges.top if not _is_contained(t, normalised))
    bottom = tuple(
        t for t in edges.bottom if not _is_contained(t, normalised)
    )
    if not top and not bottom:
        return markdown
    text = markdown
    if top:
        text = "\n".join(top) + "\n\n" + text
    if bottom:
        text = text + "\n\n" + "\n".join(bottom)
    return text


def _normalise(text: str) -> str:
    plain = unicodedata.normalize("NFKC", text)
    return "".join(
        ch for ch in plain if not ch.isspace() and ch not in _MARKDOWN_MARKUP
    )


def _is_contained(line: str, normalised_markdown: str) -> bool:
    needle = _normalise(line)
    return bool(needle) and needle in normalised_markdown


def _page_text(page: object) -> str:
    """Best-effort accessor for the Markdown text on a PyMuPDF4LLM page dict."""
    if isinstance(page, dict):
        value = page.get("text")
        if isinstance(value, str):
            return value
    return ""


def _build_chunks_from_pages(
    pages: list[object], edges: dict[int, PageEdges] | None = None
) -> list[TextChunk]:
    """Split each page's text into TextChunks using the configured size."""
    chunk_config = settings.indexing.text_chunking
    edges = edges or {}
    chunks: list[TextChunk] = []
    for index, page in enumerate(pages):
        page_num = index + 1
        text = page_text_with_edges(
            _page_text(page), edges.get(page_num)
        ).strip()
        if not text:
            continue
        sub_chunks = ContentExtractor.chunk_text(
            text,
            max_size=chunk_config.max_chunk_size,
            overlap=chunk_config.overlap,
        )
        page_chunks = [
            TextChunk(
                text=chunk,
                page=page_num,
                metadata=f"page {page_num}",
            )
            for chunk in sub_chunks
        ]
        chunks = [*chunks, *page_chunks]
    return chunks


def _join_page_markdown(
    pages: list[object], edges: dict[int, PageEdges] | None = None
) -> str:
    """Concatenate per-page Markdown with the page separator."""
    edges = edges or {}
    parts = [
        page_text_with_edges(_page_text(page), edges.get(index + 1))
        for index, page in enumerate(pages)
    ]
    return PAGE_SEPARATOR.join(parts)


def _extract_with_fitz_fallback(path: Path) -> ExtractionResult:
    """PyMuPDF raw-text extraction. Used when PyMuPDF4LLM is unusable."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return _empty_fallback()

    try:
        doc = fitz.open(str(path))
    except Exception:
        return _empty_fallback()

    try:
        chunks = _fitz_chunks(doc)
        page_count = min(getattr(doc, "page_count", 0), MAX_PAGES)
    finally:
        try:
            doc.close()
        except Exception:
            pass

    return ExtractionResult(
        chunks=chunks,
        markdown=None,
        extractor=EXTRACTOR_FALLBACK,
        page_count=page_count,
    )


def _fitz_chunks(doc: object) -> list[TextChunk]:
    """Walk a fitz document and produce TextChunks page-by-page."""
    chunk_config = settings.indexing.text_chunking
    page_count = min(getattr(doc, "page_count", 0), MAX_PAGES)
    chunks: list[TextChunk] = []

    for page_num in range(page_count):
        try:
            page = doc[page_num]
            page_text = page.get_text("text")
        except Exception:
            continue
        if not page_text or not page_text.strip():
            continue

        sub_chunks = ContentExtractor.chunk_text(
            page_text.strip(),
            max_size=chunk_config.max_chunk_size,
            overlap=chunk_config.overlap,
        )
        page_chunks = [
            TextChunk(
                text=chunk,
                page=page_num + 1,
                metadata=f"page {page_num + 1}",
            )
            for chunk in sub_chunks
        ]
        chunks = [*chunks, *page_chunks]

    return chunks
