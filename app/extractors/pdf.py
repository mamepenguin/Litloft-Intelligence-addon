"""PDF content extractor.

Primary path: PyMuPDF4LLM converts each page to Markdown so that headings,
lists, and tables survive into the embedding chunks. Fallback: PyMuPDF
(fitz) raw text extraction when PyMuPDF4LLM is unavailable or raises.
"""

import logging
from pathlib import Path

from app.config import settings
from app.extractors.base import ContentExtractor, ExtractionResult, TextChunk

logger = logging.getLogger(__name__)

# Maximum file size to process (50MB)
MAX_PDF_SIZE = 50 * 1024 * 1024

# Maximum pages to process
MAX_PAGES = 500

# Joiner placed between per-page Markdown blocks so consumers can split
# the full document back into pages if needed.
PAGE_SEPARATOR = "\n\n---\n\n"

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

    Passes ``pages=range(MAX_PAGES)`` so the page cap is enforced inside
    the library before each page's Markdown is materialised, bounding
    peak memory for hostile / pathological PDFs.
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
        )
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
    chunks = _build_chunks_from_pages(limited)
    full_markdown = _join_page_markdown(limited)
    return ExtractionResult(
        chunks=chunks,
        markdown=full_markdown,
        extractor=EXTRACTOR_PRIMARY,
        page_count=len(limited),
    )


def _page_text(page: object) -> str:
    """Best-effort accessor for the Markdown text on a PyMuPDF4LLM page dict."""
    if isinstance(page, dict):
        value = page.get("text")
        if isinstance(value, str):
            return value
    return ""


def _build_chunks_from_pages(pages: list[object]) -> list[TextChunk]:
    """Split each page's Markdown into TextChunks using the configured size."""
    chunk_config = settings.indexing.text_chunking
    chunks: list[TextChunk] = []
    for index, page in enumerate(pages):
        page_num = index + 1
        text = _page_text(page).strip()
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


def _join_page_markdown(pages: list[object]) -> str:
    """Concatenate per-page Markdown with the page separator."""
    parts = [_page_text(page) for page in pages]
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
