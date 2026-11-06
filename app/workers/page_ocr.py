"""Scanned PDF page OCR (SPEC-ADDON-069).

A page whose text layer is empty after stripping whitespace is rendered at
300 dpi (within the 50-megapixel picture cap) and read with the shared
picture-OCR engine. The text is stored per page and a forced text-content
task makes it the page's searchable text.
"""

from __future__ import annotations

import logging
import math
import os
import stat
import time
from typing import Any, Callable

from app.database import get_search_db, get_search_db_read
from app.extractors.pdf import MAX_PAGES, MAX_PDF_SIZE, PYMUPDF_LOCK
from app.models import IndexedFile, PdfPageOcrText
from app.workers.ocr import _get_engine, run_engine

logger = logging.getLogger(__name__)

MAX_PIXELS = 50_000_000
TARGET_DPI = 300


class _Unreachable(Exception):
    """The file cannot be read at all; leave it pending."""


def _reachable(path: str) -> bool:
    try:
        info = os.stat(path)
    except (OSError, ValueError):
        return False
    return stat.S_ISREG(info.st_mode)


def _render(page: Any) -> Any:
    """Render one page in RGB at 300 dpi, or the highest dpi under 50 MP."""
    import fitz

    dpi = TARGET_DPI
    points = float(page.rect.width) * float(page.rect.height)
    if points > 0:
        at_300 = (points / 72.0 / 72.0) * 300.0 * 300.0
        if at_300 > MAX_PIXELS:
            dpi = max(1, int(72.0 * math.sqrt(MAX_PIXELS / points)))
    pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
    from PIL import Image

    return Image.frombytes(
        "RGB", (pixmap.width, pixmap.height), pixmap.samples,
    )


def _store_pages(file_id: str, pages: list[tuple[int, str]]) -> bool | None:
    """A5/B5: replace the stored pages. Returns whether a text task is due.

    ``None`` means the file is no longer an active row; nothing was written.
    """
    try:
        with get_search_db() as session:
            row = session.query(IndexedFile).filter_by(
                file_id=file_id, active=True
            ).first()
            if row is None:
                return None
            earlier = session.query(PdfPageOcrText).filter_by(
                file_id=file_id
            ).all()
            had_text = any((page.text or "").strip() for page in earlier)
            session.query(PdfPageOcrText).filter_by(file_id=file_id).delete()
            for number, text in pages:
                session.add(PdfPageOcrText(
                    file_id=file_id, page=number, text=text or "",
                ))
            row.page_ocr_indexed = True
            stored_text = any((text or "").strip() for _, text in pages)
            needs_text = stored_text or had_text
            if needs_text:
                row.text_indexed = False
            return needs_text
    except Exception as exc:
        logger.error("Page OCR write failed for %s: %s", file_id, exc)
        return False


def index_page_ocr(
    file_id: str, *, is_paused: Callable[[], bool] | None = None,
) -> bool:
    """Read one PDF's scanned pages.

    Returns ``True`` when a ``TEXT_CONTENT`` task must be queued for the
    file after the write, ``False`` otherwise (including every case where
    the file stays pending).
    """
    with get_search_db_read() as session:
        file = session.query(IndexedFile).filter_by(
            file_id=file_id, active=True
        ).first()
        if file is None:
            return False
        path = file.file_path
        already = bool(file.page_ocr_indexed)
    if already:
        return False

    if not _reachable(path):
        logger.warning("Page OCR skipped for %s: file is unreachable", file_id)
        return False

    try:
        file_size = os.path.getsize(path)
    except OSError:
        logger.warning("Page OCR skipped for %s: file is unreachable", file_id)
        return False
    if file_size > MAX_PDF_SIZE:
        logger.warning(
            "Page OCR for %s: file is over %d MB", file_id,
            MAX_PDF_SIZE // (1024 * 1024),
        )
        return bool(_store_pages(file_id, []))

    import fitz

    try:
        with PYMUPDF_LOCK:
            document = fitz.open(path)
    except Exception as exc:
        logger.warning(
            "Page OCR for %s could not open the PDF (%s)",
            file_id, type(exc).__name__,
        )
        if not _reachable(path):
            logger.warning(
                "Page OCR skipped for %s: file became unreachable", file_id,
            )
            return False
        return bool(_store_pages(file_id, []))

    engine = _get_engine()
    if engine is None:
        document.close()
        return False

    stores: list[tuple[int, str]] = []
    engine_errors = 0
    try:
        page_count = min(getattr(document, "page_count", 0), MAX_PAGES)
        for index in range(page_count):
            while is_paused is not None and is_paused():
                time.sleep(0.2)
            number = index + 1
            try:
                with PYMUPDF_LOCK:
                    page = document[index]
                    if page.get_text("text").strip():
                        continue
                    image = _render(page)
            except Exception as exc:
                logger.warning(
                    "Page OCR for %s: page %d could not be rendered (%s)",
                    file_id, number, type(exc).__name__,
                )
                if not _reachable(path):
                    logger.warning(
                        "Page OCR skipped for %s: file became unreachable "
                        "while rendering page %d; nothing written",
                        file_id, number,
                    )
                    return False
                stores.append((number, ""))
                continue

            current = _get_engine()
            if current is None:
                logger.warning(
                    "Page OCR lost the engine partway for %s; nothing written",
                    file_id,
                )
                return False
            try:
                answer = run_engine(current, image)
                lines = getattr(answer, "txts", None) or ()
                text = "\n".join(lines)
            except Exception as exc:
                engine_errors += 1
                logger.warning(
                    "Page OCR for %s: the engine failed on page %d (%s)",
                    file_id, number, type(exc).__name__,
                )
                text = ""
            stores.append((number, text))
    finally:
        try:
            document.close()
        except Exception:
            pass

    if engine_errors:
        logger.warning(
            "Page OCR for %s: the engine failed on %d page(s)",
            file_id, engine_errors,
        )

    stored = _store_pages(file_id, stores)
    if stored is None:
        return False
    return stored
