"""Picture OCR worker (SPEC-ADDON-050).

Reads the text in a picture with RapidOCR's PP-OCRv6 small models and
stores the recognised lines as the picture's ``text_content`` chunks, the
same rows a document's text uses. The engine is created on the first OCR
task, ships its models inside the wheel, and is released by the idle-unload
worker like the other indexing-only models.
"""

from __future__ import annotations

import logging
import os
import stat
import threading
import time
import uuid
from typing import Any

from app.config import settings
from app.database import get_search_db, get_search_db_read, upsert_fts_text_content
from app.extractors.base import ContentExtractor
from app.models import IndexedFile
from app.workers.clip import IMAGE_TYPES
from app.workers.metadata import _remove_embeddings, _store_embedding, embed_passages
from app import extraction

logger = logging.getLogger(__name__)

# The picture set OCR applies to. HEIC/HEIF come through pillow-heif the
# way the rest of the addon reads them.
OCR_IMAGE_TYPES: frozenset[str] = frozenset(IMAGE_TYPES) | {
    "image/heic",
    "image/heif",
}

MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_PIXELS = 50_000_000
ENGINE_THREADS = 2

_lock = threading.Lock()
_engine: object | None = None
_engine_failed = False
_last_used = 0.0


class _Unreachable(Exception):
    """The file could not be read at all; leave it pending for a retry."""


class _Unreadable(Exception):
    """The file was reached but cannot yield text; close it as done."""


def _create_engine(*, threads: int = ENGINE_THREADS) -> object:
    """Build the RapidOCR engine with the PP-OCRv6 small models."""
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

    return RapidOCR(params={
        "Global.log_level": "warning",
        "Det.ocr_version": OCRVersion.PPOCRV6,
        "Det.model_type": ModelType.SMALL,
        "Det.lang_type": "japan",
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Rec.model_type": ModelType.SMALL,
        "Rec.lang_type": LangRec.JAPAN,
        "EngineConfig.onnxruntime.intra_op_num_threads": threads,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })


def _get_engine() -> object | None:
    """The shared engine, created once per process or ``None`` if it failed.

    A creation failure is remembered for the life of the process so the
    log carries one ERROR and the queued pictures are dropped without
    touching their flags; the next process start tries again.
    """
    global _engine, _engine_failed, _last_used

    if _engine is not None:
        _last_used = time.monotonic()
        return _engine
    if _engine_failed:
        return None

    with _lock:
        if _engine is not None:
            _last_used = time.monotonic()
            return _engine
        if _engine_failed:
            return None
        try:
            _engine = _create_engine(threads=ENGINE_THREADS)
        except Exception as exc:
            _engine_failed = True
            logger.error(
                "OCR engine could not be created; OCR tasks are dropped "
                "until the next process start: %s",
                exc,
            )
            return None
        _last_used = time.monotonic()
        return _engine


def unload_engine() -> None:
    """Release the engine so its memory returns to the OS."""
    global _engine
    with _lock:
        _engine = None


def check_idle_unload() -> None:
    """Release the engine after ``memory.ocr_idle_unload`` idle seconds."""
    idle_timeout = settings.memory.ocr_idle_unload
    if idle_timeout <= 0 or _engine is None:
        return
    if time.monotonic() - _last_used > idle_timeout:
        logger.info("Releasing OCR engine (idle timeout)")
        unload_engine()


def _to_rgb(image: Any) -> Any:
    """RGB with any alpha flattened onto white and palettes expanded."""
    from PIL import Image

    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def _load_picture(path: str) -> Any:
    """Decode the first frame, oriented and RGB, without the DB lock.

    Raises ``_Unreachable`` when the path cannot be read and
    ``_Unreadable`` when the file is too large or cannot be decoded.
    """
    from PIL import Image, ImageOps

    try:
        info = os.stat(path)
    except (OSError, ValueError) as exc:
        raise _Unreachable() from exc
    if not stat.S_ISREG(info.st_mode):
        raise _Unreachable() from IsADirectoryError(path)
    if info.st_size > MAX_FILE_BYTES:
        raise _Unreadable() from ValueError("file is over 50 MB")

    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass

    try:
        opened = Image.open(path)
    except (FileNotFoundError, PermissionError, IsADirectoryError) as exc:
        raise _Unreachable() from exc
    except Exception as exc:
        raise _Unreadable() from exc

    try:
        with opened:
            if opened.width * opened.height > MAX_PIXELS:
                raise _Unreadable() from ValueError("picture is over 50 megapixels")
            oriented = ImageOps.exif_transpose(opened)
            oriented.load()
            return _to_rgb(oriented)
    except _Unreadable:
        raise
    except Exception as exc:
        raise _Unreadable() from exc


def _write(
    file_id: str,
    chunks: list[str],
    vectors: list[object],
    text: str | None = None,
) -> bool:
    """Replace this picture's text rows in one short transaction.

    ``text`` is the joined string the engine read; direct callers that
    already chunked it may omit it.
    """
    picture_text = text if text is not None else "\n".join(chunks)
    with get_search_db() as session:
        row = session.query(IndexedFile).filter_by(
            file_id=file_id, active=True
        ).first()
        if row is None:
            return False

        if row.picture_text_corrected:
            if not row.picture_text_redo:
                extraction.queue_item(
                    session, file_id, extraction.KIND_PICTURE, "rebuild"
                )
                return False
            if not chunks:
                # A confirmed read that found nothing leaves the
                # correction in place and ends without a result.
                _mark_confirmed_ocr_failed(session, file_id)
                return False
            extraction.clear_corrections(
                session, file_id, extraction.KIND_PICTURE
            )
            extraction.drop_items(session, file_id, extraction.KIND_PICTURE)
            extraction.bump_generation(
                session, file_id, extraction.KIND_PICTURE
            )
        else:
            extraction.delete_original(
                session, file_id, extraction.KIND_PICTURE
            )
            extraction.bump_generation(
                session, file_id, extraction.KIND_PICTURE
            )

        extraction.set_picture_text(session, file_id, picture_text)
        _remove_embeddings(session, file_id, "text_content")
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            _store_embedding(
                session=session,
                embedding_id=f"ocr_{file_id}_{index}_{uuid.uuid4().hex[:8]}",
                file_id=file_id,
                embedding_type="text_content",
                vector_table="vec_text",
                vector=vector,
                content_preview=chunk[:200],
                page=None,
                chunk_index=index,
            )
        upsert_fts_text_content(
            session,
            file_id,
            [
                {"chunk_index": index, "page": None, "text": chunk}
                for index, chunk in enumerate(chunks)
            ],
        )
        row.ocr_indexed = True
        return True


def _mark_confirmed_ocr_failed(session: Any, file_id: str) -> None:
    extraction.clear_redo(session, file_id, extraction.KIND_PICTURE)
    extraction.mark_redo_ended(session, file_id, extraction.KIND_PICTURE)
    extraction.queue_item(
        session, file_id, extraction.KIND_PICTURE, "rebuild"
    )


def _has_redo(file_id: str) -> bool:
    with get_search_db_read() as session:
        file = session.query(IndexedFile).filter_by(
            file_id=file_id, active=True
        ).first()
        return bool(file is not None and file.picture_text_redo)


def _queue_rebuild(file_id: str) -> None:
    with get_search_db() as session:
        extraction.queue_item(
            session, file_id, extraction.KIND_PICTURE, "rebuild"
        )


def _fail_confirmed(file_id: str) -> None:
    with get_search_db() as session:
        _mark_confirmed_ocr_failed(session, file_id)

def index_ocr(file_id: str) -> bool:
    """Read one picture and store its text as ``text_content`` chunks.

    Returns ``True`` when the picture is closed as done (text stored,
    nothing found, or unreadable) and ``False`` when it stays pending
    (unreachable file, embedding failure, missing engine, or a row that
    left the index while it was read).
    """
    with get_search_db_read() as session:
        file = session.query(IndexedFile).filter_by(
            file_id=file_id, active=True
        ).first()
        if file is None:
            return False
        path = file.file_path
        protected = bool(
            file.picture_text_corrected and not file.picture_text_redo
        )

    if protected:
        _queue_rebuild(file_id)
        logger.info(
            "OCR skipped for %s: the picture text is corrected", file_id
        )
        return False

    engine = _get_engine()
    if engine is None:
        if _has_redo(file_id):
            _fail_confirmed(file_id)
        return False

    try:
        image = _load_picture(path)
    except _Unreachable as exc:
        cause = exc.__cause__ or exc
        logger.warning(
            "OCR skipped for %s: file is unreachable (%s)",
            file_id, type(cause).__name__,
        )
        return False
    except _Unreadable as exc:
        cause = exc.__cause__ or exc
        logger.warning(
            "OCR failed for %s: %s (%s)",
            file_id, type(cause).__name__, cause,
        )
        return _write(file_id, [], [])

    try:
        result = engine(image)
        lines = list(getattr(result, "txts", None) or ())
    except Exception as exc:
        logger.warning(
            "OCR failed for %s: %s", file_id, type(exc).__name__,
        )
        return _write(file_id, [], [])

    joined = "\n".join(lines)
    chunks = ContentExtractor.chunk_text(joined)
    if not chunks:
        return _write(file_id, [], [], text=joined)

    try:
        vectors = list(embed_passages(chunks))
    except Exception as exc:
        logger.error("OCR text embedding failed for %s: %s", file_id, exc)
        if _has_redo(file_id):
            _fail_confirmed(file_id)
        return False

    return _write(file_id, chunks, vectors, text=joined)
