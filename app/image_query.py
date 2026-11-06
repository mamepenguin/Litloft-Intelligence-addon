"""``POST /search/by-image``: rank a drive by a picture, in memory only.

The body is read as a stream and never spooled to a file. One semaphore per
process holds at most one body, one decoded image and one embedding at a time.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import time
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, Response
from starlette.requests import ClientDisconnect

from app import scene_segments
from app.config import settings
from app.document_sections import EPUB_MIME
from app.file_hydrate import hydrate_files
from app.schemas import (
    DocumentHit,
    ImageHit,
    ImageQueryResponse,
    ImageQueryText,
    SceneHit,
    SearchPassage,
)
from app.search import DocumentMatchData, search_documents, search_images
from app.workers import ocr
from app.workers.clip import embed_image
from app.workers.embedder import _get_prefixes, embed_texts

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pass

logger = logging.getLogger(__name__)

IMAGE_TYPES = frozenset({
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/heic",
    "image/heif",
})
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 50_000_000
IMAGE_LIMIT = 24
SCENE_LIMIT = 20
READ_LENGTH = 20
OCR_TIMEOUT_SECONDS = 10.0

_semaphore = asyncio.Semaphore(1)


class _TooManyPixels(Exception):
    """The image declares more pixels than the route accepts."""


class _Undecodable(Exception):
    """Pillow could not decode the bytes."""


@dataclass(frozen=True)
class TextPart:
    status: str
    excerpt: str | None
    read_length: int = 0
    documents: list[DocumentMatchData] = field(default_factory=list)


def _decode(body: bytes):
    from PIL import Image, ImageOps

    try:
        image = Image.open(io.BytesIO(body))
        width, height = image.size
        if width * height > MAX_PIXELS:
            raise _TooManyPixels()
        image.seek(0)
        return ocr._to_rgb(ImageOps.exif_transpose(image))
    except _TooManyPixels:
        raise
    except Exception as exc:
        logger.warning("image query decode failed: %s", type(exc).__name__)
        raise _Undecodable() from exc


def _excerpt(read_text: str) -> str | None:
    collapsed = " ".join(read_text.split())
    if not collapsed:
        return None
    return collapsed[:200] + ("…" if len(collapsed) > 200 else "")


def _visual_part(query_vector, drive: str):
    floor = settings.search.min_score_image_query
    images = search_images(
        query_vector,
        file_type="image",
        drive=drive,
        floor=floor,
        limit=IMAGE_LIMIT,
        raw_score=True,
    )
    frames = scene_segments.count_frames(drive)
    if frames == 0:
        return images.results, (), False
    scenes = scene_segments.find_segments(
        query_vector, "", drive, SCENE_LIMIT, frames, floor=floor,
    )
    return images.results, scenes.segments, scenes.partial


def _text_part(image, drive: str) -> TextPart:
    if settings.features.image_ocr == "false":
        return TextPart("off", None)

    engine = ocr._get_engine()
    if engine is None:
        logger.warning("image query OCR unavailable: no engine")
        return TextPart("failed", None)

    try:
        result = ocr.run_engine(engine, image, OCR_TIMEOUT_SECONDS)
        lines = list(getattr(result, "txts", None) or ())
    except Exception as exc:
        logger.warning("image query OCR failed: %s", type(exc).__name__)
        return TextPart("failed", None)

    read_text = "\n".join(lines)
    read_length = sum(1 for char in read_text if not char.isspace())
    excerpt = _excerpt(read_text)
    if read_length < READ_LENGTH:
        return TextPart("too_short", excerpt, read_length)

    try:
        prefix, _ = _get_prefixes()
        vector = embed_texts([f"{prefix}{read_text[:1000]}"])[0]
        documents = search_documents(vector, drive)
    except Exception as exc:
        logger.warning("image query text search failed: %s", type(exc).__name__)
        return TextPart("failed", None, read_length)
    return TextPart("read", excerpt, read_length, documents)


def _image_hits(results, hydrated: dict) -> list[ImageHit]:
    return [
        ImageHit(
            file_id=result.file_id,
            drive=result.drive,
            filename=result.filename,
            file_type=result.file_type,
            score=round(result.score, 4),
            match_types=["clip_thumbnail"],
            file=hydrated.get(result.file_id),
        )
        for result in results
    ]


def _scene_hits(segments, drive: str, hydrated: dict) -> list[SceneHit]:
    return [
        SceneHit(
            file_id=segment.file_id,
            drive=drive,
            filename=segment.filename,
            start=segment.start,
            end=segment.end,
            peak=segment.peak,
            score=round(segment.score, 4),
            frame_url=(
                f"/api/addons/intelligence/files/{segment.file_id}"
                f"/frame?t={math.floor(segment.peak)}"
            ),
            file=hydrated.get(segment.file_id),
        )
        for segment in segments
    ]


def _document_hits(matches: list[DocumentMatchData], hydrated: dict) -> list[DocumentHit]:
    return [
        DocumentHit(
            file_id=match.file_id,
            drive=match.drive,
            filename=match.filename,
            file_type=match.file_type,
            score=match.score,
            passages=[
                SearchPassage(
                    kind=passage.kind,
                    page=passage.page,
                    section=passage.section,
                    section_title=passage.section_title,
                    chunk_index=passage.chunk_index,
                    text=passage.text,
                    score=passage.score,
                    keyword=False,
                    highlight_terms=[],
                    frame_url=None,
                )
                for passage in match.passages
            ],
            file=hydrated.get(match.file_id),
        )
        for match in matches
    ]


def _content_type(request: Request) -> str:
    raw = request.headers.get("content-type") or ""
    return raw.split(";", 1)[0].strip().lower()


async def _read_body(request: Request) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_BYTES:
            raise HTTPException(
                status_code=413, detail="Image is larger than 20 MB",
            )
        body.extend(chunk)
    return bytes(body)


def _gone() -> Response:
    logger.info("image query skipped: client gone")
    return Response(status_code=204)


async def handle(request: Request, drive: str) -> ImageQueryResponse | Response:
    content_type = _content_type(request)
    if content_type not in IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image type")

    started = time.monotonic()
    async with _semaphore:
        try:
            body = await _read_body(request)
        except ClientDisconnect:
            return _gone()
        if await request.is_disconnected():
            return _gone()

        try:
            image = _decode(body)
        except _TooManyPixels as exc:
            raise HTTPException(
                status_code=413, detail="Image has too many pixels",
            ) from exc
        except _Undecodable as exc:
            raise HTTPException(
                status_code=422, detail="Image could not be decoded",
            ) from exc

        try:
            vector = await asyncio.to_thread(embed_image, image)
        except Exception as exc:
            logger.warning(
                "image query embedding failed: %s", type(exc).__name__,
            )
            raise HTTPException(
                status_code=503, detail="Embedding unavailable",
            ) from exc

        if await request.is_disconnected():
            return _gone()

        visual_task = asyncio.to_thread(_visual_part, vector, drive)
        text_task = asyncio.to_thread(_text_part, image, drive)
        try:
            visual, text = await asyncio.gather(visual_task, text_task)
        except Exception as exc:
            logger.warning(
                "image query visual ranking failed: %s", type(exc).__name__,
            )
            raise HTTPException(
                status_code=500, detail="Search failed",
            ) from exc
        images, scenes, scenes_partial = visual

        ids = list(dict.fromkeys(
            [item.file_id for item in images]
            + [item.file_id for item in scenes]
            + [match.file_id for match in text.documents]
        ))
        hydrated: dict = {}
        if ids:
            try:
                hydrated = await hydrate_files(ids)
            except Exception as exc:
                logger.warning(
                    "image query hydration failed: %s", type(exc).__name__,
                )

    answer = ImageQueryResponse(
        images=_image_hits(images, hydrated),
        scenes=_scene_hits(scenes, drive, hydrated),
        scenes_partial=scenes_partial,
        documents=_document_hits(text.documents, hydrated),
        text=ImageQueryText(status=text.status, excerpt=text.excerpt),
    )
    logger.info(
        "image query: %d bytes, %dx%d px, read %d, images %d, scenes %d, "
        "documents %d, %d ms",
        len(body), image.size[0], image.size[1],
        text.read_length,
        len(answer.images), len(answer.scenes), len(answer.documents),
        int((time.monotonic() - started) * 1000),
    )
    return answer
