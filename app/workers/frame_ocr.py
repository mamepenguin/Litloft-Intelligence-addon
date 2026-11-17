"""Video frame OCR (SPEC-ADDON-066).

Decodes a video's frames, reads the ones whose picture changed enough with
the shared picture-OCR engine, and stores the text as timed chunks. Videos
taller than 1080 lines decode only key frames and read a selected frame
scaled down to 1080 lines (the human's post-approval revision).

A frame's boxes are detected once; a box whose pixels have not changed since
its text was recognised takes that text again, and only the rest are
recognised in a second engine call.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import select
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from app.config import settings
from app.database import (
    delete_fts_text_content,
    get_search_db,
    get_search_db_read,
    upsert_fts_text_content,
)
from app.extractors.base import ContentExtractor
from app.models import IndexedFile, VideoFrameText
from app.video_frames import (
    Unreadable as _Unreadable,
    dhash as _dhash,
    probe as _probe,
    probe_geometry as _probe_geometry,
    reachable as _reachable,
)
from app.workers.metadata import _remove_embeddings, _store_embedding, embed_passages
from app.workers.ocr import _get_engine, detect_boxes, recognise_boxes
from app.workers.whisper import LOFT_MIME

logger = logging.getLogger(__name__)

STALL_SECONDS = 120.0
MAX_OCR_LINES = 1080

#: Difference-hash bits that make a later frame worth reading here.
READ_HASH_BITS = 8
#: Seconds between two read frames of one video here.
READ_GAP_SECONDS = 3.0
#: The ``frame_ocr_rule`` this reader's stored reads are written under.
FRAME_OCR_RULE = 2
#: Pixels a box grows by before its region is compared with the earlier frame.
BOX_PADDING = 4
#: Intersection over union a box needs with one of the earlier frame's boxes.
MATCH_IOU = 0.8
#: The highest mean absolute difference a window may show as unchanged.
UNCHANGED_MEAN = 2.0


class _Unreachable(Exception):
    """The file cannot be read at all; leave it pending."""


class _EngineLost(Exception):
    """The engine could not be obtained between a frame's two calls."""


def frame_read_is_outdated(
    *,
    mime_type: str,
    frame_ocr_read: bool,
    frame_ocr_indexed: bool,
    frame_ocr_rule: int | None,
    mode: str,
) -> bool:
    """Whether a stored frame read was made under an older selection rule.

    The one definition the status, index details, ``prioritize``, reindex and
    the worker share, so the backlog and the widget agree on what is pending.
    """
    from app.workers.clip import VIDEO_TYPES

    return (
        mime_type in VIDEO_TYPES
        and bool(frame_ocr_read)
        and bool(frame_ocr_indexed)
        and (frame_ocr_rule is None or int(frame_ocr_rule) < FRAME_OCR_RULE)
        and mode == "on_index"
    )


# Sentinels for the showinfo reader thread.
_BAD = object()
_END = object()
_SHOWINFO_PTS = re.compile(
    r"pts_time:([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


@dataclass
class _Segment:
    start: float
    text: str
    end: float | None = None


@dataclass
class _ReadResult:
    segments: list[_Segment] = field(default_factory=list)
    last_time: float = 0.0
    frames: int = 0
    engine_errors: int = 0
    stalled: bool = False
    exit_status: int | None = None
    engine_lost: bool = False
    times_failed: bool = False
    #: The longest gap between two decoded frames; a key-frame read's GOP.
    frame_gap: float = 0.0


def _showinfo_time(line: str) -> float | None:
    """The ``pts_time`` of one ``showinfo`` line, or ``None`` when absent."""
    match = _SHOWINFO_PTS.search(line)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _drain_showinfo(stream: Any, times: "queue.Queue[object]") -> None:
    """Read ffmpeg's stderr, putting each showinfo PTS in frame order."""
    try:
        for raw in stream:
            line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            if "Parsed_showinfo" not in line or "pts_time:" not in line:
                continue
            value = _showinfo_time(line)
            times.put(value if value is not None else _BAD)
    finally:
        times.put(_END)


def _ocr_image(image: Any) -> Any:
    """Scale a selected frame so its short side is at most 1080 lines."""
    short = min(image.width, image.height)
    if short <= MAX_OCR_LINES:
        return image
    scale = MAX_OCR_LINES / short
    width = max(2, round(image.width * scale))
    height = max(2, round(image.height * scale))
    width -= width % 2
    height -= height % 2
    return image.resize((width, height))


@dataclass
class _Box:
    """One detected box, its result, and the frame that result was recognised in."""

    rect: tuple[int, int, int, int]
    quad: Any
    line: str | None = None
    grey: Any = None


def _box_rect(quad: Any, width: int, height: int) -> tuple[int, int, int, int]:
    """The axis-aligned bounding rectangle of a quadrilateral, inside the image."""
    points = np.asarray(quad, dtype=float).reshape(-1, 2)
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    return (
        max(0, min(width, int(round(x0)))),
        max(0, min(height, int(round(y0)))),
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
    )


def _intersection_over_union(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int],
) -> float:
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - intersection
    )
    return intersection / union if union else 0.0


def _region_unchanged(
    grey: Any, reference: Any, rect: tuple[int, int, int, int],
    width: int, height: int,
) -> bool:
    """Whether every window of the padded box moved by a mean below 2.0.

    The windows are as wide as the padded box's shorter side, walked along its
    longer side, so one changed character in a long line is not averaged away
    by the line's length.
    """
    if reference is None or reference.shape != grey.shape:
        return False
    x0 = max(0, rect[0] - BOX_PADDING)
    y0 = max(0, rect[1] - BOX_PADDING)
    x1 = min(width, rect[2] + BOX_PADDING)
    y1 = min(height, rect[3] + BOX_PADDING)
    box_width, box_height = x1 - x0, y1 - y0
    if box_width <= 0 or box_height <= 0:
        return True
    difference = np.abs(
        grey[y0:y1, x0:x1].astype(np.int16)
        - reference[y0:y1, x0:x1].astype(np.int16)
    )
    side = min(box_width, box_height)
    step = max(1, side // 2)
    if box_width >= box_height:
        windows = (
            difference[:, offset:min(offset + side, box_width)]
            for offset in range(0, box_width, step)
        )
    else:
        windows = (
            difference[offset:min(offset + side, box_height), :]
            for offset in range(0, box_height, step)
        )
    return all(float(window.mean()) < UNCHANGED_MEAN for window in windows)


def _wait_while_paused(paused: Callable[[], bool] | None) -> None:
    while paused is not None and paused():
        time.sleep(0.2)


def _read_boxes(
    image: Any, previous: list[_Box], paused: Callable[[], bool] | None,
) -> list[_Box]:
    """Detect and recognise one frame's boxes, reusing the unchanged ones.

    Raises ``_EngineLost`` when the engine cannot be obtained for either call;
    an engine that raises is the caller's to treat as a failed frame.
    """
    width, height = image.size
    engine = _get_engine()
    if engine is None:
        raise _EngineLost()
    _wait_while_paused(paused)
    grey = np.asarray(image.convert("L"))
    boxes = [
        _Box(rect=_box_rect(quad, width, height), quad=quad)
        for quad in detect_boxes(engine, image)
    ]

    taken: set[int] = set()
    to_recognise: list[int] = []
    for index, box in enumerate(boxes):
        best_index = -1
        best_iou = 0.0
        for previous_index, candidate in enumerate(previous):
            if previous_index in taken:
                continue
            iou = _intersection_over_union(box.rect, candidate.rect)
            if iou >= MATCH_IOU and iou > best_iou:
                best_index, best_iou = previous_index, iou
        if best_index < 0:
            to_recognise.append(index)
            continue
        taken.add(best_index)
        picked = previous[best_index]
        if _region_unchanged(grey, picked.grey, box.rect, width, height):
            box.line = picked.line
            box.grey = picked.grey
        else:
            to_recognise.append(index)

    if to_recognise:
        engine = _get_engine()
        if engine is None:
            raise _EngineLost()
        _wait_while_paused(paused)
        answers = recognise_boxes(
            engine, image, [boxes[index].quad for index in to_recognise],
        )
        for index, answer in zip(to_recognise, answers):
            boxes[index].line = answer
            boxes[index].grey = grey
    return boxes


def _frame_command(
    path: str, width: int, height: int, keyframes: bool, needs_scale: bool,
) -> list[str]:
    if keyframes:
        # showinfo prints each decoded key frame's PTS to stderr, in the same
        # order as the frames on stdout. -fps_mode passthrough prevents the
        # rawvideo muxer from duplicating a frame to the container's rate.
        video_filter = "showinfo"
        if needs_scale:
            video_filter += f",scale={width}:{height}"
        return [
            "ffmpeg", "-hide_banner", "-v", "info",
            "-skip_frame", "nokey", "-i", path,
            "-map", "0:v:0", "-fps_mode", "passthrough",
            "-vf", video_filter,
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
        ]
    video_filter = "fps=1"
    if needs_scale:
        video_filter += f",scale={width}:{height}"
    return [
        "ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0",
        "-vf", video_filter, "-pix_fmt", "rgb24",
        "-f", "rawvideo", "-",
    ]


def _read_video(
    path: str, *, paused: Callable[[], bool] | None,
) -> _ReadResult:
    """Decode, hash, select, OCR and build segments. Holds no DB lock."""
    from PIL import Image

    width, height, needs_scale = _probe_geometry(path)
    keyframes = min(width, height) > MAX_OCR_LINES
    result = _ReadResult()
    command = _frame_command(
        path, width, height, keyframes, needs_scale,
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if keyframes else subprocess.DEVNULL,
    )
    times: queue.Queue[object] = queue.Queue()
    stderr_thread = None
    if keyframes and process.stderr is not None:
        stderr_thread = threading.Thread(
            target=_drain_showinfo,
            args=(process.stderr, times),
            daemon=True,
        )
        stderr_thread.start()
    frame_size = width * height * 3
    buffer = bytearray()
    last_data = time.monotonic()
    last_hash = None
    last_read_time: float | None = None
    open_segment: _Segment | None = None
    previous_boxes: list[_Box] = []
    index = 0
    try:
        while True:
            if paused is not None and paused():
                last_data = time.monotonic()
                time.sleep(0.2)
                continue
            ready, _, _ = select.select([process.stdout], [], [], 1.0)
            if not ready:
                if process.poll() is not None and not buffer:
                    break
                if time.monotonic() - last_data > STALL_SECONDS:
                    result.stalled = True
                    process.kill()
                    break
                continue
            chunk = os.read(process.stdout.fileno(), frame_size - len(buffer))
            if not chunk:
                break
            buffer.extend(chunk)
            last_data = time.monotonic()
            while len(buffer) >= frame_size:
                raw = bytes(buffer[:frame_size])
                del buffer[:frame_size]
                if keyframes:
                    try:
                        item = times.get(timeout=10)
                    except queue.Empty:
                        result.times_failed = True
                        process.kill()
                        return result
                    if item is _END or item is _BAD:
                        result.times_failed = True
                        process.kill()
                        return result
                    t = float(item)
                else:
                    t = float(index)
                index += 1
                result.frames += 1
                if result.frames > 1:
                    result.frame_gap = max(result.frame_gap, t - result.last_time)
                result.last_time = t
                image = Image.frombytes("RGB", (width, height), raw)
                current = _dhash(image)
                if last_hash is not None and (
                    int((current != last_hash).sum()) < READ_HASH_BITS
                    or t - (last_read_time or 0.0) < READ_GAP_SECONDS
                ):
                    continue
                try:
                    boxes = _read_boxes(
                        _ocr_image(image), previous_boxes, paused,
                    )
                except _EngineLost:
                    result.engine_lost = True
                    process.kill()
                    return result
                except Exception as exc:
                    result.engine_errors += 1
                    logger.warning(
                        "Frame OCR failed for one frame (%s)", type(exc).__name__,
                    )
                    boxes = []
                last_data = time.monotonic()
                text = "\n".join(
                    box.line for box in boxes if box.line is not None
                )
                if open_segment is not None and text != open_segment.text:
                    open_segment.end = t
                    result.segments.append(open_segment)
                    open_segment = None
                if text:
                    open_segment = _Segment(start=t, text=text)
                last_hash = current
                last_read_time = t
                previous_boxes = boxes
    finally:
        try:
            process.wait(timeout=10)
        except Exception:
            process.kill()
        result.exit_status = process.returncode
        if stderr_thread is not None:
            stderr_thread.join(timeout=5)
    if open_segment is not None:
        open_segment.end = result.last_time + 1.0
        result.segments.append(open_segment)
    return result


def _chunks_for_segments(segments: list[_Segment]) -> list[dict[str, Any]]:
    """One entry per chunk, chunk_index in time order across the video."""
    chunks: list[dict[str, Any]] = []
    for segment in segments:
        for text in ContentExtractor.chunk_text(segment.text):
            chunks.append({
                "chunk_index": len(chunks),
                "start_time": segment.start,
                "end_time": segment.end if segment.end is not None else segment.start,
                "text": text,
            })
    return chunks


def _store_read(file_id: str, chunks: list[dict[str, Any]]) -> bool:
    """A5: replace the stored read in one short transaction."""
    try:
        with get_search_db() as session:
            row = session.query(IndexedFile).filter_by(
                file_id=file_id, active=True
            ).first()
            if row is None:
                return False
            session.query(VideoFrameText).filter_by(file_id=file_id).delete()
            _remove_embeddings(session, file_id, "text_content")
            delete_fts_text_content(session, file_id)
            for chunk in chunks:
                session.add(VideoFrameText(
                    file_id=file_id,
                    chunk_index=chunk["chunk_index"],
                    start_time=chunk["start_time"],
                    end_time=chunk["end_time"],
                    text=chunk["text"],
                ))
            row.frame_ocr_read = True
            row.frame_ocr_indexed = False
            row.frame_ocr_rule = FRAME_OCR_RULE
            return True
    except Exception as exc:
        logger.error("Frame OCR write failed for %s: %s", file_id, exc)
        return False


def _keep_earlier_read(file_id: str) -> bool:
    """A re-read that was not complete: keep the read, stop retrying it.

    Only ``frame_ocr_rule`` is written, so the earlier text stays searchable
    and no automatic pass tries this video again; a reindex still reads it.
    """
    try:
        with get_search_db() as session:
            row = session.query(IndexedFile).filter_by(
                file_id=file_id, active=True
            ).first()
            if row is None:
                return False
            row.frame_ocr_rule = FRAME_OCR_RULE
            return True
    except Exception as exc:
        logger.error("Frame OCR rule write failed for %s: %s", file_id, exc)
        return False


def _read_stored_chunks(file_id: str) -> list[dict[str, Any]]:
    with get_search_db_read() as session:
        rows = (
            session.query(VideoFrameText)
            .filter_by(file_id=file_id)
            .order_by(VideoFrameText.chunk_index)
            .all()
        )
        return [
            {
                "chunk_index": row.chunk_index,
                "start_time": row.start_time,
                "end_time": row.end_time,
                "text": row.text,
            }
            for row in rows
        ]


def _build_rows(file_id: str) -> bool:
    """A6: build the text_content rows from the stored read."""
    chunks = _read_stored_chunks(file_id)
    vectors: list[Any] = []
    if chunks:
        try:
            vectors = list(embed_passages([chunk["text"] for chunk in chunks]))
        except Exception as exc:
            logger.error(
                "Frame text embedding failed for %s: %s", file_id, exc,
            )
            return False
    try:
        with get_search_db() as session:
            row = session.query(IndexedFile).filter_by(
                file_id=file_id, active=True
            ).first()
            if row is None or not row.frame_ocr_read:
                return False
            _remove_embeddings(session, file_id, "text_content")
            delete_fts_text_content(session, file_id)
            for chunk, vector in zip(chunks, vectors):
                _store_embedding(
                    session=session,
                    embedding_id=(
                        f"txt_{file_id}_{chunk['chunk_index']}_"
                        f"{uuid.uuid4().hex[:8]}"
                    ),
                    file_id=file_id,
                    embedding_type="text_content",
                    vector_table="vec_text",
                    vector=vector,
                    content_preview=chunk["text"][:200],
                    page=None,
                    chunk_index=chunk["chunk_index"],
                    timestamp_start=chunk["start_time"],
                    timestamp_end=chunk["end_time"],
                )
            if chunks:
                upsert_fts_text_content(session, file_id, [
                    {
                        "chunk_index": chunk["chunk_index"],
                        "page": None,
                        "text": chunk["text"],
                    }
                    for chunk in chunks
                ])
            row.frame_ocr_indexed = True
            return True
    except Exception as exc:
        logger.error("Frame text row write failed for %s: %s", file_id, exc)
        return False


def _reread(
    file_id: str, path: str, is_paused: Callable[[], bool] | None,
) -> bool:
    """Read a video again under the current rule, or keep its earlier read.

    A read that is anything but complete writes nothing but ``frame_ocr_rule``:
    the earlier frames and search rows stay, and no automatic pass tries the
    video again.
    """
    if not _reachable(path):
        logger.warning(
            "Frame OCR skipped for %s: file is unreachable", file_id,
        )
        return False

    engine = _get_engine()
    if engine is None:
        return False

    try:
        result = _read_video(path, paused=is_paused)
    except _Unreadable as exc:
        if not _reachable(path):
            logger.warning(
                "Frame OCR skipped for %s: file is unreachable", file_id,
            )
            return False
        cause = exc.__cause__ or exc
        logger.warning(
            "Frame OCR for %s: the file could not be decoded (%s); "
            "keeping the earlier read",
            file_id, type(cause).__name__,
        )
        return _keep_earlier_read(file_id)

    if result.times_failed:
        logger.warning(
            "Frame OCR for %s: key-frame times did not match the decoded "
            "frames; keeping the earlier read",
            file_id,
        )
        return _keep_earlier_read(file_id)
    if result.engine_lost:
        logger.warning(
            "Frame OCR lost the engine partway for %s; nothing written",
            file_id,
        )
        return False

    if result.stalled or (result.exit_status not in (0, None)):
        if not _reachable(path):
            logger.warning(
                "Frame OCR skipped for %s: file became unreachable (%s)",
                file_id,
                "stall" if result.stalled else f"ffmpeg exit {result.exit_status}",
            )
            return False
        logger.warning(
            "Frame OCR for %s ended early (%s); keeping the earlier read",
            file_id,
            "stall" if result.stalled else f"ffmpeg exit {result.exit_status}",
        )
        return _keep_earlier_read(file_id)
    if result.frames == 0:
        if not _reachable(path):
            logger.warning(
                "Frame OCR skipped for %s: file became unreachable", file_id,
            )
            return False
        logger.warning(
            "Frame OCR for %s decoded no frame; keeping the earlier read",
            file_id,
        )
        return _keep_earlier_read(file_id)

    if result.engine_errors:
        logger.warning(
            "Frame OCR for %s: the engine failed on %d frame(s); "
            "keeping the earlier read",
            file_id, result.engine_errors,
        )
        return _keep_earlier_read(file_id)

    chunks = _chunks_for_segments(result.segments)
    if not _store_read(file_id, chunks):
        return False
    return _build_rows(file_id)


def index_frame_ocr(
    file_id: str, *, is_paused: Callable[[], bool] | None = None,
    source_path: str | None = None,
) -> bool:
    """Read one video's frames, or rebuild its stored read's search rows.

    ``source_path`` is the video a ``.loft`` fetch downloaded for this file;
    the `.loft`'s own path is never decoded, and a `.loft` without a
    downloaded source is left exactly as it was.

    Returns ``True`` when the video is closed as done, ``False`` when it
    stays pending (unreachable, no engine, embedding or write failure).
    """
    with get_search_db_read() as session:
        file = session.query(IndexedFile).filter_by(
            file_id=file_id, active=True
        ).first()
        if file is None:
            return False
        path = file.file_path
        mime_type = file.mime_type
        already_read = bool(file.frame_ocr_read)
        already_indexed = bool(file.frame_ocr_indexed)
        outdated = frame_read_is_outdated(
            mime_type=mime_type,
            frame_ocr_read=already_read,
            frame_ocr_indexed=already_indexed,
            frame_ocr_rule=file.frame_ocr_rule,
            mode=settings.features.video_frame_ocr,
        )

    if already_read and not already_indexed:
        return _build_rows(file_id)
    if already_read and already_indexed:
        if not outdated:
            return True
        return _reread(file_id, path, is_paused)

    if source_path is not None:
        path = source_path
    elif mime_type == LOFT_MIME:
        return False

    if not _reachable(path):
        logger.warning(
            "Frame OCR skipped for %s: file is unreachable", file_id,
        )
        return False

    engine = _get_engine()
    if engine is None:
        return False

    try:
        result = _read_video(path, paused=is_paused)
    except _Unreadable as exc:
        if not _reachable(path):
            logger.warning(
                "Frame OCR skipped for %s: file is unreachable", file_id,
            )
            return False
        cause = exc.__cause__ or exc
        logger.warning(
            "Frame OCR for %s: the file could not be decoded (%s); "
            "closing it as read",
            file_id, type(cause).__name__,
        )
        if not _store_read(file_id, []):
            return False
        return _build_rows(file_id)

    if result.times_failed:
        logger.warning(
            "Frame OCR for %s: key-frame times did not match the decoded "
            "frames; nothing written",
            file_id,
        )
        return False
    if result.engine_lost:
        logger.warning(
            "Frame OCR lost the engine partway for %s; nothing written",
            file_id,
        )
        return False

    if result.stalled or (result.exit_status not in (0, None)):
        if not _reachable(path):
            logger.warning(
                "Frame OCR skipped for %s: file became unreachable (%s)",
                file_id,
                "stall" if result.stalled else f"ffmpeg exit {result.exit_status}",
            )
            return False
        logger.warning(
            "Frame OCR for %s ended early (%s); keeping the frames read",
            file_id,
            "stall" if result.stalled else f"ffmpeg exit {result.exit_status}",
        )
    elif result.frames == 0:
        if not _reachable(path):
            logger.warning(
                "Frame OCR skipped for %s: file became unreachable", file_id,
            )
            return False
        logger.warning(
            "Frame OCR for %s decoded no frame; closing it as read", file_id,
        )

    if result.engine_errors:
        logger.warning(
            "Frame OCR for %s: the engine failed on %d frame(s)",
            file_id, result.engine_errors,
        )

    chunks = _chunks_for_segments(result.segments)
    if not _store_read(file_id, chunks):
        return False
    return _build_rows(file_id)
