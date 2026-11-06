"""Video frame OCR (SPEC-ADDON-066).

Decodes a video's frames, reads the ones whose picture changed enough with
the shared picture-OCR engine, and stores the text as timed chunks. Videos
taller than 1080 lines decode only key frames and read a selected frame
scaled down to 1080 lines (the human's post-approval revision).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import select
import stat
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from app.database import (
    delete_fts_text_content,
    get_search_db,
    get_search_db_read,
    upsert_fts_text_content,
)
from app.extractors.base import ContentExtractor
from app.models import IndexedFile, VideoFrameText
from app.workers.metadata import _remove_embeddings, _store_embedding, embed_passages
from app.workers.ocr import _get_engine, run_engine

logger = logging.getLogger(__name__)

HASH_WIDTH = 160
HASH_BITS_THRESHOLD = 16
GAP_SECONDS = 10.0
STALL_SECONDS = 120.0
MAX_OCR_LINES = 1080


class _Unreachable(Exception):
    """The file cannot be read at all; leave it pending."""


class _Unreadable(Exception):
    """The file was reached but cannot be decoded; close it as read."""


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


def _reachable(path: str) -> bool:
    try:
        info = os.stat(path)
    except (OSError, ValueError):
        return False
    return stat.S_ISREG(info.st_mode)


def _probe(path: str) -> tuple[int, int]:
    """The first video stream's upright, even width and height."""
    return _probe_geometry(path)[:2]


def _probe_geometry(path: str) -> tuple[int, int, bool]:
    """The upright even size, and whether a scale is needed to reach it.

    ffmpeg auto-rotates on decode, so a phone video's display matrix turns a
    coded 3840x2160 into 2160x3840. The reader must cut the raw frames at
    that post-rotation size, and the same size decides the short-side path.
    An odd coded side needs the scale filter; an even one does not.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries",
             "stream=width,height:stream_side_data:stream_tags=rotate",
             "-of", "json", path],
            capture_output=True, text=True, timeout=60,
        ).stdout
        stream = (json.loads(out).get("streams") or [])[0]
        width, height = int(stream["width"]), int(stream["height"])
    except Exception as exc:
        raise _Unreadable("no readable video stream") from exc

    rotation = 0.0
    for side in stream.get("side_data_list") or ():
        value = side.get("rotation")
        if value is not None:
            try:
                rotation = float(value)
            except (TypeError, ValueError):
                rotation = 0.0
            break
    if rotation == 0.0:
        try:
            rotation = float((stream.get("tags") or {}).get("rotate") or 0)
        except (TypeError, ValueError):
            rotation = 0.0
    if abs(rotation) % 180 == 90:
        width, height = height, width

    needs_scale = bool(width % 2 or height % 2)
    width -= width % 2
    height -= height % 2
    if width <= 0 or height <= 0:
        raise _Unreadable("no readable video stream")
    return width, height, needs_scale


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


def _dhash(image: Any) -> Any:
    """256-bit difference hash: 17x16 gray, each pixel vs its right neighbour."""
    import numpy as np
    from PIL import Image

    resample = Image.Resampling.BOX
    gray = image.convert("L").resize(
        (
            HASH_WIDTH,
            max(2, round(HASH_WIDTH * image.height / image.width)),
        ),
        resample,
    )
    small = gray.resize((17, 16), resample)
    pixels = np.asarray(small, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).ravel()


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
                result.last_time = t
                image = Image.frombytes("RGB", (width, height), raw)
                current = _dhash(image)
                if last_hash is not None and (
                    int((current != last_hash).sum()) < HASH_BITS_THRESHOLD
                    or t - (last_read_time or 0.0) < GAP_SECONDS
                ):
                    continue
                engine_now = _get_engine()
                if engine_now is None:
                    result.engine_lost = True
                    process.kill()
                    return result
                text = ""
                try:
                    answer = run_engine(engine_now, _ocr_image(image))
                    text = "\n".join(getattr(answer, "txts", None) or ())
                except Exception as exc:
                    result.engine_errors += 1
                    logger.warning(
                        "Frame OCR failed for one frame (%s)", type(exc).__name__,
                    )
                if open_segment is not None and text != open_segment.text:
                    open_segment.end = t
                    result.segments.append(open_segment)
                    open_segment = None
                if text:
                    open_segment = _Segment(start=t, text=text)
                last_hash = current
                last_read_time = t
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
            return True
    except Exception as exc:
        logger.error("Frame OCR write failed for %s: %s", file_id, exc)
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


def index_frame_ocr(
    file_id: str, *, is_paused: Callable[[], bool] | None = None,
) -> bool:
    """Read one video's frames, or rebuild its stored read's search rows.

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
        already_read = bool(file.frame_ocr_read)
        already_indexed = bool(file.frame_ocr_indexed)

    if already_read and already_indexed:
        return True
    if already_read:
        return _build_rows(file_id)

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
