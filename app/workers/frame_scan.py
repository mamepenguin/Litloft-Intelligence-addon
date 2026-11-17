"""The frame scan: one difference hash and picture size per second.

The seconds whose picture changed by at least 16 bits since the last kept
second, at least 10 apart, become the file's change frames: the positions
where a representative is likely to be worth choosing. Only the selection
reads them, so they get a vector table of their own.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import logging
import os
import select
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.config import settings
from app.database import get_search_db, get_search_db_read
from app.models import IndexedFile, VideoFrameScan
from app.video_frames import (
    GAP_SECONDS,
    HASH_BITS_THRESHOLD,
    Unreadable,
    dhash,
    extract_frame,
    probe_geometry,
    reachable,
)
from app.workers.clip import VIDEO_TYPES, embed_image, load_image_model
from app.workers.metadata import _remove_embeddings, _store_embedding
from app.workers.whisper import LOFT_MIME

logger = logging.getLogger(__name__)

#: The only scan-row layout this build writes.
SCAN_VERSION = 1
SCAN_WIDTH = 160
#: The quality the stored picture sizes are measured at.
SIZE_QUALITY = 75
#: Seconds without a decoded byte before ffmpeg is killed.
STALL_SECONDS = 120.0
#: Extraction and embedding together stop after this long, keeping what they have.
EMBED_SECONDS = 600.0
CHANGE_TYPE = "clip_change"
CHANGE_TABLE = "vec_clip_change"


@dataclass
class _Decoded:
    hashes: Any = field(default_factory=lambda: np.zeros((0, 256), dtype=bool))
    sizes: list[int] = field(default_factory=list)
    stalled: bool = False
    exit_status: int | None = None

    @property
    def seconds(self) -> int:
        return len(self.sizes)


def _scan_command(path: str, width: int, height: int) -> list[str]:
    return [
        "ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0",
        "-vf", f"fps=1,scale={width}:{height},format=gray",
        "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]


def _jpeg_size(image: Any) -> int:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=SIZE_QUALITY)
    return min(buffer.tell(), 65535)


def _decode(
    path: str, *, paused: Callable[[], bool] | None,
) -> _Decoded:
    """Read one picture per second, hashing it. Holds no database session."""
    from PIL import Image

    width, height, _needs_scale = probe_geometry(path)
    # Even, so a half-pixel rounding never leaves a row the raw reader splits
    # at the wrong boundary.
    scan_width = SCAN_WIDTH
    scan_height = max(2, round(height * scan_width / width))
    scan_height -= scan_height % 2
    frame_size = scan_width * scan_height

    decoded = _Decoded()
    hashes: list[Any] = []
    process = subprocess.Popen(
        _scan_command(path, scan_width, scan_height),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    buffer = bytearray()
    last_data = time.monotonic()
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
                    decoded.stalled = True
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
                picture = Image.frombytes("L", (scan_width, scan_height), raw)
                hashes.append(dhash(picture))
                decoded.sizes.append(_jpeg_size(picture))
    finally:
        try:
            process.wait(timeout=10)
        except Exception:
            process.kill()
        decoded.exit_status = process.returncode

    if hashes:
        decoded.hashes = np.vstack(hashes)
    return decoded


def _change_seconds(hashes: Any) -> list[int]:
    seconds = [0]
    last = 0
    for second in range(1, len(hashes)):
        if second - last < GAP_SECONDS:
            continue
        changed = int(np.count_nonzero(hashes[second] != hashes[last]))
        if changed < HASH_BITS_THRESHOLD:
            continue
        seconds.append(second)
        last = second
    return seconds


def _spread(seconds: list[int], max_frames: int) -> list[int]:
    """At most ``max_frames`` of ``seconds``, evenly over the whole list."""
    if len(seconds) <= max_frames:
        return seconds
    count = len(seconds)
    return [seconds[(i * count) // max_frames] for i in range(max_frames)]


def _pack_sizes(sizes: list[int]) -> bytes:
    return np.asarray(sizes, dtype="<u2").tobytes()


def _policy_allows(drive: str) -> bool:
    """Whether the drive has the ``index`` feature on.

    The policy client is async and a scan runs off the loop, so a coroutine
    verdict is run on a loop of its own. A stand-in that answers directly is
    taken as it is.
    """
    from app.policy_client import is_feature_enabled

    verdict = is_feature_enabled(drive, "index", default_on_failure=True)
    if inspect.isawaitable(verdict):
        verdict = asyncio.run(verdict)
    return bool(verdict)


def _embed_changes(
    path: str,
    seconds: list[int],
    *,
    paused: Callable[[], bool] | None,
) -> list[tuple[int, Any]] | None:
    """Embed each change second's own frame.

    Returns ``None`` when the model cannot be loaded, before anything is
    embedded. A frame that cannot be taken or embedded is skipped; the rest
    are kept.
    """
    from PIL import Image

    if not seconds:
        return []
    try:
        load_image_model()
    except Exception:
        return None

    embedded: list[tuple[int, Any]] = []
    deadline = time.monotonic() + EMBED_SECONDS
    with tempfile.TemporaryDirectory(prefix="frame_scan_") as tmpdir:
        for second in seconds:
            if time.monotonic() > deadline:
                logger.warning(
                    "Frame scan stopped at the %ds budget after %d frame(s)",
                    int(EMBED_SECONDS), len(embedded),
                )
                break
            if paused is not None:
                while paused():
                    time.sleep(0.2)
            frame_path = Path(tmpdir) / f"{second:06d}.jpg"
            if not extract_frame(path, float(second), frame_path):
                logger.warning(
                    "Frame scan skipped the change frame at %ds", second,
                )
                continue
            try:
                with Image.open(frame_path) as opened:
                    image = opened.convert("RGB")
                vector = np.asarray(embed_image(image), dtype=np.float32)
            except Exception as exc:
                logger.warning(
                    "Frame scan skipped the change frame at %ds (%s)",
                    second, type(exc).__name__,
                )
                continue
            embedded.append((second, vector))
    return embedded


def _store(
    file_id: str,
    decoded: _Decoded,
    embedded: list[tuple[int, Any]],
) -> bool:
    """Replace the file's scan and change frames, and close its flag.

    One transaction, so a failure leaves the previous scan and change frames in
    use and the flag false.
    """
    hashes = np.packbits(decoded.hashes, axis=1).tobytes() if decoded.seconds else b""
    with get_search_db() as session:
        row = session.query(IndexedFile).filter_by(
            file_id=file_id, active=True
        ).first()
        if row is None:
            logger.warning(
                "Frame scan skipped for %s: file is no longer active", file_id,
            )
            return False
        _remove_embeddings(session, file_id, CHANGE_TYPE)
        session.query(VideoFrameScan).filter_by(file_id=file_id).delete()
        for second, vector in embedded:
            _store_embedding(
                session=session,
                embedding_id=f"chg_{file_id}_{second}",
                file_id=file_id,
                embedding_type=CHANGE_TYPE,
                vector_table=CHANGE_TABLE,
                vector=vector,
                content_preview=f"Frame at {second}s",
                timestamp_start=float(second),
            )
        # No second decoded: a scan row would claim a scan of a video that has
        # none, which is how a file with no decodable picture is closed.
        if decoded.seconds:
            session.add(VideoFrameScan(
                file_id=file_id,
                version=SCAN_VERSION,
                seconds=decoded.seconds,
                hashes=hashes,
                sizes=_pack_sizes(decoded.sizes),
            ))
        row.frame_scan_indexed = True
    return True


def index_frame_scan(
    file_id: str, *, is_paused: Callable[[], bool] | None = None,
    source_path: str | None = None,
) -> bool:
    """Scan one video's seconds and store its change frames.

    ``source_path`` is the video a ``.loft`` fetch downloaded for this file;
    the `.loft`'s own path is never decoded. A scan that stores change frames
    from a downloaded source also keeps a 320px picture of each one.

    Returns ``True`` when the file is closed as scanned, ``False`` when it
    stays pending (not a native video, already scanned, policy off, model
    unavailable, unreachable, or the write failed).
    """
    with get_search_db_read() as session:
        file = session.query(IndexedFile).filter_by(
            file_id=file_id, active=True
        ).first()
        if file is None:
            return False
        path = file.file_path
        mime_type = file.mime_type
        drive = file.drive
        already_scanned = bool(file.frame_scan_indexed)

    if source_path is not None:
        path = source_path
    elif mime_type == LOFT_MIME:
        # A `.loft` is only ever decoded from the file its fetch downloaded.
        return False
    elif mime_type not in VIDEO_TYPES:
        return False

    if already_scanned:
        return False

    try:
        permitted = _policy_allows(drive)
    except Exception:
        permitted = True
    if not permitted:
        logger.warning(
            "Frame scan skipped for %s: intelligence.index is off for drive %s",
            file_id, drive,
        )
        return False

    if not reachable(path):
        logger.warning(
            "Frame scan skipped for %s: file is unreachable", file_id,
        )
        return False

    try:
        decoded = _decode(path, paused=is_paused)
    except Unreadable as exc:
        if not reachable(path):
            logger.warning(
                "Frame scan skipped for %s: file became unreachable", file_id,
            )
            return False
        cause = exc.__cause__ or exc
        logger.warning(
            "Frame scan for %s: the file could not be decoded (%s); "
            "closing it as scanned",
            file_id, type(cause).__name__,
        )
        decoded = _Decoded()

    if decoded.stalled or (decoded.exit_status not in (0, None)):
        if not reachable(path):
            logger.warning(
                "Frame scan skipped for %s: file became unreachable (%s)",
                file_id,
                "stall" if decoded.stalled else f"ffmpeg exit {decoded.exit_status}",
            )
            return False
        logger.warning(
            "Frame scan for %s ended early (%s); keeping the seconds read",
            file_id,
            "stall" if decoded.stalled else f"ffmpeg exit {decoded.exit_status}",
        )
    elif decoded.seconds == 0:
        if not reachable(path):
            logger.warning(
                "Frame scan skipped for %s: file became unreachable", file_id,
            )
            return False
        logger.warning(
            "Frame scan for %s decoded no second; closing it as scanned", file_id,
        )

    changes = _change_seconds(decoded.hashes) if decoded.seconds else []
    kept = _spread(
        changes, settings.indexing.frame_extraction.max_frames,
    )
    embedded = _embed_changes(path, kept, paused=is_paused)
    if embedded is None:
        logger.warning(
            "Frame scan skipped for %s: the CLIP model could not be loaded",
            file_id,
        )
        return False

    if not reachable(path):
        logger.warning(
            "Frame scan skipped for %s: file became unreachable before the write",
            file_id,
        )
        return False

    try:
        stored = _store(file_id, decoded, embedded)
    except Exception as exc:
        logger.error("Frame scan write failed for %s: %s", file_id, exc)
        return False
    if stored and source_path is not None:
        _keep_small_pictures(file_id, source_path, [s for s, _ in embedded])
    return stored


def _keep_small_pictures(file_id: str, source_path: str, seconds: list[int]) -> None:
    """Keep the ``small`` frame cache picture of each stored change frame.

    Runs after the scan's row commit. An entry already there is left alone,
    and one that cannot be written is skipped: the change frame itself is
    still usable, so a picture is not worth failing the scan for.
    """
    from app.frame_cache import extract_frame_to_cache, frame_cache_path

    for second in seconds:
        cache_path = frame_cache_path(file_id, float(second))
        if cache_path.exists() and cache_path.stat().st_size > 0:
            continue
        try:
            extract_frame_to_cache(source_path, cache_path, float(second))
        except Exception as exc:
            logger.warning(
                "Frame scan skipped the small picture at %ds for %s (%s)",
                second, file_id, type(exc).__name__,
            )
