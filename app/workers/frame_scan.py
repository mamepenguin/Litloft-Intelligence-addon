"""The frame scan: one difference hash and picture size per second.

The scan stores no frame of its own. Its hashes and sizes let the selection
pick steady seconds, and the scene candidates are the file's only frame list.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import logging
import os
import select
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from app.database import get_search_db, get_search_db_read
from app.models import IndexedFile, VideoFrameScan
from app.video_frames import (
    Unreadable,
    dhash,
    probe_geometry,
    reachable,
)
from app.workers.clip import VIDEO_TYPES
from app.workers.metadata import _remove_embeddings
from app.workers.whisper import LOFT_MIME

logger = logging.getLogger(__name__)

#: The only scan-row layout this build writes.
SCAN_VERSION = 1
SCAN_WIDTH = 160
#: The quality the stored picture sizes are measured at.
SIZE_QUALITY = 75
#: Seconds without a decoded byte before ffmpeg is killed.
STALL_SECONDS = 120.0
CHANGE_TYPE = "clip_change"


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


def _store(file_id: str, decoded: _Decoded) -> bool:
    """Replace the file's scan, delete its change frames, and close its flag.

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
    """Scan one video's seconds and store its hashes and sizes.

    ``source_path`` is the video a ``.loft`` fetch downloaded for this file;
    the `.loft`'s own path is never decoded.

    Returns ``True`` when the file is closed as scanned, ``False`` when it
    stays pending (not a native video, already scanned, policy off,
    unreachable, or the write failed).
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

    if not reachable(path):
        logger.warning(
            "Frame scan skipped for %s: file became unreachable before the write",
            file_id,
        )
        return False

    try:
        return _store(file_id, decoded)
    except Exception as exc:
        logger.error("Frame scan write failed for %s: %s", file_id, exc)
        return False
