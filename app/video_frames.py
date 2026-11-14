"""Decoding pieces shared by the video workers.

The frame-OCR reader, the scene-CLIP frame extraction and the frame scan open
the same files the same way; what they share lives here so they cannot drift.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HASH_WIDTH = 160
#: Difference-hash bits that make one picture a change of another.
HASH_BITS_THRESHOLD = 16
#: Seconds that must pass before a differing picture is read as a change.
GAP_SECONDS = 10.0


class Unreadable(Exception):
    """The file was reached but cannot be decoded as video."""


def reachable(path: str) -> bool:
    """Whether ``path`` is a regular file this process can stat."""
    try:
        info = os.stat(path)
    except (OSError, ValueError):
        return False
    return stat.S_ISREG(info.st_mode)


def probe_geometry(path: str) -> tuple[int, int, bool]:
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
        raise Unreadable("no readable video stream") from exc

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
        raise Unreadable("no readable video stream")
    return width, height, needs_scale


def probe(path: str) -> tuple[int, int]:
    """The first video stream's upright, even width and height."""
    return probe_geometry(path)[:2]


def dhash(image: Any) -> Any:
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


def extract_frame(video_path: str, timestamp: float, frame_path: Path) -> bool:
    """Write the frame at ``timestamp`` as the 224x224 JPEG scene frames are.

    The single-frame decode is the one the scene-CLIP interval frames use, so a
    frame taken here is the frame that route would embed.

    Returns:
        Whether ffmpeg wrote a frame at ``frame_path``.
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-ss", str(timestamp),
                "-i", video_path,
                "-vf", "scale=224:224,format=yuvj420p",
                "-frames:v", "1",
                "-q:v", "5",
                str(frame_path),
            ],
            capture_output=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning(
            "Interval frame extraction failed at %ss: %s", timestamp, exc,
        )
        return False
    return frame_path.exists()
