"""Shared video-frame extraction + persistent disk cache.

Extracted from ``app.routers.files`` (design doc "Video Visual Index"
§4.3) so both the ``GET /files/{file_id}/frame`` HTTP endpoint and the
video-visual worker share one implementation of path safety, atomic
cache writes, and cache invalidation — there is no second copy of this
logic and no HTTP loopback from the worker to its own router.

Frames are extracted on demand via ffmpeg and cached as WebP (320px
wide, q70). Cache layout: per-file directory, immutable filename
derived from the timestamp in milliseconds. The cache is purged when
the source file is removed from the index (see ``_purge_file`` in
``app.indexer`` and ``purge_drive`` in ``app.purge``).

Why WebP 320px q70:
  - The CLIP frames grid renders at most ~260px-wide cells
    (md:grid-cols-5 inside the file detail page); 320px gives a
    small Retina headroom without paying for a full 480px frame.
  - WebP libwebp at q70 is roughly 1/3-1/4 the size of the previous
    mjpeg q3 output (~30-50KB → ~8-15KB) with no perceptible
    quality loss at this scale, and is supported by every browser
    the host targets.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from fastapi import HTTPException

logger = logging.getLogger(__name__)

_FRAME_CACHE_SUBDIR = "frames"
_FRAME_FFMPEG_TIMEOUT = 15


def frame_cache_dir(file_id: str) -> Path:
    """Return the on-disk directory for a file's cached frame thumbnails.

    Lives under ``intelligence_data_dir / frames / {file_id}``. Resolved
    fresh on every call so tests can swap ``intelligence_data_dir`` via
    the ``settings`` dataclass without monkeypatching this helper.
    """
    from app.config import settings

    return Path(settings.intelligence_data_dir) / _FRAME_CACHE_SUBDIR / file_id


def frame_cache_path(file_id: str, timestamp_seconds: float) -> Path:
    """Resolve the WebP cache path for ``(file_id, timestamp)``.

    Timestamps are quantised to milliseconds in the filename so the
    ``<img src>`` URL is byte-stable across requests (frontend always
    passes the same float, but rounding here defends against future
    callers and lets a long Cache-Control / immutable header be
    correct).
    """
    ms = int(round(timestamp_seconds * 1000))
    return frame_cache_dir(file_id) / f"{ms}.webp"


def extract_frame_to_cache(
    abs_path: str,
    cache_path: Path,
    timestamp_seconds: float,
) -> None:
    """Run ffmpeg to extract one frame and atomically write it to ``cache_path``.

    Writes to a sibling ``.tmp`` file first then ``os.replace`` so a
    concurrent reader never sees a half-written WebP. Raises
    ``HTTPException`` on timeout / non-zero ffmpeg exit so the caller
    can surface a sensible status code.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss", str(timestamp_seconds),
                "-i", abs_path,
                "-frames:v", "1",
                "-vf", "scale=320:-1",
                "-c:v", "libwebp",
                "-q:v", "70",
                # ``-f webp`` is required because the temp filename ends
                # in ``.webp.tmp`` — ffmpeg can't infer the muxer from
                # the dotted extension and otherwise errors out with
                # "Unable to choose an output format".
                "-f", "webp",
                str(tmp_path),
            ],
            capture_output=True,
            timeout=_FRAME_FFMPEG_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=504, detail="Frame extraction timed out")

    if result.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size == 0:
        logger.warning(
            "Frame extraction failed for %s at %.3fs (returncode=%s)",
            cache_path.parent.name, timestamp_seconds, result.returncode,
        )
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Frame extraction failed")

    os.replace(tmp_path, cache_path)


def ensure_frame_cached(
    file_id: str, timestamp_seconds: float, abs_path: str
) -> Path:
    """Return the cache path for ``(file_id, timestamp)``, extracting on miss.

    Shared entry point for both the HTTP frame endpoint and the
    video-visual worker: a cache hit short-circuits to the existing
    file, a miss shells out to ffmpeg via :func:`extract_frame_to_cache`.
    Raises ``HTTPException`` on extraction failure (mirrors the prior
    router-only behaviour); callers that are not FastAPI handlers
    should catch it like any other exception.
    """
    cache_path = frame_cache_path(file_id, timestamp_seconds)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    extract_frame_to_cache(abs_path, cache_path, timestamp_seconds)
    return cache_path


def purge_frame_cache(file_id: str) -> None:
    """Remove every cached frame thumbnail for ``file_id``.

    Best-effort: a missing directory is treated as already-purged. Errors
    are swallowed to avoid blocking the caller's purge transaction —
    leaving stale thumbnails behind is harmless (they get re-keyed when
    the file is re-indexed and the directory is unique per file_id).
    """
    cache_dir = frame_cache_dir(file_id)
    if not cache_dir.exists():
        return
    try:
        shutil.rmtree(cache_dir)
    except OSError as exc:
        logger.warning("purge_frame_cache: failed for %s (%s)", file_id, exc)


__all__ = [
    "ensure_frame_cached",
    "extract_frame_to_cache",
    "frame_cache_dir",
    "frame_cache_path",
    "purge_frame_cache",
]
