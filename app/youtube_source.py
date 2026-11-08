"""Frames and a storyboard for a ``.loft`` that points at a YouTube video.

Nothing is downloaded but one frame at a time (``ffmpeg -ss`` on the video
stream ``yt-dlp`` names) and the small storyboard sheets. A frame that cannot
be read is an error; no other frame is ever substituted for it.

Neither the stored URL, a signed stream URL nor a request header is ever put in
a log line: the failure classes below are what is logged.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

from fastapi import HTTPException

from app.frame_cache import FrameSize, frame_ffmpeg_argv

logger = logging.getLogger(__name__)

# ``yt-dlp``'s sheet and stream URLs are signed, and ``httpx`` prints its
# request line at INFO. Why not raise the ``httpx`` logger's level instead: that
# is the only request-level trace any other httpx caller in the process
# produces, and the leak is confined to these fetches.
_SIGNED_FETCH = re.compile(
    r"ytimg\.com|googlevideo\.com|youtube\.com|sigh=|sqp="
)


class _SignedFetchFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _SIGNED_FETCH.search(record.getMessage())


def _drop_signed_fetches() -> None:
    httpx_logger = logging.getLogger("httpx")
    if not any(isinstance(rule, _SignedFetchFilter) for rule in httpx_logger.filters):
        httpx_logger.addFilter(_SignedFetchFilter())


_drop_signed_fetches()

FRAME_DEADLINE_SECONDS = 15.0
LOOKUP_LIMIT_SECONDS = 15.0
STORYBOARD_LOOKUP_LIMIT_SECONDS = 60.0
DESCRIPTION_TTL_SECONDS = 300.0
FFMPEG_SLOTS = 4
# An ffmpeg run that starts with less than this left cannot read a frame;
# starting it only to kill it would hold a slot for nothing.
_MIN_FFMPEG_BUDGET_SECONDS = 1.0
_MAX_LOFT_BYTES = 64 * 1024
_MAX_FRAME_HEIGHT = 720

YOUTUBE_PROVIDER = "youtube"


@dataclass(frozen=True)
class LoftSource:
    provider: str
    url: str


class LoftUnreadable(Exception):
    """The ``.loft`` file could not be read."""


class LoftNotAccessible(LoftUnreadable):
    """The ``.loft`` path is outside the allowed base directories."""


class LookupFailed(Exception):
    """``yt-dlp`` gave no description; ``detail`` is safe to log."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def read_loft(
    file_path: str | None,
    drive: str | None = None,
    *,
    validate: Callable[[str], bool] | None = None,
) -> LoftSource | None:
    """The provider and URL a ``.loft`` stores; ``None`` when it holds none.

    ``drive`` resolves a drive-relative ``file_path`` the way the native frame
    route does; an already absolute stored path needs none. Raises
    :class:`LoftUnreadable` when the file cannot be read at all, which is not
    the same as a readable file that is not a usable reference. ``validate``
    is the caller's own path check where it has one (the CLIP worker's), else
    ``app.config.validate_file_path``.
    """
    from app.config import resolve_file_path, resolve_stored_path, validate_file_path

    validate = validate or validate_file_path

    path = resolve_stored_path(
        resolve_file_path(drive, file_path) if drive and file_path else file_path
    )
    if not path or not validate(path):
        raise LoftNotAccessible("loft path not accessible")
    try:
        with open(path, "rb") as handle:
            raw = handle.read(_MAX_LOFT_BYTES + 1)
    except OSError as exc:
        raise LoftUnreadable("loft unreadable") from exc
    if len(raw) > _MAX_LOFT_BYTES:
        return None
    try:
        body = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    provider, url = body.get("provider"), body.get("url")
    if not isinstance(provider, str) or not isinstance(url, str) or not url:
        return None
    return LoftSource(provider=provider, url=url)


_URL = re.compile(r"\S*://\S+")


def _scrub(line: str, source_url: str) -> str:
    """``line`` without URLs or the identifiers inside the stored URL."""
    out = _URL.sub("<url>", line)
    parts = urlsplit(source_url)
    tokens = {s for s in parts.path.split("/") if len(s) >= 6}
    for values in parse_qs(parts.query).values():
        tokens.update(v for v in values if len(v) >= 6)
    for token in sorted(tokens, key=len, reverse=True):
        out = out.replace(token, "<id>")
    return out[:300]


def run_lookup(url: str, limit: float) -> dict[str, Any]:
    """``yt-dlp -J --no-playlist <url>``, parsed.

    Raises :class:`LookupFailed` when ``yt-dlp`` is missing, fails, times out
    or prints something that is not a JSON object.
    """
    try:
        result = subprocess.run(
            ["yt-dlp", "-J", "--no-playlist", "--ignore-config", url],
            capture_output=True,
            text=True,
            timeout=limit,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        raise LookupFailed("yt-dlp is not installed") from None
    except subprocess.TimeoutExpired:
        raise LookupFailed("yt-dlp timed out") from None
    except OSError as exc:
        raise LookupFailed(type(exc).__name__) from None
    if result.returncode != 0:
        lines = [ln for ln in result.stderr.splitlines() if ln.strip()]
        last = _scrub(lines[-1].strip(), url) if lines else "no error output"
        raise LookupFailed(last)
    try:
        description = json.loads(result.stdout)
    except ValueError:
        raise LookupFailed("yt-dlp output is not JSON") from None
    if not isinstance(description, dict):
        raise LookupFailed("yt-dlp output is not an object")
    return description


def pick_format(description: dict[str, Any]) -> dict[str, Any] | None:
    """The video-only plain-HTTPS format with the greatest height up to 720.

    Among equal heights an H.264 stream wins, then an mp4 container.
    """
    best: dict[str, Any] | None = None
    best_key: tuple[int, int, int] | None = None
    formats = description.get("formats")
    if not isinstance(formats, list):
        return None
    for fmt in formats:
        if not isinstance(fmt, dict):
            continue
        height, url = fmt.get("height"), fmt.get("url")
        if (
            fmt.get("protocol") != "https"
            or fmt.get("acodec") != "none"
            or fmt.get("vcodec") in (None, "none")
            or not isinstance(url, str) or not url
            or isinstance(height, bool) or not isinstance(height, int)
            or not 0 < height <= _MAX_FRAME_HEIGHT
        ):
            continue
        key = (
            height,
            1 if str(fmt.get("vcodec")).startswith("avc1") else 0,
            1 if fmt.get("ext") == "mp4" else 0,
        )
        if best_key is None or key > best_key:
            best, best_key = fmt, key
    return best


_lock = threading.Lock()
_descriptions: dict[str, tuple[float, dict[str, Any]]] = {}
_lookups: dict[str, Future] = {}
_slots = threading.BoundedSemaphore(FFMPEG_SLOTS)


def _lookup_into(file_id: str, url: str, future: Future) -> None:
    try:
        description = run_lookup(url, LOOKUP_LIMIT_SECONDS)
    except Exception as exc:  # noqa: BLE001 - a waiter must always be released
        failure = exc if isinstance(exc, LookupFailed) else LookupFailed(type(exc).__name__)
        with _lock:
            if _lookups.get(file_id) is future:
                del _lookups[file_id]
        future.set_exception(failure)
        return
    with _lock:
        _descriptions[file_id] = (
            time.monotonic() + DESCRIPTION_TTL_SECONDS, description,
        )
        if _lookups.get(file_id) is future:
            del _lookups[file_id]
    future.set_result(description)


def _fail(file_id: str, status: int, detail: str, failure: str, extra: str = "") -> HTTPException:
    logger.warning(
        "youtube frame failed: file=%s class=%s%s",
        file_id, failure, f" ({extra})" if extra else "",
    )
    return HTTPException(status_code=status, detail=detail)


def _description(file_id: str, url: str, deadline: float) -> dict[str, Any]:
    with _lock:
        kept = _descriptions.get(file_id)
        if kept is not None:
            if time.monotonic() < kept[0]:
                return kept[1]
            del _descriptions[file_id]
        future = _lookups.get(file_id)
        if future is None:
            future = Future()
            _lookups[file_id] = future
            threading.Thread(
                target=_lookup_into, args=(file_id, url, future), daemon=True,
            ).start()
    try:
        return future.result(timeout=max(0.0, deadline - time.monotonic()))
    except FutureTimeout:
        raise _fail(file_id, 504, "Frame extraction timed out", "timeout") from None
    except LookupFailed as exc:
        raise _fail(
            file_id, 502, "Frame source unavailable", "lookup_failed", exc.detail,
        ) from None


def _drop_description(file_id: str) -> None:
    with _lock:
        _descriptions.pop(file_id, None)


def _header_option(headers: Any) -> tuple[str, ...]:
    if not isinstance(headers, dict):
        return ()
    lines = [
        f"{key}: {value}\r\n"
        for key, value in headers.items()
        if isinstance(key, str) and isinstance(value, str)
        and not re.search(r"[\r\n]", key + value)
    ]
    return ("-headers", "".join(lines)) if lines else ()


def ensure_youtube_frame(
    file_id: str, url: str, cache_path: Path, timestamp_seconds: float,
    size: FrameSize = "small",
) -> None:
    """Write the frame of the video at ``url`` to ``cache_path``, or raise.

    One deadline spans the lookup and ffmpeg. Raises ``HTTPException``: 502
    (no stream), 500 (ffmpeg), 504 (deadline).
    """
    deadline = time.monotonic() + FRAME_DEADLINE_SECONDS
    description = _description(file_id, url, deadline)
    fmt = pick_format(description)
    if fmt is None:
        _drop_description(file_id)
        raise _fail(file_id, 502, "Frame source unavailable", "no_format")

    if not _slots.acquire(timeout=max(0.0, deadline - time.monotonic())):
        raise _fail(file_id, 504, "Frame extraction timed out", "timeout")
    try:
        budget = deadline - time.monotonic()
        if budget < _MIN_FFMPEG_BUDGET_SECONDS:
            raise _fail(file_id, 504, "Frame extraction timed out", "timeout")
        _run_ffmpeg(file_id, fmt, cache_path, timestamp_seconds, size, budget)
    finally:
        _slots.release()


def _run_ffmpeg(
    file_id: str, fmt: dict[str, Any], cache_path: Path,
    timestamp_seconds: float, size: FrameSize, budget: float,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.{uuid.uuid4().hex}.tmp")
    argv = frame_ffmpeg_argv(
        fmt["url"], tmp_path, timestamp_seconds, size,
        _header_option(fmt.get("http_headers")),
    )
    try:
        result = subprocess.run(
            argv, capture_output=True, timeout=budget, stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        tmp_path.unlink(missing_ok=True)
        _drop_description(file_id)
        raise _fail(file_id, 504, "Frame extraction timed out", "timeout") from None
    except OSError:
        tmp_path.unlink(missing_ok=True)
        _drop_description(file_id)
        raise _fail(file_id, 500, "Frame extraction failed", "ffmpeg_failed") from None
    if result.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        _drop_description(file_id)
        raise _fail(
            file_id, 500, "Frame extraction failed", "ffmpeg_failed",
            f"returncode={result.returncode}",
        )
    os.replace(tmp_path, cache_path)


@dataclass(frozen=True)
class Storyboard:
    """The ``sb0`` format: sheets of ``rows * columns`` tiles, in time order."""

    sheets: tuple[tuple[str, float], ...]
    rows: int
    columns: int
    tile_width: int
    tile_height: int


@dataclass(frozen=True)
class Tile:
    sheet: int
    cell: int
    timestamp: float


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


def storyboard_of(description: dict[str, Any]) -> Storyboard | None:
    """The ``sb0`` format whatever tile size it declares; ``None`` when unusable."""
    formats = description.get("formats")
    if not isinstance(formats, list):
        return None
    for fmt in formats:
        if not isinstance(fmt, dict) or fmt.get("format_id") != "sb0":
            continue
        rows, columns = _positive_int(fmt.get("rows")), _positive_int(fmt.get("columns"))
        width, height = _positive_int(fmt.get("width")), _positive_int(fmt.get("height"))
        fragments = fmt.get("fragments")
        if not (rows and columns and width and height) or not isinstance(fragments, list):
            return None
        sheets: list[tuple[str, float]] = []
        for fragment in fragments:
            if not isinstance(fragment, dict):
                return None
            url, seconds = fragment.get("url"), _positive_number(fragment.get("duration"))
            if not isinstance(url, str) or not url or seconds is None:
                return None
            sheets.append((url, seconds))
        if not sheets:
            return None
        return Storyboard(tuple(sheets), rows, columns, width, height)
    return None


def known_duration(description: dict[str, Any], indexed: float | None) -> float | None:
    """The lookup's duration, else the indexed one; ``None`` when neither is above 0."""
    return _positive_number(description.get("duration")) or _positive_number(indexed)


def kept_tiles(board: Storyboard, duration: float, max_frames: int) -> list[Tile]:
    """The tiles below ``duration``, in time order, spread for at most ``max_frames``.

    The first tile is always kept; a later one only when it is at least
    ``duration / max_frames`` after the last kept one.
    """
    spacing = duration / max(1, max_frames)
    per_sheet = board.rows * board.columns
    kept: list[Tile] = []
    start = 0.0
    for sheet, (_, seconds) in enumerate(board.sheets):
        for cell in range(per_sheet):
            timestamp = start + cell * seconds / per_sheet
            if timestamp >= duration:
                continue
            if kept and timestamp - kept[-1].timestamp < spacing - 1e-9:
                continue
            kept.append(Tile(sheet, cell, timestamp))
        start += seconds
    return kept


__all__ = [
    "LoftNotAccessible",
    "LoftSource",
    "LoftUnreadable",
    "LookupFailed",
    "STORYBOARD_LOOKUP_LIMIT_SECONDS",
    "Storyboard",
    "Tile",
    "YOUTUBE_PROVIDER",
    "ensure_youtube_frame",
    "kept_tiles",
    "known_duration",
    "storyboard_of",
    "pick_format",
    "read_loft",
    "run_lookup",
]
