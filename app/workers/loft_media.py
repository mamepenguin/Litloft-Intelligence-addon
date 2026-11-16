"""One fetch of a `.loft` reference's video.

The reference is read and classified, its video is looked up and downloaded
once at up to 720p without audio, the frame scan and frame OCR run on that
file exactly as on a native video, and the download is deleted again. Nothing
is fetched for a reference that turns out not to point at a YouTube video.

Neither the stored URL, a signed stream URL nor a request header reaches a log
line: only the file id and the failure class below do.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app import youtube_source
from app.config import settings
from app.database import get_search_db, get_search_db_read
from app.models import IndexedFile
from app.policy_client import is_feature_enabled
from app.workers.clip import load_image_model
from app.workers.frame_ocr import index_frame_ocr
from app.workers.frame_scan import index_frame_scan
from app.workers.ocr import _get_engine
from app.workers.whisper import LOFT_MIME

logger = logging.getLogger(__name__)

FETCH_SUBDIR = "loft_media"
FAILURE_CAP = 3
LOOKUP_LIMIT_SECONDS = 60.0
UNKNOWN_DURATION_TIME_LIMIT = 7200.0
MIN_TIME_LIMIT = 600.0
KNOWN_SIZE_RESERVE = 2 * 1024 ** 3
UNKNOWN_SIZE_RESERVE = 4 * 1024 ** 3
DURATION_TOLERANCE = 0.95
_PAUSE_POLL_SECONDS = 0.2

#: File ids a `frame_ocr` reindex or prioritize asked for; a restart forgets it.
_OCR_REQUESTED: set[str] = set()
#: File ids whose fetch is running right now.
_RUNNING: set[str] = set()
#: Per file, the reindex or regenerate requests that arrived during its fetch.
_REQUESTED_DURING_RUN: dict[str, set[str]] = {}
#: Whether a lookup has succeeded in this process, for the counted-failure rule.
_lookup_succeeded = False


@dataclass(frozen=True)
class FetchResult:
    """What one fetch did, for the queue worker that ran it."""

    scan_set: bool = False
    settled: bool = False
    requeue: bool = False


def fetch_dir() -> Path:
    """The one directory a fetch downloads into."""
    return Path(settings.intelligence_data_dir) / FETCH_SUBDIR


def clear_fetch_dir() -> int:
    """Remove every file a killed fetch left behind.

    Called at startup before any fetch can be queued, so a `.part` or a
    `.mkv` from the previous process can never be mistaken for this one's.
    """
    directory = fetch_dir()
    if not directory.is_dir():
        return 0
    removed = 0
    for entry in directory.iterdir():
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            removed += 1
        except OSError as exc:
            logger.warning(
                "loft_media cleanup failed for one entry (%s)", type(exc).__name__,
            )
    return removed


def reset_worker_state() -> None:
    """Forget the in-memory state of a fetch worker that is starting."""
    global _lookup_succeeded
    _lookup_succeeded = False
    _RUNNING.clear()
    _REQUESTED_DURING_RUN.clear()


def is_fetch_running(file_id: str) -> bool:
    """Whether ``file_id``'s fetch is running in this process."""
    return file_id in _RUNNING


def record_request_during_run(file_id: str, tasks: set[str]) -> None:
    """Record a reindex or regenerate that arrived while the fetch runs."""
    _REQUESTED_DURING_RUN.setdefault(file_id, set()).update(tasks)


def mark_ocr_requested(file_id: str) -> None:
    """Remember that the next fetch of ``file_id`` must read its frames."""
    _OCR_REQUESTED.add(file_id)


def is_ocr_requested(file_id: str) -> bool:
    """Whether ``file_id`` is waiting for an on-demand OCR read."""
    return file_id in _OCR_REQUESTED


@dataclass
class _State:
    file_id: str
    drive: str
    file_path: str
    mime_type: str
    duration: float | None
    frame_scan_indexed: bool
    frame_ocr_read: bool
    frame_ocr_indexed: bool
    failures: int


def _load_state(file_id: str) -> _State | None:
    with get_search_db_read() as session:
        row = session.query(IndexedFile).filter_by(
            file_id=file_id, active=True
        ).first()
        if row is None:
            return None
        return _State(
            file_id=row.file_id,
            drive=row.drive,
            file_path=row.file_path,
            mime_type=row.mime_type,
            duration=row.duration,
            frame_scan_indexed=bool(row.frame_scan_indexed),
            frame_ocr_read=bool(row.frame_ocr_read),
            frame_ocr_indexed=bool(row.frame_ocr_indexed),
            failures=int(row.loft_fetch_failures or 0),
        )


def _write(file_id: str, values: dict) -> None:
    with get_search_db() as session:
        row = session.query(IndexedFile).filter_by(
            file_id=file_id, active=True
        ).first()
        if row is None:
            return
        for column, value in values.items():
            setattr(row, column, value)


def _flag(file_id: str, column: str) -> bool:
    with get_search_db_read() as session:
        row = session.query(IndexedFile).filter_by(
            file_id=file_id, active=True
        ).first()
        return row is not None and bool(getattr(row, column))


def _policy_allows(drive: str) -> bool:
    """Whether the drive has the ``index`` feature on.

    The policy client is async and a fetch runs off the loop, so a coroutine
    verdict is run on a loop of its own. A stand-in that answers directly is
    taken as it is.
    """
    verdict = is_feature_enabled(drive, "index", default_on_failure=True)
    if inspect.isawaitable(verdict):
        verdict = asyncio.run(verdict)
    return bool(verdict)


def _positive_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _probe_duration(path: str) -> float | None:
    """The container duration ffprobe reports, or ``None`` when unreadable."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", path,
            ],
            capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        value = float(result.stdout.strip())
    except ValueError:
        return None
    return _positive_number(value)


def _time_limit(duration: float | None) -> float:
    if duration is None:
        return UNKNOWN_DURATION_TIME_LIMIT
    return max(MIN_TIME_LIMIT, duration * 1.5)


def _count_failure(
    state: _State, values: dict, *, close_scan: bool, close_ocr: bool,
) -> None:
    """Add one counted failure and close the failed stage at the cap."""
    failures = state.failures + 1
    values["loft_fetch_failures"] = failures
    if failures >= FAILURE_CAP:
        if close_scan:
            values["frame_scan_indexed"] = True
        if close_ocr:
            values["frame_ocr_read"] = True
            values["frame_ocr_indexed"] = True
    _write(state.file_id, values)


def _stages_wanted(state: _State) -> tuple[bool, bool]:
    mode = settings.features.video_frame_ocr
    scan = not state.frame_scan_indexed
    ocr = (
        not state.frame_ocr_read
        and mode != "false"
        and (mode == "on_index" or state.file_id in _OCR_REQUESTED)
    )
    return scan, ocr


def index_loft_media(
    file_id: str, *, is_paused: Callable[[], bool] | None = None,
) -> FetchResult:
    """Fetch and process one `.loft` reference's video.

    Reads and classifies the reference, then (for YouTube) looks the video up,
    downloads the picked rendition, runs the wanted frame stages on it and
    deletes it. A counted failure increments the file's failure count; at the
    cap the failed stages are closed with no rows.
    """
    state = _load_state(file_id)
    if state is None or state.mime_type != LOFT_MIME:
        return FetchResult()

    _RUNNING.add(file_id)
    result = FetchResult()
    try:
        result = _fetch(state, is_paused)
    finally:
        _RUNNING.discard(file_id)
        requeue = _apply_recorded_requests(file_id)
    return FetchResult(
        scan_set=result.scan_set, settled=result.settled, requeue=requeue,
    )


def _fetch(
    state: _State, is_paused: Callable[[], bool] | None,
) -> FetchResult:
    global _lookup_succeeded

    file_id = state.file_id
    try:
        source = youtube_source.read_loft(state.file_path, state.drive)
    except (youtube_source.LoftNotAccessible, youtube_source.LoftUnreadable):
        return FetchResult()

    if source is None or source.provider != youtube_source.YOUTUBE_PROVIDER:
        _write(file_id, {
            "loft_source": "other",
            "frame_scan_indexed": True,
            "frame_ocr_read": True,
            "frame_ocr_indexed": True,
        })
        return FetchResult()

    _write(file_id, {"loft_source": "youtube"})

    scan_wanted, ocr_wanted = _stages_wanted(state)
    if not scan_wanted and not ocr_wanted:
        return FetchResult()

    try:
        permitted = _policy_allows(state.drive)
    except Exception:
        permitted = True
    if not permitted:
        return FetchResult()

    if scan_wanted:
        try:
            load_image_model()
        except Exception:
            return FetchResult()
    if ocr_wanted:
        try:
            if _get_engine() is None:
                return FetchResult()
        except Exception:
            return FetchResult()

    if shutil.which("yt-dlp") is None:
        # Not this video's failure: nothing was looked up.
        logger.warning("loft_media lookup unavailable (file %s)", file_id)
        return FetchResult()

    try:
        description = youtube_source.run_lookup(
            source.url, LOOKUP_LIMIT_SECONDS,
        )
    except Exception:
        logger.warning("loft_media lookup failed (file %s)", file_id)
        if _lookup_succeeded:
            _count_failure(
                state, {}, close_scan=scan_wanted, close_ocr=ocr_wanted,
            )
        return FetchResult()
    _lookup_succeeded = True

    lookup_duration = youtube_source.known_duration(description, None)
    if state.duration is None and lookup_duration is not None:
        _write(file_id, {"duration": lookup_duration})

    fmt = youtube_source.pick_format(description)
    if fmt is None:
        logger.warning("loft_media no rendition (file %s)", file_id)
        _count_failure(state, {}, close_scan=scan_wanted, close_ocr=ocr_wanted)
        return FetchResult()

    if not _space_allows(file_id, fmt):
        return FetchResult()

    directory = fetch_dir()
    directory.mkdir(parents=True, exist_ok=True)

    def _free() -> int:
        return shutil.disk_usage(str(directory)).free

    try:
        destination = youtube_source.download_rendition(
            source.url, directory, file_id,
            time_limit=_time_limit(lookup_duration),
            free_bytes=_free,
        )
    except Exception:
        destination = None
    if destination is None:
        logger.warning("loft_media download failed (file %s)", file_id)
        _count_failure(state, {}, close_scan=scan_wanted, close_ocr=ocr_wanted)
        return FetchResult()

    try:
        if lookup_duration is not None:
            probed = _probe_duration(str(destination))
            if probed is None or probed < lookup_duration * DURATION_TOLERANCE:
                logger.warning("loft_media download failed (file %s)", file_id)
                _count_failure(
                    state, {}, close_scan=scan_wanted, close_ocr=ocr_wanted,
                )
                return FetchResult()

        if is_paused is not None:
            # Discarding the download here would only fetch it again later.
            while is_paused():
                time.sleep(_PAUSE_POLL_SECONDS)

        ran_scan, ran_ocr = False, False
        scan_set = False
        ocr_set = False

        if scan_wanted:
            ran_scan = True
            try:
                index_frame_scan(
                    file_id, is_paused=is_paused, source_path=str(destination),
                )
            except Exception as exc:
                logger.error(
                    "loft_media scan failed for %s (%s)", file_id, type(exc).__name__,
                )
            scan_set = _flag(file_id, "frame_scan_indexed")

        if ocr_wanted:
            ran_ocr = True
            try:
                index_frame_ocr(
                    file_id, is_paused=is_paused, source_path=str(destination),
                )
            except Exception as exc:
                logger.error(
                    "loft_media OCR failed for %s (%s)", file_id, type(exc).__name__,
                )
            ocr_set = _flag(file_id, "frame_ocr_read")

        settled = (not ran_scan or scan_set) and (not ran_ocr or ocr_set)
        if settled:
            _write(file_id, {"loft_fetch_failures": 0})
        else:
            _count_failure(
                state,
                {},
                close_scan=scan_wanted and not scan_set,
                close_ocr=ocr_wanted and not ocr_set,
            )

        if ran_ocr:
            _OCR_REQUESTED.discard(file_id)

        return FetchResult(scan_set=scan_set, settled=settled)
    finally:
        destination.unlink(missing_ok=True)


def _space_allows(file_id: str, fmt: dict) -> bool:
    """The disk guard before a download; a refusal is not counted."""
    directory = fetch_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(str(directory)).free
    except OSError:
        free = 0
    size = _positive_int(fmt.get("filesize")) or _positive_int(
        fmt.get("filesize_approx")
    )
    required = size + KNOWN_SIZE_RESERVE if size is not None else UNKNOWN_SIZE_RESERVE
    if free < required:
        logger.warning("loft_media no space (file %s)", file_id)
        return False
    return True


def _apply_recorded_requests(file_id: str) -> bool:
    """Apply requests that arrived during the fetch; ``True`` means fetch again."""
    tasks = _REQUESTED_DURING_RUN.pop(file_id, None)
    if not tasks:
        return False
    values: dict = {"loft_fetch_failures": 0}
    if "frame_ocr" in tasks:
        _OCR_REQUESTED.add(file_id)
        values["frame_ocr_read"] = False
        values["frame_ocr_indexed"] = False
    if "clip" in tasks:
        values["loft_source"] = None
        values["frame_scan_indexed"] = False
    _write(file_id, values)
    return True


__all__ = [
    "FetchResult",
    "clear_fetch_dir",
    "fetch_dir",
    "index_loft_media",
    "is_fetch_running",
    "is_ocr_requested",
    "mark_ocr_requested",
    "record_request_during_run",
    "reset_worker_state",
]
