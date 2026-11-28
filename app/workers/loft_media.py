"""One fetch of a `.loft` reference's video.

The reference is read and classified, its video is looked up and downloaded
once at up to 720p, its scene candidates are built from that file when they
are still the storyboard, the frame scan and frame OCR run on it exactly as on
a native video, and the download is deleted again. Nothing is fetched for a
reference whose provider is generic, or that has been given up for having no
usable video format.

Neither the stored URL, a signed stream URL nor a request header reaches a log
line: only the file id and the failure class below do.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from app import youtube_source
from app.config import settings
from app.database import get_search_db, get_search_db_read
from app.frame_cache import extract_frame_to_cache, frame_cache_path, tile_cache_dir
from app.models import Embedding, IndexedFile
from app.policy_client import is_feature_enabled
from app.workers.clip import (
    _extract_frame_paths,
    _remove_clip_embeddings,
    _store_clip_embedding,
    candidates_gate_holds,
    embed_image,
    index_storyboard,
    load_image_model,
    storyboard_lock,
)
from app.workers.frame_ocr import index_frame_ocr
from app.workers.frame_scan import index_frame_scan
from app.workers.metadata import _remove_embeddings
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
PAUSE_SECONDS = 30.0
BACKOFF_START_SECONDS = 30 * 60.0
BACKOFF_MAX_SECONDS = 6 * 60 * 60.0
#: Seconds the candidates stage may spend embedding the video's frames.
EMBED_BUDGET_SECONDS = 600.0
#: The span a scene candidate keeps, as the native leg writes it.
SCENE_SECONDS = 30.0
CHANGE_TYPE = "clip_change"
_PAUSE_POLL_SECONDS = 0.2

#: File ids a `frame_ocr` reindex or prioritize asked for; a restart forgets it.
_OCR_REQUESTED: set[str] = set()
#: File ids whose fetch is running right now.
_RUNNING: set[str] = set()
#: Per file, the reindex or regenerate requests that arrived during its fetch.
_REQUESTED_DURING_RUN: dict[str, set[str]] = {}
#: Consecutive lookup or download failures across files; a success resets it.
_failure_run = 0
#: The length of the next backoff; doubled as each one starts.
_next_backoff_seconds = BACKOFF_START_SECONDS
#: Monotonic time a backoff ends, or ``None`` when none is in effect.
_backoff_until: float | None = None
#: The previous fetch result, of any file, and when it was recorded.
_last_result: str | None = None
_last_result_at: float | None = None


@dataclass(frozen=True)
class FetchResult:
    """What one fetch did, for the queue worker that ran it."""

    scan_set: bool = False
    candidates_set: bool = False
    settled: bool = False
    requeue: bool = False
    retry_backlog: bool = False


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
    global _failure_run, _next_backoff_seconds, _backoff_until
    global _last_result, _last_result_at
    _failure_run = 0
    _next_backoff_seconds = BACKOFF_START_SECONDS
    _backoff_until = None
    _last_result = None
    _last_result_at = None
    _RUNNING.clear()
    _REQUESTED_DURING_RUN.clear()


def backlog_held() -> bool:
    """Whether a backlog fetch must wait for the pause or a backoff."""
    now = time.monotonic()
    if _backoff_until is not None and now < _backoff_until:
        return True
    return (
        _last_result_at is not None
        and now - _last_result_at < PAUSE_SECONDS
    )


def _note_result(kind: str) -> bool:
    """Record a fetch result and say whether a failure counts.

    A failure counts only when the previous result, of any file, was a
    success: a run of refusals is one episode, not one failure per video.
    """
    global _failure_run, _next_backoff_seconds, _backoff_until
    global _last_result, _last_result_at
    now = time.monotonic()
    _last_result_at = now
    if kind == "success":
        _last_result = "success"
        _failure_run = 0
        _backoff_until = None
        _next_backoff_seconds = BACKOFF_START_SECONDS
        return False

    counted = _last_result == "success"
    _last_result = "failure"
    _failure_run += 1
    if _failure_run >= FAILURE_CAP:
        minutes = int(_next_backoff_seconds // 60)
        logger.info(
            "loft_media backing off %d min after %d failures in a row",
            minutes, _failure_run,
        )
        _backoff_until = now + _next_backoff_seconds
        _next_backoff_seconds = min(
            _next_backoff_seconds * 2, BACKOFF_MAX_SECONDS,
        )
    return counted


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
    candidates_done: bool
    loft_source: str | None


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
            candidates_done=bool(row.loft_candidates_done),
            loft_source=row.loft_source,
        )


class _NoLongerALoft(Exception):
    """The row's mime stopped being the `.loft` mime while its task ran."""


def _skip_no_longer_loft(file_id: str) -> None:
    logger.info("loft task skipped, no longer a .loft (file %s)", file_id)
    raise _NoLongerALoft


def _write(file_id: str, values: dict) -> None:
    with get_search_db() as session:
        row = session.query(IndexedFile).filter_by(
            file_id=file_id, active=True
        ).first()
        if row is None:
            return
        if row.mime_type != LOFT_MIME:
            _skip_no_longer_loft(file_id)
        for column, value in values.items():
            setattr(row, column, value)


def _flag(file_id: str, column: str) -> bool:
    with get_search_db_read() as session:
        row = session.query(IndexedFile).filter_by(
            file_id=file_id, active=True
        ).first()
        return row is not None and bool(getattr(row, column))


def _has_candidates(file_id: str) -> bool:
    """Whether the reference already has storyboard rows."""
    with get_search_db_read() as session:
        return (
            session.query(Embedding)
            .filter_by(file_id=file_id, embedding_type="clip")
            .first()
            is not None
        )


def _storyboard_step(
    state: _State, description: dict, candidates_wanted: bool,
) -> None:
    """Read the storyboard the lookup names, for a reference that has none.

    Best effort: a failure writes no candidate and changes nothing the fetch
    counts, flags or reports, so the download and the stages still run. The
    step is the fetch's own write, not the CLIP leg's, so it neither consults
    nor records that leg's retry gate. The model is loaded here unless the
    candidates stage, which is about to replace these rows, has loaded it.
    """
    file_id = state.file_id
    try:
        with storyboard_lock(file_id):
            # Checked under the lock: a leg finishing first may have just left
            # the reference with the very candidates this step would replace.
            if candidates_gate_holds(file_id):
                return
            if _has_candidates(file_id):
                return
            if not candidates_wanted:
                try:
                    load_image_model()
                except Exception:
                    return
            index_storyboard(file_id, description, state.duration)
    except Exception as exc:  # noqa: BLE001 - the fetch goes on regardless
        logger.warning(
            "youtube storyboard failed: file=%s class=%s",
            file_id, type(exc).__name__,
        )


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


class _CandidatesFailure(Exception):
    """A candidates stage that ends without a result; ``failure`` is the log class."""

    def __init__(self, failure: str) -> None:
        super().__init__(failure)
        self.failure = failure


def _embed_frames(
    frames: list[tuple[float, Path]], is_paused: Callable[[], bool] | None,
) -> list[tuple[float, np.ndarray]]:
    """Embed every frame that reads, holding the vectors instead of storing them."""
    deadline = time.monotonic() + EMBED_BUDGET_SECONDS
    batch_size = settings.workers.clip_frame_batch_size
    embedded: list[tuple[float, np.ndarray]] = []
    for start in range(0, len(frames), batch_size):
        if is_paused is not None:
            paused_at = time.monotonic()
            while is_paused():
                time.sleep(_PAUSE_POLL_SECONDS)
            deadline += time.monotonic() - paused_at
        for timestamp, frame_path in frames[start:start + batch_size]:
            if time.monotonic() >= deadline:
                raise _CandidatesFailure("BudgetExceeded")
            try:
                image = Image.open(str(frame_path)).convert("RGB")
                image.load()
                vector = embed_image(image)
                del image
            except Exception:  # noqa: BLE001 - one frame is not the stage
                continue
            embedded.append((timestamp, vector))
    return embedded


def _write_candidates(
    file_id: str, embedded: list[tuple[float, np.ndarray]],
) -> str:
    """Replace the file's scene candidates, or say why nothing was written."""
    with storyboard_lock(file_id):
        with get_search_db() as session:
            row = session.query(IndexedFile).filter_by(
                file_id=file_id, active=True
            ).first()
            if row is None:
                return "inactive"
            if row.mime_type != LOFT_MIME:
                _skip_no_longer_loft(file_id)
            _remove_clip_embeddings(session, file_id)
            _remove_embeddings(session, file_id, CHANGE_TYPE)
            for timestamp, vector in embedded:
                _store_clip_embedding(
                    session=session,
                    embedding_id=f"clip_{file_id}_{uuid.uuid4().hex[:8]}",
                    file_id=file_id,
                    vector=vector,
                    content_preview=f"Frame at {timestamp:.1f}s",
                    timestamp_start=timestamp,
                    timestamp_end=timestamp + SCENE_SECONDS,
                )
            row.loft_candidates_done = True
            row.clip_indexed = True
        _clear_tiles(file_id)
        from app.search import invalidate_similar_cache

        invalidate_similar_cache()
    return "committed"


def _clear_tiles(file_id: str) -> None:
    """Drop the storyboard tiles the committed rows replace."""
    directory = tile_cache_dir(file_id)
    if not directory.exists():
        return
    try:
        shutil.rmtree(directory)
    except OSError as exc:
        logger.warning(
            "loft_media tile removal failed for %s (%s)", file_id, type(exc).__name__,
        )


def _discard_entries(written: list[Path]) -> None:
    for path in written:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _stored_duration(file_id: str) -> float | None:
    with get_search_db_read() as session:
        value = session.query(IndexedFile.duration).filter_by(
            file_id=file_id, active=True
        ).scalar()
    return None if value is None else float(value)


def _candidates_stage(
    state: _State, source_path: Path, is_paused: Callable[[], bool] | None,
) -> str:
    """Build the `.loft`'s scene candidates from its downloaded video.

    Returns ``"committed"``, ``"failed"`` or ``"inactive"``. A stage that does
    not commit removes the cache entries it wrote; a committed one keeps them,
    because a scene-tile serves them for the video's frames.
    """
    file_id = state.file_id
    written: list[Path] = []
    try:
        duration = _probe_duration(str(source_path)) or _stored_duration(file_id)
        with tempfile.TemporaryDirectory(prefix="loft_frames_") as tmp:
            frames = _extract_frame_paths(str(source_path), duration, Path(tmp))
            embedded = _embed_frames(frames, is_paused)
            if not embedded:
                raise _CandidatesFailure("NoFrames")
            for timestamp, _vector in embedded:
                path = frame_cache_path(file_id, timestamp)
                if path.exists() and path.stat().st_size > 0:
                    continue
                try:
                    extract_frame_to_cache(str(source_path), path, timestamp)
                except Exception:  # noqa: BLE001 - a missing picture is not a failure
                    continue
                written.append(path)
        outcome = _write_candidates(file_id, embedded)
    except _NoLongerALoft:
        _discard_entries(written)
        raise
    except _CandidatesFailure as exc:
        _discard_entries(written)
        logger.warning("loft_media candidates failed for %s (%s)", file_id, exc.failure)
        return "failed"
    except Exception as exc:  # noqa: BLE001 - one stage is not the fetch
        _discard_entries(written)
        logger.warning(
            "loft_media candidates failed for %s (%s)", file_id, type(exc).__name__,
        )
        return "failed"
    if outcome != "committed":
        _discard_entries(written)
    return outcome


def _count_failure(
    state: _State, values: dict, *, close_scan: bool, close_ocr: bool,
    close_candidates: bool = False,
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
        if close_candidates:
            values["loft_candidates_done"] = True
    _write(state.file_id, values)


def _stages_wanted(state: _State) -> tuple[bool, bool, bool]:
    mode = settings.features.video_frame_ocr
    scan = not state.frame_scan_indexed
    ocr = (
        not state.frame_ocr_read
        and mode != "false"
        and (mode == "on_index" or state.file_id in _OCR_REQUESTED)
    )
    candidates = not state.candidates_done
    return scan, ocr, candidates


def index_loft_media(
    file_id: str, *, is_paused: Callable[[], bool] | None = None,
) -> FetchResult:
    """Fetch and process one `.loft` reference's video.

    Reads and classifies the reference, then (for a reference with a video
    provider) looks the video up, downloads the picked rendition, runs the
    wanted frame stages on it and deletes it. A counted failure increments the
    file's failure count; at the cap the failed stages are closed with no rows.
    """
    state = _load_state(file_id)
    if state is None or state.mime_type != LOFT_MIME:
        return FetchResult()

    _RUNNING.add(file_id)
    result = FetchResult()
    try:
        try:
            result = _fetch(state, is_paused)
        except _NoLongerALoft:
            result = FetchResult()
    finally:
        _RUNNING.discard(file_id)
        try:
            requeue = _apply_recorded_requests(file_id)
        except _NoLongerALoft:
            requeue = False
    return FetchResult(
        scan_set=result.scan_set, candidates_set=result.candidates_set,
        settled=result.settled, requeue=requeue,
        retry_backlog=result.retry_backlog,
    )


def _failed_result(
    state: _State, *, scan_wanted: bool, ocr_wanted: bool,
    candidates_wanted: bool,
) -> FetchResult:
    """Count a failure after a success, or send the file back to the backlog.

    A candidates-only fetch never moves the count; a counted failure closes its
    candidates flag instead, so one user-requested retry is all it gets.
    """
    if not _note_result("failure"):
        return FetchResult(retry_backlog=True)
    if not scan_wanted and not ocr_wanted:
        _write(state.file_id, {"loft_candidates_done": True})
        return FetchResult()
    _count_failure(
        state, {}, close_scan=scan_wanted, close_ocr=ocr_wanted,
        close_candidates=candidates_wanted,
    )
    return FetchResult()


def _fetch(
    state: _State, is_paused: Callable[[], bool] | None,
) -> FetchResult:
    file_id = state.file_id
    try:
        source = youtube_source.read_loft(state.file_path, state.drive)
    except (youtube_source.LoftNotAccessible, youtube_source.LoftUnreadable):
        return FetchResult()

    if source is None or source.provider == youtube_source.GENERIC_PROVIDER:
        _write(file_id, {
            "loft_source": "other",
            "frame_scan_indexed": True,
            "frame_ocr_read": True,
            "frame_ocr_indexed": True,
            "loft_candidates_done": True,
        })
        return FetchResult()

    if source.provider == youtube_source.YOUTUBE_PROVIDER:
        loft_source = "youtube"
    else:
        if state.loft_source == "other":
            return FetchResult()
        loft_source = "web"

    _write(file_id, {"loft_source": loft_source})

    scan_wanted, ocr_wanted, candidates_wanted = _stages_wanted(state)
    if not scan_wanted and not ocr_wanted and not candidates_wanted:
        return FetchResult()
    candidates_only = candidates_wanted and not scan_wanted and not ocr_wanted

    try:
        permitted = _policy_allows(state.drive)
    except Exception:
        permitted = True
    if not permitted:
        return FetchResult()

    if candidates_wanted:
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
        return _failed_result(
            state, scan_wanted=scan_wanted, ocr_wanted=ocr_wanted,
            candidates_wanted=candidates_wanted,
        )

    lookup_duration = youtube_source.known_duration(description, None)
    if state.duration is None and lookup_duration is not None:
        _write(file_id, {"duration": lookup_duration})

    if loft_source == "youtube":
        _storyboard_step(state, description, candidates_wanted)

    fmt = youtube_source.pick_format(description)
    if fmt is None:
        logger.warning("loft_media no rendition (file %s)", file_id)
        # The site answered, so the run of refusals ends; a video with nothing
        # to fetch counts against a YouTube reference, while a web one is
        # closed as having no usable format.
        _note_result("success")
        if loft_source == "web":
            _write(file_id, {
                "loft_source": "other",
                "frame_scan_indexed": True,
                "frame_ocr_read": True,
                "frame_ocr_indexed": True,
                "loft_candidates_done": True,
            })
            return FetchResult()
        if candidates_only:
            _write(file_id, {"loft_candidates_done": True})
            return FetchResult()
        _count_failure(
            state, {}, close_scan=scan_wanted, close_ocr=ocr_wanted,
            close_candidates=candidates_wanted,
        )
        return FetchResult()

    if not _space_allows(file_id, fmt):
        return FetchResult()

    directory = fetch_dir()
    directory.mkdir(parents=True, exist_ok=True)

    def _free() -> int:
        return shutil.disk_usage(str(directory)).free

    try:
        outcome = youtube_source.download_rendition(
            source.url, directory, file_id,
            time_limit=_time_limit(lookup_duration),
            free_bytes=_free,
        )
    except Exception:
        outcome = None
    if outcome is None or outcome.path is None:
        if outcome is not None and outcome.no_space:
            return FetchResult()
        logger.warning("loft_media download failed (file %s)", file_id)
        return _failed_result(
            state, scan_wanted=scan_wanted, ocr_wanted=ocr_wanted,
            candidates_wanted=candidates_wanted,
        )
    destination = outcome.path

    try:
        if lookup_duration is not None:
            probed = _probe_duration(str(destination))
            if probed is None or probed < lookup_duration * DURATION_TOLERANCE:
                logger.warning("loft_media download failed (file %s)", file_id)
                return _failed_result(
                    state, scan_wanted=scan_wanted, ocr_wanted=ocr_wanted,
                    candidates_wanted=candidates_wanted,
                )

        if is_paused is not None:
            # Discarding the download here would only fetch it again later.
            while is_paused():
                time.sleep(_PAUSE_POLL_SECONDS)

        ran_scan, ran_ocr, ran_candidates = False, False, False
        scan_set = False
        ocr_set = False
        candidates_set = False

        if candidates_wanted:
            ran_candidates = True
            candidates_outcome = _candidates_stage(state, destination, is_paused)
            candidates_set = candidates_outcome == "committed"
            if candidates_outcome == "inactive":
                return FetchResult()

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

        if ran_ocr:
            _OCR_REQUESTED.discard(file_id)

        settled = (
            (not ran_scan or scan_set)
            and (not ran_ocr or ocr_set)
            and (not ran_candidates or candidates_set)
        )
        if candidates_only:
            if not candidates_set:
                _write(file_id, {"loft_candidates_done": True})
            _note_result("success")
            return FetchResult(candidates_set=candidates_set)

        if settled:
            _write(file_id, {"loft_fetch_failures": 0})
        else:
            _count_failure(
                state,
                {},
                close_scan=scan_wanted and not scan_set,
                close_ocr=ocr_wanted and not ocr_set,
                close_candidates=ran_candidates and not candidates_set,
            )

        _note_result("success")
        return FetchResult(
            scan_set=scan_set, candidates_set=candidates_set, settled=settled,
        )
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
        values["loft_candidates_done"] = False
    _write(file_id, values)
    return True


__all__ = [
    "FetchResult",
    "backlog_held",
    "clear_fetch_dir",
    "fetch_dir",
    "index_loft_media",
    "is_fetch_running",
    "is_ocr_requested",
    "mark_ocr_requested",
    "record_request_during_run",
    "reset_worker_state",
]
