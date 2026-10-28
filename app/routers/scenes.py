"""``GET /files/{file_id}/scenes``: representative scenes without a model.

Runs the Visual Index's deterministic selection over the scene-CLIP
candidates the indexer already stored, so any indexed native video gets a
scene list whether or not a Visual Index was generated. Nothing is
written and no LLM is called.
"""

from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException, Query

from app.drive_context import require_drive
from app.routers.files import _get_indexed_file_or_404
from app.schemas import VideoSceneItem, VideoScenesResponse, VideoScenesWindow
from app.workers.video_visual import _load_candidates, _select_transcript_excerpt
from app.workers.video_visual_selection import Candidate, select_candidates
from app.workers.whisper import LOFT_MIME

router = APIRouter(tags=["scenes"])


def _check_bound(name: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value < 0):
        raise HTTPException(
            status_code=422, detail=f"{name} must be a finite number >= 0"
        )


def _resolve_window(
    start: float | None, end: float | None, duration: float | None,
) -> tuple[VideoScenesWindow | None, float | None, float | None, bool]:
    """Return ``(window, upper bound, budget duration, empty)``.

    ``upper bound`` of ``None`` means no upper bound. ``empty`` is true
    when the window lies wholly at or past the known duration.
    """
    if start is None and end is None:
        return None, None, duration, False
    lo = start if start is not None else 0.0
    if end is not None:
        return VideoScenesWindow(start=lo, end=end), end, end - lo, False
    if duration is not None:
        window = VideoScenesWindow(start=lo, end=duration)
        return window, duration, duration - lo, lo >= duration
    return VideoScenesWindow(start=lo, end=None), None, None, False


def _last_end(
    last_start: float, window: VideoScenesWindow | None, duration: float | None,
) -> float | None:
    if window is not None:
        return window.end
    if duration is not None and duration > last_start:
        return duration
    return None


@router.get("/files/{file_id}/scenes", response_model=VideoScenesResponse)
def get_scenes(
    file_id: str,
    start: float | None = Query(default=None),
    end: float | None = Query(default=None),
    drive: str = Depends(require_drive),
) -> VideoScenesResponse:
    _check_bound("start", start)
    _check_bound("end", end)
    if start is not None and end is not None and end <= start:
        raise HTTPException(status_code=422, detail="end must be greater than start")

    indexed = _get_indexed_file_or_404(file_id, drive)
    if indexed["file_type"] != "video" or indexed["mime_type"] == LOFT_MIME:
        raise HTTPException(status_code=400, detail="Not a video file")

    candidates, duration = _load_candidates(file_id)
    window, upper, budget_duration, empty = _resolve_window(start, end, duration)
    lower = window.start if window is not None else None

    kept: list[Candidate] = [
        c for c in candidates
        if (lower is None or c.timestamp >= lower)
        and (upper is None or c.timestamp < upper)
    ]
    selected = [] if empty else select_candidates(kept, duration_seconds=budget_duration)

    starts = [c.timestamp for c in selected]
    scenes = [
        VideoSceneItem(
            start=s,
            end=starts[i + 1] if i + 1 < len(starts) else _last_end(s, window, duration),
            transcript_excerpt=_select_transcript_excerpt(file_id, s, None) or None,
        )
        for i, s in enumerate(starts)
    ]
    return VideoScenesResponse(
        file_id=file_id,
        duration=duration,
        window=window,
        candidate_count=len(kept),
        scenes=scenes,
    )
