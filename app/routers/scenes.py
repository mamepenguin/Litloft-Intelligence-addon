"""``GET /files/{file_id}/scenes``: representative scenes without a model.

Runs the selection of SPEC-ADDON-112 over the rows the indexer already stored,
so any indexed native video, and an analysable ``.loft``, gets its sections and
frame times whether or not a Visual Index was generated. Nothing is written and
no LLM is called.
"""

from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException, Query

from app.drive_context import require_drive
from app.routers.files import _get_indexed_file_or_404
from app.schemas import VideoSceneItem, VideoScenesResponse, VideoScenesWindow
from app.workers.video_visual import _select_transcript_excerpt
from app.workers.video_visual_selection import load_selection_inputs, select_scenes
from app.workers.whisper import LOFT_MIME
from app.youtube_source import YOUTUBE_PROVIDER, LoftUnreadable, read_loft

router = APIRouter(tags=["scenes"])


def _check_bound(name: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value < 0):
        raise HTTPException(
            status_code=422, detail=f"{name} must be a finite number >= 0"
        )


def _known(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _resolve_window(
    start: float | None, end: float | None, duration: float | None,
) -> tuple[VideoScenesWindow | None, float, float | None]:
    """Return ``(window, span start, span end)`` for the selection.

    ``span end`` of ``None`` is unknown to the caller; the selection then ends
    the span one second after the last frame, as SPEC-ADDON-112 Terms says. A
    window whose end is still the duration gives the duration, even when the
    start is at or past it (the selection returns no sections then).
    """
    known = _known(duration)
    if start is None and end is None:
        return None, 0.0, known
    lo = start if start is not None else 0.0
    if end is not None:
        return VideoScenesWindow(start=lo, end=end), lo, end
    if known is not None:
        return VideoScenesWindow(start=lo, end=known), lo, known
    return VideoScenesWindow(start=lo, end=None), lo, None


def _require_analysable_loft(indexed: dict) -> None:
    # The fetch's classification is authoritative: re-reading the reference
    # here would fail for a drive this process cannot mount, and a `.loft`
    # of another source has no video to select from.
    classified = indexed.get("loft_source")
    if classified is not None:
        if classified in ("youtube", "web"):
            return
        raise HTTPException(status_code=400, detail="Not a video file")
    try:
        source = read_loft(indexed["file_path"], indexed["drive"])
    except LoftUnreadable:
        source = None
    if source is None or source.provider != YOUTUBE_PROVIDER:
        raise HTTPException(status_code=400, detail="Not a video file")


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
    if indexed["file_type"] != "video":
        raise HTTPException(status_code=400, detail="Not a video file")
    if indexed["mime_type"] == LOFT_MIME:
        _require_analysable_loft(indexed)

    inputs = load_selection_inputs(file_id)
    window, lo, hi = _resolve_window(start, end, inputs.duration)
    selection = select_scenes(inputs, lo, hi)

    scenes = [
        VideoSceneItem(
            start=item.section_start,
            end=item.section_end,
            frame_time=item.frame_time,
            transcript_excerpt=(
                _select_transcript_excerpt(file_id, item.frame_time, None) or None
            ),
        )
        for item in selection.navigation
    ]
    return VideoScenesResponse(
        file_id=file_id,
        duration=inputs.duration,
        window=window,
        candidate_count=selection.candidate_count,
        scenes=scenes,
    )
