"""Pickup feed endpoint.

Reads precomputed rows. No scoring happens in the request path.
"""

import asyncio
import hashlib
import logging
import random
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Response

from app.database import get_search_db_read
from app.drive_context import require_drive
from app.models import PickupItem, PickupNotInterested, PickupProfile
from app.pickup import profile as profile_mod

logger = logging.getLogger(__name__)

router = APIRouter(tags=["pickup"])

#: Rows the carousel draws its daily window from.
#:
#: Sized to clear the interleave's convergence depth, not chosen for
#: variety. A lane at the weight floor places its first item around
#: ``sum(weights) / min(weight)`` — about 33 items for the realistic
#: worst case of 24 lanes — so the head of the feed is granular and
#: over-represents the heaviest lanes. Twelve cards taken from the top
#: show 6 lanes of 24; twelve sampled from the top 40 track the intended
#: proportions. Shrinking this pool silently narrows what the carousel
#: can show.
_WINDOW_POOL = 40

_MAX_LIMIT = 60


def _daily_seed(viewer_id: str, drive: str, date: str) -> int:
    key = f"{viewer_id}\x1f{drive}\x1f{date}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


@router.get("/pickup")
async def pickup_endpoint(
    drive: str = Depends(require_drive),
    viewer_id: Annotated[str | None, Header(alias="X-Lit-Viewer-Id")] = None,
    limit: int = Query(12, ge=1, le=_MAX_LIMIT),
    offset: int = Query(0, ge=0),
    window: str | None = Query(None),
    date: str | None = Query(None),
) -> dict:
    """Return this viewer's precomputed feed for one drive.

    Args:
        limit: Page size.
        offset: Position in the feed. Ignored for ``window=daily``.
        window: ``"daily"`` selects ``limit`` items from the top
            ``min(_WINDOW_POOL, total)`` using a date-seeded shuffle.
            For the carousel only — reshuffling a paged list produces
            duplicates and gaps on the second page.
        date: The viewer's local ``YYYY-MM-DD``, so the day turns over
            at their midnight rather than the server's. Falls back to
            the server's UTC date. It only seeds a shuffle, so a
            malformed value can do nothing worse than reorder cards.

    Returns:
        ``file_ids`` in order, and ``total`` rows held once the viewer's
        current exclusion set is removed. Callers use ``total`` as the
        stock figure; counting eligible files directly would mean
        scanning every unopened file in the drive.
    """
    if not viewer_id:
        return {"file_ids": [], "total": 0}

    # A read must not take the process-wide write lock: it is held
    # across long jobs (refine keeps it for a whole forced
    # alignment), so waiting on it would hold this request for that long.
    def _read() -> tuple[int, list[str]] | None:
        with get_search_db_read() as session:
            header = (
                session.query(PickupProfile)
                .filter_by(drive_id=drive, viewer_id=viewer_id)
                .first()
            )
            if header is None or header.total == 0:
                return None
            stored = [
                row[0]
                for row in session.query(PickupItem.file_id)
                .filter_by(drive_id=drive, viewer_id=viewer_id)
                .order_by(PickupItem.rank)
                .all()
            ]
            marks = {
                row[0]
                for row in session.query(PickupNotInterested.file_id)
                .filter_by(drive_id=drive, viewer_id=viewer_id)
                .all()
            }

        excluded = profile_mod.watched_file_ids(drive, viewer_id) | marks
        kept = [file_id for file_id in stored if file_id not in excluded]
        if window == "daily":
            return len(kept), kept[:min(_WINDOW_POOL, len(kept))]
        return len(kept), kept[offset:offset + limit]

    read = await asyncio.to_thread(_read)
    if read is None:
        return {"file_ids": [], "total": 0}
    total, pool = read

    if window == "daily":
        stamp = date or datetime.now(UTC).strftime("%Y-%m-%d")
        rng = random.Random(_daily_seed(viewer_id, drive, stamp))
        rng.shuffle(pool)
        pool = pool[:limit]

    return {"file_ids": pool, "total": total}


#: The core's file-id alphabet, so a malformed id never reaches a write.
_FILE_ID_PATTERN = r"^[A-Za-z0-9_-]{12}$"


def _store_mark(drive: str, viewer_id: str, file_id: str) -> None:
    from app.database import get_search_db
    from app.models import PickupNotInterested

    now = datetime.now(UTC).replace(tzinfo=None)
    with get_search_db() as session:
        if session.get(PickupNotInterested, (drive, viewer_id, file_id)) is not None:
            # A repeated PUT keeps the first mark's time: the feed only
            # asks whether the file is marked, and the sweep's rebuild
            # should not be triggered by the second click's arrival.
            return
        session.add(PickupNotInterested(
            drive_id=drive,
            viewer_id=viewer_id,
            file_id=file_id,
            marked_at=now,
        ))


def _remove_mark(drive: str, viewer_id: str, file_id: str) -> None:
    from app.database import get_search_db
    from app.models import PickupNotInterested

    with get_search_db() as session:
        session.query(PickupNotInterested).filter_by(
            drive_id=drive, viewer_id=viewer_id, file_id=file_id,
        ).delete(synchronize_session=False)


@router.put("/pickup/not-interested/{file_id}", status_code=204)
async def mark_not_interested(
    file_id: Annotated[str, Path(pattern=_FILE_ID_PATTERN)],
    drive: str = Depends(require_drive),
    viewer_id: Annotated[str | None, Header(alias="X-Lit-Viewer-Id")] = None,
) -> Response:
    """Keep one video out of this viewer's feed in this drive.

    Answers 204 whether or not the mark already existed. A drive whose
    ``index`` policy is off answers 404 at the proxy before this runs.
    """
    if not viewer_id:
        raise HTTPException(status_code=400, detail="viewer_required")
    await asyncio.to_thread(_store_mark, drive, viewer_id, file_id)
    return Response(status_code=204)


@router.delete("/pickup/not-interested/{file_id}", status_code=204)
async def unmark_not_interested(
    file_id: Annotated[str, Path(pattern=_FILE_ID_PATTERN)],
    drive: str = Depends(require_drive),
    viewer_id: Annotated[str | None, Header(alias="X-Lit-Viewer-Id")] = None,
) -> Response:
    """Undo a mark. 204 whether or not it existed."""
    if not viewer_id:
        raise HTTPException(status_code=400, detail="viewer_required")
    await asyncio.to_thread(_remove_mark, drive, viewer_id, file_id)
    return Response(status_code=204)
