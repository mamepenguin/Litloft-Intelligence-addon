"""Pickup feed endpoint.

Reads precomputed rows. No scoring happens in the request path.
"""

import hashlib
import logging
import random
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query

from app.database import get_search_db_read
from app.drive_context import require_drive
from app.models import PickupItem, PickupProfile

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
        ``file_ids`` in order, and ``total`` rows held. Callers use
        ``total`` as the stock figure; counting eligible files directly
        would mean scanning every unopened file in the drive.
    """
    if not viewer_id:
        return {"file_ids": [], "total": 0}

    # A read must not take the process-wide write lock: it is held
    # across long jobs (refine keeps it for a whole forced
    # alignment), and this endpoint is awaited on the addon's event
    # loop, so blocking here stalls every other request too.
    with get_search_db_read() as session:
        header = (
            session.query(PickupProfile)
            .filter_by(drive_id=drive, viewer_id=viewer_id)
            .first()
        )
        if header is None or header.total == 0:
            return {"file_ids": [], "total": 0}
        total = header.total

        query = (
            session.query(PickupItem.file_id)
            .filter_by(drive_id=drive, viewer_id=viewer_id)
            .order_by(PickupItem.rank)
        )
        if window == "daily":
            pool = [row[0] for row in query.limit(min(_WINDOW_POOL, total)).all()]
        else:
            pool = [
                row[0] for row in query.offset(offset).limit(limit).all()
            ]

    if window == "daily":
        stamp = date or datetime.now(UTC).strftime("%Y-%m-%d")
        rng = random.Random(_daily_seed(viewer_id, drive, stamp))
        rng.shuffle(pool)
        pool = pool[:limit]

    return {"file_ids": pool, "total": total}
