"""Pickup (recommendation) endpoint for the dashboard widget."""

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Header

from app.database import get_search_db
from app.drive_context import require_drive
from app.models import PickupCache

logger = logging.getLogger(__name__)

router = APIRouter(tags=["pickup"])


@router.get("/pickup")
async def pickup_endpoint(
    drive: str = Depends(require_drive),
    viewer_id: Annotated[str | None, Header(alias="X-Lit-Viewer-Id")] = None,
) -> dict:
    """Return precomputed recommendation file_ids for the dashboard pickup widget.

    Returns an empty list when:
    - No viewer profile is set (viewer_id is None).
    - No cached recommendations exist for this viewer × drive yet.

    The response is always fast — this endpoint only reads from a precomputed
    cache row; no KNN computation happens in the request path.
    """
    if not viewer_id:
        return {"file_ids": []}

    with get_search_db() as session:
        cache = (
            session.query(PickupCache)
            .filter_by(drive_id=drive, viewer_id=viewer_id)
            .first()
        )

    if not cache:
        return {"file_ids": []}

    try:
        file_ids = json.loads(cache.file_ids)
    except (ValueError, TypeError):
        return {"file_ids": []}

    if not isinstance(file_ids, list):
        return {"file_ids": []}
    return {"file_ids": [x for x in file_ids if isinstance(x, str)]}
