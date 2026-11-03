"""The one path from this addon to the core's event broadcaster.

Core answers ``/api/internal/addon-events`` with 403 whenever
``CORE_INTERNAL_SECRET`` is set and the ``X-Internal-Secret`` header is
absent, and production sets it. A poster that builds its own request
therefore has every event dropped, so ``tests/test_addon_event_secret.py``
fails when one appears outside this module.

The core also broadcasts an event with no top-level ``drive`` to every
connected viewer, including viewers who cannot open the file's drive. A
drive-less post is therefore refused here rather than sent: the event
would leak a file id and its activity to every passwordless viewer.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_INTERNAL_API_DEFAULT = "http://backend:8000/api/internal"
_ADDON_EVENTS_PATH = "/addon-events"
_SECRET_HEADER = "X-Internal-Secret"
_TIMEOUT_SECONDS = 3.0


def _url() -> str:
    base = os.environ.get("HOMEVAULT_INTERNAL_API_URL", _INTERNAL_API_DEFAULT)
    return f"{base.rstrip('/')}{_ADDON_EVENTS_PATH}"


def _headers() -> dict[str, str]:
    secret = os.environ.get("CORE_INTERNAL_SECRET", "")
    return {_SECRET_HEADER: secret} if secret else {}


def _scoped_drive(data: dict[str, Any], drive: str | None) -> str | None:
    """Prefer the explicit argument, then the drive carried inside data."""
    if drive:
        return drive
    in_data = data.get("drive") if isinstance(data, dict) else None
    return in_data or None


def _payload(event: str, data: dict[str, Any], drive: str) -> dict[str, Any]:
    return {"event": event, "data": data, "drive": drive}


async def post_addon_event(
    event: str, data: dict[str, Any], *, drive: str | None = None
) -> bool:
    """POST one event; return False instead of raising when it fails.

    Callers treat a lost event as cosmetic, so a transport error or a
    non-2xx response must never fail the work that produced it. An event
    with no drive at all is not worth sending: the core would broadcast it
    to every viewer, which is exactly the leak this helper exists to stop.
    """
    scoped_drive = _scoped_drive(data, drive)
    if scoped_drive is None:
        logger.warning("addon event %s has no drive; not posted", event)
        return False
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT_SECONDS, trust_env=False
        ) as client:
            response = await client.post(
                _url(),
                headers=_headers(),
                json=_payload(event, data, scoped_drive),
            )
            response.raise_for_status()
    except Exception:
        return False
    return True


def post_addon_event_sync(
    event: str, data: dict[str, Any], *, drive: str | None = None
) -> bool:
    """Synchronous twin of :func:`post_addon_event` for sync callers."""
    scoped_drive = _scoped_drive(data, drive)
    if scoped_drive is None:
        logger.warning("addon event %s has no drive; not posted", event)
        return False
    try:
        with httpx.Client(timeout=_TIMEOUT_SECONDS, trust_env=False) as client:
            response = client.post(
                _url(),
                headers=_headers(),
                json=_payload(event, data, scoped_drive),
            )
            response.raise_for_status()
    except Exception:
        return False
    return True
