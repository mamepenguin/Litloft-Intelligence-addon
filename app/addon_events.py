"""The one path from this addon to the core's event broadcaster.

Core answers ``/api/internal/addon-events`` with 403 whenever
``CORE_INTERNAL_SECRET`` is set and the ``X-Internal-Secret`` header is
absent, and production sets it. A poster that builds its own request
therefore has every event dropped, so ``tests/test_addon_event_secret.py``
fails when one appears outside this module.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

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


def _payload(
    event: str, data: dict[str, Any], drive: str | None
) -> dict[str, Any]:
    payload: dict[str, Any] = {"event": event, "data": data}
    in_data = data.get("drive") if isinstance(data, dict) else None
    scoped_drive = drive or in_data
    if scoped_drive:
        payload["drive"] = scoped_drive
    return payload


async def post_addon_event(
    event: str, data: dict[str, Any], *, drive: str | None = None
) -> bool:
    """POST one event; return False instead of raising when it fails.

    Callers treat a lost event as cosmetic, so a transport error or a
    non-2xx response must never fail the work that produced it.
    """
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT_SECONDS, trust_env=False
        ) as client:
            response = await client.post(
                _url(), headers=_headers(), json=_payload(event, data, drive)
            )
            response.raise_for_status()
    except Exception:
        return False
    return True


def post_addon_event_sync(
    event: str, data: dict[str, Any], *, drive: str | None = None
) -> bool:
    """Synchronous twin of :func:`post_addon_event` for sync callers."""
    try:
        with httpx.Client(timeout=_TIMEOUT_SECONDS, trust_env=False) as client:
            response = client.post(
                _url(), headers=_headers(), json=_payload(event, data, drive)
            )
            response.raise_for_status()
    except Exception:
        return False
    return True
