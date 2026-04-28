"""Stage B: viewer-history client against Litloft's Internal API.

Spec: ``2026-04-26-intelligence-ask-personal-history-query.md`` §4.2.
Wraps ``GET /api/internal/viewer-history`` so the Ask service layer
can ask "which file_ids has this viewer touched in this drive within
``[after, before)``" without re-implementing HTTP boilerplate.

The host trust model
--------------------
The host's addon_proxy injects ``X-Lit-Viewer-Id`` from the
``lit_viewer`` cookie before the request reaches us, so by the time we
get here the viewer is already authenticated by the host. We just
forward the value to the host's Internal API which trusts the
``CORE_INTERNAL_SECRET`` shared secret as its only auth.

Failure modes
-------------
On any error (network, malformed response, host returned non-200) we
return an empty list and let the service layer fall back to the
graceful path. A failed history call must *never* hard-fail the whole
Ask request — the worst outcome is "we couldn't apply the personal
filter, here are the legacy results" which is what the spec's
``fallback_when_empty: graceful`` config option formalises.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Literal

import httpx

logger = logging.getLogger(__name__)


_BASE_URL_DEFAULT = "http://backend:8000/api/internal"
_TIMEOUT_SECONDS = 5.0


def _base_url() -> str:
    return os.environ.get("HOMEVAULT_INTERNAL_API_URL", _BASE_URL_DEFAULT)


def _internal_secret() -> str | None:
    """Return the shared secret used to gate Internal API calls.

    The host accepts an empty/missing secret as a no-op (dev mode), so
    sending the header only when we actually have one keeps the dev
    parity intact.
    """
    return os.environ.get("CORE_INTERNAL_SECRET") or None


def _format_naive_iso(dt: datetime) -> str:
    """Render a datetime for the Internal API's ``after``/``before`` params.

    ``WatchHistory.last_played_at`` is naive on the host side; sending
    a tz-aware ISO string would compare unfavourably against the
    naive column under SQLite (text comparison). The host strips
    tzinfo too, but mirroring the convention here removes one round
    of normalisation noise from request logs.
    """
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt.isoformat()


async def fetch_viewer_history(
    *,
    viewer_id: str,
    drive: str,
    after: datetime | None,
    before: datetime | None,
    kind: Literal["viewed", "not_viewed"],
) -> list[str]:
    """Return file_ids the viewer has (or hasn't) touched in this drive.

    Args:
        viewer_id: 16-char SHA-256 prefix produced by the host's
            ``nickname_to_viewer_id``. The host re-validates the format
            and 400s on a malformed value; we let that bubble back as
            an empty result rather than retrying.
        drive: Canonical drive name. Same value the request's
            ``X-Lit-Drive`` carries.
        after / before: Half-open ``[after, before)`` window over
            ``last_played_at``. Either side may be None for unbounded.
        kind: ``"viewed"`` for the file_ids the viewer touched in the
            window; ``"not_viewed"`` for the complementary set within
            the drive.

    Returns:
        A list of file_ids (no guaranteed ordering). Empty on every
        failure path (network error, non-200, malformed payload) so
        the caller can pass it to ``file_id_scope`` without checking
        for None.
    """
    params: dict[str, str] = {
        "viewer_id": viewer_id,
        "drive": drive,
        "kind": kind,
    }
    if after is not None:
        params["after"] = _format_naive_iso(after)
    if before is not None:
        params["before"] = _format_naive_iso(before)

    headers: dict[str, str] = {}
    secret = _internal_secret()
    if secret:
        headers["x-internal-secret"] = secret

    url = f"{_base_url()}/viewer-history"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            resp = await client.get(url, params=params, headers=headers)
    except httpx.HTTPError as exc:
        logger.warning(
            "history-client: request failed for drive=%s kind=%s: %s",
            drive, kind, type(exc).__name__,
        )
        return []

    if resp.status_code != 200:
        # 400 (bad viewer/drive/kind/after-before) and 404 (unknown
        # drive) both collapse to "no usable history" — the host has
        # already authenticated the request, so a non-200 here is a
        # contract bug worth logging but not crashing on.
        logger.warning(
            "history-client: unexpected status %d for drive=%s kind=%s",
            resp.status_code, drive, kind,
        )
        return []

    try:
        payload = resp.json()
    except ValueError:
        logger.warning("history-client: non-JSON response")
        return []

    file_ids = payload.get("file_ids") if isinstance(payload, dict) else None
    if not isinstance(file_ids, list):
        return []

    # Defensive: reject non-string entries before they propagate into
    # the retriever's ``file_id_scope`` contract (str list).
    return [fid for fid in file_ids if isinstance(fid, str)]
