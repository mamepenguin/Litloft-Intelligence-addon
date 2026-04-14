"""Per-drive intelligence policy lookup against HomeVault's Internal API.

HomeVault stores per-drive feature toggles in ``drives.json`` (see the
core ``backend/app/config.py::is_addon_feature_enabled``). This module
queries them through ``GET /api/internal/drive-policy?drive=&addon=``
and caches the result for a short TTL so worker hot paths aren't
network-bound.

Failure mode
------------
The host is on the same Docker network and the policy is small; calls
should always succeed. When they don't (network blip, host restart),
we **fail open** — return ``True`` so legitimate work keeps flowing.
This matches the host-side ``event_hooks`` filter, which also forwards
events when drive resolution fails. The actual data-protection gate is
the request-time ``X-HV-Drive`` enforcement at the router layer; this
client is an optimisation that lets background workers skip wasted
work for off-policy drives, not a security boundary.
"""

import logging
import os
import time
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

_BASE_URL_DEFAULT = "http://backend:8000/api/internal"
_TIMEOUT_SECONDS = 5.0
_TTL_SECONDS = 30.0
_ADDON_NAME = "intelligence"

# (drive, feature) -> (expires_at_monotonic, value)
_cache: dict[tuple[str, str], tuple[float, bool]] = {}


def _base_url() -> str:
    return os.environ.get("HOMEVAULT_INTERNAL_API_URL", _BASE_URL_DEFAULT)


def _evaluate_response(payload: dict, feature: str) -> bool:
    """Resolve a feature flag against the policy dict format.

    The host returns:
    - ``{}``: no policy configured → all features enabled.
    - ``{"_all": false}``: bool shorthand → every feature disabled.
    - ``{"<feature>": bool, ...}``: per-feature dict.
    Unknown features default to True (graceful degradation).
    """
    if not isinstance(payload, dict):
        return True
    if "_all" in payload:
        return bool(payload["_all"])
    if feature in payload:
        return bool(payload[feature])
    return True


async def is_feature_enabled(drive: str, feature: str) -> bool:
    """Return True when the addon feature is enabled for ``drive``.

    Cached per (drive, feature) pair for ``_TTL_SECONDS`` so worker
    hot paths don't open a new HTTP connection per task. The cache is
    in-process; restart the addon container to invalidate everything,
    or wait one TTL for stale entries to expire.
    """
    key = (drive, feature)
    now = time.monotonic()
    cached = _cache.get(key)
    if cached and cached[0] > now:
        return cached[1]

    url = f"{_base_url()}/drive-policy"
    params = {"drive": drive, "addon": _ADDON_NAME}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            resp = await client.get(url, params=params)
    except httpx.HTTPError as e:
        logger.warning(
            "policy-client: request failed for drive=%s feature=%s: %s",
            drive, feature, type(e).__name__,
        )
        return True

    if resp.status_code == 404:
        # Drive removed from drives.json. Cache as disabled to avoid
        # repeated lookups; a restart will repopulate.
        value = False
    elif resp.status_code != 200:
        logger.warning(
            "policy-client: unexpected status %d for drive=%s",
            resp.status_code, drive,
        )
        return True
    else:
        try:
            value = _evaluate_response(resp.json(), feature)
        except ValueError:
            logger.warning("policy-client: non-JSON response")
            return True

    _cache[key] = (now + _TTL_SECONDS, value)
    return value


def reset_cache() -> None:
    """Drop every cached policy entry — for tests and explicit reloads."""
    _cache.clear()


async def is_file_feature_enabled(file_id: str, feature: str) -> bool:
    """Lookup the file's drive in the local index and apply policy.

    Convenience for worker enqueue paths that have a file_id but no
    drive on hand. Returns ``True`` when the file is unknown so we
    don't accidentally suppress legitimate work for a freshly indexed
    file the cache hasn't seen yet — consistent with the fail-open
    posture of ``is_feature_enabled``.
    """
    # Local import: keeps this module importable even if the search
    # DB hasn't initialised yet (e.g. very early startup, tests that
    # don't spin up a session).
    try:
        from app.database import get_search_db
        from app.models import IndexedFile
        with get_search_db() as session:
            row = (
                session.query(IndexedFile.drive)
                .filter(IndexedFile.file_id == file_id)
                .first()
            )
    except Exception:
        return True
    if row is None:
        return True
    return await is_feature_enabled(row.drive, feature)
