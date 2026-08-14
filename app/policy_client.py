"""Per-drive intelligence policy lookup against Litloft's Internal API.

Litloft stores per-drive feature toggles in ``drives.json`` (see the
core ``backend/app/config.py::is_addon_feature_enabled``). This module
queries them through ``GET /api/internal/drive-policy?drive=&addon=``
and caches the result for a short TTL so worker hot paths aren't
network-bound.

Failure-mode taxonomy (Phase 1A of the cloud-transcription-providers
spec)
---------------------------------------------------------------------
The original module was always **fail-open**: any error returned
``True`` so legitimate work kept flowing. That's appropriate for
non-cloud features (search / indexing) where the worst case is
"running work for an off-policy drive" — already defended at the
request-time ``X-Lit-Drive`` enforcement layer.

Cloud transcription is a different threat model: "fail-open" would
mean shipping audio to a third party for a drive whose owner has
explicitly disabled cloud send. Callers therefore opt into fail-CLOSED
by passing ``default_on_failure=False``.

Cold-start grace period
-----------------------
The intelligence container can warm up before the backend HTTP
listener is healthy. With ``default_on_failure=False`` the first jobs
would be permanently marked failed even though the real cause is
"infra not ready yet". To distinguish "policy off" from "infra not
yet running" we track a 60-second grace window from process start AND
clear the grace flag the moment the policy backend has returned at
least one HTTP 200 (proof of life). While the grace flag is active,
``ConnectionError`` on a fail-closed lookup raises
:class:`TransientError` so the caller's retry path can put the job
back on the queue instead of recording a permanent failure.
"""

import logging
import os
import time
from typing import Final
from urllib.parse import quote  # noqa: F401  (kept for parity with prior import surface)

import httpx

from app.workers.transcription.errors import TransientError

logger = logging.getLogger(__name__)

_BASE_URL_DEFAULT = "http://backend:8000/api/internal"
_TIMEOUT_SECONDS = 5.0
_TTL_SECONDS = 30.0
_ADDON_NAME = "intelligence"

# Cold-start grace window (seconds). Tuned in spec
# 2026-05-07-cloud-transcription-providers.md §"Cold-start race の回避":
# 60s comfortably covers backend warm-up on a typical Compose stack
# without leaving a long permissive window after legitimate policy off.
STARTUP_GRACE_S: Final[float] = 60.0

# (drive, feature) -> (expires_at_monotonic, value)
_cache: dict[tuple[str, str], tuple[float, bool]] = {}

# Module-level startup baseline (set at import time). Tests can pin
# via ``_set_startup_for_tests``.
_startup_at: float = time.monotonic()

# Becomes True the first time the policy backend returns 200, and
# stays True until the next ``_reset_grace_period_for_tests`` call.
# When True, the grace window no longer applies — backend has proven
# it is reachable, so future ConnectionErrors are real failures.
_observed_healthy: bool = False


def _base_url() -> str:
    return os.environ.get("HOMEVAULT_INTERNAL_API_URL", _BASE_URL_DEFAULT)


def _evaluate_response(payload: dict, feature: str) -> bool:
    """Resolve a feature flag against the host's policy response.

    The host returns ``{"default": bool, "features": {<name>: bool}}``.
    A named feature wins over default; missing keys fall back to
    default; a malformed payload fails open (True) so a transient
    schema mismatch does not silently disable real work.
    """
    if not isinstance(payload, dict):
        return True
    features = payload.get("features")
    if isinstance(features, dict) and feature in features:
        return bool(features[feature])
    default = payload.get("default", True)
    return bool(default)


def _in_grace_period() -> bool:
    """True iff we are still inside the cold-start grace window AND
    have not yet observed a healthy backend response."""
    if _observed_healthy:
        return False
    return (time.monotonic() - _startup_at) < STARTUP_GRACE_S


def _resolve_failure(default_on_failure: bool, *, drive: str, feature: str) -> bool:
    """Translate a non-200 / non-cached lookup into the fail value.

    ``default_on_failure=True`` keeps the legacy fail-open behaviour
    (return True). ``default_on_failure=False`` is the cloud-policy
    fail-CLOSED posture (return False).
    """
    if default_on_failure:
        return True
    logger.warning(
        "policy-client: fail-closed for drive=%s feature=%s",
        drive, feature,
    )
    return False


async def is_feature_enabled(
    drive: str,
    feature: str,
    *,
    default_on_failure: bool = True,
) -> bool:
    """Return True when the addon feature is enabled for ``drive``.

    Cached per (drive, feature) pair for ``_TTL_SECONDS`` so worker
    hot paths don't open a new HTTP connection per task. The cache is
    in-process; restart the addon container to invalidate everything,
    or wait one TTL for stale entries to expire.

    Args:
        drive: drive name (matches drives.json).
        feature: feature flag inside ``addons.intelligence``.
        default_on_failure: ``True`` keeps the legacy fail-open
            posture used by ``index`` / ``search``. ``False`` engages
            fail-CLOSED, used by the cloud transcription gate. While
            inside the cold-start grace window AND
            ``default_on_failure=False``, ``ConnectionError`` raises
            :class:`TransientError` so the caller can retry rather
            than record a permanent failure.
    """
    global _observed_healthy

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
        # Cold-start grace path: only meaningful for fail-closed
        # callers. A connection error during warm-up should bubble up
        # as TransientError so the worker's retry loop puts the job
        # back, not record a permanent JobRecord.failed.
        if not default_on_failure and _in_grace_period():
            raise TransientError(
                "policy backend unreachable during cold-start grace period"
            ) from e
        return _resolve_failure(default_on_failure, drive=drive, feature=feature)

    if resp.status_code == 200:
        # Backend is alive — clear the grace flag forever (until the
        # process restarts or a test resets state).
        _observed_healthy = True
        try:
            value = _evaluate_response(resp.json(), feature)
        except ValueError:
            logger.warning("policy-client: non-JSON response")
            return _resolve_failure(
                default_on_failure, drive=drive, feature=feature
            )
        _cache[key] = (now + _TTL_SECONDS, value)
        return value

    if resp.status_code == 404:
        # Drive removed from drives.json. Cache as disabled to avoid
        # repeated lookups; a restart will repopulate.
        # 404 means the drive itself is gone — record the lookup so
        # we don't hammer the backend, and treat as disabled
        # regardless of fail-open/closed posture.
        _cache[key] = (now + _TTL_SECONDS, False)
        return False

    logger.warning(
        "policy-client: unexpected status %d for drive=%s",
        resp.status_code, drive,
    )
    return _resolve_failure(default_on_failure, drive=drive, feature=feature)


def reset_cache() -> None:
    """Drop every cached policy entry — for tests and explicit reloads."""
    _cache.clear()


def _reset_grace_period_for_tests() -> None:
    """Reset the cold-start grace state for test isolation.

    Tests must call this between cases to avoid the "we observed a
    healthy backend in test A" flag leaking into test B's grace
    semantics.
    """
    global _startup_at, _observed_healthy
    _startup_at = time.monotonic()
    _observed_healthy = False


def _set_startup_for_tests(value: float) -> None:
    """Pin the startup baseline timestamp (test-only)."""
    global _startup_at, _observed_healthy
    _startup_at = value
    _observed_healthy = False


async def is_file_feature_enabled(
    file_id: str,
    feature: str,
    *,
    default_on_failure: bool = True,
) -> bool:
    """Lookup the file's drive in the local index and apply policy.

    Convenience for worker enqueue paths that have a file_id but no
    drive on hand. Database failures and unknown files resolve to
    ``default_on_failure`` so callers can explicitly choose fail-open
    or fail-closed behavior.
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
        return default_on_failure
    if row is None:
        return default_on_failure
    return await is_feature_enabled(
        row.drive,
        feature,
        default_on_failure=default_on_failure,
    )
