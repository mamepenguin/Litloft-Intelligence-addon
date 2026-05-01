"""Internal API client: ``POST /api/internal/files/bulk``.

Hydrates a list of file IDs into full ``FileResponse``-shaped dicts so
the search results layer can return the same shape as core's filename
match endpoint. The frontend uses this to render semantic and
filename-match results with a single ``FileCard`` component.

Failure mode: HTTP errors and connection failures are caught and
returned as an empty dict. The caller is expected to fall back to the
``IndexedFile`` snapshot (filename / file_type / mime_type / file_size /
duration are all there). This keeps semantic search resilient when the
core service is briefly unreachable — degrading to "search works,
favorite/tags absent" instead of "search 5xx".

Cache: results are cached for 30 seconds keyed by the sorted file_ids
tuple. The TTL is short enough that user-driven mutations (favorite
toggle, tag edit) settle on the next search call. Sized for the typical
search session — a single user paging through results.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Mirror retriever.py / vision.py / refine.py / summaries.py / history_client.py
_INTERNAL_API_BASE_URL_DEFAULT = "http://backend:8000/api/internal"
_INTERNAL_API_TIMEOUT_SECONDS = 5.0
_BULK_PATH = "/files/bulk"

_CACHE_TTL_SECONDS = 30.0
# Bounded so a long-lived process can't accumulate unbounded entries.
# 200 distinct query/page hydrations covers a normal search session.
_CACHE_MAX_ENTRIES = 200


def _internal_api_base_url() -> str:
    return os.environ.get(
        "HOMEVAULT_INTERNAL_API_URL", _INTERNAL_API_BASE_URL_DEFAULT
    )


# Tiny TTL cache. We don't pull cachetools in for this — the hot path
# is single-tenant and 30s eviction is naturally small.
_CACHE: dict[tuple[str, ...], tuple[float, dict[str, dict[str, Any]]]] = {}
_CACHE_LOCK = asyncio.Lock()


def _cache_get(key: tuple[str, ...]) -> dict[str, dict[str, Any]] | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if expires_at < time.monotonic():
        _CACHE.pop(key, None)
        return None
    return value


def _cache_set(key: tuple[str, ...], value: dict[str, dict[str, Any]]) -> None:
    if len(_CACHE) >= _CACHE_MAX_ENTRIES:
        # Drop oldest entry. Linear scan is fine at this size.
        oldest = min(_CACHE.items(), key=lambda kv: kv[1][0])
        _CACHE.pop(oldest[0], None)
    _CACHE[key] = (time.monotonic() + _CACHE_TTL_SECONDS, value)


def cache_clear() -> None:
    """Drop all cached entries. Test-only helper."""
    _CACHE.clear()


async def hydrate_files(file_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Return a ``{file_id: FileResponse-dict}`` map for the given IDs.

    Returns an empty dict on transport / protocol failure (the caller
    must fall back to the ``IndexedFile`` snapshot). Returns only IDs
    that core considers active; trash/missing/purged IDs simply do not
    appear in the returned map.

    Cached for 30 seconds, keyed by the sorted ``file_ids`` tuple so
    cache hits work for repeated queries that page through the same
    semantic result set. Mutations to favorite/tags settle on the next
    cache miss.
    """
    if not file_ids:
        return {}

    key = tuple(sorted(file_ids))
    async with _CACHE_LOCK:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    url = f"{_internal_api_base_url().rstrip('/')}{_BULK_PATH}"
    payload = {"file_ids": list(file_ids)}

    try:
        async with httpx.AsyncClient(
            timeout=_INTERNAL_API_TIMEOUT_SECONDS
        ) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        logger.warning(
            "files/bulk hydrate failed (%s): %s; falling back to "
            "IndexedFile snapshot",
            type(exc).__name__,
            exc,
        )
        return {}
    except Exception as exc:  # noqa: BLE001 — never let hydrate poison search
        logger.warning(
            "files/bulk hydrate raised %s: %s; falling back to snapshot",
            type(exc).__name__,
            exc,
        )
        return {}

    files = data.get("files") or []
    result: dict[str, dict[str, Any]] = {
        item["id"]: item for item in files if isinstance(item, dict) and "id" in item
    }

    async with _CACHE_LOCK:
        _cache_set(key, result)

    return result
