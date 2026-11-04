"""Collection source sets: the file ids of a core collection, in its order.

Wraps ``GET /api/internal/collections/{collection_id}/file-ids`` so a
caller can turn a scope like ``{"kind": "collection", "id": ...}`` into
the retriever's ``file_id_scope`` allow-list.

A failure never becomes an empty list: ``[]`` means the collection has
no active file, so every error raises and the caller decides what to
tell the viewer.
"""

from __future__ import annotations

import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)


_BASE_URL_DEFAULT = "http://backend:8000/api/internal"
_TIMEOUT_SECONDS = 5.0
_COLLECTION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{12}$")


class CollectionLookupError(Exception):
    """Base type for every failure of a collection lookup."""


class CollectionNotFound(CollectionLookupError):
    """The id is malformed, or the core has no such collection in this drive."""


class CollectionLookupFailed(CollectionLookupError):
    """The lookup could not answer; the collection's contents are unknown."""


def _base_url() -> str:
    return os.environ.get("HOMEVAULT_INTERNAL_API_URL", _BASE_URL_DEFAULT)


def _internal_secret() -> str | None:
    # The host accepts a missing secret as a no-op (dev mode), so sending the
    # header only when we have one keeps that parity.
    return os.environ.get("CORE_INTERNAL_SECRET") or None


async def fetch_collection_file_ids(
    *, collection_id: str, drive: str
) -> list[str]:
    """Return the active file ids of ``drive``'s collection, in core order.

    Raises:
        CollectionNotFound: the id fails the nanoid check (no request is
            sent), or the core answers 400 or 404.
        CollectionLookupFailed: any other status, a network error or
            timeout, or a body that is not ``{"file_ids": [str, ...]}``.
    """
    if _COLLECTION_ID_RE.match(collection_id) is None:
        raise CollectionNotFound(f"malformed collection id for drive={drive}")

    headers: dict[str, str] = {}
    secret = _internal_secret()
    if secret:
        headers["x-internal-secret"] = secret

    url = f"{_base_url()}/collections/{collection_id}/file-ids"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            resp = await client.get(
                url, params={"drive": drive}, headers=headers
            )
    except httpx.HTTPError as exc:
        # The exception text can carry a file id from the request path or a
        # URL in the transport error; log the type only.
        logger.warning(
            "collection-client: request failed for drive=%s: %s",
            drive,
            type(exc).__name__,
        )
        raise CollectionLookupFailed(type(exc).__name__) from exc

    if resp.status_code in (400, 404):
        raise CollectionNotFound(
            f"status {resp.status_code} for drive={drive}"
        )

    if resp.status_code != 200:
        logger.warning(
            "collection-client: unexpected status %d for drive=%s",
            resp.status_code,
            drive,
        )
        raise CollectionLookupFailed(f"status {resp.status_code}")

    try:
        payload = resp.json()
    except ValueError:
        logger.warning(
            "collection-client: non-JSON response for drive=%s", drive
        )
        raise CollectionLookupFailed("non-JSON response") from None

    file_ids = payload.get("file_ids") if isinstance(payload, dict) else None
    if not isinstance(file_ids, list) or not all(
        isinstance(fid, str) for fid in file_ids
    ):
        logger.warning(
            "collection-client: malformed response for drive=%s", drive
        )
        raise CollectionLookupFailed("malformed response")

    return list(file_ids)
