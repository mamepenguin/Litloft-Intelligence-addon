"""Internal API client: ``POST /api/internal/files/matching``.

Resolves a search filter into the request drive's active file ids so a
filtered semantic search can rank only among them. There is no cache: a file
tagged between the listing and the meaning request must not be hidden from
the hits for a TTL, unlike ``file_hydrate``'s snapshot.
"""

from __future__ import annotations

import json
import logging
import os

import httpx
from pydantic import BaseModel, Field, StrictBool, ValidationError, model_validator

logger = logging.getLogger(__name__)

# Mirror file_hydrate.py and the other internal clients.
_INTERNAL_API_BASE_URL_DEFAULT = "http://backend:8000/api/internal"
_INTERNAL_API_TIMEOUT_SECONDS = 5.0
_MATCHING_PATH = "/files/matching"

#: Same caps as the core's request model. The listing adds one tag clause per
#: tag and SQLite refuses a deeply nested expression long before this.
TAG_MAX_COUNT = 20
TAG_MAX_LENGTH = 200
KIND_MAX_COUNT = 8

KINDS = frozenset(
    {"video", "image", "audio", "document", "archive", "other", "text", "pdf"}
)
_KIND_ALIASES = {"markdown": "text"}


class InvalidFilter(ValueError):
    """The ``filter`` query parameter is not a well-formed filter."""


class FilterUnavailable(RuntimeError):
    """The core did not answer ``/files/matching`` with a list of file ids."""


class SearchFilter(BaseModel):
    """The parsed operators of a search query, as the core's request model takes them.

    Normalisation matches core's: a blank tag is dropped, tags are deduplicated
    case-insensitively, kinds after the ``markdown`` alias, then the caps apply.
    """

    tags: list[str] = Field(default_factory=list)
    kinds: list[str] = Field(default_factory=list)
    favorite: StrictBool | None = None
    liked: StrictBool | None = None

    @model_validator(mode="after")
    def _normalise(self) -> "SearchFilter":
        tags: list[str] = []
        seen: set[str] = set()
        for raw in self.tags:
            tag = raw.strip()
            if not tag:
                continue
            key = tag.lower()
            if key in seen:
                continue
            seen.add(key)
            tags.append(tag)
        if len(tags) > TAG_MAX_COUNT:
            raise ValueError(f"at most {TAG_MAX_COUNT} tags")
        if any(len(tag) > TAG_MAX_LENGTH for tag in tags):
            raise ValueError(f"a tag is at most {TAG_MAX_LENGTH} characters")

        kinds: list[str] = []
        seen_kinds: set[str] = set()
        for raw_kind in self.kinds:
            kind = _KIND_ALIASES.get(raw_kind, raw_kind)
            if kind not in KINDS:
                raise ValueError(f"unknown kind: {raw_kind}")
            if kind in seen_kinds:
                continue
            seen_kinds.add(kind)
            kinds.append(kind)
        if len(kinds) > KIND_MAX_COUNT:
            raise ValueError(f"at most {KIND_MAX_COUNT} kinds")

        self.tags = tags
        self.kinds = kinds
        return self


def parse_search_filter(raw: str) -> SearchFilter:
    """Parse the ``filter`` query parameter, raising ``InvalidFilter`` on any defect."""
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise InvalidFilter("filter is not JSON") from exc
    if not isinstance(data, dict):
        raise InvalidFilter("filter is not a JSON object")
    try:
        return SearchFilter.model_validate(data)
    except ValidationError as exc:
        raise InvalidFilter("filter has an invalid shape") from exc


def _internal_api_base_url() -> str:
    return os.environ.get(
        "HOMEVAULT_INTERNAL_API_URL", _INTERNAL_API_BASE_URL_DEFAULT
    )


async def fetch_matching_file_ids(drive: str, search_filter: SearchFilter) -> list[str]:
    """Ask the core for ``drive``'s active file ids that satisfy ``search_filter``.

    Raises ``FilterUnavailable`` for a transport error, a timeout, a non-200
    answer, or a body without a list of string ids, so the route can answer 503
    instead of falling back to an unfiltered search.
    """
    url = f"{_internal_api_base_url().rstrip('/')}{_MATCHING_PATH}"
    payload = {"drive": drive, **search_filter.model_dump()}

    try:
        async with httpx.AsyncClient(
            timeout=_INTERNAL_API_TIMEOUT_SECONDS
        ) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "files/matching failed (%s) for drive %s: %s",
            type(exc).__name__,
            drive,
            exc,
        )
        raise FilterUnavailable(type(exc).__name__) from exc

    file_ids = data.get("file_ids") if isinstance(data, dict) else None
    if not isinstance(file_ids, list) or not all(
        isinstance(file_id, str) for file_id in file_ids
    ):
        logger.warning(
            "files/matching answered without a file_ids list for drive %s", drive
        )
        raise FilterUnavailable("file_ids missing")
    return file_ids
