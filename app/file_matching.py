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
import re
import unicodedata
from collections.abc import Callable
from decimal import Decimal

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    model_validator,
)

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
FOLDER_MAX_COUNT = 10
FOLDER_MAX_LENGTH = 1024
COLLECTION_MAX_COUNT = 10
COLLECTION_MAX_LENGTH = 100
RANGE_MAX_COUNT = 4
MAX_SIZE_BYTES = 2**53 - 1
MAX_DURATION_SECONDS = Decimal("10000000")

_WIRE_FORM = re.compile(r"^(gt|gte|lt|lte):(.*)$")
_SIZE_NUMBER = re.compile(r"^(0|[1-9]\d*)$")


def _plain_decimal(text: str, places: int) -> str | None:
    if not re.fullmatch(r"(0|[1-9]\d*)(\.\d+)?", text):
        return None
    if "." in text:
        fraction = text.split(".", 1)[1]
        if len(fraction) > places or fraction.endswith("0"):
            return None
    return text


def _parse_size(item: str) -> str:
    match = _WIRE_FORM.match(item)
    if match is None:
        raise ValueError(f"invalid size comparison: {item!r}")
    operator, amount = match.groups()
    if not _SIZE_NUMBER.fullmatch(amount) or int(amount) > MAX_SIZE_BYTES:
        raise ValueError(f"invalid size comparison: {item!r}")
    return f"{operator}:{int(amount)}"


def _parse_duration(item: str) -> str:
    match = _WIRE_FORM.match(item)
    if match is None:
        raise ValueError(f"invalid duration comparison: {item!r}")
    operator, amount = match.groups()
    text = _plain_decimal(amount, 3)
    if text is None or Decimal(text) > MAX_DURATION_SECONDS:
        raise ValueError(f"invalid duration comparison: {item!r}")
    return f"{operator}:{text}"


def _normalise_comparisons(
    values: list[str], parse: Callable[[str], str], label: str
) -> list[str]:
    comparisons: list[str] = []
    for raw in values:
        comparison = parse(raw)
        if comparison not in comparisons:
            comparisons.append(comparison)
    if len(comparisons) > RANGE_MAX_COUNT:
        raise ValueError(f"at most {RANGE_MAX_COUNT} {label}")
    return comparisons

KINDS = frozenset(
    {"video", "image", "audio", "document", "archive", "other", "text", "pdf"}
)
_KIND_ALIASES = {"markdown": "text"}


class InvalidFilter(ValueError):
    """The ``filter`` query parameter is not a well-formed filter."""


class FilterUnavailable(RuntimeError):
    """The core did not answer ``/files/matching`` with a list of file ids."""


def apply_filter_normalisation(model: BaseModel) -> None:
    """Normalise the four condition lists on ``model`` in place.

    Shared by the search ``filter`` and the Ask ``scope`` so the two name the
    same files for the same values: a blank tag is dropped, tags and
    collections are deduplicated case-insensitively, kinds go through the
    ``markdown`` alias, folders are NFC-normalised and trimmed of ``/``, then
    the caps apply. Raises ``ValueError`` for a value either model rejects.
    """
    tags: list[str] = []
    seen: set[str] = set()
    for raw in model.tags:
        tag = raw.strip()
        if not tag:
            continue
        if len(tag) > TAG_MAX_LENGTH:
            raise ValueError(f"a tag is at most {TAG_MAX_LENGTH} characters")
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag)
    if len(tags) > TAG_MAX_COUNT:
        raise ValueError(f"at most {TAG_MAX_COUNT} tags")

    kinds: list[str] = []
    seen_kinds: set[str] = set()
    for raw_kind in model.kinds:
        kind = _KIND_ALIASES.get(raw_kind, raw_kind)
        if kind not in KINDS:
            raise ValueError(f"unknown kind: {raw_kind}")
        if kind in seen_kinds:
            continue
        seen_kinds.add(kind)
        kinds.append(kind)
    if len(kinds) > KIND_MAX_COUNT:
        raise ValueError(f"at most {KIND_MAX_COUNT} kinds")

    folders: list[str] = []
    for raw_folder in model.folders:
        folder = unicodedata.normalize("NFC", raw_folder).strip("/")
        if (
            not folder
            or "\x00" in folder
            or any(segment in ("", ".", "..") for segment in folder.split("/"))
        ):
            raise ValueError(f"invalid folder: {raw_folder!r}")
        if len(folder) > FOLDER_MAX_LENGTH:
            raise ValueError(f"a folder is at most {FOLDER_MAX_LENGTH} characters")
        if folder not in folders:
            folders.append(folder)
    if len(folders) > FOLDER_MAX_COUNT:
        raise ValueError(f"at most {FOLDER_MAX_COUNT} folders")

    collections: list[str] = []
    seen_collections: set[str] = set()
    for raw_collection in model.collections:
        name = raw_collection.strip()
        if not name:
            continue
        if len(name) > COLLECTION_MAX_LENGTH:
            raise ValueError(
                f"a collection is at most {COLLECTION_MAX_LENGTH} characters"
            )
        key = name.lower()
        if key in seen_collections:
            continue
        seen_collections.add(key)
        collections.append(name)
    if len(collections) > COLLECTION_MAX_COUNT:
        raise ValueError(f"at most {COLLECTION_MAX_COUNT} collections")

    model.tags = tags
    model.kinds = kinds
    model.folders = folders
    model.collections = collections


class SearchFilter(BaseModel):
    """The parsed operators of a search query, as the core's request model takes them.

    Normalisation matches core's: a blank tag is dropped, tags are deduplicated
    case-insensitively, kinds after the ``markdown`` alias, folders are NFC-
    normalised and trimmed of ``/``, collection names are trimmed and deduped
    case-insensitively, then the caps apply.
    """

    # A field the core does not know must fail here too; otherwise this addon
    # would keep working against a core that had dropped a scope it asked for.
    model_config = ConfigDict(extra="forbid")

    tags: list[str] = Field(default_factory=list)
    kinds: list[str] = Field(default_factory=list)
    favorite: StrictBool | None = None
    liked: StrictBool | None = None
    folders: list[str] = Field(default_factory=list)
    collections: list[str] = Field(default_factory=list)
    sizes: list[str] = Field(default_factory=list)
    durations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalise(self) -> "SearchFilter":
        apply_filter_normalisation(self)
        self.sizes = _normalise_comparisons(self.sizes, _parse_size, "sizes")
        self.durations = _normalise_comparisons(
            self.durations, _parse_duration, "durations"
        )
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
