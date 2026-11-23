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
MAX_MODIFIED_SECONDS = 253402300799

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


def _parse_modified(item: str) -> str:
    match = _WIRE_FORM.match(item)
    if match is None:
        raise ValueError(f"invalid modified comparison: {item!r}")
    operator, amount = match.groups()
    if not _SIZE_NUMBER.fullmatch(amount) or int(amount) > MAX_MODIFIED_SECONDS:
        raise ValueError(f"invalid modified comparison: {item!r}")
    return f"{operator}:{int(amount)}"


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
    {"video", "image", "audio", "document", "archive", "other", "text", "pdf", "epub"}
)
_KIND_ALIASES = {"markdown": "text"}


class InvalidFilter(ValueError):
    """The ``filter`` query parameter is not a well-formed filter."""


class FilterUnavailable(RuntimeError):
    """The core did not answer ``/files/matching`` with a list of file ids."""


def _normalise_tags(values: list[str], label: str) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for raw in values:
        tag = raw.strip()
        if not tag:
            continue
        if len(tag) > TAG_MAX_LENGTH:
            raise ValueError(
                f"{label}: a tag is at most {TAG_MAX_LENGTH} characters"
            )
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag)
    if len(tags) > TAG_MAX_COUNT:
        raise ValueError(f"{label}: at most {TAG_MAX_COUNT} tags")
    return tags


def _normalise_kinds(values: list[str], label: str) -> list[str]:
    kinds: list[str] = []
    seen: set[str] = set()
    for raw_kind in values:
        kind = _KIND_ALIASES.get(raw_kind, raw_kind)
        if kind not in KINDS:
            raise ValueError(f"{label}: unknown kind: {raw_kind}")
        if kind in seen:
            continue
        seen.add(kind)
        kinds.append(kind)
    if len(kinds) > KIND_MAX_COUNT:
        raise ValueError(f"{label}: at most {KIND_MAX_COUNT} kinds")
    return kinds


def _normalise_folders(values: list[str], label: str) -> list[str]:
    folders: list[str] = []
    for raw_folder in values:
        folder = unicodedata.normalize("NFC", raw_folder).strip("/")
        if (
            not folder
            or "\x00" in folder
            or any(segment in ("", ".", "..") for segment in folder.split("/"))
        ):
            raise ValueError(f"{label}: invalid folder: {raw_folder!r}")
        if len(folder) > FOLDER_MAX_LENGTH:
            raise ValueError(
                f"{label}: a folder is at most {FOLDER_MAX_LENGTH} characters"
            )
        if folder not in folders:
            folders.append(folder)
    if len(folders) > FOLDER_MAX_COUNT:
        raise ValueError(f"{label}: at most {FOLDER_MAX_COUNT} folders")
    return folders


def _normalise_collections(values: list[str], label: str) -> list[str]:
    collections: list[str] = []
    seen: set[str] = set()
    for raw_collection in values:
        name = raw_collection.strip()
        if not name:
            continue
        if len(name) > COLLECTION_MAX_LENGTH:
            raise ValueError(
                f"{label}: a collection is at most "
                f"{COLLECTION_MAX_LENGTH} characters"
            )
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        collections.append(name)
    if len(collections) > COLLECTION_MAX_COUNT:
        raise ValueError(f"{label}: at most {COLLECTION_MAX_COUNT} collections")
    return collections


def apply_filter_normalisation(model: BaseModel) -> None:
    """Normalise the four condition lists on ``model`` in place.

    Shared by the search ``filter`` and the Ask ``scope`` so the two name the
    same files for the same values: a blank tag is dropped, tags and
    collections are deduplicated case-insensitively, kinds go through the
    ``markdown`` alias, folders are NFC-normalised and trimmed of ``/``, then
    the caps apply. Raises ``ValueError`` for a value either model rejects.
    """
    model.tags = _normalise_tags(model.tags, "tags")
    model.kinds = _normalise_kinds(model.kinds, "kinds")
    model.folders = _normalise_folders(model.folders, "folders")
    model.collections = _normalise_collections(model.collections, "collections")


class SearchFilter(BaseModel):
    """The parsed operators of a search query, as the core's request model takes them.

    Normalisation matches core's: a blank tag is dropped, tags are deduplicated
    case-insensitively, kinds after the ``markdown`` alias, folders are NFC-
    normalised and trimmed of ``/``, collection names are trimmed and deduped
    case-insensitively, then the caps apply. Each ``exclude_`` list follows its
    positive counterpart's rule and holds its own cap.
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
    modified: list[str] = Field(default_factory=list)
    exclude_tags: list[str] = Field(default_factory=list)
    exclude_kinds: list[str] = Field(default_factory=list)
    exclude_folders: list[str] = Field(default_factory=list)
    exclude_collections: list[str] = Field(default_factory=list)
    has_tag: StrictBool | None = None
    has_collection: StrictBool | None = None

    @model_validator(mode="after")
    def _normalise(self) -> "SearchFilter":
        apply_filter_normalisation(self)
        self.exclude_tags = _normalise_tags(self.exclude_tags, "exclude_tags")
        self.exclude_kinds = _normalise_kinds(self.exclude_kinds, "exclude_kinds")
        self.exclude_folders = _normalise_folders(
            self.exclude_folders, "exclude_folders"
        )
        self.exclude_collections = _normalise_collections(
            self.exclude_collections, "exclude_collections"
        )
        self.sizes = _normalise_comparisons(self.sizes, _parse_size, "sizes")
        self.durations = _normalise_comparisons(
            self.durations, _parse_duration, "durations"
        )
        self.modified = _normalise_comparisons(
            self.modified, _parse_modified, "modified"
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


_OMITTED_WHEN_UNSET = (
    "exclude_tags",
    "exclude_kinds",
    "exclude_folders",
    "exclude_collections",
    "has_tag",
    "has_collection",
)


async def fetch_matching_file_ids(drive: str, search_filter: SearchFilter) -> list[str]:
    """Ask the core for ``drive``'s active file ids that satisfy ``search_filter``.

    Raises ``FilterUnavailable`` for a transport error, a timeout, a non-200
    answer, or a body without a list of string ids, so the route can answer 503
    instead of falling back to an unfiltered search.
    """
    url = f"{_internal_api_base_url().rstrip('/')}{_MATCHING_PATH}"
    payload = {"drive": drive, **search_filter.model_dump()}
    # Sent only when set, unlike the positive keys: the core's defaults say the
    # same as an empty list or a null flag, and an unset key must not turn a
    # filter into a different request.
    for key in _OMITTED_WHEN_UNSET:
        if payload[key] is None or payload[key] == []:
            del payload[key]

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


def scope_member_ids(
    without_collections: list[str],
    with_collections: list[str] | None,
    file_ids: list[str] | None,
) -> list[str]:
    """An Ask scope's file set: ``B ∪ (A ∩ file_ids)``.

    ``without_collections`` is A, the matching call carrying the folder / tag /
    kind conditions, so the ids it holds are the request drive's active files.
    ``with_collections`` is B, the same call with the collections condition, or
    None when the scope names no collection. ``file_ids`` None means the scope
    carries none; an empty list names no file and is not an absent one.
    """
    if file_ids is None:
        return list(
            without_collections if with_collections is None else with_collections
        )
    keep = set(file_ids)
    narrowed = [file_id for file_id in without_collections if file_id in keep]
    if with_collections is None:
        return narrowed
    seen = set(with_collections)
    return list(with_collections) + [
        file_id for file_id in narrowed if file_id not in seen
    ]
