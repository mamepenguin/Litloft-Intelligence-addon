"""GUI-driven runtime overrides for the ``features`` config section.

The admin GUI lets operators flip the eight feature gates without
hand-editing search-config.yml. Persisted to
``/intelligence-data/features-overrides.json`` so the choice survives
restarts.

Allowed fields (all optional — absent ⇒ keep baseline from
search-config.yml):

* ``indexing``           — bool
* ``search``             — bool
* ``rag``                — bool
* ``auto_tags``          — "false" | "manual" | "on_index"
* ``summaries``          — same enum
* ``detailed_summaries`` — same enum
* ``transcript_refine``  — same enum
* ``vision_describe``    — same enum

Defence in depth: any other key in the on-disk file is silently
dropped during read so a buggy writer cannot smuggle additional
config knobs past the GUI scope.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from app._overrides_io import (
    delete_override_file,
    overrides_path as _overrides_path,
    read_override_payload,
    write_override_payload,
)

logger = logging.getLogger(__name__)

OVERRIDES_FILENAME = "features-overrides.json"
SCHEMA_VERSION = 1

TRISTATE_ENUM = ("false", "manual", "on_index")
BOOL_FIELDS = ("indexing", "search", "rag")
ENUM_FIELDS = (
    "auto_tags",
    "summaries",
    "detailed_summaries",
    "transcript_refine",
    "vision_describe",
)


@dataclass(frozen=True)
class FeaturesOverrides:
    """Optional override for each ``features.*`` field.

    ``None`` means "absent from the file" — caller keeps baseline.
    """

    indexing: bool | None = None
    search: bool | None = None
    rag: bool | None = None
    auto_tags: str | None = None
    summaries: str | None = None
    detailed_summaries: str | None = None
    transcript_refine: str | None = None
    vision_describe: str | None = None


def overrides_path(data_dir: str | os.PathLike[str] | None = None) -> Path:
    return _overrides_path(OVERRIDES_FILENAME, data_dir)


def read_overrides(
    data_dir: str | os.PathLike[str] | None = None,
) -> FeaturesOverrides | None:
    raw = read_override_payload(
        OVERRIDES_FILENAME,
        schema_version=SCHEMA_VERSION,
        data_dir=data_dir,
    )
    if raw is None:
        return None
    return _from_raw(raw)


def write_overrides(
    overrides: FeaturesOverrides,
    *,
    data_dir: str | os.PathLike[str] | None = None,
    updated_at: str | None = None,
) -> Path:
    payload: dict[str, Any] = {}
    for f in fields(overrides):
        value = getattr(overrides, f.name)
        if value is not None:
            payload[f.name] = value
    return write_override_payload(
        OVERRIDES_FILENAME,
        payload,
        schema_version=SCHEMA_VERSION,
        data_dir=data_dir,
        updated_at=updated_at,
    )


def delete_overrides(
    data_dir: str | os.PathLike[str] | None = None,
) -> bool:
    return delete_override_file(OVERRIDES_FILENAME, data_dir)


def _from_raw(raw: dict[str, Any]) -> FeaturesOverrides:
    """Build a typed dataclass from a parsed JSON dict, dropping
    unknown / wrongly-typed keys with a warning."""
    kwargs: dict[str, Any] = {}
    for name in BOOL_FIELDS:
        if name in raw:
            value = raw[name]
            if isinstance(value, bool):
                kwargs[name] = value
            else:
                logger.warning(
                    "Dropping non-bool features override %s=%r", name, value,
                )
    for name in ENUM_FIELDS:
        if name in raw:
            value = raw[name]
            if isinstance(value, str) and value in TRISTATE_ENUM:
                kwargs[name] = value
            else:
                logger.warning(
                    "Dropping invalid features override %s=%r "
                    "(expected one of %s)",
                    name, value, TRISTATE_ENUM,
                )
    return FeaturesOverrides(**kwargs)


def merge_into_dict(
    base: dict[str, Any],
    overrides: FeaturesOverrides | None,
) -> dict[str, Any]:
    """Apply ``overrides`` on top of a baseline dict from
    search-config.yml's ``features`` section. Returns a new dict.
    """
    out = dict(base)
    if overrides is None:
        return out
    for f in fields(overrides):
        value = getattr(overrides, f.name)
        if value is not None:
            out[f.name] = value
    return out
