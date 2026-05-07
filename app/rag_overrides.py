"""GUI-driven runtime overrides for ``rag.personal_history.enabled``
and ``rag.category_expansion.enabled``.

These two booleans are the privacy / cost levers operators flip
without touching the rest of the heavily-tuned ``rag`` block. Other
``rag.*`` knobs (top_k, max_context_chars*, hierarchical retrieval
parameters, …) stay file-only. Persisted to
``/intelligence-data/rag-overrides.json``.

The on-disk shape is flat, not nested: the GUI talks JSON keys
``personal_history_enabled`` / ``category_expansion_enabled`` so
schema_version migrations down the line don't have to reshape a
nested dict. The merge helper translates them onto the typed
``rag.personal_history.enabled`` / ``rag.category_expansion.enabled``
paths.
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

OVERRIDES_FILENAME = "rag-overrides.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RagOverrides:
    """Optional booleans for the two RAG sub-feature gates."""

    personal_history_enabled: bool | None = None
    category_expansion_enabled: bool | None = None


def overrides_path(data_dir: str | os.PathLike[str] | None = None) -> Path:
    return _overrides_path(OVERRIDES_FILENAME, data_dir)


def read_overrides(
    data_dir: str | os.PathLike[str] | None = None,
) -> RagOverrides | None:
    raw = read_override_payload(
        OVERRIDES_FILENAME,
        schema_version=SCHEMA_VERSION,
        data_dir=data_dir,
    )
    if raw is None:
        return None
    return _from_raw(raw)


def write_overrides(
    overrides: RagOverrides,
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


def _from_raw(raw: dict[str, Any]) -> RagOverrides:
    kwargs: dict[str, Any] = {}
    for name in ("personal_history_enabled", "category_expansion_enabled"):
        if name in raw:
            value = raw[name]
            if isinstance(value, bool):
                kwargs[name] = value
            else:
                logger.warning(
                    "Dropping non-bool rag override %s=%r", name, value,
                )
    return RagOverrides(**kwargs)


def merge_into_rag_dict(
    base: dict[str, Any],
    overrides: RagOverrides | None,
) -> dict[str, Any]:
    """Apply ``overrides`` on top of a baseline dict from the ``rag``
    section. Translates flat override keys back into the nested
    ``rag.{personal_history,category_expansion}.enabled`` shape used
    by the search-config.yml schema. Returns a new dict.
    """
    out = dict(base)
    if overrides is None:
        return out
    if overrides.personal_history_enabled is not None:
        ph = dict(out.get("personal_history") or {})
        ph["enabled"] = overrides.personal_history_enabled
        out["personal_history"] = ph
    if overrides.category_expansion_enabled is not None:
        ce = dict(out.get("category_expansion") or {})
        ce["enabled"] = overrides.category_expansion_enabled
        out["category_expansion"] = ce
    return out
