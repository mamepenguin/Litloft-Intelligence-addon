"""GUI-driven runtime override for the ``models.text_embedding`` knob.

Edits the single operator-facing field that decides which text
embedding model the addon uses:

* ``text_embedding`` — a HuggingFace model id, validated against the
  ``_MODEL_DIMS`` allowlist in ``app/workers/embedder.py``.

Persisted to ``/intelligence-data/embedding-overrides.json``. Sibling
``models.*`` knobs (``clip``, ``whisper``, ``blip*``, prefixes) stay
file-only; the GUI does not edit them.

The allowlist is the embedder's ``_MODEL_DIMS`` keys, *not* a
hand-maintained copy that can drift. A value not in that dict — a
typo, an unsupported model — is dropped on read with a WARN and the
field stays ``None`` so the caller keeps the ``search-config.yml``
baseline. This makes the real Ask-breaking incident (a typo silently
falling back to a 384-dim baseline, hako ``JxHJMk2V5bu603gr1HkAZ``)
structurally impossible: spec invariant §2.1-4 (no silent dim-384
fallback).
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

OVERRIDES_FILENAME = "embedding-overrides.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EmbeddingOverrides:
    """Optional override for the ``models.text_embedding`` field.

    ``None`` means "absent from the file" — the caller keeps the
    ``search-config.yml`` baseline (never a silent 384-dim fallback).
    """

    text_embedding: str | None = None


def _allowed_models() -> set[str]:
    """The allowlist IS the embedder's ``_MODEL_DIMS`` keys.

    Imported lazily so test-time path/module patching keeps working
    and so importing this module never pulls in the heavy embedder
    transitive deps unless validation actually runs.
    """
    from app.workers.embedder import _MODEL_DIMS

    return set(_MODEL_DIMS)


def overrides_path(data_dir: str | os.PathLike[str] | None = None) -> Path:
    return _overrides_path(OVERRIDES_FILENAME, data_dir)


def read_overrides(
    data_dir: str | os.PathLike[str] | None = None,
) -> EmbeddingOverrides | None:
    raw = read_override_payload(
        OVERRIDES_FILENAME,
        schema_version=SCHEMA_VERSION,
        data_dir=data_dir,
    )
    if raw is None:
        return None
    return _from_raw(raw)


def write_overrides(
    overrides: EmbeddingOverrides,
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


def _from_raw(raw: dict[str, Any]) -> EmbeddingOverrides:
    kwargs: dict[str, Any] = {}
    if "text_embedding" in raw:
        value = raw["text_embedding"]
        if isinstance(value, str) and value in _allowed_models():
            kwargs["text_embedding"] = value
        else:
            logger.warning(
                "Dropping invalid models.text_embedding override %r "
                "(not in the embedder allowlist) — keeping the "
                "search-config.yml baseline (no silent dim-384 "
                "fallback, invariant §2.1-4)",
                value,
            )
    return EmbeddingOverrides(**kwargs)


def merge_into_dict(
    base: dict[str, Any],
    overrides: EmbeddingOverrides | None,
) -> dict[str, Any]:
    """Apply ``overrides`` on top of a baseline dict from
    search-config.yml's ``models`` section. Returns a new dict;
    the input ``base`` is never mutated.
    """
    out = dict(base)
    if overrides is None:
        return out
    for f in fields(overrides):
        value = getattr(overrides, f.name)
        if value is not None:
            out[f.name] = value
    return out
