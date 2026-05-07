"""GUI-driven runtime overrides for the ``llm`` config section.

Edits the five operator-facing knobs that determine which LLM the
addon talks to:

* ``provider``         — "ollama" | "openai_compatible" | "disabled"
* ``base_url``         — string (URL, no validation here — operator's
                          choice may include LAN hostnames or IPs)
* ``model``            — string (provider-specific model id)
* ``output_language``  — "auto" | "ja" | "en"
* ``vision_model``     — string (paired vision model, may be empty)

Persisted to ``/intelligence-data/llm-overrides.json``. Tuning knobs
(temperature, retry_*, request_timeout_seconds, vision_max_tokens,
etc.) stay file-only; the GUI does not edit them. ``api_key`` is
deliberately excluded — secrets live in environment variables, not
in JSON files inside the data volume.
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

OVERRIDES_FILENAME = "llm-overrides.json"
SCHEMA_VERSION = 1

PROVIDER_ENUM = ("disabled", "ollama", "openai_compatible")
OUTPUT_LANGUAGE_ENUM = ("auto", "ja", "en")
STRING_FIELDS = ("base_url", "model", "vision_model")


@dataclass(frozen=True)
class LLMOverrides:
    """Optional override for each ``llm.*`` field the GUI exposes.

    ``None`` means "absent from the file" — caller keeps baseline.
    Empty-string values for ``base_url`` / ``model`` / ``vision_model``
    explicitly clear the baseline (e.g. switching provider to
    ``disabled`` legitimately wants to drop the URL).
    """

    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    output_language: str | None = None
    vision_model: str | None = None


def overrides_path(data_dir: str | os.PathLike[str] | None = None) -> Path:
    return _overrides_path(OVERRIDES_FILENAME, data_dir)


def read_overrides(
    data_dir: str | os.PathLike[str] | None = None,
) -> LLMOverrides | None:
    raw = read_override_payload(
        OVERRIDES_FILENAME,
        schema_version=SCHEMA_VERSION,
        data_dir=data_dir,
    )
    if raw is None:
        return None
    return _from_raw(raw)


def write_overrides(
    overrides: LLMOverrides,
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


def _from_raw(raw: dict[str, Any]) -> LLMOverrides:
    kwargs: dict[str, Any] = {}
    if "provider" in raw:
        value = raw["provider"]
        if isinstance(value, str) and value in PROVIDER_ENUM:
            kwargs["provider"] = value
        else:
            logger.warning(
                "Dropping invalid llm override provider=%r "
                "(expected one of %s)",
                value, PROVIDER_ENUM,
            )
    if "output_language" in raw:
        value = raw["output_language"]
        if isinstance(value, str) and value in OUTPUT_LANGUAGE_ENUM:
            kwargs["output_language"] = value
        else:
            logger.warning(
                "Dropping invalid llm override output_language=%r "
                "(expected one of %s)",
                value, OUTPUT_LANGUAGE_ENUM,
            )
    for name in STRING_FIELDS:
        if name in raw:
            value = raw[name]
            if isinstance(value, str):
                kwargs[name] = value
            else:
                logger.warning(
                    "Dropping non-string llm override %s=%r", name, value,
                )
    return LLMOverrides(**kwargs)


def merge_into_dict(
    base: dict[str, Any],
    overrides: LLMOverrides | None,
) -> dict[str, Any]:
    """Apply ``overrides`` on top of a baseline dict from
    search-config.yml's ``llm`` section. Returns a new dict.
    """
    out = dict(base)
    if overrides is None:
        return out
    for f in fields(overrides):
        value = getattr(overrides, f.name)
        if value is not None:
            out[f.name] = value
    return out
