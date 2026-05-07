"""Shared atomic-rename helpers for GUI-managed override JSON files.

Each of ``transcription_overrides`` / ``features_overrides`` /
``llm_overrides`` / ``rag_overrides`` writes a tiny JSON file into the
intelligence data volume to remember an operator's edits across container
restarts. The mechanics are identical:

* read may return ``None`` on missing or malformed file (caller falls back
  to the search-config.yml baseline);
* writes go through ``*.tmp`` + ``os.replace`` so a crash mid-flush never
  leaves a half-written file the loader trusts;
* deletes are idempotent (no-op when the file is already absent so the
  GUI's reset button can be clicked repeatedly).

The file format is always ``{"schema_version": int, "updated_at": iso8601,
...payload}``. ``schema_version`` lets a rolled-back image ignore a
forward-incompatible payload instead of silently honouring it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def overrides_path(
    filename: str,
    data_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the canonical path of an override file inside the
    intelligence data volume."""
    base = Path(
        data_dir
        or os.environ.get("INTELLIGENCE_DATA_DIR", "/intelligence-data")
    )
    return base / filename


def read_override_payload(
    filename: str,
    *,
    schema_version: int,
    data_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """Load and validate an override file's raw payload.

    Returns ``None`` when the file does not exist, fails to parse,
    or carries an unsupported ``schema_version``. Logs at WARN level
    on every malformed-input path so operators see why their saved
    settings are not taking effect.

    The caller is responsible for translating the dict into a typed
    dataclass and for applying field-level validation.
    """
    path = overrides_path(filename, data_dir)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Ignoring malformed overrides at %s: %s", path, exc,
        )
        return None
    if not isinstance(raw, dict):
        logger.warning(
            "Ignoring overrides at %s: top-level must be a JSON object",
            path,
        )
        return None
    schema = raw.get("schema_version")
    if schema is not None and schema != schema_version:
        logger.warning(
            "Ignoring overrides at %s: unknown schema_version %r "
            "(this build supports %d)",
            path, schema, schema_version,
        )
        return None
    return raw


def write_override_payload(
    filename: str,
    payload: dict[str, Any],
    *,
    schema_version: int,
    data_dir: str | os.PathLike[str] | None = None,
    updated_at: str | None = None,
) -> Path:
    """Persist ``payload`` to ``filename`` using atomic rename.

    The supplied ``payload`` should already contain only the section
    keys; this helper adds ``schema_version`` and (optionally)
    ``updated_at`` before writing.
    """
    path = overrides_path(filename, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    full_payload: dict[str, Any] = {"schema_version": schema_version}
    if updated_at is not None:
        full_payload["updated_at"] = updated_at
    full_payload.update(payload)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(full_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


def delete_override_file(
    filename: str,
    data_dir: str | os.PathLike[str] | None = None,
) -> bool:
    """Remove the override file. Returns True if a file was removed."""
    path = overrides_path(filename, data_dir)
    if not path.is_file():
        return False
    try:
        path.unlink()
    except OSError as exc:
        logger.warning("Could not delete %s: %s", path, exc)
        return False
    return True
