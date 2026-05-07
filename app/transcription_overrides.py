"""GUI-driven runtime overrides for the transcription provider config.

Phase 2D writes user-selected provider / language_hint / hotwords to
``/intelligence-data/transcription-overrides.json`` so the choice
persists across container restarts without modifying the immutable
``search-config.yml`` ship file (which is mounted read-only and is
out of scope for the core admin GUI per hako
``EZSuSEfDHFXkz9MrHdXF9``).

Design constraints baked in here:

* The overrides file is **optional**. Missing or malformed → caller
  gets ``None`` and falls back to the search-config.yml baseline.
  intelligence must always boot, even with corrupt overrides.
* Only the three fields the admin GUI is allowed to change are
  honoured (``provider``, ``language_hint``, ``hotwords``). Extra
  keys are silently dropped — defence in depth so a broken writer
  cannot mutate provider sub-config (model / base_url / etc.) here.
* ``language_hint`` distinguishes absent (key not present → keep
  baseline) from empty string (key present with ``""`` → caller
  set "no hint", which overrides the baseline).
* Writes use atomic rename (``*.tmp`` → ``os.replace``) so a partial
  flush cannot leave the loader staring at half-written JSON.
* Unknown ``schema_version`` values are ignored with a warning so
  rollbacks to older intelligence images do not silently honour
  forward-incompatible payloads.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OVERRIDES_FILENAME = "transcription-overrides.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TranscriptionOverrides:
    """The three knobs the admin GUI is allowed to touch.

    Each field's ``None`` means "absent from the overrides file" —
    callers should keep the baseline value. ``language_hint=""``
    explicitly sets "no hint" and overrides the baseline.
    """

    provider: str | None = None
    language_hint: str | None = None
    hotwords: tuple[str, ...] | None = None


def overrides_path(data_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the canonical path of the overrides file inside the
    intelligence data volume."""
    base = Path(
        data_dir
        or os.environ.get("INTELLIGENCE_DATA_DIR", "/intelligence-data")
    )
    return base / OVERRIDES_FILENAME


def read_overrides(
    data_dir: str | os.PathLike[str] | None = None,
) -> TranscriptionOverrides | None:
    """Load and validate the overrides file.

    Returns ``None`` when the file does not exist, fails to parse,
    or carries an unsupported schema_version. Logs at WARN level on
    every malformed-input path so operators see why their saved
    settings are not taking effect.
    """
    path = overrides_path(data_dir)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Ignoring malformed transcription overrides at %s: %s",
            path, exc,
        )
        return None
    if not isinstance(raw, dict):
        logger.warning(
            "Ignoring transcription overrides at %s: top-level must "
            "be a JSON object",
            path,
        )
        return None
    schema = raw.get("schema_version")
    if schema is not None and schema != SCHEMA_VERSION:
        logger.warning(
            "Ignoring transcription overrides at %s: unknown "
            "schema_version %r (this build supports %d)",
            path, schema, SCHEMA_VERSION,
        )
        return None

    return _from_raw(raw)


def write_overrides(
    overrides: TranscriptionOverrides,
    *,
    data_dir: str | os.PathLike[str] | None = None,
    updated_at: str | None = None,
) -> Path:
    """Persist ``overrides`` to the data dir using atomic rename.

    ``updated_at`` is ISO-8601 — usually the caller passes
    ``datetime.utcnow().isoformat() + "Z"``; if omitted we generate
    it. The function is sync because there is no benefit to making
    a single small JSON write awaitable.
    """
    path = overrides_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    if updated_at is not None:
        payload["updated_at"] = updated_at
    if overrides.provider is not None:
        payload["provider"] = overrides.provider
    if overrides.language_hint is not None:
        # Empty string is meaningful; keep it.
        payload["language_hint"] = overrides.language_hint
    if overrides.hotwords is not None:
        payload["hotwords"] = list(overrides.hotwords)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


def delete_overrides(
    data_dir: str | os.PathLike[str] | None = None,
) -> bool:
    """Remove the overrides file. Returns True if a file was removed."""
    path = overrides_path(data_dir)
    if not path.is_file():
        return False
    try:
        path.unlink()
    except OSError as exc:
        logger.warning("Could not delete %s: %s", path, exc)
        return False
    return True


def merge_into_dict(
    base: dict[str, Any],
    overrides: TranscriptionOverrides | None,
) -> dict[str, Any]:
    """Return a new dict applying ``overrides`` on top of ``base``.

    Only ``provider`` / ``language_hint`` / ``hotwords`` are touched
    — provider sub-configs (``deepgram``, ``openai_compatible`` …)
    are never altered by the overrides path.

    A ``language_hint=""`` explicitly overwrites the baseline with
    the empty string. A ``language_hint=None`` means the file did
    not include the key, so the baseline is kept.
    """
    out = dict(base)
    if overrides is None:
        return out
    if overrides.provider is not None:
        out["provider"] = overrides.provider
    if overrides.language_hint is not None:
        out["language_hint"] = overrides.language_hint
    if overrides.hotwords is not None:
        out["hotwords"] = list(overrides.hotwords)
    return out


def _from_raw(raw: dict[str, Any]) -> TranscriptionOverrides:
    """Project a parsed JSON dict into the typed dataclass.

    Skips fields that fail type checks (defensive — a corrupt write
    should degrade to "use baseline" rather than fail-loud at
    intelligence boot).
    """
    provider = raw.get("provider")
    if provider is not None and not isinstance(provider, str):
        provider = None

    language_hint = raw.get("language_hint")
    if language_hint is not None and not isinstance(language_hint, str):
        language_hint = None

    hotwords_raw = raw.get("hotwords")
    hotwords: tuple[str, ...] | None
    if hotwords_raw is None:
        hotwords = None
    elif isinstance(hotwords_raw, list) and all(
        isinstance(w, str) for w in hotwords_raw
    ):
        hotwords = tuple(hotwords_raw)
    else:
        hotwords = None

    return TranscriptionOverrides(
        provider=provider,
        language_hint=language_hint,
        hotwords=hotwords,
    )
