"""Admin endpoints for the intelligence addon.

Phase 2D: lets the core ``/admin/settings`` GUI inspect and update
the runtime transcription provider selection without touching the
read-only ``search-config.yml`` shipped with the addon. The GUI side
of the contract:

* ``GET /admin/transcription`` — current merged config + the list of
  provider names + which API keys are present in the environment.
  Reads the on-disk overrides file authoritatively so a freshly-
  saved value is reflected even before intelligence restarts (the
  imported ``settings.transcription`` value is frozen to the value
  that existed at module import).
* ``PUT /admin/transcription`` — validates the payload, persists it
  to ``/intelligence-data/transcription-overrides.json`` via atomic
  rename, and POSTs to the core's
  ``/api/internal/restart-pending`` endpoint so the user sees the
  restart banner.

The route is gated by the host addon proxy with
``pre_check: {type: admin}``; this module does not re-implement the
master-viewer judge.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import app.config as config
from app.transcription_overrides import (
    TranscriptionOverrides,
    overrides_path,
    read_overrides,
    write_overrides,
)
from app.workers.transcription import build_inner_provider, get_provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

# All providers the eval harness / runtime knows about. Kept explicit
# so a typo in ``payload.provider`` produces a 400 with the available
# list rather than failing later inside ``get_provider``.
ALL_PROVIDERS: tuple[str, ...] = (
    "whisper_local",
    "openai_compatible",
    "deepgram",
    "elevenlabs_scribe",
    "assemblyai",
    "gemini",
)

# Env vars whose presence indicates a provider can be activated. None
# means the provider does not need an env (whisper_local).
_PROVIDER_API_KEY_ENV: dict[str, str | None] = {
    "whisper_local": None,
    "openai_compatible": "OPENAI_API_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
    "elevenlabs_scribe": "ELEVENLABS_API_KEY",
    "assemblyai": "ASSEMBLYAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

# RFC 5646 lightweight tag check (BCP-47) capped at the spec's max.
# Allowing 35 chars covers ``en-GB-oxendict`` (14) plus margin.
_LANG_TAG_RE = re.compile(r"^[a-zA-Z]{2,3}(-[a-zA-Z0-9]{2,8})*$")
_LANG_TAG_MAX_LEN = 35

_HOTWORD_MAX_LEN = 64
_HOTWORDS_MAX_COUNT = 500


class TranscriptionUpdate(BaseModel):
    """Payload accepted by ``PUT /admin/transcription``.

    Pydantic enforces field types; semantic validation (provider
    enum, API-key prerequisite, BCP-47 tag, hotwords length) lives
    in :func:`_validate` so the error messages are user-friendly.
    """

    provider: str
    language_hint: str = ""
    hotwords: list[str] = Field(default_factory=list)


@router.get("/transcription")
async def get_transcription_config() -> dict[str, Any]:
    """Return the current effective transcription config + status.

    Composes the search-config.yml baseline with the on-disk
    overrides so the GUI sees what the *next* startup will use, not
    the cached ``settings.transcription`` from this process's import
    time.
    """
    base = config.settings.transcription
    overrides = read_overrides()

    if overrides is not None and overrides.provider is not None:
        provider = overrides.provider
    else:
        provider = base.provider
    if overrides is not None and overrides.language_hint is not None:
        language_hint = overrides.language_hint
    else:
        language_hint = base.language_hint
    if overrides is not None and overrides.hotwords is not None:
        hotwords = list(overrides.hotwords)
    else:
        hotwords = list(base.hotwords)

    return {
        "provider": provider,
        "language_hint": language_hint,
        "hotwords": hotwords,
        "available_providers": list(ALL_PROVIDERS),
        "api_keys_present": _check_api_keys(),
        "overrides_present": overrides is not None,
        "search_config_summary": _frozen_subconfig_summary(),
    }


@router.put("/transcription")
async def update_transcription_config(
    payload: TranscriptionUpdate,
) -> dict[str, Any]:
    """Validate and persist new transcription overrides."""
    _validate(payload)
    overrides = TranscriptionOverrides(
        provider=payload.provider,
        language_hint=payload.language_hint,
        hotwords=tuple(payload.hotwords),
    )
    written_at = datetime.now(UTC).isoformat()
    path = write_overrides(overrides, updated_at=written_at)
    logger.info(
        "Transcription overrides saved (provider=%s) at %s",
        payload.provider, path,
    )
    notify_status = await _notify_core_restart_pending()
    return {
        "status": "saved",
        "restart_required": True,
        "core_notified": notify_status,
    }


def _validate(payload: TranscriptionUpdate) -> None:
    """All-error validation per hako Ij1KDoR0QB4ElHikrxTZ8."""
    if payload.provider not in ALL_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown provider {payload.provider!r}. "
                f"Choose one of {list(ALL_PROVIDERS)}"
            ),
        )

    env_key = _PROVIDER_API_KEY_ENV.get(payload.provider)
    if env_key is not None and not os.getenv(env_key):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{payload.provider} requires the {env_key} environment "
                "variable. Add it to .env, run `docker compose up -d`, "
                "and resave."
            ),
        )

    if payload.language_hint:
        if len(payload.language_hint) > _LANG_TAG_MAX_LEN:
            raise HTTPException(
                status_code=400,
                detail=f"language_hint must be ≤{_LANG_TAG_MAX_LEN} chars",
            )
        if not _LANG_TAG_RE.match(payload.language_hint):
            raise HTTPException(
                status_code=400,
                detail=(
                    "language_hint must look like a BCP-47 tag "
                    "(e.g. 'ja', 'en-US', 'zh-Hant-HK')"
                ),
            )

    if len(payload.hotwords) > _HOTWORDS_MAX_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"hotwords list must have ≤{_HOTWORDS_MAX_COUNT} entries",
        )
    for word in payload.hotwords:
        if len(word) > _HOTWORD_MAX_LEN:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"each hotword must be ≤{_HOTWORD_MAX_LEN} chars "
                    f"(got {len(word)})"
                ),
            )
        if "\n" in word or "\r" in word or "\x00" in word:
            raise HTTPException(
                status_code=400,
                detail="hotwords must not contain control characters",
            )


def _check_api_keys() -> dict[str, bool]:
    """Map provider name → ``True`` when its API-key env is set.

    Values are ``True`` / ``False`` only; we do NOT return the actual
    secret. ``whisper_local`` always reports ``True`` because it
    needs no key.
    """
    out: dict[str, bool] = {}
    for name, env in _PROVIDER_API_KEY_ENV.items():
        if env is None:
            out[name] = True
        else:
            out[name] = bool(os.getenv(env, ""))
    return out


def _frozen_subconfig_summary() -> dict[str, Any]:
    """Read-only view of the per-provider sub-config from search-config.yml.

    The admin GUI surfaces these so the operator can see the model /
    base_url / etc. that ship with the addon. Editing them stays a
    file-edit job per hako EZSuSEfDHFXkz9MrHdXF9.
    """
    base = config.settings.transcription
    return {
        "whisper_local": {
            "model": base.whisper_local.model,
        },
        "openai_compatible": {
            "model": base.openai_compatible.model,
            "base_url": base.openai_compatible.base_url,
        },
        "deepgram": {
            "model": base.deepgram.model,
        },
        "elevenlabs_scribe": {
            "model_id": base.elevenlabs_scribe.model_id,
        },
        "assemblyai": {
            "model": base.assemblyai.model,
        },
        "gemini": {
            "model": base.gemini.model,
            "output_language": base.gemini.output_language,
        },
    }


async def _notify_core_restart_pending() -> str:
    """POST to the core's restart-pending sentinel endpoint.

    Returns one of:

    * ``"ok"`` — sentinel touched
    * ``"unconfigured"`` — env vars missing; cannot reach the core
    * ``"error"`` — request failed (network, 5xx, secret rejected)

    Failures here do **not** roll back the saved overrides — the
    user can manually run ``docker compose restart`` or wait for the
    next time the core comes up.
    """
    base = os.environ.get("HOMEVAULT_INTERNAL_URL", "").rstrip("/")
    secret = os.environ.get("CORE_INTERNAL_SECRET", "")
    if not base or not secret:
        logger.warning(
            "Cannot notify core: HOMEVAULT_INTERNAL_URL or "
            "CORE_INTERNAL_SECRET not set"
        )
        return "unconfigured"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{base}/api/internal/restart-pending",
                headers={"X-Internal-Secret": secret},
                json={
                    "source": "intelligence",
                    "reason": "transcription config updated",
                },
            )
        if resp.status_code in (200, 204):
            return "ok"
        logger.warning(
            "Core rejected restart-pending notify: HTTP %s %s",
            resp.status_code, resp.text[:200],
        )
        return "error"
    except (httpx.NetworkError, httpx.TimeoutException) as exc:
        logger.warning("Could not notify core of restart_pending: %s", exc)
        return "error"
