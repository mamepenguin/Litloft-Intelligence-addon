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
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from app.schemas import FailedJobItem, FailedJobsResponse, ResolveFailedJobResponse

import app.config as config
from app.transcription_overrides import (
    TranscriptionOverrides,
    delete_overrides,
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


@router.delete("/transcription")
async def reset_transcription_config() -> dict[str, Any]:
    """Drop the GUI overrides so search-config.yml becomes authoritative.

    Lets operators undo a GUI change without hand-editing the
    /intelligence-data volume. No-op (still returns 200) when the
    overrides file is already absent — keeps the call idempotent so
    the GUI button can be clicked repeatedly without surfacing fake
    errors.
    """
    removed = delete_overrides()
    if removed:
        logger.info("Transcription overrides reset (file removed)")
    else:
        logger.info("Transcription overrides reset requested but no file present")
    notify_status = await _notify_core_restart_pending()
    return {
        "status": "reset",
        "removed": removed,
        "restart_required": removed,
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


# ---------------------------------------------------------------------------
# /admin/features
# ---------------------------------------------------------------------------


_FEATURES_BOOL_FIELDS = ("indexing", "search", "rag")
_FEATURES_ENUM_FIELDS = (
    "auto_tags",
    "summaries",
    "detailed_summaries",
    "transcript_refine",
    "vision_describe",
    "retrieval_keywords",
)
_FEATURES_TRISTATE = ("false", "manual", "on_index")


class FeaturesUpdate(BaseModel):
    """Optional override for each ``features.*`` field. Omitted keys
    inherit from the prior overrides file or from search-config.yml."""

    indexing: bool | None = None
    search: bool | None = None
    rag: bool | None = None
    auto_tags: str | None = None
    summaries: str | None = None
    detailed_summaries: str | None = None
    transcript_refine: str | None = None
    vision_describe: str | None = None
    retrieval_keywords: str | None = None


@router.get("/features")
async def get_features_config() -> dict[str, Any]:
    """Return the effective config = baseline + on-disk overrides.

    Reads the overrides file authoritatively (NOT the frozen
    ``config.settings.features`` from import time) so a freshly-saved
    GUI value is reflected immediately, even before the container
    restart actually swaps the in-memory config. Mirrors the
    ``GET /admin/transcription`` contract.
    """
    from app import features_overrides as fo

    base = config.settings.features
    overrides = fo.read_overrides()

    def resolve(field: str) -> Any:
        if overrides is not None:
            override_value = getattr(overrides, field)
            if override_value is not None:
                return override_value
        return getattr(base, field)

    return {
        "indexing": resolve("indexing"),
        "search": resolve("search"),
        "rag": resolve("rag"),
        "auto_tags": resolve("auto_tags"),
        "summaries": resolve("summaries"),
        "detailed_summaries": resolve("detailed_summaries"),
        "transcript_refine": resolve("transcript_refine"),
        "vision_describe": resolve("vision_describe"),
        "retrieval_keywords": resolve("retrieval_keywords"),
        "tristate_values": list(_FEATURES_TRISTATE),
        "overrides_present": overrides is not None,
    }


@router.put("/features")
async def update_features_config(payload: FeaturesUpdate) -> dict[str, Any]:
    from app import features_overrides as fo

    _validate_features(payload)
    overrides = fo.FeaturesOverrides(
        indexing=payload.indexing,
        search=payload.search,
        rag=payload.rag,
        auto_tags=payload.auto_tags,
        summaries=payload.summaries,
        detailed_summaries=payload.detailed_summaries,
        transcript_refine=payload.transcript_refine,
        vision_describe=payload.vision_describe,
        retrieval_keywords=payload.retrieval_keywords,
    )
    path = fo.write_overrides(
        overrides, updated_at=datetime.now(UTC).isoformat()
    )
    logger.info("Features overrides saved at %s", path)
    notify_status = await _notify_core_restart_pending()
    return {
        "status": "saved",
        "restart_required": True,
        "core_notified": notify_status,
    }


@router.delete("/features")
async def reset_features_config() -> dict[str, Any]:
    from app import features_overrides as fo

    removed = fo.delete_overrides()
    notify_status = await _notify_core_restart_pending()
    return {
        "status": "reset",
        "removed": removed,
        "restart_required": removed,
        "core_notified": notify_status,
    }


def _validate_features(payload: FeaturesUpdate) -> None:
    for field in _FEATURES_ENUM_FIELDS:
        value = getattr(payload, field)
        if value is not None and value not in _FEATURES_TRISTATE:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"features.{field} must be one of "
                    f"{list(_FEATURES_TRISTATE)}, got {value!r}"
                ),
            )


# ---------------------------------------------------------------------------
# /admin/llm
# ---------------------------------------------------------------------------


_LLM_PROVIDERS = ("disabled", "ollama", "openai_compatible")
_LLM_OUTPUT_LANGUAGES = ("auto", "ja", "en")
_LLM_API_KEY_ENV_VAR = "LLM_API_KEY"
_LLM_BASE_URL_MAX_LEN = 2048
_LLM_MODEL_MAX_LEN = 256


class LLMUpdate(BaseModel):
    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    output_language: str | None = None
    vision_model: str | None = None


@router.get("/llm")
async def get_llm_config() -> dict[str, Any]:
    """Return the effective LLM config = baseline + on-disk overrides.

    Same idempotency guarantee as ``GET /admin/features`` /
    ``GET /admin/transcription``: a freshly-saved GUI value is
    visible before the container restart swaps the cached config.
    """
    from app import llm_overrides as lo

    base = config.settings.llm
    overrides = lo.read_overrides()

    def resolve(field: str) -> Any:
        if overrides is not None:
            override_value = getattr(overrides, field)
            if override_value is not None:
                return override_value
        return getattr(base, field)

    return {
        "provider": resolve("provider"),
        "base_url": resolve("base_url"),
        "model": resolve("model"),
        "output_language": resolve("output_language"),
        "vision_model": resolve("vision_model"),
        "available_providers": list(_LLM_PROVIDERS),
        "available_output_languages": list(_LLM_OUTPUT_LANGUAGES),
        "api_key_present": bool(os.getenv(_LLM_API_KEY_ENV_VAR, "")),
        "api_key_env_var": _LLM_API_KEY_ENV_VAR,
        "overrides_present": overrides is not None,
    }


@router.put("/llm")
async def update_llm_config(payload: LLMUpdate) -> dict[str, Any]:
    from app import llm_overrides as lo

    _validate_llm(payload)
    overrides = lo.LLMOverrides(
        provider=payload.provider,
        base_url=payload.base_url,
        model=payload.model,
        output_language=payload.output_language,
        vision_model=payload.vision_model,
    )
    path = lo.write_overrides(
        overrides, updated_at=datetime.now(UTC).isoformat()
    )
    logger.info("LLM overrides saved at %s", path)
    notify_status = await _notify_core_restart_pending()
    return {
        "status": "saved",
        "restart_required": True,
        "core_notified": notify_status,
    }


@router.delete("/llm")
async def reset_llm_config() -> dict[str, Any]:
    from app import llm_overrides as lo

    removed = lo.delete_overrides()
    notify_status = await _notify_core_restart_pending()
    return {
        "status": "reset",
        "removed": removed,
        "restart_required": removed,
        "core_notified": notify_status,
    }


def _validate_llm(payload: LLMUpdate) -> None:
    if payload.provider is not None and payload.provider not in _LLM_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"llm.provider must be one of {list(_LLM_PROVIDERS)}, "
                f"got {payload.provider!r}"
            ),
        )
    if (
        payload.output_language is not None
        and payload.output_language not in _LLM_OUTPUT_LANGUAGES
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"llm.output_language must be one of "
                f"{list(_LLM_OUTPUT_LANGUAGES)}, got "
                f"{payload.output_language!r}"
            ),
        )
    if payload.base_url is not None and len(payload.base_url) > _LLM_BASE_URL_MAX_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"llm.base_url must be <= {_LLM_BASE_URL_MAX_LEN} chars",
        )
    if payload.model is not None and len(payload.model) > _LLM_MODEL_MAX_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"llm.model must be <= {_LLM_MODEL_MAX_LEN} chars",
        )
    if (
        payload.vision_model is not None
        and len(payload.vision_model) > _LLM_MODEL_MAX_LEN
    ):
        raise HTTPException(
            status_code=400,
            detail=f"llm.vision_model must be <= {_LLM_MODEL_MAX_LEN} chars",
        )
    for field, value in (
        ("base_url", payload.base_url),
        ("model", payload.model),
        ("vision_model", payload.vision_model),
    ):
        if value is not None and ("\n" in value or "\r" in value or "\x00" in value):
            raise HTTPException(
                status_code=400,
                detail=f"llm.{field} must not contain control characters",
            )


# ---------------------------------------------------------------------------
# /admin/rag
# ---------------------------------------------------------------------------


class RagUpdate(BaseModel):
    personal_history_enabled: bool | None = None
    category_expansion_enabled: bool | None = None


@router.get("/rag")
async def get_rag_config() -> dict[str, Any]:
    """Return the effective RAG sub-feature gates = baseline +
    on-disk overrides. Mirrors the read-after-write contract of the
    other admin GET endpoints."""
    from app import rag_overrides as ro

    base = config.settings.rag
    overrides = ro.read_overrides()

    if overrides is not None and overrides.personal_history_enabled is not None:
        ph_enabled = overrides.personal_history_enabled
    else:
        ph_enabled = base.personal_history.enabled
    if overrides is not None and overrides.category_expansion_enabled is not None:
        ce_enabled = overrides.category_expansion_enabled
    else:
        ce_enabled = base.category_expansion.enabled

    return {
        "personal_history_enabled": ph_enabled,
        "category_expansion_enabled": ce_enabled,
        "overrides_present": overrides is not None,
    }


@router.put("/rag")
async def update_rag_config(payload: RagUpdate) -> dict[str, Any]:
    from app import rag_overrides as ro

    overrides = ro.RagOverrides(
        personal_history_enabled=payload.personal_history_enabled,
        category_expansion_enabled=payload.category_expansion_enabled,
    )
    path = ro.write_overrides(
        overrides, updated_at=datetime.now(UTC).isoformat()
    )
    logger.info("RAG overrides saved at %s", path)
    notify_status = await _notify_core_restart_pending()
    return {
        "status": "saved",
        "restart_required": True,
        "core_notified": notify_status,
    }


@router.delete("/rag")
async def reset_rag_config() -> dict[str, Any]:
    from app import rag_overrides as ro

    removed = ro.delete_overrides()
    notify_status = await _notify_core_restart_pending()
    return {
        "status": "reset",
        "removed": removed,
        "restart_required": removed,
        "core_notified": notify_status,
    }


# ---------------------------------------------------------------------------
# /admin/embedding
# ---------------------------------------------------------------------------


class EmbeddingUpdate(BaseModel):
    """Payload for ``PUT /admin/embedding``.

    Only ``text_embedding`` is GUI-overridable per spec §3.1 / §3.4.
    Spec §3.4 mandates HTTP 422 (not 400) for an out-of-allowlist or
    empty value so the structural guard against the silent dim-384
    fallback (§2.1-4) is symmetric with FastAPI's body-validation
    error code. The router converts the validator's ``ValueError``
    into a 422 via the explicit ``HTTPException`` raised in
    :func:`_validate_embedding`.
    """

    text_embedding: str


def _model_family(model_id: str) -> str:
    return "ja" if "ruri" in model_id else "multi"


def _model_weight(dim: int) -> str:
    if dim <= 384:
        return "light"
    if dim <= 768:
        return "normal"
    return "heavy"


def _build_catalog() -> list[dict[str, Any]]:
    """Build the catalog list from the embedder's ``_MODEL_DIMS``.

    Imported lazily to keep the heavy embedder transitive deps out of
    the module-import path of ``admin.py`` (the test conftest stubs
    them, but production loads them only when actually embedding).
    """
    from app.workers.embedder import _MODEL_DIMS

    return [
        {
            "id": model_id,
            "family": _model_family(model_id),
            "dim": dim,
            "weight": _model_weight(dim),
        }
        for model_id, dim in _MODEL_DIMS.items()
    ]


def _effective_embedding_model() -> str:
    """Resolve the effective ``models.text_embedding``.

    Mirrors the read-after-write contract of the other admin GETs:
    a freshly-saved override is reflected immediately, before the
    container restart swaps the frozen ``config.settings.models``.
    """
    from app import embedding_overrides as eo

    overrides = eo.read_overrides()
    if overrides is not None and overrides.text_embedding is not None:
        return overrides.text_embedding
    return config.settings.models.text_embedding


def _read_recorded_model() -> str | None:
    """Read the search-DB recorded text-embedding model name.

    Goes through ``app.database`` as a module so the tests' patches
    on ``get_search_db_read`` / ``_read_index_meta`` take effect
    (project rule: ``import app.database as ...``, never
    ``from app.database import X``).
    """
    import app.database as database_mod

    try:
        with database_mod.get_search_db_read() as session:
            conn = session.connection()
            return database_mod._read_index_meta(
                conn, database_mod._TEXT_EMBEDDING_MODEL_KEY
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not read recorded text embedding model: %s", exc)
        return None


def _embedding_status_payload() -> dict[str, Any]:
    from app import embedding_overrides as eo

    effective = _effective_embedding_model()
    recorded = _read_recorded_model()
    overrides = eo.read_overrides()
    # A fresh DB with no override is NOT pending: the first index run
    # will simply record the effective model. A divergence between
    # recorded and effective IS pending — either because an override
    # was just saved (recorded may still be None until restart rebuilds
    # vec_text) or because the user previously indexed with a
    # different model.
    if recorded is None and overrides is None:
        reindex_pending = False
    else:
        reindex_pending = recorded != effective
    return {
        "effective": effective,
        "catalog": _build_catalog(),
        "recorded": recorded,
        "reindex_pending": reindex_pending,
    }


def _validate_embedding(payload: EmbeddingUpdate) -> None:
    from app.workers.embedder import _MODEL_DIMS

    value = payload.text_embedding
    if not value or value not in set(_MODEL_DIMS):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown text_embedding model {value!r}. "
                f"Choose one of {sorted(_MODEL_DIMS)}"
            ),
        )


@router.get("/embedding")
async def get_embedding_config() -> dict[str, Any]:
    """Return the effective text-embedding model + catalog + status.

    Spec §3.4. Reads the on-disk override authoritatively so a
    freshly-saved value is reflected before the next restart.
    """
    return _embedding_status_payload()


@router.put("/embedding")
async def update_embedding_config(payload: EmbeddingUpdate) -> dict[str, Any]:
    """Validate against the embedder allowlist and persist the override.

    A model outside ``_MODEL_DIMS`` (including empty) returns 422 and
    does NOT touch the overrides file or the restart_pending hook
    (structural prevention of the silent dim-384 fallback, §2.1-4).
    """
    from app import embedding_overrides as eo

    _validate_embedding(payload)
    overrides = eo.EmbeddingOverrides(text_embedding=payload.text_embedding)
    written_at = datetime.now(UTC).isoformat()
    path = eo.write_overrides(overrides, updated_at=written_at)
    logger.info(
        "Embedding overrides saved (text_embedding=%s) at %s",
        payload.text_embedding, path,
    )
    notify_status = await _notify_core_restart_pending()
    body = _embedding_status_payload()
    body.update({
        "status": "saved",
        "restart_required": True,
        "core_notified": notify_status,
    })
    return body


@router.delete("/embedding")
async def reset_embedding_config() -> dict[str, Any]:
    """Drop the GUI override; search-config.yml becomes authoritative.

    Idempotent: still returns 200 (with ``removed: False``) when no
    overrides file is present, mirroring ``DELETE /admin/transcription``.
    """
    from app import embedding_overrides as eo

    removed = eo.delete_overrides()
    if removed:
        logger.info("Embedding overrides reset (file removed)")
    else:
        logger.info(
            "Embedding overrides reset requested but no file present"
        )
    notify_status = await _notify_core_restart_pending()
    body = _embedding_status_payload()
    body.update({
        "status": "reset",
        "removed": removed,
        "restart_required": removed,
        "core_notified": notify_status,
    })
    return body


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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
                    "reason": "intelligence config updated",
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


# ---------------------------------------------------------------------------
# /admin/failed-jobs (spec 2026-05-24-intelligence-reindex-controls §2.2)
# ---------------------------------------------------------------------------
#
# Aggregates ``JobRecord.status='failed'`` rows by
# (file_id, job_kind, provider), surfacing the latest row per group.
# ``status='skipped'`` is excluded because retrying an unsupported-mime
# file would just be skipped again (UnsupportedMimeType is permanent
# until the file's mime changes).

# Cap per-call payload size. ``error_message`` for a stack-trace dump
# can be multi-kilobyte; 256 chars is enough for the user to grok the
# class of failure without inflating the JSON response.
_FAILED_JOB_MESSAGE_EXCERPT_CHARS = 256

_FAILED_JOBS_DEFAULT_LIMIT = 50
_FAILED_JOBS_MAX_LIMIT = 200

_FAILED_JOB_TASK_FLAG_COLUMN: dict[str, str] = {
    "metadata": "metadata_indexed",
    "clip": "clip_indexed",
    "text": "text_indexed",
    "text_content": "text_indexed",
    "transcription": "whisper_indexed",
}


_SENSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Bearer / Authorization headers a provider SDK echoed back on 4xx.
    re.compile(r"(?i)(authorization\s*[:=]\s*)bearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
    # Generic "api_key=" / "apikey=" / "api-key=" query / form fragments.
    re.compile(r"(?i)(api[_\-]?key)\s*[:=]\s*[A-Za-z0-9._\-]+"),
    # Common provider key prefixes (OpenAI / Anthropic / etc).
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\bskey-[A-Za-z0-9_\-]{8,}\b"),
)

# Container-absolute paths that leak the deploy layout to admins. Drop
# the prefix and keep the trailing filename so the excerpt still
# identifies which file blew up.
_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/data/drives/[^\s'\"]+/"),
    re.compile(r"/app/[^\s'\"]+/"),
    re.compile(r"/intelligence-data/[^\s'\"]+/"),
)


def _scrub_excerpt(text: str) -> str:
    """Strip credentials and container-absolute path prefixes.

    Worker code stores ``str(exception)`` verbatim — provider SDKs
    occasionally echo Authorization headers / API keys, and ffmpeg /
    Python stack frames embed container-internal paths. Both surface in
    the admin Failed Jobs modal, which the master viewer reads. Scrub
    before truncation so the 256-char excerpt is always credential-free.
    """
    out = text
    for pat in _SENSITIVE_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    for pat in _PATH_PATTERNS:
        out = pat.sub("…/", out)
    return out


def _truncate_excerpt(text: str | None, max_chars: int) -> str | None:
    if text is None:
        return None
    scrubbed = _scrub_excerpt(text)
    if len(scrubbed) <= max_chars:
        return scrubbed
    return scrubbed[: max_chars - 1] + "…"


@router.get("/failed-jobs", response_model=FailedJobsResponse)
async def get_failed_jobs(
    limit: int = _FAILED_JOBS_DEFAULT_LIMIT,
    offset: int = 0,
) -> FailedJobsResponse:
    """Return failed indexing jobs, aggregated by (file, kind, provider).

    Spec §2.2. Excludes groups whose latest JobRecord is no longer
    ``failed`` (later ``succeeded`` / ``skipped`` / ``running`` rows
    close the failure), and rows whose ``file_id`` has no
    ``IndexedFile`` row (purged files have no filename / drive to link
    to). Failed groups are kept indefinitely so they cannot silently
    keep index progress below 100% after a dashboard window expires.

    ``attempts`` counts consecutive ``failed`` rows since the latest
    closing row for the same (file_id, job_kind, provider). A file that
    has never succeeded or been manually skipped reports its lifetime
    failure count.
    """
    # Clamp manually rather than relying on FastAPI's ``Query(ge=,le=)``
    # so the handler stays callable from tests that bypass the proxy.
    limit = max(1, min(limit, _FAILED_JOBS_MAX_LIMIT))
    offset = max(0, offset)

    import app.database as database_mod
    from sqlalchemy import and_, func, or_

    from app.models import IndexedFile, JobRecord

    with database_mod.get_search_db() as session:
        # Step 1: distinct groups (file_id, job_kind, provider) ordered
        # by their latest attempted_at. The aggregate considers every
        # status, then the outer query keeps only groups whose latest
        # row is still failed.
        latest_attempted = func.max(JobRecord.attempted_at).label("latest")
        latest_by_group = (
            session.query(
                JobRecord.file_id,
                JobRecord.job_kind,
                JobRecord.provider,
                latest_attempted,
            )
            .join(IndexedFile, IndexedFile.file_id == JobRecord.file_id)
            .group_by(JobRecord.file_id, JobRecord.job_kind, JobRecord.provider)
            .subquery()
        )
        provider_matches = or_(
            JobRecord.provider == latest_by_group.c.provider,
            and_(
                JobRecord.provider.is_(None),
                latest_by_group.c.provider.is_(None),
            ),
        )
        groups_query = (
            session.query(
                JobRecord.file_id,
                JobRecord.job_kind,
                JobRecord.provider,
                JobRecord.attempted_at.label("latest"),
            )
            .join(
                latest_by_group,
                and_(
                    JobRecord.file_id == latest_by_group.c.file_id,
                    JobRecord.job_kind == latest_by_group.c.job_kind,
                    provider_matches,
                    JobRecord.attempted_at == latest_by_group.c.latest,
                ),
            )
            .filter(JobRecord.status == "failed")
            .order_by(JobRecord.attempted_at.desc())
        )

        # Total = number of distinct surfaced groups. Operators read
        # this number on the IndexStatusWidget badge every 10 seconds,
        # so it must stay cheap. Subquery + COUNT(*) keeps the work in
        # SQL and avoids materialising rows just to count.
        total = (
            session.query(func.count())
            .select_from(groups_query.subquery())
            .scalar()
        ) or 0

        # Step 2: hydrate only the paged groups with filename / drive /
        # latest error_class / message, then count attempts since the
        # latest closing row per group.
        paged_groups = groups_query.offset(offset).limit(limit).all()

        items: list[FailedJobItem] = []
        for file_id, job_kind, provider, latest_at in paged_groups:
            provider_filter = (
                JobRecord.provider == provider
                if provider is not None
                else JobRecord.provider.is_(None)
            )

            # Latest failed row for this group — gives error_class /
            # excerpt and the surface IndexedFile join.
            latest_row = (
                session.query(JobRecord, IndexedFile.filename, IndexedFile.drive)
                .join(IndexedFile, IndexedFile.file_id == JobRecord.file_id)
                .filter(
                    JobRecord.file_id == file_id,
                    JobRecord.job_kind == job_kind,
                    provider_filter,
                    JobRecord.status == "failed",
                    JobRecord.attempted_at == latest_at,
                )
                .first()
            )
            if latest_row is None:
                # Race: the row was hard-deleted between the aggregate
                # and the hydrate. Skip silently.
                continue
            record, filename, drive = latest_row

            # Latest closing row (if any) for the group — used to clamp
            # the attempts counter to "since last success / manual
            # resolution" rather than lifetime.
            last_closed_row = (
                session.query(JobRecord.attempted_at)
                .filter(
                    JobRecord.file_id == file_id,
                    JobRecord.job_kind == job_kind,
                    provider_filter,
                    JobRecord.status.in_(("succeeded", "skipped")),
                )
                .order_by(JobRecord.attempted_at.desc())
                .first()
            )
            attempts_query = session.query(func.count(JobRecord.id)).filter(
                JobRecord.file_id == file_id,
                JobRecord.job_kind == job_kind,
                provider_filter,
                JobRecord.status == "failed",
            )
            if last_closed_row is not None:
                attempts_query = attempts_query.filter(
                    JobRecord.attempted_at > last_closed_row[0]
                )
            attempts = attempts_query.scalar() or 0

            items.append(
                FailedJobItem(
                    file_id=record.file_id,
                    filename=filename,
                    drive=drive,
                    job_kind=record.job_kind,
                    provider=record.provider,
                    error_class=record.error_class,
                    error_message_excerpt=_truncate_excerpt(
                        record.error_message,
                        _FAILED_JOB_MESSAGE_EXCERPT_CHARS,
                    ),
                    attempted_at=record.attempted_at,
                    attempts=attempts,
                )
            )

    return FailedJobsResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


def _job_kind_to_resolvable_task(job_kind: str) -> str:
    if job_kind == "transcription":
        return "whisper"
    if job_kind == "clip":
        return "clip"
    if job_kind in {"text", "text_content"}:
        return "text"
    if job_kind == "metadata":
        return "metadata"
    raise HTTPException(
        status_code=422,
        detail=(
            f"Unknown job_kind: {job_kind!r}. "
            "Allowed: metadata, clip, transcription, text, text_content"
        ),
    )


@router.post("/failed-jobs/resolve", response_model=ResolveFailedJobResponse)
async def resolve_failed_job(
    body: dict[str, Any] = Body(...),
) -> ResolveFailedJobResponse:
    """Manually close a permanently failing failed-job group.

    This is the operator escape hatch for files that cannot be indexed
    after retry (corrupt audio, unreadable media, provider-incompatible
    payload, etc.). It marks the corresponding IndexedFile flag as done
    so progress can reach 100%, and appends a ``status='skipped'``
    JobRecord row instead of rewriting historical failures.
    """
    file_id = body.get("file_id") if isinstance(body, dict) else None
    job_kind = body.get("job_kind") if isinstance(body, dict) else None
    provider = body.get("provider") if isinstance(body, dict) else None

    if not isinstance(file_id, str) or not file_id:
        raise HTTPException(status_code=422, detail="file_id is required")
    if not isinstance(job_kind, str) or not job_kind:
        raise HTTPException(status_code=422, detail="job_kind is required")
    if provider is not None and not isinstance(provider, str):
        raise HTTPException(status_code=422, detail="provider must be a string or null")

    task = _job_kind_to_resolvable_task(job_kind)
    flag_column = _FAILED_JOB_TASK_FLAG_COLUMN[job_kind]

    import app.database as database_mod
    from app.models import IndexedFile, JobRecord

    now = datetime.now(UTC)

    with database_mod.get_search_db() as session:
        row = (
            session.query(IndexedFile)
            .filter(
                IndexedFile.file_id == file_id,
                IndexedFile.active.is_(True),
            )
            .first()
        )
        if row is None:
            raise HTTPException(status_code=404, detail="File not indexed")

        setattr(row, flag_column, True)
        if task == "clip":
            row.clip_thumbnail_indexed = True
        elif task == "whisper":
            row.tfidf_keywords_indexed = True

        session.add(JobRecord(
            file_id=file_id,
            job_kind=job_kind,
            provider=provider,
            status="skipped",
            error_class="ManuallyResolved",
            error_message="Marked resolved from the admin failed jobs modal",
            attempted_at=now,
            completed_at=now,
        ))

    return ResolveFailedJobResponse(
        status="resolved",
        file_id=file_id,
        task=task,
    )
