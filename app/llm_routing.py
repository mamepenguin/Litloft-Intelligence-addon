"""Named LLM profiles, per-feature routing and the per-drive cloud ceiling.

Every LLM call resolves its client here with the drive of the file (or Ask)
it serves. A profile marked ``offhost`` is used only when core answers that
the drive allows ``llm_cloud``; "could not ask" defers the job and is never
read as either answer.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app import policy_client
from app.config import LLMConfig
from app.llm import LLMClient, OllamaLLMClient, create_llm_client
from app.llm_overrides import PROVIDER_ENUM

logger = logging.getLogger(__name__)

LLM_FEATURES = (
    "rag",
    "summaries",
    "detailed_summaries",
    "auto_tags",
    "transcript_refine",
    "retrieval_keywords",
    "chapter_suggestions",
    "vision_describe",
    "video_visual_index",
)
CLOUD_POLICY_FEATURE = "llm_cloud"
LEGACY_PROFILE = "default"
DEFAULT_API_KEY_ENV = "LLM_API_KEY"

_PROFILE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_PROFILE_ONLY_KEYS = ("offhost", "agentic", "api_key_env")


@dataclass(frozen=True)
class LLMProfile:
    name: str
    config: LLMConfig
    offhost: bool
    agentic: bool
    api_key_env: str


@dataclass(frozen=True)
class LLMRouting:
    profiles: Mapping[str, LLMProfile]
    default: str | None
    local_fallback: str | None
    features: Mapping[str, str]
    error: str | None = None


@dataclass(frozen=True)
class Resolved:
    profile: LLMProfile
    client: LLMClient | OllamaLLMClient


@dataclass(frozen=True)
class Skip:
    reason: str


@dataclass(frozen=True)
class Defer:
    reason: str


class _RoutingError(ValueError):
    pass


def _disabled(error: str) -> LLMRouting:
    logger.error("LLM routing disabled: %s", error)
    return LLMRouting(
        profiles={}, default=None, local_fallback=None, features={}, error=error
    )


def build_routing(
    llm_section: Mapping[str, Any],
    base: LLMConfig,
    environ: Mapping[str, str],
) -> LLMRouting:
    """Build the routing from the ``llm`` config section.

    Without ``profiles`` the whole section is one profile named
    ``default`` with ``base`` unchanged. An invalid section yields no
    profiles at all, so every LLM feature skips rather than guessing.
    """
    if "profiles" not in llm_section:
        profile = LLMProfile(
            name=LEGACY_PROFILE,
            config=base,
            offhost=llm_section.get("offhost") is not False,
            agentic=False,
            api_key_env=DEFAULT_API_KEY_ENV,
        )
        return LLMRouting(
            profiles={LEGACY_PROFILE: profile},
            default=LEGACY_PROFILE,
            local_fallback=None,
            features={},
        )
    try:
        return _build_profiles_routing(llm_section, base, environ)
    except _RoutingError as exc:
        return _disabled(str(exc))


def _build_profiles_routing(
    llm_section: Mapping[str, Any],
    base: LLMConfig,
    environ: Mapping[str, str],
) -> LLMRouting:
    raw_profiles = llm_section.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise _RoutingError("llm.profiles must be a non-empty mapping")
    profiles = {
        name: _build_profile(name, raw, base, environ)
        for name, raw in raw_profiles.items()
    }

    raw_routing = llm_section.get("routing") or {}
    if not isinstance(raw_routing, dict):
        raise _RoutingError("llm.routing must be a mapping")

    default = raw_routing.get("default")
    if default is None and len(profiles) == 1:
        default = next(iter(profiles))
    if default is None:
        raise _RoutingError("llm.routing.default is required with several profiles")
    _require_profile(profiles, default, "llm.routing.default")

    local_fallback = raw_routing.get("local_fallback")
    if local_fallback is not None:
        _require_profile(profiles, local_fallback, "llm.routing.local_fallback")
        if profiles[local_fallback].offhost:
            raise _RoutingError(
                f"llm.routing.local_fallback {local_fallback!r} must be offhost: false"
            )

    raw_features = raw_routing.get("features") or {}
    if not isinstance(raw_features, dict):
        raise _RoutingError("llm.routing.features must be a mapping")
    for feature, name in raw_features.items():
        if feature not in LLM_FEATURES:
            raise _RoutingError(f"llm.routing.features has unknown feature {feature!r}")
        _require_profile(profiles, name, f"llm.routing.features.{feature}")

    return LLMRouting(
        profiles=profiles,
        default=default,
        local_fallback=local_fallback,
        features=dict(raw_features),
    )


def _require_profile(profiles: Mapping[str, LLMProfile], name: object, where: str) -> None:
    if not isinstance(name, str) or name not in profiles:
        raise _RoutingError(f"{where} names unknown profile {name!r}")


def _build_profile(
    name: object,
    raw: object,
    base: LLMConfig,
    environ: Mapping[str, str],
) -> LLMProfile:
    if not isinstance(name, str) or not _PROFILE_NAME_RE.fullmatch(name):
        raise _RoutingError(f"invalid profile name {name!r}")
    if not isinstance(raw, dict):
        raise _RoutingError(f"llm.profiles.{name} must be a mapping")
    provider = raw.get("provider", base.provider)
    if provider not in PROVIDER_ENUM:
        raise _RoutingError(f"llm.profiles.{name}.provider {provider!r} is not one of {PROVIDER_ENUM}")

    config_fields = {f.name for f in dataclasses.fields(LLMConfig)}
    overrides = {k: v for k, v in raw.items() if k in config_fields and k != "api_key"}
    api_key_env = raw.get("api_key_env")
    if api_key_env is not None:
        if not isinstance(api_key_env, str) or not api_key_env:
            raise _RoutingError(f"llm.profiles.{name}.api_key_env must be a string")
        overrides["api_key"] = environ.get(api_key_env, "")
    return LLMProfile(
        name=name,
        config=dataclasses.replace(base, **overrides),
        offhost=raw.get("offhost") is not False,
        agentic=raw.get("agentic") is True,
        api_key_env=api_key_env or DEFAULT_API_KEY_ENV,
    )


# ---------------------------------------------------------------------------
# Current routing and resolution
# ---------------------------------------------------------------------------

_routing: LLMRouting | None = None
_clients: dict[str, LLMClient | OllamaLLMClient] = {}


def _load_from_settings() -> LLMRouting:
    import app.config as config

    return build_routing(config.settings.llm_section, config.settings.llm, os.environ)


def current_routing() -> LLMRouting:
    global _routing
    if _routing is None:
        _routing = _load_from_settings()
    return _routing


def set_routing(routing: LLMRouting | None) -> None:
    """Replace the routing for subsequent jobs; ``None`` reloads from settings.

    A job that already resolved keeps the client it was handed.
    """
    global _routing, _clients
    _routing = routing
    _clients = {}


def _client_for(profile: LLMProfile) -> LLMClient | OllamaLLMClient:
    client = _clients.get(profile.name)
    if client is None:
        client = create_llm_client(profile.config)
        _clients[profile.name] = client
    return client


def _usable(profile: LLMProfile, *, vision: bool) -> bool:
    config = profile.config
    if config.provider == "disabled" or not config.model.strip():
        return False
    return not vision or bool(config.vision_model.strip())


async def resolve(
    drive: str, feature: str, *, vision: bool = False
) -> Resolved | Skip | Defer:
    if feature not in LLM_FEATURES:
        raise ValueError(f"unknown LLM feature {feature!r}")
    routing = current_routing()
    name = routing.features.get(feature, routing.default)
    if name is None:
        return Skip(routing.error or "no LLM profile configured")
    profile = routing.profiles[name]

    if profile.offhost:
        verdict = await policy_client.lookup_feature(drive, CLOUD_POLICY_FEATURE)
        if verdict == "unknown":
            return Defer(f"{CLOUD_POLICY_FEATURE} policy for {drive!r} unavailable")
        if verdict == "denied":
            if routing.local_fallback is None:
                return Skip(f"{CLOUD_POLICY_FEATURE} is off for {drive!r}")
            profile = routing.profiles[routing.local_fallback]

    if not _usable(profile, vision=vision):
        return Skip(f"profile {profile.name!r} cannot serve {feature}")
    return Resolved(profile=profile, client=_client_for(profile))
