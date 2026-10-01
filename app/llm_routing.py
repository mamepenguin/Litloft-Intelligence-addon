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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from app import policy_client
from app.config import LLMConfig, parse_llm_config
from app.llm import LLMClient, OllamaLLMClient, create_llm_client
from app.llm_overrides import PROVIDER_ENUM

logger = logging.getLogger(__name__)

VISION_FEATURES = frozenset({"vision_describe", "video_visual_index"})
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
# Features a viewer may run on a profile of their choice instead of routing's.
REQUESTABLE_FEATURES = frozenset({"summaries", "detailed_summaries", "rag"})
CLOUD_POLICY_FEATURE = "llm_cloud"
LEGACY_PROFILE = "default"
LEGACY_API_KEY_ENV = "LLM_API_KEY"

PROFILE_NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,31}$"
_PROFILE_NAME_RE = re.compile(PROFILE_NAME_PATTERN)
# A profile may only read keys the operator named for it: any other variable
# (e.g. CORE_INTERNAL_SECRET) would be sent as a Bearer token to base_url.
_API_KEY_ENV_RE = re.compile(r"^LLM_API_KEY(_[A-Z0-9]+)*$")
_PROFILE_ONLY_KEYS = ("offhost", "agentic", "api_key_env")
# Never inherited from the top-level ``llm`` section: a profile that omitted
# its endpoint would otherwise send to, and with the key of, another one.
_CONNECTION_FIELDS = ("provider", "base_url", "api_key", "model", "vision_model")
_PROFILE_KEYS = (
    frozenset(f.name for f in dataclasses.fields(LLMConfig))
    - {"api_key", "agentic_models", "agentic_mode", "agentic_min_capability", "output_language"}
) | set(_PROFILE_ONLY_KEYS)


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
        from app.rag.agentic import agentic_capability_supported

        profile = LLMProfile(
            name=LEGACY_PROFILE,
            config=base,
            offhost=llm_section.get("offhost") is not False,
            agentic=agentic_capability_supported(base.model, base),
            api_key_env=LEGACY_API_KEY_ENV,
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

    raw_routing = _mapping(llm_section.get("routing"), "llm.routing")

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

    raw_features = _mapping(raw_routing.get("features"), "llm.routing.features")
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


def _mapping(value: object, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _RoutingError(f"{where} must be a mapping")
    return value


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
    provider = raw.get("provider")
    if provider not in PROVIDER_ENUM:
        raise _RoutingError(f"llm.profiles.{name}.provider {provider!r} is not one of {PROVIDER_ENUM}")
    unknown = set(raw) - _PROFILE_KEYS
    if unknown:
        raise _RoutingError(f"llm.profiles.{name} has unknown keys {sorted(unknown)}")
    for key in ("offhost", "agentic"):
        if key in raw and not isinstance(raw[key], bool):
            raise _RoutingError(f"llm.profiles.{name}.{key} must be true or false")
    api_key_env = raw.get("api_key_env")
    if api_key_env is not None and (
        not isinstance(api_key_env, str) or not _API_KEY_ENV_RE.fullmatch(api_key_env)
    ):
        raise _RoutingError(
            f"llm.profiles.{name}.api_key_env must be LLM_API_KEY or LLM_API_KEY_<NAME>"
        )

    knobs = dataclasses.replace(base, **{f: getattr(LLMConfig(), f) for f in _CONNECTION_FIELDS})
    values = {k: v for k, v in raw.items() if k not in _PROFILE_ONLY_KEYS}
    values["api_key"] = environ.get(api_key_env, "") if api_key_env else ""
    config = parse_llm_config(values, base=knobs)
    return LLMProfile(
        name=name,
        config=config,
        offhost=raw.get("offhost") is not False,
        agentic=raw.get("agentic") is True,
        api_key_env=api_key_env or "",
    )


# ---------------------------------------------------------------------------
# Current routing and resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Active:
    routing: LLMRouting
    clients: dict[str, LLMClient | OllamaLLMClient]
    # The settings object the routing belongs to; a different one (a reload,
    # or a test swapping ``app.config.settings``) means it is stale.
    settings: object


_active: _Active | None = None


def _load_from_settings(settings: Any) -> LLMRouting:
    return build_routing(
        getattr(settings, "llm_section", None) or {}, settings.llm, os.environ
    )


def _current() -> _Active:
    global _active
    import app.config as config

    if _active is None or _active.settings is not config.settings:
        _active = _Active(
            routing=_load_from_settings(config.settings),
            clients={},
            settings=config.settings,
        )
    return _active


def current_routing() -> LLMRouting:
    return _current().routing


def set_routing(routing: LLMRouting | None) -> None:
    """Replace the routing for subsequent jobs; ``None`` reloads from settings.

    A job that already resolved keeps the client it was handed.
    """
    global _active
    import app.config as config

    _active = (
        None
        if routing is None
        else _Active(routing=routing, clients={}, settings=config.settings)
    )


def _client_for(active: _Active, profile: LLMProfile) -> LLMClient | OllamaLLMClient:
    client = active.clients.get(profile.name)
    if client is None:
        client = create_llm_client(profile.config)
        active.clients[profile.name] = client
    return client


def _assigned(routing: LLMRouting, feature: str) -> LLMProfile | Skip:
    if feature not in LLM_FEATURES:
        raise ValueError(f"unknown LLM feature {feature!r}")
    name = routing.features.get(feature, routing.default)
    if name is None:
        return Skip(routing.error or "no LLM profile configured")
    return routing.profiles[name]


def _after_policy(
    routing: LLMRouting, profile: LLMProfile, verdict: str, drive: str
) -> LLMProfile | Skip | Defer:
    """The profile an off-host assignment yields under ``verdict``."""
    if verdict == "unknown":
        return Defer(f"{CLOUD_POLICY_FEATURE} policy for {drive!r} unavailable")
    if verdict == "denied":
        if routing.local_fallback is None:
            return Skip(f"{CLOUD_POLICY_FEATURE} is off for {drive!r}")
        return routing.profiles[routing.local_fallback]
    return profile


def _serve(active: _Active, profile: LLMProfile, feature: str) -> Resolved | Skip:
    client = _client_for(active, profile)
    needs_vision = feature in VISION_FEATURES
    if not client.enabled or (needs_vision and not profile.config.vision_model.strip()):
        return Skip(f"profile {profile.name!r} cannot serve {feature}")
    return Resolved(profile=profile, client=client)


class _Verdict:
    """The drive's ``llm_cloud`` answer, asked at most once."""

    def __init__(self, drive: str) -> None:
        self._drive = drive
        self._value: str | None = None

    async def get(self) -> str:
        if self._value is None:
            self._value = await policy_client.lookup_feature(
                self._drive, CLOUD_POLICY_FEATURE
            )
        return self._value


async def _resolve(
    active: _Active,
    drive: str,
    feature: str,
    requested: str | None,
    verdict: _Verdict,
) -> Resolved | Skip | Defer:
    if requested is not None:
        return await _resolve_requested(active, drive, feature, requested, verdict)
    profile = _assigned(active.routing, feature)
    if isinstance(profile, Skip):
        return profile
    if profile.offhost:
        picked = _after_policy(active.routing, profile, await verdict.get(), drive)
        if not isinstance(picked, LLMProfile):
            return picked
        profile = picked
    return _serve(active, profile, feature)


async def _resolve_requested(
    active: _Active,
    drive: str,
    feature: str,
    requested: str,
    verdict: _Verdict,
) -> Resolved | Skip | Defer:
    """A chosen profile is served as it is or not at all: never ``local_fallback``."""
    if feature not in REQUESTABLE_FEATURES:
        raise ValueError(f"{feature!r} does not take a requested profile")
    profile = active.routing.profiles.get(requested)
    if profile is None:
        return Skip(f"no profile named {requested!r}")
    if profile.offhost:
        answer = await verdict.get()
        if answer == "unknown":
            return Defer(f"{CLOUD_POLICY_FEATURE} policy for {drive!r} unavailable")
        if answer != "allowed":
            return Skip(f"{CLOUD_POLICY_FEATURE} is off for {drive!r}")
    return _serve(active, profile, feature)


async def resolve(
    drive: str, feature: str, requested: str | None = None
) -> Resolved | Skip | Defer:
    return await _resolve(_current(), drive, feature, requested, _Verdict(drive))


@dataclass(frozen=True)
class Choices:
    auto: LLMProfile | None
    profiles: tuple[LLMProfile, ...]


async def choices(drive: str, feature: str) -> Choices | Defer:
    """What routing picks for ``feature`` on ``drive``, and every profile a
    request naming it would be served by, in configured order.
    """
    if feature not in REQUESTABLE_FEATURES:
        raise ValueError(f"{feature!r} does not take a requested profile")
    active = _current()
    verdict = _Verdict(drive)
    # An off-host ``auto`` that defers is itself listed below and defers there.
    auto = await _resolve(active, drive, feature, None, verdict)
    served: list[LLMProfile] = []
    for name in active.routing.profiles:
        result = await _resolve(active, drive, feature, name, verdict)
        if isinstance(result, Defer):
            return result
        if isinstance(result, Resolved):
            served.append(result.profile)
    return Choices(
        auto=auto.profile if isinstance(auto, Resolved) else None,
        profiles=tuple(served),
    )


def describe_route(feature: str, verdict: str, drive: str) -> str:
    """What ``resolve`` would do for ``feature`` on a drive whose policy is
    ``verdict``: ``sends``, ``falls_back``, ``skips`` or ``unknown``.
    """
    active = _current()
    assigned = _assigned(active.routing, feature)
    if isinstance(assigned, Skip):
        return "skips"
    picked = (
        _after_policy(active.routing, assigned, verdict, drive)
        if assigned.offhost
        else assigned
    )
    if isinstance(picked, Defer):
        return "unknown"
    if isinstance(picked, Skip) or isinstance(_serve(active, picked, feature), Skip):
        return "skips"
    return "sends" if picked is assigned else "falls_back"


def has_vision_profile(settings: Any | None = None) -> bool:
    """True when some profile can describe images under ``settings``."""
    import app.config as config

    if settings is None or settings is config.settings:
        active = _current()
    else:
        active = _Active(
            routing=_load_from_settings(settings), clients={}, settings=settings
        )
    return any(
        p.config.vision_model.strip() and _client_for(active, p).enabled
        for p in active.routing.profiles.values()
    )


_bound: ContextVar[Resolved | None] = ContextVar("llm_routing_bound", default=None)


@contextmanager
def bound(resolved: Resolved) -> Iterator[None]:
    """Hand ``resolved`` to the helpers one Ask calls, for its duration."""
    token = _bound.set(resolved)
    try:
        yield
    finally:
        _bound.reset(token)


def bound_resolved() -> Resolved | None:
    return _bound.get()


def bound_client() -> LLMClient | OllamaLLMClient | None:
    resolved = _bound.get()
    return resolved.client if resolved is not None else None


def default_status() -> tuple[LLMProfile | None, bool]:
    """The profile unassigned features use, and whether its client works."""
    active = _current()
    name = active.routing.default
    profile = active.routing.profiles.get(name) if name is not None else None
    if profile is None:
        return None, False
    return profile, _client_for(active, profile).enabled


def resolve_without_ceiling(feature: str) -> Resolved | Skip:
    """The profile ``feature`` is routed to, ignoring every drive's ceiling.

    For operator-run tools only (the eval runner); nothing reachable from
    a route or a worker may call it.
    """
    active = _current()
    profile = _assigned(active.routing, feature)
    if isinstance(profile, Skip):
        return profile
    return _serve(active, profile, feature)