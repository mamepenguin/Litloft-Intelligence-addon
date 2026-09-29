"""LLM profiles, routing and the per-drive cloud ceiling."""

from __future__ import annotations

from typing import Any

import pytest

from app import llm_routing
from app.config import LLMConfig
from app.llm_routing import Defer, Resolved, Skip, build_routing

BASE = LLMConfig(
    provider="openai_compatible",
    base_url="https://api.example/v1",
    api_key="base-key",
    model="base-model",
    vision_model="base-vision",
    temperature=0.3,
)

LOCAL = {
    "provider": "ollama",
    "base_url": "http://ollama:11434",
    "model": "qwen3:14b",
    "vision_model": "gemma3:12b",
    "offhost": False,
}
CLOUD = {
    "provider": "openai_compatible",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-mini",
    "vision_model": "gpt-mini",
    "api_key_env": "CLOUD_KEY",
    "offhost": True,
}


def _section(**routing: Any) -> dict[str, Any]:
    return {
        "profiles": {"local": dict(LOCAL), "cloud": dict(CLOUD)},
        "routing": {"default": "local", "local_fallback": "local", **routing},
    }


# ---------------------------------------------------------------------------
# build_routing
# ---------------------------------------------------------------------------


def test_legacy_section_becomes_one_default_profile_with_the_same_config() -> None:
    routing = build_routing({"model": "base-model"}, BASE, environ={})

    assert routing.error is None
    assert list(routing.profiles) == ["default"]
    assert routing.profiles["default"].config == BASE
    assert routing.default == "default"
    assert routing.features == {}


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ({}, True),
        ({"offhost": False}, False),
        ({"offhost": True}, True),
    ],
)
def test_legacy_profile_is_offhost_unless_declared(
    section: dict[str, Any], expected: bool
) -> None:
    routing = build_routing(section, BASE, environ={})
    assert routing.profiles["default"].offhost is expected


def test_profile_without_offhost_is_treated_as_offhost() -> None:
    section = _section()
    del section["profiles"]["cloud"]["offhost"]

    routing = build_routing(section, BASE, environ={})

    assert routing.profiles["cloud"].offhost is True


def test_profile_inherits_unset_knobs_and_reads_its_own_key_env() -> None:
    routing = build_routing(_section(), BASE, environ={"CLOUD_KEY": "k-cloud"})

    cloud = routing.profiles["cloud"].config
    assert cloud.model == "gpt-mini"
    assert cloud.temperature == 0.3
    assert cloud.api_key == "k-cloud"
    assert routing.profiles["local"].config.api_key == "base-key"


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda s: s["routing"].update(default="nope"), "nope"),
        (lambda s: s["routing"].update(features={"rag": "nope"}), "nope"),
        (lambda s: s["routing"].update(features={"ragg": "cloud"}), "ragg"),
        (lambda s: s["routing"].update(local_fallback="cloud"), "local_fallback"),
        (lambda s: s["routing"].pop("default"), "default"),
        (lambda s: s["profiles"].update({"Bad Name": dict(LOCAL)}), "Bad Name"),
        (lambda s: s["profiles"]["local"].update(provider="nope"), "provider"),
        (lambda s: s.update(profiles={}), "profiles"),
    ],
)
def test_invalid_routing_disables_every_profile_and_says_why(mutate, needle) -> None:
    section = _section()
    mutate(section)

    routing = build_routing(section, BASE, environ={})

    assert routing.profiles == {}
    assert routing.error is not None and needle in routing.error


def test_single_profile_needs_no_default() -> None:
    section = {"profiles": {"local": dict(LOCAL)}}

    routing = build_routing(section, BASE, environ={})

    assert routing.error is None
    assert routing.default == "local"


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


class _Policy:
    def __init__(self) -> None:
        self.verdicts: dict[str, str] = {}
        self.asked: list[tuple[str, str]] = []

    async def lookup(self, drive: str, feature: str) -> str:
        self.asked.append((drive, feature))
        return self.verdicts[drive]


@pytest.fixture
def policy(monkeypatch: pytest.MonkeyPatch) -> _Policy:
    fake = _Policy()
    monkeypatch.setattr(llm_routing.policy_client, "lookup_feature", fake.lookup)
    return fake


@pytest.fixture(autouse=True)
def _restore_routing():
    yield
    llm_routing.set_routing(None)


def _install(section: dict[str, Any]) -> None:
    routing = build_routing(section, BASE, environ={"CLOUD_KEY": "k"})
    assert routing.error is None
    llm_routing.set_routing(routing)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("assigned", "fallback", "verdict", "vision", "expected"),
    [
        ("cloud", "local", "allowed", False, "cloud"),
        ("cloud", "local", "denied", False, "local"),
        ("cloud", None, "denied", False, Skip),
        ("cloud", "local", "unknown", False, Defer),
        ("cloud", None, "unknown", False, Defer),
        ("local", "local", None, False, "local"),
        ("cloud", "local-novision", "denied", True, Skip),
        ("cloud", "local-novision", "denied", False, "local-novision"),
    ],
)
async def test_resolve_never_sends_offhost_without_an_allowed_answer(
    policy, assigned, fallback, verdict, vision, expected
) -> None:
    section = _section(features={"summaries": assigned})
    section["profiles"]["local-novision"] = {**LOCAL, "vision_model": ""}
    if fallback is None:
        section["routing"].pop("local_fallback")
    else:
        section["routing"]["local_fallback"] = fallback
    _install(section)
    if verdict is not None:
        policy.verdicts["d"] = verdict

    result = await llm_routing.resolve("d", "summaries", vision=vision)

    if isinstance(expected, str):
        assert isinstance(result, Resolved)
        assert result.profile.name == expected
    else:
        assert type(result) is expected


@pytest.mark.asyncio
async def test_resolve_does_not_ask_the_policy_for_an_onhost_profile(policy) -> None:
    _install(_section(features={"auto_tags": "local"}))

    await llm_routing.resolve("d", "auto_tags")

    assert policy.asked == []


@pytest.mark.asyncio
async def test_resolve_asks_the_llm_cloud_feature_for_the_jobs_drive(policy) -> None:
    _install(_section(features={"rag": "cloud"}))
    policy.verdicts["private"] = "denied"

    await llm_routing.resolve("private", "rag")

    assert policy.asked == [("private", "llm_cloud")]


@pytest.mark.asyncio
async def test_unassigned_feature_uses_the_default_profile(policy) -> None:
    _install(_section(default="cloud", features={}))
    policy.verdicts["d"] = "allowed"

    result = await llm_routing.resolve("d", "retrieval_keywords")

    assert isinstance(result, Resolved) and result.profile.name == "cloud"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profile_patch",
    [{"provider": "disabled"}, {"model": ""}],
)
async def test_unusable_profile_skips(policy, profile_patch) -> None:
    section = _section(features={"summaries": "local"})
    section["profiles"]["local"].update(profile_patch)
    _install(section)

    assert isinstance(await llm_routing.resolve("d", "summaries"), Skip)


@pytest.mark.asyncio
async def test_invalid_routing_skips_everything(policy) -> None:
    section = _section()
    section["routing"]["default"] = "nope"
    llm_routing.set_routing(build_routing(section, BASE, environ={}))

    assert isinstance(await llm_routing.resolve("d", "summaries"), Skip)


@pytest.mark.asyncio
async def test_unknown_feature_is_a_programming_error(policy) -> None:
    _install(_section())

    with pytest.raises(ValueError):
        await llm_routing.resolve("d", "not_a_feature")


@pytest.mark.asyncio
async def test_swapping_routing_drops_cached_clients(policy) -> None:
    _install(_section(features={"summaries": "local"}))
    first = await llm_routing.resolve("d", "summaries")
    again = await llm_routing.resolve("d", "summaries")
    assert isinstance(first, Resolved) and isinstance(again, Resolved)
    assert first.client is again.client

    _install(_section(features={"summaries": "local"}))
    after = await llm_routing.resolve("d", "summaries")

    assert isinstance(after, Resolved)
    assert after.client is not first.client
