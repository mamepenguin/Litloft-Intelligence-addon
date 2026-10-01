"""LLM profiles, routing and the per-drive cloud ceiling."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from app import llm_routing
from app.config import LLMConfig
from app.llm import OllamaLLMClient
from app.llm_routing import Defer, Resolved, Skip, build_routing

BASE = LLMConfig(
    provider="openai_compatible",
    base_url="https://api.example/v1",
    api_key="base-key",
    model="base-model",
    vision_model="base-vision",
    temperature=0.7,
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
    "api_key_env": "LLM_API_KEY_CLOUD",
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


def test_profile_inherits_tuning_knobs_only() -> None:
    routing = build_routing(_section(), BASE, environ={"LLM_API_KEY_CLOUD": "k-cloud"})

    cloud = routing.profiles["cloud"].config
    assert cloud.model == "gpt-mini"
    assert cloud.temperature == 0.7
    assert cloud.api_key == "k-cloud"


@pytest.mark.parametrize("written", [None, 123])
@pytest.mark.parametrize("field", ["base_url", "model", "vision_model"])
def test_profile_never_inherits_the_top_level_endpoint_or_key(
    field: str, written: object
) -> None:
    section = {"profiles": {"bare": {"provider": "ollama", "offhost": False, field: written}}}

    routing = build_routing(section, BASE, environ={"LLM_API_KEY": "top-key"})

    config = routing.profiles["bare"].config
    assert (getattr(config, field), config.api_key) == ("", "")


@pytest.mark.parametrize(
    ("field", "written"),
    [
        ("temperature", None),
        ("temperature", "hot"),
        ("temperature", True),
        ("max_tokens", True),
    ],
)
def test_profile_knob_left_blank_or_mistyped_inherits_the_top_level_value(
    field: str, written: object
) -> None:
    section = _section()
    section["profiles"]["cloud"][field] = written

    routing = build_routing(section, BASE, environ={})

    assert routing.error is None
    assert getattr(routing.profiles["cloud"].config, field) == getattr(BASE, field)


def test_a_mistyped_key_never_reaches_the_log(caplog: pytest.LogCaptureFixture) -> None:
    from app.config import parse_llm_config

    with caplog.at_level("WARNING"):
        parse_llm_config({"api_key": 98765432101234})

    assert "98765432101234" not in caplog.text


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
        (lambda s: s["profiles"]["local"].pop("provider"), "provider"),
        (lambda s: s["profiles"]["cloud"].update(offhost="no"), "offhost"),
        (lambda s: s["profiles"]["cloud"].update(agentic="yes"), "agentic"),
        (lambda s: s["profiles"]["cloud"].update(modle="x"), "modle"),
        (lambda s: s["profiles"]["cloud"].update(output_language="en"), "output_language"),
        (lambda s: s["profiles"].update(cloud="not a mapping"), "cloud"),
        (lambda s: s.update(routing=[]), "llm.routing must be a mapping"),
        (lambda s: s.update(routing="x"), "llm.routing must be a mapping"),
        (lambda s: s["routing"].update(features=[]), "features must be a mapping"),
        (lambda s: s["profiles"]["local"].update(api_key="inline"), "api_key"),
        (lambda s: s["profiles"]["cloud"].update(api_key_env=""), "api_key_env"),
        (lambda s: s["profiles"]["cloud"].update(api_key_env="CORE_INTERNAL_SECRET"), "api_key_env"),
        (lambda s: s["profiles"]["cloud"].update(api_key_env="LLM_API_KEYS"), "api_key_env"),
        (lambda s: s["profiles"]["cloud"].update(api_key_env="llm_api_key_x"), "api_key_env"),
    ],
)
def test_invalid_routing_disables_every_profile_and_says_why(mutate, needle) -> None:
    section = _section()
    mutate(section)

    routing = build_routing(section, BASE, environ={})

    assert routing.profiles == {}
    assert routing.error is not None and needle in routing.error


def test_named_but_unset_key_env_gives_no_key() -> None:
    routing = build_routing(_section(), BASE, environ={"LLM_API_KEY": "top-key"})

    assert routing.profiles["cloud"].config.api_key == ""


@pytest.mark.parametrize(
    ("patch", "field", "expected"),
    [
        ({"temperature": 0}, "temperature", 0),
        ({"request_timeout_seconds": 60}, "request_timeout_seconds", 60),
        ({"reasoning": "auto"}, "reasoning", "auto"),
        ({"reasoning": "bogus"}, "reasoning", "disabled"),
    ],
)
def test_profile_values_go_through_the_llm_config_parser(patch, field, expected) -> None:
    section = _section()
    section["profiles"]["cloud"].update(patch)

    routing = build_routing(section, BASE, environ={})

    assert routing.error is None
    assert getattr(routing.profiles["cloud"].config, field) == expected


def test_null_routing_keys_count_as_absent() -> None:
    section = {"profiles": {"local": dict(LOCAL)}, "routing": None}

    routing = build_routing(section, BASE, environ={})

    assert routing.error is None
    assert routing.features == {}


def test_agentic_is_off_unless_declared() -> None:
    routing = build_routing(_section(), BASE, environ={})

    assert routing.profiles["cloud"].agentic is False


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
    routing = build_routing(section, BASE, environ={"LLM_API_KEY_CLOUD": "k"})
    assert routing.error is None
    llm_routing.set_routing(routing)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("assigned", "fallback", "verdict", "feature", "expected", "described"),
    [
        ("cloud", "local", "allowed", "summaries", "cloud", "sends"),
        ("cloud", "local", "denied", "summaries", "local", "falls_back"),
        ("cloud", None, "denied", "summaries", Skip, "skips"),
        ("cloud", "local", "unknown", "summaries", Defer, "unknown"),
        ("cloud", None, "unknown", "summaries", Defer, "unknown"),
        ("local", "local", None, "summaries", "local", "sends"),
        ("cloud", "local-novision", "denied", "vision_describe", Skip, "skips"),
        ("cloud", "local-novision", "denied", "summaries", "local-novision", "falls_back"),
    ],
)
async def test_resolve_never_sends_offhost_without_an_allowed_answer(
    policy, assigned, fallback, verdict, feature, expected, described
) -> None:
    section = _section(features={feature: assigned})
    section["profiles"]["local-novision"] = {**LOCAL, "vision_model": ""}
    if fallback is None:
        section["routing"].pop("local_fallback")
    else:
        section["routing"]["local_fallback"] = fallback
    _install(section)
    if verdict is not None:
        policy.verdicts["d"] = verdict

    result = await llm_routing.resolve("d", feature)

    if isinstance(expected, str):
        assert isinstance(result, Resolved)
        assert result.profile.name == expected
    else:
        assert type(result) is expected
    assert llm_routing.describe_route(feature, verdict or "allowed", "d") == described


def _requestable_section() -> dict[str, Any]:
    section = _section(features={"summaries": "local", "detailed_summaries": "local", "rag": "local"})
    section["profiles"]["off"] = {**LOCAL, "provider": "disabled"}
    section["profiles"]["cloud-novision"] = {**CLOUD, "vision_model": ""}
    return section


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", ["summaries", "detailed_summaries", "rag"])
@pytest.mark.parametrize(
    ("requested", "verdict", "expected", "asks_policy"),
    [
        ("local", "denied", "local", False),
        ("cloud", "allowed", "cloud", True),
        ("cloud-novision", "allowed", "cloud-novision", True),
        ("cloud", "denied", Skip, True),
        ("cloud", "unknown", Defer, True),
        ("off", "allowed", Skip, False),
        ("nope", "allowed", Skip, False),
    ],
)
async def test_a_requested_profile_is_served_as_is_or_not_at_all(
    policy, feature, requested, verdict, expected, asks_policy
) -> None:
    _install(_requestable_section())
    policy.verdicts["d"] = verdict

    result = await llm_routing.resolve("d", feature, requested)

    if isinstance(expected, str):
        assert isinstance(result, Resolved)
        assert result.profile.name == expected
    else:
        assert type(result) is expected
    assert policy.asked == ([("d", "llm_cloud")] if asks_policy else [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feature", sorted(set(llm_routing.LLM_FEATURES) - {"summaries", "detailed_summaries", "rag"})
)
async def test_only_the_three_choosable_features_take_a_requested_profile(
    policy, feature
) -> None:
    _install(_requestable_section())

    with pytest.raises(ValueError):
        await llm_routing.resolve("d", feature, "local")


# ``routing`` assigns every choosable feature to ``auto``; the drive answers ``verdict``.
@pytest.mark.asyncio
@pytest.mark.parametrize("feature", ["summaries", "detailed_summaries", "rag"])
@pytest.mark.parametrize(
    ("auto", "fallback", "verdict", "expected_auto", "expected_names"),
    [
        ("local", "local", "allowed", "local", ["local", "cloud", "cloud-novision"]),
        ("cloud", "local", "allowed", "cloud", ["local", "cloud", "cloud-novision"]),
        ("cloud", "local", "denied", "local", ["local"]),
        ("cloud", None, "denied", None, ["local"]),
        ("off", None, "denied", None, ["local"]),
    ],
)
async def test_choices_are_the_profiles_a_request_would_be_served_by(
    policy, feature, auto, fallback, verdict, expected_auto, expected_names
) -> None:
    section = _requestable_section()
    section["routing"]["features"] = {feature: auto}
    if fallback is None:
        section["routing"].pop("local_fallback")
    _install(section)
    policy.verdicts["d"] = verdict

    result = await llm_routing.choices("d", feature)

    assert not isinstance(result, Defer)
    assert (result.auto.name if result.auto else None) == expected_auto
    assert [p.name for p in result.profiles] == expected_names
    assert len(policy.asked) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("auto", ["local", "cloud"])
async def test_choices_defer_when_the_policy_cannot_be_read(policy, auto) -> None:
    section = _requestable_section()
    section["routing"]["features"] = {"rag": auto}
    _install(section)
    policy.verdicts["d"] = "unknown"

    assert isinstance(await llm_routing.choices("d", "rag"), Defer)


@pytest.mark.asyncio
async def test_choices_refuse_a_feature_that_takes_no_choice(policy) -> None:
    _install(_requestable_section())

    with pytest.raises(ValueError):
        await llm_routing.choices("d", "auto_tags")


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
    [{"provider": "disabled"}, {"model": ""}, {"base_url": ""}],
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


@pytest.mark.asyncio
async def test_swap_during_a_policy_lookup_never_hands_out_the_old_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(_section(features={"summaries": "cloud"}))
    replacement = _section(features={"summaries": "cloud"})
    replacement["profiles"]["cloud"] = dict(LOCAL)
    replacement["routing"].pop("local_fallback")

    async def _lookup_then_swap(drive: str, feature: str) -> str:
        _install(replacement)
        return "allowed"

    monkeypatch.setattr(llm_routing.policy_client, "lookup_feature", _lookup_then_swap)
    await llm_routing.resolve("d", "summaries")

    after = await llm_routing.resolve("d", "summaries")

    assert isinstance(after, Resolved)
    assert isinstance(after.client, OllamaLLMClient)


def test_load_settings_builds_routing_from_yaml_and_v1_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import json

    import yaml

    from app import config

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_file = tmp_path / "search-config.yml"
    config_file.write_text(yaml.safe_dump({"llm": {"provider": "ollama", "model": "yaml-model"}}))
    (data_dir / "llm-overrides.json").write_text(
        json.dumps({"schema_version": 1, "model": "gui-model"})
    )
    monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))
    monkeypatch.setattr(config, "settings", config.load_settings())
    llm_routing.set_routing(None)

    routing = llm_routing.current_routing()

    assert list(routing.profiles) == ["default"]
    assert routing.profiles["default"].config.model == "gui-model"
    assert routing.profiles["default"].config.provider == "ollama"


def test_load_settings_passes_profiles_from_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import yaml

    from app import config

    config_file = tmp_path / "search-config.yml"
    config_file.write_text(yaml.safe_dump({"llm": _section(features={"rag": "cloud"})}))
    monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))
    monkeypatch.setattr(config, "settings", config.load_settings())
    llm_routing.set_routing(None)

    routing = llm_routing.current_routing()

    assert set(routing.profiles) == {"local", "cloud"}
    assert routing.features == {"rag": "cloud"}
    assert routing.local_fallback == "local"


def test_manifest_declares_the_policy_features() -> None:
    import json
    from pathlib import Path

    manifest = json.loads((Path(__file__).parent.parent / "manifest.json").read_text())

    assert {
        f["name"]: (f["default"], f["i18n_key"]) for f in manifest["policy_features"]
    } == {
        "transcription_cloud": (True, "intelligence.policyFeatures.transcriptionCloud"),
        "chapter_suggestions": (True, "intelligence.policyFeatures.chapterSuggestions"),
        "llm_cloud": (True, "intelligence.policyFeatures.llmCloud"),
    }


@pytest.mark.parametrize(
    ("llm", "provider", "offhost"),
    [
        ({"provider": "ollama", "model": "m", "offhost": False}, "ollama", False),
        ({"provider": "ollama", "model": "m"}, "ollama", True),
        ({"provider": "ollama", "model": 123}, "ollama", True),
        ({"provider": "ollama", "model": "m", "agentic_mode": False}, "ollama", True),
        ({"provider": "ollama", "model": "m", "output_language": False}, "ollama", True),
        ({"provider": "ollama", "model": "m", "api_key": None}, "ollama", True),
        ({"provider": "ollama", "model": "m", "vision_model": None}, "ollama", True),
        ({"provider": "ollama", "model": "m", "reasoning": None}, "ollama", True),
        ({"provider": "ollama", "model": "m", "agentic_models": None}, "ollama", True),
        ({"provider": "ollama", "model": "m", "agentic_models": 5}, "ollama", True),
    ],
)
def test_load_settings_legacy_section(
    monkeypatch: pytest.MonkeyPatch, tmp_path, llm, provider, offhost
) -> None:
    import yaml

    from app import config

    config_file = tmp_path / "search-config.yml"
    config_file.write_text(yaml.safe_dump({"llm": llm}))
    monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))
    monkeypatch.setattr(config, "settings", config.load_settings())
    llm_routing.set_routing(None)

    profile = llm_routing.current_routing().profiles["default"]

    assert (profile.config.provider, profile.offhost) == (provider, offhost)



@pytest.mark.parametrize(
    ("section", "expected"),
    [
        (_section(), ("local", True)),
        (_section(default="cloud"), ("cloud", True)),
        (
            {"profiles": {"off": {**LOCAL, "provider": "disabled"}}},
            ("off", False),
        ),
    ],
)
def test_default_status_reports_the_default_profile(section, expected) -> None:
    _install(section)

    profile, enabled = llm_routing.default_status()

    assert (profile.name, enabled) == expected


def test_default_status_with_invalid_routing_is_disabled() -> None:
    section = _section()
    section["routing"]["default"] = "nope"
    llm_routing.set_routing(build_routing(section, BASE, environ={}))

    assert llm_routing.default_status() == (None, False)


def test_resolve_without_ceiling_ignores_the_drive_policy(policy) -> None:
    _install(_section(features={"rag": "cloud"}))

    result = llm_routing.resolve_without_ceiling("rag")

    assert isinstance(result, Resolved) and result.profile.name == "cloud"
    assert policy.asked == []


@pytest.mark.parametrize(
    ("mode", "allowlist", "expected"),
    [
        ("auto", ("base-model",), True),
        ("auto", ("other",), False),
        ("off", ("base-model",), False),
    ],
)
def test_the_legacy_profile_is_agentic_by_the_allowlist(mode, allowlist, expected) -> None:
    from app.config import AgenticModelEntry

    base = dataclasses.replace(
        BASE,
        agentic_mode=mode,
        agentic_models=tuple(AgenticModelEntry(name=n) for n in allowlist),
    )

    routing = build_routing({}, base, environ={})

    assert routing.profiles["default"].agentic is expected


def test_a_profile_is_not_agentic_by_the_inherited_allowlist() -> None:
    from app.config import AgenticModelEntry

    base = dataclasses.replace(
        BASE, agentic_mode="auto", agentic_models=(AgenticModelEntry(name="gpt-mini"),)
    )

    routing = build_routing(_section(), base, environ={})

    assert routing.profiles["cloud"].agentic is False


@pytest.mark.parametrize(
    ("patch", "feature"),
    [({"provider": "disabled"}, "summaries"), ({"vision_model": ""}, "vision_describe")],
)
def test_resolve_without_ceiling_skips_an_unusable_profile(policy, patch, feature) -> None:
    section = _section(features={feature: "cloud"})
    section["profiles"]["cloud"].update(patch)
    _install(section)

    assert isinstance(llm_routing.resolve_without_ceiling(feature), Skip)


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        (_section(default="cloud"), ("openai_compatible", "gpt-mini", True)),
        (
            {"profiles": {"off": {**LOCAL, "provider": "disabled"}}},
            ("disabled", "qwen3:14b", False),
        ),
    ],
)
def test_status_reports_the_default_profile(section, expected) -> None:
    from app.main import _llm_status

    _install(section)

    status = _llm_status()

    assert (status.provider, status.model, status.enabled) == expected


def test_saved_profiles_replace_the_yamls_wholesale(monkeypatch, tmp_path) -> None:
    import yaml

    from app import config
    from app.llm_overrides import write_profiles

    config_file = tmp_path / "search-config.yml"
    config_file.write_text(yaml.safe_dump({"llm": _section(features={"rag": "cloud"})}))
    monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))
    monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path))
    write_profiles(
        {"only": dict(LOCAL)}, {"default": "only"}, None, data_dir=tmp_path
    )

    section, _ = config.load_llm_sources()

    assert section["profiles"] == {"only": LOCAL}
    assert section["routing"] == {"default": "only"}


# ---------------------------------------------------------------------------
# GET /llm/choices/*
# ---------------------------------------------------------------------------


def _choices_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.routers import llm_choices

    app = FastAPI()
    app.include_router(llm_choices.router)
    return TestClient(app)


@pytest.mark.parametrize("feature", ["summaries", "detailed_summaries", "rag"])
@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        (
            "allowed",
            {
                "auto": "cloud",
                "choices": [
                    {"name": "local", "model": "qwen3:14b", "offhost": False},
                    {"name": "cloud", "model": "gpt-mini", "offhost": True},
                ],
            },
        ),
        (
            "denied",
            {
                "auto": None,
                "choices": [{"name": "local", "model": "qwen3:14b", "offhost": False}],
            },
        ),
    ],
)
def test_choices_endpoint_lists_names_and_models_only(
    policy, feature, verdict, expected
) -> None:
    section = _section(features={feature: "cloud"})
    section["routing"].pop("local_fallback")
    section["profiles"]["off"] = {**LOCAL, "provider": "disabled"}
    _install(section)
    policy.verdicts["d"] = verdict

    response = _choices_client().get(
        f"/llm/choices/{feature}", headers={"X-Lit-Drive": "d"}
    )

    assert response.status_code == 200
    assert response.json() == expected


@pytest.mark.parametrize("feature", ["summaries", "detailed_summaries", "rag"])
def test_choices_endpoint_is_unavailable_while_the_policy_is_unknown(
    policy, feature
) -> None:
    _install(_section(features={feature: "local"}))
    policy.verdicts["d"] = "unknown"

    response = _choices_client().get(
        f"/llm/choices/{feature}", headers={"X-Lit-Drive": "d"}
    )

    assert response.status_code == 503


@pytest.mark.parametrize("feature", ["summaries", "detailed_summaries", "rag"])
def test_choices_endpoint_needs_a_drive(policy, feature) -> None:
    _install(_section())

    response = _choices_client().get(f"/llm/choices/{feature}")

    assert response.status_code == 400
    assert policy.asked == []


def test_choices_routes_are_drive_scoped_and_gated_by_their_feature() -> None:
    import json
    from pathlib import Path

    manifest = json.loads((Path(__file__).parent.parent / "manifest.json").read_text())
    routes = {
        r["path"]: r for r in manifest["proxy"]["routes"] if r["path"].startswith("/llm/")
    }

    assert routes == {
        f"/llm/choices/{feature}": {
            "path": f"/llm/choices/{feature}",
            "methods": ["GET"],
            "pre_check": {"type": "addon_feature", "feature": feature},
            "response_filter": None,
        }
        for feature in ("summaries", "detailed_summaries", "rag")
    }


@pytest.mark.parametrize("feature", ["summaries", "detailed_summaries", "rag"])
def test_the_app_serves_the_choices_routes(policy, feature) -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    _install(_section(features={feature: "local"}))
    policy.verdicts["d"] = "allowed"

    response = TestClient(app).get(f"/llm/choices/{feature}", headers={"X-Lit-Drive": "d"})

    assert response.status_code == 200
    assert response.json()["auto"] == "local"
