"""Tests for the admin transcription endpoint."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stub heavy ML deps before importing anything that pulls them in.
for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from fastapi.testclient import TestClient

from app.transcription_overrides import (
    TranscriptionOverrides,
    overrides_path,
    read_overrides,
    write_overrides,
)


# Build a tiny FastAPI app that mounts only the admin router so the
# tests don't have to spin up the whole intelligence service.

@pytest.fixture()
def admin_app(tmp_path, monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path))
    # Reload the router module so the env var is picked up.
    sys.modules.pop("app.routers.admin", None)
    from fastapi import FastAPI

    from app.routers import admin
    app = FastAPI()
    app.include_router(admin.router)
    return app, tmp_path


@pytest.fixture()
def client(admin_app):
    app, _ = admin_app
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /admin/transcription
# ---------------------------------------------------------------------------


def test_get_returns_baseline_when_no_overrides(client, monkeypatch) -> None:
    """Without an overrides file the baseline (search-config.yml)
    values come through unchanged."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    response = client.get("/admin/transcription")
    assert response.status_code == 200
    body = response.json()
    assert "provider" in body
    assert "available_providers" in body
    assert set(body["available_providers"]) == {
        "whisper_local", "openai_compatible", "deepgram",
        "elevenlabs_scribe", "assemblyai", "gemini",
    }
    assert body["api_keys_present"] == {
        "whisper_local": True,
        "openai_compatible": False,
        "deepgram": False,
        "elevenlabs_scribe": False,
        "assemblyai": False,
        "gemini": False,
    }
    assert body["overrides_present"] is False


def test_get_reflects_saved_overrides_before_restart(
    client, admin_app
) -> None:
    """Phase 2D contract: GET reads the on-disk file authoritatively
    (R1 review H1), so a freshly-saved value is visible right away."""
    _app, data_dir = admin_app
    write_overrides(
        TranscriptionOverrides(
            provider="deepgram",
            language_hint="ja",
            hotwords=("Foo",),
        ),
        data_dir=data_dir,
    )
    response = client.get("/admin/transcription")
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "deepgram"
    assert body["language_hint"] == "ja"
    assert body["hotwords"] == ["Foo"]
    assert body["overrides_present"] is True


def test_get_reports_api_key_presence(client, monkeypatch) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.get("/admin/transcription")
    body = response.json()
    assert body["api_keys_present"]["deepgram"] is True
    assert body["api_keys_present"]["openai_compatible"] is False


# ---------------------------------------------------------------------------
# PUT /admin/transcription
# ---------------------------------------------------------------------------


def _ok_notify():
    return AsyncMock(return_value="ok")


def test_put_persists_valid_payload(client, admin_app, monkeypatch) -> None:
    _app, data_dir = admin_app
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    from app.routers import admin as admin_module

    monkeypatch.setattr(
        admin_module, "_notify_core_restart_pending", _ok_notify()
    )
    response = client.put(
        "/admin/transcription",
        json={
            "provider": "deepgram",
            "language_hint": "ja",
            "hotwords": ["Litloft"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "saved"
    assert body["restart_required"] is True
    assert body["core_notified"] == "ok"

    persisted = read_overrides(data_dir)
    assert persisted is not None
    assert persisted.provider == "deepgram"
    assert persisted.language_hint == "ja"
    assert persisted.hotwords == ("Litloft",)


def test_put_rejects_unknown_provider(client) -> None:
    response = client.put(
        "/admin/transcription",
        json={"provider": "totally_made_up", "hotwords": []},
    )
    assert response.status_code == 400
    assert "Unknown provider" in response.json()["detail"]


def test_put_requires_api_key_for_cloud_provider(
    client, monkeypatch
) -> None:
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    response = client.put(
        "/admin/transcription",
        json={"provider": "deepgram", "hotwords": []},
    )
    assert response.status_code == 400
    assert "DEEPGRAM_API_KEY" in response.json()["detail"]


def test_put_rejects_invalid_language_hint(client, monkeypatch) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    response = client.put(
        "/admin/transcription",
        json={
            "provider": "deepgram",
            "language_hint": "not a tag",
            "hotwords": [],
        },
    )
    assert response.status_code == 400
    assert "BCP-47" in response.json()["detail"] or "language_hint" in response.json()["detail"]


def test_put_accepts_long_bcp47_tag(client, monkeypatch) -> None:
    """``zh-Hant-HK`` is 10 chars, valid BCP-47 — must not 400."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    from app.routers import admin as admin_module

    monkeypatch.setattr(
        admin_module, "_notify_core_restart_pending", _ok_notify()
    )
    response = client.put(
            "/admin/transcription",
            json={
                "provider": "deepgram",
                "language_hint": "zh-Hant-HK",
                "hotwords": [],
            },
        )
    assert response.status_code == 200


def test_put_accepts_empty_language_hint(client, monkeypatch) -> None:
    """Empty string explicitly clears the hint and overrides baseline."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    from app.routers import admin as admin_module

    monkeypatch.setattr(
        admin_module, "_notify_core_restart_pending", _ok_notify()
    )
    response = client.put(
            "/admin/transcription",
            json={
                "provider": "deepgram",
                "language_hint": "",
                "hotwords": [],
            },
        )
    assert response.status_code == 200


def test_put_rejects_oversize_hotword(client, monkeypatch) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    response = client.put(
        "/admin/transcription",
        json={
            "provider": "deepgram",
            "hotwords": ["x" * 100],  # > 64
        },
    )
    assert response.status_code == 400
    assert "hotword" in response.json()["detail"]


def test_put_rejects_too_many_hotwords(client, monkeypatch) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    response = client.put(
        "/admin/transcription",
        json={
            "provider": "deepgram",
            "hotwords": ["x"] * 1000,  # > 500
        },
    )
    assert response.status_code == 400


def test_put_rejects_control_chars_in_hotwords(client, monkeypatch) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    response = client.put(
        "/admin/transcription",
        json={
            "provider": "deepgram",
            "hotwords": ["foo\nbar"],
        },
    )
    assert response.status_code == 400


def test_put_handles_notify_failure_without_rollback(
    client, admin_app, monkeypatch
) -> None:
    """Core unreachable: overrides are still saved, status returns
    ``"error"`` so the GUI can warn but the user can manually
    restart."""
    _app, data_dir = admin_app
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")

    from app.routers import admin as admin_module

    monkeypatch.setattr(
        admin_module,
        "_notify_core_restart_pending",
        AsyncMock(return_value="error"),
    )
    response = client.put(
        "/admin/transcription",
        json={"provider": "deepgram", "hotwords": []},
    )
    assert response.status_code == 200
    assert response.json()["core_notified"] == "error"
    # Overrides STILL persisted
    assert read_overrides(data_dir) is not None


# ---------------------------------------------------------------------------
# DELETE /admin/transcription
# ---------------------------------------------------------------------------


def test_delete_removes_existing_overrides(client, admin_app, monkeypatch) -> None:
    """A DELETE drops the overrides file so search-config.yml takes back
    over on the next restart. The core is notified that a restart is
    needed because the actual transcribe job switches over only after
    intelligence reloads ``settings.transcription``."""
    _app, data_dir = admin_app
    write_overrides(
        TranscriptionOverrides(provider="deepgram"),
        data_dir=data_dir,
    )
    assert overrides_path(data_dir).is_file()

    from app.routers import admin as admin_module

    notify = _ok_notify()
    monkeypatch.setattr(admin_module, "_notify_core_restart_pending", notify)

    response = client.delete("/admin/transcription")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "reset"
    assert body["removed"] is True
    assert body["restart_required"] is True
    assert body["core_notified"] == "ok"
    assert not overrides_path(data_dir).is_file()
    notify.assert_awaited_once()


def test_delete_is_noop_when_no_overrides_present(
    client, admin_app, monkeypatch
) -> None:
    """Calling DELETE without an existing overrides file returns 200
    with ``removed=False`` so the GUI can debounce repeated clicks
    without surfacing fake errors. Restart is not required because
    nothing actually changed."""
    _app, data_dir = admin_app
    assert not overrides_path(data_dir).is_file()

    from app.routers import admin as admin_module

    notify = _ok_notify()
    monkeypatch.setattr(admin_module, "_notify_core_restart_pending", notify)

    response = client.delete("/admin/transcription")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "reset"
    assert body["removed"] is False
    assert body["restart_required"] is False
    notify.assert_awaited_once()


def test_delete_get_roundtrip_clears_overrides_present(
    client, admin_app, monkeypatch
) -> None:
    """After a successful DELETE, GET reports ``overrides_present=False``
    so the GUI banner disappears immediately (well before the
    container restart actually swaps providers)."""
    _app, data_dir = admin_app
    write_overrides(
        TranscriptionOverrides(provider="deepgram", language_hint="ja"),
        data_dir=data_dir,
    )
    pre = client.get("/admin/transcription").json()
    assert pre["overrides_present"] is True

    from app.routers import admin as admin_module

    monkeypatch.setattr(
        admin_module, "_notify_core_restart_pending", _ok_notify()
    )
    client.delete("/admin/transcription")

    post = client.get("/admin/transcription").json()
    assert post["overrides_present"] is False


def test_whisper_local_does_not_require_api_key(
    client, admin_app, monkeypatch
) -> None:
    """The on-host provider needs no env. Selecting it must succeed
    even with every cloud key unset."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from app.routers import admin as admin_module

    monkeypatch.setattr(
        admin_module, "_notify_core_restart_pending", _ok_notify()
    )
    response = client.put(
            "/admin/transcription",
            json={"provider": "whisper_local", "hotwords": []},
        )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# /admin/features
# ---------------------------------------------------------------------------


def test_features_get_returns_baseline(client) -> None:
    response = client.get("/admin/features")
    assert response.status_code == 200
    body = response.json()
    for key in (
        "indexing", "search", "rag", "auto_tags", "summaries",
        "detailed_summaries", "transcript_refine", "vision_describe",
        "retrieval_keywords",
    ):
        assert key in body
    assert body["overrides_present"] is False
    assert body["tristate_values"] == ["false", "manual", "on_index"]


def test_features_get_reflects_saved_overrides_before_restart(
    client, admin_app
) -> None:
    """Without this read-after-write guarantee the GUI form would snap
    back to the YAML baseline as soon as the user saves, even though
    the override has been persisted. Mirrors the transcription
    endpoint's contract (R1 review H1)."""
    _app, data_dir = admin_app
    from app.features_overrides import (
        FeaturesOverrides,
        write_overrides as write_features,
    )
    write_features(
        FeaturesOverrides(indexing=False, vision_describe="on_index"),
        data_dir=data_dir,
    )
    body = client.get("/admin/features").json()
    assert body["indexing"] is False
    assert body["vision_describe"] == "on_index"
    assert body["overrides_present"] is True


def test_features_put_persists_payload(client, admin_app, monkeypatch) -> None:
    _app, data_dir = admin_app
    from app.routers import admin as admin_module

    monkeypatch.setattr(
        admin_module, "_notify_core_restart_pending", _ok_notify()
    )

    response = client.put(
        "/admin/features",
        json={
            "indexing": False,
            "auto_tags": "on_index",
            "vision_describe": "false",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "saved"

    from app.features_overrides import read_overrides as read_features
    persisted = read_features(data_dir=data_dir)
    assert persisted is not None
    assert persisted.indexing is False
    assert persisted.auto_tags == "on_index"
    assert persisted.vision_describe == "false"


def test_features_put_rejects_invalid_enum(client) -> None:
    response = client.put(
        "/admin/features",
        json={"auto_tags": "always"},
    )
    assert response.status_code == 400
    assert "auto_tags" in response.json()["detail"]


def test_features_delete_removes_overrides(
    client, admin_app, monkeypatch
) -> None:
    _app, data_dir = admin_app
    from app.features_overrides import (
        FeaturesOverrides,
        overrides_path as features_overrides_path,
        write_overrides as write_features,
    )
    write_features(
        FeaturesOverrides(indexing=False), data_dir=data_dir,
    )
    assert features_overrides_path(data_dir).is_file()

    from app.routers import admin as admin_module

    monkeypatch.setattr(
        admin_module, "_notify_core_restart_pending", _ok_notify()
    )

    response = client.delete("/admin/features")
    assert response.status_code == 200
    body = response.json()
    assert body["removed"] is True
    assert not features_overrides_path(data_dir).is_file()


# ---------------------------------------------------------------------------
# /admin/llm
# ---------------------------------------------------------------------------


@pytest.fixture()
def llm_env(monkeypatch, tmp_path):
    from app import llm_routing

    monkeypatch.setenv("SEARCH_CONFIG_PATH", str(tmp_path / "missing.yml"))
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_API_KEY_CLOUD", "sk-cloud")
    llm_routing.set_routing(None)
    yield
    llm_routing.set_routing(None)


_LOCAL = {
    "provider": "ollama",
    "base_url": "http://ollama:11434",
    "model": "qwen3:14b",
    "offhost": False,
}
_CLOUD = {
    "provider": "openai_compatible",
    "base_url": "https://api.example/v1",
    "model": "gpt-mini",
    "offhost": True,
    "api_key_env": "LLM_API_KEY_CLOUD",
}
_ROUTED = {
    "profiles": {"local": _LOCAL, "cloud": _CLOUD},
    "routing": {"default": "local", "local_fallback": "local", "features": {"rag": "cloud"}},
}


def test_llm_get_shows_the_legacy_section_as_one_profile(client, llm_env) -> None:
    body = client.get("/admin/llm").json()

    assert body["legacy"] is True
    assert body["error"] is None
    assert body["overrides_present"] is False
    assert body["routing"] == {"default": "default"}
    assert body["profiles"]["default"]["api_key_env"] == "LLM_API_KEY"
    assert body["profiles"]["default"]["api_key_present"] is True
    assert body["profiles"]["default"]["offhost"] is True
    assert body["available_output_languages"] == ["auto", "ja", "en"]


def test_llm_get_reads_v1_overrides_into_the_legacy_profile(
    client, admin_app, llm_env
) -> None:
    _app, data_dir = admin_app
    from app.llm_overrides import LLMOverrides, write_overrides

    write_overrides(
        LLMOverrides(provider="ollama", model="gemma4:e4b"), data_dir=data_dir
    )

    profile = client.get("/admin/llm").json()["profiles"]["default"]

    assert (profile["provider"], profile["model"]) == ("ollama", "gemma4:e4b")


def test_llm_put_saves_and_applies_without_a_restart(
    client, admin_app, llm_env, monkeypatch
) -> None:
    _app, data_dir = admin_app
    from app import llm_routing
    from app.llm_overrides import read_profiles
    from app.routers import admin as admin_module

    notify = _ok_notify()
    monkeypatch.setattr(admin_module, "_notify_core_restart_pending", notify)

    response = client.put("/admin/llm", json=_ROUTED)

    assert response.status_code == 200
    assert response.json()["restart_required"] is False
    notify.assert_not_awaited()
    assert read_profiles(data_dir=data_dir) == _ROUTED
    routing = llm_routing.current_routing()
    assert (set(routing.profiles), routing.features) == ({"local", "cloud"}, {"rag": "cloud"})

    monkeypatch.delenv("LLM_API_KEY_CLOUD")
    body = client.get("/admin/llm").json()
    assert body["legacy"] is False
    assert body["profiles"]["cloud"]["api_key_present"] is False
    assert body["routing"] == _ROUTED["routing"]


def test_llm_put_with_a_new_output_language_asks_for_a_restart(
    client, llm_env, monkeypatch
) -> None:
    from app.routers import admin as admin_module

    notify = _ok_notify()
    monkeypatch.setattr(admin_module, "_notify_core_restart_pending", notify)
    language = "en" if admin_module.config.settings.llm.output_language != "en" else "ja"

    response = client.put("/admin/llm", json={**_ROUTED, "output_language": language})

    assert response.json()["restart_required"] is True
    notify.assert_awaited_once()


@pytest.mark.parametrize(
    ("patch", "needle"),
    [
        ({"routing": {"default": "nope"}}, "nope"),
        ({"output_language": "fr"}, "output_language"),
        ({"profiles": {"local": {**_LOCAL, "base_url": "http://x\ny"}}}, "control"),
        (
            {"profiles": {f"p{i}": _LOCAL for i in range(17)}, "routing": {"default": "p0"}},
            "16",
        ),
        (
            {"profiles": {"local": _LOCAL, "cloud": {**_CLOUD, "api_key_env": "CORE_INTERNAL_SECRET"}}},
            "api_key_env",
        ),
        ({"routing": {**_ROUTED["routing"], "junk": "x" * 70000}}, "bytes"),
    ],
)
def test_llm_put_rejects_and_changes_nothing(
    client, admin_app, llm_env, patch, needle
) -> None:
    _app, data_dir = admin_app
    from app import llm_routing
    from app.llm_overrides import overrides_path

    before = llm_routing.current_routing()

    response = client.put("/admin/llm", json={**_ROUTED, **patch})

    assert response.status_code == 400
    assert needle in response.json()["detail"]
    assert not overrides_path(data_dir).is_file()
    assert llm_routing.current_routing() is before


def test_llm_get_sees_a_yaml_key_and_the_legacy_agentic_flag(
    client, llm_env, monkeypatch, tmp_path
) -> None:
    import yaml

    config_file = tmp_path / "with-key.yml"
    config_file.write_text(yaml.safe_dump({"llm": {
        "provider": "openai_compatible", "base_url": "https://x/v1", "model": "m",
        "api_key": "yaml-key", "agentic_mode": "auto", "agentic_models": [{"name": "m"}],
    }}))
    monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))
    monkeypatch.delenv("LLM_API_KEY")

    profile = client.get("/admin/llm").json()["profiles"]["default"]

    assert (profile["api_key_present"], profile["agentic"]) == (True, True)


def test_llm_get_body_can_be_saved_back_unchanged(client, llm_env, monkeypatch) -> None:
    from app.routers import admin as admin_module

    monkeypatch.setattr(admin_module, "_notify_core_restart_pending", _ok_notify())
    client.put("/admin/llm", json={**_ROUTED, "profiles": {
        "local": {**_LOCAL, "agentic": True}, "cloud": _CLOUD,
    }})
    body = client.get("/admin/llm").json()

    response = client.put("/admin/llm", json={
        "profiles": body["profiles"], "routing": body["routing"],
    })

    assert response.status_code == 200
    saved = client.get("/admin/llm").json()["profiles"]
    assert saved["local"]["agentic"] is True


def test_llm_restart_is_reported_until_the_language_is_applied(
    client, llm_env, monkeypatch
) -> None:
    from app.routers import admin as admin_module

    import dataclasses

    settings = admin_module.config.settings
    monkeypatch.setattr(
        admin_module.config,
        "settings",
        dataclasses.replace(
            settings, llm=dataclasses.replace(settings.llm, output_language="en")
        ),
    )
    notify = _ok_notify()
    monkeypatch.setattr(admin_module, "_notify_core_restart_pending", notify)

    client.put("/admin/llm", json={**_ROUTED, "output_language": "en"})
    assert client.get("/admin/llm").json()["output_language_restart_pending"] is False
    notify.assert_not_awaited()

    response = client.delete("/admin/llm")

    assert response.json()["restart_required"] is True
    notify.assert_awaited_once()
    assert client.get("/admin/llm").json()["output_language_restart_pending"] is True


def test_llm_delete_returns_to_the_yaml_immediately(
    client, admin_app, llm_env, monkeypatch
) -> None:
    _app, data_dir = admin_app
    from app import llm_routing
    from app.llm_overrides import overrides_path
    from app.routers import admin as admin_module

    monkeypatch.setattr(admin_module, "_notify_core_restart_pending", _ok_notify())
    client.put("/admin/llm", json=_ROUTED)

    response = client.delete("/admin/llm")

    assert response.json()["removed"] is True
    assert not overrides_path(data_dir).is_file()
    assert set(llm_routing.current_routing().profiles) == {"default"}


def test_llm_exposure_says_what_each_drive_does(client, llm_env, monkeypatch) -> None:
    from app import policy_client
    from app.routers import admin as admin_module

    import dataclasses

    monkeypatch.setattr(
        admin_module.config,
        "settings",
        dataclasses.replace(
            admin_module.config.settings,
            drive_mounts={"a": "/d/a", "b": "/d/b", "c": "/d/c"},
        ),
    )
    monkeypatch.setattr(admin_module, "_notify_core_restart_pending", _ok_notify())
    client.put("/admin/llm", json=_ROUTED)
    verdicts = {"a": "allowed", "b": "denied", "c": "unknown"}

    async def _lookup(drive, feature):
        assert feature == "llm_cloud"
        return verdicts[drive]

    monkeypatch.setattr(policy_client, "lookup_feature", _lookup)

    body = client.get("/admin/llm/exposure").json()

    assert body["features"]["rag"] == {
        "profile": "cloud",
        "offhost": True,
        "drives": {"a": "sends", "b": "falls_back", "c": "unknown"},
    }
    assert body["features"]["summaries"] == {"profile": "local", "offhost": False}


# ---------------------------------------------------------------------------
# /admin/rag
# ---------------------------------------------------------------------------


def test_rag_get_returns_baseline(client) -> None:
    response = client.get("/admin/rag")
    assert response.status_code == 200
    body = response.json()
    assert "personal_history_enabled" in body
    assert "category_expansion_enabled" in body
    assert body["overrides_present"] is False


def test_rag_get_reflects_saved_overrides_before_restart(
    client, admin_app
) -> None:
    _app, data_dir = admin_app
    from app.rag_overrides import (
        RagOverrides,
        write_overrides as write_rag,
    )
    write_rag(
        RagOverrides(
            personal_history_enabled=False,
            category_expansion_enabled=True,
        ),
        data_dir=data_dir,
    )
    body = client.get("/admin/rag").json()
    assert body["personal_history_enabled"] is False
    assert body["category_expansion_enabled"] is True
    assert body["overrides_present"] is True


def test_rag_put_persists_payload(client, admin_app, monkeypatch) -> None:
    _app, data_dir = admin_app
    from app.routers import admin as admin_module

    monkeypatch.setattr(
        admin_module, "_notify_core_restart_pending", _ok_notify()
    )

    response = client.put(
        "/admin/rag",
        json={
            "personal_history_enabled": False,
            "category_expansion_enabled": True,
        },
    )
    assert response.status_code == 200
    from app.rag_overrides import read_overrides as read_rag
    persisted = read_rag(data_dir=data_dir)
    assert persisted is not None
    assert persisted.personal_history_enabled is False
    assert persisted.category_expansion_enabled is True


def test_rag_delete_removes_overrides(
    client, admin_app, monkeypatch
) -> None:
    _app, data_dir = admin_app
    from app.rag_overrides import (
        RagOverrides,
        overrides_path as rag_overrides_path,
        write_overrides as write_rag,
    )
    write_rag(
        RagOverrides(personal_history_enabled=False),
        data_dir=data_dir,
    )
    assert rag_overrides_path(data_dir).is_file()

    from app.routers import admin as admin_module

    monkeypatch.setattr(
        admin_module, "_notify_core_restart_pending", _ok_notify()
    )

    response = client.delete("/admin/rag")
    assert response.status_code == 200
    assert response.json()["removed"] is True
    assert not rag_overrides_path(data_dir).is_file()
