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
