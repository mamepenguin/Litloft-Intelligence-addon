"""SPEC-ADDON-016: the admin transcription summary names the Whisper model the worker loads."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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

from app.workers import whisper as whisper_mod
from tests.whisper_fakes import load_settings_from, use_settings

SMALL = "openai/whisper-small"
LARGE_V3 = "openai/whisper-large-v3"
DEFAULT_WHISPER = "openai/whisper-large-v3-turbo"

OTHER_PROVIDERS = {
    "openai_compatible": {"base_url": "https://stt.example/v1", "model": "stt-model-a"},
    "deepgram": {"model": "nova-2"},
    "elevenlabs_scribe": {"model_id": "scribe_v1"},
    "assemblyai": {"model": "nano"},
    "gemini": {"model": "gemini-2.5-pro", "output_language": "en"},
}


def _client(tmp_path, monkeypatch, data: dict) -> TestClient:
    settings = load_settings_from(tmp_path, monkeypatch, data)
    use_settings(monkeypatch, settings)
    monkeypatch.delitem(sys.modules, "app.routers.admin", raising=False)
    from fastapi import FastAPI

    from app.routers import admin

    if hasattr(admin, "settings"):
        monkeypatch.setattr(admin, "settings", settings)
    app = FastAPI()
    app.include_router(admin.router)
    return TestClient(app)


def _summary(tmp_path, monkeypatch, data: dict) -> dict:
    response = _client(tmp_path, monkeypatch, data).get("/admin/transcription")
    assert response.status_code == 200
    return response.json()["search_config_summary"]


def _model_passed_to_resolver(tmp_path, monkeypatch, data: dict) -> str:
    use_settings(monkeypatch, load_settings_from(tmp_path, monkeypatch, data))
    fake = types.ModuleType("faster_whisper")

    class WhisperModel:
        def __init__(self, *args, **kwargs):
            self.feature_extractor = types.SimpleNamespace(sampling_rate=16000)

    class BatchedInferencePipeline:
        def __init__(self, *args, **kwargs):
            pass

    fake.WhisperModel = WhisperModel
    fake.BatchedInferencePipeline = BatchedInferencePipeline
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setattr(whisper_mod, "_model", None)
    monkeypatch.setattr(whisper_mod, "_batched_pipeline", None)
    monkeypatch.setattr(whisper_mod, "_loaded", False)

    seen: list[str] = []
    real = whisper_mod._resolve_model_size

    def spy(name):
        seen.append(name)
        return real(name)

    monkeypatch.setattr(whisper_mod, "_resolve_model_size", spy)
    whisper_mod._ensure_loaded()
    assert len(seen) == 1
    return seen[0]


def test_spec_addon_016_i1_shows_models_whisper_not_the_unused_key(tmp_path, monkeypatch):
    summary = _summary(
        tmp_path,
        monkeypatch,
        {
            "models": {"whisper": SMALL},
            "transcription": {"whisper_local": {"model": LARGE_V3}},
        },
    )

    assert summary["whisper_local"] == {"models.whisper": SMALL}


def test_spec_addon_016_i2_shows_the_default_when_neither_key_is_set(tmp_path, monkeypatch):
    summary = _summary(tmp_path, monkeypatch, {})

    assert summary["whisper_local"] == {"models.whisper": DEFAULT_WHISPER}


@pytest.mark.parametrize(
    "models_section, expected",
    [
        ({}, DEFAULT_WHISPER),
        ({"models": {"whisper": SMALL}}, SMALL),
    ],
    ids=["models-absent", "models-set"],
)
def test_spec_addon_016_i2_legacy_indexing_whisper_model_is_not_shown(
    tmp_path, monkeypatch, models_section, expected
):
    data = {**models_section, "indexing": {"whisper": {"model": LARGE_V3}}}

    summary = _summary(tmp_path, monkeypatch, data)

    assert summary["whisper_local"] == {"models.whisper": expected}


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"models": {"whisper": SMALL}},
        {"models": {"whisper": SMALL}, "transcription": {"whisper_local": {"model": LARGE_V3}}},
        {"indexing": {"whisper": {"model": LARGE_V3}}},
        {"models": {"whisper": "deepdml/faster-whisper-large-v3-turbo-ct2"}},
    ],
    ids=["empty", "models-only", "both-keys", "legacy-only", "hf-repo-id"],
)
def test_spec_addon_016_i3_shown_value_is_what_the_worker_loads(tmp_path, monkeypatch, data):
    loaded = _model_passed_to_resolver(tmp_path, monkeypatch, data)

    summary = _summary(tmp_path, monkeypatch, data)

    assert summary["whisper_local"] == {"models.whisper": loaded}


def test_spec_addon_016_i4_other_providers_entries_are_unchanged(tmp_path, monkeypatch):
    summary = _summary(
        tmp_path,
        monkeypatch,
        {
            "models": {"whisper": SMALL},
            "transcription": {"whisper_local": {"model": LARGE_V3}, **OTHER_PROVIDERS},
        },
    )

    assert {k: v for k, v in summary.items() if k != "whisper_local"} == OTHER_PROVIDERS


@pytest.mark.parametrize(
    "unused",
    [
        {"transcription": {"whisper_local": {"model": LARGE_V3}}},
        {"indexing": {"whisper": {"model": LARGE_V3}}},
    ],
    ids=["transcription-whisper_local-model", "legacy-indexing-whisper-model"],
)
def test_spec_addon_016_i5_unused_key_parses_and_changes_nothing_loaded(
    tmp_path, monkeypatch, unused
):
    without = _model_passed_to_resolver(tmp_path, monkeypatch, {"models": {"whisper": SMALL}})
    with_unused = _model_passed_to_resolver(
        tmp_path, monkeypatch, {"models": {"whisper": SMALL}, **unused}
    )

    assert without == with_unused == SMALL


def test_spec_addon_016_reads_the_loaded_settings_not_the_file_on_disk(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, {"models": {"whisper": SMALL}})
    Path(os.environ["SEARCH_CONFIG_PATH"]).write_text(
        f"models:\n  whisper: {LARGE_V3}\n", encoding="utf-8"
    )

    summary = client.get("/admin/transcription").json()["search_config_summary"]

    assert summary["whisper_local"] == {"models.whisper": SMALL}
