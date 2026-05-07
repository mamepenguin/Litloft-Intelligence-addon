"""Phase 1A foundation tests for the indexing.whisper -> transcription shim.

Spec ``2026-05-07-cloud-transcription-providers.md`` §"旧キー → 新
キー後方互換 shim". The new ``transcription`` config tree replaces
the old ``indexing.whisper`` section but a 1-release shim preserves
backward compatibility:

* Old ``indexing.whisper.*`` is parsed (no behaviour change for
  existing deployments).
* If ``transcription.whisper_local.*`` is also present, the new key
  wins (new takes precedence over old when both are defined).
* If only the old keys are present, a deprecation warning is logged
  exactly once per process load.
* The module-level ``settings.transcription`` always has a populated
  ``whisper_local`` sub-config so callers (refine.py, future workers)
  can rely on a single read site.
"""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock

import pytest

# Stub heavy ML deps before importing app.
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


def _reset_warning_flag():
    """Clear the once-per-process deprecation flag between tests."""
    import app.config as cfg
    cfg._whisper_deprecation_logged = False


@pytest.fixture(autouse=True)
def _reset_flag():
    _reset_warning_flag()
    yield
    _reset_warning_flag()


def test_transcription_config_has_whisper_local_default() -> None:
    """A clean Settings has a populated transcription tree."""
    from app.config import Settings, TranscriptionConfig

    tc = TranscriptionConfig()
    # whisper_local must exist as a sub-config and carry the same
    # defaults as WhisperIndexConfig (so existing call sites that
    # read .beam_size / .initial_prompt etc. keep working).
    assert tc.whisper_local.beam_size == 1
    assert tc.whisper_local.initial_prompt == ""
    assert tc.provider == "whisper_local"


def test_old_indexing_whisper_populates_new_transcription() -> None:
    """When only legacy keys are set, ``transcription.whisper_local`` is
    populated from them so refine.py / workers can read a single site."""
    import app.config as cfg

    config_data = {
        "indexing": {
            "whisper": {
                "beam_size": 5,
                "initial_prompt": "legacy prompt",
                "no_speech_threshold": 0.6,
            }
        }
    }
    transcription = cfg._parse_transcription(config_data)
    assert transcription.whisper_local.beam_size == 5
    assert transcription.whisper_local.initial_prompt == "legacy prompt"
    assert transcription.whisper_local.no_speech_threshold == 0.6


def test_new_keys_take_precedence_over_old() -> None:
    """Both keys defined: new wins."""
    import app.config as cfg

    config_data = {
        "indexing": {
            "whisper": {"beam_size": 5, "initial_prompt": "old"},
        },
        "transcription": {
            "whisper_local": {"beam_size": 7, "initial_prompt": "new"},
        },
    }
    transcription = cfg._parse_transcription(config_data)
    assert transcription.whisper_local.beam_size == 7
    assert transcription.whisper_local.initial_prompt == "new"


def test_only_new_keys_no_deprecation_warning(caplog) -> None:
    """No old keys present -> no deprecation log."""
    import app.config as cfg

    config_data = {
        "transcription": {
            "whisper_local": {"beam_size": 4},
        }
    }
    with caplog.at_level(logging.WARNING):
        transcription = cfg._parse_transcription(config_data)

    assert transcription.whisper_local.beam_size == 4
    deprecation_messages = [
        r.message for r in caplog.records if "deprecated" in r.message.lower()
    ]
    assert deprecation_messages == []


def test_old_keys_emit_deprecation_warning_once(caplog) -> None:
    """Old keys present -> warning logged once with the removal date."""
    import app.config as cfg

    config_data = {
        "indexing": {
            "whisper": {"beam_size": 3},
        }
    }
    with caplog.at_level(logging.WARNING):
        cfg._parse_transcription(config_data)
        # Second call in the same process must not log again.
        cfg._parse_transcription(config_data)

    deprecation_messages = [
        r.message for r in caplog.records
        if "indexing.whisper" in r.message and "deprecated" in r.message.lower()
    ]
    assert len(deprecation_messages) == 1
    assert "2026-07-07" in deprecation_messages[0]


def test_provider_field_defaults_to_whisper_local() -> None:
    from app.config import TranscriptionConfig
    assert TranscriptionConfig().provider == "whisper_local"


def test_provider_field_reads_yaml_section() -> None:
    import app.config as cfg

    config_data = {
        "transcription": {
            "provider": "deepgram",
        }
    }
    transcription = cfg._parse_transcription(config_data)
    assert transcription.provider == "deepgram"


def test_settings_carry_transcription_field() -> None:
    """``load_settings`` populates ``settings.transcription`` end-to-end."""
    from app.config import Settings

    fields = Settings.__dataclass_fields__
    assert "transcription" in fields


def test_transcription_subconfigs_have_provider_specific_fields() -> None:
    """Spec lists 4 provider sub-configs with their own fields."""
    from app.config import (
        TranscriptionConfig,
        WhisperLocalConfig,
        OpenAICompatibleProviderConfig,
        DeepgramProviderConfig,
        ElevenLabsScribeProviderConfig,
    )

    tc = TranscriptionConfig()
    assert isinstance(tc.whisper_local, WhisperLocalConfig)
    assert isinstance(tc.openai_compatible, OpenAICompatibleProviderConfig)
    assert isinstance(tc.deepgram, DeepgramProviderConfig)
    assert isinstance(tc.elevenlabs_scribe, ElevenLabsScribeProviderConfig)

    # Spot-check provider-specific fields from the spec.
    assert tc.openai_compatible.base_url == "https://api.openai.com/v1"
    assert tc.openai_compatible.model == "whisper-1"
    assert tc.deepgram.model == "nova-3"
    assert tc.deepgram.diarize is True
    assert tc.elevenlabs_scribe.model_id == "scribe_v1"


def test_load_settings_with_only_old_keys_populates_transcription(
    tmp_path, monkeypatch,
) -> None:
    """End-to-end: a YAML with only old ``indexing.whisper`` keys
    produces a Settings whose ``transcription.whisper_local`` is
    populated from those values."""
    import app.config as cfg

    yml = tmp_path / "search-config.yml"
    yml.write_text(
        "indexing:\n"
        "  whisper:\n"
        "    beam_size: 9\n"
        "    initial_prompt: from-old-key\n"
    )
    monkeypatch.setenv("SEARCH_CONFIG_PATH", str(yml))
    monkeypatch.setenv(
        "INTELLIGENCE_DATA_DIR", str(tmp_path / "intelligence-data"),
    )
    settings = cfg.load_settings()
    assert settings.transcription.whisper_local.beam_size == 9
    assert settings.transcription.whisper_local.initial_prompt == "from-old-key"
    # Old indexing.whisper still parsed for legacy reads.
    assert settings.indexing.whisper.beam_size == 9
