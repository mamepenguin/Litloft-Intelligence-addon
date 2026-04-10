"""Tests for RAG-related configuration.

Covers the new RAG fields on FeaturesConfig, the new RagConfig
dataclass (with defaults defined in spec Phase A), and YAML
parsing integration with load_settings().
"""

import pytest

from app.config import (
    FeaturesConfig,
    LLMConfig,
    RagConfig,
    Settings,
    _parse_nested,
    load_settings,
)


# ---------------------------------------------------------------------------
# FeaturesConfig.rag
# ---------------------------------------------------------------------------


class TestFeaturesConfigRag:
    """Tests for the new `rag` bool field on FeaturesConfig."""

    def test_default_is_false(self):
        # RAG must default to disabled for security (file content
        # is sent to the LLM, so opt-in only).
        cfg = FeaturesConfig()
        assert cfg.rag is False

    def test_can_enable(self):
        cfg = FeaturesConfig(rag=True)
        assert cfg.rag is True

    def test_coexists_with_existing_fields(self):
        # Adding rag must not break other feature flags.
        cfg = FeaturesConfig(
            indexing=True,
            search=True,
            auto_tags="on_index",
            summaries="manual",
            rag=True,
        )
        assert cfg.indexing is True
        assert cfg.search is True
        assert cfg.auto_tags == "on_index"
        assert cfg.summaries == "manual"
        assert cfg.rag is True

    def test_parsed_from_yaml_section(self):
        data = {"features": {"rag": True}}
        result = _parse_nested(data, "features", FeaturesConfig)

        assert result.rag is True
        # Other defaults preserved.
        assert result.auto_tags == "false"
        assert result.summaries == "false"

    def test_yaml_false_remains_false(self):
        data = {"features": {"rag": False}}
        result = _parse_nested(data, "features", FeaturesConfig)

        assert result.rag is False

    def test_yaml_missing_section_keeps_default(self):
        result = _parse_nested({}, "features", FeaturesConfig)

        assert result.rag is False


# ---------------------------------------------------------------------------
# RagConfig dataclass
# ---------------------------------------------------------------------------


class TestRagConfigDefaults:
    """The RagConfig defaults must match spec Phase A values exactly."""

    def test_top_k_default(self):
        assert RagConfig().top_k == 5

    def test_max_context_chars_per_file_default(self):
        assert RagConfig().max_context_chars_per_file == 2000

    def test_max_total_context_chars_default(self):
        assert RagConfig().max_total_context_chars == 10000

    def test_max_tokens_default(self):
        assert RagConfig().max_tokens == 1024

    def test_transcript_window_seconds_default(self):
        assert RagConfig().transcript_window_seconds == 30.0

    def test_is_frozen(self):
        # Spec mandates frozen=True for all config dataclasses so
        # runtime mutation cannot silently change behavior.
        cfg = RagConfig()
        with pytest.raises(Exception):
            cfg.top_k = 10  # type: ignore[misc]


class TestRagConfigOverrides:
    """Overrides passed at construction should take effect."""

    def test_top_k_override(self):
        cfg = RagConfig(top_k=8)
        assert cfg.top_k == 8

    def test_max_context_chars_per_file_override(self):
        cfg = RagConfig(max_context_chars_per_file=5000)
        assert cfg.max_context_chars_per_file == 5000

    def test_max_total_context_chars_override(self):
        cfg = RagConfig(max_total_context_chars=20000)
        assert cfg.max_total_context_chars == 20000

    def test_max_tokens_override(self):
        cfg = RagConfig(max_tokens=2048)
        assert cfg.max_tokens == 2048

    def test_transcript_window_seconds_override(self):
        cfg = RagConfig(transcript_window_seconds=60.0)
        assert cfg.transcript_window_seconds == 60.0


# ---------------------------------------------------------------------------
# Settings.rag field
# ---------------------------------------------------------------------------


class TestSettingsRagField:
    """Settings must expose a .rag attribute of type RagConfig."""

    def test_settings_has_rag_attribute(self, make_settings):
        settings = make_settings()
        assert hasattr(settings, "rag")
        assert isinstance(settings.rag, RagConfig)

    def test_settings_rag_default_is_default_rag_config(self, make_settings):
        settings = make_settings()
        assert settings.rag == RagConfig()

    def test_settings_rag_override(self, make_settings):
        settings = make_settings(rag=RagConfig(top_k=10, max_tokens=4096))
        assert settings.rag.top_k == 10
        assert settings.rag.max_tokens == 4096


# ---------------------------------------------------------------------------
# YAML integration via load_settings
# ---------------------------------------------------------------------------


class TestLoadSettingsRag:
    """End-to-end YAML parsing for the RAG section."""

    def test_parses_rag_section_from_yaml(self, tmp_path, monkeypatch):
        config_file = tmp_path / "search-config.yml"
        config_file.write_text(
            "features:\n"
            "  rag: true\n"
            "rag:\n"
            "  top_k: 8\n"
            "  max_context_chars_per_file: 1500\n"
            "  max_total_context_chars: 12000\n"
            "  max_tokens: 2048\n"
            "  transcript_window_seconds: 45.0\n"
        )

        monkeypatch.setenv("SEARCH_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))

        result = load_settings()

        assert result.features.rag is True
        assert result.rag.top_k == 8
        assert result.rag.max_context_chars_per_file == 1500
        assert result.rag.max_total_context_chars == 12000
        assert result.rag.max_tokens == 2048
        assert result.rag.transcript_window_seconds == 45.0

    def test_missing_rag_section_uses_defaults(self, tmp_path, monkeypatch):
        config_file = tmp_path / "search-config.yml"
        config_file.write_text(
            "features:\n"
            "  rag: false\n"
        )

        monkeypatch.setenv("SEARCH_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))

        result = load_settings()

        # features.rag picked up, rag section missing -> defaults.
        assert result.features.rag is False
        assert result.rag == RagConfig()

    def test_partial_rag_section_keeps_unspecified_defaults(
        self, tmp_path, monkeypatch
    ):
        config_file = tmp_path / "search-config.yml"
        config_file.write_text(
            "features:\n"
            "  rag: true\n"
            "rag:\n"
            "  top_k: 3\n"
        )

        monkeypatch.setenv("SEARCH_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))

        result = load_settings()

        assert result.rag.top_k == 3
        # Unspecified -> defaults preserved.
        assert result.rag.max_context_chars_per_file == 2000
        assert result.rag.max_total_context_chars == 10000
        assert result.rag.max_tokens == 1024
        assert result.rag.transcript_window_seconds == 30.0

    def test_yaml_rag_enabled_without_llm_still_parses(
        self, tmp_path, monkeypatch
    ):
        # The config layer must not reject rag=true on its own — the
        # runtime gate (router 400) is what enforces the LLM requirement.
        config_file = tmp_path / "search-config.yml"
        config_file.write_text(
            "features:\n"
            "  rag: true\n"
        )

        monkeypatch.setenv("SEARCH_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))

        result = load_settings()

        assert result.features.rag is True
        # LLM defaults remain disabled-looking (no provider).
        assert result.llm.provider == "disabled"
