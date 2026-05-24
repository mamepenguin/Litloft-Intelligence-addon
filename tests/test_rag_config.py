"""Tests for RAG-related configuration.

Covers the RAG fields on FeaturesConfig, the tuned RagConfig defaults,
and YAML parsing integration with load_settings().
"""

import pytest

from app.config import (
    CategoryExpansionConfig,
    FeaturesConfig,
    HierarchicalRagConfig,
    LLMConfig,
    PersonalHistoryConfig,
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

    def test_default_is_enabled(self):
        # RAG is enabled by default; runtime still requires an LLM provider.
        cfg = FeaturesConfig()
        assert cfg.rag is True

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
        assert result.auto_tags == "manual"
        assert result.summaries == "manual"

    def test_yaml_false_remains_false(self):
        data = {"features": {"rag": False}}
        result = _parse_nested(data, "features", FeaturesConfig)

        assert result.rag is False

    def test_yaml_missing_section_keeps_default(self):
        result = _parse_nested({}, "features", FeaturesConfig)

        assert result.rag is True


# ---------------------------------------------------------------------------
# RagConfig dataclass
# ---------------------------------------------------------------------------


class TestRagConfigDefaults:
    """The RagConfig defaults must match the current tuned values."""

    def test_top_k_default(self):
        assert RagConfig().top_k == 5

    def test_max_context_chars_per_file_default(self):
        assert RagConfig().max_context_chars_per_file == 3500

    def test_max_total_context_chars_default(self):
        assert RagConfig().max_total_context_chars == 17500

    def test_max_tokens_default(self):
        assert RagConfig().max_tokens == 2048

    def test_transcript_window_seconds_default(self):
        assert RagConfig().transcript_window_seconds == 60.0

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

        monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path / "data"))
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

        monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path / "data"))
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

        monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))

        result = load_settings()

        assert result.rag.top_k == 3
        # Unspecified -> defaults preserved.
        assert result.rag.max_context_chars_per_file == 3500
        assert result.rag.max_total_context_chars == 17500
        assert result.rag.max_tokens == 2048
        assert result.rag.transcript_window_seconds == 60.0

    def test_parses_hierarchical_section_from_yaml(self, tmp_path, monkeypatch):
        config_file = tmp_path / "search-config.yml"
        config_file.write_text(
            "features:\n"
            "  rag: true\n"
            "rag:\n"
            "  hierarchical:\n"
            "    enabled: true\n"
            "    coarse_top_k: 30\n"
            "    coarse_score_threshold: 0.4\n"
            "    min_drive_files_for_shortlist: 25\n"
            "    fallback_full_search: false\n"
            "    clue_count: 5\n"
        )

        monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))

        result = load_settings()

        h = result.rag.hierarchical
        assert h.enabled is True
        assert h.coarse_top_k == 30
        assert h.coarse_score_threshold == 0.4
        assert h.min_drive_files_for_shortlist == 25
        assert h.fallback_full_search is False
        assert h.clue_count == 5

    def test_missing_hierarchical_section_keeps_defaults(
        self, tmp_path, monkeypatch
    ):
        config_file = tmp_path / "search-config.yml"
        config_file.write_text(
            "rag:\n"
            "  top_k: 3\n"
        )

        monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))

        result = load_settings()

        # Defaults preserved when the nested block is absent.
        assert result.rag.hierarchical == HierarchicalRagConfig()
        assert result.rag.hierarchical.enabled is True

    def test_parses_personal_history_section_from_yaml(
        self, tmp_path, monkeypatch
    ):
        config_file = tmp_path / "search-config.yml"
        config_file.write_text(
            "rag:\n"
            "  personal_history:\n"
            "    enabled: true\n"
            "    max_lookback_days: 90\n"
            "    fallback_when_empty: strict\n"
        )

        monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))

        result = load_settings()

        ph = result.rag.personal_history
        assert ph.enabled is True
        assert ph.max_lookback_days == 90
        assert ph.fallback_when_empty == "strict"

    def test_missing_personal_history_section_keeps_defaults(
        self, tmp_path, monkeypatch
    ):
        config_file = tmp_path / "search-config.yml"
        config_file.write_text("rag:\n  top_k: 3\n")

        monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))

        result = load_settings()

        assert result.rag.personal_history == PersonalHistoryConfig()
        assert result.rag.personal_history.enabled is True
        assert result.rag.personal_history.max_lookback_days == 365
        assert result.rag.personal_history.fallback_when_empty == "graceful"

    def test_parses_category_expansion_section_from_yaml(
        self, tmp_path, monkeypatch
    ):
        config_file = tmp_path / "search-config.yml"
        config_file.write_text(
            "rag:\n"
            "  category_expansion:\n"
            "    enabled: true\n"
            "    max_terms: 12\n"
        )

        monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))

        result = load_settings()

        ce = result.rag.category_expansion
        assert ce.enabled is True
        assert ce.max_terms == 12

    def test_missing_category_expansion_section_keeps_defaults(
        self, tmp_path, monkeypatch
    ):
        config_file = tmp_path / "search-config.yml"
        config_file.write_text("rag:\n  top_k: 3\n")

        monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))

        result = load_settings()

        assert result.rag.category_expansion == CategoryExpansionConfig()
        assert result.rag.category_expansion.enabled is False
        assert result.rag.category_expansion.max_terms == 8

    def test_unknown_keys_in_personal_history_are_ignored(
        self, tmp_path, monkeypatch
    ):
        # _parse_nested filters by __dataclass_fields__ so a typo
        # ("max_look_back_days") must not blow up dataclass construction.
        config_file = tmp_path / "search-config.yml"
        config_file.write_text(
            "rag:\n"
            "  personal_history:\n"
            "    enabled: true\n"
            "    max_look_back_days: 30\n"
        )

        monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))

        result = load_settings()

        # Recognised key took effect, typo silently dropped.
        assert result.rag.personal_history.enabled is True
        assert result.rag.personal_history.max_lookback_days == 365

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

        monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))

        result = load_settings()

        assert result.features.rag is True
        # LLM defaults remain disabled-looking (no provider).
        assert result.llm.provider == "disabled"
