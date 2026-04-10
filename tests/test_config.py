"""Tests for app.config module.

Covers YAML loading, settings construction, nested dataclass parsing,
file path resolution, and path traversal validation.
"""

import os
from pathlib import Path

import pytest

from app.config import (
    FeaturesConfig,
    LLMConfig,
    ModelConfig,
    SearchConfig,
    Settings,
    _parse_nested,
    load_config_file,
    load_settings,
    resolve_file_path,
    validate_file_path,
)


# ---------------------------------------------------------------------------
# _parse_nested
# ---------------------------------------------------------------------------


class TestParseNested:
    """Tests for _parse_nested: dict -> frozen dataclass conversion."""

    def test_parses_known_fields(self):
        data = {"llm": {"provider": "openai_compatible", "model": "gpt-4"}}
        result = _parse_nested(data, "llm", LLMConfig)

        assert result.provider == "openai_compatible"
        assert result.model == "gpt-4"
        # Defaults preserved for unspecified fields
        assert result.base_url == ""
        assert result.temperature == 0.3

    def test_ignores_unknown_keys(self):
        data = {"llm": {"provider": "openai_compatible", "bogus_key": "ignored"}}
        result = _parse_nested(data, "llm", LLMConfig)

        assert result.provider == "openai_compatible"
        assert not hasattr(result, "bogus_key")

    def test_returns_default_for_missing_section(self):
        result = _parse_nested({}, "llm", LLMConfig)

        assert result == LLMConfig()

    def test_returns_default_for_empty_section(self):
        result = _parse_nested({"llm": {}}, "llm", LLMConfig)

        assert result == LLMConfig()

    def test_returns_default_when_section_is_not_dict(self):
        result = _parse_nested({"llm": "not-a-dict"}, "llm", LLMConfig)

        assert result == LLMConfig()

    def test_returns_default_when_section_is_none(self):
        result = _parse_nested({"llm": None}, "llm", LLMConfig)

        assert result == LLMConfig()

    def test_works_with_different_dataclass(self):
        data = {"search": {"default_limit": 50, "max_limit": 200}}
        result = _parse_nested(data, "search", SearchConfig)

        assert result.default_limit == 50
        assert result.max_limit == 200
        assert result.alpha == 0.7  # default preserved


# ---------------------------------------------------------------------------
# load_config_file
# ---------------------------------------------------------------------------


class TestLoadConfigFile:
    """Tests for load_config_file: YAML file reading with graceful fallbacks."""

    def test_loads_valid_yaml(self, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text("models:\n  whisper: openai/whisper-large\n")

        result = load_config_file(config_file)

        assert result == {"models": {"whisper": "openai/whisper-large"}}

    def test_returns_empty_dict_for_missing_file(self, tmp_path):
        result = load_config_file(tmp_path / "nonexistent.yml")

        assert result == {}

    def test_returns_empty_dict_for_invalid_yaml(self, tmp_path):
        config_file = tmp_path / "bad.yml"
        config_file.write_text(":\n  :\n    [invalid yaml {{{\n")

        result = load_config_file(config_file)

        assert result == {}

    def test_returns_empty_dict_when_yaml_is_not_dict(self, tmp_path):
        config_file = tmp_path / "list.yml"
        config_file.write_text("- item1\n- item2\n")

        result = load_config_file(config_file)

        assert result == {}

    def test_returns_empty_dict_for_empty_file(self, tmp_path):
        config_file = tmp_path / "empty.yml"
        config_file.write_text("")

        result = load_config_file(config_file)

        assert result == {}


# ---------------------------------------------------------------------------
# load_settings
# ---------------------------------------------------------------------------


class TestLoadSettings:
    """Tests for load_settings: env vars + YAML -> Settings."""

    def test_loads_from_yaml_and_env(self, tmp_path, monkeypatch):
        config_file = tmp_path / "search-config.yml"
        config_file.write_text(
            "models:\n"
            "  whisper: openai/whisper-large\n"
            "llm:\n"
            "  provider: openai_compatible\n"
            "  base_url: http://localhost:11434/v1\n"
            "  model: llama3\n"
            "  output_language: ja\n"
        )

        monkeypatch.setenv("SEARCH_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))
        monkeypatch.setenv("ALLOWED_BASE_DIRS", "/drives/,/mnt/")
        monkeypatch.setenv("DRIVE_MOUNTS", "Videos=/drives/default,Photos=/drives/photos")

        result = load_settings()

        assert result.search_data_dir == Path(str(tmp_path / "data"))
        assert result.homevault_db_path == Path(str(tmp_path / "hv.db"))
        assert result.models.whisper == "openai/whisper-large"
        assert result.allowed_base_dirs == ("/drives/", "/mnt/")
        assert result.drive_mounts == {
            "Videos": "/drives/default",
            "Photos": "/drives/photos",
        }
        assert result.llm.provider == "openai_compatible"
        assert result.llm.output_language == "ja"

    def test_llm_api_key_env_overrides_yaml(self, tmp_path, monkeypatch):
        config_file = tmp_path / "search-config.yml"
        config_file.write_text(
            "llm:\n"
            "  provider: openai_compatible\n"
            "  api_key: yaml-key\n"
            "  base_url: http://localhost:11434/v1\n"
            "  model: llama3\n"
            "  output_language: en\n"
        )

        monkeypatch.setenv("SEARCH_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))
        monkeypatch.setenv("LLM_API_KEY", "env-override-key")

        result = load_settings()

        assert result.llm.api_key == "env-override-key"
        # Other LLM fields preserved from YAML
        assert result.llm.provider == "openai_compatible"
        assert result.llm.model == "llama3"
        assert result.llm.output_language == "en"

    def test_llm_api_key_env_empty_does_not_override(self, tmp_path, monkeypatch):
        config_file = tmp_path / "search-config.yml"
        config_file.write_text(
            "llm:\n"
            "  api_key: yaml-key\n"
        )

        monkeypatch.setenv("SEARCH_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))
        monkeypatch.setenv("LLM_API_KEY", "")

        result = load_settings()

        assert result.llm.api_key == "yaml-key"

    def test_drive_mounts_empty_string(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SEARCH_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(tmp_path / "missing.yml"))
        monkeypatch.setenv("DRIVE_MOUNTS", "")

        result = load_settings()

        assert result.drive_mounts == {}

    def test_allowed_base_dirs_strips_whitespace(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SEARCH_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
        monkeypatch.setenv("SEARCH_CONFIG_PATH", str(tmp_path / "missing.yml"))
        monkeypatch.setenv("ALLOWED_BASE_DIRS", "  /a/ , /b/  ,  ")

        result = load_settings()

        assert result.allowed_base_dirs == ("/a/", "/b/")

    def test_defaults_when_no_env_vars(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SEARCH_DATA_DIR", raising=False)
        monkeypatch.delenv("HOMEVAULT_DB_PATH", raising=False)
        monkeypatch.delenv("SEARCH_CONFIG_PATH", raising=False)
        monkeypatch.delenv("ALLOWED_BASE_DIRS", raising=False)
        monkeypatch.delenv("DRIVE_MOUNTS", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        result = load_settings()

        assert result.search_data_dir == Path("/search-data")
        assert result.homevault_db_path == Path("/data/homevault.db")
        assert result.allowed_base_dirs == ("/drives/",)


# ---------------------------------------------------------------------------
# resolve_file_path
# ---------------------------------------------------------------------------


class TestResolveFilePath:
    """Tests for resolve_file_path: drive mount + relative -> absolute path."""

    def test_resolves_existing_drive(self, monkeypatch, make_settings):
        settings = make_settings(
            drive_mounts={"Videos": "/drives/videos"}
        )
        monkeypatch.setattr("app.config.settings", settings)

        result = resolve_file_path("Videos", "subfolder/file.mp4")

        assert result == "/drives/videos/subfolder/file.mp4"

    def test_returns_none_for_missing_drive(self, monkeypatch, make_settings):
        settings = make_settings(drive_mounts={"Videos": "/drives/videos"})
        monkeypatch.setattr("app.config.settings", settings)

        result = resolve_file_path("NonExistent", "file.mp4")

        assert result is None

    def test_returns_none_for_empty_mounts(self, monkeypatch, make_settings):
        settings = make_settings(drive_mounts={})
        monkeypatch.setattr("app.config.settings", settings)

        result = resolve_file_path("Videos", "file.mp4")

        assert result is None


# ---------------------------------------------------------------------------
# validate_file_path
# ---------------------------------------------------------------------------


class TestValidateFilePath:
    """Tests for validate_file_path: path traversal prevention."""

    def test_allows_path_under_allowed_dir(self, monkeypatch, make_settings):
        settings = make_settings(allowed_base_dirs=("/drives/",))
        monkeypatch.setattr("app.config.settings", settings)

        result = validate_file_path("/drives/videos/file.mp4")

        assert result is True

    def test_rejects_path_outside_allowed_dirs(self, monkeypatch, make_settings):
        settings = make_settings(allowed_base_dirs=("/drives/",))
        monkeypatch.setattr("app.config.settings", settings)

        result = validate_file_path("/etc/passwd")

        assert result is False

    def test_rejects_traversal_attack(self, monkeypatch, make_settings, tmp_path):
        # Create a real directory to test realpath resolution
        allowed = tmp_path / "drives"
        allowed.mkdir()
        settings = make_settings(allowed_base_dirs=(str(allowed),))
        monkeypatch.setattr("app.config.settings", settings)

        # Attempt traversal: /drives/../etc/passwd resolves to /etc/passwd
        result = validate_file_path(str(allowed / ".." / "etc" / "passwd"))

        assert result is False

    def test_allows_multiple_base_dirs(self, monkeypatch, make_settings):
        settings = make_settings(
            allowed_base_dirs=("/drives/", "/mnt/storage/")
        )
        monkeypatch.setattr("app.config.settings", settings)

        assert validate_file_path("/mnt/storage/file.txt") is True
        assert validate_file_path("/drives/file.txt") is True
        assert validate_file_path("/other/file.txt") is False

    def test_rejects_empty_allowed_dirs(self, monkeypatch, make_settings):
        settings = make_settings(allowed_base_dirs=())
        monkeypatch.setattr("app.config.settings", settings)

        result = validate_file_path("/drives/file.mp4")

        assert result is False
