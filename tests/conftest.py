"""Shared fixtures for intelligence addon tests.

Provides factories and helpers for creating test configurations
without importing heavy ML dependencies.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.config import (
    FeaturesConfig,
    IndexingConfig,
    LLMConfig,
    MemoryConfig,
    ModelConfig,
    SearchConfig,
    Settings,
    WorkerConfig,
)


@pytest.fixture()
def llm_config_factory():
    """Factory fixture for creating LLMConfig instances with overrides."""

    def _create(**overrides) -> LLMConfig:
        defaults = {
            "provider": "openai_compatible",
            "base_url": "http://localhost:11434/v1",
            "api_key": "test-key",
            "model": "test-model",
            "max_tokens": 2048,
            "temperature": 0.3,
            "tag_language": "auto",
        }
        return LLMConfig(**{**defaults, **overrides})

    return _create


@pytest.fixture()
def make_settings(tmp_path):
    """Factory fixture for creating Settings instances with sensible defaults."""

    def _create(**overrides) -> Settings:
        defaults = {
            "search_data_dir": tmp_path / "search-data",
            "homevault_db_path": tmp_path / "homevault.db",
            "model_cache_dir": tmp_path / "models",
            "search_db_path": tmp_path / "search.db",
            "allowed_base_dirs": ("/drives/",),
            "drive_mounts": {},
            "models": ModelConfig(),
            "search": SearchConfig(),
            "indexing": IndexingConfig(),
            "workers": WorkerConfig(),
            "memory": MemoryConfig(),
            "features": FeaturesConfig(),
            "llm": LLMConfig(),
        }
        return Settings(**{**defaults, **overrides})

    return _create


@pytest.fixture()
def mock_search_db():
    """Create a mock get_search_db context manager yielding a MagicMock session."""
    session = MagicMock()

    @contextmanager
    def _get_search_db():
        yield session

    return _get_search_db, session


@pytest.fixture()
def mock_homevault_db():
    """Create a mock get_homevault_db context manager yielding a MagicMock session."""
    session = MagicMock()

    @contextmanager
    def _get_homevault_db():
        yield session

    return _get_homevault_db, session
