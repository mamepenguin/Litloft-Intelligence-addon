"""Shared fixtures for intelligence addon tests.

Provides factories and helpers for creating test configurations
without importing heavy ML dependencies.
"""

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

# Stub heavy ML dependencies at collection time so individual test
# files don't need to repeat the boilerplate. Crucially, numpy's
# `bool_` attribute is set to the real `bool` type so pytest.approx's
# `isinstance(val, np.bool_)` check keeps working — a plain MagicMock
# attribute would break every float comparison in the suite.
_ml_stubs = (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
)
for _mod in _ml_stubs:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

if "numpy" not in sys.modules:
    try:
        import numpy  # noqa: F401  — prefer the real package when installed
    except ImportError:
        _numpy_stub = MagicMock()
        _numpy_stub.bool_ = bool  # Real type for pytest.approx
        sys.modules["numpy"] = _numpy_stub

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
            "output_language": "auto",
        }
        return LLMConfig(**{**defaults, **overrides})

    return _create


@pytest.fixture()
def make_settings(tmp_path):
    """Factory fixture for creating Settings instances with sensible defaults."""

    def _create(**overrides) -> Settings:
        defaults = {
            "intelligence_data_dir": tmp_path / "intelligence-data",
            "litloft_db_path": tmp_path / "litloft.db",
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
def mock_litloft_db():
    """Create a mock get_litloft_db context manager yielding a MagicMock session."""
    session = MagicMock()

    @contextmanager
    def _get_litloft_db():
        yield session

    return _get_litloft_db, session


@pytest.fixture(autouse=True)
def _clean_vision_capability_cache():
    """A probed verdict must not outlive the test that provoked it.

    ``app.llm`` caches capability per (base_url, model) for the process,
    so without this a stub in one test answers the probe in another —
    and the two are usually configured identically. Session-wide rather
    than per-file: the cache is module state, so any test that reaches
    ``_classify_vision_rejection`` inherits it, not only the ones that
    meant to.
    """
    from app.llm import reset_vision_capability_cache

    reset_vision_capability_cache()
    yield
    reset_vision_capability_cache()
