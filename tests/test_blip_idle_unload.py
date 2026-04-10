"""Tests for BLIP idle unload behavior.

Verifies that check_idle_unload() respects the configured timeout
and only unloads when the model has been idle long enough.
"""

import sys
from unittest.mock import MagicMock

import pytest

# Stub out heavy dependencies before importing app.workers.blip
for _mod in (
    "PIL", "PIL.Image", "torch", "transformers",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app import config as app_config
from app.workers import blip


@pytest.fixture(autouse=True)
def reset_blip_state():
    """Reset BLIP module state before each test."""
    blip._model = None
    blip._processor = None
    blip._loaded = False
    blip._last_used = 0.0
    yield
    blip._model = None
    blip._processor = None
    blip._loaded = False
    blip._last_used = 0.0


def _patch_memory(
    monkeypatch: pytest.MonkeyPatch, blip_idle_unload: int,
) -> None:
    """Patch settings.memory.blip_idle_unload in the blip module."""
    monkeypatch.setattr(
        blip, "settings", app_config.Settings(
            search_data_dir=app_config.settings.search_data_dir,
            homevault_db_path=app_config.settings.homevault_db_path,
            model_cache_dir=app_config.settings.model_cache_dir,
            search_db_path=app_config.settings.search_db_path,
            memory=app_config.MemoryConfig(
                whisper_idle_unload=300,
                blip_idle_unload=blip_idle_unload,
            ),
        ),
    )


class TestUnloadModel:
    """Tests for unload_model()."""

    def test_unload_when_loaded(self):
        """unload_model clears state when model is loaded."""
        blip._model = MagicMock()
        blip._processor = MagicMock()
        blip._loaded = True

        blip.unload_model()

        assert blip._model is None
        assert blip._processor is None
        assert blip._loaded is False

    def test_unload_when_not_loaded_is_safe(self):
        """unload_model is safe to call when no model is loaded."""
        blip._model = None
        blip._processor = None
        blip._loaded = False

        # Should not raise
        blip.unload_model()

        assert blip._model is None


class TestCheckIdleUnload:
    """Tests for check_idle_unload()."""

    def test_no_unload_when_timeout_zero(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        """timeout=0 disables idle unload entirely."""
        _patch_memory(monkeypatch, blip_idle_unload=0)
        blip._model = MagicMock()
        blip._processor = MagicMock()
        blip._loaded = True
        blip._last_used = 0.0  # very old

        blip.check_idle_unload()

        # Model should still be loaded
        assert blip._loaded is True
        assert blip._model is not None

    def test_no_unload_when_not_loaded(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        """Skip gracefully when no model is loaded."""
        _patch_memory(monkeypatch, blip_idle_unload=300)
        blip._model = None
        blip._loaded = False

        # Should not raise
        blip.check_idle_unload()

        assert blip._loaded is False

    def test_no_unload_when_recently_used(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        """Model stays loaded if used within the idle window."""
        _patch_memory(monkeypatch, blip_idle_unload=300)

        import time
        blip._model = MagicMock()
        blip._processor = MagicMock()
        blip._loaded = True
        blip._last_used = time.monotonic() - 100  # 100 seconds ago (< 300)

        blip.check_idle_unload()

        assert blip._loaded is True
        assert blip._model is not None

    def test_unload_when_idle_exceeded(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        """Model is unloaded once idle time exceeds the timeout."""
        _patch_memory(monkeypatch, blip_idle_unload=300)

        import time
        blip._model = MagicMock()
        blip._processor = MagicMock()
        blip._loaded = True
        blip._last_used = time.monotonic() - 400  # 400 seconds ago (> 300)

        blip.check_idle_unload()

        assert blip._loaded is False
        assert blip._model is None
        assert blip._processor is None

    def test_unload_exact_boundary(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        """Idle exactly at timeout does NOT unload (uses strict >)."""
        _patch_memory(monkeypatch, blip_idle_unload=300)

        import time
        blip._model = MagicMock()
        blip._processor = MagicMock()
        blip._loaded = True
        # Make elapsed just under the threshold
        blip._last_used = time.monotonic() - 299

        blip.check_idle_unload()

        assert blip._loaded is True
