"""Config parsing tests for the vision_describe feature.

``LLMConfig`` must grow three vision-specific fields parsed from
``search-config.yml``'s ``llm`` section:

* ``vision_model`` (str, default "")
* ``vision_max_tokens`` (int, default 1024)
* ``vision_temperature`` (float, default 0.1)

``FeaturesConfig`` must grow a ``vision_describe`` field with the same
3-mode shape as ``detailed_summaries`` / ``transcript_refine``:

* "false" — fully disabled
* "manual" — UI-triggered only
* "on_index" — auto-run after indexing completes

Graceful degradation: when ``vision_model`` is empty/unset, the feature
must be reported as unavailable regardless of ``features.vision_describe``
value. This is the runtime check the worker/router performs; we pin it
here as a helper so implementation can use a single predicate.
"""

from __future__ import annotations

import sys
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

from app.config import (  # noqa: E402
    FeaturesConfig,
    LLMConfig,
    _parse_nested,
)


class TestLLMConfigVisionFields:
    """LLMConfig must carry vision_model / vision_max_tokens / vision_temperature."""

    def test_defaults(self):
        cfg = LLMConfig()
        # vision_model empty by default so auto/manual requests degrade
        # gracefully until a provider is configured.
        assert cfg.vision_model == ""
        # Tight but generous defaults per spec.
        assert cfg.vision_max_tokens == 1024
        assert cfg.vision_temperature == pytest.approx(0.1)

    def test_parses_vision_fields_from_yaml(self):
        data = {
            "llm": {
                "provider": "openai_compatible",
                "base_url": "http://localhost:11434/v1",
                "model": "gemma2:27b",
                "vision_model": "llava:13b",
                "vision_max_tokens": 2048,
                "vision_temperature": 0.0,
            }
        }
        cfg = _parse_nested(data, "llm", LLMConfig)

        assert cfg.vision_model == "llava:13b"
        assert cfg.vision_max_tokens == 2048
        assert cfg.vision_temperature == pytest.approx(0.0)
        # Text-mode fields still respected.
        assert cfg.model == "gemma2:27b"

    def test_vision_fields_independent_of_text_fields(self):
        """Text model/temperature must not leak into vision overrides."""
        cfg = LLMConfig(
            provider="openai_compatible",
            model="gemma2:27b",
            max_tokens=2048,
            temperature=0.3,
            vision_model="llava:13b",
            vision_max_tokens=512,
            vision_temperature=0.2,
        )
        assert cfg.model == "gemma2:27b"
        assert cfg.vision_model == "llava:13b"
        assert cfg.max_tokens == 2048
        assert cfg.vision_max_tokens == 512
        assert cfg.temperature == pytest.approx(0.3)
        assert cfg.vision_temperature == pytest.approx(0.2)


class TestFeaturesConfigVisionDescribe:
    """features.vision_describe must mirror the 3-mode string pattern."""

    def test_default_is_manual(self):
        """Default allows manual runs once a vision model is configured."""
        cfg = FeaturesConfig()
        assert cfg.vision_describe == "manual"

    @pytest.mark.parametrize("mode", ["false", "manual", "on_index"])
    def test_accepts_three_modes(self, mode):
        cfg = FeaturesConfig(vision_describe=mode)
        assert cfg.vision_describe == mode

    def test_parsed_from_yaml(self):
        data = {"features": {"vision_describe": "manual"}}
        cfg = _parse_nested(data, "features", FeaturesConfig)
        assert cfg.vision_describe == "manual"


class TestVisionFeatureAvailability:
    """Graceful degradation: vision_model gates feature availability.

    Implementation expected to expose ``is_vision_describe_available(settings)``
    (or equivalent) returning True iff ``features.vision_describe != "false"``
    AND ``llm.vision_model`` is set to a non-empty string.

    This test locks in the interface name so the worker/router can
    single-point the check.
    """

    def _available(self, features_mode: str, vision_model: str) -> bool:
        """Thin shim — mirror of the planned helper."""
        try:
            from app.config import is_vision_describe_available
        except ImportError:
            pytest.skip("is_vision_describe_available not yet implemented")

        from app.config import (
            FeaturesConfig,
            LLMConfig,
        )

        # Build a settings-like object carrying only the fields the
        # predicate cares about.
        class _Fake:
            features = FeaturesConfig(vision_describe=features_mode)
            llm = LLMConfig(vision_model=vision_model)

        return is_vision_describe_available(_Fake())

    def test_false_mode_always_unavailable(self):
        assert self._available("false", "llava:13b") is False
        assert self._available("false", "") is False

    def test_manual_requires_vision_model(self):
        assert self._available("manual", "") is False
        assert self._available("manual", "llava:13b") is True

    def test_on_index_requires_vision_model(self):
        assert self._available("on_index", "") is False
        assert self._available("on_index", "llava:13b") is True

    def test_empty_vision_model_equivalent_to_missing(self):
        # Whitespace-only value should behave like "" (operator error).
        assert self._available("manual", "   ") is False
