"""The split threshold must respect our memory ceiling, not just the API's.

``ProviderCapabilities.max_input_bytes`` records what the remote API
accepts. Several upload paths buffer the whole body in memory, so a
provider whose API cap is generous (assemblyai: 5 GB) or absent
(deepgram, elevenlabs_scribe) used to skip ``SplittingTranscriber``
entirely and read the source file whole.

That OOM-killed the container: a 4 GB video queued for transcription
took RSS from 166 MiB to 5.5 GiB in ~15 s, and with ``whisper_parallel``
> 1 several ran at once. The container restarted, re-queued the same
job, and looped.

These tests pin the invariant that made it possible: no provider is
ever handed more bytes in one call than ``MAX_INPUT_MEMORY_BYTES``.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

for _mod in ("google", "google.genai", "assemblyai"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.workers.transcription import (  # noqa: E402
    MAX_INPUT_MEMORY_BYTES,
    _effective_input_cap,
    get_provider,
)
from app.workers.transcription.splitting_transcriber import (  # noqa: E402
    SplittingTranscriber,
)

# Providers constructible without a real SDK/network, mirroring the set
# already exercised in test_provider_capabilities.py.
_PROVIDERS = [
    ("whisper_local", None),
    ("openai_compatible", "OPENAI_API_KEY"),
    ("deepgram", "DEEPGRAM_API_KEY"),
    ("elevenlabs_scribe", "ELEVENLABS_API_KEY"),
    ("assemblyai", "ASSEMBLYAI_API_KEY"),
]


class TestEffectiveInputCap:
    def test_absent_api_cap_falls_back_to_the_memory_ceiling(self):
        """None must not mean "unbounded read"."""
        assert _effective_input_cap(None) == MAX_INPUT_MEMORY_BYTES

    def test_generous_api_cap_is_clamped_to_the_memory_ceiling(self):
        """assemblyai's 5 GB is the regression that bit us."""
        assert (
            _effective_input_cap(5 * 1024 * 1024 * 1024)
            == MAX_INPUT_MEMORY_BYTES
        )

    def test_stricter_api_cap_wins(self):
        """A 25 MB API limit must not be relaxed up to our ceiling."""
        strict = 25 * 1024 * 1024
        assert strict < MAX_INPUT_MEMORY_BYTES
        assert _effective_input_cap(strict) == strict


class TestFactoryAlwaysBoundsInputSize:
    @pytest.mark.parametrize("name,env_key", _PROVIDERS)
    def test_provider_is_wrapped_with_a_bounded_cap(
        self, name, env_key, monkeypatch
    ):
        if env_key:
            monkeypatch.setenv(env_key, "test-key")

        provider = get_provider(name)

        assert isinstance(provider, SplittingTranscriber), (
            f"{name} is not wrapped, so an oversized file would be "
            "passed to the provider whole"
        )
        cap = provider._cap_bytes
        assert cap is not None, f"{name} resolved an unbounded cap"
        assert cap <= MAX_INPUT_MEMORY_BYTES, (
            f"{name} would buffer up to {cap} bytes, above the "
            f"{MAX_INPUT_MEMORY_BYTES}-byte memory ceiling"
        )

    def test_deepgram_specifically_no_longer_resolves_to_none(
        self, monkeypatch
    ):
        """The exact shape of the outage: API cap None, so no wrapper."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
        provider = get_provider("deepgram")

        assert provider.name == "deepgram"
        # The inner provider still reports the API's own limit...
        assert provider._inner.capabilities.max_input_bytes is None
        # ...but what actually governs a single call is bounded.
        assert provider._cap_bytes == MAX_INPUT_MEMORY_BYTES


class TestMemoryCeilingIsConfigurable:
    def test_env_var_overrides_the_default(self):
        """Operators with more RAM can trade memory for fewer calls."""
        import importlib

        import app.workers.transcription as transcription

        os.environ["TRANSCRIPTION_MAX_INPUT_MEMORY_BYTES"] = str(
            128 * 1024 * 1024
        )
        try:
            reloaded = importlib.reload(transcription)
            assert reloaded.MAX_INPUT_MEMORY_BYTES == 128 * 1024 * 1024
        finally:
            del os.environ["TRANSCRIPTION_MAX_INPUT_MEMORY_BYTES"]
            importlib.reload(transcription)
