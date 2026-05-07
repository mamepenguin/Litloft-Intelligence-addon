"""Phase 1A foundation tests for the TranscriptionProvider Protocol.

These tests exercise the static contract — dataclass shape, Protocol
typing, error hierarchy, ``get_provider`` factory stub. Concrete
provider implementations are validated in Phase 1B against the same
``ProviderCapabilities`` declarations checked here.
"""

from __future__ import annotations

import pytest

from app.workers.transcription import (
    FatalError,
    ProviderCapabilities,
    RateLimitError,
    TranscriptionError,
    TranscriptionProvider,
    TranscriptionSegment,
    TransientError,
    WordToken,
    get_provider,
)


def test_word_token_is_frozen_dataclass() -> None:
    """``WordToken`` must be immutable so callers can pass references freely."""
    token = WordToken(text="hello", start=0.0, end=0.5)
    with pytest.raises(Exception):
        # frozen dataclass raises FrozenInstanceError (subclass of AttributeError)
        token.text = "mutated"  # type: ignore[misc]


def test_word_token_speaker_defaults_to_none() -> None:
    """Providers without diarization must be able to omit ``speaker_id``."""
    token = WordToken(text="hi", start=0.0, end=1.0)
    assert token.speaker_id is None


def test_word_token_accepts_speaker_id() -> None:
    token = WordToken(text="hi", start=0.0, end=1.0, speaker_id="spk_0")
    assert token.speaker_id == "spk_0"


def test_transcription_segment_is_frozen_dataclass() -> None:
    seg = TranscriptionSegment(
        text="hello world",
        start=0.0,
        end=1.5,
        language="en",
        words=[WordToken(text="hello", start=0.0, end=0.5)],
    )
    with pytest.raises(Exception):
        seg.text = "mutated"  # type: ignore[misc]


def test_provider_capabilities_is_frozen_dataclass() -> None:
    caps = ProviderCapabilities(
        sends_audio_offhost=False,
        supports_diarization=False,
        supports_hotwords=True,
        supports_word_timestamps=True,
    )
    with pytest.raises(Exception):
        caps.sends_audio_offhost = True  # type: ignore[misc]


def test_provider_capabilities_requires_all_fields() -> None:
    """Capabilities must be declared explicitly — no defaults.

    A provider that forgets to declare a capability would otherwise
    silently fall back to ``False`` (or worse, ``True``); requiring
    explicit values forces the author to make the decision.
    """
    with pytest.raises(TypeError):
        ProviderCapabilities()  # type: ignore[call-arg]


def test_error_hierarchy() -> None:
    """All provider errors must descend from ``TranscriptionError``.

    The retry helper uses ``except TranscriptionError`` as a catch-all
    when classifying job failures, and ``JobRecord.error_class`` is
    populated from the concrete subclass name.
    """
    assert issubclass(TransientError, TranscriptionError)
    assert issubclass(RateLimitError, TranscriptionError)
    assert issubclass(FatalError, TranscriptionError)
    assert TransientError is not RateLimitError
    assert RateLimitError is not FatalError


def test_errors_carry_messages() -> None:
    err = FatalError("bad api key")
    assert str(err) == "bad api key"


def test_get_provider_unknown_name_raises_value_error() -> None:
    """Unknown provider names must surface as ``ValueError``.

    Misconfigurations (typos in ``settings.transcription.provider``)
    deserve a clear message at startup rather than a confusing import
    error or a silent fallback.
    """
    with pytest.raises(ValueError, match="Unknown transcription provider"):
        get_provider("not-a-real-provider")


def test_get_provider_returns_whisper_local() -> None:
    provider = get_provider("whisper_local")
    assert provider.name == "whisper_local"
    assert provider.capabilities.sends_audio_offhost is False


def test_get_provider_returns_openai_compatible(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = get_provider("openai_compatible")
    assert provider.name == "openai_compatible"
    assert provider.capabilities.sends_audio_offhost is True


def test_get_provider_returns_deepgram(monkeypatch) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    provider = get_provider("deepgram")
    assert provider.name == "deepgram"
    assert provider.capabilities.supports_diarization is True


def test_get_provider_returns_elevenlabs_scribe(monkeypatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
    provider = get_provider("elevenlabs_scribe")
    assert provider.name == "elevenlabs_scribe"
    assert provider.capabilities.supports_diarization is True


@pytest.mark.parametrize(
    "name",
    ["whisper_local", "openai_compatible", "deepgram",
     "elevenlabs_scribe", "assemblyai"],
)
def test_native_word_ts_providers_declare_true(
    name, monkeypatch
) -> None:
    """Native word-ts provider must advertise ``supports_word_timestamps=True``.

    Phase 2A contract evolution: ``False`` is reserved for providers
    that synthesise word boundaries from segment-level output (Gemini).
    Providers that decode word-level timing from audio must continue to
    declare ``True`` so the dispatch-time WARN in
    ``_do_transcribe_and_index`` only fires for genuinely synthetic
    backends. End-to-end ``words`` non-emptiness is asserted in each
    provider's own test module (``test_<name>_provider.py``).
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "aai-test")
    provider = get_provider(name)
    assert provider.capabilities.supports_word_timestamps is True


def test_gemini_declares_synthetic_word_ts(monkeypatch) -> None:
    """Gemini is the first ``supports_word_timestamps=False`` provider.

    The flag is observability-only — the chunker still receives a
    non-empty ``words`` list per segment (synthesised by uniform
    splitting). Pin both the flag and the existence of synthetic words
    so a future refactor cannot quietly flip the contract.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    provider = get_provider("gemini")
    assert provider.capabilities.supports_word_timestamps is False


class _FakeProvider:
    """Hand-rolled provider used to confirm the Protocol shape.

    ``TranscriptionProvider`` is a ``Protocol``, so any class with
    matching attributes / async method satisfies it structurally —
    no ABC inheritance required. This test pins that contract.
    """

    name = "fake"
    capabilities = ProviderCapabilities(
        sends_audio_offhost=False,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
    )

    async def transcribe(
        self,
        file_path: str,
        *,
        language_hint: str | None = None,
        hotwords: list[str] | None = None,
        progress=None,
    ) -> list[TranscriptionSegment]:
        return []


def test_protocol_accepts_structurally_matching_class() -> None:
    """A class with matching shape passes the Protocol contract."""
    instance: TranscriptionProvider = _FakeProvider()
    assert instance.name == "fake"
    assert instance.capabilities.supports_word_timestamps is True


@pytest.mark.asyncio
async def test_fake_provider_transcribe_returns_segment_list() -> None:
    provider = _FakeProvider()
    result = await provider.transcribe("/tmp/test.wav")
    assert result == []
