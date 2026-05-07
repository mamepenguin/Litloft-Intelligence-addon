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


def test_get_provider_stub_raises_not_implemented() -> None:
    """Phase 1A foundation: factory exists but cannot resolve providers yet."""
    with pytest.raises(NotImplementedError):
        get_provider("whisper_local")


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
