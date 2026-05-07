"""Provider Protocol + dataclasses for transcription.

Phase 1A foundation. The Protocol intentionally captures only the
contract that all providers must satisfy (cloud or local); concrete
implementations land in Phase 1B.

Design notes:

* All dataclasses are ``frozen=True`` so call sites can hand them
  around without worrying about mutation. Word/segment lists are still
  mutable Python lists but their contents are immutable.
* ``progress`` is included from the start so streaming providers in
  Phase 2 do not require a Protocol expansion (Protocol changes are a
  breaking surface).
* ``ProviderCapabilities`` is a dataclass (not flags / enums) so
  feature negotiation reads naturally at the call site
  (``provider.capabilities.sends_audio_offhost``).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class WordToken:
    """One word-level timestamp returned by a provider.

    ``speaker_id`` is provider-supplied (e.g. Deepgram diarization);
    providers that do not support diarization always set it to ``None``
    and the chunker treats ``None`` as "no speaker change boundary".
    """

    text: str
    start: float
    end: float
    speaker_id: str | None = None


@dataclass(frozen=True)
class TranscriptionSegment:
    """One segment (sentence-ish span) returned by a provider.

    ``language`` is the provider's detected language code; consumers
    should not assume a specific ISO format because providers vary
    (Whisper "ja" vs Deepgram "ja-JP" vs ElevenLabs "japanese").
    Normalisation, when needed, is the caller's responsibility.
    """

    text: str
    start: float
    end: float
    language: str
    words: list[WordToken]


@dataclass(frozen=True)
class ProviderCapabilities:
    """Static capability declaration for a provider.

    The ``sends_audio_offhost`` flag drives the per-drive cloud policy
    gate (``transcription_cloud``). Providers that run entirely on the
    Litloft host (``whisper_local``) set it to ``False``; providers
    that POST audio to a third-party API (``deepgram``,
    ``elevenlabs_scribe``, ``openai_compatible``) set it to ``True``.

    ``supports_word_timestamps`` is a hard requirement of the indexer
    pipeline (subtitles + word seek). A provider declaring ``False``
    here must fail at startup, not silently produce empty word lists.
    """

    sends_audio_offhost: bool
    supports_diarization: bool
    supports_hotwords: bool
    supports_word_timestamps: bool


class TranscriptionProvider(Protocol):
    """Common interface for every transcription backend.

    Implementations live in sibling modules
    (``transcription/whisper_local.py`` etc.) and are wired through
    :func:`get_provider`. The ``progress`` callback is optional in
    Phase 1 — providers may ignore it; it exists in the Protocol so
    Phase 2 streaming backends can plug in without an interface change.
    """

    name: str
    capabilities: ProviderCapabilities

    async def transcribe(
        self,
        file_path: str,
        *,
        language_hint: str | None = None,
        hotwords: list[str] | None = None,
        progress: Callable[[float], None] | None = None,
    ) -> list[TranscriptionSegment]:
        """Transcribe ``file_path`` and return per-segment results."""
        ...
