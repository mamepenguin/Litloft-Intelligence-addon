"""Transcription provider abstraction.

Phase 1B: the four concrete provider classes (``WhisperLocalProvider``,
``OpenAICompatibleProvider``, ``DeepgramProvider``,
``ElevenLabsScribeProvider``) are wired through :func:`get_provider`.
Imports inside the factory are lazy so a misconfigured cloud provider
(missing API key, missing dependency) only fails when that name is
actually requested — startup of the intelligence container does not
load every cloud SDK regardless of which one the user picked.
"""

from app.workers.transcription.base import (
    ProviderCapabilities,
    TranscriptionProvider,
    TranscriptionSegment,
    WordToken,
)
from app.workers.transcription.errors import (
    FatalError,
    RateLimitError,
    TranscriptionError,
    TransientError,
)

__all__ = [
    "FatalError",
    "ProviderCapabilities",
    "RateLimitError",
    "TranscriptionError",
    "TranscriptionProvider",
    "TranscriptionSegment",
    "TransientError",
    "WordToken",
    "get_provider",
]


def get_provider(name: str) -> TranscriptionProvider:
    """Look up a registered transcription provider by name.

    Lazy imports are intentional: importing :mod:`whisper_local` would
    eagerly pull in faster-whisper / ctranslate2 even on a cloud-only
    deployment, and importing :mod:`openai_compatible` instantiates
    the OpenAI SDK at the top level. The match arms only pay each
    cost when the user has actually selected that provider.
    """
    match name:
        case "whisper_local":
            from app.workers.transcription.whisper_local import (
                WhisperLocalProvider,
            )
            return WhisperLocalProvider()
        case "openai_compatible":
            from app.workers.transcription.openai_compatible import (
                OpenAICompatibleProvider,
            )
            return OpenAICompatibleProvider()
        case "deepgram":
            from app.workers.transcription.deepgram import DeepgramProvider
            return DeepgramProvider()
        case "elevenlabs_scribe":
            from app.workers.transcription.elevenlabs_scribe import (
                ElevenLabsScribeProvider,
            )
            return ElevenLabsScribeProvider()
        case _:
            raise ValueError(f"Unknown transcription provider: {name}")
