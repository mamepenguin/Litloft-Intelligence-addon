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

    Phase 2B: when the resolved provider declares a hard
    ``max_input_bytes``, wrap it in :class:`SplittingTranscriber` so
    long-form inputs are split via ffmpeg before delegation. The
    splitter handles per-chunk retry internally and advertises
    ``handles_own_retry=True`` so the dispatch layer skips its outer
    ``transcribe_with_retry`` wrap.
    """
    inner = _build_inner(name)
    if inner.capabilities.max_input_bytes is not None:
        from app.workers.transcription.splitting_transcriber import (
            SplittingTranscriber,
        )
        return SplittingTranscriber(inner)
    return inner


def _build_inner(name: str) -> TranscriptionProvider:
    """Resolve the concrete provider (no Phase 2B wrapping)."""
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
        case "assemblyai":
            from app.workers.transcription.assemblyai import (
                AssemblyAIProvider,
            )
            return AssemblyAIProvider()
        case "gemini":
            from app.workers.transcription.gemini import GeminiProvider
            return GeminiProvider()
        case _:
            raise ValueError(f"Unknown transcription provider: {name}")
