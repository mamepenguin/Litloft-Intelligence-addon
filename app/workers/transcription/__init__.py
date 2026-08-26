"""Transcription provider abstraction.

Phase 1B: the four concrete provider classes (``WhisperLocalProvider``,
``OpenAICompatibleProvider``, ``DeepgramProvider``,
``ElevenLabsScribeProvider``) are wired through :func:`get_provider`.
Imports inside the factory are lazy so a misconfigured cloud provider
(missing API key, missing dependency) only fails when that name is
actually requested — startup of the intelligence container does not
load every cloud SDK regardless of which one the user picked.
"""

import os

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

# Ceiling on how many bytes of audio we hand a provider in one call.
#
# ``ProviderCapabilities.max_input_bytes`` records what the *remote
# API* accepts, which for several providers is far more than this
# process can hold: the upload paths buffer the whole body in memory
# (``deepgram`` and ``assemblyai`` read it into ``bytes`` outright,
# and httpx materialises multipart bodies), and ``whisper_parallel``
# runs several of those at once. A 4 GB source file therefore OOM-
# killed the container even though Deepgram would have accepted it.
#
# Capping here routes oversized inputs through ``SplittingTranscriber``,
# which normalises to 16 kHz mono FLAC on disk with ffmpeg and slices
# on silence, so peak RSS is bounded by the chunk size rather than by
# the source file. At the FLAC rate (~32 kB/s) the default is roughly
# a 27-minute chunk.
#
# Raise it on a host with memory to spare (fewer, longer chunks means
# fewer API round-trips); lower it if transcription still crowds the
# box.
MAX_INPUT_MEMORY_BYTES = int(
    os.environ.get("TRANSCRIPTION_MAX_INPUT_MEMORY_BYTES", 64 * 1024 * 1024)
)

__all__ = [
    "FatalError",
    "MAX_INPUT_MEMORY_BYTES",
    "ProviderCapabilities",
    "RateLimitError",
    "TranscriptionError",
    "TranscriptionProvider",
    "TranscriptionSegment",
    "TransientError",
    "WordToken",
    "build_inner_provider",
    "get_provider",
]


def build_inner_provider(name: str) -> "TranscriptionProvider":
    """Public alias for :func:`_build_inner` — needed by Phase 2C.

    The eval harness reaches past the auto-wrap logic to build the
    raw provider and apply a temporary capability override (forcing a
    custom ``max_input_bytes`` for split-test runs). Exposing the
    inner builder lets evaluators do this without re-implementing the
    factory's match arms.
    """
    return _build_inner(name)


def get_provider(name: str) -> TranscriptionProvider:
    """Look up a registered transcription provider by name.

    Lazy imports are intentional: importing :mod:`whisper_local` would
    eagerly pull in faster-whisper / ctranslate2 even on a cloud-only
    deployment, and importing :mod:`openai_compatible` instantiates
    the OpenAI SDK at the top level. The match arms only pay each
    cost when the user has actually selected that provider.

    Phase 2B: the resolved provider is wrapped in
    :class:`SplittingTranscriber` so long-form inputs are split via
    ffmpeg before delegation. The splitter handles per-chunk retry
    internally and advertises ``handles_own_retry=True`` so the
    dispatch layer skips its outer ``transcribe_with_retry`` wrap.

    The split threshold is the stricter of the provider's own
    ``max_input_bytes`` (what the remote API accepts) and
    :data:`MAX_INPUT_MEMORY_BYTES` (what we are willing to hold in
    memory). Taking the minimum matters: providers whose API cap is
    generous or absent used to skip the wrapper entirely and buffer
    the whole source file, which OOM-killed the container on
    multi-gigabyte inputs.
    """
    inner = _build_inner(name)
    cap = _effective_input_cap(inner.capabilities.max_input_bytes)
    from app.workers.transcription.splitting_transcriber import (
        SplittingTranscriber,
    )
    return SplittingTranscriber(inner, cap_bytes=cap)


def _effective_input_cap(api_cap: int | None) -> int:
    """Resolve the split threshold for a provider.

    ``api_cap`` is the remote API's limit, or None when it has no
    documented one. The memory ceiling always applies, so an absent
    API cap is not an unbounded read.
    """
    if api_cap is None:
        return MAX_INPUT_MEMORY_BYTES
    return min(api_cap, MAX_INPUT_MEMORY_BYTES)


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
