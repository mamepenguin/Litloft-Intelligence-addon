"""Transcription provider abstraction.

Phase 1A foundation: only the Protocol, dataclasses, errors and
circuit breaker are exposed here. Concrete provider implementations
(``WhisperLocalProvider``, ``OpenAICompatibleProvider``,
``DeepgramProvider``, ``ElevenLabsScribeProvider``) land in Phase 1B.

``get_provider`` is a stub that raises ``NotImplementedError`` until
Phase 1B wires the factory through to the concrete classes.
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

    Phase 1A returns ``NotImplementedError`` for every name so callers
    can already import the symbol but cannot instantiate providers
    until Phase 1B lands.
    """
    raise NotImplementedError(
        f"Transcription provider {name!r} not implemented yet "
        "(Phase 1B will register concrete provider classes)."
    )
