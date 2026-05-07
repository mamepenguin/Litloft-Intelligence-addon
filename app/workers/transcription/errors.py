"""Transcription error hierarchy shared by every provider.

Concrete provider implementations classify the upstream HTTP / network
failure into one of these three kinds. The retry / circuit-breaker
machinery in ``circuit_breaker.py`` and the eventual
``transcribe_with_retry`` wrapper rely on this taxonomy:

* :class:`TransientError` — 5xx, connection refused, generic timeouts.
  Worth retrying with exponential backoff.
* :class:`RateLimitError` — 429 Too Many Requests. Worth retrying with
  backoff, but also feeds the circuit breaker so a wholesale-rate-
  limited provider gets paused instead of compounding the queue.
* :class:`FatalError` — 4xx that will not improve on retry: bad
  credentials, unsupported audio format, payload too large, etc.

The base class :class:`TranscriptionError` is intentionally instantiable
so callers can ``except TranscriptionError`` to catch every flavour at
once (e.g. for ``JobRecord.error_class`` recording).
"""


class TranscriptionError(Exception):
    """Base class for every transcription-provider failure."""


class TransientError(TranscriptionError):
    """Retryable failure: network blip, 5xx, connection timeout."""


class RateLimitError(TranscriptionError):
    """Provider responded 429 / equivalent quota signal."""


class FatalError(TranscriptionError):
    """Permanent failure: bad credentials, unsupported file, etc.

    These must NOT be retried — retrying compounds the JobRecord with
    duplicate failures and can rack up real money on metered providers.
    """
