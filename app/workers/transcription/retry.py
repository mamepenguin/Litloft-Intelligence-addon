"""Retry + circuit-breaker wrapper around ``provider.transcribe``.

Spec ``2026-05-07-cloud-transcription-providers.md`` §"Retry policy" /
§"429 連続時の circuit breaker". The wrapper:

* attempts ``transcribe`` up to ``MAX_ATTEMPTS`` times with
  exponential backoff on :class:`TransientError` /
  :class:`RateLimitError`
* records each :class:`RateLimitError` against the per-provider
  circuit breaker
* short-circuits with :class:`CircuitBreakerOpen` when the breaker
  is already open at call time, so a wholesale-rate-limited provider
  cannot exhaust queue capacity by getting 429s on every job

:class:`FatalError` propagates immediately — credentials problems and
unsupported file shapes are the operator's fault and retrying just
spends quota on guaranteed failures.

A module-level :class:`ProviderCircuitBreaker` singleton holds state
for the whole intelligence process. Tests can swap it via
:func:`reset_circuit_breaker` (and the ``circuit_breaker`` argument
on :func:`transcribe_with_retry`).
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable

from app.workers.transcription.base import TranscriptionSegment
from app.workers.transcription.circuit_breaker import ProviderCircuitBreaker
from app.workers.transcription.errors import (
    FatalError,
    RateLimitError,
    TranscriptionError,
    TransientError,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
INITIAL_BACKOFF_S = 1.0
BACKOFF_CAP_S = 30.0


class CircuitBreakerOpen(TranscriptionError):
    """Raised when the per-provider circuit breaker is open.

    Distinct subclass so :class:`JobRecord.error_class` can record
    ``"CircuitBreakerOpen"`` without mistaking it for the underlying
    :class:`RateLimitError` that caused it.
    """


# Module-level singleton; reset between tests via reset_circuit_breaker().
_breaker = ProviderCircuitBreaker()


def get_circuit_breaker() -> ProviderCircuitBreaker:
    """Return the process-wide circuit breaker instance."""
    return _breaker


def reset_circuit_breaker() -> None:
    """Reset the process-wide breaker to a clean state (test hook)."""
    _breaker.reset()


async def transcribe_with_retry(
    provider,
    file_path: str,
    *,
    language_hint: str | None = None,
    hotwords: list[str] | None = None,
    progress: Callable[[float], None] | None = None,
    circuit_breaker: ProviderCircuitBreaker | None = None,
    sleep: Callable[[float], "asyncio.Future"] | None = None,
) -> list[TranscriptionSegment]:
    """Call ``provider.transcribe`` with retry and circuit-breaker gating.

    Args:
        provider: Any object satisfying the
            :class:`TranscriptionProvider` Protocol.
        file_path: Audio path; passed through verbatim.
        language_hint / hotwords / progress: Forwarded to the provider.
        circuit_breaker: Override the module singleton; tests pass an
            isolated instance so they don't leak state.
        sleep: Override ``asyncio.sleep`` so backoff tests can run
            instantly.

    Raises:
        CircuitBreakerOpen: Breaker was already open at call time, or
            ``RateLimitError`` ate through the retry budget and tripped
            it.
        FatalError / TransientError / RateLimitError: Final classified
            error after the retry budget is exhausted. Caller writes
            the class name into ``JobRecord.error_class``.
    """
    breaker = circuit_breaker if circuit_breaker is not None else _breaker
    sleep_fn = sleep if sleep is not None else asyncio.sleep

    if breaker.is_open(provider.name):
        raise CircuitBreakerOpen(
            f"Provider {provider.name} circuit breaker is open"
        )

    delay = INITIAL_BACKOFF_S
    last_exc: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return await provider.transcribe(
                file_path,
                language_hint=language_hint,
                hotwords=hotwords,
                progress=progress,
            )
        except RateLimitError as exc:
            last_exc = exc
            breaker.record_failure(provider.name)
            if breaker.is_open(provider.name):
                # 21st failure flipped the breaker — fail fast rather
                # than spending the rest of the retry budget.
                raise CircuitBreakerOpen(
                    f"Provider {provider.name} circuit breaker tripped "
                    f"during retry"
                ) from exc
            if attempt == MAX_ATTEMPTS - 1:
                raise
            await sleep_fn(delay + random.uniform(0, 0.5))
            delay = min(delay * 2, BACKOFF_CAP_S)
        except TransientError as exc:
            last_exc = exc
            if attempt == MAX_ATTEMPTS - 1:
                raise
            await sleep_fn(delay + random.uniform(0, 0.5))
            delay = min(delay * 2, BACKOFF_CAP_S)
        except FatalError:
            # Fatal: do not retry, do not touch breaker.
            raise

    # Defensive: should not be reachable — the loop either returns,
    # raises FatalError, or re-raises on the last iteration.
    if last_exc is not None:
        raise last_exc
    raise TransientError("retry budget exhausted")
