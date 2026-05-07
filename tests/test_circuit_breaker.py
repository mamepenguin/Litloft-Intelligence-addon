"""Phase 1A foundation tests for the per-provider circuit breaker.

The circuit breaker pauses cloud transcription when a provider is
wholesale rate-limited (consistent 429 stream). It is *per-provider*
because Deepgram being down should not stop ElevenLabs Scribe from
serving a different drive.

Defaults follow the spec ("20/60s threshold, 60s pause") — both are
constructor args so tests can shrink the window and accelerate.
"""

from __future__ import annotations

import pytest

from app.workers.transcription.circuit_breaker import ProviderCircuitBreaker


def test_breaker_starts_closed() -> None:
    breaker = ProviderCircuitBreaker(threshold=3, window_s=60, pause_s=60)
    assert breaker.is_open("deepgram") is False


def test_breaker_opens_after_threshold_exceeded() -> None:
    """Strictly *more* failures than the threshold open the breaker.

    20/60s in production means "20 failures in a window are tolerated;
    the 21st trips the breaker". This matches the spec wording
    ("20 件超で 60 秒 pause").
    """
    breaker = ProviderCircuitBreaker(threshold=3, window_s=60, pause_s=60)
    for _ in range(3):
        breaker.record_failure("deepgram")
    # At threshold — still closed.
    assert breaker.is_open("deepgram") is False
    breaker.record_failure("deepgram")
    assert breaker.is_open("deepgram") is True


def test_breaker_is_per_provider() -> None:
    """A provider's failures must not trip a different provider's breaker."""
    breaker = ProviderCircuitBreaker(threshold=2, window_s=60, pause_s=60)
    for _ in range(5):
        breaker.record_failure("deepgram")
    assert breaker.is_open("deepgram") is True
    assert breaker.is_open("elevenlabs_scribe") is False


def test_breaker_old_failures_age_out_without_tripping() -> None:
    """Failures outside ``window_s`` no longer count toward the threshold
    *when the breaker has not yet tripped*.

    Uses an injected ``now`` clock so the test is deterministic and
    does not depend on ``time.monotonic`` advancing in real time.
    A tripped breaker holds open for ``pause_s`` regardless — that
    behaviour is covered by ``test_breaker_pause_period_keeps_open_after_open``.
    """
    clock = {"t": 1000.0}
    breaker = ProviderCircuitBreaker(
        threshold=3, window_s=10, pause_s=60, now=lambda: clock["t"]
    )
    # 2 failures — under threshold, so breaker has not tripped.
    breaker.record_failure("deepgram")
    breaker.record_failure("deepgram")
    assert breaker.is_open("deepgram") is False

    # Advance clock past the window; old failures should age out so
    # the rolling count no longer keeps us anywhere near threshold.
    clock["t"] += 11.0
    assert breaker.is_open("deepgram") is False

    # And one new failure after aging is still under threshold.
    breaker.record_failure("deepgram")
    assert breaker.is_open("deepgram") is False


def test_breaker_pause_period_keeps_open_after_open() -> None:
    """Once tripped, the breaker stays open for ``pause_s`` even with
    no further failures, then auto-closes."""
    clock = {"t": 0.0}
    breaker = ProviderCircuitBreaker(
        threshold=1, window_s=60, pause_s=30, now=lambda: clock["t"]
    )
    breaker.record_failure("deepgram")
    breaker.record_failure("deepgram")
    assert breaker.is_open("deepgram") is True

    # Within pause window — still open even though failures are gone
    # from the count window (it's the "pause" guarantee, not the
    # rolling count).
    clock["t"] += 10.0
    assert breaker.is_open("deepgram") is True

    # After pause expires, breaker closes again (no new failures).
    clock["t"] += 100.0
    assert breaker.is_open("deepgram") is False


def test_default_constants_match_spec() -> None:
    """Spec calls for 20 / 60s / 60s defaults; pin them in code so a
    drift in either spec or implementation surfaces in test."""
    breaker = ProviderCircuitBreaker()
    assert breaker.threshold == 20
    assert breaker.window_s == 60
    assert breaker.pause_s == 60


def test_breaker_handles_unknown_provider_gracefully() -> None:
    """Querying a provider that has never failed must not crash."""
    breaker = ProviderCircuitBreaker(threshold=3, window_s=60, pause_s=60)
    assert breaker.is_open("brand-new-provider") is False


@pytest.mark.parametrize("threshold", [1, 5, 20, 100])
def test_threshold_parametric(threshold: int) -> None:
    breaker = ProviderCircuitBreaker(
        threshold=threshold, window_s=60, pause_s=60
    )
    for _ in range(threshold):
        breaker.record_failure("p")
    assert breaker.is_open("p") is False
    breaker.record_failure("p")
    assert breaker.is_open("p") is True
