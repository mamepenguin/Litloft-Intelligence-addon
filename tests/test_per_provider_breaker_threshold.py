"""Per-provider ``circuit_breaker_threshold`` override.

Spec ``2026-05-07-cloud-transcription-providers.md`` R2-4 requires
that operators can tune the rate-limit circuit breaker on a per-
provider basis: a tight quota on Deepgram should not force a 20-
failure window on ElevenLabs Scribe (and vice versa).

The retry wrapper looks up
``settings.transcription.<provider>.circuit_breaker_threshold`` on
first use and constructs a per-provider
``ProviderCircuitBreaker(threshold=...)``. ``None`` falls back to the
spec default (20).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import app.config as config
from app.workers.transcription import errors
from app.workers.transcription.circuit_breaker import ProviderCircuitBreaker
from app.workers.transcription.retry import (
    CircuitBreakerOpen,
    get_breaker_for,
    reset_circuit_breaker,
    transcribe_with_retry,
)


@pytest.fixture(autouse=True)
def _reset_breaker_registry():
    reset_circuit_breaker()
    yield
    reset_circuit_breaker()


class _AlwaysRateLimited:
    """Stub provider that always raises ``RateLimitError``."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def transcribe(self, *args, **kwargs):
        raise errors.RateLimitError("simulated 429")


def _patch_threshold(monkeypatch, provider: str, threshold: int | None) -> None:
    """Inject a circuit_breaker_threshold override on a provider config."""
    base = config.settings.transcription
    sub = getattr(base, provider)
    new_sub = replace(sub, circuit_breaker_threshold=threshold)
    new_transcription = replace(base, **{provider: new_sub})
    new_settings = replace(config.settings, transcription=new_transcription)
    monkeypatch.setattr(config, "settings", new_settings)


def test_get_breaker_for_uses_default_when_no_override(monkeypatch) -> None:
    _patch_threshold(monkeypatch, "deepgram", None)
    breaker = get_breaker_for("deepgram")
    assert isinstance(breaker, ProviderCircuitBreaker)
    # Spec default is 20.
    assert breaker.threshold == 20


def test_get_breaker_for_honours_per_provider_override(monkeypatch) -> None:
    _patch_threshold(monkeypatch, "deepgram", 5)
    breaker = get_breaker_for("deepgram")
    assert breaker.threshold == 5


def test_per_provider_thresholds_isolated(monkeypatch) -> None:
    """Setting threshold on Deepgram must not change ElevenLabs's threshold."""
    _patch_threshold(monkeypatch, "deepgram", 5)
    _patch_threshold(monkeypatch, "elevenlabs_scribe", None)

    dg = get_breaker_for("deepgram")
    el = get_breaker_for("elevenlabs_scribe")

    assert dg.threshold == 5
    assert el.threshold == 20


def test_invalid_threshold_falls_back_to_default(monkeypatch) -> None:
    """Zero / negative thresholds are ignored (would trip on first failure)."""
    _patch_threshold(monkeypatch, "deepgram", 0)
    breaker = get_breaker_for("deepgram")
    assert breaker.threshold == 20


@pytest.mark.asyncio
async def test_breaker_opens_at_configured_threshold_not_default(
    monkeypatch,
) -> None:
    """End-to-end: threshold=2 opens the breaker on the 3rd failure, not 21st.

    With the default threshold (20) plus the 3-attempt retry budget,
    a single ``transcribe_with_retry`` call would record 3 failures
    and stay closed. With threshold=2, the 3rd failure inside the
    same call must trip the breaker — surfacing as
    ``CircuitBreakerOpen`` instead of ``RateLimitError``.
    """
    _patch_threshold(monkeypatch, "deepgram", 2)
    provider = _AlwaysRateLimited("deepgram")

    async def _no_sleep(_d):  # accelerate: skip backoff
        return None

    with pytest.raises((CircuitBreakerOpen, errors.RateLimitError)) as exc_info:
        await transcribe_with_retry(
            provider, "/dev/null", sleep=_no_sleep
        )

    # The 3rd failure (3 attempts at threshold=2) trips the breaker
    # mid-retry and surfaces as CircuitBreakerOpen.
    assert isinstance(exc_info.value, CircuitBreakerOpen)
    breaker = get_breaker_for("deepgram")
    assert breaker.is_open("deepgram") is True


@pytest.mark.asyncio
async def test_breaker_does_not_open_under_default_threshold(monkeypatch) -> None:
    """Inverse: with threshold=20 default, 3 retries do not trip."""
    _patch_threshold(monkeypatch, "deepgram", None)
    provider = _AlwaysRateLimited("deepgram")

    async def _no_sleep(_d):
        return None

    with pytest.raises(errors.RateLimitError):
        await transcribe_with_retry(
            provider, "/dev/null", sleep=_no_sleep
        )

    breaker = get_breaker_for("deepgram")
    # 3 failures under threshold=20 must not open.
    assert breaker.is_open("deepgram") is False
