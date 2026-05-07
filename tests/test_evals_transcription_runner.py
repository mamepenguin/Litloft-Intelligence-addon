"""Tests for the runner orchestration logic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.evals_transcription.loader import Case, SplitTest
from app.evals_transcription.runner import (
    ALL_PROVIDERS,
    _ProviderInstance,
    resolve_providers,
    run_eval,
)
from app.evals_transcription.metrics import SpeakerSegment
from app.workers.transcription.base import (
    ProviderCapabilities,
    TranscriptionSegment,
    WordToken,
)
from app.workers.transcription.errors import FatalError, TransientError


def _segment(text: str, start: float = 0.0, end: float = 1.0):
    return TranscriptionSegment(
        text=text,
        start=start,
        end=end,
        language="en",
        words=[WordToken(text=text, start=start, end=end)],
    )


def _case(
    name: str,
    *,
    language: str = "en",
    duration_s: float = 5.0,
    speakers=(),
    split_test: SplitTest | None = None,
) -> Case:
    return Case(
        name=name,
        case_path=f"/tmp/{name}.yml",
        audio_path=f"/tmp/audio/{name}.wav",
        language=language,
        duration_s=duration_s,
        tier="short",
        reference_transcript="hello world",
        speakers=speakers,
        split_test=split_test,
    )


# ---------------------------------------------------------------------------
# resolve_providers
# ---------------------------------------------------------------------------


def test_resolve_providers_unknown_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown providers"):
        resolve_providers(["totally_made_up"])


def test_resolve_providers_skips_when_get_provider_raises_fatal() -> None:
    """API-key-less provider's __init__ raises FatalError → skip row,
    other providers still resolved."""
    def fake_get_provider(name):
        if name == "deepgram":
            raise FatalError("DEEPGRAM_API_KEY not configured")
        mock = MagicMock()
        mock.name = name
        return mock

    with patch(
        "app.evals_transcription.runner.get_provider",
        side_effect=fake_get_provider,
    ):
        result = resolve_providers(["whisper_local", "deepgram"])

    assert result[0].name == "whisper_local"
    assert result[0].provider is not None
    assert result[1].name == "deepgram"
    assert result[1].provider is None
    assert "DEEPGRAM_API_KEY" in (result[1].skipped_reason or "")


# ---------------------------------------------------------------------------
# run_eval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_eval_emits_skipped_row_for_missing_provider() -> None:
    case = _case("c1")
    pi = _ProviderInstance(
        name="deepgram",
        provider=None,
        skipped_reason="DEEPGRAM_API_KEY not configured",
    )
    results = await run_eval([case], [pi])
    assert len(results) == 1
    r = results[0]
    assert r.skipped is True
    assert r.skipped_reason == "DEEPGRAM_API_KEY not configured"
    assert r.case_name == "c1"
    assert r.provider_name == "deepgram"


@pytest.mark.asyncio
async def test_run_eval_scores_default_mode_when_provider_succeeds() -> None:
    case = _case("c1", language="en")
    provider = MagicMock()
    provider.name = "openai_compatible"
    provider.transcribe = AsyncMock(return_value=[_segment("hello world")])

    pi = _ProviderInstance(
        name="openai_compatible", provider=provider, skipped_reason=None
    )
    results = await run_eval([case], [pi])
    assert len(results) == 1
    r = results[0]
    assert r.skipped is False
    assert r.error is None
    assert r.wer == 0.0
    assert r.cer == 0.0
    assert r.latency_s is not None and r.latency_s >= 0


@pytest.mark.asyncio
async def test_run_eval_records_provider_error_without_aborting() -> None:
    """One provider raising TranscriptionError must not stop other
    case×provider runs."""
    case_a = _case("a")
    case_b = _case("b")
    failing = MagicMock()
    failing.name = "openai_compatible"
    failing.transcribe = AsyncMock(side_effect=FatalError("401"))

    pi = _ProviderInstance(name="openai_compatible", provider=failing)
    results = await run_eval([case_a, case_b], [pi])
    assert len(results) == 2
    assert all(r.error == "FatalError" for r in results)
    assert all(r.skipped is False for r in results)


@pytest.mark.asyncio
async def test_run_eval_handles_unsupported_language_as_error() -> None:
    """ValueError from normalize() (unsupported language) must be
    converted into a CaseResult with `error` set, not propagated."""
    case = _case("c1", language="es")
    provider = MagicMock()
    provider.name = "openai_compatible"
    provider.transcribe = AsyncMock(return_value=[_segment("hola")])

    pi = _ProviderInstance(name="openai_compatible", provider=provider)
    results = await run_eval([case], [pi])
    assert len(results) == 1
    assert results[0].error and "ValueError" in (
        results[0].error or ""
    ) or "Unsupported" in (results[0].error or "")


@pytest.mark.asyncio
async def test_run_eval_split_test_runs_no_split_and_split_modes() -> None:
    """When a case has a split_test block, the runner produces three
    rows for the matching provider: default, no_split, split."""
    case = _case(
        "long",
        split_test=SplitTest(
            forced_cap_bytes=10,
            providers=("openai_compatible",),
        ),
    )
    inner = MagicMock()
    inner.name = "openai_compatible"
    inner.capabilities = ProviderCapabilities(
        sends_audio_offhost=True,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
        max_input_bytes=25 * 1024 * 1024,
        accepts_initial_prompt=True,
        handles_own_retry=False,
    )
    inner.transcribe = AsyncMock(return_value=[_segment("hello world")])

    # The default-mode resolve uses get_provider (which would normally
    # wrap inner in SplittingTranscriber); the build_inner_provider
    # call inside _split_test_runs returns our raw inner.
    provider = MagicMock()
    provider.name = "openai_compatible"
    provider.capabilities = inner.capabilities
    provider.transcribe = AsyncMock(return_value=[_segment("hello world")])

    pi = _ProviderInstance(
        name="openai_compatible", provider=provider
    )

    with patch(
        "app.evals_transcription.runner.build_inner_provider",
        return_value=inner,
    ), patch(
        "app.evals_transcription.runner.SplittingTranscriber"
    ) as splitting_mock:
        splitting_instance = MagicMock()
        splitting_instance.name = "openai_compatible"
        splitting_instance.transcribe = AsyncMock(
            return_value=[_segment("hello world")]
        )
        splitting_mock.return_value = splitting_instance

        results = await run_eval([case], [pi])

    modes = [r.mode for r in results]
    assert "default" in modes
    assert "no_split" in modes
    assert "split" in modes
    # The split_test is silently skipped only when the forced cap >=
    # inner cap; here 10 < 25 MB so split fires.
    splitting_mock.assert_called_once()
