"""Tests for :class:`OpenAICompatibleProvider`.

Covered surface:

* Capabilities declaration (sends_audio_offhost=True, no diarization)
* Missing ``OPENAI_API_KEY`` env → ``FatalError`` (fail-loud, never silent)
* The 25 MB pre-check fires only when ``base_url`` points at the
  official OpenAI endpoint — Groq / Fireworks / self-hosted endpoints
  must not be blocked
* Word-timestamp parity: response with ``words`` → TranscriptionSegment
  with WordToken list
* Empty-words guard (Groq / Fireworks broken implementations) →
  ``FatalError`` so we never silently lose word seek + subtitle data
* HTTP error mapping: 5xx / network / timeout → TransientError, 429 →
  RateLimitError, 401 / 400 / 413 / 422 → FatalError
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Importing openai exceptions for mock plumbing — same set the LLM
# wrapper classifies against, so we keep the taxonomy consistent.
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
)
from openai import (
    RateLimitError as OpenAIRateLimitError,
)

from app.workers.transcription import (
    FatalError,
    ProviderCapabilities,
    RateLimitError,
    TranscriptionSegment,
    TransientError,
    WordToken,
)
from app.workers.transcription.openai_compatible import (
    OpenAICompatibleProvider,
)


@pytest.fixture()
def with_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    yield


@pytest.fixture()
def fake_audio_file(tmp_path):
    p = tmp_path / "clip.wav"
    p.write_bytes(b"\x00" * 1024)  # 1 KB — well under 25 MB
    return str(p)


def _verbose_response(words: list[dict] | None = None) -> SimpleNamespace:
    """Build a minimal SDK-shaped response (verbose_json + word ts).

    The OpenAI Python SDK returns Pydantic models with ``segments`` and
    ``words`` attributes. SimpleNamespace mimics the attribute-access
    shape closely enough for the provider's parser.
    """
    if words is None:
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.5},
        ]
    word_objs = [SimpleNamespace(**w) for w in words]
    seg = SimpleNamespace(
        text="hello world",
        start=0.0,
        end=1.5,
        words=word_objs,
    )
    return SimpleNamespace(
        text="hello world",
        language="en",
        segments=[seg],
        words=word_objs,
    )


def _patch_client(provider: OpenAICompatibleProvider, response):
    """Install a mocked AsyncOpenAI on a provider instance."""
    provider._client = MagicMock()
    provider._client.audio = MagicMock()
    provider._client.audio.transcriptions = MagicMock()
    provider._client.audio.transcriptions.create = AsyncMock(return_value=response)
    return provider._client


def test_provider_declared_name() -> None:
    assert OpenAICompatibleProvider.name == "openai_compatible"


def test_provider_capabilities_match_spec() -> None:
    assert OpenAICompatibleProvider.capabilities == ProviderCapabilities(
        sends_audio_offhost=True,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
        max_input_bytes=25 * 1024 * 1024,
        accepts_initial_prompt=True,
        handles_own_retry=False,
    )


def test_init_requires_api_key(monkeypatch) -> None:
    """No ``OPENAI_API_KEY`` env → fail-loud at instantiation.

    Per spec we must never silently swap providers; the indexer records
    a JobRecord with FatalError and stops.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(FatalError, match="OPENAI_API_KEY"):
        OpenAICompatibleProvider()


def test_init_with_empty_api_key_is_fatal(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(FatalError, match="OPENAI_API_KEY"):
        OpenAICompatibleProvider()


def test_init_with_api_key_succeeds(with_api_key) -> None:
    provider = OpenAICompatibleProvider()
    assert provider.name == "openai_compatible"


@pytest.mark.asyncio
async def test_transcribe_converts_response_to_segments(
    with_api_key, fake_audio_file
) -> None:
    provider = OpenAICompatibleProvider()
    _patch_client(provider, _verbose_response())

    result = await provider.transcribe(fake_audio_file)

    assert len(result) == 1
    assert isinstance(result[0], TranscriptionSegment)
    assert result[0].text == "hello world"
    assert result[0].start == 0.0
    assert result[0].end == 1.5
    assert result[0].language == "en"
    assert result[0].words == [
        WordToken(text="hello", start=0.0, end=0.5, speaker_id=None),
        WordToken(text="world", start=0.5, end=1.5, speaker_id=None),
    ]


@pytest.mark.asyncio
async def test_transcribe_passes_language_hint(with_api_key, fake_audio_file) -> None:
    provider = OpenAICompatibleProvider()
    client = _patch_client(provider, _verbose_response())

    await provider.transcribe(fake_audio_file, language_hint="ja")

    kwargs = client.audio.transcriptions.create.await_args.kwargs
    assert kwargs.get("language") == "ja"
    assert kwargs.get("response_format") == "verbose_json"
    assert kwargs.get("timestamp_granularities") == ["word"]


@pytest.mark.asyncio
async def test_transcribe_omits_language_when_not_provided(
    with_api_key, fake_audio_file
) -> None:
    provider = OpenAICompatibleProvider()
    client = _patch_client(provider, _verbose_response())

    await provider.transcribe(fake_audio_file)

    kwargs = client.audio.transcriptions.create.await_args.kwargs
    # Either omitted entirely or explicitly None — both signal "auto".
    assert kwargs.get("language") in (None,)


@pytest.mark.asyncio
async def test_25mb_precheck_blocks_official_openai(
    with_api_key, tmp_path
) -> None:
    """File >25 MB targeted at api.openai.com → FatalError before HTTP call."""
    big = tmp_path / "big.wav"
    big.write_bytes(b"\x00" * (26 * 1024 * 1024))

    with patch(
        "app.workers.transcription.openai_compatible.config"
    ) as fake_config:
        fake_config.settings.transcription.openai_compatible.base_url = (
            "https://api.openai.com/v1"
        )
        fake_config.settings.transcription.openai_compatible.model = "whisper-1"
        fake_config.settings.transcription.openai_compatible.timeout_s = 600
        provider = OpenAICompatibleProvider()
        _patch_client(provider, _verbose_response())  # would succeed if reached

        with pytest.raises(FatalError, match="25MB"):
            await provider.transcribe(str(big))


@pytest.mark.asyncio
async def test_25mb_precheck_skips_non_openai_endpoints(
    with_api_key, tmp_path
) -> None:
    """26 MB on Groq endpoint → call proceeds (Groq has no 25 MB cap)."""
    big = tmp_path / "big.wav"
    big.write_bytes(b"\x00" * (26 * 1024 * 1024))

    with patch(
        "app.workers.transcription.openai_compatible.config"
    ) as fake_config:
        fake_config.settings.transcription.openai_compatible.base_url = (
            "https://api.groq.com/openai/v1"
        )
        fake_config.settings.transcription.openai_compatible.model = (
            "whisper-large-v3-turbo"
        )
        fake_config.settings.transcription.openai_compatible.timeout_s = 600
        provider = OpenAICompatibleProvider()
        _patch_client(provider, _verbose_response())

        result = await provider.transcribe(str(big))
        assert len(result) == 1


@pytest.mark.asyncio
async def test_empty_word_timestamps_is_fatal(
    with_api_key, fake_audio_file
) -> None:
    """Provider returned segments but no words anywhere → FatalError.

    Catches Groq / Fireworks endpoints whose
    ``timestamp_granularities=["word"]`` support is broken — silent
    skip would lose subtitle + word-seek data downstream.
    """
    bad = SimpleNamespace(
        text="hello world",
        language="en",
        segments=[SimpleNamespace(text="hi", start=0.0, end=1.0, words=[])],
        words=[],
    )
    provider = OpenAICompatibleProvider()
    _patch_client(provider, bad)

    with pytest.raises(FatalError, match="word timestamps"):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_empty_response_returns_empty_list(
    with_api_key, fake_audio_file
) -> None:
    """No segments at all (silence) is succeeded-with-zero, not an error."""
    empty = SimpleNamespace(text="", language="en", segments=[], words=[])
    provider = OpenAICompatibleProvider()
    _patch_client(provider, empty)

    assert await provider.transcribe(fake_audio_file) == []


def _build_status_error(code: int) -> APIStatusError:
    """Construct an APIStatusError via the SDK constructor surface."""
    response = httpx.Response(
        status_code=code,
        request=httpx.Request("POST", "https://api.openai.com/v1/audio"),
    )
    return APIStatusError(
        message=f"http {code}", response=response, body=None
    )


@pytest.mark.asyncio
async def test_429_maps_to_rate_limit_error(
    with_api_key, fake_audio_file
) -> None:
    response = httpx.Response(
        status_code=429,
        request=httpx.Request("POST", "https://api.openai.com/v1/audio"),
    )
    err = OpenAIRateLimitError(
        message="rate", response=response, body=None
    )
    provider = OpenAICompatibleProvider()
    provider._client = MagicMock()
    provider._client.audio = MagicMock()
    provider._client.audio.transcriptions = MagicMock()
    provider._client.audio.transcriptions.create = AsyncMock(side_effect=err)

    with pytest.raises(RateLimitError):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [400, 401, 413, 422])
async def test_4xx_maps_to_fatal_error(
    with_api_key, fake_audio_file, code
) -> None:
    err = _build_status_error(code)
    provider = OpenAICompatibleProvider()
    provider._client = MagicMock()
    provider._client.audio = MagicMock()
    provider._client.audio.transcriptions = MagicMock()
    provider._client.audio.transcriptions.create = AsyncMock(side_effect=err)

    with pytest.raises(FatalError):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_5xx_maps_to_transient_error(
    with_api_key, fake_audio_file
) -> None:
    response = httpx.Response(
        status_code=503,
        request=httpx.Request("POST", "https://api.openai.com/v1/audio"),
    )
    err = InternalServerError(
        message="boom", response=response, body=None
    )
    provider = OpenAICompatibleProvider()
    provider._client = MagicMock()
    provider._client.audio = MagicMock()
    provider._client.audio.transcriptions = MagicMock()
    provider._client.audio.transcriptions.create = AsyncMock(side_effect=err)

    with pytest.raises(TransientError):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_timeout_maps_to_transient_error(
    with_api_key, fake_audio_file
) -> None:
    err = APITimeoutError(
        request=httpx.Request("POST", "https://api.openai.com/v1/audio"),
    )
    provider = OpenAICompatibleProvider()
    provider._client = MagicMock()
    provider._client.audio = MagicMock()
    provider._client.audio.transcriptions = MagicMock()
    provider._client.audio.transcriptions.create = AsyncMock(side_effect=err)

    with pytest.raises(TransientError):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_connection_error_maps_to_transient_error(
    with_api_key, fake_audio_file
) -> None:
    err = APIConnectionError(
        request=httpx.Request("POST", "https://api.openai.com/v1/audio"),
    )
    provider = OpenAICompatibleProvider()
    provider._client = MagicMock()
    provider._client.audio = MagicMock()
    provider._client.audio.transcriptions = MagicMock()
    provider._client.audio.transcriptions.create = AsyncMock(side_effect=err)

    with pytest.raises(TransientError):
        await provider.transcribe(fake_audio_file)
