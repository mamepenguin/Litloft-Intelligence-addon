"""Tests for :class:`ElevenLabsScribeProvider`.

Covered surface:

* Capabilities (cloud, diarization=True, word ts)
* Missing ``ELEVENLABS_API_KEY`` env → ``FatalError``
* Wire shape: POST /v1/speech-to-text, ``xi-api-key`` header,
  multipart body with ``file``, ``model_id``, ``diarize``,
  ``timestamps_granularity``
* Parity: response → list[TranscriptionSegment] with speaker_id
  forwarded verbatim (already a string, e.g. "speaker_0")
* Empty words → empty list (silent audio)
* HTTP error mapping (5xx/timeout = transient, 429 = rate limit, 4xx = fatal)
* Non-word entries (``type != "word"``) are skipped — Scribe also
  emits ``"spacing"`` / ``"audio_event"`` rows that are not real words
"""

from __future__ import annotations

import httpx
import pytest

from app.workers.transcription import (
    FatalError,
    ProviderCapabilities,
    RateLimitError,
    TranscriptionSegment,
    TransientError,
    WordToken,
)
from app.workers.transcription.elevenlabs_scribe import (
    ElevenLabsScribeProvider,
)


@pytest.fixture()
def with_api_key(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test-fake")
    yield


@pytest.fixture()
def fake_audio_file(tmp_path):
    p = tmp_path / "clip.wav"
    p.write_bytes(b"\x00" * 1024)
    return str(p)


def _scribe_response(words: list[dict] | None = None) -> dict:
    if words is None:
        words = [
            {
                "text": "Hello",
                "type": "word",
                "start": 0.10,
                "end": 0.50,
                "speaker_id": "speaker_0",
            },
            {
                "text": "world",
                "type": "word",
                "start": 0.50,
                "end": 1.50,
                "speaker_id": "speaker_0",
            },
        ]
    return {
        "language_code": "en",
        "language_probability": 0.95,
        "text": "Hello world",
        "words": words,
    }


def _make_provider_with_transport(
    transport: httpx.MockTransport,
) -> ElevenLabsScribeProvider:
    provider = ElevenLabsScribeProvider()
    provider._client = httpx.AsyncClient(transport=transport, timeout=10.0)
    return provider


def test_provider_declared_name() -> None:
    assert ElevenLabsScribeProvider.name == "elevenlabs_scribe"


def test_provider_capabilities_match_spec() -> None:
    assert ElevenLabsScribeProvider.capabilities == ProviderCapabilities(
        sends_audio_offhost=True,
        supports_diarization=True,
        supports_hotwords=False,
        supports_word_timestamps=True,
    )


def test_init_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(FatalError, match="ELEVENLABS_API_KEY"):
        ElevenLabsScribeProvider()


def test_init_with_empty_api_key_is_fatal(monkeypatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "")
    with pytest.raises(FatalError, match="ELEVENLABS_API_KEY"):
        ElevenLabsScribeProvider()


@pytest.mark.asyncio
async def test_transcribe_posts_with_xi_api_key_header_and_multipart(
    with_api_key, fake_audio_file
) -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["content"] = request.content  # multipart-encoded bytes
        return httpx.Response(200, json=_scribe_response())

    provider = _make_provider_with_transport(httpx.MockTransport(handler))
    await provider.transcribe(fake_audio_file)

    assert seen["method"] == "POST"
    assert "https://api.elevenlabs.io/v1/speech-to-text" in seen["url"]
    assert seen["headers"]["xi-api-key"] == "el-test-fake"
    # Multipart body must contain the configured fields.
    body = seen["content"]
    assert b"name=\"model_id\"" in body
    assert b"scribe_v1" in body  # default model id
    assert b"name=\"diarize\"" in body
    assert b"name=\"timestamps_granularity\"" in body
    assert b"word" in body  # granularity value
    assert b"name=\"file\"" in body


@pytest.mark.asyncio
async def test_transcribe_converts_words_to_segment(
    with_api_key, fake_audio_file
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_scribe_response())

    provider = _make_provider_with_transport(httpx.MockTransport(handler))
    result = await provider.transcribe(fake_audio_file)

    assert len(result) == 1
    seg = result[0]
    assert isinstance(seg, TranscriptionSegment)
    assert seg.language == "en"
    assert seg.words == [
        WordToken(text="Hello", start=0.10, end=0.50, speaker_id="speaker_0"),
        WordToken(text="world", start=0.50, end=1.50, speaker_id="speaker_0"),
    ]


@pytest.mark.asyncio
async def test_non_word_types_are_skipped(with_api_key, fake_audio_file) -> None:
    """Scribe interleaves ``type=spacing`` rows; provider must skip them."""
    body = _scribe_response(
        words=[
            {
                "text": "Hi",
                "type": "word",
                "start": 0.0,
                "end": 0.4,
                "speaker_id": "speaker_0",
            },
            {
                "text": " ",
                "type": "spacing",
                "start": 0.4,
                "end": 0.5,
            },
            {
                "text": "[applause]",
                "type": "audio_event",
                "start": 0.5,
                "end": 1.0,
            },
            {
                "text": "there",
                "type": "word",
                "start": 1.0,
                "end": 1.5,
                "speaker_id": "speaker_0",
            },
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    provider = _make_provider_with_transport(httpx.MockTransport(handler))
    result = await provider.transcribe(fake_audio_file)

    assert [w.text for w in result[0].words] == ["Hi", "there"]


@pytest.mark.asyncio
async def test_speaker_none_passes_through(with_api_key, fake_audio_file) -> None:
    body = _scribe_response(
        words=[
            {"text": "hi", "type": "word", "start": 0.0, "end": 0.5}
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    provider = _make_provider_with_transport(httpx.MockTransport(handler))
    result = await provider.transcribe(fake_audio_file)

    assert result[0].words[0].speaker_id is None


@pytest.mark.asyncio
async def test_empty_words_returns_empty_list(
    with_api_key, fake_audio_file
) -> None:
    body = _scribe_response(words=[])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    provider = _make_provider_with_transport(httpx.MockTransport(handler))
    assert await provider.transcribe(fake_audio_file) == []


@pytest.mark.asyncio
async def test_429_maps_to_rate_limit_error(
    with_api_key, fake_audio_file
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"err": "rate"})

    provider = _make_provider_with_transport(httpx.MockTransport(handler))
    with pytest.raises(RateLimitError):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [400, 401, 403, 413, 422])
async def test_4xx_maps_to_fatal(with_api_key, fake_audio_file, code) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, json={"err": "bad"})

    provider = _make_provider_with_transport(httpx.MockTransport(handler))
    with pytest.raises(FatalError):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [500, 502, 503])
async def test_5xx_maps_to_transient(with_api_key, fake_audio_file, code) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, json={"err": "boom"})

    provider = _make_provider_with_transport(httpx.MockTransport(handler))
    with pytest.raises(TransientError):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_timeout_maps_to_transient(with_api_key, fake_audio_file) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    provider = _make_provider_with_transport(httpx.MockTransport(handler))
    with pytest.raises(TransientError):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_connect_error_maps_to_transient(
    with_api_key, fake_audio_file
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    provider = _make_provider_with_transport(httpx.MockTransport(handler))
    with pytest.raises(TransientError):
        await provider.transcribe(fake_audio_file)
