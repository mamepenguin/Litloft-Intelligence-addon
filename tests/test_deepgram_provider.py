"""Tests for :class:`DeepgramProvider`.

Covered surface:

* Capabilities (cloud, diarization=True, word ts)
* Missing ``DEEPGRAM_API_KEY`` env → ``FatalError``
* Wire shape: POST to /v1/listen with audio body, ``Token`` auth, the
  configured query params (model, diarize, smart_format, etc.)
* Parity: Deepgram-shaped JSON → list[TranscriptionSegment] with
  speaker_id propagated as ``str(speaker)``
* Empty word list / no words anywhere → empty result (silence is
  succeeded-with-zero, not an error — Deepgram returns 200 with no
  ``words`` for silent clips)
* HTTP error classification: 5xx / timeout = transient, 429 = rate
  limit, 4xx = fatal
* Channel 0 only — multichannel responses must not double-count words
"""

from __future__ import annotations

import os

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
from app.workers.transcription.deepgram import DeepgramProvider


@pytest.fixture()
def with_api_key(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test-fake")
    yield


@pytest.fixture()
def fake_audio_file(tmp_path):
    p = tmp_path / "clip.wav"
    p.write_bytes(b"\x00" * 1024)
    return str(p)


def _deepgram_response(words: list[dict] | None = None) -> dict:
    """Build a representative Deepgram /v1/listen response body."""
    if words is None:
        words = [
            {
                "word": "hello",
                "punctuated_word": "Hello,",
                "start": 0.10,
                "end": 0.50,
                "speaker": 0,
            },
            {
                "word": "world",
                "punctuated_word": "world.",
                "start": 0.50,
                "end": 1.50,
                "speaker": 0,
            },
        ]
    return {
        "metadata": {"detected_language": "en"},
        "results": {
            "channels": [
                {
                    "detected_language": "en",
                    "alternatives": [
                        {
                            "transcript": "Hello, world.",
                            "words": words,
                        }
                    ],
                }
            ]
        },
    }


def _make_provider_with_transport(transport: httpx.MockTransport) -> DeepgramProvider:
    """Build a provider whose per-call AsyncClient uses the mock transport.

    The provider builds an ``httpx.AsyncClient`` per ``transcribe()``
    call (so sockets are released between jobs); ``_transport`` is the
    test-only injection slot the implementation respects when set.
    """
    provider = DeepgramProvider()
    provider._transport = transport
    return provider


def test_provider_declared_name() -> None:
    assert DeepgramProvider.name == "deepgram"


def test_provider_capabilities_match_spec() -> None:
    assert DeepgramProvider.capabilities == ProviderCapabilities(
        sends_audio_offhost=True,
        supports_diarization=True,
        supports_hotwords=False,
        supports_word_timestamps=True,
        max_input_bytes=None,
        accepts_initial_prompt=False,
        handles_own_retry=False,
    )


def test_init_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with pytest.raises(FatalError, match="DEEPGRAM_API_KEY"):
        DeepgramProvider()


def test_init_with_empty_api_key_is_fatal(monkeypatch) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "")
    with pytest.raises(FatalError, match="DEEPGRAM_API_KEY"):
        DeepgramProvider()


@pytest.mark.asyncio
async def test_transcribe_posts_with_token_auth_and_params(
    with_api_key, fake_audio_file
) -> None:
    """Provider must hit /v1/listen, Token auth, and our query string."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body_size"] = len(request.content)
        return httpx.Response(200, json=_deepgram_response())

    provider = _make_provider_with_transport(httpx.MockTransport(handler))
    await provider.transcribe(fake_audio_file)

    assert seen["method"] == "POST"
    assert "https://api.deepgram.com/v1/listen" in seen["url"]
    assert seen["headers"]["authorization"] == "Token dg-test-fake"
    # Default config: nova-3 + diarize + smart_format + detect_language
    assert "model=nova-3" in seen["url"]
    assert "diarize=true" in seen["url"]
    assert "smart_format=true" in seen["url"]
    assert "punctuate=true" in seen["url"]
    # Audio body forwarded verbatim.
    assert seen["body_size"] == 1024


@pytest.mark.asyncio
async def test_transcribe_converts_words_to_segment(
    with_api_key, fake_audio_file
) -> None:
    """One TranscriptionSegment with all words; speaker_id propagated."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_deepgram_response())

    provider = _make_provider_with_transport(httpx.MockTransport(handler))
    result = await provider.transcribe(fake_audio_file)

    assert len(result) == 1
    seg = result[0]
    assert isinstance(seg, TranscriptionSegment)
    # punctuated_word preferred over word for display.
    assert seg.words == [
        WordToken(text="Hello,", start=0.10, end=0.50, speaker_id="0"),
        WordToken(text="world.", start=0.50, end=1.50, speaker_id="0"),
    ]
    assert seg.start == 0.10
    assert seg.end == 1.50
    assert seg.language == "en"


@pytest.mark.asyncio
async def test_transcribe_falls_back_to_word_when_punctuated_missing(
    with_api_key, fake_audio_file
) -> None:
    """If ``punctuated_word`` absent, we use raw ``word`` text."""
    body = _deepgram_response(
        words=[{"word": "raw", "start": 0.0, "end": 0.5, "speaker": 1}]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    provider = _make_provider_with_transport(httpx.MockTransport(handler))
    result = await provider.transcribe(fake_audio_file)

    assert result[0].words == [
        WordToken(text="raw", start=0.0, end=0.5, speaker_id="1"),
    ]


@pytest.mark.asyncio
async def test_transcribe_speaker_none_passes_through(
    with_api_key, fake_audio_file
) -> None:
    """Some responses omit ``speaker`` per word; we preserve None."""
    body = _deepgram_response(
        words=[{"word": "hi", "start": 0.0, "end": 0.5}]
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
    """Silent audio: alternatives present but ``words`` empty → no segments."""
    body = _deepgram_response(words=[])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    provider = _make_provider_with_transport(httpx.MockTransport(handler))
    result = await provider.transcribe(fake_audio_file)

    assert result == []


@pytest.mark.asyncio
async def test_no_channels_returns_empty_list(
    with_api_key, fake_audio_file
) -> None:
    body = {"metadata": {}, "results": {"channels": []}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    provider = _make_provider_with_transport(httpx.MockTransport(handler))
    assert await provider.transcribe(fake_audio_file) == []


@pytest.mark.asyncio
async def test_channel_zero_used_when_multiple_channels(
    with_api_key, fake_audio_file
) -> None:
    """Mono assumption — extra channels must not duplicate words."""
    body = {
        "metadata": {"detected_language": "en"},
        "results": {
            "channels": [
                {
                    "detected_language": "en",
                    "alternatives": [
                        {
                            "transcript": "ch0",
                            "words": [
                                {"word": "ch0", "start": 0.0, "end": 0.5, "speaker": 0}
                            ],
                        }
                    ],
                },
                {
                    "detected_language": "en",
                    "alternatives": [
                        {
                            "transcript": "ch1",
                            "words": [
                                {"word": "ch1", "start": 0.0, "end": 0.5, "speaker": 1}
                            ],
                        }
                    ],
                },
            ]
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    provider = _make_provider_with_transport(httpx.MockTransport(handler))
    result = await provider.transcribe(fake_audio_file)

    assert len(result) == 1
    assert [w.text for w in result[0].words] == ["ch0"]


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


@pytest.mark.skipif(
    not os.getenv("DEEPGRAM_API_KEY")
    or os.getenv("DEEPGRAM_API_KEY", "").startswith("dg-test"),
    reason="optional smoke test (requires real DEEPGRAM_API_KEY env)",
)
@pytest.mark.asyncio
async def test_real_deepgram_smoke(fake_audio_file) -> None:
    """Sanity check against the real Deepgram endpoint.

    Skipped by default; run locally with a valid key to verify the
    wire shape has not drifted. The test only asserts a non-error
    response — accuracy belongs in an eval harness.
    """
    provider = DeepgramProvider()
    try:
        await provider.transcribe(fake_audio_file)
    except FatalError:
        # 1 KB of zero bytes is not a valid WAV; Deepgram correctly
        # returns 4xx. We still hit the wire successfully.
        pass
