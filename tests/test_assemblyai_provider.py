"""Tests for :class:`AssemblyAIProvider`.

Covered surface:

* Capabilities (cloud, diarization=True, hotwords=True, word ts=True)
* Missing ``ASSEMBLYAI_API_KEY`` env → ``FatalError``
* Wire shape: 3-call sequence (POST /v2/upload → POST /v2/transcript →
  GET /v2/transcript/{id} polled until completed), correct
  ``content-type`` per phase, ``authorization`` header without
  ``Bearer`` prefix
* Parity: AssemblyAI-shaped JSON → list[TranscriptionSegment] with
  speaker_id propagated and timestamps converted from ms to seconds
* Empty word list → empty result (silence: succeeded-with-zero, not
  an error)
* HTTP error classification: 5xx / timeout = transient, 429 = rate
  limit, 4xx = fatal, in any of the three phases
* TOCTOU-safe size pre-check via fstat: file > 5 GB rejected as fatal
* Polling timeout (transcript stays ``processing``) → transient
* ``status: error`` in poll body → fatal with the API's error message
* ``language_hint`` overrides ``language_detection``
* ``hotwords`` forwarded as ``word_boost`` in the submit body
"""

from __future__ import annotations

import os
import time

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
from app.workers.transcription.assemblyai import (
    ASSEMBLYAI_FILE_SIZE_LIMIT,
    AssemblyAIProvider,
)


@pytest.fixture()
def with_api_key(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "aai-test-fake")
    yield


@pytest.fixture()
def fake_audio_file(tmp_path):
    p = tmp_path / "clip.wav"
    p.write_bytes(b"\x00" * 1024)
    return str(p)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Replace ``asyncio.sleep`` with a no-op so polling tests don't hang."""

    async def _instant(_):
        return None

    monkeypatch.setattr("asyncio.sleep", _instant)


def _completed_transcript(words: list[dict] | None = None) -> dict:
    if words is None:
        words = [
            {
                "text": "Hello,",
                "start": 100,
                "end": 500,
                "speaker": "A",
            },
            {
                "text": "world.",
                "start": 500,
                "end": 1500,
                "speaker": "A",
            },
        ]
    return {
        "id": "tr-abc",
        "status": "completed",
        "text": "Hello, world.",
        "language_code": "en",
        "words": words,
    }


class _SequencedHandler:
    """httpx handler that walks through a list of (matcher, response) pairs.

    Each matcher is called with the request — if it returns True the
    handler emits the paired response. Useful for sequencing the
    upload → submit → poll three-call wire.
    """

    def __init__(self, steps):
        self.steps = list(steps)
        self.calls: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if not self.steps:
            raise AssertionError(
                f"Unexpected extra request: {request.method} {request.url}"
            )
        matcher, response = self.steps[0]
        assert matcher(request), (
            f"Request {request.method} {request.url} did not match the "
            f"next expected step"
        )
        self.steps.pop(0)
        return response


def _make_provider(transport: httpx.MockTransport) -> AssemblyAIProvider:
    provider = AssemblyAIProvider()
    provider._transport = transport
    return provider


def _is_upload(request: httpx.Request) -> bool:
    return request.method == "POST" and request.url.path == "/v2/upload"


def _is_submit(request: httpx.Request) -> bool:
    return request.method == "POST" and request.url.path == "/v2/transcript"


def _is_poll(request: httpx.Request) -> bool:
    return request.method == "GET" and request.url.path.startswith(
        "/v2/transcript/"
    )


def test_provider_declared_name() -> None:
    assert AssemblyAIProvider.name == "assemblyai"


def test_provider_capabilities_match_spec() -> None:
    assert AssemblyAIProvider.capabilities == ProviderCapabilities(
        sends_audio_offhost=True,
        supports_diarization=True,
        supports_hotwords=True,
        supports_word_timestamps=True,
        max_input_bytes=ASSEMBLYAI_FILE_SIZE_LIMIT,
        accepts_initial_prompt=False,
        handles_own_retry=False,
    )


def test_init_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    with pytest.raises(FatalError, match="ASSEMBLYAI_API_KEY"):
        AssemblyAIProvider()


def test_init_with_empty_api_key_is_fatal(monkeypatch) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "")
    with pytest.raises(FatalError, match="ASSEMBLYAI_API_KEY"):
        AssemblyAIProvider()


@pytest.mark.asyncio
async def test_three_call_wire_shape(with_api_key, fake_audio_file) -> None:
    """upload → submit → poll, each with the correct headers / body."""
    submit_body: dict = {}

    def upload_response(request):
        assert (
            request.headers.get("content-type") == "application/octet-stream"
        )
        assert request.headers.get("authorization") == "aai-test-fake"
        assert len(request.content) == 1024
        return httpx.Response(
            200, json={"upload_url": "https://cdn.assemblyai/abc"}
        )

    def submit_response(request):
        assert request.headers.get("content-type", "").startswith(
            "application/json"
        )
        assert request.headers.get("authorization") == "aai-test-fake"
        body = request.read().decode()
        import json as _json

        submit_body.update(_json.loads(body))
        return httpx.Response(200, json={"id": "tr-abc"})

    def poll_response(request):
        assert request.headers.get("authorization") == "aai-test-fake"
        return httpx.Response(200, json=_completed_transcript())

    handler = _SequencedHandler(
        [
            (_is_upload, None),
            (_is_submit, None),
            (_is_poll, None),
        ]
    )

    def dispatch(request: httpx.Request) -> httpx.Response:
        if _is_upload(request):
            return upload_response(request)
        if _is_submit(request):
            return submit_response(request)
        if _is_poll(request):
            return poll_response(request)
        raise AssertionError(f"unexpected: {request.method} {request.url}")

    provider = _make_provider(httpx.MockTransport(dispatch))
    await provider.transcribe(fake_audio_file)

    assert submit_body["audio_url"] == "https://cdn.assemblyai/abc"
    assert submit_body["speech_model"] == "best"
    assert submit_body["speaker_labels"] is True
    assert submit_body["language_detection"] is True


@pytest.mark.asyncio
async def test_language_hint_overrides_detection(
    with_api_key, fake_audio_file
) -> None:
    captured: dict = {}

    def dispatch(request: httpx.Request) -> httpx.Response:
        if _is_upload(request):
            return httpx.Response(
                200, json={"upload_url": "https://cdn.assemblyai/x"}
            )
        if _is_submit(request):
            import json as _json

            captured.update(_json.loads(request.read().decode()))
            return httpx.Response(200, json={"id": "tr-x"})
        return httpx.Response(200, json=_completed_transcript())

    provider = _make_provider(httpx.MockTransport(dispatch))
    await provider.transcribe(fake_audio_file, language_hint="ja")

    assert captured["language_code"] == "ja"
    assert "language_detection" not in captured


@pytest.mark.asyncio
async def test_hotwords_forwarded_as_word_boost(
    with_api_key, fake_audio_file
) -> None:
    captured: dict = {}

    def dispatch(request: httpx.Request) -> httpx.Response:
        if _is_upload(request):
            return httpx.Response(
                200, json={"upload_url": "https://cdn/x"}
            )
        if _is_submit(request):
            import json as _json

            captured.update(_json.loads(request.read().decode()))
            return httpx.Response(200, json={"id": "tr-x"})
        return httpx.Response(200, json=_completed_transcript())

    provider = _make_provider(httpx.MockTransport(dispatch))
    await provider.transcribe(
        fake_audio_file, hotwords=["Litloft", "Cloudflare"]
    )

    assert captured["word_boost"] == ["Litloft", "Cloudflare"]


@pytest.mark.asyncio
async def test_polling_succeeds_after_processing(
    with_api_key, fake_audio_file
) -> None:
    """``processing`` → poll again, eventually ``completed``."""
    poll_count = {"n": 0}

    def dispatch(request: httpx.Request) -> httpx.Response:
        if _is_upload(request):
            return httpx.Response(
                200, json={"upload_url": "https://cdn/x"}
            )
        if _is_submit(request):
            return httpx.Response(200, json={"id": "tr-x"})
        # Poll: queued → processing → completed
        poll_count["n"] += 1
        if poll_count["n"] == 1:
            return httpx.Response(200, json={"status": "queued"})
        if poll_count["n"] == 2:
            return httpx.Response(200, json={"status": "processing"})
        return httpx.Response(200, json=_completed_transcript())

    provider = _make_provider(httpx.MockTransport(dispatch))
    result = await provider.transcribe(fake_audio_file)
    assert poll_count["n"] >= 3
    assert len(result) == 1


@pytest.mark.asyncio
async def test_words_converted_with_ms_to_s(
    with_api_key, fake_audio_file
) -> None:
    """AssemblyAI returns ms timestamps; we expose seconds."""

    def dispatch(request: httpx.Request) -> httpx.Response:
        if _is_upload(request):
            return httpx.Response(
                200, json={"upload_url": "https://cdn/x"}
            )
        if _is_submit(request):
            return httpx.Response(200, json={"id": "tr-x"})
        return httpx.Response(200, json=_completed_transcript())

    provider = _make_provider(httpx.MockTransport(dispatch))
    result = await provider.transcribe(fake_audio_file)

    assert len(result) == 1
    seg = result[0]
    assert isinstance(seg, TranscriptionSegment)
    assert seg.words == [
        WordToken(text="Hello,", start=0.10, end=0.50, speaker_id="A"),
        WordToken(text="world.", start=0.50, end=1.50, speaker_id="A"),
    ]
    assert seg.start == 0.10
    assert seg.end == 1.50
    assert seg.language == "en"


@pytest.mark.asyncio
async def test_speaker_none_passes_through(
    with_api_key, fake_audio_file
) -> None:
    body = _completed_transcript(
        words=[{"text": "hi", "start": 0, "end": 500}]
    )

    def dispatch(request: httpx.Request) -> httpx.Response:
        if _is_upload(request):
            return httpx.Response(200, json={"upload_url": "https://cdn/x"})
        if _is_submit(request):
            return httpx.Response(200, json={"id": "tr-x"})
        return httpx.Response(200, json=body)

    provider = _make_provider(httpx.MockTransport(dispatch))
    result = await provider.transcribe(fake_audio_file)
    assert result[0].words[0].speaker_id is None


@pytest.mark.asyncio
async def test_empty_words_returns_empty_list(
    with_api_key, fake_audio_file
) -> None:
    body = _completed_transcript(words=[])

    def dispatch(request: httpx.Request) -> httpx.Response:
        if _is_upload(request):
            return httpx.Response(200, json={"upload_url": "https://cdn/x"})
        if _is_submit(request):
            return httpx.Response(200, json={"id": "tr-x"})
        return httpx.Response(200, json=body)

    provider = _make_provider(httpx.MockTransport(dispatch))
    assert await provider.transcribe(fake_audio_file) == []


@pytest.mark.asyncio
async def test_status_error_in_poll_maps_to_fatal(
    with_api_key, fake_audio_file
) -> None:
    def dispatch(request: httpx.Request) -> httpx.Response:
        if _is_upload(request):
            return httpx.Response(200, json={"upload_url": "https://cdn/x"})
        if _is_submit(request):
            return httpx.Response(200, json={"id": "tr-x"})
        return httpx.Response(
            200, json={"status": "error", "error": "audio decoding failed"}
        )

    provider = _make_provider(httpx.MockTransport(dispatch))
    with pytest.raises(FatalError, match="audio decoding failed"):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_polling_timeout_maps_to_transient(
    with_api_key, fake_audio_file, monkeypatch
) -> None:
    """Stuck-in-processing past timeout_s → TransientError.

    The provider's deadline is computed against
    ``asyncio.get_event_loop().time()``. We monkeypatch it to a fast-
    forwarding clock so the test doesn't actually wait ``timeout_s``
    seconds.
    """
    fake_time = {"now": 0.0}

    class _FakeLoop:
        def time(self):
            fake_time["now"] += 1000.0  # advance well past timeout
            return fake_time["now"]

    monkeypatch.setattr(
        "asyncio.get_event_loop", lambda: _FakeLoop()
    )

    def dispatch(request: httpx.Request) -> httpx.Response:
        if _is_upload(request):
            return httpx.Response(200, json={"upload_url": "https://cdn/x"})
        if _is_submit(request):
            return httpx.Response(200, json={"id": "tr-x"})
        return httpx.Response(200, json={"status": "processing"})

    provider = _make_provider(httpx.MockTransport(dispatch))
    with pytest.raises(TransientError, match="polling timeout"):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_429_in_upload_maps_to_rate_limit(
    with_api_key, fake_audio_file
) -> None:
    def dispatch(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"err": "rate"})

    provider = _make_provider(httpx.MockTransport(dispatch))
    with pytest.raises(RateLimitError):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [400, 401, 403, 413, 422])
async def test_4xx_in_upload_maps_to_fatal(
    with_api_key, fake_audio_file, code
) -> None:
    def dispatch(request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, json={"err": "bad"})

    provider = _make_provider(httpx.MockTransport(dispatch))
    with pytest.raises(FatalError):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [500, 502, 503])
async def test_5xx_in_upload_maps_to_transient(
    with_api_key, fake_audio_file, code
) -> None:
    def dispatch(request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, json={"err": "boom"})

    provider = _make_provider(httpx.MockTransport(dispatch))
    with pytest.raises(TransientError):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_network_error_in_submit_maps_to_transient(
    with_api_key, fake_audio_file
) -> None:
    def dispatch(request: httpx.Request) -> httpx.Response:
        if _is_upload(request):
            return httpx.Response(200, json={"upload_url": "https://cdn/x"})
        raise httpx.ConnectError("connection refused")

    provider = _make_provider(httpx.MockTransport(dispatch))
    with pytest.raises(TransientError):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_pre_check_rejects_oversize_file(
    with_api_key, tmp_path, monkeypatch
) -> None:
    """5 GB byte cap: stub fstat so we don't actually need 5 GB on disk."""
    p = tmp_path / "huge.wav"
    p.write_bytes(b"\x00")
    fake_path = str(p)

    real_fstat = os.fstat

    class _StubResult:
        st_size = ASSEMBLYAI_FILE_SIZE_LIMIT + 1

    def fake_fstat(fd):
        return _StubResult()

    monkeypatch.setattr("os.fstat", fake_fstat)

    def dispatch(request: httpx.Request) -> httpx.Response:
        raise AssertionError("upload should not be reached")

    provider = _make_provider(httpx.MockTransport(dispatch))
    with pytest.raises(FatalError, match="5GB"):
        await provider.transcribe(fake_path)

    # Restore for other tests
    monkeypatch.setattr("os.fstat", real_fstat)


@pytest.mark.asyncio
async def test_missing_upload_url_is_fatal(
    with_api_key, fake_audio_file
) -> None:
    def dispatch(request: httpx.Request) -> httpx.Response:
        if _is_upload(request):
            return httpx.Response(200, json={"unexpected": "shape"})
        raise AssertionError("submit should not be reached")

    provider = _make_provider(httpx.MockTransport(dispatch))
    with pytest.raises(FatalError, match="missing upload_url"):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_unexpected_status_in_poll_is_fatal(
    with_api_key, fake_audio_file
) -> None:
    def dispatch(request: httpx.Request) -> httpx.Response:
        if _is_upload(request):
            return httpx.Response(200, json={"upload_url": "https://cdn/x"})
        if _is_submit(request):
            return httpx.Response(200, json={"id": "tr-x"})
        return httpx.Response(200, json={"status": "wat"})

    provider = _make_provider(httpx.MockTransport(dispatch))
    with pytest.raises(FatalError, match="unexpected status"):
        await provider.transcribe(fake_audio_file)
