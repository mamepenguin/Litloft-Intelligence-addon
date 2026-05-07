"""Tests for :class:`GeminiProvider`.

Covered surface:

* Capabilities (cloud, diarization=False, hotwords=True, word ts=False)
* Missing ``GEMINI_API_KEY`` env → ``FatalError``
* Lazy SDK import: missing ``google-genai`` → ``FatalError`` (not bare
  ``ImportError``) so the dispatch layer's ``except FatalError`` catches it
* TOCTOU-safe size pre-check via fstat: file > 2 GB → fatal
* Wire flow: ``client.files.upload`` → ``_wait_for_active`` polling →
  ``client.models.generate_content`` → ``client.files.delete`` cleanup
* Cleanup runs even when generate_content raises (try/finally)
* Synthetic word generation: whitespace split for English / latin,
  grapheme-cluster split for ja/zh/ko/th, language-driven detection
* JSON parse: structured-output payload → list[TranscriptionSegment]
  with synthetic words; missing / malformed JSON → ``FatalError``
* ACTIVE polling: queued/processing → ACTIVE; FAILED → fatal; timeout
  → transient
* Error mapping: 429 → RateLimit, ServerError → Transient,
  ClientError(other) → Fatal
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import MagicMock

import pytest

from app.workers.transcription import (
    FatalError,
    ProviderCapabilities,
    RateLimitError,
    TranscriptionSegment,
    TransientError,
    WordToken,
)


# ---------------------------------------------------------------------------
# Module fixture: install a stub ``google.genai`` SDK before any test in
# this file imports the provider. The real ``google-genai`` is added to
# requirements.txt so the production container sees it; tests run against
# a hand-rolled stub so we can drive ``client.files.upload`` /
# ``client.files.get`` / ``client.models.generate_content`` /
# ``client.files.delete`` shapes deterministically.
# ---------------------------------------------------------------------------


class _FakeFileState:
    def __init__(self, name: str = "ACTIVE") -> None:
        self.name = name


class _FakeUploadedFile:
    def __init__(self, name: str = "files/abc", state: str = "ACTIVE") -> None:
        self.name = name
        self.state = _FakeFileState(state)


class _FakeFiles:
    def __init__(self) -> None:
        self.uploaded: list[str] = []
        self.deleted: list[str] = []
        self.get_states: list[str] = ["ACTIVE"]
        self.upload_exception: Exception | None = None
        self.get_exception: Exception | None = None
        self._initial_file = _FakeUploadedFile()

    def upload(self, file=None, **_):
        if self.upload_exception is not None:
            raise self.upload_exception
        self.uploaded.append(file)
        return self._initial_file

    def get(self, name=None, **_):
        if self.get_exception is not None:
            raise self.get_exception
        state = (
            self.get_states.pop(0)
            if self.get_states
            else "ACTIVE"
        )
        return _FakeUploadedFile(name=name, state=state)

    def delete(self, name=None, **_):
        self.deleted.append(name)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.exception: Exception | None = None
        self.next_response: _FakeResponse | None = None

    def generate_content(self, *, model, contents, config):
        self.calls.append({
            "model": model,
            "contents": contents,
            "config": config,
        })
        if self.exception is not None:
            raise self.exception
        if self.next_response is None:
            return _FakeResponse(
                '{"segments": [{"start": 0.0, "end": 1.0, '
                '"text": "hello world", "language": "en"}]}'
            )
        return self.next_response


class _FakeClient:
    def __init__(self) -> None:
        self.files = _FakeFiles()
        self.models = _FakeModels()


class _FakeClientError(Exception):
    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


class _FakeServerError(Exception):
    pass


def _install_fake_genai_sdk() -> _FakeClient:
    """Install a stub google-genai SDK in ``sys.modules`` and return a
    new ``_FakeClient`` instance. Tests pre-build the client and inject
    it via ``provider._client`` so the production lifecycle (``new
    Client per call``) is bypassed deterministically.
    """
    fake_client = _FakeClient()

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    errors_module = types.ModuleType("google.genai.errors")
    types_module = types.ModuleType("google.genai.types")

    class _GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    types_module.GenerateContentConfig = _GenerateContentConfig
    errors_module.ClientError = _FakeClientError
    errors_module.ServerError = _FakeServerError
    genai_module.errors = errors_module
    genai_module.types = types_module

    class _Client:
        def __init__(self, api_key=None):
            self._api_key = api_key

        files = fake_client.files
        models = fake_client.models

    genai_module.Client = lambda api_key=None: fake_client
    google_module.genai = genai_module

    sys.modules["google"] = google_module
    sys.modules["google.genai"] = genai_module
    sys.modules["google.genai.errors"] = errors_module
    sys.modules["google.genai.types"] = types_module

    return fake_client


@pytest.fixture()
def fake_genai():
    yield _install_fake_genai_sdk()
    # Tests in unrelated modules never import google.genai, so
    # leaving the stubs in sys.modules is harmless.


@pytest.fixture()
def with_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-test-fake")
    yield


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Replace ``asyncio.sleep`` with a no-op so polling tests run fast.

    The provider's polling loop interleaves ``asyncio.sleep`` between
    ``client.files.get`` calls; without this fixture each test in the
    polling group would block for the configured interval. Tests that
    need to verify timeout behaviour drive the loop's clock via
    ``loop.time`` instead.
    """

    async def _instant(_):
        return None

    monkeypatch.setattr("asyncio.sleep", _instant)


@pytest.fixture()
def fake_audio_file(tmp_path):
    p = tmp_path / "clip.wav"
    p.write_bytes(b"\x00" * 1024)
    return str(p)


def _make_provider(fake_client: _FakeClient):
    from app.workers.transcription.gemini import GeminiProvider

    provider = GeminiProvider()
    provider._client = fake_client
    # Polling sleep is mocked to no-op via the autouse ``_no_real_sleep``
    # fixture, so the wait budget only matters for explicit timeout
    # tests (which override ``_upload_wait_sec`` directly).
    provider._upload_wait_sec = 30
    provider._timeout_s = 30
    return provider


# ---------------------------------------------------------------------------
# Static surface
# ---------------------------------------------------------------------------


def test_provider_declared_name(fake_genai, with_api_key) -> None:
    from app.workers.transcription.gemini import GeminiProvider

    assert GeminiProvider.name == "gemini"


def test_provider_capabilities_match_spec(fake_genai, with_api_key) -> None:
    from app.workers.transcription.gemini import GeminiProvider

    assert GeminiProvider.capabilities == ProviderCapabilities(
        sends_audio_offhost=True,
        supports_diarization=False,
        supports_hotwords=True,
        supports_word_timestamps=False,
    )


def test_init_requires_api_key(fake_genai, monkeypatch) -> None:
    from app.workers.transcription.gemini import GeminiProvider

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(FatalError, match="GEMINI_API_KEY"):
        GeminiProvider()


def test_missing_sdk_maps_to_fatal(monkeypatch, with_api_key) -> None:
    """Lazy SDK import: missing google-genai must surface as FatalError.

    The dispatch layer in ``whisper.py`` catches ``(ValueError,
    FatalError)``; bare ImportError would escape unwrapped and
    JobRecord would record the wrong error class.
    """
    import builtins

    # Ensure no SDK module is installed for this test.
    monkeypatch.delitem(sys.modules, "google", raising=False)
    monkeypatch.delitem(sys.modules, "google.genai", raising=False)
    monkeypatch.delitem(
        sys.modules, "app.workers.transcription.gemini", raising=False
    )

    real_import = builtins.__import__

    def fail_genai(name, *a, **kw):
        if name == "google.genai" or name.startswith("google.genai"):
            raise ImportError("no google-genai available")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fail_genai)

    from app.workers.transcription.gemini import GeminiProvider

    with pytest.raises(FatalError, match="google-genai SDK not installed"):
        GeminiProvider()


# ---------------------------------------------------------------------------
# Synthetic word splitter
# ---------------------------------------------------------------------------


def test_synthetic_words_whitespace_split(fake_genai, with_api_key) -> None:
    from app.workers.transcription.gemini import _synthetic_words

    words = _synthetic_words("hello world", 0.0, 2.0, "en")
    assert [w.text for w in words] == ["hello", "world"]
    assert words[0].start == 0.0
    assert words[0].end == 1.0
    assert words[1].start == 1.0
    assert words[1].end == 2.0
    assert all(w.speaker_id is None for w in words)


def test_synthetic_words_japanese_grapheme_split(
    fake_genai, with_api_key
) -> None:
    """Japanese has no whitespace word boundaries — grapheme cluster
    split keeps combining marks attached and gives roughly per-character
    timing."""
    from app.workers.transcription.gemini import _synthetic_words

    words = _synthetic_words("こんにちは", 0.0, 5.0, "ja")
    assert [w.text for w in words] == ["こ", "ん", "に", "ち", "は"]
    assert words[0].start == 0.0
    # Each grapheme gets 1 second.
    assert pytest.approx(words[1].start, rel=1e-9) == 1.0


def test_synthetic_words_chinese_uses_grapheme_split(
    fake_genai, with_api_key
) -> None:
    from app.workers.transcription.gemini import _synthetic_words

    words = _synthetic_words("你好世界", 0.0, 4.0, "zh-CN")
    assert [w.text for w in words] == ["你", "好", "世", "界"]


def test_synthetic_words_korean_uses_grapheme_split(
    fake_genai, with_api_key
) -> None:
    from app.workers.transcription.gemini import _synthetic_words

    words = _synthetic_words("안녕", 0.0, 2.0, "ko")
    assert [w.text for w in words] == ["안", "녕"]


def test_synthetic_words_mixed_japanese_english(
    fake_genai, with_api_key
) -> None:
    """Language wins: ja → grapheme split even when English tokens are
    present in the segment text."""
    from app.workers.transcription.gemini import _synthetic_words

    words = _synthetic_words("hello 世界", 0.0, 6.0, "ja")
    # Whitespace and English chars become individual graphemes too.
    assert any(w.text == "h" for w in words)
    assert any(w.text == "世" for w in words)


def test_synthetic_words_mixed_english_takes_whitespace(
    fake_genai, with_api_key
) -> None:
    """Language wins (en) → whitespace split even with kanji present."""
    from app.workers.transcription.gemini import _synthetic_words

    words = _synthetic_words("hello 世界", 0.0, 4.0, "en")
    assert [w.text for w in words] == ["hello", "世界"]


def test_synthetic_words_empty_returns_empty(
    fake_genai, with_api_key
) -> None:
    from app.workers.transcription.gemini import _synthetic_words

    assert _synthetic_words("", 0.0, 1.0, "en") == []
    assert _synthetic_words("   ", 0.0, 1.0, "en") == []


def test_synthetic_words_language_none_falls_back_to_whitespace(
    fake_genai, with_api_key
) -> None:
    from app.workers.transcription.gemini import _synthetic_words

    words = _synthetic_words("hello world", 0.0, 2.0, None)
    assert [w.text for w in words] == ["hello", "world"]


# ---------------------------------------------------------------------------
# transcribe() end-to-end (against the fake SDK)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcribe_uploads_and_calls_generate_content(
    fake_genai, with_api_key, fake_audio_file
) -> None:
    fake_genai.models.next_response = _FakeResponse(
        '{"segments": [{"start": 0.0, "end": 2.0, '
        '"text": "Hello world", "language": "en"}]}'
    )
    provider = _make_provider(fake_genai)
    result = await provider.transcribe(fake_audio_file)

    assert fake_genai.files.uploaded == [fake_audio_file]
    assert len(fake_genai.models.calls) == 1
    assert fake_genai.models.calls[0]["model"] == "gemini-2.5-flash"
    # generate_content config carries the schema + json mime
    cfg = fake_genai.models.calls[0]["config"]
    assert cfg.kwargs["response_mime_type"] == "application/json"
    assert "response_schema" in cfg.kwargs
    # cleanup ran
    assert fake_genai.files.deleted == ["files/abc"]
    # 1 segment with synthetic words
    assert len(result) == 1
    assert isinstance(result[0], TranscriptionSegment)
    assert [w.text for w in result[0].words] == ["Hello", "world"]


@pytest.mark.asyncio
async def test_cleanup_runs_when_generate_content_fails(
    fake_genai, with_api_key, fake_audio_file
) -> None:
    fake_genai.models.exception = _FakeServerError("internal")
    provider = _make_provider(fake_genai)
    with pytest.raises(TransientError):
        await provider.transcribe(fake_audio_file)
    # cleanup must have run despite the exception
    assert fake_genai.files.deleted == ["files/abc"]


@pytest.mark.asyncio
async def test_wait_for_active_handles_processing_then_active(
    fake_genai, with_api_key, fake_audio_file
) -> None:
    fake_genai.files.get_states = ["PROCESSING", "PROCESSING", "ACTIVE"]
    provider = _make_provider(fake_genai)
    result = await provider.transcribe(fake_audio_file)
    assert len(result) == 1


@pytest.mark.asyncio
async def test_wait_for_active_fails_on_failed_state(
    fake_genai, with_api_key, fake_audio_file
) -> None:
    fake_genai.files.get_states = ["PROCESSING", "FAILED"]
    provider = _make_provider(fake_genai)
    with pytest.raises(FatalError, match="processing failed"):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_wait_for_active_times_out_to_transient(
    fake_genai, with_api_key, fake_audio_file, monkeypatch
) -> None:
    """Stuck-in-processing past upload_wait_sec → TransientError."""
    # Always return PROCESSING.
    def always_processing(name=None, **_):
        return _FakeUploadedFile(name=name, state="PROCESSING")

    fake_genai.files.get = always_processing
    provider = _make_provider(fake_genai)
    provider._upload_wait_sec = 0  # immediate timeout

    with pytest.raises(TransientError, match="did not become ACTIVE"):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_429_in_generate_content_maps_to_rate_limit(
    fake_genai, with_api_key, fake_audio_file
) -> None:
    fake_genai.models.exception = _FakeClientError("rate limit", 429)
    provider = _make_provider(fake_genai)
    with pytest.raises(RateLimitError):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_400_in_generate_content_maps_to_fatal(
    fake_genai, with_api_key, fake_audio_file
) -> None:
    fake_genai.models.exception = _FakeClientError("bad request", 400)
    provider = _make_provider(fake_genai)
    with pytest.raises(FatalError):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_5xx_in_upload_maps_to_transient(
    fake_genai, with_api_key, fake_audio_file
) -> None:
    fake_genai.files.upload_exception = _FakeServerError("boom")
    provider = _make_provider(fake_genai)
    with pytest.raises(TransientError):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_invalid_json_response_maps_to_fatal(
    fake_genai, with_api_key, fake_audio_file
) -> None:
    fake_genai.models.next_response = _FakeResponse("not json at all")
    provider = _make_provider(fake_genai)
    with pytest.raises(FatalError, match="not valid JSON"):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_missing_segments_field_maps_to_fatal(
    fake_genai, with_api_key, fake_audio_file
) -> None:
    fake_genai.models.next_response = _FakeResponse('{"foo": "bar"}')
    provider = _make_provider(fake_genai)
    with pytest.raises(FatalError, match="missing 'segments'"):
        await provider.transcribe(fake_audio_file)


@pytest.mark.asyncio
async def test_empty_segments_returns_empty_list(
    fake_genai, with_api_key, fake_audio_file
) -> None:
    """Silence: empty segments must succeed-with-zero, not error."""
    fake_genai.models.next_response = _FakeResponse('{"segments": []}')
    provider = _make_provider(fake_genai)
    result = await provider.transcribe(fake_audio_file)
    assert result == []


@pytest.mark.asyncio
async def test_pre_check_rejects_oversize_file(
    fake_genai, with_api_key, tmp_path, monkeypatch
) -> None:
    """2 GB byte cap: stub fstat so we don't need 2 GB on disk."""
    from app.workers.transcription.gemini import GEMINI_FILE_SIZE_LIMIT

    p = tmp_path / "huge.wav"
    p.write_bytes(b"\x00")

    class _StubResult:
        st_size = GEMINI_FILE_SIZE_LIMIT + 1

    monkeypatch.setattr("os.fstat", lambda fd: _StubResult())

    provider = _make_provider(fake_genai)
    with pytest.raises(FatalError, match="2GB"):
        await provider.transcribe(str(p))


@pytest.mark.asyncio
async def test_hotwords_appear_in_prompt(
    fake_genai, with_api_key, fake_audio_file
) -> None:
    provider = _make_provider(fake_genai)
    await provider.transcribe(
        fake_audio_file, hotwords=["Litloft", "Cloudflare"]
    )
    contents = fake_genai.models.calls[0]["contents"]
    # contents = [uploaded_file, prompt_string]
    assert any("Litloft" in c for c in contents if isinstance(c, str))
    assert any("Cloudflare" in c for c in contents if isinstance(c, str))


@pytest.mark.asyncio
async def test_language_hint_overrides_output_language(
    fake_genai, with_api_key, fake_audio_file
) -> None:
    provider = _make_provider(fake_genai)
    await provider.transcribe(fake_audio_file, language_hint="en")
    prompt = next(
        c for c in fake_genai.models.calls[0]["contents"] if isinstance(c, str)
    )
    assert "出力言語: en" in prompt


@pytest.mark.asyncio
async def test_returns_non_empty_words_per_segment(
    fake_genai, with_api_key, fake_audio_file
) -> None:
    """Phase 2A invariant: every non-empty segment returns non-empty
    words (synthetic) so the chunker never receives an empty list."""
    fake_genai.models.next_response = _FakeResponse(
        '{"segments": ['
        '{"start": 0.0, "end": 2.0, "text": "Hello world", "language": "en"},'
        '{"start": 2.0, "end": 5.0, "text": "やあ世界", "language": "ja"}'
        ']}'
    )
    provider = _make_provider(fake_genai)
    result = await provider.transcribe(fake_audio_file)
    assert len(result) == 2
    for seg in result:
        assert len(seg.words) > 0
