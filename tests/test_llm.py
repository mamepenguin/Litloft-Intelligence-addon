"""Tests for app.llm module.

Covers LLMClient initialization (enabled/disabled detection),
text generation with mocked AsyncOpenAI, JSON parsing with
regex fallback, retry with exponential backoff, and rate limiting.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from app.config import LLMConfig
from app.llm import LLMClient, OllamaLLMClient, create_llm_client


def _make_response_obj(text: str) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    return response


def _make_status_error(status_code: int) -> APIStatusError:
    """Build an APIStatusError with a specific status code."""
    request = httpx.Request("POST", "http://test/chat/completions")
    response = httpx.Response(
        status_code=status_code,
        request=request,
        content=b'{"error": {"message": "test"}}',
    )
    return APIStatusError(
        message=f"HTTP {status_code}",
        response=response,
        body=None,
    )


def _make_rate_limit_error() -> RateLimitError:
    """Build a RateLimitError (429)."""
    request = httpx.Request("POST", "http://test/chat/completions")
    response = httpx.Response(
        status_code=429,
        request=request,
        content=b'{"error": {"message": "Rate limit exceeded"}}',
    )
    return RateLimitError(
        message="Rate limit exceeded",
        response=response,
        body=None,
    )


def _make_internal_server_error() -> InternalServerError:
    """Build an InternalServerError (500)."""
    request = httpx.Request("POST", "http://test/chat/completions")
    response = httpx.Response(
        status_code=500,
        request=request,
        content=b'{"error": {"message": "Internal error"}}',
    )
    return InternalServerError(
        message="Internal server error",
        response=response,
        body=None,
    )


def _mock_completion(client: LLMClient, content: str | None) -> None:
    """Answer at the transport seam, where a real request would go."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].finish_reason = "stop"
    response.usage = None
    client._client = MagicMock()
    client._client.chat.completions.create = AsyncMock(return_value=response)


def _make_client(
    retry_attempts: int = 3,
    retry_base_delay: float = 0.01,
    retry_max_delay: float = 0.1,
    min_request_interval_ms: int = 0,
) -> LLMClient:
    """Build an enabled LLMClient with fast retry for tests."""
    config = LLMConfig(
        provider="openai_compatible",
        base_url="http://localhost:11434/v1",
        model="llama3",
        retry_attempts=retry_attempts,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
        min_request_interval_ms=min_request_interval_ms,
    )
    return LLMClient(config)


# ---------------------------------------------------------------------------
# LLMClient.__init__ (enabled/disabled detection)
# ---------------------------------------------------------------------------


class TestLLMClientInit:
    """Tests for LLMClient enabled/disabled state based on config."""

    def test_disabled_when_provider_is_disabled(self):
        config = LLMConfig(
            provider="disabled",
            base_url="http://localhost:11434/v1",
            model="llama3",
        )

        client = LLMClient(config)

        assert client.enabled is False

    def test_disabled_when_base_url_empty(self):
        config = LLMConfig(
            provider="openai_compatible",
            base_url="",
            model="llama3",
        )

        client = LLMClient(config)

        assert client.enabled is False

    def test_disabled_when_model_empty(self):
        config = LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="",
        )

        client = LLMClient(config)

        assert client.enabled is False

    def test_enabled_when_all_set(self):
        config = LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
        )

        client = LLMClient(config)

        assert client.enabled is True

    def test_disabled_when_all_defaults(self):
        config = LLMConfig()

        client = LLMClient(config)

        assert client.enabled is False


# ---------------------------------------------------------------------------
# LLMClient.generate
# ---------------------------------------------------------------------------


class TestLLMClientGenerate:
    """Tests for LLMClient.generate: async text completion."""

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self):
        config = LLMConfig(provider="disabled")
        client = LLMClient(config)

        result = await client.generate("system", "user")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_response_text_on_success(self):
        config = LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
        )
        client = LLMClient(config)

        # Mock the internal AsyncOpenAI client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated text"

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await client.generate("Be helpful.", "Hello")

        assert result == "Generated text"
        client._client.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        config = LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
        )
        client = LLMClient(config)

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        result = await client.generate("system", "user")

        assert result is None


# ---------------------------------------------------------------------------
# LLMClient.generate_json
# ---------------------------------------------------------------------------


class TestLLMClientGenerateJson:
    """Tests for LLMClient.generate_json: JSON parsing with fallback."""

    @pytest.mark.asyncio
    async def test_parses_valid_json_array(self):
        config = LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
        )
        client = LLMClient(config)
        _mock_completion(client, '["tag1", "tag2", "tag3"]')

        result = await client.generate_json("system", "user")

        assert result == ["tag1", "tag2", "tag3"]

    @pytest.mark.asyncio
    async def test_parses_valid_json_dict(self):
        config = LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
        )
        client = LLMClient(config)
        _mock_completion(client, '{"key": "value"}')

        result = await client.generate_json("system", "user")

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_regex_fallback_extracts_array(self):
        config = LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
        )
        client = LLMClient(config)
        # LLM returns array with surrounding text
        _mock_completion(
            client, 'Here are some tags: ["tag1", "tag2"] hope that helps!'
        )

        result = await client.generate_json("system", "user")

        assert result == ["tag1", "tag2"]

    @pytest.mark.asyncio
    async def test_returns_none_for_completely_invalid(self):
        config = LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
        )
        client = LLMClient(config)
        _mock_completion(client, "This is not JSON at all")

        result = await client.generate_json("system", "user")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_generate_returns_none(self):
        config = LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
        )
        client = LLMClient(config)
        _mock_completion(client, None)

        result = await client.generate_json("system", "user")

        assert result is None

    @pytest.mark.asyncio
    async def test_regex_fallback_with_multiline_array(self):
        config = LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
        )
        client = LLMClient(config)
        _mock_completion(client, 'Sure!\n[\n  "tag1",\n  "tag2"\n]\nDone.')

        result = await client.generate_json("system", "user")

        assert result == ["tag1", "tag2"]


# ---------------------------------------------------------------------------
# LLMClient.generate retry logic
# ---------------------------------------------------------------------------


class TestLLMClientRetry:
    """Tests for retry behavior on transient failures."""

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_error(self):
        """RateLimitError (429) triggers retry and eventually succeeds."""
        client = _make_client(retry_attempts=3)
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=[
                _make_rate_limit_error(),
                _make_rate_limit_error(),
                _make_response_obj("success"),
            ]
        )

        result = await client.generate("system", "user")

        assert result == "success"
        assert client._client.chat.completions.create.await_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self):
        """APITimeoutError triggers retry and eventually succeeds."""
        client = _make_client(retry_attempts=2)
        client._client = MagicMock()
        request = httpx.Request("POST", "http://test")
        client._client.chat.completions.create = AsyncMock(
            side_effect=[
                APITimeoutError(request=request),
                _make_response_obj("ok"),
            ]
        )

        result = await client.generate("system", "user")

        assert result == "ok"
        assert client._client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self):
        """APIConnectionError triggers retry."""
        client = _make_client(retry_attempts=1)
        client._client = MagicMock()
        request = httpx.Request("POST", "http://test")
        client._client.chat.completions.create = AsyncMock(
            side_effect=[
                APIConnectionError(request=request),
                _make_response_obj("ok"),
            ]
        )

        result = await client.generate("system", "user")

        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_internal_server_error(self):
        """InternalServerError (500) triggers retry."""
        client = _make_client(retry_attempts=1)
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=[
                _make_internal_server_error(),
                _make_response_obj("ok"),
            ]
        )

        result = await client.generate("system", "user")

        assert result == "ok"

    @pytest.mark.asyncio
    async def test_no_retry_on_insufficient_balance(self):
        """402 Insufficient Balance is permanent — no retry."""
        client = _make_client(retry_attempts=3)
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=_make_status_error(402)
        )

        result = await client.generate("system", "user")

        assert result is None
        # Called only once (no retries)
        assert client._client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_bad_request(self):
        """400 is permanent once our own body field is ruled out.

        Reasoning suppression is on by default, so the first 400 buys one
        re-send without that field. A 400 that survives it is the
        provider rejecting the request itself, and is not retried.
        """
        client = _make_client(retry_attempts=3)
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=_make_status_error(400)
        )

        result = await client.generate("system", "user")

        calls = client._client.chat.completions.create.await_args_list
        assert result is None
        assert len(calls) == 2
        assert "extra_body" in calls[0].kwargs
        assert "extra_body" not in calls[1].kwargs

    @pytest.mark.asyncio
    async def test_no_retry_on_unauthorized(self):
        """401 Unauthorized is permanent — no retry."""
        client = _make_client(retry_attempts=3)
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=_make_status_error(401)
        )

        result = await client.generate("system", "user")

        assert result is None
        assert client._client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_retry_exhausted_returns_none(self):
        """All retries failing returns None."""
        client = _make_client(retry_attempts=2)
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=_make_rate_limit_error()
        )

        result = await client.generate("system", "user")

        assert result is None
        # Initial call + 2 retries = 3 total
        assert client._client.chat.completions.create.await_count == 3

    @pytest.mark.asyncio
    async def test_zero_retries(self):
        """retry_attempts=0 means one attempt total, no retries."""
        client = _make_client(retry_attempts=0)
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=_make_rate_limit_error()
        )

        result = await client.generate("system", "user")

        assert result is None
        assert client._client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_generic_exception_not_retried(self):
        """Unknown exceptions fall through to the generic catch — no retry."""
        client = _make_client(retry_attempts=3)
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=ValueError("unexpected")
        )

        result = await client.generate("system", "user")

        assert result is None
        assert client._client.chat.completions.create.await_count == 1

    def test_backoff_delay_doubles(self):
        """Backoff delay doubles on each attempt."""
        client = _make_client(
            retry_base_delay=1.0, retry_max_delay=100.0,
        )
        assert client._backoff_delay(0) == 1.0
        assert client._backoff_delay(1) == 2.0
        assert client._backoff_delay(2) == 4.0
        assert client._backoff_delay(3) == 8.0

    def test_backoff_delay_capped_at_max(self):
        """Backoff delay is capped at retry_max_delay."""
        client = _make_client(
            retry_base_delay=1.0, retry_max_delay=5.0,
        )
        assert client._backoff_delay(0) == 1.0
        assert client._backoff_delay(1) == 2.0
        assert client._backoff_delay(2) == 4.0
        # Would be 8.0, capped at 5.0
        assert client._backoff_delay(3) == 5.0
        assert client._backoff_delay(10) == 5.0

    @pytest.mark.asyncio
    async def test_retry_uses_exponential_backoff(self):
        """Verify asyncio.sleep is called with exponentially increasing delays."""
        client = _make_client(
            retry_attempts=3,
            retry_base_delay=1.0,
            retry_max_delay=100.0,
        )
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=[
                _make_rate_limit_error(),
                _make_rate_limit_error(),
                _make_rate_limit_error(),
                _make_response_obj("ok"),
            ]
        )

        with patch("app.llm.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await client.generate("system", "user")

        assert result == "ok"
        # Three retries → three backoff delays: 1.0, 2.0, 4.0
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# LLMClient rate limiting
# ---------------------------------------------------------------------------


class TestLLMClientRateLimit:
    """Tests for minimum request interval enforcement."""

    @pytest.mark.asyncio
    async def test_no_rate_limit_by_default(self):
        """min_request_interval_ms=0 means no delay between requests."""
        client = _make_client(min_request_interval_ms=0)
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            return_value=_make_response_obj("ok")
        )

        with patch("app.llm.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await client.generate("system", "user")
            await client.generate("system", "user")

        # No rate-limiting sleeps should happen
        assert mock_sleep.await_count == 0

    @pytest.mark.asyncio
    async def test_rate_limit_enforces_interval(self):
        """Successive requests wait for the configured interval."""
        client = _make_client(min_request_interval_ms=500)
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            return_value=_make_response_obj("ok")
        )

        # Mock time: first call sets _last_request_time to 0.0,
        # second call sees elapsed=0.1 - 0.0 = 0.1, waits 0.4s
        with patch("app.llm.time.monotonic") as mock_time, \
             patch("app.llm.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # First gen: 1 monotonic() (save last_time=0.0)
            # Second gen: 2 monotonic() (compute elapsed=0.1, then save=0.5)
            mock_time.side_effect = [0.0, 0.1, 0.5]
            await client.generate("system", "user")
            await client.generate("system", "user")

        sleep_calls = [
            call.args[0] for call in mock_sleep.call_args_list
            if call.args[0] > 0
        ]
        assert len(sleep_calls) == 1
        assert sleep_calls[0] == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_rate_limit_skipped_if_enough_time_elapsed(self):
        """If interval has already passed, no sleep occurs."""
        client = _make_client(min_request_interval_ms=100)
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            return_value=_make_response_obj("ok")
        )

        with patch("app.llm.time.monotonic") as mock_time, \
             patch("app.llm.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # First gen saves t=0.0, second gen sees elapsed=1.0 (> 0.1)
            mock_time.side_effect = [0.0, 1.0, 1.0]
            await client.generate("system", "user")
            await client.generate("system", "user")

        positive_sleeps = [
            call.args[0] for call in mock_sleep.call_args_list
            if call.args[0] > 0
        ]
        assert positive_sleeps == []

    @pytest.mark.asyncio
    async def test_rate_limit_first_request_no_wait(self):
        """First request never waits even with rate limit configured."""
        client = _make_client(min_request_interval_ms=1000)
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            return_value=_make_response_obj("ok")
        )

        with patch("app.llm.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await client.generate("system", "user")

        positive_sleeps = [
            call.args[0] for call in mock_sleep.call_args_list
            if call.args[0] > 0
        ]
        assert positive_sleeps == []


# ---------------------------------------------------------------------------
# generate_stream
# ---------------------------------------------------------------------------


def _make_stream_chunk(content: str | None):
    """Build a mock ChatCompletionChunk with the given delta content.

    A real OpenAI streaming chunk has ``chunk.choices[0].delta.content``.
    The SDK emits a None content on the first (role-only) and last
    (finish) chunks; we mimic that shape so the filter in
    generate_stream is exercised.
    """
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = content
    return chunk


async def _async_iter(chunks):
    for c in chunks:
        yield c


class TestGenerateStream:
    @pytest.mark.asyncio
    async def test_yields_content_deltas_in_order(self):
        client = _make_client()
        chunks = [
            _make_stream_chunk(None),      # role-only first chunk
            _make_stream_chunk("Hello, "),
            _make_stream_chunk("world"),
            _make_stream_chunk(None),      # finish chunk
        ]
        client._client.chat.completions.create = AsyncMock(
            return_value=_async_iter(chunks)
        )

        deltas: list[str] = []
        async for delta in client.generate_stream("sys", "usr"):
            deltas.append(delta)

        assert deltas == ["Hello, ", "world"]

    @pytest.mark.asyncio
    async def test_disabled_yields_nothing(self):
        config = LLMConfig(provider="disabled")
        client = LLMClient(config)

        deltas = [d async for d in client.generate_stream("sys", "usr")]

        assert deltas == []

    @pytest.mark.asyncio
    async def test_open_failure_yields_nothing(self):
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(
            side_effect=_make_rate_limit_error()
        )

        deltas = [d async for d in client.generate_stream("sys", "usr")]

        # Stream open failed; generator terminates cleanly without
        # retry (the caller has to decide what to do with a short
        # or empty stream).
        assert deltas == []

    @pytest.mark.asyncio
    async def test_mid_stream_error_returns_partial_output(self):
        """A crash mid-iteration must not propagate — we keep what we got."""
        client = _make_client()

        async def _broken_iter():
            yield _make_stream_chunk("partial ")
            yield _make_stream_chunk("output")
            raise RuntimeError("transport died")

        client._client.chat.completions.create = AsyncMock(
            return_value=_broken_iter()
        )

        deltas = [d async for d in client.generate_stream("sys", "usr")]

        # The two successfully-received chunks survive; the crash is
        # swallowed and the iterator terminates cleanly.
        assert deltas == ["partial ", "output"]


# ---------------------------------------------------------------------------
# OllamaLLMClient
# ---------------------------------------------------------------------------


def _make_ollama_client(
    base_url: str = "http://localhost:11434",
    retry_attempts: int = 3,
    retry_base_delay: float = 0.01,
    retry_max_delay: float = 0.1,
) -> OllamaLLMClient:
    """Build an enabled OllamaLLMClient with fast retry for tests."""
    config = LLMConfig(
        provider="ollama",
        base_url=base_url,
        model="gemma4:e4b",
        retry_attempts=retry_attempts,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
    )
    return OllamaLLMClient(config)


class TestOllamaLLMClientInit:
    """Initialisation / config handling for OllamaLLMClient."""

    def test_enabled_when_configured(self):
        client = _make_ollama_client()
        assert client.enabled is True

    def test_disabled_when_provider_disabled(self):
        config = LLMConfig(
            provider="disabled",
            base_url="http://localhost:11434",
            model="gemma4:e4b",
        )
        assert OllamaLLMClient(config).enabled is False

    def test_base_url_strips_v1_suffix(self):
        """Users who paste an openai_compatible URL shouldn't get /v1/api/chat."""
        client = _make_ollama_client(base_url="http://localhost:11434/v1")
        assert client._base_url == "http://localhost:11434"

    def test_base_url_strips_trailing_slash(self):
        client = _make_ollama_client(base_url="http://localhost:11434/")
        assert client._base_url == "http://localhost:11434"


class TestOllamaLLMClientGenerate:
    """Non-streaming generate() behaviour."""

    @pytest.mark.asyncio
    async def test_sends_think_false(self):
        """The body MUST include think: false to skip reasoning."""
        client = _make_ollama_client()

        captured: dict = {}

        async def fake_post(url, json):
            captured["url"] = url
            captured["json"] = json
            return httpx.Response(
                status_code=200,
                content=b'{"message": {"content": "hi"}}',
            )

        client._http.post = fake_post

        result = await client.generate("sys", "usr")

        assert result == "hi"
        assert captured["url"] == "http://localhost:11434/api/chat"
        assert captured["json"]["think"] is False
        assert captured["json"]["stream"] is False
        assert captured["json"]["model"] == "gemma4:e4b"
        assert captured["json"]["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]

    @pytest.mark.asyncio
    async def test_translates_openai_json_object_to_ollama_json(self):
        """response_format={'type': 'json_object'} → format='json'."""
        client = _make_ollama_client()

        captured: dict = {}

        async def fake_post(url, json):
            captured["json"] = json
            return httpx.Response(
                status_code=200,
                content=b'{"message": {"content": "{}"}}',
            )

        client._http.post = fake_post

        await client.generate(
            "sys", "usr", response_format={"type": "json_object"}
        )

        assert captured["json"]["format"] == "json"

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self):
        config = LLMConfig(provider="disabled", base_url="", model="")
        client = OllamaLLMClient(config)
        assert await client.generate("sys", "usr") is None

    @pytest.mark.asyncio
    async def test_retries_on_transient_status(self):
        """5xx and 429 should trigger exponential backoff retries."""
        client = _make_ollama_client(retry_attempts=2)

        call_count = {"n": 0}

        async def fake_post(url, json):
            call_count["n"] += 1
            if call_count["n"] < 3:
                return httpx.Response(
                    status_code=503, content=b'{"error": "busy"}'
                )
            return httpx.Response(
                status_code=200,
                content=b'{"message": {"content": "ok"}}',
            )

        client._http.post = fake_post

        result = await client.generate("sys", "usr")

        assert result == "ok"
        assert call_count["n"] == 3

    @pytest.mark.asyncio
    async def test_permanent_error_not_retried(self):
        """400/401/403/404 should fail immediately."""
        client = _make_ollama_client(retry_attempts=3)

        call_count = {"n": 0}

        async def fake_post(url, json):
            call_count["n"] += 1
            return httpx.Response(
                status_code=404, content=b'{"error": "not found"}'
            )

        client._http.post = fake_post

        result = await client.generate("sys", "usr")

        assert result is None
        assert call_count["n"] == 1


class TestOllamaLLMClientStream:
    """Streaming generate_stream() behaviour."""

    @pytest.mark.asyncio
    async def test_yields_content_deltas_from_ndjson(self):
        """Each {"message": {"content": "..."}} line → one yielded delta."""
        client = _make_ollama_client()

        class FakeStreamResponse:
            status_code = 200

            async def aiter_lines(self):
                yield '{"message": {"content": "Hello"}, "done": false}'
                yield '{"message": {"content": " world"}, "done": false}'
                yield '{"message": {"content": "!"}, "done": true}'

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        captured: dict = {}

        def fake_stream(method, url, json):
            captured["method"] = method
            captured["url"] = url
            captured["json"] = json
            return FakeStreamResponse()

        client._http.stream = fake_stream

        deltas = [d async for d in client.generate_stream("sys", "usr")]

        assert deltas == ["Hello", " world", "!"]
        assert captured["method"] == "POST"
        assert captured["url"] == "http://localhost:11434/api/chat"
        assert captured["json"]["stream"] is True
        assert captured["json"]["think"] is False

    @pytest.mark.asyncio
    async def test_stream_handles_malformed_lines(self):
        """Non-JSON lines should be skipped without breaking the stream."""
        client = _make_ollama_client()

        class FakeStreamResponse:
            status_code = 200

            async def aiter_lines(self):
                yield ""
                yield "not json"
                yield '{"message": {"content": "ok"}, "done": true}'

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        client._http.stream = lambda method, url, json: FakeStreamResponse()

        deltas = [d async for d in client.generate_stream("sys", "usr")]

        assert deltas == ["ok"]

    @pytest.mark.asyncio
    async def test_stream_yields_nothing_on_failed_open(self):
        """Non-200 response → empty iterator, no exception."""
        client = _make_ollama_client()

        class FakeStreamResponse:
            status_code = 500

            async def aiter_lines(self):
                return
                yield  # unreachable

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        client._http.stream = lambda method, url, json: FakeStreamResponse()

        deltas = [d async for d in client.generate_stream("sys", "usr")]

        assert deltas == []


class TestCreateLLMClient:
    """Factory dispatch based on config.provider."""

    def test_ollama_provider_returns_ollama_client(self):
        config = LLMConfig(
            provider="ollama",
            base_url="http://localhost:11434",
            model="gemma4:e4b",
        )
        client = create_llm_client(config)
        assert isinstance(client, OllamaLLMClient)

    def test_openai_compatible_returns_llm_client(self):
        config = LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="gemma4:e4b",
        )
        client = create_llm_client(config)
        assert isinstance(client, LLMClient)

    def test_disabled_returns_llm_client(self):
        """Disabled provider gets LLMClient with enabled=False."""
        config = LLMConfig(provider="disabled", base_url="", model="")
        client = create_llm_client(config)
        assert isinstance(client, LLMClient)
        assert client.enabled is False


# ---------------------------------------------------------------------------
# llm.reasoning -> request body
# ---------------------------------------------------------------------------


def _reasoning_client(reasoning: str) -> LLMClient:
    """An enabled client whose request kwargs can be inspected."""
    client = LLMClient(
        LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
            vision_model="llava",
            retry_attempts=0,
            reasoning=reasoning,
        )
    )
    client._client = MagicMock()
    client._client.chat.completions.create = AsyncMock(
        return_value=_make_response_obj("{}")
    )
    return client


def _make_tool_response_obj() -> MagicMock:
    """A tool-free assistant turn, enough for chat_with_tools to parse."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "done"
    response.choices[0].message.tool_calls = []
    response.choices[0].finish_reason = "stop"
    return response


def _sent_kwargs(client: LLMClient) -> dict:
    return client._client.chat.completions.create.await_args.kwargs


_REASONING_OFF = {"reasoning": {"enabled": False}}


class TestReasoningSuppression:
    """`reasoning: "disabled"` reaches every OpenAI-compatible call."""

    @pytest.mark.asyncio
    async def test_generate_sends_extension_when_disabled(self):
        client = _reasoning_client("disabled")

        await client.generate("system", "user")

        assert _sent_kwargs(client)["extra_body"] == _REASONING_OFF

    @pytest.mark.asyncio
    async def test_generate_omits_extension_under_auto(self):
        """"auto" is the escape hatch for providers that reject it."""
        client = _reasoning_client("auto")

        await client.generate("system", "user")

        assert "extra_body" not in _sent_kwargs(client)

    @pytest.mark.asyncio
    async def test_generate_json_sends_extension_when_disabled(self):
        client = _reasoning_client("disabled")

        await client.generate_json("system", "user")

        assert _sent_kwargs(client)["extra_body"] == _REASONING_OFF

    @pytest.mark.asyncio
    async def test_chat_with_tools_sends_extension_when_disabled(self):
        client = _reasoning_client("disabled")
        client._client.chat.completions.create = AsyncMock(
            return_value=_make_tool_response_obj()
        )

        await client.chat_with_tools([{"role": "user", "content": "hi"}])

        assert _sent_kwargs(client)["extra_body"] == _REASONING_OFF

    @pytest.mark.asyncio
    async def test_stream_sends_extension_when_disabled(self):
        client = _reasoning_client("disabled")
        client._client.chat.completions.create = AsyncMock(
            return_value=_async_iter([])
        )

        async for _ in client.generate_stream("system", "user"):
            pass

        assert _sent_kwargs(client)["extra_body"] == _REASONING_OFF

    @pytest.mark.asyncio
    async def test_stream_omits_extension_under_auto(self):
        client = _reasoning_client("auto")
        client._client.chat.completions.create = AsyncMock(
            return_value=_async_iter([])
        )

        async for _ in client.generate_stream("system", "user"):
            pass

        assert "extra_body" not in _sent_kwargs(client)

    @pytest.mark.asyncio
    async def test_vision_sends_extension_when_disabled(self):
        client = _reasoning_client("disabled")

        await client.generate_vision(b"\x89PNG", "image/png", "system", "user")

        assert _sent_kwargs(client)["extra_body"] == _REASONING_OFF

    @pytest.mark.asyncio
    async def test_a_rejecting_provider_is_retried_without_the_extension(self):
        """OpenAI 400s on an unknown body field; that must not kill it."""
        client = _reasoning_client("disabled")
        good = _make_classified_response("done")
        client._client.chat.completions.create = AsyncMock(
            side_effect=[_make_status_error(400), good, good]
        )

        first = await client.generate("system", "user")
        await client.generate("system", "user")

        calls = client._client.chat.completions.create.await_args_list
        assert first == "done"
        assert "extra_body" in calls[0].kwargs
        assert "extra_body" not in calls[1].kwargs
        # The rejection is remembered, so the cost is one request total.
        assert "extra_body" not in calls[2].kwargs

    @pytest.mark.asyncio
    async def test_a_400_without_the_extension_still_fails(self):
        """The retry is for the field, not a way to ignore real 400s."""
        client = _reasoning_client("auto")
        client._client.chat.completions.create = AsyncMock(
            side_effect=_make_status_error(400)
        )

        assert await client.generate("system", "user") is None
        assert client._client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_a_rejected_stream_is_reopened_without_the_extension(self):
        client = _reasoning_client("disabled")
        client._client.chat.completions.create = AsyncMock(
            side_effect=[
                _make_status_error(400),
                _async_iter([_make_stream_chunk("hello")]),
            ]
        )

        deltas = [d async for d in client.generate_stream("system", "user")]

        calls = client._client.chat.completions.create.await_args_list
        assert deltas == ["hello"]
        assert "extra_body" in calls[0].kwargs
        assert "extra_body" not in calls[1].kwargs

    @pytest.mark.asyncio
    async def test_a_stream_400_without_the_extension_yields_nothing(self):
        client = _reasoning_client("auto")
        client._client.chat.completions.create = AsyncMock(
            side_effect=_make_status_error(400)
        )

        deltas = [d async for d in client.generate_stream("system", "user")]

        assert deltas == []
        assert client._client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_tool_calls_survive_a_rejected_extension(self):
        client = _reasoning_client("disabled")
        client._client.chat.completions.create = AsyncMock(
            side_effect=[_make_status_error(400), _make_tool_response_obj()]
        )

        result = await client.chat_with_tools(
            [{"role": "user", "content": "hi"}]
        )

        calls = client._client.chat.completions.create.await_args_list
        assert result is not None
        assert result.content == "done"
        assert "extra_body" in calls[0].kwargs
        assert "extra_body" not in calls[1].kwargs

    @pytest.mark.asyncio
    async def test_the_request_the_provider_sent_is_left_alone(self):
        """Dropping the field builds a new body; it does not edit one."""
        client = _reasoning_client("disabled")
        sent: list[dict] = []

        async def capture(**kwargs):
            sent.append(kwargs)
            if len(sent) == 1:
                raise _make_status_error(400)
            return _make_classified_response("done")

        client._client.chat.completions.create = AsyncMock(side_effect=capture)

        await client.generate("system", "user")

        # The first dict is the one handed to the provider; a helper that
        # mutated its argument would have emptied it retroactively.
        assert sent[0]["extra_body"] == {"reasoning": {"enabled": False}}
        assert "extra_body" not in sent[1]

    def test_ollama_body_is_unchanged_by_the_knob(self):
        """Ollama always sends think: false; the extension is OpenAI-only."""
        client = create_llm_client(
            LLMConfig(
                provider="ollama",
                base_url="http://localhost:11434",
                model="llama3",
                reasoning="disabled",
            )
        )

        body = client._build_body(
            "sys",
            "usr",
            stream=False,
            temperature=0.1,
            max_tokens=128,
            response_format=None,
        )

        assert body["think"] is False
        assert "reasoning" not in body
        assert "extra_body" not in body


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def _make_classified_response(
    content: str | None,
    *,
    finish_reason: str = "stop",
    completion_tokens: int = 10,
    reasoning_tokens: int = 0,
) -> MagicMock:
    """A response carrying the fields the classifier reads."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].finish_reason = finish_reason
    response.usage.completion_tokens = completion_tokens
    response.usage.completion_tokens_details.reasoning_tokens = reasoning_tokens
    return response


def _classifying_client(response: MagicMock) -> LLMClient:
    client = _reasoning_client("auto")
    client._client.chat.completions.create = AsyncMock(return_value=response)
    return client


class TestJsonFailureClassification:
    """A silent None is the bug; every failure has to name itself."""

    @pytest.mark.asyncio
    async def test_budget_spent_on_reasoning_is_named(self, caplog):
        client = _classifying_client(
            _make_classified_response(
                None,
                finish_reason="length",
                completion_tokens=2048,
                reasoning_tokens=2250,
            )
        )

        with caplog.at_level(logging.WARNING):
            result = await client.generate_json_result("system", "user")

        assert result.value is None
        assert result.failure == "token_budget"
        assert "length" in caplog.text
        assert "2250" in caplog.text

    @pytest.mark.asyncio
    async def test_empty_content_without_length_is_empty(self):
        client = _classifying_client(
            _make_classified_response(None, finish_reason="stop")
        )

        result = await client.generate_json_result("system", "user")

        assert result.failure == "empty"

    @pytest.mark.asyncio
    async def test_unparseable_content_is_malformed(self):
        client = _classifying_client(
            _make_classified_response("not json at all")
        )

        result = await client.generate_json_result("system", "user")

        assert result.value is None
        assert result.failure == "malformed"

    @pytest.mark.asyncio
    async def test_truncated_json_reports_the_budget_not_the_syntax(self):
        """The remedy is a budget change, so the cause outranks the symptom."""
        client = _classifying_client(
            _make_classified_response(
                '{"chapters": [{"start_time": 0.24, "end_',
                finish_reason="length",
                completion_tokens=2048,
                reasoning_tokens=1900,
            )
        )

        result = await client.generate_json_result("system", "user")

        assert result.value is None
        assert result.failure == "token_budget"

    @pytest.mark.asyncio
    async def test_good_response_carries_no_failure(self):
        client = _classifying_client(
            _make_classified_response('{"chapters": []}')
        )

        result = await client.generate_json_result("system", "user")

        assert result.value == {"chapters": []}
        assert result.failure is None

    @pytest.mark.asyncio
    async def test_request_failure_is_named(self):
        client = _reasoning_client("auto")
        client._client.chat.completions.create = AsyncMock(
            side_effect=_make_status_error(401)
        )

        result = await client.generate_json_result("system", "user")

        assert result.value is None
        assert result.failure == "request_failed"

    @pytest.mark.asyncio
    async def test_missing_usage_fields_do_not_crash_the_classifier(self):
        """Providers omit usage details; absence must not raise."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = None
        response.choices[0].finish_reason = "length"
        response.usage = None
        client = _classifying_client(response)

        result = await client.generate_json_result("system", "user")

        assert result.failure == "token_budget"


class TestClassificationHonesty:
    """A classification that overstates its evidence misdirects the fix."""

    @pytest.mark.asyncio
    async def test_truncated_text_is_not_reported_as_absent(self, caplog):
        """generate() hands the text back, so the log must not deny it."""
        client = _classifying_client(
            _make_classified_response(
                "A summary that ran out of ro", finish_reason="length"
            )
        )

        with caplog.at_level(logging.WARNING):
            text = await client.generate("system", "user")

        assert text == "A summary that ran out of ro"
        assert "truncated" in caplog.text.lower()
        assert "no usable output" not in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_reasoning_without_truncation_is_merely_empty(self):
        """Thinking is not proof the budget ran out; finish_reason is."""
        client = _classifying_client(
            _make_classified_response(
                None,
                finish_reason="stop",
                completion_tokens=500,
                reasoning_tokens=500,
            )
        )

        result = await client.generate_json_result("system", "user")

        assert result.failure == "empty"


class TestGenerateJsonDelegation:
    """The existing entry points keep their shape."""

    @pytest.mark.asyncio
    async def test_generate_json_returns_the_value_only(self):
        client = _classifying_client(
            _make_classified_response('{"ok": true}')
        )

        assert await client.generate_json("system", "user") == {"ok": True}

    @pytest.mark.asyncio
    async def test_generate_json_returns_none_on_failure(self):
        client = _classifying_client(
            _make_classified_response(None, finish_reason="length")
        )

        assert await client.generate_json("system", "user") is None

    @pytest.mark.asyncio
    async def test_generate_still_returns_truncated_text(self):
        """Classification observes; it does not withhold what came back."""
        client = _classifying_client(
            _make_classified_response("half an ans", finish_reason="length")
        )

        assert await client.generate("system", "user") == "half an ans"


class TestOllamaJsonFailureClassification:
    """The ollama client exposes the same result shape."""

    def _client(self, handler) -> OllamaLLMClient:
        client = create_llm_client(
            LLMConfig(
                provider="ollama",
                base_url="http://localhost:11434",
                model="llama3",
                retry_attempts=0,
            )
        )
        client._http = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        return client

    def _responds(self, content: str, done_reason: str = "stop"):
        return lambda request: httpx.Response(
            200,
            json={
                "message": {"content": content},
                "done_reason": done_reason,
            },
        )

    @pytest.mark.asyncio
    async def test_unparseable_content_is_malformed(self):
        client = self._client(self._responds("not json"))

        result = await client.generate_json_result("system", "user")

        assert result.value is None
        assert result.failure == "malformed"

    @pytest.mark.asyncio
    async def test_request_failure_is_named(self):
        def refuse(request):
            raise httpx.ConnectError("refused", request=request)

        client = self._client(refuse)

        result = await client.generate_json_result("system", "user")

        assert result.value is None
        assert result.failure == "request_failed"

    @pytest.mark.asyncio
    async def test_num_predict_truncation_is_a_budget_failure(self):
        """think: false stops reasoning, not an over-long answer."""
        client = self._client(
            self._responds('{"chapters": [{"start_', done_reason="length")
        )

        result = await client.generate_json_result("system", "user")

        assert result.value is None
        assert result.failure == "token_budget"

    @pytest.mark.asyncio
    async def test_complete_response_carries_no_failure(self):
        client = self._client(self._responds('{"ok": true}'))

        result = await client.generate_json_result("system", "user")

        assert result.value == {"ok": True}
        assert result.failure is None
