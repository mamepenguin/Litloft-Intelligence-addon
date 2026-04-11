"""Tests for app.llm module.

Covers LLMClient initialization (enabled/disabled detection),
text generation with mocked AsyncOpenAI, JSON parsing with
regex fallback, retry with exponential backoff, and rate limiting.
"""

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
from app.llm import LLMClient


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
        client.generate = AsyncMock(return_value='["tag1", "tag2", "tag3"]')

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
        client.generate = AsyncMock(return_value='{"key": "value"}')

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
        client.generate = AsyncMock(
            return_value='Here are some tags: ["tag1", "tag2"] hope that helps!'
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
        client.generate = AsyncMock(return_value="This is not JSON at all")

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
        client.generate = AsyncMock(return_value=None)

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
        client.generate = AsyncMock(
            return_value='Sure!\n[\n  "tag1",\n  "tag2"\n]\nDone.'
        )

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
        """400 Bad Request is permanent — no retry."""
        client = _make_client(retry_attempts=3)
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=_make_status_error(400)
        )

        result = await client.generate("system", "user")

        assert result is None
        assert client._client.chat.completions.create.await_count == 1

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
