"""Tests for app.llm module.

Covers LLMClient initialization (enabled/disabled detection),
text generation with mocked AsyncOpenAI, and JSON parsing with
regex fallback.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import LLMConfig
from app.llm import LLMClient


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
