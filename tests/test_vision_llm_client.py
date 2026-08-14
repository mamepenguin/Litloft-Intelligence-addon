"""Vision LLM client tests.

Both ``LLMClient`` (OpenAI-compatible) and ``OllamaLLMClient`` must grow
a ``generate_vision`` coroutine:

.. code-block:: python

    async def generate_vision(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        output_language: str = "auto",
    ) -> str | None | object:
        ...

Contract:

* Returns the description text on success.
* Returns ``None`` on transient errors (timeouts, rate limits).
* Returns the sentinel ``VISION_UNSUPPORTED`` when the provider signals
  it cannot handle vision content (HTTP 400/404 after a vision payload,
  "does not support images" upstream error).
* Uses ``config.vision_model`` (NOT ``config.model``) and the vision-
  specific max_tokens / temperature.
* ``openai_compatible`` path builds ``messages[].content`` as a list
  containing ``{"type": "image_url", "image_url": {"url":
  "data:{mime};base64,..."}}`` and a ``{"type": "text", "text": ...}``
  block.
* ``ollama`` path uses the native ``/api/chat`` format with
  ``messages[].images`` as a list of base64 strings and the prompt as
  plain string content.
* System prompt is English per spec (keeps non-English models stable).
  Only ``output_language`` controls output-language of the description.
"""

from __future__ import annotations

import base64
import sys
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.config import LLMConfig  # noqa: E402
from app.llm import LLMClient, OllamaLLMClient  # noqa: E402


# Implementation is expected to expose a module-level sentinel so
# callers don't have to inspect status strings.
try:
    from app.llm import VISION_UNSUPPORTED  # noqa: F401
except ImportError:
    VISION_UNSUPPORTED = "__vision_unsupported__"  # placeholder for RED phase


_TINY_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


def _make_response_obj(text: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    return response


def _make_openai_client(**overrides) -> LLMClient:
    defaults = dict(
        provider="openai_compatible",
        base_url="http://localhost:11434/v1",
        model="gemma2:27b",
        max_tokens=2048,
        temperature=0.3,
        vision_model="llava:13b",
        vision_max_tokens=1024,
        vision_temperature=0.1,
        retry_attempts=0,
        retry_base_delay=0.01,
        retry_max_delay=0.05,
    )
    config = LLMConfig(**{**defaults, **overrides})
    return LLMClient(config)


def _make_ollama_client(**overrides) -> OllamaLLMClient:
    defaults = dict(
        provider="ollama",
        base_url="http://localhost:11434",
        model="gemma2:27b",
        vision_model="llava:13b",
        vision_max_tokens=1024,
        vision_temperature=0.1,
        retry_attempts=0,
        retry_base_delay=0.01,
        retry_max_delay=0.05,
    )
    config = LLMConfig(**{**defaults, **overrides})
    return OllamaLLMClient(config)


# ---------------------------------------------------------------------------
# LLMClient.generate_vision (openai_compatible)
# ---------------------------------------------------------------------------


class TestOpenAICompatibleGenerateVision:
    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self):
        """provider=disabled or missing vision_model → None."""
        client = _make_openai_client(vision_model="")
        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_builds_data_url_image_content(self):
        """The request messages must carry a data-URL image_url part."""
        client = _make_openai_client()

        captured: dict = {}

        async def fake_create(**kwargs):
            captured.update(kwargs)
            return _make_response_obj("A red apple on a table.")

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(side_effect=fake_create)

        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )

        assert result == "A red apple on a table."

        # User message content must be a list including an image_url part.
        messages = captured["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        content = user_msg["content"]
        assert isinstance(content, list)
        image_part = next(
            p for p in content if p.get("type") == "image_url"
        )
        url = image_part["image_url"]["url"]
        expected_b64 = base64.b64encode(_TINY_JPEG).decode("ascii")
        assert url == f"data:image/jpeg;base64,{expected_b64}"

    @pytest.mark.asyncio
    async def test_uses_vision_model_not_text_model(self):
        """The vision request uses config.vision_model, not config.model."""
        client = _make_openai_client(
            model="gemma2:27b", vision_model="llava:13b"
        )

        captured: dict = {}

        async def fake_create(**kwargs):
            captured.update(kwargs)
            return _make_response_obj("ok")

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(side_effect=fake_create)

        await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )

        assert captured["model"] == "llava:13b"

    @pytest.mark.asyncio
    async def test_uses_vision_max_tokens_and_temperature(self):
        client = _make_openai_client(
            vision_max_tokens=512, vision_temperature=0.0,
            max_tokens=2048, temperature=0.9,
        )

        captured: dict = {}

        async def fake_create(**kwargs):
            captured.update(kwargs)
            return _make_response_obj("ok")

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(side_effect=fake_create)

        await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )

        # Either legacy or new name depending on model; accept both.
        tokens = captured.get("max_tokens", captured.get("max_completion_tokens"))
        assert tokens == 512
        assert captured["temperature"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_system_prompt_is_english(self):
        """Spec: system prompt is English regardless of output_language."""
        client = _make_openai_client()

        captured: dict = {}

        async def fake_create(**kwargs):
            captured.update(kwargs)
            return _make_response_obj("ok")

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(side_effect=fake_create)

        await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image.",
            output_language="ja",
        )

        messages = captured["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        system_text = system_msg["content"]
        if isinstance(system_text, list):
            # Some impls put system as a list of parts; concat text parts.
            system_text = "".join(
                p.get("text", "") for p in system_text if p.get("type") == "text"
            )
        # Required English phrasing from the spec.
        lowered = system_text.lower()
        assert "describe" in lowered
        assert "detail" in lowered or "detailed" in lowered
        assert "speculate" in lowered or "speculat" in lowered
        assert "invent" in lowered or "quantit" in lowered

    @pytest.mark.asyncio
    async def test_output_language_is_substituted_into_prompt(self):
        """{output_language} placeholder must be filled in the system prompt."""
        client = _make_openai_client()

        captured: dict = {}

        async def fake_create(**kwargs):
            captured.update(kwargs)
            return _make_response_obj("ok")

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(side_effect=fake_create)

        await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image.",
            output_language="ja",
        )

        system_text = captured["messages"][0]["content"]
        if isinstance(system_text, list):
            system_text = "".join(
                p.get("text", "") for p in system_text if p.get("type") == "text"
            )
        # The configured value is interpreted as a standard language tag;
        # no feature-specific Japanese/English hard-code is needed.
        assert 'BCP 47 language tag "ja"' in system_text
        assert "Do not choose a different output language" in system_text
        assert "{language_requirement}" not in system_text

    @pytest.mark.asyncio
    async def test_returns_unsupported_sentinel_on_400(self):
        """Vision-incapable providers answer 400 → sentinel, not None."""
        client = _make_openai_client()
        request = httpx.Request("POST", "http://test/chat/completions")
        response = httpx.Response(
            status_code=400,
            request=request,
            content=b'{"error": {"message": "images not supported"}}',
        )
        from openai import APIStatusError

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=APIStatusError(
                message="HTTP 400", response=response, body=None,
            )
        )

        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )

        # The caller persists status = "unsupported" to avoid wasteful
        # retries. Don't collapse to None (which would invite retry).
        assert result == VISION_UNSUPPORTED

    @pytest.mark.asyncio
    async def test_returns_unsupported_sentinel_on_404(self):
        """Some ollama /v1 deployments return 404 for missing vision model."""
        client = _make_openai_client()
        request = httpx.Request("POST", "http://test/chat/completions")
        response = httpx.Response(
            status_code=404,
            request=request,
            content=b'{"error": {"message": "model not found"}}',
        )
        from openai import APIStatusError

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=APIStatusError(
                message="HTTP 404", response=response, body=None,
            )
        )

        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert result == VISION_UNSUPPORTED

    @pytest.mark.asyncio
    async def test_returns_none_on_transient_500(self):
        """5xx is transient; caller marks status=failed and retries later."""
        from openai import InternalServerError

        client = _make_openai_client()
        request = httpx.Request("POST", "http://test/chat/completions")
        response = httpx.Response(
            status_code=500,
            request=request,
            content=b'{"error": {"message": "boom"}}',
        )
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=InternalServerError(
                message="boom", response=response, body=None,
            )
        )

        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert result is None


class TestOpenAICompatibleGenerateVideoSceneJson:
    @pytest.mark.asyncio
    async def test_retries_without_json_mode_when_provider_rejects_it(self):
        from openai import APIStatusError

        client = _make_openai_client()
        request = httpx.Request("POST", "http://test/chat/completions")
        response = httpx.Response(
            status_code=400,
            request=request,
            content=b'{"error": {"message": "response_format unsupported"}}',
        )
        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=[
                APIStatusError(message="HTTP 400", response=response, body=None),
                _make_response_obj(
                    '{"scene_label":"Architecture diagram","visible_text":"","scene_type":"slide"}'
                ),
            ]
        )

        result = await client.generate_video_scene_json(
            _TINY_JPEG, "image/jpeg", "system", "user",
        )

        assert result["scene_label"] == "Architecture diagram"
        first, second = client._client.chat.completions.create.await_args_list
        assert first.kwargs["response_format"] == {"type": "json_object"}
        assert "response_format" not in second.kwargs


# ---------------------------------------------------------------------------
# OllamaLLMClient.generate_vision (native /api/chat with images: [base64...])
# ---------------------------------------------------------------------------


class TestOllamaGenerateVision:
    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self):
        client = _make_ollama_client(vision_model="")
        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_posts_to_api_chat_with_images_field(self):
        """Ollama native format: images is a list of base64 strings, NOT data URLs."""
        client = _make_ollama_client()

        captured: dict = {}

        async def fake_post(url, json):
            captured["url"] = url
            captured["json"] = json
            return httpx.Response(
                status_code=200,
                content=b'{"message": {"content": "A dog on grass."}}',
            )

        client._http.post = fake_post

        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )

        assert result == "A dog on grass."
        assert captured["url"] == "http://localhost:11434/api/chat"
        assert captured["json"]["model"] == "llava:13b"
        assert captured["json"]["stream"] is False

        messages = captured["json"]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        images = user_msg.get("images")
        assert isinstance(images, list) and len(images) == 1
        expected_b64 = base64.b64encode(_TINY_JPEG).decode("ascii")
        # Ollama wants raw base64, no data-URL prefix.
        assert images[0] == expected_b64
        assert "data:" not in images[0]

    @pytest.mark.asyncio
    async def test_system_prompt_is_english(self):
        client = _make_ollama_client()

        captured: dict = {}

        async def fake_post(url, json):
            captured["json"] = json
            return httpx.Response(
                status_code=200,
                content=b'{"message": {"content": "ok"}}',
            )

        client._http.post = fake_post

        await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image.",
            output_language="ja",
        )

        messages = captured["json"]["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        lowered = system_msg["content"].lower()
        assert "describe" in lowered
        assert "speculate" in lowered or "speculat" in lowered

    @pytest.mark.asyncio
    async def test_output_language_embedded_in_prompt(self):
        client = _make_ollama_client()

        captured: dict = {}

        async def fake_post(url, json):
            captured["json"] = json
            return httpx.Response(
                status_code=200,
                content=b'{"message": {"content": "ok"}}',
            )

        client._http.post = fake_post

        await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image.",
            output_language="en",
        )

        messages = captured["json"]["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert 'BCP 47 language tag "en"' in system_msg["content"]
        assert "Do not choose a different output language" in system_msg["content"]
        assert "{language_requirement}" not in system_msg["content"]

    @pytest.mark.asyncio
    async def test_unsupported_sentinel_on_400(self):
        client = _make_ollama_client()

        async def fake_post(url, json):
            return httpx.Response(
                status_code=400,
                content=b'{"error": "model does not support images"}',
            )

        client._http.post = fake_post

        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert result == VISION_UNSUPPORTED

    @pytest.mark.asyncio
    async def test_unsupported_sentinel_on_404_for_missing_model(self):
        client = _make_ollama_client()

        async def fake_post(url, json):
            return httpx.Response(
                status_code=404,
                content=b'{"error": "model \\"llava:13b\\" not found"}',
            )

        client._http.post = fake_post

        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert result == VISION_UNSUPPORTED

    @pytest.mark.asyncio
    async def test_empty_content_returns_none(self):
        """Blank response → None so the worker marks status=failed."""
        client = _make_ollama_client()

        async def fake_post(url, json):
            return httpx.Response(
                status_code=200,
                content=b'{"message": {"content": ""}}',
            )

        client._http.post = fake_post

        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_uses_vision_temperature_via_options(self):
        """Ollama carries temperature in the 'options' dict."""
        client = _make_ollama_client(
            vision_temperature=0.0, vision_max_tokens=256,
        )

        captured: dict = {}

        async def fake_post(url, json):
            captured["json"] = json
            return httpx.Response(
                status_code=200,
                content=b'{"message": {"content": "ok"}}',
            )

        client._http.post = fake_post

        await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )

        options = captured["json"].get("options", {})
        # Temperature propagates; num_predict is Ollama's max_tokens analog.
        assert options.get("temperature") == pytest.approx(0.0)
        assert options.get("num_predict") == 256


class TestOllamaGenerateVideoSceneJson:
    @pytest.mark.asyncio
    async def test_retries_without_json_format_when_provider_rejects_it(self):
        client = _make_ollama_client()
        bodies: list[dict] = []

        async def fake_post(url, json):
            bodies.append(json)
            if len(bodies) == 1:
                return httpx.Response(
                    status_code=400,
                    content=b'{"error": "format unsupported"}',
                )
            return httpx.Response(
                status_code=200,
                content=(
                    b'{"message":{"content":"{\\"scene_label\\":'
                    b'\\"Architecture diagram\\",\\"visible_text\\":\\"\\",'
                    b'\\"scene_type\\":\\"slide\\"}"}}'
                ),
            )

        client._http.post = fake_post

        result = await client.generate_video_scene_json(
            _TINY_JPEG, "image/jpeg", "system", "user",
        )

        assert result["scene_label"] == "Architecture diagram"
        assert bodies[0]["format"] == "json"
        assert "format" not in bodies[1]
