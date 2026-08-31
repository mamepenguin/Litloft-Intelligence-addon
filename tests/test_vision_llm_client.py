"""Vision LLM client tests.

Both ``LLMClient`` (OpenAI-compatible) and ``OllamaLLMClient`` expose a
``generate_vision`` coroutine:

.. code-block:: python

    async def generate_vision(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        output_language: str = "auto",
    ) -> VisionGeneration:
        ...

Contract:

* Returns ``VisionGeneration(text, failure)`` — the description and, when
  the call produced nothing usable, why.
* ``failure`` is ``FAILURE_REQUEST_FAILED`` for transient errors
  (timeouts, rate limits, 5xx) and ``FAILURE_EMPTY`` for a blank answer.
* A provider rejection (400/404) is not a verdict on its own. It is
  referred to the capability probe, which sends a reference image to the
  same model and reports ``FAILURE_VISION_UNSUPPORTED`` (the model will
  not take images), ``FAILURE_MODEL_MISSING`` (the model is absent), or
  ``FAILURE_IMAGE_REJECTED`` (the model reads the reference, so this
  request's image was the problem).
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
from app.llm import (  # noqa: E402
    _PROBE_CONFIRMATIONS,
    _PROBE_IMAGE_B64,
    FAILURE_EMPTY,
    JsonGeneration,
    FAILURE_IMAGE_REJECTED,
    FAILURE_MODEL_MISSING,
    FAILURE_REQUEST_FAILED,
    FAILURE_VISION_REJECTED,
    FAILURE_VISION_UNSUPPORTED,
    LLMClient,
    OllamaLLMClient,
    VisionGeneration,
    reset_vision_capability_cache,
)


_TINY_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


@pytest.fixture(autouse=True)
def _clean_capability_cache():
    """A verdict cached by one test must not answer another's probe."""
    reset_vision_capability_cache()
    yield
    reset_vision_capability_cache()


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
    async def test_reports_request_failed_when_disabled(self):
        """provider=disabled or missing vision_model → nothing was sent."""
        client = _make_openai_client(vision_model="")
        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert result.text is None
        assert result.failure == FAILURE_REQUEST_FAILED

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

        assert result.text == "A red apple on a table."
        assert result.failure is None

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
        # Reasoning suppression is on by default, so the first 400 is
        # read as the provider refusing that field and costs one re-send.
        # A 400 that survives without it goes to the probe, which is
        # rejected too — so the model, not the image, is the cause.
        client._client.chat.completions.create = AsyncMock(
            side_effect=[
                APIStatusError(
                    message="HTTP 400", response=response, body=None,
                ),
                APIStatusError(
                    message="HTTP 400", response=response, body=None,
                ),
                # Two probes: the only permanent verdict has to be
                # rejected twice before it is believed.
                APIStatusError(
                    message="HTTP 400", response=response, body=None,
                ),
                APIStatusError(
                    message="HTTP 400", response=response, body=None,
                ),
            ]
        )

        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )

        # The caller persists status = "unsupported"; this is the one
        # verdict it is allowed to latch.
        assert result.text is None
        assert result.failure == FAILURE_VISION_UNSUPPORTED
        first, second, probe, _confirm = (
            client._client.chat.completions.create.await_args_list
        )
        assert "extra_body" in first.kwargs
        assert "extra_body" not in second.kwargs
        # The probe carries its own reference image, not the caller's.
        probe_content = probe.kwargs["messages"][0]["content"]
        probe_url = next(
            p for p in probe_content if p.get("type") == "image_url"
        )["image_url"]["url"]
        assert probe_url.startswith("data:image/png;base64,")
        assert base64.b64encode(_TINY_JPEG).decode("ascii") not in probe_url

    @pytest.mark.asyncio
    async def test_rejected_image_is_not_blamed_on_the_model(self):
        """A capable model refusing one image must stay retryable.

        Measured 2026-08-31: ollama answers 400 "Failed to load image or
        audio file" for a corrupt PNG sent to a vision-capable model —
        the same status a text-only model answers.
        """
        client = _make_openai_client()
        request = httpx.Request("POST", "http://test/chat/completions")
        response = httpx.Response(
            status_code=400,
            request=request,
            content=b'{"error": {"message": "Failed to load image"}}',
        )
        from openai import APIStatusError

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=[
                APIStatusError(message="HTTP 400", response=response, body=None),
                APIStatusError(message="HTTP 400", response=response, body=None),
                # The probe's reference image is taken, so the model sees.
                _make_response_obj("Red"),
            ]
        )

        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert result.failure == FAILURE_IMAGE_REJECTED

    @pytest.mark.asyncio
    async def test_probe_verdict_ignores_an_empty_body(self):
        """A 2xx is capability, whatever the body says.

        Measured 2026-08-31: gemma4:e4b answers 200 with empty content
        over the OpenAI-compatible transport and "Red" over ollama's
        native one. Reading the body would misjudge the first.
        """
        client = _make_openai_client()
        request = httpx.Request("POST", "http://test/chat/completions")
        response = httpx.Response(
            status_code=400,
            request=request,
            content=b'{"error": {"message": "bad image"}}',
        )
        from openai import APIStatusError

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=[
                APIStatusError(message="HTTP 400", response=response, body=None),
                APIStatusError(message="HTTP 400", response=response, body=None),
                _make_response_obj(""),
            ]
        )

        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert result.failure == FAILURE_IMAGE_REJECTED

    @pytest.mark.asyncio
    async def test_404_is_a_missing_model_not_an_incapable_one(self):
        """An absent model clears when the operator pulls it."""
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
        assert result.failure == FAILURE_MODEL_MISSING

    @pytest.mark.asyncio
    async def test_a_404_needs_no_probe_and_outranks_a_cached_verdict(self):
        """The model went away after it had been probed as capable.

        Nothing about the earlier verdict survives the model's removal,
        so answering the 404 from cache would blame the image for the
        model's absence and send the operator looking at the file.
        """
        client = _make_openai_client()
        request = httpx.Request("POST", "http://test/chat/completions")
        bad_image = httpx.Response(
            status_code=400, request=request, content=b"{}",
        )
        gone = httpx.Response(
            status_code=404, request=request, content=b"{}",
        )
        from openai import APIStatusError

        probe_calls = 0
        model_present = True

        async def fake_create(**kwargs):
            nonlocal probe_calls
            if len(kwargs["messages"]) == 1:
                probe_calls += 1
                if model_present:
                    return _make_response_obj("Red")
                raise APIStatusError(
                    message="HTTP 404", response=gone, body=None,
                )
            raise APIStatusError(
                message="HTTP 400" if model_present else "HTTP 404",
                response=bad_image if model_present else gone,
                body=None,
            )

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=fake_create
        )

        first = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert first.failure == FAILURE_IMAGE_REJECTED
        assert probe_calls == 1

        # The operator removes the model. Nothing is restarted.
        model_present = False
        second = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert second.failure == FAILURE_MODEL_MISSING
        # The 404 answered itself; no probe was needed for it.
        assert probe_calls == 1

    @pytest.mark.asyncio
    async def test_a_404_evicts_the_verdict_it_contradicts(self):
        """The 404 says the cached verdict was about a model that is gone.

        Leaving it would let "the model can see" answer for a model
        that is no longer installed, so a later rejection would be
        blamed on the image and the operator sent to look at the file.
        """
        client = _make_openai_client()
        request = httpx.Request("POST", "http://test/chat/completions")
        rejected = httpx.Response(
            status_code=400, request=request, content=b"{}",
        )
        gone = httpx.Response(
            status_code=404, request=request, content=b"{}",
        )
        from openai import APIStatusError

        # 1: rejected + a probe that reads the reference -> capable is
        # cached. 2: the model is removed. 3: it comes back, but this
        # time it cannot see.
        stage = 1

        async def fake_create(**kwargs):
            probing = len(kwargs["messages"]) == 1
            if stage == 2:
                raise APIStatusError(
                    message="HTTP 404", response=gone, body=None,
                )
            if probing and stage == 1:
                return _make_response_obj("Red")
            raise APIStatusError(
                message="HTTP 400", response=rejected, body=None,
            )

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=fake_create
        )

        first = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert first.failure == FAILURE_IMAGE_REJECTED

        stage = 2
        gone_result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert gone_result.failure == FAILURE_MODEL_MISSING

        stage = 3
        # Without eviction the stale "capable" answers here and this
        # comes back as image_rejected.
        after = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert after.failure == FAILURE_VISION_UNSUPPORTED

    @pytest.mark.asyncio
    async def test_one_bad_moment_does_not_condemn_a_capable_model(self):
        """The only permanent verdict must not rest on one request.

        A provider refusing once under load would otherwise latch
        "cannot see" for the life of the process and stamp it onto
        every file that follows — while every other 400 in this module
        gets retried.
        """
        client = _make_openai_client()
        request = httpx.Request("POST", "http://test/chat/completions")
        response = httpx.Response(
            status_code=400, request=request, content=b"{}",
        )
        from openai import APIStatusError

        probe_calls = 0

        async def fake_create(**kwargs):
            nonlocal probe_calls
            if len(kwargs["messages"]) == 1:
                probe_calls += 1
                if probe_calls == 1:
                    # One bad moment.
                    raise APIStatusError(
                        message="HTTP 400", response=response, body=None,
                    )
                return _make_response_obj("Red")
            raise APIStatusError(
                message="HTTP 400", response=response, body=None,
            )

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=fake_create
        )

        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert result.failure == FAILURE_IMAGE_REJECTED
        assert probe_calls == 2

    @pytest.mark.asyncio
    async def test_probe_runs_once_per_model_and_not_on_success(self):
        """The probe is a failure-path cost, paid once per model."""
        client = _make_openai_client()
        request = httpx.Request("POST", "http://test/chat/completions")
        response = httpx.Response(
            status_code=400,
            request=request,
            content=b'{"error": {"message": "no"}}',
        )
        from openai import APIStatusError

        rejection = APIStatusError(
            message="HTTP 400", response=response, body=None,
        )
        probe_calls = 0

        async def fake_create(**kwargs):
            nonlocal probe_calls
            messages = kwargs["messages"]
            is_probe = (
                len(messages) == 1
                and any(
                    p.get("type") == "image_url"
                    and "png" in p["image_url"]["url"]
                    for p in messages[0]["content"]
                )
            )
            if is_probe:
                probe_calls += 1
            raise rejection

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=fake_create
        )

        for _ in range(3):
            result = await client.generate_vision(
                _TINY_JPEG, "image/jpeg", "Describe this image."
            )
            assert result.failure == FAILURE_VISION_UNSUPPORTED

        # One classification, confirmed once; the other two files are
        # answered from the cache.
        assert probe_calls == _PROBE_CONFIRMATIONS

    @pytest.mark.asyncio
    async def test_concurrent_failures_share_one_probe(self):
        """A bulk run must not launch a probe per file."""
        import asyncio

        client = _make_openai_client()
        request = httpx.Request("POST", "http://test/chat/completions")
        response = httpx.Response(
            status_code=400,
            request=request,
            content=b'{"error": {"message": "no"}}',
        )
        from openai import APIStatusError

        probe_calls = 0

        async def fake_create(**kwargs):
            nonlocal probe_calls
            messages = kwargs["messages"]
            if len(messages) == 1:
                probe_calls += 1
                # Hold the probe open so every caller queues behind it.
                await asyncio.sleep(0.05)
            raise APIStatusError(
                message="HTTP 400", response=response, body=None,
            )

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=fake_create
        )

        results = await asyncio.gather(*[
            client.generate_vision(
                _TINY_JPEG, "image/jpeg", "Describe this image."
            )
            for _ in range(5)
        ])

        assert all(r.failure == FAILURE_VISION_UNSUPPORTED for r in results)
        # Five files, one classification — confirmed once, not five times.
        assert probe_calls == _PROBE_CONFIRMATIONS

    @pytest.mark.asyncio
    async def test_an_undetermined_rejection_is_not_a_lost_request(self):
        """The request landed and was refused; only the reason is missing.

        Collapsing that into request_failed loses the fact that the
        provider rejected it, which is what the JSON-mode fallback keys
        on.
        """
        client = _make_openai_client()
        request = httpx.Request("POST", "http://test/chat/completions")
        response = httpx.Response(
            status_code=400, request=request, content=b"{}",
        )
        from openai import APIStatusError

        async def fake_create(**kwargs):
            if len(kwargs["messages"]) == 1:
                raise RuntimeError("probe could not be carried out")
            raise APIStatusError(
                message="HTTP 400", response=response, body=None,
            )

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=fake_create
        )

        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert result.failure == FAILURE_VISION_REJECTED

    @pytest.mark.asyncio
    async def test_failed_probe_is_not_cached(self):
        """A probe that could not be carried out says nothing about the model."""
        client = _make_openai_client()
        request = httpx.Request("POST", "http://test/chat/completions")
        response = httpx.Response(
            status_code=400,
            request=request,
            content=b'{"error": {"message": "no"}}',
        )
        from openai import APIStatusError

        probe_calls = 0

        async def fake_create(**kwargs):
            nonlocal probe_calls
            if len(kwargs["messages"]) == 1:
                probe_calls += 1
                raise RuntimeError("connection reset")
            raise APIStatusError(
                message="HTTP 400", response=response, body=None,
            )

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=fake_create
        )

        for _ in range(2):
            result = await client.generate_vision(
                _TINY_JPEG, "image/jpeg", "Describe this image."
            )
            # Rejected, cause undetermined — never cached, so the next
            # rejection probes again.
            assert result.failure == FAILURE_VISION_REJECTED

        assert probe_calls == 2

    @pytest.mark.asyncio
    async def test_a_missing_model_verdict_is_not_cached(self):
        """`ollama pull` clears the condition without a restart.

        Caching it would latch a self-healing condition — the same
        mistake, one layer up, that the classification exists to undo.
        """
        client = _make_openai_client()
        request = httpx.Request("POST", "http://test/chat/completions")
        gone = httpx.Response(
            status_code=404, request=request, content=b"{}",
        )
        rejected = httpx.Response(
            status_code=400, request=request, content=b"{}",
        )
        from openai import APIStatusError

        installed = False

        async def fake_create(**kwargs):
            if len(kwargs["messages"]) == 1:
                # The probe only runs once the model is back.
                return _make_response_obj("Red")
            raise APIStatusError(
                message="rejected",
                response=rejected if installed else gone,
                body=None,
            )

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=fake_create
        )

        first = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert first.failure == FAILURE_MODEL_MISSING

        # The operator pulls the model; nothing is restarted. The next
        # rejection must be classified afresh rather than answered from
        # the earlier verdict.
        installed = True
        second = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert second.failure == FAILURE_IMAGE_REJECTED

    @pytest.mark.asyncio
    async def test_transient_500_is_a_failed_request(self):
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
        assert result.text is None
        assert result.failure == FAILURE_REQUEST_FAILED


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
        # The first 400 is spent ruling out our own reasoning field. The
        # second goes to the probe, which the model answers — so json
        # mode, not the image or the model, is what was refused.
        client._client.chat.completions.create = AsyncMock(
            side_effect=[
                APIStatusError(message="HTTP 400", response=response, body=None),
                APIStatusError(message="HTTP 400", response=response, body=None),
                _make_response_obj("Red"),
                _make_response_obj(
                    '{"scene_label":"Architecture diagram","visible_text":"","scene_type":"slide"}'
                ),
            ]
        )

        result = await client.generate_video_scene_json(
            _TINY_JPEG, "image/jpeg", "system", "user",
        )

        assert result.value["scene_label"] == "Architecture diagram"
        assert result.failure is None
        calls = client._client.chat.completions.create.await_args_list
        assert calls[0].kwargs["response_format"] == {"type": "json_object"}
        assert "extra_body" in calls[0].kwargs
        assert "extra_body" not in calls[1].kwargs
        # calls[2] is the probe; the real retry drops response_format.
        assert "response_format" not in calls[3].kwargs


# ---------------------------------------------------------------------------
# OllamaLLMClient.generate_vision (native /api/chat with images: [base64...])
# ---------------------------------------------------------------------------


class TestOllamaGenerateVision:
    @pytest.mark.asyncio
    async def test_reports_request_failed_when_disabled(self):
        client = _make_ollama_client(vision_model="")
        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert result.text is None
        assert result.failure == FAILURE_REQUEST_FAILED

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

        assert result.text == "A dog on grass."
        assert result.failure is None
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
    async def test_400_goes_to_the_probe(self):
        """A rejected model is only unsupported once the probe agrees."""
        client = _make_ollama_client()

        async def fake_post(url, json, **kwargs):
            return httpx.Response(
                status_code=400,
                content=b'{"error": "model does not support images"}',
            )

        client._http.post = fake_post

        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert result.failure == FAILURE_VISION_UNSUPPORTED

    @pytest.mark.asyncio
    async def test_400_with_a_capable_model_blames_the_image(self):
        client = _make_ollama_client()
        seen: list[dict] = []

        async def fake_post(url, json, **kwargs):
            seen.append(json)
            if json["messages"][0].get("images") == [_PROBE_IMAGE_B64]:
                return httpx.Response(
                    status_code=200,
                    content=b'{"message": {"content": "Red"}}',
                )
            return httpx.Response(
                status_code=400,
                content=b'{"error": "Failed to load image or audio file"}',
            )

        client._http.post = fake_post

        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert result.failure == FAILURE_IMAGE_REJECTED
        # The probe sent its own reference image, not the caller's.
        assert seen[-1]["messages"][0]["images"] == [_PROBE_IMAGE_B64]

    @pytest.mark.asyncio
    async def test_404_is_a_missing_model(self):
        client = _make_ollama_client()

        async def fake_post(url, json, **kwargs):
            return httpx.Response(
                status_code=404,
                content=b'{"error": "model \\"llava:13b\\" not found"}',
            )

        client._http.post = fake_post

        result = await client.generate_vision(
            _TINY_JPEG, "image/jpeg", "Describe this image."
        )
        assert result.failure == FAILURE_MODEL_MISSING

    @pytest.mark.asyncio
    async def test_empty_content_is_reported_as_empty(self):
        """Blank response → the worker marks status=failed."""
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
        assert result.text is None
        assert result.failure == FAILURE_EMPTY

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

        async def fake_post(url, json, **kwargs):
            bodies.append(json)
            if len(bodies) == 1:
                return httpx.Response(
                    status_code=400,
                    content=b'{"error": "format unsupported"}',
                )
            if json["messages"][0].get("images") == [_PROBE_IMAGE_B64]:
                return httpx.Response(
                    status_code=200,
                    content=b'{"message": {"content": "Red"}}',
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

        assert result.value["scene_label"] == "Architecture diagram"
        assert result.failure is None
        assert bodies[0]["format"] == "json"
        # bodies[1] is the probe; the real retry drops the format field.
        assert "format" not in bodies[2]


# ---------------------------------------------------------------------------
# Regression detector: the vision path must stay on the failure taxonomy
# ---------------------------------------------------------------------------


class TestVisionClassificationCannotBeBypassed:
    """Guards the seam this module's whole contract rests on.

    The defect these tests exist for was structural, not a typo: the
    vision path returned a bare sentinel while every other generation
    path had moved to a classified result, so a rejection reached the
    worker with no way to ask what it meant. A future edit that returns
    a naked value again would silently restore that.
    """

    def test_no_bare_sentinel_survives(self):
        import app.llm as llm

        assert not hasattr(llm, "VISION_UNSUPPORTED"), (
            "a sentinel cannot carry a reason; return VisionGeneration"
        )

    @pytest.mark.parametrize(
        "cls,method,expected",
        [
            (LLMClient, "generate_vision", VisionGeneration),
            (LLMClient, "_vision_chat", VisionGeneration),
            (LLMClient, "generate_video_scene_json", JsonGeneration),
            (OllamaLLMClient, "generate_vision", VisionGeneration),
            (OllamaLLMClient, "_vision_chat", VisionGeneration),
            (OllamaLLMClient, "generate_video_scene_json", JsonGeneration),
        ],
    )
    def test_vision_entry_points_return_a_classified_result(
        self, cls, method, expected
    ):
        annotation = getattr(cls, method).__annotations__.get("return")
        assert annotation is expected, (
            f"{cls.__name__}.{method} returns {annotation!r}, not "
            f"{expected.__name__}; a caller cannot distinguish transient "
            f"from terminal"
        )


class TestProbeImageIsActuallyDecodable:
    """The reference image is the instrument every verdict rests on.

    Every other test in this module mocks the probe's response, so none
    of them would notice the literal itself going bad — and a corrupt
    reference makes the probe condemn capable models: the provider
    rejects the unreadable image with a 400, which is read as
    "this model will not take images" and cached for the process.

    Decoded here with the standard library rather than Pillow, which the
    suite replaces with a MagicMock.
    """

    def _chunks(self):
        import struct
        import zlib

        data = base64.b64decode(_PROBE_IMAGE_B64)
        assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
        out = []
        offset = 8
        while offset < len(data):
            (length,) = struct.unpack(">I", data[offset:offset + 4])
            tag = data[offset + 4:offset + 8]
            body = data[offset + 8:offset + 8 + length]
            (crc,) = struct.unpack(
                ">I", data[offset + 8 + length:offset + 12 + length]
            )
            assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF, (
                f"{tag!r} chunk fails its own checksum"
            )
            out.append((tag, body))
            offset += 12 + length
        assert offset == len(data), "trailing bytes after the last chunk"
        return out

    def test_chunk_structure_is_intact(self):
        tags = [tag for tag, _ in self._chunks()]
        assert tags[0] == b"IHDR"
        assert tags[-1] == b"IEND"
        assert b"IDAT" in tags

    def test_header_declares_a_64px_truecolour_image(self):
        import struct

        header = next(body for tag, body in self._chunks() if tag == b"IHDR")
        width, height, depth, colour = struct.unpack(">IIBB", header[:10])
        # Measured 2026-08-31: gemma4:e4b answers 200 at this size over
        # both transports. Smaller images still pass, but this is the
        # size the recorded measurements were taken at.
        assert (width, height) == (64, 64)
        assert (depth, colour) == (8, 2), "expected 8-bit truecolour RGB"

    def test_pixel_data_inflates_to_a_full_solid_image(self):
        import zlib

        payload = b"".join(
            body for tag, body in self._chunks() if tag == b"IDAT"
        )
        raw = zlib.decompress(payload)
        stride = 1 + 64 * 3
        assert len(raw) == 64 * stride, "scanlines are truncated"
        for row in range(64):
            line = raw[row * stride:(row + 1) * stride]
            assert line[0] == 0, "expected the None filter on every scanline"
            assert set(line[1:]) == {204, 51}, "expected one solid colour"



class TestStructuredVisionKeepsTheUpstreamReason:
    """Truncation can leave a parseable object the domain still rejects.

    Reporting that as malformed output sends the operator to the prompt
    when the remedy is the token budget. The parser hands over both the
    value and the reason and lets the domain validator decide.
    """

    def test_a_parsed_value_keeps_a_truncation_reason(self):
        from app.llm import (
            FAILURE_TOKEN_BUDGET,
            VisionGeneration,
            _vision_json_result,
        )

        result = _vision_json_result(
            VisionGeneration('{"visible_text":"x"}', FAILURE_TOKEN_BUDGET)
        )
        assert result.value == {"visible_text": "x"}
        assert result.failure == FAILURE_TOKEN_BUDGET

    def test_a_clean_parse_carries_no_reason(self):
        from app.llm import VisionGeneration, _vision_json_result

        result = _vision_json_result(
            VisionGeneration('{"scene_label":"A kitchen"}', None)
        )
        assert result.value == {"scene_label": "A kitchen"}
        assert result.failure is None

    def test_an_unparseable_body_reports_the_upstream_cause(self):
        from app.llm import (
            FAILURE_TOKEN_BUDGET,
            VisionGeneration,
            _vision_json_result,
        )

        result = _vision_json_result(
            VisionGeneration('{"scene_label":"A kit', FAILURE_TOKEN_BUDGET)
        )
        assert result.value is None
        assert result.failure == FAILURE_TOKEN_BUDGET



class TestJsonModeFallbackSurvivesAnUndeterminedProbe:
    """A provider whose only fault is lacking JSON mode still gets its retry.

    The fallback keys on the rejection having happened, not on the probe
    having reached a conclusion — otherwise a probe that times out takes
    the compatibility path down with it.
    """

    @pytest.mark.asyncio
    async def test_retry_without_response_format_still_happens(self):
        client = _make_openai_client()
        request = httpx.Request("POST", "http://test/chat/completions")
        rejected = httpx.Response(
            status_code=400, request=request, content=b"{}",
        )
        from openai import APIStatusError

        async def fake_create(**kwargs):
            if len(kwargs["messages"]) == 1:
                raise RuntimeError("probe timed out")
            if "response_format" in kwargs:
                raise APIStatusError(
                    message="no json mode", response=rejected, body=None,
                )
            return _make_response_obj('{"scene_label":"A kitchen"}')

        client._client = MagicMock()
        client._client.chat.completions.create = AsyncMock(
            side_effect=fake_create
        )

        result = await client.generate_video_scene_json(
            _TINY_JPEG, "image/jpeg", "system", "user",
        )
        assert result.value == {"scene_label": "A kitchen"}
        assert result.failure is None
