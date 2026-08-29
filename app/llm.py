"""Async LLM clients with pluggable provider backends.

Two backends are available, selected via ``config.provider``:

* ``openai_compatible`` — ``LLMClient`` uses the official OpenAI SDK.
  Works with OpenAI, ollama's /v1 layer, vLLM, LM Studio, DeepSeek, etc.
* ``ollama`` — ``OllamaLLMClient`` uses ollama's native ``/api/chat``
  HTTP API via httpx, sending ``think: false`` to skip chain-of-thought
  reasoning. This eliminates the 10-20 second thinking delay that
  reasoning models (Gemma 4, DeepSeek-R1, QwQ) would otherwise incur.

Both classes expose the same interface (``generate``,
``generate_stream``, ``generate_json``, ``enabled``).

Use ``create_llm_client(config)`` to instantiate the correct backend.
"""

import asyncio
import base64
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from app.config import LLMConfig
from app.output_language import configured_language_requirement
from app.prompt_loader import render

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatToolCall:
    """One tool call surfaced by the LLM during an agentic turn."""

    id: str
    name: str
    arguments_raw: str  # Raw JSON string; the loop layer json.loads


@dataclass(frozen=True)
class ChatTurnResult:
    """Outcome of a single ``chat_with_tools`` round-trip.

    ``finish_reason`` mirrors OpenAI's vocabulary so the loop can branch
    deterministically:
      * ``"stop"`` → ``content`` carries the final answer.
      * ``"tool_calls"`` → ``tool_calls`` is non-empty; ``content`` may
        accompany them but is typically empty.
      * Any other value (``"length"``, ``"content_filter"`` …) is a
        non-fatal end-of-turn the loop should treat like ``"stop"``.
    """

    content: str | None
    tool_calls: tuple[ChatToolCall, ...]
    finish_reason: str

# Exceptions that indicate transient failures and should trigger a retry.
_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)

# HTTP status codes that indicate permanent failures (don't retry).
_PERMANENT_STATUS_CODES = frozenset({400, 401, 402, 403, 404})


# Model name prefixes that require ``max_completion_tokens`` instead of
# the legacy ``max_tokens`` parameter (OpenAI's next-gen families refuse
# to accept ``max_tokens`` since early 2025). Match is case-insensitive
# against the configured model string.
_MAX_COMPLETION_TOKENS_PREFIXES: tuple[str, ...] = (
    "gpt-5",
    "o1",
    "o3",
    "o4",
)


def _uses_max_completion_tokens(model: str) -> bool:
    """Decide which token-budget parameter name the target model accepts.

    Keeping the check on the client side lets us support OpenAI's new
    models without a config flag while leaving the many OpenAI-compatible
    backends (ollama, vLLM, LM Studio, older gpt-4 / gpt-3.5) on the
    legacy ``max_tokens`` parameter they already accept.
    """
    lowered = (model or "").lower()
    return any(lowered.startswith(p) for p in _MAX_COMPLETION_TOKENS_PREFIXES)


def _parse_json_response(raw: str) -> list | dict | None:
    """Parse a raw LLM response as JSON with regex fallback.

    Tries direct json.loads first. On failure, scans for a
    ``{…}`` object then ``[…]`` array with a greedy regex — this
    handles the common cases where the model wraps JSON in prose or
    a Markdown code fence. Returns None if nothing parses.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: try object first (dict), then array (list).
    # Object match is prioritized because array fallback was
    # historically the only option and we don't want to regress.
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError):
                continue

    preview = (raw[:40] + "…") if len(raw) > 40 else raw
    logger.warning(
        "Failed to parse LLM response as JSON (len=%d, head=%r)",
        len(raw), preview,
    )
    return None


# Why a generation produced nothing usable. The distinction matters
# because the remedies differ: a budget spent on chain-of-thought is
# fixed by ``llm.reasoning`` or a larger ``max_tokens``, while malformed
# output is a prompt or model-capability problem.
FAILURE_TOKEN_BUDGET = "token_budget"
FAILURE_MALFORMED = "malformed"
FAILURE_EMPTY = "empty"
FAILURE_REQUEST_FAILED = "request_failed"


@dataclass(frozen=True)
class TextGeneration:
    """A completion plus why it is unusable, if it is.

    ``text`` is whatever came back, truncation included — classification
    observes the response, it never withholds it from callers who can
    still make use of a partial answer.
    """

    text: str | None
    failure: str | None


@dataclass(frozen=True)
class JsonGeneration:
    """A parsed JSON payload plus why it is missing, if it is."""

    value: list | dict | None
    failure: str | None


def _usage_reasoning_tokens(response: object) -> int | None:
    """Reasoning-token count, or None when the provider omits it.

    Every field here is optional in practice: OpenAI-compatible backends
    differ on whether they report usage at all, and the SDK models leave
    absent fields as None.
    """
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    tokens = getattr(details, "reasoning_tokens", None)
    return tokens if isinstance(tokens, int) else None


def _classify_completion(
    content: str | None,
    finish_reason: object,
) -> str | None:
    """Name the reason a completion is unusable, or None if it is fine.

    ``finish_reason == "length"`` is the only evidence that the budget
    ran out. Reasoning tokens say where the budget went, not that it was
    exhausted, so a clean stop with no content is an empty answer rather
    than a budget failure — the two have different remedies.

    A response truncated by the ceiling is a budget failure even when
    some content arrived, because a structured answer cut mid-value is
    unusable to the caller that asked for one. Callers wanting prose can
    still use the partial text; ``TextGeneration`` hands it to them.
    """
    if finish_reason == "length":
        return FAILURE_TOKEN_BUDGET
    if content is not None and content.strip():
        return None
    return FAILURE_EMPTY


# Sentinel return value from ``generate_vision`` when the upstream model
# signals it cannot handle image content (HTTP 400 / 404 after a vision
# payload, "images not supported" errors). Callers use it to persist
# ``visual_description_status = "unsupported"`` and avoid wasteful retries
# against the same model. A distinct object (not ``None``) so the "empty
# response" and "not vision-capable" cases don't collide.
VISION_UNSUPPORTED: object = object()


# Vision status codes that mean "this provider/model can't do vision".
# Both are sticky: caller marks status=unsupported and won't retry with
# the same model. 5xx stays in the transient/retry path.
_VISION_UNSUPPORTED_STATUS_CODES = frozenset({400, 404})


def _build_vision_system_prompt(output_language: str) -> str:
    """Construct the English system prompt for vision description.

    The prompt is English for stability across multi-language models
    (matches the auto_tags / summaries convention). The configured
    output-language tag controls only the generated description.

    Spec: docs/superpowers/specs/2026-04-23-intelligence-vision-describe.md
    """
    return render(
        "vision/system.jinja2",
        language_requirement=configured_language_requirement(
            output_language,
            auto_requirement=(
                "Use the same language as the filename and existing tags, "
                "defaulting to English."
            ),
        ),
    )


class LLMClient:
    """Async LLM client wrapping AsyncOpenAI for OpenAI-compatible APIs."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._enabled = (
            config.provider != "disabled"
            and bool(config.base_url)
            and bool(config.model)
        )
        if self._enabled:
            # Explicit timeout (see LLMConfig docstring for rationale).
            # httpx.Timeout accepts a total timeout + a separate connect
            # timeout so slow TCP handshakes fail faster than slow body
            # reads. Falling back to the SDK default (~600s) on
            # non-finite values keeps tests that pass a MagicMock config
            # from crashing on initialization.
            timeout: httpx.Timeout | None
            total = config.request_timeout_seconds
            connect = config.request_connect_timeout_seconds
            if isinstance(total, (int, float)) and total > 0:
                timeout = httpx.Timeout(
                    float(total),
                    connect=(
                        float(connect)
                        if isinstance(connect, (int, float)) and connect > 0
                        else float(total)
                    ),
                )
            else:
                timeout = None
            self._client = AsyncOpenAI(
                base_url=config.base_url,
                api_key=config.api_key or "not-needed",
                timeout=timeout,
            )
        # Rate-limiting state (None = no previous request yet)
        self._rate_lock = asyncio.Lock()
        self._last_request_time: float | None = None
        # Latched once a provider answers 400 to our opt-in body fields.
        self._extras_rejected = False

    def _provider_extras(self) -> dict:
        """Provider-specific body fields, empty unless opted into.

        ``reasoning`` is an OpenRouter extension. OpenAI answers 400 to
        an unrecognised body field, so an operator who has not asked for
        suppression must see a request that looks exactly as it did
        before. Callers merge the result into their request kwargs; an
        empty dict adds nothing.
        """
        if self._config.reasoning != "disabled" or self._extras_rejected:
            return {}
        return {"extra_body": {"reasoning": {"enabled": False}}}

    def _drop_rejected_extras(
        self, error: APIStatusError, request_kwargs: dict
    ) -> bool:
        """Give up the opt-in extras once a provider refuses them.

        Reasoning is suppressed by default because thinking buys nothing
        here, but OpenAI answers 400 to a body field it does not know.
        Rather than leave every operator to discover that, read the first
        such 400 as the provider saying it does not speak this field,
        remember that for the life of the client, and carry on without
        it. Returns whether the caller should send the request again.
        """
        if error.status_code != 400 or "extra_body" not in request_kwargs:
            return False
        request_kwargs.pop("extra_body")
        self._extras_rejected = True
        logger.info(
            "Provider rejected the reasoning body field (400); continuing "
            "without it (model=%s)",
            self._config.model,
        )
        return True

    @property
    def enabled(self) -> bool:
        """True if LLM is properly configured and not disabled."""
        return self._enabled

    async def _wait_for_rate_limit(self) -> None:
        """Enforce minimum interval between requests if configured."""
        interval = self._config.min_request_interval_ms / 1000.0
        if interval <= 0:
            return

        async with self._rate_lock:
            if self._last_request_time is not None:
                elapsed = time.monotonic() - self._last_request_time
                wait = interval - elapsed
                if wait > 0:
                    await asyncio.sleep(wait)
            self._last_request_time = time.monotonic()

    async def _generate_result(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens_override: int | None = None,
        response_format: dict | None = None,
        temperature: float | None = None,
    ) -> TextGeneration:
        """Generate a completion and say why it is unusable, if it is.

        Retries on timeouts, rate limits (429), and server errors
        (500-504) using exponential backoff. Permanent errors
        (400, 401, 402, 403, 404) are not retried.

        Args:
            system_prompt: System message for the model.
            user_prompt: User message for the model.
            max_tokens_override: Optional per-call override for
                max_tokens. When None, falls back to the configured
                ``self._config.max_tokens``. RAG answer generation uses
                this to get a longer token budget than the default
                summary/tag budget without mutating global config.
            response_format: Optional OpenAI-style response_format dict
                (e.g. ``{"type": "json_object"}``). Forwarded to the
                Chat Completions API when non-None. Providers that
                don't support it typically ignore the field; providers
                that do will refuse to return malformed JSON.

        Returns:
            The completion text (truncation included) paired with a
            failure reason, or ``None`` text when the request itself
            never produced a response.
        """
        if not self._enabled:
            return TextGeneration(None, FAILURE_REQUEST_FAILED)

        max_attempts = max(1, self._config.retry_attempts + 1)
        effective_max_tokens = (
            max_tokens_override
            if max_tokens_override is not None
            else self._config.max_tokens
        )
        effective_temperature = (
            temperature
            if temperature is not None
            else self._config.temperature
        )

        # Only include response_format in the kwargs when specified, so
        # providers that 400 on unknown keys are not broken for non-JSON
        # callers like RAG streaming.
        extra_kwargs: dict = self._provider_extras()
        if response_format is not None:
            extra_kwargs["response_format"] = response_format

        # OpenAI's gpt-5 / o-series families reject ``max_tokens`` and
        # require ``max_completion_tokens``. Every other OpenAI-compatible
        # backend (ollama, vLLM, LM Studio, gpt-4, gpt-3.5) still speaks
        # the legacy parameter, so we route by model prefix rather than
        # sending both.
        if _uses_max_completion_tokens(self._config.model):
            extra_kwargs["max_completion_tokens"] = effective_max_tokens
        else:
            extra_kwargs["max_tokens"] = effective_max_tokens

        attempt = 0
        while attempt < max_attempts:
            await self._wait_for_rate_limit()

            try:
                response = await self._client.chat.completions.create(
                    model=self._config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=effective_temperature,
                    **extra_kwargs,
                )
                choice = response.choices[0]
                content = choice.message.content
                finish_reason = getattr(choice, "finish_reason", None)
                reasoning_tokens = _usage_reasoning_tokens(response)
                failure = _classify_completion(content, finish_reason)
                if failure is not None:
                    self._log_unusable_completion(
                        failure, content, finish_reason,
                        response, reasoning_tokens,
                    )
                return TextGeneration(content, failure)
            except _RETRY_EXCEPTIONS as e:
                if attempt + 1 >= max_attempts:
                    logger.error(
                        "LLM generation failed after %d attempts: %s",
                        max_attempts, e,
                    )
                    return TextGeneration(None, FAILURE_REQUEST_FAILED)
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "LLM generation attempt %d/%d failed (%s), "
                    "retrying in %.1fs",
                    attempt + 1, max_attempts, type(e).__name__, delay,
                )
                await asyncio.sleep(delay)
                attempt += 1
            except APIStatusError as e:
                if self._drop_rejected_extras(e, extra_kwargs):
                    # The provider does not know the field; the request
                    # itself was fine. Send it as it would have looked
                    # without the opt-in, and do not spend a retry on
                    # our own doing.
                    continue
                if e.status_code in _PERMANENT_STATUS_CODES:
                    logger.warning(
                        "LLM generation failed with permanent error %d: %s",
                        e.status_code, e,
                    )
                    return TextGeneration(None, FAILURE_REQUEST_FAILED)
                if attempt + 1 >= max_attempts:
                    logger.error(
                        "LLM generation failed after %d attempts (status %d): %s",
                        max_attempts, e.status_code, e,
                    )
                    return TextGeneration(None, FAILURE_REQUEST_FAILED)
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "LLM generation attempt %d/%d failed with status %d, "
                    "retrying in %.1fs",
                    attempt + 1, max_attempts, e.status_code, delay,
                )
                await asyncio.sleep(delay)
                attempt += 1
            except Exception as e:
                logger.warning("LLM generation failed: %s", e)
                return TextGeneration(None, FAILURE_REQUEST_FAILED)

        return TextGeneration(None, FAILURE_REQUEST_FAILED)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens_override: int | None = None,
        response_format: dict | None = None,
        temperature: float | None = None,
    ) -> str | None:
        """Generate a text completion; see ``_generate_result``.

        Returns the text alone, so a caller that has no use for the
        failure reason is unaffected by the classification.
        """
        result = await self._generate_result(
            system_prompt,
            user_prompt,
            max_tokens_override=max_tokens_override,
            response_format=response_format,
            temperature=temperature,
        )
        return result.text

    def _log_unusable_completion(
        self,
        failure: str,
        content: str | None,
        finish_reason: object,
        response: object,
        reasoning_tokens: int | None,
    ) -> None:
        """Say what came back, so the cause is not left to guesswork.

        Without this the whole class of provider-side failures reaches
        the caller as a bare ``None`` and leaves no trace in the log.
        """
        usage = getattr(response, "usage", None)
        # Truncated output is still handed back, so saying it is absent
        # would send a reader looking for the wrong problem.
        headline = (
            "LLM output was truncated (%s)"
            if content and content.strip()
            else "LLM produced no usable output (%s)"
        )
        logger.warning(
            headline + ": finish_reason=%s, completion_tokens=%s, "
            "reasoning_tokens=%s, content_len=%d, model=%s",
            failure,
            finish_reason,
            getattr(usage, "completion_tokens", None),
            reasoning_tokens,
            len(content or ""),
            self._config.model,
        )

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens_override: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream a text completion token-by-token.

        Unlike ``generate()``, this method does **not** retry on
        transient failures: the client has already started receiving
        bytes, and silently restarting would produce a disjoint output
        stream. On any error the iterator terminates cleanly and the
        caller is responsible for treating a short/empty stream as a
        failure — the SSE wrapper logs the exception type so the
        streamed error manifests as "no answer" rather than a 500.

        Args:
            system_prompt: System message for the model.
            user_prompt: User message for the model.
            max_tokens_override: Optional per-call override for
                ``max_tokens`` (see ``generate``).

        Yields:
            Content delta strings, in order, as the provider emits them.
            Empty deltas (some providers send empty strings as
            keep-alives) are filtered out so downstream SSE consumers
            do not see redundant events.
        """
        if not self._enabled:
            return

        await self._wait_for_rate_limit()

        effective_max_tokens = (
            max_tokens_override
            if max_tokens_override is not None
            else self._config.max_tokens
        )
        effective_temperature = (
            temperature
            if temperature is not None
            else self._config.temperature
        )

        stream_kwargs: dict = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": effective_temperature,
            "stream": True,
            **self._provider_extras(),
        }
        if _uses_max_completion_tokens(self._config.model):
            stream_kwargs["max_completion_tokens"] = effective_max_tokens
        else:
            stream_kwargs["max_tokens"] = effective_max_tokens

        try:
            stream = await self._client.chat.completions.create(**stream_kwargs)
        except _RETRY_EXCEPTIONS as e:
            logger.warning(
                "LLM stream open failed (%s); yielding nothing",
                type(e).__name__,
            )
            return
        except APIStatusError as e:
            if not self._drop_rejected_extras(e, stream_kwargs):
                logger.warning(
                    "LLM stream open failed with status %d; yielding nothing",
                    e.status_code,
                )
                return
            try:
                stream = await self._client.chat.completions.create(
                    **stream_kwargs
                )
            except Exception as retry_error:
                logger.warning(
                    "LLM stream open failed after dropping the reasoning "
                    "field: %s",
                    retry_error,
                )
                return
        except Exception as e:
            logger.warning("LLM stream open failed: %s", e)
            return

        try:
            async for chunk in stream:
                # The OpenAI SDK normalizes streaming chunks into an
                # object with ``choices[0].delta.content`` — which is
                # None on the very first and very last chunks (role
                # announcement + finish marker). Skip those quietly so
                # the caller only sees real deltas.
                try:
                    choices = chunk.choices
                except AttributeError:
                    continue
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if content:
                    yield content
        except Exception as e:
            # Mid-stream failure. Log the exception type and terminate
            # the iterator. The caller has whatever was already yielded;
            # the SSE layer converts the short output into a best-effort
            # partial answer (or None if nothing came through at all).
            logger.warning(
                "LLM stream interrupted (%s); terminating",
                type(e).__name__,
            )
            return

    async def _vision_chat(
        self,
        image_bytes: bytes,
        mime_type: str,
        system_prompt: str,
        user_prompt: str,
        *,
        response_format: dict | None = None,
    ) -> str | object | None:
        """Shared OpenAI-compatible vision transport.

        Private generic-image-message helper factored out of
        ``generate_vision`` (design doc "Video Visual Index" §4.2) so
        the video-scene structured-output path can reuse the same
        retry / unsupported-detection semantics without duplicating
        them. The public ``generate_vision`` contract (input, prompt,
        status handling, return values) is unchanged.

        Returns the response text on success, :data:`VISION_UNSUPPORTED`
        when the provider answers 400/404 (model is not vision-capable),
        or ``None`` on disabled state, empty response, or exhausted
        transient-failure retries.
        """
        if not self._enabled or not self._config.vision_model:
            return None

        # Encode once — the same base64 payload is used on every retry
        # so we don't pay O(bytes) per attempt.
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{b64}"

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ]

        extra_kwargs: dict = self._provider_extras()
        if _uses_max_completion_tokens(self._config.vision_model):
            extra_kwargs["max_completion_tokens"] = self._config.vision_max_tokens
        else:
            extra_kwargs["max_tokens"] = self._config.vision_max_tokens
        if response_format is not None:
            extra_kwargs["response_format"] = response_format

        max_attempts = max(1, self._config.retry_attempts + 1)
        attempt = 0
        while attempt < max_attempts:
            await self._wait_for_rate_limit()

            try:
                response = await self._client.chat.completions.create(
                    model=self._config.vision_model,
                    messages=messages,
                    temperature=self._config.vision_temperature,
                    **extra_kwargs,
                )
            except _RETRY_EXCEPTIONS as e:
                if attempt + 1 >= max_attempts:
                    logger.error(
                        "Vision generation failed after %d attempts: %s",
                        max_attempts, type(e).__name__,
                    )
                    return None
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Vision generation attempt %d/%d failed (%s), "
                    "retrying in %.1fs",
                    attempt + 1, max_attempts, type(e).__name__, delay,
                )
                await asyncio.sleep(delay)
                attempt += 1
                continue
            except APIStatusError as e:
                # Before concluding the model cannot see, rule out our
                # own body field: a 400 from that would otherwise mark a
                # perfectly capable model unsupported for good.
                if self._drop_rejected_extras(e, extra_kwargs):
                    continue
                if e.status_code in _VISION_UNSUPPORTED_STATUS_CODES:
                    logger.info(
                        "Vision generation rejected by provider (status %d); "
                        "marking unsupported",
                        e.status_code,
                    )
                    return VISION_UNSUPPORTED
                if e.status_code in _PERMANENT_STATUS_CODES:
                    logger.warning(
                        "Vision generation failed with permanent error %d",
                        e.status_code,
                    )
                    return None
                if attempt + 1 >= max_attempts:
                    return None
                delay = self._backoff_delay(attempt)
                await asyncio.sleep(delay)
                attempt += 1
                continue
            except Exception as e:
                logger.warning("Vision generation failed: %s", type(e).__name__)
                return None

            try:
                content = response.choices[0].message.content
            except (AttributeError, IndexError):
                return None
            if not isinstance(content, str):
                return None
            text_out = content.strip()
            if not text_out:
                return None
            return text_out

        return None

    async def generate_vision(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        output_language: str = "auto",
    ) -> str | object | None:
        """Generate a description for an image via a vision-capable LLM.

        Uses the OpenAI Chat Completions "image_url" content block with
        a data-URL embedding. Returns:

        * the description text on success,
        * :data:`VISION_UNSUPPORTED` when the provider answers 400/404
          (model is not vision-capable),
        * ``None`` on disabled state, empty response, or transient
          failures (timeouts, 5xx).

        The call uses ``self._config.vision_model`` (NOT ``model``) and
        the vision-specific ``vision_max_tokens`` / ``vision_temperature``
        so the operator can run a text model alongside a vision model.

        Args:
            image_bytes: Raw image bytes. Should be pre-processed
                (resized, re-encoded) by the caller.
            mime_type: MIME type for the data URL (e.g. ``image/jpeg``).
            prompt: User-side instruction. Kept generic so the caller
                decides phrasing.
            output_language: Language tag threaded into the system
                prompt. ``"auto"`` resolves to a filename-derived hint.
        """
        system_prompt = _build_vision_system_prompt(output_language)
        return await self._vision_chat(image_bytes, mime_type, system_prompt, prompt)

    async def generate_video_scene_json(
        self,
        image_bytes: bytes,
        mime_type: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict | list | object | None:
        """Structured-output vision call for one video-visual-index scene.

        Reuses :meth:`_vision_chat` (same transport, retry, and
        unsupported-detection as ``generate_vision``) but requests JSON
        object mode and parses the result via the same
        ``_parse_json_response`` fallback used by ``generate_json``.
        The video path uses a dedicated prompt/parser and never touches
        ``generate_vision``'s own contract (design doc §4.2).

        Returns the parsed JSON value, :data:`VISION_UNSUPPORTED`, or
        ``None`` on failure / malformed / empty output. A provider that
        rejects JSON mode is retried once without ``response_format``
        before it is classified as vision-unsupported. Retry/repair
        policy for malformed JSON remains the caller's responsibility.
        """
        result = await self._vision_chat(
            image_bytes, mime_type, system_prompt, user_prompt,
            response_format={"type": "json_object"},
        )
        if result is VISION_UNSUPPORTED:
            logger.info(
                "Vision request rejected with json_object mode; "
                "retrying without response_format"
            )
            result = await self._vision_chat(
                image_bytes, mime_type, system_prompt, user_prompt,
            )
        if result is VISION_UNSUPPORTED or result is None:
            return result
        return _parse_json_response(result)

    def _backoff_delay(self, attempt: int) -> float:
        """Compute exponential backoff delay for a given attempt.

        Args:
            attempt: Zero-based attempt number (0 = first retry).

        Returns:
            Delay in seconds, capped at retry_max_delay.
        """
        delay = self._config.retry_base_delay * (2**attempt)
        return min(delay, self._config.retry_max_delay)

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens_override: int | None = None,
        temperature: float | None = None,
    ) -> list | dict | None:
        """Generate a completion and parse the result as JSON.

        Attempts direct JSON parsing first. On failure, tries to
        extract either a JSON object or array using regex as a fallback
        (in case the model wraps output in ```json ... ``` or prose).

        Args:
            system_prompt: System message for the model.
            user_prompt: User message for the model.
            max_tokens_override: Optional per-call override for
                max_tokens. Forwarded to ``generate()``. None means
                use the configured default.

        Returns:
            Parsed JSON (list or dict), or None on failure.
        """
        # Request JSON object mode so compliant providers (OpenAI,
        # ollama, vLLM, LM Studio) refuse to emit non-JSON output.
        # Providers that don't support the field typically ignore it,
        # so we still rely on the regex fallback below as a safety net.
        return (
            await self.generate_json_result(
                system_prompt,
                user_prompt,
                max_tokens_override=max_tokens_override,
                temperature=temperature,
            )
        ).value

    async def generate_json_result(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens_override: int | None = None,
        temperature: float | None = None,
    ) -> JsonGeneration:
        """Generate JSON and say why it is missing, if it is.

        Callers that can act on the distinction — offering the operator
        a budget remedy rather than a generic retry — use this; the rest
        stay on ``generate_json``.
        """
        result = await self._generate_result(
            system_prompt,
            user_prompt,
            max_tokens_override=max_tokens_override,
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        raw = result.text
        if raw is None:
            return JsonGeneration(
                None, result.failure or FAILURE_REQUEST_FAILED
            )

        # Some OpenAI-compatible backends (certain ollama versions in
        # particular) accept response_format={"type": "json_object"} but
        # silently return an empty body instead of enforcing JSON. When
        # that happens, retry once without response_format so the model
        # can obey the prompt-level JSON instruction. Compliant providers
        # never hit this path, so latency is only paid by broken ones.
        if not raw.strip():
            logger.info(
                "LLM returned empty body with json_object mode; "
                "retrying without response_format"
            )
            result = await self._generate_result(
                system_prompt,
                user_prompt,
                max_tokens_override=max_tokens_override,
                temperature=temperature,
            )
            raw = result.text
            if raw is None:
                return JsonGeneration(
                    None, result.failure or FAILURE_REQUEST_FAILED
                )

        parsed = _parse_json_response(raw)
        if parsed is None:
            # A body cut off by the token ceiling is unparseable for a
            # reason the caller can act on, so the upstream cause wins
            # over the parse symptom.
            return JsonGeneration(None, result.failure or FAILURE_MALFORMED)
        return JsonGeneration(parsed, None)

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        temperature: float | None = None,
        max_tokens_override: int | None = None,
        tool_choice: str | dict = "auto",
        response_format: dict | None = None,
    ) -> ChatTurnResult | None:
        """One turn of the agentic loop (Phase 1.C).

        Sends the full message list to the OpenAI-compatible Chat
        Completions endpoint with the ``tools`` parameter. ``messages``
        is the loop's running transcript — system, user, assistant
        (tool_calls), tool (results), …

        Returns the parsed ``ChatTurnResult`` or ``None`` on permanent
        / retried-out failure. The retry policy mirrors ``generate()``;
        callers should treat ``None`` as fail-loud (stop the loop).
        """
        if not self._enabled:
            return None

        max_attempts = max(1, self._config.retry_attempts + 1)
        effective_max_tokens = (
            max_tokens_override
            if max_tokens_override is not None
            else self._config.max_tokens
        )
        effective_temperature = (
            temperature
            if temperature is not None
            else self._config.temperature
        )

        extra_kwargs: dict = self._provider_extras()
        if _uses_max_completion_tokens(self._config.model):
            extra_kwargs["max_completion_tokens"] = effective_max_tokens
        else:
            extra_kwargs["max_tokens"] = effective_max_tokens
        if tools is not None:
            extra_kwargs["tools"] = tools
            extra_kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            extra_kwargs["response_format"] = response_format

        attempt = 0
        while attempt < max_attempts:
            await self._wait_for_rate_limit()
            try:
                response = await self._client.chat.completions.create(
                    model=self._config.model,
                    messages=messages,
                    temperature=effective_temperature,
                    **extra_kwargs,
                )
                choice = response.choices[0]
                msg = choice.message
                tool_calls_raw = getattr(msg, "tool_calls", None) or []
                parsed_calls: list[ChatToolCall] = []
                for tc in tool_calls_raw:
                    fn = getattr(tc, "function", None)
                    if fn is None:
                        continue
                    parsed_calls.append(
                        ChatToolCall(
                            id=getattr(tc, "id", "") or "",
                            name=getattr(fn, "name", "") or "",
                            arguments_raw=getattr(fn, "arguments", "") or "",
                        )
                    )
                return ChatTurnResult(
                    content=getattr(msg, "content", None),
                    tool_calls=tuple(parsed_calls),
                    finish_reason=choice.finish_reason or "stop",
                )
            except _RETRY_EXCEPTIONS as e:
                if attempt + 1 >= max_attempts:
                    logger.error(
                        "chat_with_tools failed after %d attempts: %s",
                        max_attempts, e,
                    )
                    return None
                await asyncio.sleep(self._backoff_delay(attempt))
                attempt += 1
            except APIStatusError as e:
                if self._drop_rejected_extras(e, extra_kwargs):
                    continue
                if e.status_code in _PERMANENT_STATUS_CODES:
                    logger.warning(
                        "chat_with_tools permanent error %d: %s",
                        e.status_code, e,
                    )
                    return None
                if attempt + 1 >= max_attempts:
                    return None
                await asyncio.sleep(self._backoff_delay(attempt))
                attempt += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("chat_with_tools failed: %s", e)
                return None
        return None


# ---------------------------------------------------------------------------
# Ollama native API backend
# ---------------------------------------------------------------------------


# Transient HTTP status codes worth retrying for ollama's native API.
_OLLAMA_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class OllamaLLMClient:
    """Async LLM client using ollama's native ``/api/chat`` HTTP API.

    The key reason this exists alongside the OpenAI-compatible
    :class:`LLMClient` is that ollama's ``/v1`` compatibility layer
    silently ignores ``think: false`` during streaming, so reasoning
    models (Gemma 4, DeepSeek-R1, QwQ) incur 10-20s of thinking delay
    before the first visible token. The native ``/api/chat`` endpoint
    honours ``think: false``, eliminating that delay entirely.

    Same public interface as :class:`LLMClient` — callers obtain
    either instance via :func:`create_llm_client` and never branch
    on the provider.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._enabled = (
            config.provider != "disabled"
            and bool(config.base_url)
            and bool(config.model)
        )
        # Normalise base_url: strip trailing slash and any /v1 suffix
        # so users can paste the same URL they had for openai_compatible
        # and the native-API paths still resolve correctly.
        base = (config.base_url or "").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        self._base_url = base

        if self._enabled:
            total = config.request_timeout_seconds
            connect = config.request_connect_timeout_seconds
            timeout = httpx.Timeout(
                float(total) if isinstance(total, (int, float)) and total > 0 else 90.0,
                connect=(
                    float(connect)
                    if isinstance(connect, (int, float)) and connect > 0
                    else 10.0
                ),
            )
            self._http = httpx.AsyncClient(timeout=timeout)
        # Rate-limiting state (None = no previous request yet)
        self._rate_lock = asyncio.Lock()
        self._last_request_time: float | None = None

    @property
    def enabled(self) -> bool:
        """True if LLM is properly configured and not disabled."""
        return self._enabled

    async def _wait_for_rate_limit(self) -> None:
        """Enforce minimum interval between requests if configured."""
        interval = self._config.min_request_interval_ms / 1000.0
        if interval <= 0:
            return

        async with self._rate_lock:
            if self._last_request_time is not None:
                elapsed = time.monotonic() - self._last_request_time
                wait = interval - elapsed
                if wait > 0:
                    await asyncio.sleep(wait)
            self._last_request_time = time.monotonic()

    def _backoff_delay(self, attempt: int) -> float:
        """Compute exponential backoff delay for a given attempt."""
        delay = self._config.retry_base_delay * (2**attempt)
        return min(delay, self._config.retry_max_delay)

    def _build_body(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        stream: bool,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
    ) -> dict:
        """Construct the request body for /api/chat.

        ``think: false`` is always set so reasoning models skip the
        chain-of-thought phase. Ollama's format field takes a string
        ("json") rather than OpenAI's ``{"type": "json_object"}``, so
        we translate here.
        """
        body: dict = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": stream,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if response_format is not None:
            # OpenAI sends {"type": "json_object"}; ollama wants "json".
            fmt = response_format.get("type", "json")
            body["format"] = "json" if fmt in ("json_object", "json") else fmt
        return body

    async def _generate_result(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens_override: int | None = None,
        response_format: dict | None = None,
        temperature: float | None = None,
    ) -> TextGeneration:
        """Generate via /api/chat and say why the output is unusable.

        ``done_reason: "length"`` means the answer hit
        ``options.num_predict``. ``think: false`` keeps chain-of-thought
        out of that budget, but an over-long answer still runs into it,
        and the remedy is a budget change rather than a prompt change.
        """
        if not self._enabled:
            return TextGeneration(None, FAILURE_REQUEST_FAILED)

        max_attempts = max(1, self._config.retry_attempts + 1)
        effective_max_tokens = (
            max_tokens_override
            if max_tokens_override is not None
            else self._config.max_tokens
        )
        effective_temperature = (
            temperature
            if temperature is not None
            else self._config.temperature
        )

        body = self._build_body(
            system_prompt,
            user_prompt,
            stream=False,
            temperature=effective_temperature,
            max_tokens=effective_max_tokens,
            response_format=response_format,
        )

        for attempt in range(max_attempts):
            await self._wait_for_rate_limit()

            try:
                resp = await self._http.post(
                    f"{self._base_url}/api/chat",
                    json=body,
                )
            except httpx.TimeoutException as e:
                if attempt + 1 >= max_attempts:
                    logger.error(
                        "Ollama generation timed out after %d attempts: %s",
                        max_attempts, e,
                    )
                    return TextGeneration(None, FAILURE_REQUEST_FAILED)
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Ollama generation attempt %d/%d timed out, "
                    "retrying in %.1fs",
                    attempt + 1, max_attempts, delay,
                )
                await asyncio.sleep(delay)
                continue
            except httpx.HTTPError as e:
                if attempt + 1 >= max_attempts:
                    logger.error(
                        "Ollama generation failed after %d attempts: %s",
                        max_attempts, e,
                    )
                    return TextGeneration(None, FAILURE_REQUEST_FAILED)
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Ollama generation attempt %d/%d failed (%s), "
                    "retrying in %.1fs",
                    attempt + 1, max_attempts, type(e).__name__, delay,
                )
                await asyncio.sleep(delay)
                continue

            if resp.status_code in _PERMANENT_STATUS_CODES:
                logger.warning(
                    "Ollama generation failed with permanent error %d",
                    resp.status_code,
                )
                return TextGeneration(None, FAILURE_REQUEST_FAILED)

            if resp.status_code in _OLLAMA_RETRY_STATUSES:
                if attempt + 1 >= max_attempts:
                    logger.error(
                        "Ollama generation failed after %d attempts (status %d)",
                        max_attempts, resp.status_code,
                    )
                    return TextGeneration(None, FAILURE_REQUEST_FAILED)
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Ollama generation attempt %d/%d got status %d, "
                    "retrying in %.1fs",
                    attempt + 1, max_attempts, resp.status_code, delay,
                )
                await asyncio.sleep(delay)
                continue

            if resp.status_code != 200:
                logger.warning(
                    "Ollama generation got unexpected status %d",
                    resp.status_code,
                )
                return TextGeneration(None, FAILURE_REQUEST_FAILED)

            try:
                data = resp.json()
            except Exception:
                logger.warning(
                    "Ollama returned non-JSON response (status %d)",
                    resp.status_code,
                )
                return TextGeneration(None, FAILURE_REQUEST_FAILED)

            content = data.get("message", {}).get("content", "")
            done_reason = data.get("done_reason")
            failure = _classify_completion(content, done_reason)
            if failure is not None:
                headline = (
                    "Ollama output was truncated (%s)"
                    if content and content.strip()
                    else "Ollama produced no usable output (%s)"
                )
                logger.warning(
                    headline + ": done_reason=%s, content_len=%d, model=%s",
                    failure, done_reason, len(content or ""),
                    self._config.model,
                )
            return TextGeneration(content, failure)

        return TextGeneration(None, FAILURE_REQUEST_FAILED)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens_override: int | None = None,
        response_format: dict | None = None,
        temperature: float | None = None,
    ) -> str | None:
        """Generate a non-streamed completion; see ``_generate_result``."""
        result = await self._generate_result(
            system_prompt,
            user_prompt,
            max_tokens_override=max_tokens_override,
            response_format=response_format,
            temperature=temperature,
        )
        return result.text

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens_override: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion via /api/chat (NDJSON response).

        Ollama's streaming response is newline-delimited JSON, one
        object per line with ``{"message": {"content": "…"}, "done": bool}``.
        We yield each non-empty content delta and stop when ``done``
        becomes true.

        Like :meth:`LLMClient.generate_stream`, this does not retry
        after bytes have started flowing — silently restarting would
        produce a disjoint output stream.
        """
        if not self._enabled:
            return

        await self._wait_for_rate_limit()

        effective_max_tokens = (
            max_tokens_override
            if max_tokens_override is not None
            else self._config.max_tokens
        )
        effective_temperature = (
            temperature
            if temperature is not None
            else self._config.temperature
        )

        body = self._build_body(
            system_prompt,
            user_prompt,
            stream=True,
            temperature=effective_temperature,
            max_tokens=effective_max_tokens,
            response_format=None,
        )

        try:
            async with self._http.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json=body,
            ) as resp:
                if resp.status_code != 200:
                    logger.warning(
                        "Ollama stream open failed with status %d",
                        resp.status_code,
                    )
                    return

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content

                    if data.get("done", False):
                        return
        except httpx.TimeoutException:
            logger.warning("Ollama stream timed out; terminating")
            return
        except httpx.HTTPError as e:
            logger.warning(
                "Ollama stream interrupted (%s); terminating",
                type(e).__name__,
            )
            return
        except Exception as e:
            logger.warning(
                "Ollama stream error (%s); terminating",
                type(e).__name__,
            )
            return

    async def _vision_chat(
        self,
        image_bytes: bytes,
        mime_type: str,
        system_prompt: str,
        user_prompt: str,
        *,
        response_format: dict | None = None,
    ) -> str | object | None:
        """Shared ollama-native vision transport.

        Private generic-image-message helper factored out of
        ``generate_vision`` (design doc "Video Visual Index" §4.2), used
        by both ``generate_vision`` and ``generate_video_scene_json``.
        ``mime_type`` is accepted for API parity; ollama ignores it and
        sniffs from the bytes.
        """
        if not self._enabled or not self._config.vision_model:
            return None

        b64 = base64.b64encode(image_bytes).decode("ascii")

        body: dict = {
            "model": self._config.vision_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [b64],
                },
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": self._config.vision_temperature,
                "num_predict": self._config.vision_max_tokens,
            },
        }
        if response_format is not None:
            # OpenAI sends {"type": "json_object"}; ollama wants "json".
            fmt = response_format.get("type", "json")
            body["format"] = "json" if fmt in ("json_object", "json") else fmt

        max_attempts = max(1, self._config.retry_attempts + 1)
        for attempt in range(max_attempts):
            await self._wait_for_rate_limit()

            try:
                resp = await self._http.post(
                    f"{self._base_url}/api/chat",
                    json=body,
                )
            except httpx.TimeoutException:
                if attempt + 1 >= max_attempts:
                    return None
                await asyncio.sleep(self._backoff_delay(attempt))
                continue
            except httpx.HTTPError as e:
                if attempt + 1 >= max_attempts:
                    logger.warning(
                        "Ollama vision generation failed: %s", type(e).__name__,
                    )
                    return None
                await asyncio.sleep(self._backoff_delay(attempt))
                continue

            if resp.status_code in _VISION_UNSUPPORTED_STATUS_CODES:
                # Ollama returns 404 for an uninstalled model and 400 for
                # a model that exists but doesn't handle images. Both
                # are sticky failures for this (model, provider) pair —
                # caller records status="unsupported" to suppress retry.
                return VISION_UNSUPPORTED

            if resp.status_code in _OLLAMA_RETRY_STATUSES:
                if attempt + 1 >= max_attempts:
                    return None
                await asyncio.sleep(self._backoff_delay(attempt))
                continue

            if resp.status_code != 200:
                logger.warning(
                    "Ollama vision unexpected status %d", resp.status_code,
                )
                return None

            try:
                data = resp.json()
            except Exception:
                return None

            content = data.get("message", {}).get("content", "")
            if not isinstance(content, str):
                return None
            text_out = content.strip()
            if not text_out:
                return None
            return text_out

        return None

    async def generate_vision(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        output_language: str = "auto",
    ) -> str | object | None:
        """Generate an image description via ollama's native ``/api/chat``.

        Ollama's native protocol embeds images as a list of base64
        strings on the message (no data-URL prefix), distinct from the
        OpenAI-compatible ``image_url`` block. Response contract matches
        :meth:`LLMClient.generate_vision`: text on success,
        :data:`VISION_UNSUPPORTED` on 400/404, ``None`` otherwise.
        """
        system_prompt = _build_vision_system_prompt(output_language)
        return await self._vision_chat(image_bytes, mime_type, system_prompt, prompt)

    async def generate_video_scene_json(
        self,
        image_bytes: bytes,
        mime_type: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict | list | object | None:
        """Structured-output vision call for one video-visual-index scene.

        Same contract as :meth:`LLMClient.generate_video_scene_json`:
        JSON-mode requested and parsed via ``_parse_json_response``.
        Providers that reject JSON mode are retried once without it.
        Returns the parsed value, :data:`VISION_UNSUPPORTED`, or ``None``.
        """
        result = await self._vision_chat(
            image_bytes, mime_type, system_prompt, user_prompt,
            response_format={"type": "json_object"},
        )
        if result is VISION_UNSUPPORTED:
            logger.info(
                "Ollama vision request rejected with format=json; "
                "retrying without format"
            )
            result = await self._vision_chat(
                image_bytes, mime_type, system_prompt, user_prompt,
            )
        if result is VISION_UNSUPPORTED or result is None:
            return result
        return _parse_json_response(result)

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens_override: int | None = None,
        temperature: float | None = None,
    ) -> list | dict | None:
        """Generate a completion and parse the result as JSON."""
        return (
            await self.generate_json_result(
                system_prompt,
                user_prompt,
                max_tokens_override=max_tokens_override,
                temperature=temperature,
            )
        ).value

    async def generate_json_result(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens_override: int | None = None,
        temperature: float | None = None,
    ) -> JsonGeneration:
        """Generate JSON and say why it is missing, if it is.

        The native body always carries ``think: false``, so chain-of-
        thought never consumes the budget here — but an answer that runs
        past ``options.num_predict`` still does, and that arrives as
        ``done_reason: "length"`` rather than as a parse error.
        """
        result = await self._generate_result(
            system_prompt,
            user_prompt,
            max_tokens_override=max_tokens_override,
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        raw = result.text
        if raw is None:
            return JsonGeneration(
                None, result.failure or FAILURE_REQUEST_FAILED
            )

        if not raw.strip():
            logger.info(
                "Ollama returned empty body with format=json; "
                "retrying without format"
            )
            result = await self._generate_result(
                system_prompt,
                user_prompt,
                max_tokens_override=max_tokens_override,
                temperature=temperature,
            )
            raw = result.text
            if raw is None:
                return JsonGeneration(
                    None, result.failure or FAILURE_REQUEST_FAILED
                )
            if not raw.strip():
                return JsonGeneration(None, result.failure or FAILURE_EMPTY)

        parsed = _parse_json_response(raw)
        if parsed is None:
            return JsonGeneration(None, result.failure or FAILURE_MALFORMED)
        return JsonGeneration(parsed, None)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_llm_client(config: LLMConfig) -> LLMClient | OllamaLLMClient:
    """Instantiate the correct LLM client based on ``config.provider``.

    Supported providers:
    * ``"openai_compatible"`` — :class:`LLMClient` (OpenAI SDK)
    * ``"ollama"`` — :class:`OllamaLLMClient` (native API with think=false)
    * ``"disabled"`` — returns a disabled :class:`LLMClient`
      (the ``enabled`` flag is False; all methods no-op)

    Both classes expose the same interface, so callers do not need
    to branch on the return type.
    """
    if config.provider == "ollama":
        return OllamaLLMClient(config)
    return LLMClient(config)
