"""Async LLM client using OpenAI-compatible API.

Supports any OpenAI-compatible endpoint including ollama, vLLM,
LM Studio, and OpenAI itself via configurable base_url.

Includes retry with exponential backoff for transient failures
(timeouts, rate limits, server errors) and an optional minimum
request interval to rate-limit outbound requests. Permanent errors
(400, 401, 402, 403, 404) are not retried to avoid wasting API calls.
"""

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator

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

logger = logging.getLogger(__name__)

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

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens_override: int | None = None,
        response_format: dict | None = None,
        temperature: float | None = None,
    ) -> str | None:
        """Generate a text completion with retry on transient failures.

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
            Completion text, or None if disabled or on final failure.
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

        # Only include response_format in the kwargs when specified, so
        # providers that 400 on unknown keys are not broken for non-JSON
        # callers like RAG streaming.
        extra_kwargs: dict = {}
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

        for attempt in range(max_attempts):
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
                return response.choices[0].message.content
            except _RETRY_EXCEPTIONS as e:
                if attempt + 1 >= max_attempts:
                    logger.error(
                        "LLM generation failed after %d attempts: %s",
                        max_attempts, e,
                    )
                    return None
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "LLM generation attempt %d/%d failed (%s), "
                    "retrying in %.1fs",
                    attempt + 1, max_attempts, type(e).__name__, delay,
                )
                await asyncio.sleep(delay)
            except APIStatusError as e:
                if e.status_code in _PERMANENT_STATUS_CODES:
                    logger.warning(
                        "LLM generation failed with permanent error %d: %s",
                        e.status_code, e,
                    )
                    return None
                if attempt + 1 >= max_attempts:
                    logger.error(
                        "LLM generation failed after %d attempts (status %d): %s",
                        max_attempts, e.status_code, e,
                    )
                    return None
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "LLM generation attempt %d/%d failed with status %d, "
                    "retrying in %.1fs",
                    attempt + 1, max_attempts, e.status_code, delay,
                )
                await asyncio.sleep(delay)
            except Exception as e:
                logger.warning("LLM generation failed: %s", e)
                return None

        return None

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
            logger.warning(
                "LLM stream open failed with status %d; yielding nothing",
                e.status_code,
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
        raw = await self.generate(
            system_prompt,
            user_prompt,
            max_tokens_override=max_tokens_override,
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        if raw is None:
            return None

        # Try direct JSON parse
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

        # Log only a short excerpt so error logs don't contain long
        # slices of file content on parse failures — LLM output often
        # echoes bits of the prompt context.
        preview = (raw[:40] + "…") if len(raw) > 40 else raw
        logger.warning(
            "Failed to parse LLM response as JSON (len=%d, head=%r)",
            len(raw), preview,
        )
        return None
