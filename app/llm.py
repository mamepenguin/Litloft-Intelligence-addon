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
            self._client = AsyncOpenAI(
                base_url=config.base_url,
                api_key=config.api_key or "not-needed",
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
        self, system_prompt: str, user_prompt: str
    ) -> str | None:
        """Generate a text completion with retry on transient failures.

        Retries on timeouts, rate limits (429), and server errors
        (500-504) using exponential backoff. Permanent errors
        (400, 401, 402, 403, 404) are not retried.

        Args:
            system_prompt: System message for the model.
            user_prompt: User message for the model.

        Returns:
            Completion text, or None if disabled or on final failure.
        """
        if not self._enabled:
            return None

        max_attempts = max(1, self._config.retry_attempts + 1)

        for attempt in range(max_attempts):
            await self._wait_for_rate_limit()

            try:
                response = await self._client.chat.completions.create(
                    model=self._config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
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
        self, system_prompt: str, user_prompt: str
    ) -> list | dict | None:
        """Generate a completion and parse the result as JSON.

        Attempts direct JSON parsing first. On failure, tries to
        extract either a JSON object or array using regex as a fallback
        (in case the model wraps output in ```json ... ``` or prose).

        Args:
            system_prompt: System message for the model.
            user_prompt: User message for the model.

        Returns:
            Parsed JSON (list or dict), or None on failure.
        """
        raw = await self.generate(system_prompt, user_prompt)
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
