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
            raw = await self.generate(
                system_prompt,
                user_prompt,
                max_tokens_override=max_tokens_override,
                temperature=temperature,
            )
            if raw is None:
                return None

        return _parse_json_response(raw)


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

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens_override: int | None = None,
        response_format: dict | None = None,
        temperature: float | None = None,
    ) -> str | None:
        """Generate a non-streamed completion via /api/chat."""
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
                    return None
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
                    return None
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
                return None

            if resp.status_code in _OLLAMA_RETRY_STATUSES:
                if attempt + 1 >= max_attempts:
                    logger.error(
                        "Ollama generation failed after %d attempts (status %d)",
                        max_attempts, resp.status_code,
                    )
                    return None
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
                return None

            try:
                data = resp.json()
            except Exception:
                logger.warning(
                    "Ollama returned non-JSON response (status %d)",
                    resp.status_code,
                )
                return None

            return data.get("message", {}).get("content", "")

        return None

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

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens_override: int | None = None,
        temperature: float | None = None,
    ) -> list | dict | None:
        """Generate a completion and parse the result as JSON."""
        raw = await self.generate(
            system_prompt,
            user_prompt,
            max_tokens_override=max_tokens_override,
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        if raw is None:
            return None

        if not raw.strip():
            logger.info(
                "Ollama returned empty body with format=json; "
                "retrying without format"
            )
            raw = await self.generate(
                system_prompt,
                user_prompt,
                max_tokens_override=max_tokens_override,
                temperature=temperature,
            )
            if raw is None:
                return None

        return _parse_json_response(raw)


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
