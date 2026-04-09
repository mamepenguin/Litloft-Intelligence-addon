"""Async LLM client using OpenAI-compatible API.

Supports any OpenAI-compatible endpoint including ollama, vLLM,
LM Studio, and OpenAI itself via configurable base_url.
All errors are handled gracefully (log warning, return None).
"""

import json
import logging
import re

from openai import AsyncOpenAI

from app.config import LLMConfig

logger = logging.getLogger(__name__)


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

    @property
    def enabled(self) -> bool:
        """True if LLM is properly configured and not disabled."""
        return self._enabled

    async def generate(
        self, system_prompt: str, user_prompt: str
    ) -> str | None:
        """Generate a text completion.

        Args:
            system_prompt: System message for the model.
            user_prompt: User message for the model.

        Returns:
            Completion text, or None if disabled or on error.
        """
        if not self._enabled:
            return None

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
        except Exception as e:
            logger.warning("LLM generation failed: %s", e)
            return None

    async def generate_json(
        self, system_prompt: str, user_prompt: str
    ) -> list | dict | None:
        """Generate a completion and parse the result as JSON.

        Attempts direct JSON parsing first. On failure, tries to
        extract a JSON array using regex as a fallback.

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

        # Fallback: extract JSON array with regex
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError):
                pass

        logger.warning(
            "Failed to parse LLM response as JSON: %.200s", raw
        )
        return None
