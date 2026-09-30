"""Build a resolved LLM profile around a test double."""

from __future__ import annotations

from app import llm_routing
from app.config import LLMConfig


def resolved_with(client, *, model: str = "test-llm", vision_model: str = ""):
    profile = llm_routing.LLMProfile(
        name="test",
        config=LLMConfig(
            provider="openai_compatible",
            base_url="http://llm.test/v1",
            model=model,
            vision_model=vision_model,
        ),
        offhost=False,
        agentic=False,
        api_key_env="",
    )
    return llm_routing.Resolved(profile=profile, client=client)
