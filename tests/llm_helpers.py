"""Build a resolved LLM profile around a test double."""

from __future__ import annotations

from app import llm_routing
from app.config import LLMConfig


def resolved_with(
    client,
    *,
    model: str = "test-llm",
    vision_model: str = "",
    agentic: bool = False,
    config: LLMConfig | None = None,
):
    profile = llm_routing.LLMProfile(
        name="test",
        config=config
        or LLMConfig(
            provider="openai_compatible",
            base_url="http://llm.test/v1",
            model=model,
            vision_model=vision_model,
        ),
        offhost=False,
        agentic=agentic,
        api_key_env="",
    )
    return llm_routing.Resolved(profile=profile, client=client)


def bind_llm(
    monkeypatch,
    factory,
    *,
    model: str = "test-llm",
    agentic: bool | None = None,
    config: LLMConfig | None = None,
) -> None:
    """Serve ``factory()`` as the Ask's client and as every drive's resolve.

    ``factory`` raising ``RuntimeError`` stands for "no client"; a client
    whose ``enabled`` is False resolves to ``Skip``. With ``config`` and no
    ``agentic``, the profile is agentic by the legacy allowlist rule.
    """
    if agentic is None:
        from app.rag.agentic import agentic_capability_supported

        agentic = config is not None and agentic_capability_supported(
            config.model, config
        )

    def _client():
        try:
            return factory()
        except RuntimeError:
            return None

    def _resolved():
        client = _client()
        return (
            resolved_with(client, model=model, agentic=agentic, config=config)
            if client is not None
            else None
        )

    async def _resolve(drive, feature, requested=None):
        client = _client()
        if client is None or not getattr(client, "enabled", True):
            return llm_routing.Skip("disabled")
        return resolved_with(client, model=model, agentic=agentic, config=config)

    monkeypatch.setattr(llm_routing, "bound_client", _client)
    monkeypatch.setattr(llm_routing, "bound_resolved", _resolved)
    monkeypatch.setattr(llm_routing, "resolve", _resolve)
