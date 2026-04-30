"""Contract tests for intelligence → knowledge active-summary clear.

Spec ``2026-04-30-file-active-summary-to-knowledge`` moved the pointer
from core to the knowledge addon. ``regenerate_detailed_summary`` calls
``_clear_knowledge_active_summary(file_id)`` which directly hits
``DELETE http://knowledge:8200/internal/file_active_summary/{file_id}``
gated by ``KNOWLEDGE_WEBHOOK_SECRET``.

Two layers (per hako VHE7K0KWjIzV3M1CyfDAN):
1. Wire shape — URL, method, secret header.
2. Failure handling parity — best-effort behaviour preserved when the
   downstream returns 404 / 502 / network error so ``regenerate``
   never fails just because the pointer happened to be absent.
"""

from __future__ import annotations

import httpx
import pytest

from app.routers import summaries as summaries_router


def _install_transport(monkeypatch, handler):
    orig_async_client = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return orig_async_client(*args, **kwargs)

    monkeypatch.setattr(summaries_router.httpx, "AsyncClient", _factory)


@pytest.mark.asyncio
async def test_clear_targets_knowledge_internal_route(monkeypatch):
    received: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        received["url"] = str(req.url)
        received["method"] = req.method
        return httpx.Response(204)

    _install_transport(monkeypatch, handler)
    monkeypatch.delenv("KNOWLEDGE_SERVICE_URL", raising=False)
    monkeypatch.delenv("KNOWLEDGE_WEBHOOK_SECRET", raising=False)

    await summaries_router._clear_knowledge_active_summary("abc123456789")

    assert received["method"] == "DELETE"
    assert received["url"].endswith(
        "/internal/file_active_summary/abc123456789"
    )
    # Default service URL is the Docker-network knowledge container.
    assert received["url"].startswith("http://knowledge:8200")


@pytest.mark.asyncio
async def test_clear_sends_secret_header_when_configured(monkeypatch):
    received: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        received.update(dict(req.headers))
        return httpx.Response(204)

    _install_transport(monkeypatch, handler)
    monkeypatch.setenv("KNOWLEDGE_WEBHOOK_SECRET", "shared-secret")

    await summaries_router._clear_knowledge_active_summary("abc123456789")

    # Same header name knowledge's webhook handlers expect — the
    # service-to-service DELETE rides the same auth channel.
    assert received.get("x-webhook-secret") == "shared-secret"


@pytest.mark.asyncio
async def test_clear_omits_secret_when_unset(monkeypatch):
    received: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        received.update(dict(req.headers))
        return httpx.Response(204)

    _install_transport(monkeypatch, handler)
    monkeypatch.delenv("KNOWLEDGE_WEBHOOK_SECRET", raising=False)

    await summaries_router._clear_knowledge_active_summary("abc123456789")
    assert "x-webhook-secret" not in received


@pytest.mark.asyncio
async def test_clear_swallows_404(monkeypatch):
    """404 is the expected branch when the user never promoted the
    summary to knowledge. ``regenerate`` must continue, not fail."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    _install_transport(monkeypatch, handler)
    monkeypatch.delenv("KNOWLEDGE_WEBHOOK_SECRET", raising=False)

    # No exception raised — best-effort semantics preserved.
    await summaries_router._clear_knowledge_active_summary("abc123456789")


@pytest.mark.asyncio
async def test_clear_swallows_5xx_and_network_errors(monkeypatch):
    """5xx and connection errors are swallowed; the regenerate flow
    must not be coupled to knowledge availability."""

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("knowledge unreachable")

    _install_transport(monkeypatch, handler)

    await summaries_router._clear_knowledge_active_summary("abc123456789")


@pytest.mark.asyncio
async def test_clear_uses_KNOWLEDGE_SERVICE_URL_override(monkeypatch):
    received: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        received["url"] = str(req.url)
        return httpx.Response(204)

    _install_transport(monkeypatch, handler)
    monkeypatch.setenv("KNOWLEDGE_SERVICE_URL", "http://kn-test:9999")
    monkeypatch.delenv("KNOWLEDGE_WEBHOOK_SECRET", raising=False)

    await summaries_router._clear_knowledge_active_summary("abc123456789")

    assert received["url"].startswith("http://kn-test:9999")
