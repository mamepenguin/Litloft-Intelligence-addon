"""Every addon event must carry the core's internal secret.

The core's ``POST /api/internal/addon-events`` rejects a request with 403
when ``CORE_INTERNAL_SECRET`` is set and the ``X-Internal-Secret`` header
is missing. Production sets it, so a poster that forgets the header has
every event silently dropped (SPEC-ADDON-057 R-5 saw this on
``processing.updated``).

One helper owns the endpoint, the URL resolution, the header and the
best-effort failure posture. The detector below fails when a module posts
to ``addon-events`` on its own, so a new poster cannot bypass the helper
and reintroduce the 403.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
HELPER_FILE = APP_DIR / "addon_events.py"

_REAL_ASYNC_CLIENT = httpx.AsyncClient
_REAL_CLIENT = httpx.Client


def _factory(transport: httpx.MockTransport, real):
    def _build(*args: Any, **kwargs: Any):
        kwargs.pop("transport", None)
        return real(transport=transport, **kwargs)

    return _build


def _install(monkeypatch, captured: list[dict], *, fail: bool = False) -> None:
    """Record every request and answer 200, or fail the connection."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append({
            "url": str(request.url),
            "headers": dict(request.headers),
            "json": json.loads(request.content) if request.content else None,
        })
        if fail:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "AsyncClient", _factory(transport, _REAL_ASYNC_CLIENT))
    monkeypatch.setattr(httpx, "Client", _factory(transport, _REAL_CLIENT))


def test_only_the_shared_helper_posts_to_addon_events() -> None:
    """A poster outside the helper would skip the header and be dropped."""
    offenders = sorted(
        str(path.relative_to(APP_DIR))
        for path in APP_DIR.rglob("*.py")
        if path != HELPER_FILE and re.search(r"addon-events", path.read_text("utf-8"))
    )
    assert offenders == [], (
        "these modules post to /addon-events directly; route them through "
        f"app.addon_events: {offenders}"
    )


async def test_async_post_sends_secret_header_when_configured(
    monkeypatch,
) -> None:
    from app.addon_events import post_addon_event

    monkeypatch.setenv("CORE_INTERNAL_SECRET", "s3cret")
    captured: list[dict] = []
    _install(monkeypatch, captured)

    await post_addon_event("intelligence.test.event", {"file_id": "f-1"})

    assert len(captured) == 1
    assert captured[0]["headers"].get("x-internal-secret") == "s3cret"


async def test_async_post_omits_secret_header_when_unset(monkeypatch) -> None:
    from app.addon_events import post_addon_event

    monkeypatch.delenv("CORE_INTERNAL_SECRET", raising=False)
    captured: list[dict] = []
    _install(monkeypatch, captured)

    await post_addon_event("intelligence.test.event", {"file_id": "f-1"})

    assert len(captured) == 1
    assert "x-internal-secret" not in captured[0]["headers"]


async def test_async_post_swallows_transport_failure(monkeypatch) -> None:
    from app.addon_events import post_addon_event

    monkeypatch.setenv("CORE_INTERNAL_SECRET", "s3cret")
    captured: list[dict] = []
    _install(monkeypatch, captured, fail=True)

    await post_addon_event("intelligence.test.event", {"file_id": "f-1"})


def test_sync_post_sends_secret_header_when_configured(monkeypatch) -> None:
    from app.addon_events import post_addon_event_sync

    monkeypatch.setenv("CORE_INTERNAL_SECRET", "s3cret")
    captured: list[dict] = []
    _install(monkeypatch, captured)

    post_addon_event_sync("intelligence.test.event", {"file_id": "f-1"})

    assert len(captured) == 1
    assert captured[0]["headers"].get("x-internal-secret") == "s3cret"


def test_sync_post_omits_secret_header_when_unset(monkeypatch) -> None:
    from app.addon_events import post_addon_event_sync

    monkeypatch.delenv("CORE_INTERNAL_SECRET", raising=False)
    captured: list[dict] = []
    _install(monkeypatch, captured)

    post_addon_event_sync("intelligence.test.event", {"file_id": "f-1"})

    assert len(captured) == 1
    assert "x-internal-secret" not in captured[0]["headers"]


def test_sync_post_swallows_transport_failure(monkeypatch) -> None:
    from app.addon_events import post_addon_event_sync

    monkeypatch.setenv("CORE_INTERNAL_SECRET", "s3cret")
    captured: list[dict] = []
    _install(monkeypatch, captured, fail=True)

    post_addon_event_sync("intelligence.test.event", {"file_id": "f-1"})


def test_processing_updated_carries_the_secret_header(monkeypatch) -> None:
    """SPEC-ADDON-057 R-5: without the header the core 403s every event."""
    from app.processing_events import emit_processing_updated

    monkeypatch.setenv("CORE_INTERNAL_SECRET", "s3cret")
    captured: list[dict] = []
    _install(monkeypatch, captured)

    emit_processing_updated("d1", ["f-1"])

    assert len(captured) == 1
    assert captured[0]["json"]["event"] == "processing.updated"
    assert captured[0]["headers"].get("x-internal-secret") == "s3cret"
