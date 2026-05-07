"""WS event drive-scoping contract tests.

The host's ``ConnectionManager.broadcast`` filters outbound WS events
by the ``AddonEventRequest.drive`` top-level field. When an addon
worker emits an event whose ``data`` carries a per-file ``drive``,
the helper MUST also lift that drive to the request's top-level so
viewers without access to the protected drive are excluded.

Without this lift, a Whisper / refine / summaries event for a
protected drive would broadcast to every connected viewer, leaking
indexing activity / file_ids to passwordless viewers.

Hako pattern: ``HpeftQ_io8n7sJ5xxlasC``.

These tests assert the wire shape of the POST body the helper sends
to the host's ``/api/internal/addon-events`` endpoint.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.workers import refine, summaries, whisper


def _capture_post(captured: list[dict[str, Any]]) -> httpx.MockTransport:
    """Build a MockTransport that records request JSON bodies."""

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            captured.append(json.loads(request.content))
        except json.JSONDecodeError:
            captured.append({"_raw": request.content.decode("utf-8", "replace")})
        return httpx.Response(200, json={"ok": True})

    return httpx.MockTransport(handler)


_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _patched_factory(transport: httpx.MockTransport):
    """Factory that mimics ``httpx.AsyncClient(...)`` while injecting transport."""

    def _factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return _REAL_ASYNC_CLIENT(transport=transport, **kwargs)

    return _factory


@pytest.fixture()
def patched_httpx(monkeypatch):
    """Yield a ``(transport_setter, captured_list)`` pair.

    monkeypatch.setattr is more reliable than ``with patch.object`` in
    this codebase (the workers ``import httpx`` lazily inside the
    helper, and re-entering a ``with patch.object`` block per
    parametrize seems to cause undelivered state in some
    pytest-asyncio + python-3.12 combos).
    """
    captured: list[dict[str, Any]] = []

    def install(transport: httpx.MockTransport) -> None:
        monkeypatch.setattr(httpx, "AsyncClient", _patched_factory(transport))

    return install, captured


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module",
    [whisper, refine, summaries],
    ids=["whisper", "refine", "summaries"],
)
async def test_emit_ws_event_lifts_drive_to_top_level(
    module, patched_httpx
) -> None:
    install, captured = patched_httpx
    install(_capture_post(captured))

    await module._emit_ws_event(
        "intelligence.test.event",
        {"file_id": "f-1", "drive": "private", "extra": 42},
    )

    assert len(captured) == 1, f"{module.__name__} must POST exactly once"
    body = captured[0]
    # The drive must be at top level so ConnectionManager.broadcast
    # can filter on it. Without this, passwordless viewers would
    # receive activity events for protected drives.
    assert body.get("drive") == "private", (
        f"{module.__name__}._emit_ws_event must lift drive to top level"
    )
    assert body.get("event") == "intelligence.test.event"
    # The original drive remains inside data so existing consumers
    # that read data.drive (frontend hooks) keep working.
    assert body["data"]["drive"] == "private"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module",
    [whisper, refine, summaries],
    ids=["whisper", "refine", "summaries"],
)
async def test_emit_ws_event_omits_drive_when_data_has_none(
    module, patched_httpx
) -> None:
    """Global events (no drive in data) must NOT set top-level drive.

    Setting drive=None on the AddonEventRequest would change the
    broadcast filter behaviour from "global broadcast" to "drive-
    scoped broadcast for None", which is undefined and risks dropping
    events.
    """
    install, captured = patched_httpx
    install(_capture_post(captured))

    await module._emit_ws_event(
        "intelligence.test.global",
        {"file_id": "f-1"},
    )

    assert len(captured) == 1
    body = captured[0]
    assert "drive" not in body or body["drive"] is None or body["drive"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module",
    [whisper, refine, summaries],
    ids=["whisper", "refine", "summaries"],
)
async def test_emit_ws_event_swallows_transport_failures(
    module, monkeypatch
) -> None:
    """The host endpoint is best-effort; transport failures must not raise."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "AsyncClient", _patched_factory(transport))
    # Must not raise — workers cannot fail on flaky core.
    await module._emit_ws_event(
        "intelligence.test.event",
        {"file_id": "f-1", "drive": "default"},
    )
