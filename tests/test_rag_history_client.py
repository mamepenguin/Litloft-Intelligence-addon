"""Tests for app.rag.history_client.

The client wraps a single httpx GET against the host's Internal API
``/viewer-history`` endpoint. The contract is intentionally narrow:
return ``list[str]`` on success, return ``[]`` on every failure path
so the caller can pass the result straight into ``file_id_scope``
without checking for None.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub heavy ML deps before importing the module under test.
for _mod in (
    "PIL",
    "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome",
    "janome.tokenizer",
    "numpy",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import httpx  # noqa: E402

from app.rag import history_client  # noqa: E402


VIEWER = "0123456789abcdef"
DRIVE = "test"


class _FakeResponse:
    """Minimal httpx-compatible response for asserting on status + json."""

    def __init__(self, *, status_code: int, payload: object | None = None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeClient:
    """Records the GET call args and returns a pre-baked response.

    Matches httpx.AsyncClient's ``async with`` context-manager protocol
    so ``async with httpx.AsyncClient(...) as c:`` works in production
    code paths without changes.
    """

    def __init__(self, response: _FakeResponse | Exception):
        self._response = response
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def get(self, url, *, params=None, headers=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _patch_client(monkeypatch, response):
    """Install a single ``_FakeClient`` instance for one fetch call."""
    fake = _FakeClient(response)

    def _factory(*args, **kwargs):
        return fake

    monkeypatch.setattr(history_client.httpx, "AsyncClient", _factory)
    return fake


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestFetchViewerHistoryHappy:
    @pytest.mark.asyncio
    async def test_returns_file_ids_on_200(self, monkeypatch):
        _patch_client(
            monkeypatch,
            _FakeResponse(
                status_code=200,
                payload={"file_ids": ["fid1", "fid2", "fid3"]},
            ),
        )
        result = await history_client.fetch_viewer_history(
            viewer_id=VIEWER,
            drive=DRIVE,
            after=None,
            before=None,
            kind="viewed",
        )
        assert result == ["fid1", "fid2", "fid3"]

    @pytest.mark.asyncio
    async def test_passes_required_query_params(self, monkeypatch):
        fake = _patch_client(
            monkeypatch,
            _FakeResponse(status_code=200, payload={"file_ids": []}),
        )
        await history_client.fetch_viewer_history(
            viewer_id=VIEWER,
            drive=DRIVE,
            after=None,
            before=None,
            kind="not_viewed",
        )
        assert len(fake.calls) == 1
        params = fake.calls[0]["params"]
        assert params["viewer_id"] == VIEWER
        assert params["drive"] == DRIVE
        assert params["kind"] == "not_viewed"
        # No window → after/before omitted entirely; the host treats
        # missing keys as unbounded.
        assert "after" not in params
        assert "before" not in params

    @pytest.mark.asyncio
    async def test_serialises_window_as_naive_iso(self, monkeypatch):
        fake = _patch_client(
            monkeypatch,
            _FakeResponse(status_code=200, payload={"file_ids": []}),
        )
        # Aware datetime — the client must strip tz to match the host's
        # naive ``last_played_at`` column convention.
        await history_client.fetch_viewer_history(
            viewer_id=VIEWER,
            drive=DRIVE,
            after=datetime(2026, 4, 19, 0, 0, 0, tzinfo=UTC),
            before=datetime(2026, 4, 26, 0, 0, 0, tzinfo=UTC),
            kind="viewed",
        )
        params = fake.calls[0]["params"]
        # Naive ISO has no offset suffix.
        assert params["after"] == "2026-04-19T00:00:00"
        assert params["before"] == "2026-04-26T00:00:00"

    @pytest.mark.asyncio
    async def test_attaches_internal_secret_header_when_set(
        self, monkeypatch
    ):
        monkeypatch.setenv("CORE_INTERNAL_SECRET", "shared-secret")
        fake = _patch_client(
            monkeypatch,
            _FakeResponse(status_code=200, payload={"file_ids": []}),
        )
        await history_client.fetch_viewer_history(
            viewer_id=VIEWER,
            drive=DRIVE,
            after=None,
            before=None,
            kind="viewed",
        )
        headers = fake.calls[0]["headers"]
        assert headers.get("x-internal-secret") == "shared-secret"

    @pytest.mark.asyncio
    async def test_omits_secret_header_when_unset(self, monkeypatch):
        monkeypatch.delenv("CORE_INTERNAL_SECRET", raising=False)
        fake = _patch_client(
            monkeypatch,
            _FakeResponse(status_code=200, payload={"file_ids": []}),
        )
        await history_client.fetch_viewer_history(
            viewer_id=VIEWER,
            drive=DRIVE,
            after=None,
            before=None,
            kind="viewed",
        )
        headers = fake.calls[0]["headers"]
        assert "x-internal-secret" not in headers


# ---------------------------------------------------------------------------
# Failure paths — every one must collapse to []
# ---------------------------------------------------------------------------


class TestFetchViewerHistoryFailures:
    @pytest.mark.asyncio
    async def test_http_error_returns_empty(self, monkeypatch):
        _patch_client(
            monkeypatch,
            httpx.ConnectError("connection refused"),
        )
        result = await history_client.fetch_viewer_history(
            viewer_id=VIEWER,
            drive=DRIVE,
            after=None,
            before=None,
            kind="viewed",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_non_200_status_returns_empty(self, monkeypatch):
        _patch_client(
            monkeypatch,
            _FakeResponse(status_code=400, payload={"detail": "bad"}),
        )
        result = await history_client.fetch_viewer_history(
            viewer_id=VIEWER,
            drive=DRIVE,
            after=None,
            before=None,
            kind="viewed",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_404_drive_unknown_returns_empty(self, monkeypatch):
        _patch_client(
            monkeypatch,
            _FakeResponse(status_code=404, payload=None),
        )
        result = await history_client.fetch_viewer_history(
            viewer_id=VIEWER,
            drive=DRIVE,
            after=None,
            before=None,
            kind="viewed",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_non_json_body_returns_empty(self, monkeypatch):
        _patch_client(
            monkeypatch,
            _FakeResponse(status_code=200, payload=ValueError("not json")),
        )
        result = await history_client.fetch_viewer_history(
            viewer_id=VIEWER,
            drive=DRIVE,
            after=None,
            before=None,
            kind="viewed",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_payload_missing_file_ids_returns_empty(self, monkeypatch):
        _patch_client(
            monkeypatch,
            _FakeResponse(status_code=200, payload={"other": "shape"}),
        )
        result = await history_client.fetch_viewer_history(
            viewer_id=VIEWER,
            drive=DRIVE,
            after=None,
            before=None,
            kind="viewed",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_drops_non_string_entries(self, monkeypatch):
        _patch_client(
            monkeypatch,
            _FakeResponse(
                status_code=200,
                payload={"file_ids": ["good", 42, None, "good2"]},
            ),
        )
        result = await history_client.fetch_viewer_history(
            viewer_id=VIEWER,
            drive=DRIVE,
            after=None,
            before=None,
            kind="viewed",
        )
        assert result == ["good", "good2"]
