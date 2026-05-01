"""Contract tests for intelligence → core ``POST /api/internal/files/bulk``.

Two layers per hako ``VHE7K0KWjIzV3M1CyfDAN`` and ``70vp3pXn2iod7ehhxcYF5``:

1. Wire shape — URL, HTTP method, request body, response parse.
2. Validator parity — every field the frontend ``FileItem`` type expects
   on the wire is accepted by the client and returned unchanged. Catches
   silent drift between core's ``FileResponse`` and the addon's consumer.

Failure tolerance: HTTP 5xx, connection errors, JSON parse errors must
all degrade to an empty hydrate map. ``execute_search`` is expected to
fall through to the ``IndexedFile`` snapshot — semantic search must
keep working when core is briefly unreachable.
"""

from __future__ import annotations

import httpx
import pytest

from app import file_hydrate


@pytest.fixture(autouse=True)
def _clear_hydrate_cache():
    """Each test starts with a fresh cache — TTL is 30s otherwise."""
    file_hydrate.cache_clear()
    yield
    file_hydrate.cache_clear()


def _install_transport(monkeypatch, handler):
    orig_async_client = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return orig_async_client(*args, **kwargs)

    monkeypatch.setattr(file_hydrate.httpx, "AsyncClient", _factory)


# Reference shape: the contract that must hold between core's
# FileResponse and the frontend's FileItem TypeScript interface.
# If a field is added/removed in either, this fixture surfaces the
# drift.
_FILE_ITEM_KEYS = {
    "id",
    "filename",
    "title",
    "description",
    "drive",
    "folder_path",
    "file_type",
    "mime_type",
    "thumbnail_url",
    "has_thumbnail",
    "file_size",
    "duration",
    "likes",
    "is_favorite",
    "tags",
    "subtitles",
    "created_at",
    "updated_at",
    "deleted_at",
    "missing_since",
}


def _make_file_item(file_id: str = "abc123def456", **overrides) -> dict:
    base = {
        "id": file_id,
        "filename": "movie.mp4",
        "title": "Movie",
        "description": "",
        "drive": "default",
        "folder_path": "videos/2026",
        "file_type": "video",
        "mime_type": "video/mp4",
        "thumbnail_url": f"/api/files/{file_id}/thumbnail",
        "has_thumbnail": True,
        "file_size": 1024,
        "duration": 120.5,
        "likes": 0,
        "is_favorite": False,
        "tags": [],
        "subtitles": [],
        "created_at": "2026-04-15T10:00:00",
        "updated_at": "2026-04-30T05:00:00",
        "deleted_at": None,
        "missing_since": None,
    }
    base.update(overrides)
    return base


# --- Layer 1: Wire shape ---


@pytest.mark.asyncio
async def test_targets_internal_files_bulk_endpoint(monkeypatch):
    received: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        received["url"] = str(req.url)
        received["method"] = req.method
        return httpx.Response(200, json={"files": [], "not_found": []})

    _install_transport(monkeypatch, handler)
    monkeypatch.delenv("HOMEVAULT_INTERNAL_API_URL", raising=False)

    await file_hydrate.hydrate_files(["abc123def456"])

    assert received["method"] == "POST"
    assert received["url"].endswith("/api/internal/files/bulk")
    assert received["url"].startswith("http://backend:8000")


@pytest.mark.asyncio
async def test_request_body_uses_file_ids_key(monkeypatch):
    received: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json as _json
        received["body"] = _json.loads(req.content.decode())
        return httpx.Response(200, json={"files": [], "not_found": []})

    _install_transport(monkeypatch, handler)

    await file_hydrate.hydrate_files(["a", "b", "c"])

    # Core's BulkFilesRequest expects ``{file_ids: [...]}`` — drift
    # check.
    assert received["body"] == {"file_ids": ["a", "b", "c"]}


@pytest.mark.asyncio
async def test_returns_empty_dict_for_empty_input(monkeypatch):
    """Short-circuit: empty input must not hit the network."""
    called = False

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"files": [], "not_found": []})

    _install_transport(monkeypatch, handler)
    result = await file_hydrate.hydrate_files([])

    assert result == {}
    assert called is False


# --- Layer 2: Validator parity ---


@pytest.mark.asyncio
async def test_parses_file_response_into_id_keyed_map(monkeypatch):
    item_a = _make_file_item(file_id="aaa111", filename="A.mp4")
    item_b = _make_file_item(file_id="bbb222", filename="B.mp4")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"files": [item_a, item_b], "not_found": ["ccc333"]},
        )

    _install_transport(monkeypatch, handler)

    result = await file_hydrate.hydrate_files(["aaa111", "bbb222", "ccc333"])

    assert set(result.keys()) == {"aaa111", "bbb222"}
    assert result["aaa111"]["filename"] == "A.mp4"
    assert result["bbb222"]["filename"] == "B.mp4"


@pytest.mark.asyncio
async def test_preserves_all_file_item_fields_unchanged(monkeypatch):
    """Parity: every key in the core FileResponse shape arrives at the
    consumer untouched. Drift detector if either side adds a field
    without the other.
    """
    item = _make_file_item(
        file_id="xyz789abc012",
        is_favorite=True,
        tags=["family", "2026"],
        subtitles=[
            {"index": 0, "language": "ja", "format": "vtt", "label": "JA"}
        ],
    )

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"files": [item], "not_found": []})

    _install_transport(monkeypatch, handler)
    result = await file_hydrate.hydrate_files(["xyz789abc012"])

    received = result["xyz789abc012"]
    # Every FileItem key the frontend expects is preserved through the
    # hydrate client. If a key is missing here, either core dropped it
    # or our client filtered it.
    assert set(received.keys()) >= _FILE_ITEM_KEYS
    # Values pass through unchanged (no normalization, no field drop).
    assert received == item


# --- Failure tolerance ---


@pytest.mark.asyncio
async def test_returns_empty_on_5xx(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    _install_transport(monkeypatch, handler)

    result = await file_hydrate.hydrate_files(["any"])
    assert result == {}


@pytest.mark.asyncio
async def test_returns_empty_on_4xx(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "bad request"})

    _install_transport(monkeypatch, handler)

    result = await file_hydrate.hydrate_files(["any"])
    assert result == {}


@pytest.mark.asyncio
async def test_returns_empty_on_connection_error(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _install_transport(monkeypatch, handler)

    result = await file_hydrate.hydrate_files(["any"])
    assert result == {}


@pytest.mark.asyncio
async def test_returns_empty_on_malformed_response(monkeypatch):
    """Core promises ``{files: [...], not_found: [...]}`` but if a future
    drift removes ``files``, we must not crash search.
    """
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    _install_transport(monkeypatch, handler)

    result = await file_hydrate.hydrate_files(["any"])
    assert result == {}


# --- Cache behavior ---


@pytest.mark.asyncio
async def test_cache_hit_skips_network(monkeypatch):
    """Two consecutive calls for the same id set hit the network once."""
    call_count = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(
            200,
            json={"files": [_make_file_item("x")], "not_found": []},
        )

    _install_transport(monkeypatch, handler)

    await file_hydrate.hydrate_files(["x"])
    await file_hydrate.hydrate_files(["x"])

    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_cache_keys_by_sorted_id_set(monkeypatch):
    """Two calls with the same ids in different order share a cache slot.
    Pagination over the same result page hits the cache regardless of
    presentation order.
    """
    call_count = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(
            200,
            json={
                "files": [_make_file_item("x"), _make_file_item("y")],
                "not_found": [],
            },
        )

    _install_transport(monkeypatch, handler)

    await file_hydrate.hydrate_files(["x", "y"])
    await file_hydrate.hydrate_files(["y", "x"])

    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_cache_miss_on_different_id_set(monkeypatch):
    call_count = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json={"files": [], "not_found": []})

    _install_transport(monkeypatch, handler)

    await file_hydrate.hydrate_files(["x"])
    await file_hydrate.hydrate_files(["y"])

    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_cache_failure_not_persisted(monkeypatch):
    """A failed call must not poison the cache — the next call retries."""
    state = {"calls": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={"files": [_make_file_item("x")], "not_found": []},
        )

    _install_transport(monkeypatch, handler)

    first = await file_hydrate.hydrate_files(["x"])
    assert first == {}

    second = await file_hydrate.hydrate_files(["x"])
    assert "x" in second


# --- Configurable base URL ---


@pytest.mark.asyncio
async def test_respects_homevault_internal_api_url_env(monkeypatch):
    received: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        received["url"] = str(req.url)
        return httpx.Response(200, json={"files": [], "not_found": []})

    _install_transport(monkeypatch, handler)
    monkeypatch.setenv(
        "HOMEVAULT_INTERNAL_API_URL", "http://core:9000/api/internal"
    )

    await file_hydrate.hydrate_files(["x"])

    assert received["url"].startswith("http://core:9000/api/internal")
