"""SPEC-ADDON-036 I8: at most two requests to ``GET /files/{file_id}/document-text``
extract at once, and a third waits up to 10 s for a slot before a 429."""

from __future__ import annotations

import asyncio
import threading
import time
import zipfile

import httpx
import pytest

from tests.document_fixtures import EPUB
from tests.document_text_api import (  # noqa: F401
    DRIVE,
    api,
    path_of,
    reader,
    search_index,
    seed,
)
from tests.epub_fixtures import Item, make_epub, xhtml


class _Gate:
    """Holds every read of an archive whose name contains ``slow`` until released."""

    def __init__(self):
        self.release = threading.Event()
        self._held: set[int] = set()
        self._lock = threading.Lock()

    def held(self) -> int:
        with self._lock:
            return len(self._held)

    def install(self, monkeypatch) -> None:
        real_open = zipfile.ZipFile.open
        gate = self

        def gated(self, name, *args, **kwargs):
            if "slow" in str(self.filename):
                with gate._lock:
                    gate._held.add(threading.get_ident())
                gate.release.wait(timeout=60)
            return real_open(self, name, *args, **kwargs)

        monkeypatch.setattr(zipfile.ZipFile, "open", gated)


@pytest.fixture()
def books(reader, tmp_path):
    for file_id in ("slow1", "slow2", "fast"):
        reader.register(
            file_id,
            make_epub(tmp_path / f"{file_id}.epub", [Item("c1", "c1.xhtml", xhtml("<p>MARK</p>"))]),
            EPUB,
        )


@pytest.fixture()
def gate(monkeypatch, books):
    g = _Gate()
    g.install(monkeypatch)
    yield g
    g.release.set()


def _client() -> httpx.AsyncClient:
    from app.main import app

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://intelligence",
        headers={"X-Lit-Drive": DRIVE},
        timeout=60,
    )


async def _until_two_are_held(gate: _Gate) -> None:
    deadline = time.monotonic() + 10
    while gate.held() < 2:
        if time.monotonic() > deadline:
            pytest.fail(f"two extractions never started; {gate.held()} held")
        await asyncio.sleep(0.05)


async def test_spec_addon_036_i8_a_third_request_gets_429_after_waiting_10s(gate):
    async with _client() as client:
        slow = [asyncio.create_task(client.get(path_of(f))) for f in ("slow1", "slow2")]
        try:
            await _until_two_are_held(gate)

            started = time.monotonic()
            third = await asyncio.wait_for(client.get(path_of("fast")), timeout=30)
            waited = time.monotonic() - started
        finally:
            gate.release.set()
        first, second = await asyncio.gather(*slow)

    assert third.status_code == 429, third.text
    assert third.json()["detail"] == "Document reader busy"
    assert 9.5 <= waited < 20
    assert (first.status_code, second.status_code) == (200, 200)


async def test_spec_addon_036_i8_a_waiting_request_takes_a_slot_freed_within_10s(gate):
    async with _client() as client:
        slow = [asyncio.create_task(client.get(path_of(f))) for f in ("slow1", "slow2")]
        try:
            await _until_two_are_held(gate)
            third = asyncio.create_task(client.get(path_of("fast")))
            await asyncio.sleep(1.5)
            assert not third.done()
            gate.release.set()
            response = await asyncio.wait_for(third, timeout=30)
        finally:
            gate.release.set()
        await asyncio.gather(*slow)

    assert response.status_code == 200, response.text
    assert response.json()["units"][0]["text"].strip() == "MARK"


async def test_spec_addon_036_i8_two_requests_extract_at_once(gate):
    async with _client() as client:
        slow = [asyncio.create_task(client.get(path_of(f))) for f in ("slow1", "slow2")]
        try:
            await _until_two_are_held(gate)
        finally:
            gate.release.set()
        responses = await asyncio.gather(*slow)

    assert [r.status_code for r in responses] == [200, 200]
