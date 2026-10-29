"""PyMuPDF is not thread-safe, so no two PDF conversions in the addon overlap:
not two reads of ``GET /files/{file_id}/document-text``, and not a read and indexing.
A read that cannot take PyMuPDF within its 10 s wait gets 429."""

from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest

from tests.document_fixtures import PDF, make_pdf
from tests.document_text_api import (  # noqa: F401
    DRIVE,
    api,
    path_of,
    reader,
    search_index,
    seed,
)


class _Overlap:
    """Wraps ``pymupdf4llm.to_markdown`` and records how many calls ran at once."""

    def __init__(self, hold: float):
        self.hold = hold
        self.running = 0
        self.most = 0
        self.calls = 0
        self._lock = threading.Lock()

    def install(self, monkeypatch) -> None:
        import pymupdf4llm

        real = pymupdf4llm.to_markdown
        overlap = self

        def spy(*args, **kwargs):
            with overlap._lock:
                overlap.running += 1
                overlap.calls += 1
                overlap.most = max(overlap.most, overlap.running)
            try:
                time.sleep(overlap.hold)
                return real(*args, **kwargs)
            finally:
                with overlap._lock:
                    overlap.running -= 1

        monkeypatch.setattr(pymupdf4llm, "to_markdown", spy)


def _client() -> httpx.AsyncClient:
    from app.main import app

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://intelligence",
        headers={"X-Lit-Drive": DRIVE},
        timeout=60,
    )


@pytest.fixture()
def pdfs(reader, tmp_path):
    paths = {}
    for file_id in ("p1", "p2"):
        path = make_pdf(tmp_path / f"{file_id}.pdf", [f"MARK-{file_id}-a", f"MARK-{file_id}-b"])
        reader.register(file_id, path, PDF)
        paths[file_id] = path
    return paths


async def test_two_pdf_reads_never_convert_at_once(pdfs, monkeypatch):
    overlap = _Overlap(hold=0.3)
    overlap.install(monkeypatch)

    async with _client() as client:
        responses = await asyncio.gather(
            client.get(path_of("p1"), params={"count": 2}),
            client.get(path_of("p2"), params={"count": 2}),
        )

    assert [r.status_code for r in responses] == [200, 200]
    assert overlap.calls == 4
    assert overlap.most == 1
    for response in responses:
        assert {u["extractor"] for u in response.json()["units"]} == {"pymupdf4llm"}


async def test_a_pdf_read_never_converts_while_indexing_converts(pdfs, monkeypatch):
    from app.extractors.pdf import PdfExtractor

    overlap = _Overlap(hold=0.5)
    overlap.install(monkeypatch)

    async with _client() as client:
        indexed, response = await asyncio.gather(
            asyncio.to_thread(PdfExtractor().extract, str(pdfs["p1"])),
            client.get(path_of("p2"), params={"count": 2}),
        )

    assert response.status_code == 200, response.text
    assert indexed.extractor == "pymupdf4llm"
    assert overlap.most == 1


async def test_a_read_that_cannot_take_pymupdf_within_10s_gets_429(pdfs, monkeypatch):
    from app.extractors import pdf

    taken = threading.Event()
    release = threading.Event()

    def hold_pymupdf():
        with pdf.PYMUPDF_LOCK:
            taken.set()
            release.wait(timeout=60)

    holder = threading.Thread(target=hold_pymupdf)
    holder.start()
    try:
        assert taken.wait(timeout=5)
        async with _client() as client:
            started = time.monotonic()
            response = await client.get(path_of("p1"))
            waited = time.monotonic() - started
    finally:
        release.set()
        holder.join()

    assert response.status_code == 429, response.text
    assert response.json()["detail"] == "Document reader busy"
    assert 9.5 <= waited < 20
