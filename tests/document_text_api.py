"""``GET /files/{file_id}/document-text`` behind the addon's HTTP app, with the
files it reads registered in a real search DB."""

from __future__ import annotations

from pathlib import Path

import pytest

import app.database as database
from app.models import IndexedFile
from tests.search_endpoint_db import (  # noqa: F401
    DRIVE,
    OTHER,
    api,
    patch_everywhere,
    search_index,
    seed,
    statements,
)


def path_of(file_id: str) -> str:
    return f"/files/{file_id}/document-text"


class Reader:
    def __init__(self, get, seed_):  # noqa: F811
        self._get = get
        self.seed = seed_

    def register(
        self, file_id: str, path: Path, mime: str, *, drive: str = DRIVE, active: bool = True,
    ) -> None:
        size = path.stat().st_size if path.exists() else 1
        with database.get_search_db() as s:
            s.add(IndexedFile(
                file_id=file_id, drive=drive, filename=path.name, file_path=str(path),
                file_type="document", mime_type=mime, file_size=size, active=active,
            ))

    def get(self, file_id: str, drive: str | None = DRIVE, **params):
        return self._get(path_of(file_id), params, drive=drive)

    def ok(self, file_id: str, **params) -> dict:
        response = self.get(file_id, **params)
        assert response.status_code == 200, response.text
        return response.json()

    def walk(self, file_id: str, **params) -> list[dict]:
        """Every unit returned by following ``next`` from ``params`` until it is null."""
        units: list[dict] = []
        cursor = dict(params)
        for _ in range(10_000):
            body = self.ok(file_id, **cursor)
            units.extend(body["units"])
            if body["next"] is None:
                return units
            cursor = {**cursor, **body["next"]}
        raise AssertionError("next never became null")

    def full_texts(self, file_id: str) -> dict[int, str]:
        """Each unit's whole text, read one unit at a time with the largest max_chars."""
        body = self.ok(file_id, max_chars=200_000, count=1)
        count = body["unit_count"]
        texts: dict[int, str] = {}
        for number in range(1, count + 1):
            piece = self.ok(file_id, start=number, count=1, max_chars=200_000)
            assert [u["number"] for u in piece["units"]] == [number]
            texts[number] = piece["units"][0]["text"]
        return texts


@pytest.fixture()
def reader(api, seed, monkeypatch):  # noqa: F811
    patch_everywhere(monkeypatch, "validate_file_path", lambda _path: True)
    return Reader(api, seed)
