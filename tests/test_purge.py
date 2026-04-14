"""Drive-level purge: bulk removal triggered by per-drive policy off."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app import policy_client, purge


@pytest.fixture(autouse=True)
def _reset_policy_cache():
    policy_client.reset_cache()
    yield
    policy_client.reset_cache()


def _stub_search_db(file_ids_for_drive: dict[str, list[str]], monkeypatch):
    """Patch app.purge.get_search_db with an in-memory dict-backed stub.

    Captures which file_ids the SUT discovered per drive and which it
    asked _purge_file to delete, without standing up a real SQLite DB.
    """
    asked_drive: dict[str, str] = {}

    class _Query:
        def __init__(self_inner, payload):
            self_inner._payload = payload
            self_inner._drive_filter: str | None = None
            self_inner._distinct = False

        def filter(self_inner, *args, **kwargs):
            # SQLAlchemy filter API: the test sees something like
            # IndexedFile.drive == "work". The stub just records the
            # rhs string by introspecting the BinaryExpression.
            for arg in args:
                rhs = getattr(arg, "right", None)
                value = getattr(rhs, "value", None) if rhs is not None else None
                if isinstance(value, str):
                    self_inner._drive_filter = value
            return self_inner

        def distinct(self_inner):
            self_inner._distinct = True
            return self_inner

        def all(self_inner):
            if self_inner._distinct:
                return [SimpleNamespace(drive=d) for d in self_inner._payload]
            ids = self_inner._payload.get(self_inner._drive_filter, [])
            return [SimpleNamespace(file_id=fid) for fid in ids]

    class _Session:
        def query(self_inner, _col):
            return _Query(file_ids_for_drive)

    @contextmanager
    def _get_search_db():
        yield _Session()

    monkeypatch.setattr(purge, "get_search_db", _get_search_db, raising=False)
    # Also patch the late-imported version inside purge_drive / purge_disabled_drives
    import app.database as database_module
    monkeypatch.setattr(database_module, "get_search_db", _get_search_db)
    return asked_drive


def test_purge_drive_no_files_returns_zero(monkeypatch):
    _stub_search_db({"work": []}, monkeypatch)
    deleted: list[str] = []
    monkeypatch.setattr("app.indexer._purge_file", lambda fid: deleted.append(fid))
    monkeypatch.setattr("app.search.invalidate_similar_cache", lambda: 0)

    assert purge.purge_drive("work") == 0
    assert deleted == []


def test_purge_drive_calls_per_file(monkeypatch):
    _stub_search_db({"work": ["a", "b", "c"]}, monkeypatch)
    deleted: list[str] = []
    monkeypatch.setattr("app.indexer._purge_file", lambda fid: deleted.append(fid))
    invalidated = MagicMock()
    monkeypatch.setattr("app.search.invalidate_similar_cache", invalidated)

    assert purge.purge_drive("work") == 3
    assert deleted == ["a", "b", "c"]
    invalidated.assert_called_once()


def test_purge_drive_continues_on_per_file_failure(monkeypatch):
    _stub_search_db({"work": ["a", "b", "c"]}, monkeypatch)
    deleted: list[str] = []

    def boom_on_b(fid):
        if fid == "b":
            raise RuntimeError("boom")
        deleted.append(fid)

    monkeypatch.setattr("app.indexer._purge_file", boom_on_b)
    monkeypatch.setattr("app.search.invalidate_similar_cache", lambda: 0)

    # 2 succeeded, b failed → total purged = 2
    assert purge.purge_drive("work") == 2
    assert deleted == ["a", "c"]


@pytest.mark.asyncio
async def test_purge_disabled_drives_only_purges_off(monkeypatch):
    _stub_search_db({"work": ["w1"], "private": ["p1"]}, monkeypatch)

    async def fake_policy(drive, feature):
        # work is off, private stays on
        return drive != "work"

    monkeypatch.setattr(purge, "is_feature_enabled", fake_policy, raising=False)
    monkeypatch.setattr(policy_client, "is_feature_enabled", fake_policy)

    purged: list[str] = []
    monkeypatch.setattr("app.indexer._purge_file", lambda fid: purged.append(fid))
    monkeypatch.setattr("app.search.invalidate_similar_cache", lambda: 0)

    result = await purge.purge_disabled_drives()
    assert result == {"work": 1}
    assert purged == ["w1"]


@pytest.mark.asyncio
async def test_purge_disabled_drives_skips_on_policy_lookup_failure(monkeypatch):
    """A policy lookup failure must not silently destroy data."""
    _stub_search_db({"work": ["w1"]}, monkeypatch)

    async def boom(drive, feature):
        raise RuntimeError("network down")

    monkeypatch.setattr(policy_client, "is_feature_enabled", boom)

    purged: list[str] = []
    monkeypatch.setattr("app.indexer._purge_file", lambda fid: purged.append(fid))
    monkeypatch.setattr("app.search.invalidate_similar_cache", lambda: 0)

    result = await purge.purge_disabled_drives()
    assert result == {}
    assert purged == []
