"""SPEC-ADDON-001: a `.loft` finds its `.vtt` whatever the normalization or glob syntax of either name."""
from __future__ import annotations

import logging
import os
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import IndexedFile, TranscriptChunk


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


CAFE = "Café ガイド"
# é decomposed, ガ composed: a name that is neither NFC nor NFD.
CAFE_MIXED = "Café ガイド"
assert CAFE_MIXED not in (nfc(CAFE), nfd(CAFE))


@pytest.fixture()
def Session(tmp_path, monkeypatch):
    eng = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(eng)
    with eng.begin() as conn:
        for table in ("fts_transcripts", "fts_transcripts_word"):
            conn.execute(text(
                f"CREATE TABLE IF NOT EXISTS {table} "
                "(file_id TEXT, chunk_index INTEGER, text TEXT)"
            ))
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS vec_text "
            "(embedding_id TEXT PRIMARY KEY, vector BLOB)"
        ))
    S = sessionmaker(bind=eng, expire_on_commit=False)

    @contextmanager
    def _get_search_db():
        s = S()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    @contextmanager
    def _get_search_db_read():
        s = S()
        try:
            yield s
        finally:
            s.close()

    for target in ("app.database", "app.workers.whisper", "app.indexer"):
        monkeypatch.setattr(f"{target}.get_search_db", _get_search_db, raising=False)
        monkeypatch.setattr(
            f"{target}.get_search_db_read", _get_search_db_read, raising=False
        )
    yield S
    eng.dispose()


@pytest.fixture()
def unlistable(monkeypatch):
    """Make the given directories fail to list or search, as a permission error would.

    Listing the directory and stat-ing anything inside it both raise, so a
    check that stats a path in the folder before listing it is caught too.
    """
    blocked: set[str] = set()
    real_scandir, real_listdir, real_stat = os.scandir, os.listdir, os.stat

    def _deny(path):
        raise PermissionError(13, "Permission denied", os.fsdecode(path))

    def _scandir(path="."):
        if os.path.abspath(os.fsdecode(path)) in blocked:
            _deny(path)
        return real_scandir(path)

    def _listdir(path="."):
        if os.path.abspath(os.fsdecode(path)) in blocked:
            _deny(path)
        return real_listdir(path)

    def _stat(path, *args, **kwargs):
        if not isinstance(path, int):
            if os.path.dirname(os.path.abspath(os.fsdecode(path))) in blocked:
                _deny(path)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", _scandir)
    monkeypatch.setattr(os, "listdir", _listdir)
    monkeypatch.setattr(os, "stat", _stat)

    def _block(directory: Path) -> None:
        blocked.add(os.path.abspath(directory))

    return _block


def _vtt(path: Path, cue: str) -> None:
    path.write_text(
        f"WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n{cue}\n", encoding="utf-8"
    )


def _seed_loft(Session, file_id: str, loft: Path, *, whisper_indexed: bool) -> None:
    from app.workers.whisper import LOFT_MIME

    s = Session()
    s.add(IndexedFile(
        file_id=file_id,
        drive="d1",
        filename=loft.name,
        file_path=str(loft),
        file_type="other",
        mime_type=LOFT_MIME,
        file_size=2,
        active=True,
        metadata_indexed=True,
        clip_indexed=False,
        whisper_indexed=whisper_indexed,
        text_indexed=False,
        title=loft.stem,
        description="",
        tags_text="",
    ))
    s.commit()
    s.close()


def _make_loft(directory: Path, stem: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    loft = directory / f"{stem}.loft"
    loft.write_text("{}", encoding="utf-8")
    return loft


async def _index(file_id: str) -> None:
    from app.workers import whisper as whisper_module

    with (
        patch.object(whisper_module, "validate_file_path", return_value=True),
        patch(
            "app.policy_client.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
    ):
        await whisper_module.index_whisper(file_id)


def _chunks(Session, file_id: str) -> list[str]:
    s = Session()
    try:
        return [
            c.text for c in s.query(TranscriptChunk)
            .filter_by(file_id=file_id).order_by(TranscriptChunk.chunk_index)
        ]
    finally:
        s.close()


def _whisper_indexed(Session, file_id: str) -> bool:
    s = Session()
    try:
        return s.query(IndexedFile).filter_by(file_id=file_id).one().whisper_indexed
    finally:
        s.close()


def _reconcile_vtt() -> None:
    from app.indexer import IndexManager

    IndexManager.__new__(IndexManager)._reset_loft_refs_with_new_vtt()


# (stored .loft stem, on-disk .vtt name)
FINDS = [
    pytest.param(nfc(CAFE), nfd(CAFE) + ".vtt", id="nfc-stored-nfd-vtt"),
    pytest.param(nfd(CAFE), nfc(CAFE) + ".vtt", id="nfd-stored-nfc-vtt"),
    pytest.param(nfc(CAFE), CAFE_MIXED + ".vtt", id="mixed-form-vtt"),
    pytest.param("Live [2026] *best?", "Live [2026] *best?.vtt", id="glob-metacharacters"),
    pytest.param(nfc("Été [ライブ]"), nfd("Été [ライブ]") + ".vtt", id="metacharacters-and-nfd"),
]


class TestIndexing:
    @pytest.mark.parametrize("stored, vtt_name", FINDS)
    async def test_spec_addon_001_indexes_adjacent_vtt(
        self, Session, tmp_path, stored, vtt_name
    ):
        loft = _make_loft(tmp_path / "dir", stored)
        _vtt(tmp_path / "dir" / vtt_name, "the transcript")
        _seed_loft(Session, "f1", loft, whisper_indexed=False)

        await _index("f1")

        assert _chunks(Session, "f1") == ["the transcript"]

    async def test_spec_addon_001_metacharacters_never_widen_the_match(
        self, Session, tmp_path
    ):
        loft = _make_loft(tmp_path / "dir", "Title[x]")
        _vtt(tmp_path / "dir" / "Titlex.vtt", "the transcript")
        _seed_loft(Session, "f1", loft, whisper_indexed=False)

        await _index("f1")

        assert _chunks(Session, "f1") == []

    async def test_spec_addon_001_exact_stem_vtt_wins_after_nfc(self, Session, tmp_path):
        # The NFD sibling sorts first on disk ('e' < 'é'), so only the
        # exact-name preference picks the NFC one.
        d = tmp_path / "dir"
        loft = _make_loft(d, nfd("Café"))
        _vtt(d / (nfc("Café") + ".vtt"), "exact")
        _vtt(d / (nfd("Café") + " (1).vtt"), "prefix sibling")
        assert sorted(os.listdir(d))[0] == nfd("Café") + " (1).vtt"
        _seed_loft(Session, "f1", loft, whisper_indexed=False)

        await _index("f1")

        assert _chunks(Session, "f1") == ["exact"]

    async def test_spec_addon_001_unlistable_directory_marks_indexed_and_warns(
        self, Session, tmp_path, unlistable, caplog
    ):
        loft = _make_loft(tmp_path / "locked", "Clip")
        _vtt(tmp_path / "locked" / "Clip.vtt", "unreachable")
        _seed_loft(Session, "flocked00001", loft, whisper_indexed=False)
        unlistable(tmp_path / "locked")

        with caplog.at_level(logging.WARNING):
            await _index("flocked00001")

        assert _whisper_indexed(Session, "flocked00001") is True
        assert _chunks(Session, "flocked00001") == []
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "flocked00001" in warnings[0].getMessage()


class TestReconcile:
    @pytest.mark.parametrize("stored, vtt_name", FINDS)
    def test_spec_addon_001_requeues_loft_with_adjacent_vtt(
        self, Session, tmp_path, stored, vtt_name
    ):
        loft = _make_loft(tmp_path / "dir", stored)
        _vtt(tmp_path / "dir" / vtt_name, "late transcript")
        _seed_loft(Session, "f1", loft, whisper_indexed=True)

        _reconcile_vtt()

        assert _whisper_indexed(Session, "f1") is False

    def test_spec_addon_001_metacharacters_never_widen_the_requeue(
        self, Session, tmp_path
    ):
        loft = _make_loft(tmp_path / "dir", "Title[x]")
        _vtt(tmp_path / "dir" / "Titlex.vtt", "the transcript")
        _seed_loft(Session, "f1", loft, whisper_indexed=True)

        _reconcile_vtt()

        assert _whisper_indexed(Session, "f1") is True

    def test_spec_addon_001_loft_with_chunks_is_not_requeued(self, Session, tmp_path):
        loft = _make_loft(tmp_path / "dir", nfc(CAFE))
        _vtt(tmp_path / "dir" / (nfd(CAFE) + ".vtt"), "already indexed")
        _seed_loft(Session, "f1", loft, whisper_indexed=True)
        s = Session()
        s.add(TranscriptChunk(
            file_id="f1", chunk_index=0, text="already indexed",
            timestamp_start=0.0, timestamp_end=2.0,
        ))
        s.commit()
        s.close()

        _reconcile_vtt()

        assert _whisper_indexed(Session, "f1") is True

    @pytest.mark.parametrize("locked_count", [0, 1])
    def test_spec_addon_001_reconcile_warns_only_when_a_folder_was_skipped(
        self, Session, tmp_path, unlistable, caplog, locked_count
    ):
        for i in range(locked_count):
            locked = _make_loft(tmp_path / f"locked{i}", "Clip")
            _seed_loft(Session, f"locked{i}", locked, whisper_indexed=True)
            unlistable(tmp_path / f"locked{i}")
        ok = _make_loft(tmp_path / "open", "Clip")
        _seed_loft(Session, "open", ok, whisper_indexed=True)

        with caplog.at_level(logging.WARNING):
            _reconcile_vtt()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == locked_count

    def test_spec_addon_001_unlistable_directories_are_skipped_and_counted_once(
        self, Session, tmp_path, unlistable, caplog
    ):
        for i in (1, 2):
            locked = _make_loft(tmp_path / f"locked{i}", "Clip")
            _vtt(tmp_path / f"locked{i}" / "Clip.vtt", "unreachable")
            _seed_loft(Session, f"locked{i}", locked, whisper_indexed=True)
            unlistable(tmp_path / f"locked{i}")
        ok = _make_loft(tmp_path / "open", nfc(CAFE))
        _vtt(tmp_path / "open" / (nfd(CAFE) + ".vtt"), "late transcript")
        _seed_loft(Session, "open", ok, whisper_indexed=True)

        with caplog.at_level(logging.WARNING):
            _reconcile_vtt()

        assert _whisper_indexed(Session, "open") is False
        assert _whisper_indexed(Session, "locked1") is True
        assert _whisper_indexed(Session, "locked2") is True
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "2" in warnings[0].getMessage()
