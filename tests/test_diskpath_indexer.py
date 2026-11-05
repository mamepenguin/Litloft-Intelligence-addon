"""SPEC-ADDON-062 I14: the indexer stores the resolved on-disk path, and
reconcile corrects an NFC row to it without clearing flags or embeddings and
without queueing a job."""

from __future__ import annotations

import os
import unicodedata

import pytest
from sqlalchemy import text

from tests.idf_term_db import idx  # noqa: F401
from tests.nfd_fs import make_file, require_distinguishing_filesystem
from tests.test_idf_term_marks import WORDS, _manager, litloft  # noqa: F401

DRIVE = "a"
REL = "がぎ/ぱぴ.mp4"
NFD_REL = "nfd:がぎ/nfd:ぱぴ.mp4"


@pytest.fixture()
def mount(tmp_path, monkeypatch, litloft):  # noqa: F811
    root = tmp_path / "mount"
    root.mkdir()
    require_distinguishing_filesystem(root)
    from app import config

    monkeypatch.setitem(config.settings.drive_mounts, DRIVE, str(root))
    # The litloft fixture fakes the join; these tests need the real mount map.
    monkeypatch.setattr("app.indexer.resolve_file_path", config.resolve_file_path)
    return root


def _stored_path(idx, file_id="k1"):
    return idx.rows(
        f"SELECT file_path FROM indexed_files WHERE file_id = '{file_id}'"
    )[0][0]


def _seed(idx, litloft, stored):
    idx.add_file("k1", drive=DRIVE, filename="ぱぴ.mp4", transcript=WORDS, indexed=True, gen=3)
    idx.put_terms("k1", {"共通語"})
    idx.sql(f"UPDATE indexed_files SET file_path = '{stored}' WHERE file_id = 'k1'")
    litloft("k1", drive=DRIVE, file_path=REL, filename="ぱぴ.mp4")


def _job_count(idx):
    return idx.rows("SELECT count(*) FROM job_records")[0][0]


@pytest.mark.asyncio
async def test_spec_addon_062_reconcile_corrects_an_nfc_row_to_the_on_disk_path(
    idx, litloft, mount,  # noqa: F811
):
    on_disk = make_file(mount, NFD_REL)
    _seed(idx, litloft, os.fspath(mount / unicodedata.normalize("NFC", REL)))
    jobs_before = _job_count(idx)

    await _manager().reconcile()

    assert _stored_path(idx) == os.fspath(on_disk)
    assert _job_count(idx) == jobs_before


@pytest.mark.asyncio
async def test_spec_addon_062_the_correction_keeps_flags_embeddings_and_terms(
    idx, litloft, mount,  # noqa: F811
):
    on_disk = make_file(mount, NFD_REL)
    _seed(idx, litloft, os.fspath(mount / unicodedata.normalize("NFC", REL)))

    await _manager().reconcile()

    assert _stored_path(idx) == os.fspath(on_disk)
    assert idx.state("k1")[0] is True
    assert idx.terms("k1") == {"共通語"}
    assert idx.rows("SELECT count(*) FROM transcript_chunks WHERE file_id = 'k1'") == [(1,)]
    assert idx.rows("SELECT active FROM indexed_files WHERE file_id = 'k1'") == [(1,)]


@pytest.mark.asyncio
async def test_spec_addon_062_a_row_already_holding_the_on_disk_path_is_not_drift(
    idx, litloft, mount, monkeypatch,  # noqa: F811
):
    on_disk = make_file(mount, NFD_REL)
    _seed(idx, litloft, os.fspath(on_disk))
    manager = _manager()
    repaired = []

    async def spy(file_ids):
        repaired.append(list(file_ids))

    monkeypatch.setattr(manager, "handle_files_moved", spy)

    await manager.reconcile()

    assert repaired == []
    assert _stored_path(idx) == os.fspath(on_disk)


@pytest.mark.asyncio
async def test_spec_addon_062_an_unreadable_drive_for_one_pass_leaves_the_stored_path(
    idx, litloft, mount, monkeypatch, tmp_path,  # noqa: F811
):
    on_disk = make_file(mount, NFD_REL)
    _seed(idx, litloft, os.fspath(on_disk))
    hidden = tmp_path / "gone"
    mount.rename(hidden)  # the drive is not mounted for this pass

    await _manager().reconcile()

    assert _stored_path(idx) == os.fspath(on_disk)
    assert idx.state("k1")[0] is True
    assert idx.rows("SELECT active FROM indexed_files WHERE file_id = 'k1'") == [(1,)]


@pytest.mark.asyncio
async def test_spec_addon_062_a_listing_error_for_one_pass_leaves_the_stored_path(
    idx, litloft, mount, monkeypatch,  # noqa: F811
):
    on_disk = make_file(mount, NFD_REL)
    _seed(idx, litloft, os.fspath(on_disk))

    def denied(*args, **kwargs):
        raise PermissionError("unreadable")

    monkeypatch.setattr(os, "scandir", denied)
    monkeypatch.setattr(os, "listdir", denied)

    await _manager().reconcile()

    monkeypatch.undo()
    assert _stored_path(idx) == os.fspath(on_disk)


@pytest.mark.asyncio
async def test_spec_addon_062_a_move_stores_the_resolved_on_disk_path(
    idx, litloft, mount,  # noqa: F811
):
    on_disk = make_file(mount, NFD_REL)
    idx.add_file("k1", drive=DRIVE, filename="old.mp4", transcript=WORDS, indexed=True, gen=1)
    litloft("k1", drive=DRIVE, file_path=REL, filename="ぱぴ.mp4")

    await _manager().handle_files_moved(["k1"])

    assert _stored_path(idx) == os.fspath(on_disk)
