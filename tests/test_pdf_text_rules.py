"""SPEC-ADDON-060 I6: the startup check that re-extracts PDFs once per rule version."""

from __future__ import annotations

import logging

import pytest

from app import database
from app.indexer import IndexManager, TaskType
from tests import pdf_footnote_api as api
from tests.ocr_fixtures import build_search_db, seed_indexed_file

PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TEXT = "text/plain"
PNG = "image/png"

OTHERS = ("docx", "text", "picture")


def _seed(db) -> None:
    seed_indexed_file(db, "pdfactive", PDF, "/drives/a/active.pdf", text_indexed=True)
    seed_indexed_file(
        db, "pdftrashed", PDF, "/drives/a/trashed.pdf", active=False, text_indexed=True,
    )
    seed_indexed_file(db, "docx", DOCX, "/drives/a/a.docx", text_indexed=True)
    seed_indexed_file(db, "text", TEXT, "/drives/a/a.txt", text_indexed=True)
    seed_indexed_file(db, "picture", PNG, "/drives/a/a.png", text_indexed=True)


def _version(db) -> str | None:
    tables = db.rows(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='index_meta'"
    )
    if not tables:
        return None
    return db.scalar("SELECT value FROM index_meta WHERE key = 'pdf_text_rules'")


def test_spec_addon_060_i6_the_first_start_clears_every_pdf_and_no_other_file(
    tmp_path, monkeypatch, caplog,
):
    db = build_search_db(tmp_path, monkeypatch)
    _seed(db)

    with caplog.at_level(logging.INFO):
        api.check_text_rules()

    assert db.flag("pdfactive", "text_indexed") == 0
    assert db.flag("pdftrashed", "text_indexed") == 0
    assert [db.flag(file_id, "text_indexed") for file_id in OTHERS] == [1, 1, 1]
    assert _version(db) == api.rules_version()
    assert any(
        record.levelno == logging.INFO and api.rules_version() in record.getMessage()
        for record in caplog.records
    )


def test_spec_addon_060_i6_a_later_start_with_the_same_version_changes_nothing(
    tmp_path, monkeypatch,
):
    db = build_search_db(tmp_path, monkeypatch)
    _seed(db)
    api.check_text_rules()
    db.set_flag("pdfactive", "text_indexed", True)
    db.set_flag("pdftrashed", "text_indexed", True)

    api.check_text_rules()

    assert db.flag("pdfactive", "text_indexed") == 1
    assert db.flag("pdftrashed", "text_indexed") == 1
    assert _version(db) == api.rules_version()


def test_spec_addon_060_i6_the_eval_snapshot_records_the_version_without_a_reset(
    tmp_path, monkeypatch,
):
    db = build_search_db(tmp_path, monkeypatch)
    _seed(db)
    monkeypatch.setenv("INTELLIGENCE_SEARCH_DB_PATH", str(db.path))

    api.check_text_rules()

    assert db.flag("pdfactive", "text_indexed") == 1
    assert db.flag("pdftrashed", "text_indexed") == 1
    assert _version(db) == api.rules_version()


def test_spec_addon_060_i6_a_failed_check_leaves_both_the_flags_and_the_version(
    tmp_path, monkeypatch, caplog,
):
    db = build_search_db(tmp_path, monkeypatch)
    _seed(db)

    def broken(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(database, "_upsert_index_meta", broken)

    with caplog.at_level(logging.ERROR):
        api.check_text_rules()

    assert db.flag("pdfactive", "text_indexed") == 1
    assert db.flag("pdftrashed", "text_indexed") == 1
    assert _version(db) is None
    assert any(
        record.levelno == logging.ERROR and record.exc_info
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_spec_addon_060_i6_the_next_resume_queues_the_reset_pdfs(
    tmp_path, monkeypatch,
):
    db = build_search_db(tmp_path, monkeypatch)
    _seed(db)
    api.check_text_rules()
    manager = IndexManager()

    await manager._resume_incomplete()
    tasks = await manager._collect_batch(TaskType.TEXT_CONTENT, 10)

    assert [task.file_id for task in tasks] == ["pdfactive"]
