"""SPEC-ADDON-060 I11: the resumed text backlog runs below new uploads."""

from __future__ import annotations

from app.indexer import IndexManager, TaskType
from tests.ocr_fixtures import build_search_db, seed_indexed_file

PDF = "application/pdf"

ALL_DONE = dict(
    metadata_indexed=True,
    clip_indexed=True,
    clip_thumbnail_indexed=True,
    whisper_indexed=True,
    tfidf_keywords_indexed=True,
)


async def test_spec_addon_060_i11_a_resumed_text_task_queues_below_a_new_file(
    tmp_path, monkeypatch,
):
    db = build_search_db(tmp_path, monkeypatch)
    seed_indexed_file(
        db, "backlog", PDF, "/drives/a/backlog.pdf",
        text_indexed=False, **ALL_DONE,
    )
    manager = IndexManager()

    await manager._resume_incomplete()
    await manager._queue_file_tasks({"id": "new", "mime_type": PDF})
    tasks = await manager._collect_batch(TaskType.TEXT_CONTENT, 10)

    assert [(task.file_id, task.priority) for task in tasks] == [
        ("new", 0),
        ("backlog", -10),
    ]
