"""Unit tests for IndexManager._enqueue deduplication logic.

Verifies that repeated calls with the same (file_id, task_type) do not
accumulate duplicate entries in the asyncio.PriorityQueue, and that
force=True bypasses the guard for prioritize() use cases.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


@pytest.fixture()
def manager(monkeypatch):
    """IndexManager with settings stubbed out."""
    workers_ns = SimpleNamespace(
        whisper_parallel=1,
        clip_parallel=1,
        metadata_batch_size=10,
    )
    settings_ns = SimpleNamespace(
        workers=workers_ns,
        indexing=SimpleNamespace(reconciliation_interval=3600),
        features=SimpleNamespace(
            auto_tags="false",
            summaries="false",
            detailed_summaries="false",
            vision_describe="false",
        ),
    )
    monkeypatch.setattr("app.indexer.settings", settings_ns)

    from app.indexer import IndexManager
    return IndexManager()


@pytest.mark.asyncio
async def test_duplicate_enqueue_blocked(manager):
    """Second enqueue of the same (file_id, task_type) is a no-op."""
    from app.indexer import IndexTask, TaskType

    task = IndexTask(file_id="abc", task_type=TaskType.WHISPER)
    await manager._enqueue(task)
    await manager._enqueue(task)  # duplicate

    assert manager._queues[TaskType.WHISPER].qsize() == 1
    assert "abc" in manager._queued_by_type[TaskType.WHISPER]


@pytest.mark.asyncio
async def test_different_task_types_both_queued(manager):
    """Same file_id with different task_type are independent entries."""
    from app.indexer import IndexTask, TaskType

    await manager._enqueue(IndexTask(file_id="abc", task_type=TaskType.WHISPER))
    await manager._enqueue(IndexTask(file_id="abc", task_type=TaskType.METADATA))

    assert manager._queues[TaskType.WHISPER].qsize() == 1
    assert manager._queues[TaskType.METADATA].qsize() == 1


@pytest.mark.asyncio
async def test_processing_blocks_enqueue(manager):
    """file_id in _processing_by_type prevents re-queuing."""
    from app.indexer import IndexTask, TaskType

    manager._processing_by_type[TaskType.WHISPER].append("xyz")
    await manager._enqueue(IndexTask(file_id="xyz", task_type=TaskType.WHISPER))

    assert manager._queues[TaskType.WHISPER].qsize() == 0


@pytest.mark.asyncio
async def test_force_bypasses_dedup(manager):
    """force=True allows re-inserting even if file_id is already queued."""
    from app.indexer import IndexTask, TaskType

    task = IndexTask(file_id="abc", task_type=TaskType.WHISPER)
    await manager._enqueue(task)
    await manager._enqueue(IndexTask(file_id="abc", task_type=TaskType.WHISPER, priority=100), force=True)

    assert manager._queues[TaskType.WHISPER].qsize() == 2


@pytest.mark.asyncio
async def test_collect_batch_removes_from_queued_set(manager):
    """Dequeuing via _collect_batch clears the file_id from _queued_by_type."""
    from app.indexer import IndexTask, TaskType

    task = IndexTask(file_id="abc", task_type=TaskType.METADATA)
    await manager._enqueue(task)
    assert "abc" in manager._queued_by_type[TaskType.METADATA]

    collected = await manager._collect_batch(TaskType.METADATA, 10)

    assert len(collected) == 1
    assert "abc" not in manager._queued_by_type[TaskType.METADATA]


@pytest.mark.asyncio
async def test_after_collect_reenqueue_allowed(manager):
    """After _collect_batch removes a file from queued, it can be re-enqueued."""
    from app.indexer import IndexTask, TaskType

    task = IndexTask(file_id="abc", task_type=TaskType.METADATA)
    await manager._enqueue(task)
    await manager._collect_batch(TaskType.METADATA, 10)

    # Should be accepted again (not blocked by stale queued set)
    await manager._enqueue(task)
    assert manager._queues[TaskType.METADATA].qsize() == 1
