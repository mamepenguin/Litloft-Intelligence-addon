"""Tests for app.webhook module.

Covers all webhook handlers: scan_complete, files_deleted,
files_restored, and files_purged. Verifies that each handler
calls the correct IndexManager method and invalidates cache.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stub out heavy ML/image dependencies before importing app modules
for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "numpy",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.webhook import (
    FilesDeletedPayload,
    FilesPurgedPayload,
    FilesRestoredPayload,
    ScanCompletePayload,
    handle_files_deleted,
    handle_files_purged,
    handle_files_restored,
    handle_scan_complete,
)


@pytest.fixture()
def mock_index_manager():
    """Create a mock IndexManager with async methods."""
    manager = MagicMock()
    manager.handle_scan_complete = AsyncMock()
    manager.handle_files_deleted = AsyncMock()
    manager.handle_files_restored = AsyncMock()
    manager.handle_files_purged = AsyncMock()
    return manager


# ---------------------------------------------------------------------------
# handle_scan_complete
# ---------------------------------------------------------------------------


class TestHandleScanComplete:
    """Tests for handle_scan_complete webhook handler."""

    @pytest.mark.asyncio
    async def test_calls_index_manager_with_drive(self, mock_index_manager):
        payload = ScanCompletePayload(drive="Videos", added=5, removed=2)

        with patch("app.webhook.invalidate_similar_cache") as mock_invalidate:
            result = await handle_scan_complete(payload, mock_index_manager)

        mock_index_manager.handle_scan_complete.assert_awaited_once_with("Videos")
        mock_invalidate.assert_called_once()
        assert result["status"] == "accepted"

    @pytest.mark.asyncio
    async def test_returns_accepted_status(self, mock_index_manager):
        payload = ScanCompletePayload(drive="Photos")

        with patch("app.webhook.invalidate_similar_cache"):
            result = await handle_scan_complete(payload, mock_index_manager)

        assert result == {
            "status": "accepted",
            "message": "Reconciliation triggered",
        }


# ---------------------------------------------------------------------------
# handle_files_deleted
# ---------------------------------------------------------------------------


class TestHandleFilesDeleted:
    """Tests for handle_files_deleted webhook handler."""

    @pytest.mark.asyncio
    async def test_calls_index_manager_with_file_ids_and_type(
        self, mock_index_manager
    ):
        payload = FilesDeletedPayload(
            file_ids=("id1", "id2"), type="soft_delete"
        )

        with patch("app.webhook.invalidate_similar_cache") as mock_invalidate:
            result = await handle_files_deleted(payload, mock_index_manager)

        mock_index_manager.handle_files_deleted.assert_awaited_once_with(
            ["id1", "id2"], "soft_delete"
        )
        mock_invalidate.assert_called_once()
        assert result["status"] == "accepted"

    @pytest.mark.asyncio
    async def test_message_includes_file_count(self, mock_index_manager):
        payload = FilesDeletedPayload(file_ids=("a", "b", "c"))

        with patch("app.webhook.invalidate_similar_cache"):
            result = await handle_files_deleted(payload, mock_index_manager)

        assert "3 files" in result["message"]

    @pytest.mark.asyncio
    async def test_default_type_is_soft_delete(self, mock_index_manager):
        payload = FilesDeletedPayload(file_ids=("id1",))

        with patch("app.webhook.invalidate_similar_cache"):
            await handle_files_deleted(payload, mock_index_manager)

        mock_index_manager.handle_files_deleted.assert_awaited_once_with(
            ["id1"], "soft_delete"
        )


# ---------------------------------------------------------------------------
# handle_files_restored
# ---------------------------------------------------------------------------


class TestHandleFilesRestored:
    """Tests for handle_files_restored webhook handler."""

    @pytest.mark.asyncio
    async def test_calls_index_manager_with_file_ids(self, mock_index_manager):
        payload = FilesRestoredPayload(file_ids=("id1", "id2"))

        with patch("app.webhook.invalidate_similar_cache") as mock_invalidate:
            result = await handle_files_restored(payload, mock_index_manager)

        mock_index_manager.handle_files_restored.assert_awaited_once_with(
            ["id1", "id2"]
        )
        mock_invalidate.assert_called_once()
        assert result["status"] == "accepted"

    @pytest.mark.asyncio
    async def test_message_includes_file_count(self, mock_index_manager):
        payload = FilesRestoredPayload(file_ids=("x",))

        with patch("app.webhook.invalidate_similar_cache"):
            result = await handle_files_restored(payload, mock_index_manager)

        assert "1 files restored" in result["message"]


# ---------------------------------------------------------------------------
# handle_files_purged
# ---------------------------------------------------------------------------


class TestHandleFilesPurged:
    """Tests for handle_files_purged webhook handler."""

    @pytest.mark.asyncio
    async def test_calls_index_manager_with_file_ids(self, mock_index_manager):
        payload = FilesPurgedPayload(file_ids=("id1", "id2", "id3"))

        with patch("app.webhook.invalidate_similar_cache") as mock_invalidate:
            result = await handle_files_purged(payload, mock_index_manager)

        mock_index_manager.handle_files_purged.assert_awaited_once_with(
            ["id1", "id2", "id3"]
        )
        mock_invalidate.assert_called_once()
        assert result["status"] == "accepted"

    @pytest.mark.asyncio
    async def test_message_includes_file_count(self, mock_index_manager):
        payload = FilesPurgedPayload(file_ids=("a", "b"))

        with patch("app.webhook.invalidate_similar_cache"):
            result = await handle_files_purged(payload, mock_index_manager)

        assert "2 files purged" in result["message"]

    @pytest.mark.asyncio
    async def test_returns_accepted_dict_format(self, mock_index_manager):
        payload = FilesPurgedPayload(file_ids=("id1",))

        with patch("app.webhook.invalidate_similar_cache"):
            result = await handle_files_purged(payload, mock_index_manager)

        assert "status" in result
        assert "message" in result
        assert result["status"] == "accepted"
