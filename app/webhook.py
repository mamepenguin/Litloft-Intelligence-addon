"""Webhook endpoint handlers for HomeVault integration.

Processes incoming webhooks from HomeVault's backend for scan completion,
file deletion, restoration, and purging events.
"""

import logging
from dataclasses import dataclass

from app.indexer import IndexManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanCompletePayload:
    """Payload for scan-complete webhook."""

    drive: str
    added: int = 0
    removed: int = 0


@dataclass(frozen=True)
class FilesDeletedPayload:
    """Payload for files-deleted webhook."""

    file_ids: tuple[str, ...]
    type: str = "soft_delete"


@dataclass(frozen=True)
class FilesRestoredPayload:
    """Payload for files-restored webhook."""

    file_ids: tuple[str, ...]


@dataclass(frozen=True)
class FilesPurgedPayload:
    """Payload for files-purged webhook."""

    file_ids: tuple[str, ...]


@dataclass(frozen=True)
class PrioritizePayload:
    """Payload for queue prioritize request."""

    file_id: str


async def handle_scan_complete(
    payload: ScanCompletePayload,
    index_manager: IndexManager,
) -> dict[str, str]:
    """Handle scan-complete webhook from HomeVault.

    Triggers reconciliation to detect new and removed files.

    Args:
        payload: Scan completion details.
        index_manager: The index manager instance.

    Returns:
        Acknowledgment dict.
    """
    logger.info(
        "Received scan-complete webhook: drive=%s, added=%d, removed=%d",
        payload.drive, payload.added, payload.removed,
    )

    await index_manager.handle_scan_complete(payload.drive)

    return {"status": "accepted", "message": "Reconciliation triggered"}


async def handle_files_deleted(
    payload: FilesDeletedPayload,
    index_manager: IndexManager,
) -> dict[str, str]:
    """Handle files-deleted webhook from HomeVault.

    Marks files as inactive (soft delete) or removes them entirely.

    Args:
        payload: Deletion details.
        index_manager: The index manager instance.

    Returns:
        Acknowledgment dict.
    """
    logger.info(
        "Received files-deleted webhook: %d files, type=%s",
        len(payload.file_ids), payload.type,
    )

    await index_manager.handle_files_deleted(
        list(payload.file_ids), payload.type
    )

    return {
        "status": "accepted",
        "message": f"{len(payload.file_ids)} files processed",
    }


async def handle_files_restored(
    payload: FilesRestoredPayload,
    index_manager: IndexManager,
) -> dict[str, str]:
    """Handle files-restored webhook from HomeVault.

    Reactivates previously soft-deleted files in the index.

    Args:
        payload: Restoration details.
        index_manager: The index manager instance.

    Returns:
        Acknowledgment dict.
    """
    logger.info(
        "Received files-restored webhook: %d files",
        len(payload.file_ids),
    )

    await index_manager.handle_files_restored(list(payload.file_ids))

    return {
        "status": "accepted",
        "message": f"{len(payload.file_ids)} files restored",
    }


async def handle_files_purged(
    payload: FilesPurgedPayload,
    index_manager: IndexManager,
) -> dict[str, str]:
    """Handle files-purged webhook from HomeVault.

    Permanently removes files and all associated data from the index.

    Args:
        payload: Purge details.
        index_manager: The index manager instance.

    Returns:
        Acknowledgment dict.
    """
    logger.info(
        "Received files-purged webhook: %d files",
        len(payload.file_ids),
    )

    await index_manager.handle_files_purged(list(payload.file_ids))

    return {
        "status": "accepted",
        "message": f"{len(payload.file_ids)} files purged",
    }
