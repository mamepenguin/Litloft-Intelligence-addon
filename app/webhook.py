"""Webhook endpoint handlers for HomeVault integration.

Processes incoming webhooks from HomeVault's backend for scan completion,
file deletion, restoration, and purging events.
"""

import logging
from dataclasses import dataclass

from app.indexer import IndexManager
from app.search import invalidate_similar_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanCompletePayload:
    """Payload for scan-complete webhook."""

    drive: str
    added: int = 0
    # ``removed`` kept for backwards compatibility with older HomeVault
    # builds. New builds emit ``missing`` / ``recovered`` instead.
    removed: int = 0
    missing: int = 0
    recovered: int = 0


@dataclass(frozen=True)
class FilesMissingPayload:
    """Payload for files-missing webhook (scanner detected vanished files)."""

    file_ids: tuple[str, ...]


@dataclass(frozen=True)
class FilesRecoveredPayload:
    """Payload for files-recovered webhook (scanner detected reappearance)."""

    file_ids: tuple[str, ...]


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
        "Received scan-complete webhook: drive=%s, added=%d, missing=%d, recovered=%d",
        payload.drive, payload.added, payload.missing, payload.recovered,
    )

    await index_manager.handle_scan_complete(payload.drive)
    invalidate_similar_cache()

    return {"status": "accepted", "message": "Reconciliation triggered"}


async def handle_files_missing(
    payload: FilesMissingPayload,
    index_manager: IndexManager,
) -> dict[str, str]:
    """Handle files-missing webhook from HomeVault.

    Marks files as inactive in the index. Embeddings and transcripts are
    preserved so the file can be fully restored if it reappears on disk.
    """
    logger.info(
        "Received files-missing webhook: %d files",
        len(payload.file_ids),
    )
    await index_manager.handle_files_missing(list(payload.file_ids))
    invalidate_similar_cache()
    return {
        "status": "accepted",
        "message": f"{len(payload.file_ids)} files marked missing",
    }


async def handle_files_recovered(
    payload: FilesRecoveredPayload,
    index_manager: IndexManager,
) -> dict[str, str]:
    """Handle files-recovered webhook from HomeVault.

    Reactivates previously missing files in the index.
    """
    logger.info(
        "Received files-recovered webhook: %d files",
        len(payload.file_ids),
    )
    await index_manager.handle_files_recovered(list(payload.file_ids))
    invalidate_similar_cache()
    return {
        "status": "accepted",
        "message": f"{len(payload.file_ids)} files recovered",
    }


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
    invalidate_similar_cache()

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
    invalidate_similar_cache()

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
    invalidate_similar_cache()

    return {
        "status": "accepted",
        "message": f"{len(payload.file_ids)} files purged",
    }
