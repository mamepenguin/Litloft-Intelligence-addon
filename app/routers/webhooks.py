"""Webhook endpoints for HomeVault event notifications."""

from fastapi import APIRouter, Depends

from app.dependencies import get_index_manager, verify_webhook_secret
from app.schemas import (
    MessageResponse,
    WebhookFilesDeleted,
    WebhookFilesMissing,
    WebhookFilesPurged,
    WebhookFilesRecovered,
    WebhookFilesRestored,
    WebhookScanComplete,
)
from app.webhook import (
    FilesDeletedPayload,
    FilesMissingPayload,
    FilesPurgedPayload,
    FilesRecoveredPayload,
    FilesRestoredPayload,
    ScanCompletePayload,
    handle_files_deleted,
    handle_files_missing,
    handle_files_purged,
    handle_files_recovered,
    handle_files_restored,
    handle_scan_complete,
)

router = APIRouter(tags=["webhooks"])


@router.post("/webhook/scan-complete", response_model=MessageResponse)
async def webhook_scan_complete(
    body: WebhookScanComplete,
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Handle scan-complete webhook from HomeVault."""
    manager = get_index_manager()
    payload = ScanCompletePayload(
        drive=body.drive,
        added=body.added,
        removed=body.removed,
        missing=body.missing,
        recovered=body.recovered,
    )
    result = await handle_scan_complete(payload, manager)
    return MessageResponse(**result)


@router.post("/webhook/files-deleted", response_model=MessageResponse)
async def webhook_files_deleted(
    body: WebhookFilesDeleted,
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Handle files-deleted webhook from HomeVault."""
    manager = get_index_manager()
    payload = FilesDeletedPayload(
        file_ids=tuple(body.file_ids), type=body.type
    )
    result = await handle_files_deleted(payload, manager)
    return MessageResponse(**result)


@router.post("/webhook/files-restored", response_model=MessageResponse)
async def webhook_files_restored(
    body: WebhookFilesRestored,
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Handle files-restored webhook from HomeVault."""
    manager = get_index_manager()
    payload = FilesRestoredPayload(file_ids=tuple(body.file_ids))
    result = await handle_files_restored(payload, manager)
    return MessageResponse(**result)


@router.post("/webhook/files-purged", response_model=MessageResponse)
async def webhook_files_purged(
    body: WebhookFilesPurged,
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Handle files-purged webhook from HomeVault."""
    manager = get_index_manager()
    payload = FilesPurgedPayload(file_ids=tuple(body.file_ids))
    result = await handle_files_purged(payload, manager)
    return MessageResponse(**result)


@router.post("/webhook/files-missing", response_model=MessageResponse)
async def webhook_files_missing(
    body: WebhookFilesMissing,
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Handle files-missing webhook from HomeVault."""
    manager = get_index_manager()
    payload = FilesMissingPayload(file_ids=tuple(body.file_ids))
    result = await handle_files_missing(payload, manager)
    return MessageResponse(**result)


@router.post("/webhook/files-recovered", response_model=MessageResponse)
async def webhook_files_recovered(
    body: WebhookFilesRecovered,
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Handle files-recovered webhook from HomeVault."""
    manager = get_index_manager()
    payload = FilesRecoveredPayload(file_ids=tuple(body.file_ids))
    result = await handle_files_recovered(payload, manager)
    return MessageResponse(**result)
