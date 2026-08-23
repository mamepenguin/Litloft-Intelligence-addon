"""Webhook endpoints for Litloft event notifications.

These are gated by ``verify_webhook_secret``, which compares
``X-Webhook-Secret`` against ``SEARCH_WEBHOOK_SECRET``. The gate is opt-in:
unset, it is a no-op, matching how core treats ``CORE_INTERNAL_SECRET`` on
its own internal read endpoints.

To turn it on, both ends have to agree. Core sends the header only when the
listener in ``event-hooks.json`` declares ``"secret_env":
"SEARCH_WEBHOOK_SECRET"`` (see ``_build_headers`` in core's
``event_hooks.py``, and the example in that module's docstring). Neither
``configure.py`` nor this addon's manifest emits that today, so a default
install runs ungated — but a hand-written ``event-hooks.json`` plus the
environment variable does work, and is a supported configuration.

The gate deliberately does **not** cover ``/queue/*``. Those routes are
called from the browser through core's addon proxy, which never attaches
``X-Webhook-Secret``, so guarding them here meant that setting the variable
returned 403 for the queue controls — a failure that looks like an addon
bug and is hard to trace back to a config value. Their authorization comes
from the proxy instead: the manifest marks all three with
``pre_check: {"type": "admin"}``.
"""

from fastapi import APIRouter, Depends

import asyncio

from app import dependencies
from app.dependencies import get_index_manager, verify_webhook_secret
from app.schemas import (
    MessageResponse,
    WebhookFilesDeleted,
    WebhookFilesMissing,
    WebhookFilesMoved,
    WebhookFilesPurged,
    WebhookFilesRecovered,
    WebhookFilesRestored,
    WebhookScanComplete,
)
from app.webhook import (
    FilesDeletedPayload,
    FilesMissingPayload,
    FilesMovedPayload,
    FilesPurgedPayload,
    FilesRecoveredPayload,
    FilesRestoredPayload,
    ScanCompletePayload,
    handle_files_deleted,
    handle_files_missing,
    handle_files_moved,
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
    """Handle scan-complete webhook from Litloft."""
    manager = get_index_manager()
    payload = ScanCompletePayload(
        drive=body.drive,
        added=body.added,
        removed=body.removed,
        missing=body.missing,
        recovered=body.recovered,
    )
    result = await handle_scan_complete(payload, manager)

    if dependencies._pickup_worker is not None:
        await dependencies._pickup_worker.schedule_drive(body.drive)

    return MessageResponse(**result)


@router.post("/webhook/files-deleted", response_model=MessageResponse)
async def webhook_files_deleted(
    body: WebhookFilesDeleted,
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Handle files-deleted webhook from Litloft."""
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
    """Handle files-restored webhook from Litloft."""
    manager = get_index_manager()
    payload = FilesRestoredPayload(file_ids=tuple(body.file_ids))
    result = await handle_files_restored(payload, manager)
    return MessageResponse(**result)


@router.post("/webhook/files-purged", response_model=MessageResponse)
async def webhook_files_purged(
    body: WebhookFilesPurged,
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Handle files-purged webhook from Litloft."""
    manager = get_index_manager()
    payload = FilesPurgedPayload(file_ids=tuple(body.file_ids))
    result = await handle_files_purged(payload, manager)
    return MessageResponse(**result)


@router.post("/webhook/files-missing", response_model=MessageResponse)
async def webhook_files_missing(
    body: WebhookFilesMissing,
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Handle files-missing webhook from Litloft."""
    manager = get_index_manager()
    payload = FilesMissingPayload(file_ids=tuple(body.file_ids))
    result = await handle_files_missing(payload, manager)
    return MessageResponse(**result)


@router.post("/webhook/files-recovered", response_model=MessageResponse)
async def webhook_files_recovered(
    body: WebhookFilesRecovered,
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Handle files-recovered webhook from Litloft."""
    manager = get_index_manager()
    payload = FilesRecoveredPayload(file_ids=tuple(body.file_ids))
    result = await handle_files_recovered(payload, manager)
    return MessageResponse(**result)


@router.post("/webhook/files-moved", response_model=MessageResponse)
async def webhook_files_moved(
    body: WebhookFilesMoved,
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Handle files-moved webhook from Litloft (rename / move / folder ops)."""
    manager = get_index_manager()
    payload = FilesMovedPayload(file_ids=tuple(body.file_ids))
    result = await handle_files_moved(payload, manager)
    return MessageResponse(**result)
