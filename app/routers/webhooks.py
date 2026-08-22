"""Webhook endpoints for Litloft event notifications.

These are reachable only from the Docker-internal network — core posts to
them from ``event_hooks``, and nothing outside the compose network can
route here. They carry no shared-secret check.

There used to be one, gated on ``SEARCH_WEBHOOK_SECRET``, but it was never
wired: nothing generated the value, no manifest declared ``secret_env`` for
it, and the variable was empty in every deployment, so the check was a
permanent no-op. Worse, the same dependency also guarded the ``/queue/*``
admin routes, which core's proxy calls **from the browser** without that
header — so setting the variable would have started returning 403 for the
queue controls while appearing to secure the webhooks. A setting that
breaks the application when you turn it on is worse than no setting.

If the threat model changes (a shared Docker network with untrusted peers,
or third-party addon containers), add the gate deliberately: a dependency
applied to *these* routes only, an env var emitted by ``configure.py``, and
``secret_env`` in the manifest so core sends the header. Do not reinstate
the old shape. The knowledge addon's ``KNOWLEDGE_WEBHOOK_SECRET`` is the
working reference for all three pieces.
"""

from fastapi import APIRouter, Depends

import asyncio

from app import dependencies
from app.dependencies import get_index_manager
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
) -> MessageResponse:
    """Handle files-restored webhook from Litloft."""
    manager = get_index_manager()
    payload = FilesRestoredPayload(file_ids=tuple(body.file_ids))
    result = await handle_files_restored(payload, manager)
    return MessageResponse(**result)


@router.post("/webhook/files-purged", response_model=MessageResponse)
async def webhook_files_purged(
    body: WebhookFilesPurged,
) -> MessageResponse:
    """Handle files-purged webhook from Litloft."""
    manager = get_index_manager()
    payload = FilesPurgedPayload(file_ids=tuple(body.file_ids))
    result = await handle_files_purged(payload, manager)
    return MessageResponse(**result)


@router.post("/webhook/files-missing", response_model=MessageResponse)
async def webhook_files_missing(
    body: WebhookFilesMissing,
) -> MessageResponse:
    """Handle files-missing webhook from Litloft."""
    manager = get_index_manager()
    payload = FilesMissingPayload(file_ids=tuple(body.file_ids))
    result = await handle_files_missing(payload, manager)
    return MessageResponse(**result)


@router.post("/webhook/files-recovered", response_model=MessageResponse)
async def webhook_files_recovered(
    body: WebhookFilesRecovered,
) -> MessageResponse:
    """Handle files-recovered webhook from Litloft."""
    manager = get_index_manager()
    payload = FilesRecoveredPayload(file_ids=tuple(body.file_ids))
    result = await handle_files_recovered(payload, manager)
    return MessageResponse(**result)


@router.post("/webhook/files-moved", response_model=MessageResponse)
async def webhook_files_moved(
    body: WebhookFilesMoved,
) -> MessageResponse:
    """Handle files-moved webhook from Litloft (rename / move / folder ops)."""
    manager = get_index_manager()
    payload = FilesMovedPayload(file_ids=tuple(body.file_ids))
    result = await handle_files_moved(payload, manager)
    return MessageResponse(**result)
