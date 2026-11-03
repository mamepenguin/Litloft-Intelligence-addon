"""Best-effort ``processing.updated`` notices for the core upload drawer.

SPEC-ADDON-057 I6: a write that moves ``metadata_indexed``,
``text_indexed`` or ``whisper_indexed`` from false to true, or ends a
transcription job ``failed``, tells the core so the drawer can re-ask
``POST /files/processing`` instead of waiting for its next tick.

The post uses the same core ``addon-events`` endpoint as the existing
``_emit_ws_event`` helpers in the worker modules, with the drive lifted
to the top level so the host's broadcast filter can scope it.
"""

import logging
import os
from collections.abc import Iterable

logger = logging.getLogger(__name__)

PROCESSING_EVENT = "processing.updated"

# SPEC-CORE-031 step 4 takes at most 100 ids per request, so a write of
# 105 changed files sends two events rather than one oversized one.
MAX_IDS_PER_EVENT = 100


def emit_processing_updated(drive: str, file_ids: Iterable[str]) -> None:
    """Post ``processing.updated`` for ``file_ids`` in ``drive``, best effort.

    Splits the ids into events of at most :data:`MAX_IDS_PER_EVENT`.
    A failed post is logged and swallowed: the flag write has already
    committed, and a lost notice only costs the drawer one tick.
    """
    ids = list(dict.fromkeys(file_ids))
    if not ids or not drive:
        return

    base = os.environ.get(
        "HOMEVAULT_INTERNAL_API_URL", "http://backend:8000/api/internal"
    )
    url = f"{base}/addon-events"

    for start in range(0, len(ids), MAX_IDS_PER_EVENT):
        chunk = ids[start:start + MAX_IDS_PER_EVENT]
        payload = {
            "event": PROCESSING_EVENT,
            "data": {"file_ids": chunk},
            "drive": drive,
        }
        try:
            import httpx

            with httpx.Client(timeout=3.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
        except Exception as exc:
            # INFO, not WARNING: the core is allowed to be down, and the
            # drawer re-asks on its next tick, so this is not a defect.
            logger.info(
                "%s post failed for drive %s (%d ids): %s",
                PROCESSING_EVENT,
                drive,
                len(chunk),
                exc,
            )
