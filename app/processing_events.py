"""Best-effort ``processing.updated`` notices for the core upload drawer.

SPEC-ADDON-057 I6: a write that moves ``metadata_indexed``,
``text_indexed`` or ``whisper_indexed`` from false to true, or ends a
transcription job ``failed``, tells the core so the drawer can re-ask
``POST /files/processing`` instead of waiting for its next tick.

The event goes through :mod:`app.addon_events` like every other core
event, with the drive lifted to the top level so the host's broadcast
filter can scope it.
"""

import logging
from collections.abc import Iterable

from app.addon_events import post_addon_event_sync

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

    for start in range(0, len(ids), MAX_IDS_PER_EVENT):
        chunk = ids[start:start + MAX_IDS_PER_EVENT]
        posted = post_addon_event_sync(
            PROCESSING_EVENT, {"file_ids": chunk}, drive=drive
        )
        if not posted:
            # INFO, not WARNING: the core is allowed to be down, and the
            # drawer re-asks on its next tick, so this is not a defect.
            logger.info(
                "%s post failed for drive %s (%d ids)",
                PROCESSING_EVENT,
                drive,
                len(chunk),
            )
