"""Bulk purge of an entire drive from the intelligence index.

Used when a HomeVault operator flips ``addons.intelligence`` (or its
``index`` sub-key) to ``false`` for a drive in ``drives.json`` and
restarts the addon container. The user explicitly opted to purge
existing artefacts (see plan ``kind-sparking-shore``) so the drive
can be re-enabled later from a clean slate.

Tables emptied per drive:
- ``indexed_files`` (drive column)
- ``embeddings`` and their vector-table backing rows (joined via file_id)
- ``transcript_chunks`` (file_id)
- ``suggested_tags`` (file_id)
- ``file_summaries`` (file_id)
- FTS5 mirrors (``fts_files``, ``fts_transcripts``, ``fts_text_content``)
- ``similar_cache`` (invalidated wholesale; cache is cheap to rebuild)

The implementation reuses ``_purge_file`` per file. Volumes are small
in practice (a few thousand files per disabled drive at most) and a
batch SQL would need to duplicate the vector-table cleanup logic.
"""

import logging

from app.models import IndexedFile

logger = logging.getLogger(__name__)


def purge_drive(drive: str) -> int:
    """Remove every intelligence artefact tied to ``drive``.

    Returns the number of indexed files purged. Safe to call when the
    drive has no indexed data (returns 0). Best-effort: per-file
    failures are logged and skipped so a single bad row does not block
    cleanup for the whole drive.
    """
    from app.database import get_search_db
    from app.indexer import _purge_file
    from app.search import invalidate_similar_cache

    with get_search_db() as session:
        file_ids = [
            row.file_id
            for row in session.query(IndexedFile.file_id)
            .filter(IndexedFile.drive == drive)
            .all()
        ]

    if not file_ids:
        return 0

    purged = 0
    for file_id in file_ids:
        try:
            _purge_file(file_id)
            purged += 1
        except Exception as e:
            logger.error(
                "purge_drive: failed to purge %s on drive %s: %s",
                file_id, drive, e,
            )

    invalidate_similar_cache()
    logger.info(
        "purge_drive: removed %d/%d intelligence rows for drive %s",
        purged, len(file_ids), drive,
    )
    return purged


async def purge_disabled_drives() -> dict[str, int]:
    """Purge every drive whose intelligence.index policy is currently off.

    Discovers candidate drives by scanning the local index for distinct
    drive values, then asks the host's policy endpoint per drive. Drives
    that aren't in ``drives.json`` anymore (404 from the host) are also
    purged — they cannot be the source of new data and their leftover
    rows would only show up as dead weight.

    Returns ``{drive_name: purged_count}`` for every drive that was
    actually purged (entries with 0 purged are omitted).
    """
    from app.database import get_search_db
    from app.policy_client import is_feature_enabled

    with get_search_db() as session:
        drives = [
            row.drive
            for row in session.query(IndexedFile.drive)
            .distinct()
            .all()
        ]

    results: dict[str, int] = {}
    for drive in drives:
        if not drive:
            continue
        try:
            allowed = await is_feature_enabled(drive, "index")
        except Exception:
            logger.exception(
                "purge_disabled_drives: policy lookup failed for %s "
                "(skipping to avoid accidental data loss)",
                drive,
            )
            continue
        if allowed:
            continue
        purged = purge_drive(drive)
        if purged:
            results[drive] = purged

    return results
