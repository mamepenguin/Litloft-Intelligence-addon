"""Bulk purge of an entire drive from the intelligence index.

Used when a Litloft operator flips ``addons.intelligence`` (or its
``index`` sub-key) to ``false`` for a drive in ``drives.json`` and
restarts the addon container. The user explicitly opted to purge
existing artefacts (see plan ``kind-sparking-shore``) so the drive
can be re-enabled later from a clean slate.

Tables emptied per drive:
- ``indexed_files`` (drive column)
- ``embeddings`` and their vector-table backing rows (joined via file_id)
- ``transcript_chunks`` and ``transcript_words`` (file_id)
- ``suggested_tags`` (file_id)
- ``suggested_chapters`` (file_id)
- ``file_summaries`` (file_id)
- ``file_insights`` (file_id, via ``_purge_file`` cascade)
- FTS5 mirrors (``fts_files``, ``fts_transcripts``, ``fts_text_content``)
- in-memory similar-files cache (invalidated wholesale; cheap to rebuild)

The implementation reuses ``_purge_file`` per file. Volumes are small
in practice (a few thousand files per disabled drive at most) and a
batch SQL would need to duplicate the vector-table cleanup logic.
"""

import logging

from sqlalchemy.exc import OperationalError

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


def purge_vision_for_drive(drive: str) -> int:
    """Clear vision_describe artefacts for every file in ``drive``.

    Used when per-drive policy flips vision_describe off but the
    umbrella index feature stays on — we only wipe the vision columns
    and ``vision_description`` embeddings so unrelated data (short /
    long / detailed summaries, whisper chunks, CLIP frames) survives.

    Returns the number of files touched.
    """
    from sqlalchemy import text as sql_text

    from app.database import get_search_db
    from app.search import invalidate_similar_cache

    touched = 0
    with get_search_db() as session:
        file_ids = [
            row.file_id
            for row in session.query(IndexedFile.file_id)
            .filter(IndexedFile.drive == drive)
            .all()
        ]

        for file_id in file_ids:
            row = session.execute(
                sql_text(
                    "SELECT visual_description, visual_description_status "
                    "FROM file_summaries WHERE file_id = :fid"
                ),
                {"fid": file_id},
            ).fetchone()
            had_columns = (
                row is not None and (row[0] is not None or row[1] is not None)
            )
            if had_columns:
                session.execute(
                    sql_text(
                        "UPDATE file_summaries SET "
                        "visual_description = NULL, "
                        "visual_description_status = NULL, "
                        "visual_description_model = NULL, "
                        "visual_description_generated_at = NULL "
                        "WHERE file_id = :fid"
                    ),
                    {"fid": file_id},
                )

            from app.models import Embedding

            embeddings = (
                session.query(Embedding)
                .filter(
                    Embedding.file_id == file_id,
                    Embedding.embedding_type == "vision_description",
                )
                .all()
            )
            had_embedding = False
            for emb in embeddings:
                table = emb.vector_table or ""
                if table.startswith("vec_"):
                    try:
                        session.execute(
                            sql_text(
                                f"DELETE FROM {table} WHERE embedding_id = :id"
                            ),
                            {"id": emb.id},
                        )
                    except OperationalError as e:
                        # Missing vec table (narrow test harness) —
                        # log and keep deleting metadata rows.
                        logger.warning(
                            "purge_vision_for_drive: vec delete failed "
                            "for %s (%s)",
                            table, type(e).__name__,
                        )
                session.delete(emb)
                had_embedding = True

            if had_columns or had_embedding:
                touched += 1

    try:
        invalidate_similar_cache()
    except Exception as e:
        # Cache invalidation is best-effort. Logging at warning keeps
        # genuine crashes visible while tolerating the rare case where
        # the cache module hasn't finished warming up yet.
        logger.warning(
            "purge_vision_for_drive: invalidate_similar_cache failed "
            "(%s)", type(e).__name__,
        )
    return touched


async def purge_disabled_vision_drives() -> dict[str, int]:
    """Scan indexed drives and wipe vision artefacts on policy-off drives.

    Mirrors :func:`purge_disabled_drives` but scoped to the
    ``vision_describe`` feature. Callers wire this in at startup so
    flipping a drive's ``addons.intelligence.vision_describe`` to false
    in ``drives.json`` + restart is the one-shot "turn it off and purge"
    workflow. Policy lookup errors cause the drive to be skipped (not
    purged) to avoid accidental data loss.
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
            allowed = await is_feature_enabled(drive, "vision_describe")
        except Exception:
            logger.exception(
                "purge_disabled_vision_drives: policy lookup failed "
                "for %s (skipping to avoid accidental data loss)",
                drive,
            )
            continue
        if allowed:
            continue
        purged = purge_vision_for_drive(drive)
        if purged:
            results[drive] = purged
    return results


def purge_video_visual_for_drive(drive: str) -> int:
    """Clear video_visual_index artefacts for every file in ``drive``.

    Used when per-drive policy flips ``video_visual_index`` off but the
    umbrella index feature stays on. Deletes ``video_visual_runs`` rows
    for the drive's files (``video_visual_scenes`` cascade via FK) and
    their ``video_visual_scene`` embeddings. Unrelated CLIP, transcript,
    summary, image-vision data, and the shared frame cache survive
    untouched — the frame cache belongs to the base scene-CLIP lifecycle
    (design doc §10).

    Returns the number of files touched (had at least one run or
    embedding removed).
    """
    from sqlalchemy import text as sql_text

    from app.database import get_search_db
    from app.models import Embedding, VideoVisualRun
    from app.search import invalidate_similar_cache

    touched = 0
    with get_search_db() as session:
        file_ids = [
            row.file_id
            for row in session.query(IndexedFile.file_id)
            .filter(IndexedFile.drive == drive)
            .all()
        ]

        for file_id in file_ids:
            had_run = (
                session.query(VideoVisualRun.id)
                .filter(VideoVisualRun.file_id == file_id)
                .first()
                is not None
            )
            if had_run:
                session.query(VideoVisualRun).filter(
                    VideoVisualRun.file_id == file_id
                ).delete(synchronize_session=False)

            embeddings = (
                session.query(Embedding)
                .filter(
                    Embedding.file_id == file_id,
                    Embedding.embedding_type == "video_visual_scene",
                )
                .all()
            )
            had_embedding = False
            for emb in embeddings:
                table = emb.vector_table or ""
                if table.startswith("vec_"):
                    try:
                        session.execute(
                            sql_text(f"DELETE FROM {table} WHERE embedding_id = :id"),
                            {"id": emb.id},
                        )
                    except OperationalError as e:
                        logger.warning(
                            "purge_video_visual_for_drive: vec delete failed "
                            "for %s (%s)",
                            table, type(e).__name__,
                        )
                session.delete(emb)
                had_embedding = True

            if had_run or had_embedding:
                touched += 1

    try:
        invalidate_similar_cache()
    except Exception as e:
        logger.warning(
            "purge_video_visual_for_drive: invalidate_similar_cache failed "
            "(%s)", type(e).__name__,
        )
    return touched


async def purge_disabled_video_visual_drives() -> dict[str, int]:
    """Scan indexed drives and wipe video-visual artefacts on policy-off drives.

    Mirrors :func:`purge_disabled_vision_drives`, scoped to the
    ``video_visual_index`` feature. Policy lookup errors cause the drive
    to be skipped (not purged) to avoid accidental data loss.
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
            allowed = await is_feature_enabled(drive, "video_visual_index")
        except Exception:
            logger.exception(
                "purge_disabled_video_visual_drives: policy lookup failed "
                "for %s (skipping to avoid accidental data loss)",
                drive,
            )
            continue
        if allowed:
            continue
        purged = purge_video_visual_for_drive(drive)
        if purged:
            results[drive] = purged
    return results


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
