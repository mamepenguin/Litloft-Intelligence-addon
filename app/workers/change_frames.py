"""Move stored change frames into their file's scene candidates at start-up.

Each frame more than 3 s from every scene candidate becomes a scene candidate
with the same id and time; the rest are deleted. The move runs once per file
under the search write lock, so a file's selection pool is the same before and
after its own commit.
"""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path

from sqlalchemy import text

from app.database import get_search_db, get_search_db_read, get_search_engine
from app.models import Embedding
from app.vec_partition import _FREE_SPACE_FACTOR, insert_vector_sql, read_vectors

logger = logging.getLogger(__name__)

CHANGE_TYPE = "clip_change"
CHANGE_TABLE = "vec_clip_change"
SCENE_TYPE = "clip"
SCENE_TABLE = "vec_clip"
#: A change frame this close to a scene candidate adds nothing to the pool.
CHANGE_GAP_SECONDS = 3.0
#: The span a scene candidate keeps.
SCENE_SECONDS = 30.0


def _vector_rows(session, table: str, embedding_ids: list[str]) -> dict[str, bytes]:
    if not embedding_ids:
        return {}
    return read_vectors(session, table, embedding_ids)


def _scene_times(session, file_id: str) -> list[float]:
    rows = session.execute(
        text(
            "SELECT id, timestamp_start FROM embeddings "
            "WHERE file_id = :f AND embedding_type = :t AND vector_table = :v"
        ),
        {"f": file_id, "t": SCENE_TYPE, "v": SCENE_TABLE},
    ).fetchall()
    finite = [
        (embedding_id, float(timestamp))
        for embedding_id, timestamp in rows
        if timestamp is not None and math.isfinite(float(timestamp))
    ]
    if not finite:
        return []
    vectors = _vector_rows(session, SCENE_TABLE, [row[0] for row in finite])
    return [timestamp for embedding_id, timestamp in finite if embedding_id in vectors]


def _change_rows(session, file_id: str) -> list[tuple[str, float | None]]:
    return [
        (embedding_id, timestamp)
        for embedding_id, timestamp in session.execute(
            text(
                "SELECT id, timestamp_start FROM embeddings "
                "WHERE file_id = :f AND embedding_type = :t"
            ),
            {"f": file_id, "t": CHANGE_TYPE},
        ).fetchall()
    ]


def _movable(timestamp: float | None, scene_times: list[float]) -> bool:
    return (
        timestamp is not None
        and math.isfinite(float(timestamp))
        and all(abs(float(timestamp) - time) > CHANGE_GAP_SECONDS for time in scene_times)
    )


def _files_with_change_frames() -> list[str]:
    with get_search_db_read() as session:
        return [
            row[0]
            for row in session.execute(
                text(
                    "SELECT DISTINCT file_id FROM embeddings "
                    "WHERE embedding_type = :t ORDER BY file_id"
                ),
                {"t": CHANGE_TYPE},
            ).fetchall()
        ]


def _plan(file_ids: list[str]) -> tuple[int, int, int, int]:
    """Files, frames moved, frames deleted and the bytes the moved vectors take."""
    files = moved = deleted = total_bytes = 0
    for file_id in file_ids:
        with get_search_db_read() as session:
            scene_times = _scene_times(session, file_id)
            rows = _change_rows(session, file_id)
            if not rows:
                continue
            vectors = _vector_rows(session, CHANGE_TABLE, [row[0] for row in rows])
            files += 1
            for embedding_id, timestamp in rows:
                vector = vectors.get(embedding_id)
                if vector is not None and _movable(timestamp, scene_times):
                    moved += 1
                    total_bytes += len(vector)
                else:
                    deleted += 1
    return files, moved, deleted, total_bytes


def _data_volume() -> Path:
    database_path = get_search_engine().url.database
    if database_path:
        return Path(database_path).parent
    import app.config as config

    return config.settings.intelligence_data_dir


def _move_file(file_id: str) -> tuple[int, int]:
    with get_search_db() as session:
        scene_times = _scene_times(session, file_id)
        rows = session.query(Embedding).filter_by(
            file_id=file_id, embedding_type=CHANGE_TYPE,
        ).all()
        if not rows:
            return 0, 0
        vectors = _vector_rows(session, CHANGE_TABLE, [row.id for row in rows])
        moved = deleted = 0
        for row in rows:
            vector = vectors.get(row.id)
            if vector is not None and _movable(row.timestamp_start, scene_times):
                session.execute(
                    text(insert_vector_sql(SCENE_TABLE)),
                    {"id": row.id, "type": SCENE_TYPE, "vec": vector},
                )
                session.execute(
                    text(f"DELETE FROM {CHANGE_TABLE} WHERE embedding_id = :id"),
                    {"id": row.id},
                )
                row.embedding_type = SCENE_TYPE
                row.vector_table = SCENE_TABLE
                row.timestamp_end = float(row.timestamp_start) + SCENE_SECONDS
                moved += 1
            else:
                session.execute(
                    text(f"DELETE FROM {CHANGE_TABLE} WHERE embedding_id = :id"),
                    {"id": row.id},
                )
                session.delete(row)
                deleted += 1
    return moved, deleted


def move_change_frames() -> None:
    """Move every stored change frame into its file's scene candidates."""
    try:
        file_ids = _files_with_change_frames()
        if not file_ids:
            return
        files, planned_moved, planned_deleted, estimated = _plan(file_ids)
        if files == 0:
            return
        required = estimated * _FREE_SPACE_FACTOR
        free = shutil.disk_usage(_data_volume()).free
        if free < required:
            logger.warning(
                "change frame move skipped: not enough free disk: free=%d required=%d",
                free, required,
            )
            return
    except Exception as exc:  # noqa: BLE001 - start-up continues
        logger.error(
            "change frame move failed to start (%s)", type(exc).__name__,
        )
        return

    logger.info(
        "change frame move started: files=%d frames_moved=%d frames_deleted=%d",
        files, planned_moved, planned_deleted,
    )
    moved_files = moved_frames = deleted_frames = 0
    for file_id in file_ids:
        try:
            moved, deleted = _move_file(file_id)
        except Exception as exc:  # noqa: BLE001 - one file's failure is isolated
            logger.error(
                "change frame move failed for %s (%s)",
                file_id, type(exc).__name__,
            )
            continue
        if moved:
            moved_files += 1
        moved_frames += moved
        deleted_frames += deleted
    logger.info(
        "change frame move done: files_moved=%d frames_moved=%d frames_deleted=%d",
        moved_files, moved_frames, deleted_frames,
    )
    if moved_frames:
        from app.search import invalidate_similar_cache

        invalidate_similar_cache()
