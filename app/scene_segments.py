"""Time segments of a drive's videos that sit closest to a text query.

Reads the scene-frame vectors the indexer already stored; nothing is written.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy import text as sql_text

from app.database import get_search_db_read, get_search_engine
from app.models import Embedding, IndexedFile
from app.schemas import SceneSegment, SceneSegmentsResponse
from app.search import _CLIP_DRIVE_FETCH_MAX, _l2_to_cosine_similarity

MAX_GAP_SECONDS = 45.0
SEGMENT_TAIL_SECONDS = 30.0
MAX_FILES = 30
MAX_SEGMENTS_PER_FILE = 2

# Well under SQLite's default bound-variable limit of 999 on older builds.
_IN_CHUNK = 500

_KNN_SQL = (
    "SELECT embedding_id, distance FROM vec_clip "
    "WHERE vector MATCH :vec AND k = :limit AND embedding_type = 'clip' "
    "ORDER BY distance"
)


@dataclass(frozen=True)
class _Seed:
    file_id: str
    timestamp: float
    score: float


def _eligible(drive: str):
    return (
        Embedding.embedding_type == "clip",
        Embedding.timestamp_start.is_not(None),
        IndexedFile.drive == drive,
        IndexedFile.active.is_(True),
    )


def count_frames(drive: str) -> int:
    with get_search_db_read() as session:
        return session.execute(
            select(func.count())
            .select_from(Embedding)
            .join(IndexedFile, IndexedFile.file_id == Embedding.file_id)
            .where(*_eligible(drive))
        ).scalar_one()


def required_seeds(frames: int) -> int:
    return min(max(min(50, math.ceil(frames / 10)), frames // 100), _CLIP_DRIVE_FETCH_MAX)


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _seeds(query_vector, drive: str) -> tuple[list[_Seed], int, float | None]:
    """Eligible KNN rows with their scores, best first, the row count the KNN
    returned, and the score of its last row."""
    with get_search_engine().connect() as conn:
        rows = conn.execute(
            sql_text(_KNN_SQL),
            {"vec": query_vector.tobytes(), "limit": _CLIP_DRIVE_FETCH_MAX},
        ).fetchall()
    scores = {row[0]: _l2_to_cosine_similarity(row[1]) for row in rows}
    seeds: dict[str, _Seed] = {}
    with get_search_db_read() as session:
        for chunk in _chunks(list(scores), _IN_CHUNK):
            found = session.execute(
                select(Embedding.id, Embedding.file_id, Embedding.timestamp_start)
                .join(IndexedFile, IndexedFile.file_id == Embedding.file_id)
                .where(Embedding.id.in_(chunk), *_eligible(drive))
            ).all()
            for emb_id, file_id, timestamp in found:
                seeds[emb_id] = _Seed(file_id, timestamp, scores[emb_id])
    ordered = [seeds[i] for i in scores if i in seeds]
    last_score = _l2_to_cosine_similarity(rows[-1][1]) if rows else None
    return ordered, len(rows), last_score


def _frame_times(file_ids: list[str]) -> dict[str, list[float]]:
    times: dict[str, list[float]] = defaultdict(list)
    with get_search_db_read() as session:
        for chunk in _chunks(file_ids, _IN_CHUNK):
            rows = session.execute(
                select(Embedding.file_id, Embedding.timestamp_start)
                .where(
                    Embedding.file_id.in_(chunk),
                    Embedding.embedding_type == "clip",
                    Embedding.timestamp_start.is_not(None),
                )
                .order_by(Embedding.file_id, Embedding.timestamp_start)
            ).all()
            for file_id, timestamp in rows:
                times[file_id].append(timestamp)
    return times


def _file_meta(file_ids: list[str]) -> dict[str, tuple[float | None, str]]:
    with get_search_db_read() as session:
        rows = session.execute(
            select(IndexedFile.file_id, IndexedFile.duration, IndexedFile.filename)
            .where(IndexedFile.file_id.in_(file_ids))
        ).all()
    return {file_id: (duration, filename) for file_id, duration, filename in rows}


def _segment_end(last: float, frame_times: list[float], duration: float | None) -> float:
    candidates = [last + SEGMENT_TAIL_SECONDS]
    later = [t for t in frame_times if t > last]
    if later:
        candidates.append(min(later))
    if duration is not None and duration > last:
        candidates.append(duration)
    return min(candidates)


def _groups(seeds: list[_Seed]) -> list[list[_Seed]]:
    groups: list[list[_Seed]] = []
    for seed in sorted(seeds, key=lambda s: s.timestamp):
        if groups and seed.timestamp - groups[-1][-1].timestamp <= MAX_GAP_SECONDS:
            groups[-1].append(seed)
        else:
            groups.append([seed])
    return groups


def find_segments(
    query_vector, query: str, drive: str, limit: int, frames: int,
    *, floor: float | None = None,
) -> SceneSegmentsResponse:
    required = required_seeds(frames)
    seeds, fetched, last_score = _seeds(query_vector, drive)
    if floor is None:
        partial = fetched >= _CLIP_DRIVE_FETCH_MAX and len(seeds) < required
    else:
        seeds = [seed for seed in seeds if seed.score >= floor]
        partial = (
            fetched >= _CLIP_DRIVE_FETCH_MAX
            and last_score is not None
            and last_score >= floor
        )
    if not seeds:
        return SceneSegmentsResponse(query=query, segments=[], partial=partial)

    threshold = seeds[min(required, len(seeds)) - 1].score
    best: dict[str, float] = {}
    for seed in seeds:
        best[seed.file_id] = max(best.get(seed.file_id, seed.score), seed.score)
    kept = sorted(best, key=lambda f: (-best[f], f))[:MAX_FILES]
    kept_set = set(kept)

    by_file: dict[str, list[_Seed]] = defaultdict(list)
    for seed in seeds:
        if seed.score >= threshold and seed.file_id in kept_set:
            by_file[seed.file_id].append(seed)

    frame_times = _frame_times(kept)
    meta = _file_meta(kept)

    built: list[tuple[float, str, float, SceneSegment]] = []
    for file_id, file_seeds in by_file.items():
        duration, filename = meta[file_id]
        for group in _groups(file_seeds):
            peak = min(group, key=lambda s: (-s.score, s.timestamp))
            start = group[0].timestamp
            built.append((
                peak.score, file_id, start,
                SceneSegment(
                    file_id=file_id,
                    filename=filename,
                    start=start,
                    end=_segment_end(group[-1].timestamp, frame_times[file_id], duration),
                    peak=peak.timestamp,
                    score=round(peak.score, 4),
                ),
            ))
    built.sort(key=lambda b: (-b[0], b[1], b[2]))

    taken: dict[str, int] = defaultdict(int)
    segments: list[SceneSegment] = []
    for _, file_id, _, segment in built:
        if taken[file_id] >= MAX_SEGMENTS_PER_FILE:
            continue
        taken[file_id] += 1
        segments.append(segment)
        if len(segments) == limit:
            break
    return SceneSegmentsResponse(query=query, segments=segments, partial=partial)
