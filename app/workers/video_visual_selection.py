"""Representative frames that follow a video's changes (SPEC-ADDON-112).

The file's stored frames are cut into sections at its strongest visual and
topic changes, with a coverage guarantee; one frame per section is chosen to be
typical of its section, unlike frames already chosen, and not a momentary or
near-empty picture; adjacent sections showing the same thing are merged.

:func:`select_scenes` is pure: the same stored rows, in any order, give the same
navigation and Ask lists (SPEC-ADDON-112 I2). :func:`load_selection_inputs` is
the only public function here that touches the database. Nothing decodes a
video or calls a model, so a request can run the selection.
"""

from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass

import numpy as np

from app.models import Embedding, IndexedFile, VideoFrameScan, VideoFrameText
from app.vec_partition import read_vectors

#: Sections aimed at one per this many seconds of span (SPEC-ADDON-112 step 2).
DENSITY_SECONDS = 75.0
MIN_SECTIONS = 8
MAX_SECTIONS = 40
#: Sections as cut never exceed this, however many coverage splits run.
SECTION_CEILING = 40
#: A change frame this close to a scene candidate adds nothing to the pool.
CHANGE_GAP_SECONDS = 3.0
#: A section long enough to split is this many times the span's share.
SPLIT_RATIO = 2.5
#: Head and tail of the span, whose boundaries are discounted.
HEAD_MARGIN_SECONDS = 20.0
HEAD_MARGIN_FRACTION = 0.03
TAIL_MARGIN_SECONDS = 30.0
TAIL_MARGIN_FRACTION = 0.06
EDGE_SCORE_FACTOR = 0.3
#: A topic boundary this strong may join a near visual boundary.
TOPIC_JOIN_RANK = 0.7
#: ... and only a visual boundary this strong takes it.
VISUAL_JOIN_RANK = 0.8
JOIN_WINDOW_SECONDS = 5.0
JOIN_BONUS = 0.25
#: Merging neighbours and dropping an Ask frame use these cosines.
MERGE_COSINE = 0.93
ASK_COSINE = 0.95
#: Textual representatives are compared by bigrams instead of by CLIP.
TEXTUAL_MIN_CHARS = 200
TEXTUAL_JACCARD = 0.85
#: A representative keeps clear of a picture whose next second differs this much.
STEADY_HASH_BITS = 24
#: ... and of one below this multiple of the file's median picture size.
STEADY_SIZE_FRACTION = 0.45
#: The representative is looked for this many seconds into a section.
ELIGIBLE_DELAY_SECONDS = 2.0
#: How many of a section's best frames may be tried for a steady second.
STEADY_TRIES = 4
#: Against an already chosen representative, a frame's score loses this share
#: of its similarity, which spreads the frames across the video.
ALREADY_CHOSEN_WEIGHT = 0.7

#: The only scan-row layout this build reads.
SCAN_VERSION = 1


@dataclass(frozen=True)
class Frame:
    """One stored CLIP frame: a scene candidate or a change frame."""

    embedding_id: str
    timestamp: float
    vector: np.ndarray
    embedding_type: str = "clip"


#: The Visual Index worker's name for a scene-candidate frame.
Candidate = Frame


@dataclass(frozen=True)
class TranscriptEmbedding:
    """One stored transcript embedding at its timestamp."""

    embedding_id: str
    timestamp: float
    vector: np.ndarray


@dataclass(frozen=True)
class FrameText:
    """One stored frame-OCR chunk of one segment."""

    start_time: float
    end_time: float
    text: str


@dataclass(frozen=True)
class Scan:
    """A file's per-second difference hashes and picture sizes (version 1)."""

    seconds: int
    hashes: np.ndarray
    sizes: np.ndarray


@dataclass(frozen=True)
class SelectionInputs:
    """Everything the selection reads for one file, with no database behind it."""

    scene_candidates: tuple[Frame, ...] = ()
    change_frames: tuple[Frame, ...] = ()
    scan: Scan | None = None
    transcript: tuple[TranscriptEmbedding, ...] = ()
    frame_texts: tuple[FrameText, ...] = ()
    duration: float | None = None


@dataclass(frozen=True)
class SelectedFrame:
    """One item of the navigation or Ask list."""

    section_start: float
    section_end: float
    frame_time: float
    embedding_id: str
    embedding_type: str
    kind: str
    textual: bool


@dataclass(frozen=True)
class Selection:
    """The navigation list, the Ask list, and the pool they were chosen from."""

    navigation: tuple[SelectedFrame, ...]
    ask: tuple[SelectedFrame, ...]
    candidate_count: int


@dataclass
class _Boundary:
    time: float
    kind: str
    score: float = 0.0
    topic_strength: float = 0.0


@dataclass
class _Section:
    start: float
    end: float
    kind: str
    topic_strength: float


@dataclass
class _Representative:
    frame: Frame
    frame_time: float
    textual: bool
    bigrams: frozenset[tuple[str, str]]
    #: Length of the section as cut that chose this frame; merging compares it.
    section_length: float


def compute_candidate_fingerprint(candidates: list[Frame]) -> str:
    """Hash the ordered candidate embedding IDs and timestamps.

    Order-sensitive by design: rebuilding scene CLIP changes candidate
    content/order even when the id set happens to coincide, and the
    fingerprint must catch that. Callers pass candidates in a stable order
    (the CLIP worker's storage order, i.e. ``timestamp_start``).
    """
    h = hashlib.sha256()
    for c in candidates:
        h.update(c.embedding_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(f"{c.timestamp:.6f}".encode("ascii"))
        h.update(b"\x01")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Reading the stored inputs
# ---------------------------------------------------------------------------


def _load_vectors(table: str, embedding_ids: list[str]) -> dict[str, np.ndarray]:
    """The stored vectors of ``embedding_ids`` in one partition-key vec table."""
    # Read at call time: ``app.database`` rebinds these names (the test
    # doubles do it per test), and a module-level copy would go stale.
    from app.database import get_search_engine

    wanted = sorted(set(embedding_ids))
    with get_search_engine().connect() as conn:
        stored = read_vectors(conn, table, wanted)
    return {
        embedding_id: np.frombuffer(blob, dtype=np.float32)
        for embedding_id, blob in stored.items()
    }


def _parse_scan(row: VideoFrameScan | None) -> Scan | None:
    """Parse a version-1 scan row; another version or a wrong length is no scan."""
    if row is None or row.version != SCAN_VERSION:
        return None
    seconds = int(row.seconds or 0)
    hashes = row.hashes or b""
    sizes = row.sizes or b""
    if seconds < 0 or len(hashes) != seconds * 32 or len(sizes) != seconds * 2:
        return None
    if seconds:
        bits = np.unpackbits(np.frombuffer(hashes, dtype=np.uint8)).reshape(seconds, 256)
        stored_sizes = np.frombuffer(sizes, dtype="<u2").astype(np.int64)
    else:
        bits = np.zeros((0, 256), dtype=np.uint8)
        stored_sizes = np.zeros((0,), dtype=np.int64)
    return Scan(seconds=seconds, hashes=bits.astype(bool), sizes=stored_sizes)


def _frames(
    rows: list[tuple[str, float | None]], table: str, embedding_type: str,
) -> tuple[Frame, ...]:
    vectors = _load_vectors(table, [row[0] for row in rows])
    out = [
        Frame(
            embedding_id=embedding_id,
            timestamp=float(timestamp),
            vector=vectors[embedding_id],
            embedding_type=embedding_type,
        )
        for embedding_id, timestamp in rows
        if timestamp is not None and embedding_id in vectors
    ]
    out.sort(key=lambda frame: (frame.timestamp, frame.embedding_id))
    return tuple(out)


def load_selection_inputs(file_id: str) -> SelectionInputs:
    """Read the stored rows the selection needs for one file, in a stable order.

    A missing vector drops its row, as the vector table and the metadata row can
    disagree after a partial write; a scan row of another version or with blobs
    of the wrong length is read as no scan.
    """
    # Read at call time, for the same reason as in :func:`_load_vectors`.
    from app.database import get_search_db_read

    with get_search_db_read() as session:
        duration = session.query(IndexedFile.duration).filter(
            IndexedFile.file_id == file_id
        ).scalar()
        clip_rows = session.query(Embedding.id, Embedding.timestamp_start).filter(
            Embedding.file_id == file_id,
            Embedding.embedding_type == "clip",
            Embedding.vector_table == "vec_clip",
        ).all()
        change_rows = session.query(Embedding.id, Embedding.timestamp_start).filter(
            Embedding.file_id == file_id,
            Embedding.embedding_type == "clip_change",
            Embedding.vector_table == "vec_clip_change",
        ).all()
        transcript_rows = session.query(Embedding.id, Embedding.timestamp_start).filter(
            Embedding.file_id == file_id,
            Embedding.embedding_type == "whisper",
            Embedding.vector_table == "vec_text",
        ).all()
        scan_row = session.query(VideoFrameScan).filter_by(file_id=file_id).first()
        text_rows = session.query(
            VideoFrameText.start_time, VideoFrameText.end_time, VideoFrameText.text,
        ).filter(VideoFrameText.file_id == file_id).all()
        scan = _parse_scan(scan_row)

    transcript_vectors = _load_vectors("vec_text", [row[0] for row in transcript_rows])
    transcript = tuple(sorted(
        (
            TranscriptEmbedding(
                embedding_id=embedding_id,
                timestamp=float(timestamp),
                vector=transcript_vectors[embedding_id],
            )
            for embedding_id, timestamp in transcript_rows
            if timestamp is not None and embedding_id in transcript_vectors
        ),
        key=lambda item: (item.timestamp, item.embedding_id),
    ))
    texts = tuple(sorted(
        (FrameText(float(start), float(end), body or "") for start, end, body in text_rows),
        key=lambda item: (item.start_time, item.end_time, item.text),
    ))
    return SelectionInputs(
        scene_candidates=_frames([(r[0], r[1]) for r in clip_rows], "vec_clip", "clip"),
        change_frames=_frames(
            [(r[0], r[1]) for r in change_rows], "vec_clip_change", "clip_change",
        ),
        scan=scan,
        transcript=transcript,
        frame_texts=texts,
        duration=duration,
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mean(vectors: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(vectors), axis=0)


def _ranks(entries: list[tuple[float, float, str]]) -> list[float]:
    """``(value + 1)/n`` by ascending value, ties by earlier time then id."""
    order = sorted(
        range(len(entries)),
        key=lambda i: (entries[i][0], entries[i][1], entries[i][2]),
    )
    ranks = [0.0] * len(entries)
    for position, index in enumerate(order):
        ranks[index] = (position + 1) / len(entries)
    return ranks


def _segment_chunks(texts: tuple[FrameText, ...], time: float) -> tuple[str, ...]:
    groups: dict[tuple[float, float], list[str]] = {}
    for item in texts:
        if item.start_time <= time < item.end_time:
            groups.setdefault((item.start_time, item.end_time), []).append(item.text)
    if not groups:
        return ()
    key = min(groups, key=lambda bounds: (bounds[0], bounds[1]))
    return tuple(groups[key])


def _bigrams(chunks: tuple[str, ...]) -> frozenset[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for chunk in chunks:
        body = "".join(chunk.split())
        out.update((body[i], body[i + 1]) for i in range(len(body) - 1))
    return frozenset(out)


def _is_textual(texts: tuple[FrameText, ...], time: float) -> bool:
    chunks = _segment_chunks(texts, time)
    return bool(chunks) and max(len(chunk) for chunk in chunks) >= TEXTUAL_MIN_CHARS


def _same(left: _Representative, right: _Representative, cosine: float) -> bool:
    """Whether two representatives show the same thing at ``cosine`` (step 7)."""
    if left.textual and right.textual:
        union = left.bigrams | right.bigrams
        if not union:
            return False
        return len(left.bigrams & right.bigrams) / len(union) >= TEXTUAL_JACCARD
    if left.textual != right.textual:
        return False
    return _cosine(left.frame.vector, right.frame.vector) >= cosine


def _median(sizes: np.ndarray) -> float:
    if sizes.size == 0:
        return 0.0
    return float(np.median(sizes))


def _steady_second(time: float, section_end: float, scan: Scan | None) -> float | None:
    """The first steady second of a frame at ``time``, or ``None`` (step 6)."""
    if scan is None or scan.seconds == 0:
        return None
    floor = STEADY_SIZE_FRACTION * _median(scan.sizes)
    first = math.ceil(time)
    for second in (first, first + 1, first + 2):
        if second >= section_end:
            continue
        if second < 0 or second + 1 >= scan.seconds:
            continue
        if scan.sizes[second] < floor:
            continue
        changed = int(np.count_nonzero(
            scan.hashes[second] != scan.hashes[second + 1]
        ))
        if changed >= STEADY_HASH_BITS:
            continue
        return float(second)
    return None


def _frames_both_sides(
    times: list[float], start: float, time: float, end: float,
) -> bool:
    """Whether a cut at ``time`` leaves a pool frame on each side of it.

    ``times`` is the pool's timestamps in pool order, so a cut is a bisection
    rather than a scan of the whole pool.
    """
    index = bisect.bisect_left(times, time)
    return (
        index > 0 and times[index - 1] >= start
        and index < len(times) and times[index] < end
    )


# ---------------------------------------------------------------------------
# The selection
# ---------------------------------------------------------------------------


def _pool(inputs: SelectionInputs, lo: float, hi: float) -> list[Frame]:
    scene = [
        frame for frame in inputs.scene_candidates
        if frame.vector is not None and lo <= frame.timestamp < hi
    ]
    scene_times = [
        frame.timestamp for frame in inputs.scene_candidates
        if frame.vector is not None and math.isfinite(frame.timestamp)
    ]
    changes = [
        frame for frame in inputs.change_frames
        if frame.vector is not None
        and lo <= frame.timestamp < hi
        and all(abs(frame.timestamp - time) > CHANGE_GAP_SECONDS for time in scene_times)
    ]
    pool = scene + changes
    pool.sort(key=lambda frame: (
        frame.timestamp,
        0 if frame.embedding_type == "clip" else 1,
        frame.embedding_id,
    ))
    return pool


def selection_pool(inputs: SelectionInputs) -> tuple[Frame, ...]:
    """The whole file's pool, in pool order (SPEC-ADDON-112 step 1).

    The span ends at the file's duration, or one second past its last frame
    when the duration is unknown, as :func:`select_scenes` ends it.
    """
    hi = inputs.duration
    if hi is None:
        times = [
            frame.timestamp
            for frame in (*inputs.scene_candidates, *inputs.change_frames)
            if frame.vector is not None and math.isfinite(frame.timestamp)
        ]
        hi = max(times, default=0.0) + 1.0
    return tuple(_pool(inputs, 0.0, hi))


def _visual_boundaries(pool: list[Frame]) -> list[_Boundary]:
    entries: list[tuple[float, float, str]] = []
    for index in range(len(pool) - 1):
        earlier = _mean([pool[j].vector for j in range(max(0, index - 1), index + 1)])
        later = _mean([
            pool[j].vector for j in range(index + 1, min(len(pool), index + 3))
        ])
        later_frame = pool[index + 1]
        entries.append((
            1.0 - _cosine(earlier, later),
            later_frame.timestamp,
            later_frame.embedding_id,
        ))
    ranks = _ranks(entries)
    return [
        _Boundary(time=entry[1], kind="visual", score=rank)
        for entry, rank in zip(entries, ranks)
    ]


def _topic_boundaries(
    inputs: SelectionInputs, lo: float, hi: float,
) -> list[_Boundary]:
    embeddings = sorted(
        (
            item for item in inputs.transcript
            if item.vector is not None and lo <= item.timestamp < hi
        ),
        key=lambda item: (item.timestamp, item.embedding_id),
    )
    if len(embeddings) <= 6:
        return []
    entries: list[tuple[float, float, str]] = []
    for index in range(len(embeddings) - 1):
        earlier = _mean([
            embeddings[j].vector for j in range(max(0, index - 2), index + 1)
        ])
        later = _mean([
            embeddings[j].vector
            for j in range(index + 1, min(len(embeddings), index + 4))
        ])
        later_item = embeddings[index + 1]
        entries.append((
            1.0 - _cosine(earlier, later),
            later_item.timestamp,
            later_item.embedding_id,
        ))
    ranks = _ranks(entries)
    return [
        _Boundary(time=entry[1], kind="topic", score=rank, topic_strength=rank)
        for entry, rank in zip(entries, ranks)
    ]


def _boundaries(
    inputs: SelectionInputs, pool: list[Frame], lo: float, hi: float,
) -> list[_Boundary]:
    """Every boundary with its kind, score, and topic strength (step 3)."""
    visual = _visual_boundaries(pool)
    topic = _topic_boundaries(inputs, lo, hi)

    joiners: dict[int, list[_Boundary]] = {}
    kept_topic: list[_Boundary] = []
    for boundary in topic:
        if boundary.score < TOPIC_JOIN_RANK:
            kept_topic.append(boundary)
            continue
        options = [
            (index, candidate) for index, candidate in enumerate(visual)
            if candidate.score >= VISUAL_JOIN_RANK
            and abs(candidate.time - boundary.time) <= JOIN_WINDOW_SECONDS
        ]
        if not options:
            kept_topic.append(boundary)
            continue
        best_index, _best = min(
            options, key=lambda item: (-item[1].score, item[1].time),
        )
        joiners.setdefault(best_index, []).append(boundary)

    boundaries: list[_Boundary] = []
    for index, boundary in enumerate(visual):
        joined = joiners.get(index)
        if joined:
            top = min(joined, key=lambda item: (-item.score, item.time))
            boundary.kind = "visual_topic"
            boundary.topic_strength = top.score
            boundary.score = max(boundary.score, top.score) + JOIN_BONUS
        boundaries.append(boundary)
    boundaries.extend(kept_topic)

    head = max(HEAD_MARGIN_SECONDS, HEAD_MARGIN_FRACTION * (hi - lo))
    tail = max(TAIL_MARGIN_SECONDS, TAIL_MARGIN_FRACTION * (hi - lo))
    for boundary in boundaries:
        if boundary.time - lo < head or hi - boundary.time < tail:
            boundary.score *= EDGE_SCORE_FACTOR
    return boundaries


def _sections(
    pool: list[Frame], boundaries: list[_Boundary], lo: float, hi: float,
    count: int, margin: float,
) -> list[_Section]:
    """Cut the span into sections (step 4)."""
    times = [frame.timestamp for frame in pool]
    taken: list[_Boundary] = []
    for boundary in sorted(boundaries, key=lambda item: (-item.score, item.time)):
        if len(taken) >= count - 1:
            break
        if boundary.time - lo < margin or hi - boundary.time < margin:
            continue
        if any(abs(boundary.time - other.time) < margin for other in taken):
            continue
        if not _frames_both_sides(times, lo, boundary.time, hi):
            continue
        taken.append(boundary)
    taken.sort(key=lambda item: item.time)

    starts = [(lo, "start", 0.0)]
    starts.extend(
        (boundary.time, boundary.kind, boundary.topic_strength)
        for boundary in taken
    )
    sections = [
        _Section(start=start, end=end, kind=kind, topic_strength=strength)
        for (start, kind, strength), end in zip(
            starts, [time for time, _, _ in starts[1:]] + [hi],
        )
    ]

    limit = min(math.ceil(1.5 * count), SECTION_CEILING)
    threshold = SPLIT_RATIO * (hi - lo) / count
    taken_times = {boundary.time for boundary in taken}
    while len(sections) < limit:
        split: tuple[int, float, str, float] | None = None
        for index in sorted(
            range(len(sections)),
            key=lambda i: (-(sections[i].end - sections[i].start), sections[i].start),
        ):
            section = sections[index]
            if section.end - section.start <= threshold:
                break
            inside = [
                boundary for boundary in boundaries
                if boundary.time not in taken_times
                and section.start + margin <= boundary.time <= section.end - margin
                and _frames_both_sides(
                    times, section.start, boundary.time, section.end,
                )
            ]
            if inside:
                best = min(inside, key=lambda item: (-item.score, item.time))
                split = (index, best.time, best.kind, best.topic_strength)
                break
            frames = [
                frame for frame in pool
                if section.start + margin <= frame.timestamp <= section.end - margin
                and _frames_both_sides(
                    times, section.start, frame.timestamp, section.end,
                )
            ]
            if frames:
                middle = (section.start + section.end) / 2
                best_frame = min(
                    frames,
                    key=lambda frame: (abs(frame.timestamp - middle), frame.timestamp),
                )
                split = (index, best_frame.timestamp, "coverage", 0.0)
                break
            # Neither a boundary nor a frame: the next longest section is tried.
        if split is None:
            break
        index, time, kind, strength = split
        section = sections[index]
        sections[index] = _Section(
            start=section.start, end=time, kind=section.kind,
            topic_strength=section.topic_strength,
        )
        sections.insert(
            index + 1, _Section(
                start=time, end=section.end, kind=kind, topic_strength=strength,
            ),
        )
        if kind in ("visual", "topic", "visual_topic"):
            taken_times.add(time)
    return sections


def _representative(
    section: _Section, eligible: list[Frame], chosen: list[_Representative],
    scan: Scan | None, texts: tuple[FrameText, ...],
) -> _Representative:
    mean = _mean([frame.vector for frame in eligible])
    scored = []
    for index, frame in enumerate(eligible):
        score = _cosine(frame.vector, mean)
        if chosen:
            score -= ALREADY_CHOSEN_WEIGHT * max(
                _cosine(frame.vector, rep.frame.vector) for rep in chosen
            )
        scored.append((score, index, frame))
    scored.sort(key=lambda item: (-item[0], item[1]))

    def build(frame: Frame, frame_time: float) -> _Representative:
        return _Representative(
            frame=frame,
            frame_time=frame_time,
            textual=_is_textual(texts, frame_time),
            bigrams=_bigrams(_segment_chunks(texts, frame_time)),
            section_length=section.end - section.start,
        )

    best = [frame for _score, _index, frame in scored[:STEADY_TRIES]]
    if scan is not None:
        for frame in best:
            second = _steady_second(frame.timestamp, section.end, scan)
            if second is not None:
                return build(frame, second)
    return build(best[0], best[0].timestamp)


def _representatives(
    sections: list[_Section], pool: list[Frame], scan: Scan | None,
    texts: tuple[FrameText, ...],
) -> list[_Representative]:
    order = sorted(
        range(len(sections)),
        key=lambda i: (-(sections[i].end - sections[i].start), sections[i].start),
    )
    chosen: list[_Representative] = []
    reps: list[_Representative | None] = [None] * len(sections)
    for index in order:
        section = sections[index]
        eligible = [
            frame for frame in pool
            if section.start + ELIGIBLE_DELAY_SECONDS <= frame.timestamp < section.end
        ]
        if not eligible:
            eligible = [
                frame for frame in pool
                if section.start <= frame.timestamp < section.end
            ]
        rep = _representative(section, eligible, chosen, scan, texts)
        reps[index] = rep
        chosen.append(rep)
    return [rep for rep in reps if rep is not None]


def _merge(
    sections: list[_Section], reps: list[_Representative],
) -> list[tuple[_Section, _Representative]]:
    """Merge adjacent sections showing the same thing (step 8)."""
    merged: list[tuple[_Section, _Representative]] = []
    for section, rep in zip(sections, reps):
        if merged and section.topic_strength < TOPIC_JOIN_RANK:
            current, held = merged[-1]
            if _same(held, rep, MERGE_COSINE):
                merged[-1] = (
                    _Section(
                        start=current.start, end=section.end, kind=current.kind,
                        topic_strength=current.topic_strength,
                    ),
                    rep if rep.section_length > held.section_length else held,
                )
                continue
        merged.append((section, rep))
    return merged


def _ask(
    merged: list[tuple[_Section, _Representative]],
) -> list[tuple[_Section, _Representative]]:
    """Drop a section showing what a longer kept one shows (step 9)."""
    kept: list[tuple[_Section, _Representative]] = []
    order = sorted(
        range(len(merged)),
        key=lambda i: (-(merged[i][0].end - merged[i][0].start), merged[i][0].start),
    )
    for index in order:
        section, rep = merged[index]
        if any(_same(rep, other, ASK_COSINE) for _kept_section, other in kept):
            continue
        kept.append((section, rep))
    kept.sort(key=lambda item: item[0].start)
    return kept


def _item(section: _Section, rep: _Representative) -> SelectedFrame:
    return SelectedFrame(
        section_start=section.start,
        section_end=section.end,
        frame_time=rep.frame_time,
        embedding_id=rep.frame.embedding_id,
        embedding_type=rep.frame.embedding_type,
        kind=section.kind,
        textual=rep.textual,
    )


def select_scenes(
    inputs: SelectionInputs, lo: float = 0.0, hi: float | None = None,
) -> Selection:
    """The navigation list and Ask list over the span ``[lo, hi)``, in time order.

    ``hi`` of ``None`` is the last scene-candidate or change-frame time at or
    after ``lo`` plus one second; with no frame at or after ``lo``, and when a
    known ``hi`` does not exceed ``lo``, both lists are empty. Pure: no
    database, file or model is read.
    """
    if hi is None:
        times = [
            frame.timestamp
            for frame in (*inputs.scene_candidates, *inputs.change_frames)
            if frame.vector is not None
            and math.isfinite(frame.timestamp)
            and frame.timestamp >= lo
        ]
        if not times:
            return Selection(navigation=(), ask=(), candidate_count=0)
        hi = max(times) + 1.0
    if not math.isfinite(hi) or hi <= lo:
        return Selection(navigation=(), ask=(), candidate_count=0)

    pool = _pool(inputs, lo, hi)
    if not pool:
        return Selection(navigation=(), ask=(), candidate_count=0)

    span = hi - lo
    count = min(
        max(MIN_SECTIONS, min(MAX_SECTIONS, math.ceil(span / DENSITY_SECONDS))),
        len(pool),
    )
    margin = span / (3 * count)

    boundaries = _boundaries(inputs, pool, lo, hi)
    sections = _sections(pool, boundaries, lo, hi, count, margin)
    reps = _representatives(sections, pool, inputs.scan, inputs.frame_texts)
    merged = _merge(sections, reps)
    ask = _ask(merged)
    return Selection(
        navigation=tuple(_item(section, rep) for section, rep in merged),
        ask=tuple(_item(section, rep) for section, rep in ask),
        candidate_count=len(pool),
    )


__all__ = [
    "Candidate",
    "Frame",
    "FrameText",
    "Scan",
    "SelectedFrame",
    "Selection",
    "SelectionInputs",
    "TranscriptEmbedding",
    "compute_candidate_fingerprint",
    "load_selection_inputs",
    "select_scenes",
    "selection_pool",
]
