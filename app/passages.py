"""Passage-level links between a file and the sources the viewer vouched for.

Answers "which passages of what I am reading connect to what I already
have?" — a reading aid, not a judgement. Each row pairs one chunk of the
file being read with one chunk of a **verified** file, both reproduced
verbatim. No LLM is called and nothing is summarised: the feature points
at places, it does not write words (hako ``DPcjrRgspKAXqHjHOkJ8L``).

Two stages, so the cost stays bounded:

1. Average this file's chunk vectors into a centroid and run **one** KNN
   over ``vec_text`` to pick candidate files. Drive scoping happens in
   that query; trust and access narrowing happen right after, through
   the same Internal API call Ask uses.
2. Load the surviving candidates' chunk vectors and compute every
   source×candidate similarity as a single matrix product. Vectors are
   L2-normalised at write time, so a dot product *is* cosine similarity.

The alternative — one semantic search per paragraph — is what this
replaces. It cost one round trip per paragraph, which forced a cap on
how many paragraphs were looked at, which meant only a document's
opening ever got examined.

Spec ``2026-08-29-related-passages.md``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass

import numpy as np
from sqlalchemy import text as sql_text

from app.config import settings
from app.credentials import CallerCredential
from app.database import get_search_engine
from app.passage_terms import overlap_terms
# Reused rather than reimplemented: this function's fail-closed handling
# (network error, non-200, and above all a response that does not confirm
# the trust filter) is exactly the part that must never drift between
# callers.
from app.rag.retriever import _filter_file_ids_via_internal_api

logger = logging.getLogger(__name__)

#: The chunk kinds that carry prose. ``vec_text`` also holds metadata,
#: tfidf_keywords and vision_description vectors; none of those is a
#: passage a reader can be pointed at.
_PASSAGE_TYPES = ("text_content", "whisper")

#: Eligible rows the KNN aims to return after drive, kind and
#: self-exclusion narrow what it fetched. ``MATCH`` is a **global** KNN
#: and every joined predicate is applied post-fetch (the same behaviour
#: ``app.citations`` documents), so this is headroom, not a guarantee.
_KNN_POOL = 400

#: Ceiling sqlite-vec puts on ``k``.
_KNN_K_MAX = 4096

#: How much wider than the candidate count the KNN is asked for, before
#: trust and access drop rows. Those filters run after ranking, so
#: without headroom a run of unverified neighbours at the top empties
#: the list while verified files sit just below the cut — the reason
#: ``retriever._search_pool_size`` exists.
_CANDIDATE_OVERSAMPLE = 4

#: Rows returned when the caller does not say.
_DEFAULT_LIMIT = 5

#: How much wider than ``limit`` to rank, so the recurrence check has
#: something to fall back on. Measured over sixty channel videos it
#: rejects 53% of ranked rows; four times the ask covers that with
#: room to spare, and costs only the text resolution of rows that are
#: never reached.
_BOILERPLATE_HEADROOM = 4

#: How much of the reachable z range the bar may occupy on a small
#: field. Below 1.0 so that passing still means standing out rather than
#: merely being the maximum.
_Z_CEILING_HEADROOM = 0.9

#: A YAML key on the line after an opening ``---``. Frontmatter is
#: metadata that happens to sit in the text stream; two files' frontmatter
#: blocks resemble each other far more than their prose does.
_YAML_KEY = re.compile(r"\s*[A-Za-z_][\w .-]*:(\s|$)")


@dataclass(frozen=True)
class _Chunk:
    """One embedded passage, with everything needed to find its text again."""

    embedding_id: str
    file_id: str
    embedding_type: str
    chunk_index: int | None
    timestamp_start: float | None
    page: int | None
    #: First 200 characters of the chunk, stored at index time. Used
    #: only to recognise fragments and metadata blocks — never to
    #: display, because it is truncated.
    preview: str
    vector: np.ndarray


@dataclass(frozen=True)
class PassagePair:
    """A passage of the file being read, beside one it echoes."""

    text: str
    page: int | None
    timestamp: float | None
    other_file_id: str
    other_drive: str
    other_filename: str
    other_text: str
    other_page: int | None
    other_timestamp: float | None
    score: float
    #: Words present, word for word, in both passages, longest first.
    #: Empty whenever the two share nothing, or the tokeniser's premise
    #: does not hold (``passage_terms.has_kana``).
    overlap: list[str]


def _is_degenerate(preview: str) -> bool:
    """Whether a chunk is a fragment or a metadata block.

    Both match everything. An ellipsis, a stray caption line, or a
    two-word transcript fragment has no content to be similar *about*,
    and two files' YAML frontmatter blocks resemble each other far more
    than their prose does — one such pair scored 0.995 against a video
    it had nothing to do with.

    ``find_similar`` guards the same way when it skips whisper
    similarity for transcripts under 20 characters: "BGM-only files
    often produce a single spurious word whose embedding matches
    everything".
    """
    stripped = preview.strip()
    if len(stripped) < settings.related_passages.min_passage_chars:
        return True
    if not stripped.startswith("---"):
        return False
    for line in stripped.splitlines()[1:6]:
        if line.strip() in ("", "---"):
            continue
        return bool(_YAML_KEY.match(line))
    return False


def _sample(rows: list, cap: int) -> list:
    """At most ``cap`` rows, spread evenly across the whole file.

    Taking the first ``cap`` instead would put the opening of a long
    document in front of a reader and leave its middle unexamined —
    which is the exact failure this feature was built to remove.
    """
    if len(rows) <= cap:
        return rows
    stride = len(rows) / cap
    return [rows[int(i * stride)] for i in range(cap)]


def _passage_count(file_id: str) -> int:
    """How many passage chunks a file has, before any sampling.

    The KNN budget is computed from this: a file's own chunks are, by
    construction, the nearest things to its own centroid, so they fill
    the global top-k before the ``file_id != :self`` predicate ever runs.
    """
    engine = get_search_engine()
    types = ",".join(f"'{t}'" for t in _PASSAGE_TYPES)
    with engine.connect() as conn:
        return int(
            conn.execute(
                sql_text(
                    "SELECT COUNT(*) FROM embeddings "
                    f"WHERE file_id = :f AND embedding_type IN ({types})"
                ),
                {"f": file_id},
            ).scalar()
            or 0
        )


def _load_chunks(file_ids: list[str], cap: int) -> dict[str, list[_Chunk]]:
    """Read each file's passage chunks, vectors included.

    Metadata and vectors are fetched separately: ``vec_text`` is a
    virtual table and the proven way to read vectors out of one is an
    ``embedding_id IN (...)`` lookup, not a join.
    """
    if not file_ids:
        return {}

    engine = get_search_engine()
    placeholders = ",".join(f":f{i}" for i in range(len(file_ids)))
    params = {f"f{i}": fid for i, fid in enumerate(file_ids)}
    types = ",".join(f"'{t}'" for t in _PASSAGE_TYPES)

    with engine.connect() as conn:
        rows = conn.execute(
            sql_text(
                "SELECT id, file_id, embedding_type, chunk_index, "
                "       timestamp_start, page, content_preview "
                "FROM embeddings "
                f"WHERE file_id IN ({placeholders}) "
                f"  AND embedding_type IN ({types}) "
                "ORDER BY file_id, chunk_index, timestamp_start"
            ),
            params,
        ).fetchall()

        by_file: dict[str, list[tuple]] = {}
        for row in rows:
            # Dropped before sampling, so the cap counts usable
            # passages rather than being spent on fragments.
            if _is_degenerate(row[6] or ""):
                continue
            by_file.setdefault(row[1], []).append(row)

        kept = [
            row
            for bucket in by_file.values()
            for row in _sample(bucket, cap)
        ]
        if not kept:
            return {}

        vec_placeholders = ",".join(f":v{i}" for i in range(len(kept)))
        vec_params = {f"v{i}": row[0] for i, row in enumerate(kept)}
        vec_rows = conn.execute(
            sql_text(
                "SELECT embedding_id, vector FROM vec_text "
                f"WHERE embedding_id IN ({vec_placeholders})"
            ),
            vec_params,
        ).fetchall()

    vectors = {
        row[0]: np.frombuffer(row[1], dtype=np.float32)
        for row in vec_rows
        if row[1]
    }

    chunks: dict[str, list[_Chunk]] = {}
    for row in kept:
        vector = vectors.get(row[0])
        if vector is None:
            # The metadata row outlived its vector. Nothing can be
            # scored against it.
            continue
        chunks.setdefault(row[1], []).append(
            _Chunk(
                embedding_id=row[0],
                file_id=row[1],
                embedding_type=row[2],
                chunk_index=row[3],
                timestamp_start=row[4],
                page=row[5],
                preview=row[6] or "",
                vector=vector,
            )
        )
    return chunks


def _knn_budgets(source_rows: int) -> list[int]:
    """The ``k`` values to try, in order.

    ``MATCH`` is a global KNN: sqlite-vec picks ``k`` rows across the
    whole table and only then does the join apply drive, kind and
    ``!= :self``. The source's own chunks are the nearest things to
    their own centroid, so on a long document they occupy the entire
    budget and the self-exclusion empties it. Budget for them.

    A second, wider attempt covers the rest of the post-filter loss
    (other drives, metadata rows). Two queries is the ceiling: past
    sqlite-vec's own ``k`` cap there is nothing further to ask for.
    """
    first = min(_KNN_K_MAX, source_rows + _KNN_POOL)
    if first >= _KNN_K_MAX:
        return [_KNN_K_MAX]
    return [first, _KNN_K_MAX]


def _nearest_files(
    centroid: np.ndarray,
    *,
    file_id: str,
    drive: str,
    limit: int | None = None,
    source_rows: int = 0,
) -> list[str]:
    """Files whose passages sit closest to this file's centre of mass.

    A drive is a security boundary, so the candidate set never leaves
    the request's drive. The source file is excluded: a document is
    trivially closest to itself.
    """
    engine = get_search_engine()
    types = ",".join(f"'{t}'" for t in _PASSAGE_TYPES)
    if limit is None:
        limit = settings.related_passages.candidate_files * _CANDIDATE_OVERSAMPLE

    found: list[str] = []
    with engine.connect() as conn:
        for k in _knn_budgets(source_rows):
            rows = conn.execute(
                sql_text(
                    "SELECT e.file_id, MIN(v.distance) AS d "
                    "FROM vec_text v "
                    # No CAST on e.id. Both sides are already TEXT, and
                    # wrapping the column disables the primary-key index:
                    # SQLite falls back to scanning all of ``embeddings``
                    # once per vector row. Measured on a 54k-vector index,
                    # that is 12.6s versus 0.16s, and it grows with k.
                    "JOIN embeddings e ON e.id = v.embedding_id "
                    "JOIN indexed_files f ON f.file_id = e.file_id "
                    "WHERE v.vector MATCH :vec AND k = :k "
                    f"  AND e.embedding_type IN ({types}) "
                    "  AND e.file_id != :self "
                    "  AND f.drive = :drive "
                    "  AND f.active = 1 "
                    "GROUP BY e.file_id "
                    "ORDER BY d "
                    "LIMIT :limit"
                ),
                {
                    "vec": centroid.tobytes(),
                    "k": k,
                    "self": file_id,
                    "drive": drive,
                    "limit": limit,
                },
            ).fetchall()
            found = [row[0] for row in rows]
            # Only a completely empty result means the budget was eaten
            # by the post-fetch filters. Fewer files than the pool wants
            # is the ordinary state of a small drive, and re-running the
            # widest possible query on every request to rediscover that
            # would be pure waste.
            if found:
                break

    return found


def _file_meta(file_ids: list[str]) -> dict[str, tuple[str, str]]:
    """``file_id -> (drive, filename)`` for the files being linked to."""
    if not file_ids:
        return {}
    engine = get_search_engine()
    placeholders = ",".join(f":f{i}" for i in range(len(file_ids)))
    params = {f"f{i}": fid for i, fid in enumerate(file_ids)}
    with engine.connect() as conn:
        rows = conn.execute(
            sql_text(
                "SELECT file_id, drive, filename FROM indexed_files "
                f"WHERE file_id IN ({placeholders})"
            ),
            params,
        ).fetchall()
    return {row[0]: (row[1], row[2]) for row in rows}


def _resolve_text(chunk: _Chunk) -> str | None:
    """The chunk's full text, or None when it cannot be found.

    None is a real answer, and the caller drops the pair rather than
    substituting ``content_preview``: that column is truncated at 200
    characters, so showing it would put a prefix on screen while the
    hidden remainder is what actually produced the score.
    """
    engine = get_search_engine()
    with engine.connect() as conn:
        if chunk.embedding_type == "text_content":
            if chunk.chunk_index is None:
                # Indexed before the chunk key existed; re-indexing fills
                # it in.
                return None
            row = conn.execute(
                sql_text(
                    "SELECT text FROM fts_text_content "
                    "WHERE file_id = :f AND chunk_index = :ci LIMIT 1"
                ),
                # Every FTS5 column is text, chunk_index included.
                {"f": chunk.file_id, "ci": str(chunk.chunk_index)},
            ).fetchone()
        else:
            if chunk.timestamp_start is None:
                return None
            row = conn.execute(
                sql_text(
                    "SELECT text FROM transcript_chunks "
                    "WHERE file_id = :f AND timestamp_start = :s LIMIT 1"
                ),
                {"f": chunk.file_id, "s": chunk.timestamp_start},
            ).fetchone()

    return row[0] if row else None


def _z_ceiling(n: int) -> float:
    """The largest z a population of ``n`` values can produce.

    One value sitting above ``n - 1`` identical others gives
    ``(n - 1) / sqrt(n)``, so a request comparing a handful of passages
    cannot reach a z of 3 no matter how good its best pair is.
    """
    return (n - 1) / math.sqrt(n) if n > 1 else 0.0


def _drop_recurring(
    keep: np.ndarray,
    recurrence: np.ndarray,
    candidates: list[_Chunk],
    cap: int,
) -> None:
    """Silence source passages that pair with too many different files.

    A channel sign-off — "subscribe, links in the description" — is
    prose of ordinary length, so the fragment guard lets it through, and
    it recurs near-verbatim across every video that carries it, so it
    scores *higher* than real subject matter: 0.98 against 0.93 on a
    measured library. No threshold separates them, because the boilerplate
    is on the wrong side of it.

    Recurrence does separate them. A passage that appears in many files
    is not about any of them — the same reasoning that gives a common
    word a low IDF. Measured over a sample, source passages matched one
    file (50 of them) or two (2), and every passage that matched three
    was a sign-off.

    Only the source side is checked. Both halves of such a pair are the
    same boilerplate, so dropping the row here removes it whichever end
    it was noticed from.

    Counting happens over ``recurrence``, which still holds the
    near-duplicates that ``keep`` has already dropped. A scripted
    sign-off is the most likely thing in the library to appear *word for
    word* in several files, and those copies are the strongest evidence
    that it is boilerplate — counting only the inexact ones would let
    two verbatim copies hide, leaving the third to pass the cap alone.

    Modifies ``keep`` in place.
    """
    index: dict[str, int] = {}
    file_of = np.array(
        [index.setdefault(c.file_id, len(index)) for c in candidates]
    )
    for i in np.flatnonzero(keep.any(axis=1)):
        columns = np.flatnonzero(recurrence[i])
        if np.unique(file_of[columns]).size > cap:
            keep[i] = False


def _rank_pairs(
    source: list[_Chunk],
    candidates: list[_Chunk],
    limit: int,
) -> list[tuple[float, _Chunk, _Chunk]]:
    """Best pairs, gated by how far they stand out from this request.

    Vectors are L2-normalised at write time, so the matrix product is
    already cosine similarity — but its **absolute** value does not
    separate related passages from unrelated ones. Measured on a real
    drive: unrelated passages have a median of 0.770 and a p99 of 0.852,
    while a genuinely related pair scored 0.928 and the best unrelated
    pairs in the same request reached 0.896. The bands touch, so no
    fixed floor works. A floor at 0.80 admitted a fifth of all random
    pairs; a floor at 0.95 returned one frontmatter block and no prose.

    What separates them is position within the request's own
    distribution. The matrix is overwhelmingly made of unrelated pairs,
    so its mean and spread describe the noise, and a real match is an
    outlier against it. Across three measured files the true matches sat
    at z = 5.5-6.2 while the noise stopped at 4.5.

    This is the gap check ``search._find_similar_by_embedding`` already
    applies — does the best score stand out from the average? — with the
    spread divided out, which is necessary here because passage scores
    are packed far more tightly than the file-level scores it works on.

    At most one row per other file, always. And at most one row per
    source passage *while the document has enough passages to fill the
    list* — otherwise one paragraph with several close matches would
    crowd out the rest of a long document. A short note has nothing else
    to spread across, and capping it at one row would hide every
    connection but the strongest.
    """
    cfg = settings.related_passages

    matrix = np.stack([c.vector for c in source]) @ np.stack(
        [c.vector for c in candidates]
    ).T

    # The distribution has to describe the noise, so duplicates are taken
    # out of it before it is measured — not merely excluded from the
    # results afterwards. A single pair at 1.0 drags the mean and the
    # spread up with it, and the raised bar then suppresses the genuine
    # match it was supposed to be measured against.
    field = matrix[matrix < cfg.near_duplicate_score]
    if field.size == 0:
        return []

    mean = float(field.mean())
    std = float(field.std())
    z_floor = (
        cfg.min_z if field.size >= cfg.min_pairs_for_z else cfg.small_sample_z
    )
    # A population of n values cannot produce a z above (n-1)/sqrt(n),
    # so on a small field the configured bar can be unreachable — the
    # test would then reject every pair however good it is, which reads
    # as a confident "no connections" when the truth is "too little to
    # judge". Clamp below the ceiling so the comparison stays possible,
    # keeping headroom so passing still means standing out.
    z_floor = min(z_floor, _z_ceiling(field.size) * _Z_CEILING_HEADROOM)

    if std <= 0.0:
        # No spread to measure against. Across a wide field that means
        # every pair is equally (un)related and there is no outlier to
        # find; across a handful it means there was never a distribution
        # in the first place, and the sanity floor is all we have.
        if field.size >= cfg.min_pairs_for_z:
            return []
        threshold = cfg.min_score
    else:
        threshold = max(cfg.min_score, mean + z_floor * std)

    # Two masks: one for what may be shown, one for what counts as
    # evidence of recurrence. They differ by the near-duplicates, which
    # are never shown but are the loudest signal that a passage is
    # boilerplate.
    recurrence = matrix >= threshold
    keep = recurrence & (matrix < cfg.near_duplicate_score)
    _drop_recurring(keep, recurrence, candidates, cfg.max_passage_files)

    hits = np.argwhere(keep)
    if hits.size == 0:
        return []

    order = np.argsort(-matrix[hits[:, 0], hits[:, 1]])
    spread_across_source = len(source) >= limit
    used_source: set[int] = set()
    used_file: set[str] = set()
    pairs: list[tuple[float, _Chunk, _Chunk]] = []

    for idx in order:
        i, j = int(hits[idx][0]), int(hits[idx][1])
        other = candidates[j]
        if other.file_id in used_file:
            continue
        if spread_across_source and i in used_source:
            continue
        used_source.add(i)
        used_file.add(other.file_id)
        pairs.append((float(matrix[i][j]), source[i], other))
        if len(pairs) >= limit:
            break

    return pairs


def _recurs_across_drive(chunk: _Chunk, drive: str) -> int:
    """How many *other* files in ``drive`` say this passage too.

    ``_drop_recurring`` already knows that recurrence is what separates a
    channel sign-off from subject matter, and that the bar is two files.
    It counts over the candidates that survived into this request,
    though, which is a much smaller population than the drive: a sign-off
    carried by two hundred videos still shows up here if only one of them
    reached the ranking. Measured on a real drive, a sign-off recurring
    in fifteen files was displayed to a viewer while the filter saw one.

    Counting against the drive is what the filter meant. It costs one
    KNN per pair that survived ranking — at most five in a request — and
    stays inside the caller's drive, so nothing crosses the boundary.
    """
    engine = get_search_engine()
    cfg = settings.related_passages
    types = ",".join(f"'{t}'" for t in _PASSAGE_TYPES)
    try:
        rows = _recurrence_rows(engine, chunk, drive, cfg, types)
    except Exception as exc:
        # Fail open, as every optional narrowing in this module does. A
        # vector table that cannot answer should cost a boilerplate row,
        # not the whole section.
        logger.debug("Recurrence check unavailable for %s: %s", chunk.file_id, exc)
        return 0
    return len(rows)


def _recurrence_rows(engine, chunk: _Chunk, drive: str, cfg, types: str):
    with engine.connect() as conn:
        return conn.execute(
            sql_text(
                "SELECT DISTINCT e.file_id "
                "FROM vec_text v "
                "JOIN embeddings e ON e.id = v.embedding_id "
                "JOIN indexed_files f ON f.file_id = e.file_id "
                "WHERE v.vector MATCH :vec AND k = :k "
                f"  AND e.embedding_type IN ({types}) "
                "  AND f.drive = :drive "
                "  AND f.active = 1 "
                "  AND e.file_id != :self "
                "  AND v.distance <= :max_distance"
            ),
            {
                "vec": chunk.vector.astype(np.float32).tobytes(),
                # ``MATCH`` is a global KNN and every predicate below
                # is applied after it, so budget for what they discard —
                # the same reasoning as ``_knn_budgets``. The source's
                # own chunks are the nearest things to one of its own,
                # and other drives sit in the same table.
                "k": min(
                    _KNN_K_MAX,
                    _passage_count(chunk.file_id) + cfg.recurrence_k,
                ),
                "drive": drive,
                "self": chunk.file_id,
                # sqlite-vec reports plain L2, not its square: search.py
                # inverts it as cos = 1 - d²/2, so the bound is the root.
                "max_distance": math.sqrt(
                    max(0.0, 2.0 - 2.0 * cfg.recurrence_score)
                ),
            },
        ).fetchall()


def _build_pairs(
    source: list[_Chunk], candidate_ids: list[str], limit: int, drive: str = ""
) -> list[PassagePair]:
    """Stage 2, plus text resolution. Runs off the event loop."""
    cfg = settings.related_passages
    by_file = _load_chunks(candidate_ids, cfg.max_candidate_chunks)
    candidates = [c for fid in candidate_ids for c in by_file.get(fid, [])]
    if not candidates:
        return []

    # Rank a wider pool than the caller asked for: the recurrence check
    # below rejects rows *after* ranking, and on a channel's output it
    # rejects half of them. Truncating first would answer with two rows
    # while four good ones sat just under the cut.
    ranked = _rank_pairs(source, candidates, limit * _BOILERPLATE_HEADROOM)
    if not ranked:
        return []

    meta = _file_meta([other.file_id for _, _, other in ranked])

    cfg = settings.related_passages
    pairs: list[PassagePair] = []
    for score, mine, other in ranked:
        my_text = _resolve_text(mine)
        other_text = _resolve_text(other)
        if my_text is None or other_text is None:
            continue
        if drive and _recurs_across_drive(mine, drive) > cfg.max_passage_files:
            continue
        drive, filename = meta.get(other.file_id, ("", ""))
        pairs.append(
            PassagePair(
                text=my_text,
                page=mine.page,
                timestamp=mine.timestamp_start,
                other_file_id=other.file_id,
                other_drive=drive,
                other_filename=filename,
                other_text=other_text,
                other_page=other.page,
                other_timestamp=other.timestamp_start,
                score=score,
                overlap=overlap_terms(my_text, other_text),
            )
        )
        if len(pairs) >= limit:
            break
    return pairs


def _source_and_candidates(file_id: str, drive: str) -> tuple[list[_Chunk], list[str]]:
    """Stage 1. Runs off the event loop."""
    source = _load_chunks([file_id], settings.related_passages.max_source_chunks).get(file_id, [])
    if not source:
        return [], []

    centroid = np.mean(np.stack([c.vector for c in source]), axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0.0:
        return source, []
    centroid = (centroid / norm).astype(np.float32)

    return source, _nearest_files(
        centroid,
        file_id=file_id,
        drive=drive,
        source_rows=_passage_count(file_id),
    )


async def find_related_passages(
    file_id: str,
    drive: str,
    credential: CallerCredential | None,
    limit: int = _DEFAULT_LIMIT,
) -> list[PassagePair]:
    """Passages of ``file_id`` paired with passages of verified files.

    Returns an empty list rather than an error whenever there is nothing
    to say: an unindexed file, a file whose closest neighbours are all
    unvouched, a library with nothing similar in it.
    """
    source, candidate_ids = await asyncio.to_thread(
        _source_and_candidates, file_id, drive
    )
    if not source or not candidate_ids:
        return []

    allowed = await _filter_file_ids_via_internal_api(
        candidate_ids, credential, trust_tier="verified"
    )
    # The cap lands here, not on the KNN: a verified file just below a
    # run of unverified neighbours has to survive long enough to be
    # asked about.
    verified = [fid for fid in candidate_ids if fid in allowed][: settings.related_passages.candidate_files]
    if not verified:
        return []

    return await asyncio.to_thread(_build_pairs, source, verified, limit, drive)
