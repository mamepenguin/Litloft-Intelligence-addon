"""Citation linker for detailed_summary segments.

Given a ``detailed_summary`` Markdown document and the file it was
generated from, compute the best-matching source chunks for each
segment and persist them as ``detailed_summary_citations`` rows.

Design:

* Parsing happens in :mod:`app.summary_parser` — this module only
  consumes ``Segment`` objects.
* Embeddings are produced by the shared ``text_embedding`` model
  (:mod:`app.workers.embedder`), the same one used to index
  transcripts and documents. Using the same model keeps the cosine
  space consistent.
* Candidate chunks are pulled via a hybrid two-stage pipeline
  (``_retrieve_candidates``): dense KNN oversamples a pool of
  ``citation_top_k_internal`` candidates, then BM25 over the FTS5
  mirrors reranks within that pool using RRF (reciprocal rank
  fusion). Dense cosine is preserved as ``top_score`` so the existing
  ``citation_threshold`` keeps its meaning. When
  ``citation_hybrid_enabled`` is False the pipeline falls back to
  dense-only. CLIP (image) vectors are always excluded because
  detailed_summary is text-only.
* A segment with top-1 score below ``summaries.citation_threshold``
  still gets a row with ``has_citation = False`` so the UI can render
  a "no strong source" warning.

The writer replaces-in-place: it wipes all existing citations for
``file_id`` and writes the freshly computed set in one transaction.
This keeps the ``UNIQUE (file_id, section_path)`` invariant simple —
regeneration / edit / revert always start from a clean slate.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np
from sqlalchemy import text as sql_text

from app.config import settings
from app.database import get_search_db
from app.summary_parser import Segment, parse_segments

logger = logging.getLogger(__name__)

# Embedding id formats used by the indexer. We parse these to recover
# a human-readable chunk identifier for ``citation_chunk_ids``.
#
# Transcript (Whisper) rows:      wh_{file_id}_{chunk_index}_{hex_hash}
# Text-content (document) rows:   txt_{file_id}_{chunk_index}_{hex_hash}
#
# ``file_id`` can contain underscores (e.g. nanoid defaults include
# ``_``, ``-`` and alphanumerics — a real-world id like
# ``KtVKUiry6S_d`` has a ``_`` in the middle). The original
# ``[^_]+`` pattern greedily stopped at the first underscore and
# silently failed to parse any embedding id for such files, which
# made citation retrieval return zero candidates for them. We now
# anchor on the *trailing* shape — digit chunk_index followed by a
# lowercase-hex hash to end — and let ``.+`` back off through any
# embedded underscores inside the file_id.
_WHISPER_RE = re.compile(r"^wh_.+_(\d+)_[0-9a-f]+$")
_TEXT_CONTENT_RE = re.compile(r"^txt_.+_(\d+)_[0-9a-f]+$")


def _make_chunk_id(embedding_id: str) -> str | None:
    """Derive a UI-friendly chunk identifier from a ``vec_text`` row.

    Transcripts become ``transcript:{chunk_index}`` and document chunks
    become ``document:{chunk_index}``. The prefix lets the frontend
    choose an appropriate jump target (seek-to-timestamp vs
    scroll-to-chunk). Returns ``None`` for embedding ids that don't
    match either known format.
    """
    match = _WHISPER_RE.match(embedding_id)
    if match:
        return f"transcript:{match.group(1)}"
    match = _TEXT_CONTENT_RE.match(embedding_id)
    if match:
        return f"document:{match.group(1)}"
    return None


def _fetch_file_vectors(
    file_id: str,
    chunk_range: tuple[int, int] | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Fetch every text embedding for a file as ``(chunk_id, vector)`` pairs.

    Joins ``vec_text`` against ``embeddings`` to pull the raw vector
    bytes directly, bypassing the ``MATCH`` KNN operator. The result
    set contains **all** of the file's whisper / text_content chunks —
    the caller then scores them in numpy for an exhaustive per-file
    top-K.

    Why not KNN? sqlite-vec's ``MATCH`` is a **global** KNN; passing
    ``file_id`` through the join filters post-fetch, so if the target
    file's chunks rank below the global top-K they're silently
    dropped. For a DB with tens of thousands of vectors and many
    files, a generic summary segment can easily have all its best
    in-file matches buried behind other files' chunks. Exhaustively
    scoring the target file's ~50-500 chunks in memory is cheaper and,
    crucially, **independent of DB size** — the dev environment will
    behave the same as a production DB with 10× or 100× the vectors.

    ``chunk_range`` narrows the fetch to chunk indices inside an
    inclusive ``(lo, hi)`` window. This is the Phase 2-D' section
    anchor filter — applied in SQL so we don't pay for decoding
    vectors we'll immediately discard.

    Returns an empty list when the file has no text embeddings or
    when the DB query fails (the caller treats that as "no dense
    candidates", same as the previous KNN path's error behaviour).
    """
    # chunk_index is embedded in ``embedding_id`` (e.g. ``wh_<fid>_7_<hex>``)
    # rather than exposed as a separate column on ``embeddings``. We
    # apply the range filter on whichever side is cheapest: transcripts
    # have a parallel ``transcript_chunks.chunk_index`` int column we
    # could join, but a simple Python-side filter after
    # ``_make_chunk_id`` stays close to legacy behaviour and avoids
    # schema-dependent JOIN logic. The cost is trivial because the SQL
    # already narrows by ``file_id``.
    try:
        with get_search_db() as session:
            rows = session.execute(
                sql_text(
                    "SELECT v.embedding_id, v.vector "
                    "FROM vec_text v "
                    "JOIN embeddings e "
                    "ON CAST(e.id AS TEXT) = v.embedding_id "
                    "WHERE e.file_id = :fid "
                    "AND e.embedding_type IN ('whisper', 'text_content')"
                ),
                {"fid": file_id},
            ).fetchall()
    except Exception as e:  # noqa: BLE001 — fail soft, don't break worker
        logger.warning(
            "Citation vector fetch failed for %s: %s", file_id, e
        )
        return []

    pairs: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for embedding_id, vec_bytes in rows:
        chunk_id = _make_chunk_id(embedding_id or "")
        if chunk_id is None or chunk_id in seen:
            continue
        if chunk_range is not None:
            # ``chunk_id`` is ``transcript:<idx>`` or ``document:<idx>``.
            _, _, raw_idx = chunk_id.partition(":")
            try:
                idx = int(raw_idx)
            except ValueError:
                continue
            lo, hi = chunk_range
            if not (lo <= idx <= hi):
                continue
        if vec_bytes is None:
            continue
        try:
            arr = np.frombuffer(vec_bytes, dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if arr.size == 0:
            continue
        seen = {*seen, chunk_id}
        # ``frombuffer`` returns a read-only view; we copy so later
        # in-place ops (e.g. normalisation in tests) don't fail.
        pairs = [*pairs, (chunk_id, arr.copy())]
    return pairs


def _query_top_chunks_dense(
    file_id: str,
    query_vector: np.ndarray,
    top_k: int,
    chunk_range: tuple[int, int] | None = None,
    file_vectors: list[tuple[str, np.ndarray]] | None = None,
) -> list[tuple[str, float]]:
    """Return ``[(chunk_id, cosine_score), ...]`` for the top-K file chunks.

    Exhaustive per-file cosine: fetches every text embedding of
    ``file_id`` (optionally range-filtered) and scores each against
    ``query_vector`` in numpy. This replaces the previous sqlite-vec
    ``MATCH`` path, which was a *global* KNN + post-filter and thus
    missed in-file candidates that ranked below the global top-K once
    the DB grew. The new path's cost scales with the file's chunk
    count, not the DB's vector count — so citation quality stays
    stable as the database grows.

    Both query and corpus vectors are normalised before the dot
    product so the output is a true cosine similarity in ``[0, 1]``
    (clamped against float noise). Scores are returned in descending
    order, top-K only.

    ``file_vectors`` is an optional already-fetched list of the file's
    ``(chunk_id, vector)`` pairs. When supplied we skip the DB round-
    trip and filter the pool in memory instead — used by the caller
    that pre-fetches once per file for the hierarchical range map.
    """
    if top_k <= 0:
        return []

    try:
        query = np.asarray(query_vector, dtype=np.float32)
    except (TypeError, ValueError) as e:
        logger.warning("Invalid query vector for %s: %s", file_id, e)
        return []
    if query.ndim != 1 or query.size == 0:
        logger.warning(
            "Query vector has unexpected shape for %s: %r",
            file_id, getattr(query, "shape", None),
        )
        return []
    qnorm = float(np.linalg.norm(query))
    if qnorm <= 0:
        return []
    query_unit = query / qnorm

    if file_vectors is None:
        pairs = _fetch_file_vectors(file_id, chunk_range=chunk_range)
    else:
        pairs = _filter_vectors_by_range(file_vectors, chunk_range)
    if not pairs:
        return []

    # Drop any stored vector whose dimensionality doesn't match the
    # query — shouldn't happen when the indexer and the retriever
    # share ``text_embedding`` config, but a mismatch would otherwise
    # crash the stack instead of failing soft.
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    for cid, v in pairs:
        if v.shape[0] != query_unit.shape[0]:
            continue
        ids = [*ids, cid]
        vecs = [*vecs, v]
    if not ids:
        return []

    matrix = np.stack(vecs)  # shape (N, D)
    norms = np.linalg.norm(matrix, axis=1)
    # Replace zero norms with 1 so the division doesn't NaN;
    # those rows will score ~0 after the dot product.
    safe_norms = np.where(norms > 0, norms, 1.0)
    unit_matrix = matrix / safe_norms[:, None]
    sims = unit_matrix @ query_unit  # shape (N,)
    # Clamp float noise near identical vectors so the threshold test
    # can't be tripped by 1.0000001.
    sims = np.clip(sims, 0.0, 1.0)

    # Top-K via argpartition for O(N) selection, then final small sort.
    k = min(top_k, sims.shape[0])
    if k <= 0:
        return []
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(ids[int(i)], float(sims[int(i)])) for i in top_idx]


# Salient-token patterns for BM25 query synthesis. We deliberately keep
# this shallow (no morphological analyser) because the segment text is
# short (bullet / paragraph / row — typically under 200 chars) and the
# FTS5 trigram tokenizer already handles partial matches. What we need
# is *which* tokens to hand it: kanji runs, katakana runs, ASCII words
# of length >= 2, and numbers optionally followed by a unit. Hiragana-
# only tokens (grammar particles, fillers) are excluded because they
# generate extremely noisy matches.
_SEG_TOKEN_RE = re.compile(
    # Number + unit (e.g. "3日", "30%", "0.5秒") as one token so the
    # unit character survives the len >= 2 filter. Ordered first so a
    # kanji unit isn't absorbed by the bare kanji-run alternative.
    r"\d+(?:\.\d+)?[一-龥ァ-ヴー%]"
    r"|[一-龥々]+"  # kanji run (nouns, tech terms)
    r"|[ァ-ヴー]+"  # katakana run (loanwords, names)
    r"|[A-Za-z][A-Za-z0-9]+"  # latin identifiers / acronyms
    r"|\d+(?:\.\d+)?"  # bare numbers (dropped by len >= 2 unless >= 10)
)
_FTS_MAX_TOKENS = 20


def _build_segment_fts_query(segment_text: str) -> str:
    """Extract salient tokens from a summary segment for BM25 matching.

    Returns an FTS5 query string joining unique tokens with ``OR`` so a
    chunk matches if it contains any of them. Tokens shorter than two
    characters are dropped to cut noise (single kanji / lone digits
    over-match and blow up the query on FTS5's trigram tokenizer).

    Returns an empty string when no qualifying tokens are found — the
    caller treats this as "BM25 contributed no signal" and falls back
    to dense-only ranking for the segment.
    """
    tokens = _SEG_TOKEN_RE.findall(segment_text)
    tokens = [t for t in tokens if len(t) >= 2]
    if not tokens:
        return ""
    # Preserve first-seen order while deduping, then cap the query
    # length to keep FTS5 fast even on pathological segments.
    unique = list(dict.fromkeys(tokens))[:_FTS_MAX_TOKENS]
    # Each token is quoted so the trigram tokenizer treats it as a
    # subphrase match; we OR them together to get union semantics.
    return " OR ".join(f'"{t}"' for t in unique)


def _query_top_chunks_bm25(
    file_id: str, segment_text: str, top_k: int,
    chunk_range: tuple[int, int] | None = None,
) -> list[str]:
    """Return chunk ids ordered by BM25 rank (best first).

    Queries both ``fts_transcripts`` and ``fts_text_content`` scoped
    to ``file_id`` — the caller doesn't need to know which kind of
    source the file has. Ranks are returned as a list (position = rank)
    rather than absolute scores because FTS5's ``bm25()`` score has no
    fixed cosine-like range and we only need the relative ordering for
    RRF fusion.

    When ``chunk_range`` is given, only transcript chunks whose
    ``chunk_index`` falls inside the inclusive ``(lo, hi)`` range are
    kept. ``fts_text_content`` uses string chunk indices (docs
    sometimes embed page markers) and is not range-filtered in SQL;
    we drop non-int rows the same way we do without a range so the
    Python side is the only gate.

    Returns an empty list on tokeniser failure, FTS5 error, or empty
    query. Callers must treat that as "no BM25 signal" and fall back
    to dense-only.
    """
    if top_k <= 0:
        return []

    fts_query = _build_segment_fts_query(segment_text)
    if not fts_query:
        return []

    results: list[str] = []
    seen: set[str] = set()
    try:
        with get_search_db() as session:
            # Transcript chunks (whisper). chunk_index is an integer
            # column, so range filtering can happen in SQL.
            tr_params = {"fid": file_id, "q": fts_query, "k": top_k}
            if chunk_range is None:
                tr_sql = (
                    "SELECT chunk_index FROM fts_transcripts "
                    "WHERE file_id = :fid AND fts_transcripts MATCH :q "
                    "ORDER BY rank LIMIT :k"
                )
            else:
                tr_sql = (
                    "SELECT chunk_index FROM fts_transcripts "
                    "WHERE file_id = :fid AND fts_transcripts MATCH :q "
                    "AND chunk_index BETWEEN :lo AND :hi "
                    "ORDER BY rank LIMIT :k"
                )
                tr_params = {
                    **tr_params,
                    "lo": int(chunk_range[0]),
                    "hi": int(chunk_range[1]),
                }
            rows = session.execute(sql_text(tr_sql), tr_params).fetchall()
            for row in rows:
                cid = f"transcript:{int(row[0])}"
                if cid not in seen:
                    seen = {*seen, cid}
                    results = [*results, cid]

            # Document chunks (text_content). ``chunk_index`` is a
            # string column here (may embed page/section markers), so
            # we try to parse it as an int to match the ``document:{N}``
            # convention; rows that don't parse are skipped rather than
            # surfacing an ambiguous chunk id to the UI. Range filter
            # is applied in Python after the int parse.
            rows = session.execute(
                sql_text(
                    "SELECT chunk_index FROM fts_text_content "
                    "WHERE file_id = :fid AND fts_text_content MATCH :q "
                    "ORDER BY rank LIMIT :k"
                ),
                {"fid": file_id, "q": fts_query, "k": top_k},
            ).fetchall()
            for row in rows:
                try:
                    idx = int(row[0])
                except (TypeError, ValueError):
                    continue
                if chunk_range is not None and not (
                    chunk_range[0] <= idx <= chunk_range[1]
                ):
                    continue
                cid = f"document:{idx}"
                if cid not in seen:
                    seen = {*seen, cid}
                    results = [*results, cid]
    except Exception as e:  # noqa: BLE001 — fail soft: BM25 is optional
        logger.warning(
            "Citation BM25 query failed for %s: %s", file_id, e
        )
        return []

    return results[:top_k]


def _filter_vectors_by_range(
    vectors: list[tuple[str, np.ndarray]],
    chunk_range: tuple[int, int] | None,
) -> list[tuple[str, np.ndarray]]:
    """Drop ``(chunk_id, vector)`` pairs whose index falls outside the range.

    ``chunk_id`` format is ``transcript:<idx>`` or ``document:<idx>``;
    anything that doesn't parse as an int is treated as out-of-range
    so malformed rows can't leak through.
    """
    if chunk_range is None:
        return vectors
    lo, hi = chunk_range
    result: list[tuple[str, np.ndarray]] = []
    for cid, v in vectors:
        _, _, raw = cid.partition(":")
        try:
            idx = int(raw)
        except ValueError:
            continue
        if lo <= idx <= hi:
            result = [*result, (cid, v)]
    return result


def _pick_dense_cluster(
    scored_indices: list[tuple[int, float]],
    gap: int,
    union_ratio: float,
) -> list[int]:
    """Split ``scored_indices`` into contiguous clusters and return the
    indices of the strongest one (optionally unioned with a near-tied
    runner-up).

    ``scored_indices`` is a list of ``(chunk_index, score)`` pairs. A
    new cluster starts wherever the gap between consecutive
    chunk_index values exceeds ``gap``. Each cluster's weight is the
    sum of its scores; the cluster with the largest weight wins.

    When a second-place cluster's weight is at least ``union_ratio``
    of the winner's, it gets merged into the returned index list —
    preserving legitimate "two-ended" sections (a まとめ that
    references both an intro and an outro) without letting an actual
    outlier double-back into the range.

    Returns an empty list only when the input is empty.
    """
    if not scored_indices:
        return []
    # Sort by chunk_index so contiguous clusters come out in order.
    sorted_pairs = sorted(scored_indices, key=lambda p: p[0])

    clusters: list[list[tuple[int, float]]] = [[sorted_pairs[0]]]
    for idx, score in sorted_pairs[1:]:
        prev_idx = clusters[-1][-1][0]
        if idx - prev_idx > gap:
            clusters = [*clusters, [(idx, score)]]
        else:
            clusters[-1] = [*clusters[-1], (idx, score)]

    if len(clusters) == 1:
        return [i for i, _ in clusters[0]]

    weighted = [
        (sum(s for _, s in cluster), cluster) for cluster in clusters
    ]
    weighted.sort(key=lambda x: x[0], reverse=True)
    primary_weight, primary_cluster = weighted[0]

    selected = list(primary_cluster)
    # Union near-tied clusters so a legitimately two-ended section
    # keeps both of its anchors. Guard against a zero-weight primary.
    if primary_weight > 0:
        for weight, cluster in weighted[1:]:
            if weight / primary_weight >= union_ratio:
                selected = [*selected, *cluster]
            else:
                break
    return [i for i, _ in selected]


def _find_range_from_pool(
    pool_vec: np.ndarray,
    file_vectors: list[tuple[str, np.ndarray]],
    parent_range: tuple[int, int] | None,
    top_m: int,
    score_floor: float,
    cluster_gap: int = 5,
    cluster_union_ratio: float = 0.8,
    sibling_pools: list[np.ndarray] | None = None,
    disc_margin: float = 0.01,
) -> tuple[tuple[int, int] | None, float]:
    """Return ``(range, top_1_cosine)`` for a pooled section embedding.

    Computes cosine between ``pool_vec`` and every vector in
    ``file_vectors`` (optionally restricted to ``parent_range``),
    picks the top-M, filters them, then runs dense-cluster detection
    on the surviving chunk indices. The range is the min/max of the
    strongest contiguous cluster (plus a near-tied runner-up) — not
    of the raw top-M — so a handful of high-scoring outliers on the
    other side of the file can't open the range up to cover the
    whole video.

    When ``sibling_pools`` is supplied, scoring switches from raw
    cosine to *discriminative* cosine: ``disc = cos_this -
    max(cos_sibling)``. A chunk is only kept when its raw cosine
    clears ``score_floor`` **and** its discriminative score clears
    ``disc_margin``. This is the fix for "the whole video shares one
    topic": all sibling sections score equally high on every chunk,
    so the absolute cosine can't tell them apart — but the relative
    score (this section vs the best sibling) can.

    ``top_1_cosine`` is always the **raw** top-1 cosine (for the
    caller's threshold check to keep its documented semantics).
    Internal ranking and cluster weights use the effective score
    (raw or disc) so the range reflects "strongest distinctive match".

    Returns ``(None, top_1_cosine)`` when no chunk clears the filter
    combo (caller falls back). Returns ``(None, 0.0)`` when the
    filtered vector pool is empty up front.
    """
    vectors = _filter_vectors_by_range(file_vectors, parent_range)
    if not vectors:
        return None, 0.0
    ids = [cid for cid, _ in vectors]
    matrix = np.stack([v for _, v in vectors])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    unit = matrix / safe
    pnorm = float(np.linalg.norm(pool_vec))
    if pnorm <= 0:
        return None, 0.0
    pool_unit = pool_vec / pnorm
    raw_sims = np.clip(unit @ pool_unit, 0.0, 1.0)

    # Discriminative score: subtract per-chunk max cosine against any
    # sibling pool. Skipped when no usable siblings are available (e.g.
    # top-level singleton prefix); in that case the effective score is
    # just the raw cosine.
    use_disc = False
    effective_sims = raw_sims
    if sibling_pools:
        sibling_unit_list = []
        for sp in sibling_pools:
            if sp is None:
                continue
            sn = float(np.linalg.norm(sp))
            if sn > 0:
                sibling_unit_list = [*sibling_unit_list, sp / sn]
        if sibling_unit_list:
            sib_matrix = np.stack(sibling_unit_list)
            # Per-chunk max cosine over all siblings.
            sib_sims = unit @ sib_matrix.T
            max_sib = np.max(sib_sims, axis=1)
            disc_sims = raw_sims - max_sib
            effective_sims = disc_sims
            use_disc = True

    # top_1 raw cosine is returned regardless of scoring mode so
    # callers can continue to compare against the raw threshold.
    top_raw_score = float(raw_sims.max())

    # Top-M selection uses effective score (raw or disc). This is the
    # key switch: when disc is active, chunks that merely share the
    # whole-video topic no longer dominate the top-M.
    k = min(top_m, effective_sims.shape[0])
    if k <= 0:
        return None, top_raw_score
    idx = np.argpartition(-effective_sims, k - 1)[:k]
    idx = idx[np.argsort(-effective_sims[idx])]

    # Filter: raw cosine must clear the absolute on-topic floor, and
    # disc (if active) must clear the margin. We scan all top-M
    # rather than breaking early because raw and disc scores aren't
    # monotonically aligned once disc is active.
    scored: list[tuple[int, float]] = []
    for i in idx:
        raw = float(raw_sims[int(i)])
        if raw < score_floor:
            continue
        eff = float(effective_sims[int(i)])
        if use_disc and eff < disc_margin:
            continue
        _, _, raw_str = ids[int(i)].partition(":")
        try:
            # Cluster weight uses the effective score so disc-aware
            # mode rewards distinctive matches, not blanket matches.
            scored = [*scored, (int(raw_str), eff)]
        except ValueError:
            continue
    if not scored:
        return None, top_raw_score

    picked = _pick_dense_cluster(
        scored, gap=cluster_gap, union_ratio=cluster_union_ratio,
    )
    if not picked:
        return None, top_raw_score
    return (min(picked), max(picked)), top_raw_score


def _pool_segment_vectors(
    segment_vectors: list[np.ndarray | None],
) -> np.ndarray | None:
    """Mean-pool a list of (possibly None) segment vectors.

    Each vector is unit-normalised first so a single long segment
    with a high-magnitude embedding can't dominate the pool. The
    result is also renormalised so it behaves as a proper unit vector
    for the downstream cosine comparison.
    """
    live = [v for v in segment_vectors if v is not None]
    if not live:
        return None
    matrix = np.stack(live)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    unit = matrix / safe
    pooled = np.mean(unit, axis=0)
    pnorm = float(np.linalg.norm(pooled))
    if pnorm <= 0:
        return None
    return pooled / pnorm


def _viterbi_monotonic_path(emissions: np.ndarray) -> list[int]:
    """Strict-monotonic Viterbi alignment over ``emissions[T, K]``.

    The path starts forced at state 0 and can only stay (k → k) or
    advance by one (k → k+1); no backward transition, no skip. The
    result is the list of states (one per chunk) that maximises the
    summed emission. The ending state is left free — the Viterbi
    picks whichever terminal maximises the cumulative score.

    Use when:

    * The summary's sections are written in chronological order of the
      source.
    * We want every chunk assigned to exactly one section, with no
      section appearing out of order.

    ``emissions[t, k]`` is the affinity of chunk ``t`` to section ``k``
    (higher = better match). Can be raw cosine or discriminative
    cosine — the DP doesn't care.

    Returns an empty list for empty ``emissions``. Single-chunk inputs
    return ``[0]`` (forced start).
    """
    if emissions.size == 0:
        return []
    T, K = emissions.shape
    if T == 0 or K == 0:
        return []

    neg_inf = np.float64(-1e18)
    dp = np.full((T, K), neg_inf, dtype=np.float64)
    backptr = np.full((T, K), -1, dtype=np.int32)

    # Forced start at state 0: no other state is reachable at t=0.
    dp[0, 0] = float(emissions[0, 0])

    for t in range(1, T):
        # State 0 can only be reached from state 0 (stay).
        prev0 = dp[t - 1, 0]
        if prev0 > neg_inf:
            dp[t, 0] = prev0 + float(emissions[t, 0])
            backptr[t, 0] = 0
        for k in range(1, K):
            stay = dp[t - 1, k]
            advance = dp[t - 1, k - 1]
            if stay >= advance and stay > neg_inf:
                dp[t, k] = stay + float(emissions[t, k])
                backptr[t, k] = k
            elif advance > neg_inf:
                dp[t, k] = advance + float(emissions[t, k])
                backptr[t, k] = k - 1

    final = int(np.argmax(dp[T - 1]))
    path: list[int] = [0] * T
    path[T - 1] = final
    for t in range(T - 1, 0, -1):
        path[t - 1] = int(backptr[t, path[t]])
    return path


def _align_sibling_group(
    sibling_pools: list[np.ndarray],
    file_vectors: list[tuple[str, np.ndarray]],
    discriminative: bool,
) -> list[list[int]]:
    """Assign each chunk to exactly one sibling via monotonic Viterbi.

    ``sibling_pools`` must be in the summary's section order (first
    prefix that appears in the markdown first). Returns one list of
    chunk indices per sibling; the order within each list is ascending
    chunk_index. When two adjacent sections compete for the same chunk,
    monotonic DP gives it to whichever wins the *path* maximisation —
    typically the one whose pool is matched around that timestamp and
    whose emergence matches the summary's ordering.

    Discriminative mode (``discriminative=True``) subtracts the max
    emission across the **other** siblings from each sibling's
    emission. This makes the DP favour chunks where the target
    section is distinctively the best fit — the same signal that
    ``_find_range_from_pool`` uses for the pool-based path, carried
    through to the alignment layer.
    """
    K = len(sibling_pools)
    if K == 0:
        return []

    # Parse ``(chunk_index, vector)`` pairs and sort by chunk_index.
    parsed: list[tuple[int, np.ndarray]] = []
    for cid, vec in file_vectors:
        _, _, raw = cid.partition(":")
        try:
            parsed = [*parsed, (int(raw), vec)]
        except ValueError:
            continue
    parsed.sort(key=lambda x: x[0])
    if not parsed:
        return [[] for _ in range(K)]

    T = len(parsed)

    # Normalise each pool into a unit vector; a ``None`` slot becomes
    # a zero row so the DP can never pick that section.
    pool_units: list[np.ndarray | None] = []
    for p in sibling_pools:
        if p is None:
            pool_units = [*pool_units, None]
            continue
        pn = float(np.linalg.norm(p))
        pool_units = [*pool_units, (p / pn) if pn > 0 else None]

    # Build ``T × K`` emission matrix = raw cosine of each chunk
    # against each sibling's pool.
    emissions = np.zeros((T, K), dtype=np.float64)
    for t, (_idx, vec) in enumerate(parsed):
        vn = float(np.linalg.norm(vec))
        if vn <= 0:
            # All-zero vector can't match any pool; leave the row at 0.
            continue
        unit = vec / vn
        for k, pu in enumerate(pool_units):
            if pu is None:
                emissions[t, k] = -1.0  # effectively mask this state
                continue
            emissions[t, k] = float(np.clip(np.dot(unit, pu), 0.0, 1.0))

    # Discriminative adjustment: for each ``k`` subtract the max
    # emission over the other siblings. A chunk that matches this
    # section *more than any sibling* keeps a positive score; one
    # that matches a sibling more becomes negative and loses to
    # staying in the neighbouring state.
    if discriminative and K >= 2:
        adjusted = np.empty_like(emissions)
        for k in range(K):
            if k == 0:
                others = emissions[:, 1:]
            elif k == K - 1:
                others = emissions[:, :-1]
            else:
                others = np.concatenate(
                    [emissions[:, :k], emissions[:, k + 1:]], axis=1
                )
            max_other = others.max(axis=1)
            adjusted[:, k] = emissions[:, k] - max_other
        emissions = adjusted

    path = _viterbi_monotonic_path(emissions)

    assignments: list[list[int]] = [[] for _ in range(K)]
    for t, state in enumerate(path):
        chunk_idx, _ = parsed[t]
        assignments[state] = [*assignments[state], chunk_idx]
    return assignments


def _build_hierarchical_range_map(
    segments: list[Segment],
    segment_vectors: list[np.ndarray | None],
    file_vectors: list[tuple[str, np.ndarray]],
) -> dict[tuple[str, ...], tuple[int, int] | None]:
    """Build a ``prefix → chunk_range`` map with each prefix resolved independently.

    For every prefix of every segment's ``ancestor_headings`` chain,
    pool the embeddings of segments under that prefix and find the
    tightest chunk range where the pool's content is discussed. Each
    prefix is resolved **independently against the full file**; a
    parent prefix's range is *not* used to constrain a child's
    search. The asymmetry between prefixes and segments is:

    * **Prefixes** are resolved independently of one another. Two
      sibling prefixes may land on overlapping or disjoint ranges,
      and that's correct — each one's pool is its own topic signal.
    * **Segments** look up the deepest-prefix's range. A segment's
      search is scoped to "its own section's range", not to
      "everything above it in the hierarchy".

    This avoids the recipe-video regression where a structural
    parent like ``## 詳細内容`` (a pure container holding six
    unrelated recipes under ``###``) mixed six distinct topics into
    one pool; cluster detection picked one recipe's zone and
    cascade-constrained every child section to that zone. With no
    cascade, each child section's own pool finds its own recipe zone.

    Prefixes whose pool can't cluster above ``citation_section_narrow_threshold``
    map to ``None``; the caller treats that as "no narrowing, full-
    file search" for segments hanging off that prefix.

    The returned map uses the full ``ancestor_headings`` tuple as the
    key so ``prefix_range[seg.ancestor_headings]`` gives the segment's
    direct section range. Segments with empty ancestors aren't in the
    map; the caller also treats that as full-file search.
    """
    threshold = settings.summaries.citation_section_narrow_threshold
    top_m = settings.summaries.citation_section_range_top_m
    cluster_gap = settings.summaries.citation_section_cluster_gap
    cluster_union_ratio = settings.summaries.citation_section_cluster_union_ratio
    disc_enabled = settings.summaries.citation_section_discriminative_enabled
    disc_margin = settings.summaries.citation_section_disc_margin
    alignment_enabled = settings.summaries.citation_section_alignment_enabled
    boundary_margin = settings.summaries.citation_section_boundary_margin

    # 1. Enumerate unique prefixes (every prefix of every ancestor chain).
    unique_prefixes: set[tuple[str, ...]] = set()
    for seg in segments:
        for depth in range(1, len(seg.ancestor_headings) + 1):
            unique_prefixes = unique_prefixes | {
                seg.ancestor_headings[:depth]
            }
    if not unique_prefixes:
        return {}

    # 2. For each prefix, collect the segments below it and pool their
    #    already-computed embeddings. We look up by index so a segment
    #    missing a valid embedding is skipped cleanly.
    seg_index = {id(s): i for i, s in enumerate(segments)}
    prefix_pool: dict[tuple[str, ...], np.ndarray] = {}
    for prefix in unique_prefixes:
        plen = len(prefix)
        members = [
            s for s in segments
            if len(s.ancestor_headings) >= plen
            and s.ancestor_headings[:plen] == prefix
        ]
        vecs = [segment_vectors[seg_index[id(s)]] for s in members]
        pooled = _pool_segment_vectors(vecs)
        if pooled is not None:
            prefix_pool[prefix] = pooled

    # 3. Resolve each prefix's range.
    #
    #    When ``alignment_enabled`` and a parent has two or more
    #    children with usable pools, run Viterbi monotonic alignment
    #    across that sibling group: each chunk gets assigned to the
    #    single sibling whose pool it best matches, *subject to the
    #    summary order*. A section's range is then the ``[min, max]``
    #    of its assigned chunks. This directly implements the user
    #    intuition "sections appear in chronological order of the
    #    source" — and is robust against the case where a section's
    #    pool broadly matches the whole video (the DP path picks the
    #    zone consistent with the sections that come before and
    #    after, not just the highest absolute cosine).
    #
    #    When only a single usable sibling exists, or alignment is
    #    disabled, the prefix falls back to ``_find_range_from_pool``
    #    (pool + cluster detection + optional discriminative).
    #
    #    Siblings are the other prefixes at the same depth under the
    #    same parent. Summary order is reconstructed from the first
    #    occurrence of each prefix in the parsed segments list.
    prefix_range: dict[tuple[str, ...], tuple[int, int] | None] = {}

    # Group prefixes by parent.
    prefixes_by_parent: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
    for prefix in unique_prefixes:
        parent = prefix[:-1]
        prefixes_by_parent.setdefault(parent, []).append(prefix)

    # Summary order of each prefix (first index of a segment with that
    # prefix on its ancestor chain). Stable tie-break by current order.
    first_seen: dict[tuple[str, ...], int] = {}
    for i, seg in enumerate(segments):
        for depth in range(1, len(seg.ancestor_headings) + 1):
            p = seg.ancestor_headings[:depth]
            if p not in first_seen:
                first_seen[p] = i

    for parent, siblings in prefixes_by_parent.items():
        ordered = sorted(siblings, key=lambda p: first_seen.get(p, 0))
        with_pool = [(p, prefix_pool[p]) for p in ordered if p in prefix_pool]
        without_pool = [p for p in ordered if p not in prefix_pool]
        for p in without_pool:
            prefix_range[p] = None

        if alignment_enabled and len(with_pool) >= 2:
            ordered_prefixes = [p for p, _ in with_pool]
            ordered_pools = [pool for _, pool in with_pool]
            assignments = _align_sibling_group(
                ordered_pools, file_vectors, discriminative=disc_enabled,
            )
            for idx, prefix in enumerate(ordered_prefixes):
                chunks = assignments[idx] if idx < len(assignments) else []
                if chunks:
                    lo = min(chunks)
                    hi = max(chunks)
                    if boundary_margin > 0:
                        lo = max(0, lo - boundary_margin)
                        hi = hi + boundary_margin
                    prefix_range[prefix] = (lo, hi)
                else:
                    prefix_range[prefix] = None
            continue

        # Pool-based fallback (single sibling or alignment disabled).
        for prefix, pool_vec in with_pool:
            sibling_pools: list[np.ndarray] | None = None
            if disc_enabled:
                sibs = [pool for p, pool in with_pool if p != prefix]
                sibling_pools = sibs if sibs else None
            new_range, top_score = _find_range_from_pool(
                pool_vec, file_vectors, None, top_m,
                score_floor=threshold,
                cluster_gap=cluster_gap,
                cluster_union_ratio=cluster_union_ratio,
                sibling_pools=sibling_pools,
                disc_margin=disc_margin,
            )
            if new_range is None or top_score < threshold:
                prefix_range[prefix] = None
            else:
                prefix_range[prefix] = new_range
    return prefix_range


def _retrieve_candidates(
    file_id: str,
    segment: Segment,
    query_vector: np.ndarray,
    top_k: int,
    section_range: tuple[int, int] | None = None,
    file_vectors: list[tuple[str, np.ndarray]] | None = None,
) -> list[tuple[str, float]]:
    """Return ``[(chunk_id, cosine_score), ...]`` for the final top-K.

    Hybrid pipeline:

    1. Pull a dense candidate pool of size
       ``citation_top_k_internal`` (>= ``top_k``). Each candidate
       comes with its cosine similarity. When ``section_range`` is
       given, the pool is drawn only from chunks inside that range —
       this prevents "another recipe's 保存方法 chunk" from entering
       the pool at all.
    2. If hybrid retrieval is enabled, run a BM25 query against the
       FTS5 mirrors (also range-filtered) and take the ranks of the
       chunks that are already in the dense pool (BM25-only
       candidates are discarded to keep ``top_score`` = dense cosine,
       so ``citation_threshold`` keeps its documented meaning).
    3. RRF-fuse the two rankings and reorder the dense pool by the
       fused score. Cosine is preserved as the returned score.

    Graceful degradation for ``section_range``: if the range is given
    but produces zero dense candidates (heading was anchored wrong, or
    the section content is entirely in a different modality), we
    retry without the range rather than returning empty. This trades
    "no citation" for "possibly-wrong citation" only when the anchor
    is demonstrably broken.

    When ``citation_hybrid_enabled`` is False, the function degrades
    to ``_query_top_chunks_dense`` directly — no FTS5 access and no
    BM25 dependency. Same for files that produce no BM25 hits (empty
    FTS5 query / no salient tokens).
    """
    if top_k <= 0:
        return []

    pool_size = max(
        top_k, settings.summaries.citation_top_k_internal
    )
    dense = _query_top_chunks_dense(
        file_id, query_vector, pool_size,
        chunk_range=section_range,
        file_vectors=file_vectors,
    )
    if not dense and section_range is not None:
        # Anchor produced an empty pool — fall back to full-file
        # search. Better to return a possibly-distant citation than to
        # drop the segment to ⚠ purely because of a bad heading match.
        dense = _query_top_chunks_dense(
            file_id, query_vector, pool_size, file_vectors=file_vectors,
        )
    if not dense:
        return []

    if not settings.summaries.citation_hybrid_enabled:
        return dense[:top_k]

    sparse = _query_top_chunks_bm25(
        file_id,
        segment.segment_text,
        pool_size,
        chunk_range=section_range,
    )
    if not sparse:
        # Fail soft: without BM25 signal the dense order is the
        # safest bet, same as legacy behaviour.
        return dense[:top_k]

    rrf_k = settings.summaries.citation_rrf_k
    rrf_scores: dict[str, float] = {}
    for rank, (cid, _) in enumerate(dense):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (
            rrf_k + rank + 1
        )
    for rank, cid in enumerate(sparse):
        if cid in rrf_scores:
            # Reward candidates that appear in both rankings.
            rrf_scores[cid] = rrf_scores[cid] + 1.0 / (
                rrf_k + rank + 1
            )

    dense_score_map = dict(dense)
    reordered = sorted(
        rrf_scores.items(), key=lambda kv: kv[1], reverse=True
    )
    return [
        (cid, dense_score_map[cid])
        for cid, _rrf in reordered
        if cid in dense_score_map
    ][:top_k]


def _pool_cell_vectors(vectors) -> np.ndarray | None:
    """Element-wise max-pool a list of cell vectors, then renormalise.

    ``embed_passages`` returns L2-normalised unit vectors (shared
    convention with transcript / document indexing). Element-wise max
    across them yields a vector whose every dimension reflects the
    strongest cell signal in that dimension; renormalising restores a
    comparable magnitude for cosine lookups. This is an approximation
    of "match any cell", cheap enough to run inline per row.

    Accepts either a list of 1-D vectors or a pre-stacked 2-D
    ``numpy.ndarray``. Returns ``None`` when the input is empty or
    every pooled dimension is zero (shouldn't happen in practice but
    keeps the caller branch simple). Uses identity / length checks
    rather than truthiness so numpy arrays don't raise the
    "truth value ambiguous" error.
    """
    if vectors is None:
        return None
    try:
        arr = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    pooled = np.max(arr, axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm <= 0:
        return None
    return pooled / norm


# Punctuation used to split compound bullets into sub-anchors. Full-stop
# 「。」 and the common CJK list separators are the signal we trust;
# latin commas / semicolons are also accepted so mixed-script bullets
# still split. We deliberately do NOT split on 「と」「や」「・」-prefix
# particles or 「て」-form verbs — those are grammatical links that
# connect parts of one claim, not separators that enumerate independent
# anchors, so splitting on them inflates the sub-segment count with
# fragments that hurt retrieval.
_COMPOUND_BULLET_SPLIT_RE = re.compile(r"[。、，,・；;]+")


def _split_compound_segment(segment: Segment) -> list[str]:
    """Return sub-segment texts for a compound bullet, or ``[segment_text]``.

    A summary bullet often carries multiple sub-anchors on one line —
    e.g. "洗って芯を切り落とし、葉と芯を分けて千切りにする" (four
    kitchen operations) or "にんじんは3本、手元の分量でもよい" (two
    separate facts). A single embedding of the joined text blurs
    across those anchors; the retriever ends up matching a
    neighbouring "theme" chunk whose register matches the summary's
    declarative tone rather than the imperative chunks where each
    anchor actually lives.

    Splitting on CJK punctuation (``、 。 ・ ， , ； ;``) recovers each
    anchor as its own fragment, which the caller can then embed and
    retrieve independently. The result is unioned by max-score so a
    compound bullet ends up with chunk citations for each of its
    sub-anchors instead of one under-determined top-1.

    Rules for keeping a sub-fragment:

    * it must be at least ``citation_multi_anchor_min_len`` characters
    * it must contain at least one salient token (kanji / katakana
      run, numeric-with-unit, or latin identifier) — hiragana-only
      fragments are grammatical, not anchors.

    Skipped unconditionally:

    * table rows (``segment.cells`` is set) — per-cell pooling in
      ``_pool_cell_vectors`` already handles multi-cell row semantics.
    * non-bullets (paragraphs) — claim-vs-example bias on paragraphs
      is a separate structural problem; splitting a paragraph on a
      list-style 、 would over-fire.

    Returns a list of length >= 1. A single-element list means the
    caller should use the segment's full text unchanged (no multi-
    anchor fan-out).
    """
    text = segment.segment_text.strip()
    if not text:
        return [text]
    if segment.cells is not None:
        return [text]
    if segment.segment_type != "bullet":
        return [text]

    min_len = settings.summaries.citation_multi_anchor_min_len
    raw_parts = _COMPOUND_BULLET_SPLIT_RE.split(text)

    kept: list[str] = []
    for part in raw_parts:
        stripped = part.strip()
        if len(stripped) < min_len:
            continue
        if not _SEG_TOKEN_RE.search(stripped):
            continue
        kept = [*kept, stripped]

    if len(kept) < 2:
        return [text]
    return kept


def _multi_anchor_retrieve(
    file_id: str,
    segment: Segment,
    sub_texts: list[str],
    top_k: int,
    section_range: tuple[int, int] | None,
    file_vectors: list[tuple[str, np.ndarray]] | None,
) -> list[tuple[str, float]]:
    """Retrieve candidates per sub-segment and union by max score.

    Embeds each ``sub_texts`` entry separately, runs
    ``_retrieve_candidates`` once per sub-segment (using the same
    ``section_range``), and merges the results into a single
    ``[(chunk_id, score), ...]`` list. A chunk appearing under
    multiple sub-segments keeps its highest observed cosine — which
    naturally boosts chunks that serve more than one anchor, without
    double-counting.

    The embedding call is per sub-segment on purpose: the whole point
    of splitting is to ask the index about each anchor independently
    rather than about their mean. Returns an empty list when every
    sub-segment fails to embed (caller falls back).
    """
    try:
        from app.workers.embedder import embed_passages
    except Exception as e:  # noqa: BLE001 — fail soft
        logger.warning(
            "Multi-anchor embed import failed for %s: %s",
            segment.section_path, e,
        )
        return []

    try:
        sub_vectors = embed_passages(sub_texts)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Multi-anchor embed failed for %s: %s",
            segment.section_path, e,
        )
        return []
    if sub_vectors is None or len(sub_vectors) == 0:
        return []

    best_score: dict[str, float] = {}
    # Each sub-segment needs its own ``Segment`` shape so
    # ``_retrieve_candidates`` can build a BM25 query from the sub-
    # text rather than from the compound bullet's joined tokens.
    for i, sub_vec in enumerate(sub_vectors):
        try:
            arr = np.asarray(sub_vec, dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if arr.ndim != 1 or arr.size == 0:
            continue
        sub_seg = Segment(
            section_path=segment.section_path,
            segment_type=segment.segment_type,
            segment_text=sub_texts[i],
            ancestor_headings=segment.ancestor_headings,
        )
        candidates = _retrieve_candidates(
            file_id, sub_seg, arr, top_k,
            section_range=section_range,
            file_vectors=file_vectors,
        )
        for cid, score in candidates:
            prev = best_score.get(cid)
            if prev is None or score > prev:
                best_score[cid] = score

    merged = sorted(best_score.items(), key=lambda kv: kv[1], reverse=True)
    return merged[:top_k]


def _embed_segment(segment: Segment) -> np.ndarray | None:
    """Embed one segment's text using the shared passage encoder.

    Table rows with two or more distinct cells are embedded cell-by-
    cell and max-pooled so header labels don't dominate the joined
    string. For paragraphs, bullets, and single-cell rows the method
    falls back to a single embedding of ``segment_text``.

    Imports ``embed_passages`` lazily so tests that don't exercise the
    real embedder (most of the suite) can stub it out before the first
    citation call. Returns ``None`` on embedding failure so we can
    persist a ``has_citation = False`` row instead of crashing.
    """
    text = segment.segment_text.strip()
    if not text:
        return None

    cell_texts: list[str] = []
    if segment.cells and len(segment.cells) >= 2:
        cell_texts = [c.strip() for c in segment.cells if c.strip()]

    try:
        from app.workers.embedder import embed_passages

        if len(cell_texts) >= 2:
            cell_vectors = embed_passages(cell_texts)
            pooled = _pool_cell_vectors(cell_vectors)
            if pooled is not None:
                return pooled
            # Pooling failed (e.g. all-zero cells) — fall through to
            # the joined-text embed so the segment still gets a shot.
        vectors = embed_passages([text])
    except Exception as e:  # noqa: BLE001 — fail soft
        logger.warning(
            "Citation embedding failed for %s: %s", segment.section_path, e
        )
        return None
    if vectors is None or len(vectors) == 0:
        return None
    return np.asarray(vectors[0], dtype=np.float32)


def compute_citations(
    file_id: str, detailed_summary: str
) -> list[dict]:
    """Compute (but do not persist) citations for a detailed_summary.

    Split into its own function so the worker path can persist via
    :func:`write_citations` while tests can assert on the raw output.

    Each returned dict has the keys used by ``write_citations``:

    * ``section_path``         — from the parser
    * ``segment_type``         — from the parser
    * ``segment_text``         — trimmed copy of the raw segment
    * ``citation_chunk_ids``   — list of chunk ids whose score passes threshold
    * ``top_score``            — the top-1 cosine similarity (0.0 if no chunks)
    * ``has_citation``         — True iff top-1 >= threshold
    """
    threshold = settings.summaries.citation_threshold
    top_k = settings.summaries.citation_top_k
    margin_gate = settings.summaries.citation_margin_gate
    margin_bypass = settings.summaries.citation_margin_bypass_score

    segments = parse_segments(detailed_summary)
    results: list[dict] = []

    # Pre-compute all segment embeddings once so the hierarchical pool
    # (which uses the same embeddings) and per-segment retrieval share
    # work. Missing embeddings become None and are reported as ⚠.
    segment_vectors: list[np.ndarray | None] = [
        _embed_segment(seg) for seg in segments
    ]

    # Pre-fetch the file's vectors once; both the hierarchical range
    # map and every ``_retrieve_candidates`` call read from this
    # shared pool instead of re-querying the DB per segment.
    file_vectors: list[tuple[str, np.ndarray]] = []
    section_ranges: dict[tuple[str, ...], tuple[int, int] | None] = {}
    if settings.summaries.citation_section_anchor_enabled:
        file_vectors = _fetch_file_vectors(file_id)
        if file_vectors:
            section_ranges = _build_hierarchical_range_map(
                segments, segment_vectors, file_vectors,
            )

    for i, seg in enumerate(segments):
        vector = segment_vectors[i]
        if vector is None:
            results.append(
                {
                    "section_path": seg.section_path,
                    "segment_type": seg.segment_type,
                    "segment_text": seg.segment_text,
                    "citation_chunk_ids": [],
                    "top_score": 0.0,
                    "has_citation": False,
                }
            )
            continue

        section_range = (
            section_ranges.get(seg.ancestor_headings)
            if seg.ancestor_headings
            else None
        )

        # Compound-bullet multi-anchor retrieval (see
        # ``_split_compound_segment``). When a bullet carries several
        # punctuation-separated sub-anchors, run retrieval per sub-
        # segment AND keep the baseline joined-text candidates, then
        # union both pools by max score. Each sub-text surfaces
        # chunks that specifically match one anchor; the joined text
        # still contributes chunks that only rank because they
        # weakly match multiple anchors together (signal that
        # disappears when any single anchor is considered alone).
        # Max-score union doesn't guarantee a baseline chunk keeps
        # its rank — in principle several strong multi chunks can
        # push a baseline chunk out of top-K — but it's far less
        # aggressive than letting multi's ordering fully occupy the
        # top slots, and it was the approach whose eval showed no
        # per-segment regressions on the curated cases. Falls through
        # to the single-embedding path when the split yields 0-1
        # usable sub-segments or when the feature is disabled.
        candidates = _retrieve_candidates(
            file_id, seg, vector, top_k,
            section_range=section_range,
            file_vectors=file_vectors or None,
        )
        if settings.summaries.citation_multi_anchor_enabled:
            sub_texts = _split_compound_segment(seg)
            if len(sub_texts) >= 2:
                multi = _multi_anchor_retrieve(
                    file_id, seg, sub_texts, top_k,
                    section_range=section_range,
                    file_vectors=file_vectors or None,
                )
                if multi:
                    merged: dict[str, float] = dict(candidates)
                    for cid, score in multi:
                        prev = merged.get(cid)
                        if prev is None or score > prev:
                            merged[cid] = score
                    candidates = sorted(
                        merged.items(), key=lambda kv: kv[1], reverse=True
                    )[:top_k]
        if not candidates:
            results.append(
                {
                    "section_path": seg.section_path,
                    "segment_type": seg.segment_type,
                    "segment_text": seg.segment_text,
                    "citation_chunk_ids": [],
                    "top_score": 0.0,
                    "has_citation": False,
                }
            )
            continue

        top_score = candidates[0][1]
        has_citation = top_score >= threshold
        # Margin gate: when the top-1 score is in the borderline band
        # (threshold .. margin_bypass) and the gap to top-2 is small,
        # the pick is low-confidence — likely multiple chunks look
        # comparably close and picking any one of them would mislead.
        # Flip to ⚠ so the UI is honest about uncertainty. Clearly
        # strong matches (``top_score >= margin_bypass``) bypass the
        # gate because a close runner-up there usually means the
        # segment legitimately has several matching sources.
        if (
            has_citation
            and margin_gate > 0
            and len(candidates) >= 2
            and top_score < margin_bypass
        ):
            margin = candidates[0][1] - candidates[1][1]
            if margin < margin_gate:
                has_citation = False
        # Only persist chunks whose individual score clears the bar.
        # This keeps the UI promise ("shown citations are strong") but
        # still records top_score so the ⚠ marker can fire even when
        # the chunk list is empty. When the margin gate demotes the
        # segment we also drop the chunk_ids so the UI doesn't show
        # suggestive but unreliable anchors.
        if has_citation:
            passing = [
                cid for cid, score in candidates if score >= threshold
            ]
        else:
            passing = []
        results.append(
            {
                "section_path": seg.section_path,
                "segment_type": seg.segment_type,
                "segment_text": seg.segment_text,
                "citation_chunk_ids": passing,
                "top_score": top_score,
                "has_citation": has_citation,
            }
        )
    return results


def write_citations(file_id: str, citations: list[dict]) -> tuple[int, int]:
    """Replace all stored citations for ``file_id``.

    Returns ``(citation_count, no_citation_count)`` for use in the
    ``citations_ready`` WebSocket payload. ``citation_count`` is the
    number of segments with ``has_citation = True``; the "no" count
    is the complement (segments the LLM produced without a strong
    source anchor).

    Safe to call with an empty ``citations`` list: the existing rows
    are still wiped, leaving the file in a "no citations computed"
    state. The API endpoint renders this identically to "no detailed
    summary" so the UI stays quiet.
    """
    with_count = sum(1 for c in citations if c["has_citation"])
    without_count = len(citations) - with_count

    with get_search_db() as session:
        session.execute(
            sql_text(
                "DELETE FROM detailed_summary_citations "
                "WHERE file_id = :fid"
            ),
            {"fid": file_id},
        )
        for cit in citations:
            session.execute(
                sql_text(
                    "INSERT INTO detailed_summary_citations "
                    "(file_id, section_path, segment_type, segment_text, "
                    "citation_chunk_ids, top_score, has_citation) "
                    "VALUES (:fid, :section_path, :segment_type, "
                    ":segment_text, :citation_chunk_ids, :top_score, "
                    ":has_citation)"
                ),
                {
                    "fid": file_id,
                    "section_path": cit["section_path"],
                    "segment_type": cit["segment_type"],
                    "segment_text": cit["segment_text"],
                    "citation_chunk_ids": json.dumps(
                        cit["citation_chunk_ids"]
                    ),
                    "top_score": float(cit["top_score"]),
                    "has_citation": bool(cit["has_citation"]),
                },
            )
    return (with_count, without_count)


def calculate_and_store(
    file_id: str, detailed_summary: str
) -> tuple[int, int]:
    """Compute citations for ``file_id`` and persist them.

    Convenience wrapper for workers: equivalent to
    ``write_citations(file_id, compute_citations(file_id, summary))``.
    Returns ``(citation_count, no_citation_count)`` so the caller can
    emit the ``citations_ready`` WebSocket event without a second
    database round trip.
    """
    citations = compute_citations(file_id, detailed_summary)
    return write_citations(file_id, citations)


def get_citations(file_id: str) -> list[dict]:
    """Fetch all citations for ``file_id`` in section order.

    Rows are returned as plain dicts with the fields shaped for the
    API response (JSON array of chunk ids is decoded; booleans are
    coerced to native Python bools). Returns an empty list when no
    citations have been computed yet.
    """
    with get_search_db() as session:
        rows = session.execute(
            sql_text(
                "SELECT section_path, segment_type, segment_text, "
                "citation_chunk_ids, top_score, has_citation "
                "FROM detailed_summary_citations "
                "WHERE file_id = :fid "
                "ORDER BY id"
            ),
            {"fid": file_id},
        ).fetchall()

    results: list[dict] = []
    for row in rows:
        try:
            chunk_ids = json.loads(row[3]) if row[3] else []
        except (TypeError, ValueError):
            chunk_ids = []
        results.append(
            {
                "section_path": row[0],
                "segment_type": row[1],
                "segment_text": row[2],
                "chunk_ids": chunk_ids,
                "top_score": float(row[4]),
                "has_citation": bool(row[5]),
            }
        )
    return results


def delete_citations(file_id: str) -> int:
    """Delete all citations for ``file_id``. Returns row count deleted."""
    with get_search_db() as session:
        result = session.execute(
            sql_text(
                "DELETE FROM detailed_summary_citations "
                "WHERE file_id = :fid"
            ),
            {"fid": file_id},
        )
        return int(result.rowcount or 0)
