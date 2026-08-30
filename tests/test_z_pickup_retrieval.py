"""Pickup's candidate generation, held to the guarantees the feed needs.

The feed asks a different question from the file-detail "similar files"
section, and the two need opposite answers when the evidence is weak.
The section is right to say nothing rather than something thin; the feed
exists to produce quantity, and a lane that goes silent contributes
nothing to the interleave.

It also cannot use a k-nearest-neighbour query. sqlite-vec caps k at
4096 rows in the extension itself, and the exclusion set here *is* the
viewer's history while the query vector is built out of that same
history — so the nearest rows are overwhelmingly rows that must be
dropped. Scope and channel are settled in one join instead, and one
matmul scores every lane.

Every fixture that proves a file is excluded first proves the same file
is retrievable on its own terms. Without that control a test passes
when some *other* filter swallows the fixture, which is how the drive
boundary — the only security boundary in this code — was once covered
by three tests that all passed with the boundary deleted.
"""

from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pytest

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.models import Base, Embedding, IndexedFile  # noqa: E402
from app.pickup import retrieval  # noqa: E402

DIM = 8


def _query() -> np.ndarray:
    """The lane centroid every fixture is positioned against."""
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = 1.0
    return v


def _at_cosine(cos: float, axis: int = 0, off_axis: int = 1) -> np.ndarray:
    """A unit vector whose cosine with basis ``axis`` is exactly ``cos``."""
    v = np.zeros(DIM, dtype=np.float32)
    v[axis] = cos
    v[off_axis] = math.sqrt(max(0.0, 1.0 - cos * cos))
    return v


class _Index:
    """A two-drive index with stand-ins for both vector tables."""

    def __init__(self, tmp_path):
        self.engine = create_engine(
            f"sqlite:///{tmp_path / 'search.db'}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self._maker = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.engine.begin() as conn:
            for table in ("vec_clip", "vec_text"):
                conn.execute(text(
                    f"CREATE TABLE {table} "
                    "(embedding_id TEXT PRIMARY KEY, vector BLOB)"
                ))

    def add_file(
        self,
        file_id: str,
        drive: str,
        *,
        cosines: list[float] | None = None,
        channel: str = "clip_thumbnail",
        active: bool = True,
        vectors: list[np.ndarray] | None = None,
    ):
        """Register a file with one embedding per requested cosine."""
        rows = (
            vectors if vectors is not None
            else [_at_cosine(c) for c in (cosines or [])]
        )
        session = self._maker()
        try:
            if not session.get(IndexedFile, file_id):
                session.add(IndexedFile(
                    file_id=file_id, drive=drive, filename=f"{file_id}.mp4",
                    file_path=f"/drives/{drive}/{file_id}.mp4",
                    file_type="video", mime_type="video/mp4",
                    file_size=1, active=active,
                ))
            for i in range(len(rows)):
                session.add(Embedding(
                    id=f"e_{file_id}_{channel}_{i}", file_id=file_id,
                    embedding_type=channel,
                    vector_table=retrieval.vector_table_for(channel),
                    content_preview="x",
                ))
            session.commit()
        finally:
            session.close()

        table = retrieval.vector_table_for(channel)
        with self.engine.begin() as conn:
            for i, vector in enumerate(rows):
                conn.execute(
                    text(f"INSERT INTO {table} VALUES (:i, :v)"),
                    {"i": f"e_{file_id}_{channel}_{i}", "v": vector.tobytes()},
                )

    def add_many(self, prefix: str, count: int, drive: str,
                 *, cos_start: float, step: float,
                 channel: str = "clip_thumbnail"):
        """Register ``count`` files in one transaction.

        The per-file path opens a session each time, which is fine for a
        handful and far too slow for the thousands a ceiling-free search
        has to be shown handling.
        """
        table = retrieval.vector_table_for(channel)
        session = self._maker()
        try:
            for i in range(count):
                file_id = f"{prefix}{i}"
                session.add(IndexedFile(
                    file_id=file_id, drive=drive, filename=f"{file_id}.mp4",
                    file_path=f"/drives/{drive}/{file_id}.mp4",
                    file_type="video", mime_type="video/mp4",
                    file_size=1, active=True,
                ))
                session.add(Embedding(
                    id=f"e_{file_id}_{channel}_0", file_id=file_id,
                    embedding_type=channel, vector_table=table,
                    content_preview="x",
                ))
            session.commit()
        finally:
            session.close()
        with self.engine.begin() as conn:
            conn.execute(
                text(f"INSERT INTO {table} VALUES (:i, :v)"),
                [
                    {"i": f"e_{prefix}{i}_{channel}_0",
                     "v": _at_cosine(cos_start + i * step).tobytes()}
                    for i in range(count)
                ],
            )

    @contextmanager
    def search_db(self):
        session = self._maker()
        try:
            yield session
        finally:
            session.close()


@pytest.fixture()
def index(monkeypatch, tmp_path):
    idx = _Index(tmp_path)
    monkeypatch.setattr(retrieval, "get_search_engine", lambda: idx.engine)
    monkeypatch.setattr(retrieval, "get_search_db_read", idx.search_db)
    return idx


def _top(drive="a", channel="clip_thumbnail", exclude=(), limit=10,
         centroids=None):
    candidates = retrieval.load_candidates(drive=drive, channel=channel)
    lanes = retrieval.score_lanes(
        candidates,
        centroids if centroids is not None else [_query()],
        exclude_file_ids=set(exclude),
        limit=limit,
    )
    return [[file_id for file_id, _ in lane] for lane in lanes]


# ---------------------------------------------------------------------------
# The drive boundary
# ---------------------------------------------------------------------------


def test_another_drives_files_are_never_scored(index):
    """The boundary is structural: those rows never enter the matrix.

    The first assertion is the control, and it is load-bearing. Choose
    cosines that trip some *other* filter and the boundary is never
    asked, so the test passes with the boundary deleted.
    """
    for i in range(5):
        index.add_file(f"b{i}", "b", cosines=[0.95 - i * 0.01])
    index.add_file("a1", "a", cosines=[0.80])
    index.add_file("a2", "a", cosines=[0.70])

    assert _top(drive="b")[0] == ["b0", "b1", "b2", "b3", "b4"]
    assert _top(drive="a")[0] == ["a1", "a2"]


def test_a_drive_in_the_minority_still_finds_its_own(index):
    """No page at which the search gives up.

    A KNN would spend its 4096-row budget on the other drive's rows
    before reaching this drive's. Here they were never candidates.
    """
    index.add_many("b", 5000, "b", cos_start=0.99, step=-0.00001)
    index.add_file("a1", "a", cosines=[0.50])

    assert _top(drive="a", limit=1)[0] == ["a1"]


def test_the_boundary_is_the_only_thing_keeping_them_out(index):
    """A guard on the guard, so the fixtures above cannot go vacuous."""
    for i in range(5):
        index.add_file(f"b{i}", "b", cosines=[0.95 - i * 0.01])

    assert len(_top(drive="b")[0]) == 5
    assert _top(drive="a")[0] == []


def test_inactive_files_are_not_candidates(index):
    index.add_file("gone", "a", cosines=[0.95], active=False)
    index.add_file("here", "a", cosines=[0.70])

    assert _top()[0] == ["here"]


# ---------------------------------------------------------------------------
# Chunks reduce to files by max, not by mean
# ---------------------------------------------------------------------------


def test_a_document_is_scored_by_its_best_chunk(index):
    """One chunk squarely on the subject is what makes a file a candidate.

    Averaging 53 chunks would blur a long document into a mediocre score
    against every lane. The profile takes the mean, because there the
    question is what a document is about as a whole; here it is whether
    any part of it is about this.
    """
    index.add_file(
        "long", "a",
        cosines=[0.95] + [0.10] * 52, channel="text_content",
    )
    index.add_file("short", "a", cosines=[0.60], channel="text_content")

    lane = retrieval.score_lanes(
        retrieval.load_candidates(drive="a", channel="text_content"),
        [_query()], exclude_file_ids=set(), limit=10,
    )[0]

    assert [f for f, _ in lane] == ["long", "short"]
    assert lane[0][1] == pytest.approx(0.95, abs=1e-5)


def test_a_file_appears_once_however_many_chunks_it_has(index):
    index.add_file(
        "doc", "a", cosines=[0.9, 0.85, 0.8], channel="text_content",
    )

    lane = _top(channel="text_content")[0]

    assert lane == ["doc"]


# ---------------------------------------------------------------------------
# Weak evidence must still produce candidates
# ---------------------------------------------------------------------------


def test_a_flat_neighbourhood_still_yields_candidates(index):
    """Scores alike to three decimals are normal around a cluster centre.

    ``find_similar`` reads that shape as "this embedding cannot tell
    these files apart" and discards every candidate, twice — once by the
    gap check and once by the coefficient of variation.
    """
    for i in range(10):
        index.add_file(f"a{i}", "a", cosines=[0.700 + i * 0.0001])

    assert len(_top(limit=10)[0]) == 10


def test_candidates_far_below_the_top_score_are_kept(index):
    """No margin cutoff, the largest single cause of an empty feed."""
    index.add_file("close", "a", cosines=[0.98])
    index.add_file("middling", "a", cosines=[0.70])
    index.add_file("distant", "a", cosines=[0.46])

    assert _top()[0] == ["close", "middling", "distant"]


def test_candidates_below_the_floor_are_dropped(index):
    index.add_file("good", "a", cosines=[0.80])
    index.add_file("noise", "a", cosines=[0.44])

    assert _top()[0] == ["good"]
    assert retrieval._FEED_MIN_SCORE == pytest.approx(0.45)


def test_near_identical_is_judged_on_the_reduced_score(index):
    """Per chunk it would only skip one row of a duplicated document.

    A duplicate whose first chunk matches at 1.0 and whose second
    matches at 0.98 must not come back on the strength of the second.
    """
    index.add_file(
        "dupe", "a", cosines=[1.0, 0.98], channel="text_content",
    )
    index.add_file("real", "a", cosines=[0.80], channel="text_content")

    assert _top(channel="text_content")[0] == ["real"]


# ---------------------------------------------------------------------------
# The exclusion set
# ---------------------------------------------------------------------------


def test_watched_files_are_excluded(index):
    index.add_file("seen", "a", cosines=[0.95])
    index.add_file("unseen", "a", cosines=[0.70])

    # Control: without the exclusion both are candidates, and the
    # watched one ranks first.
    assert _top()[0] == ["seen", "unseen"]

    assert _top(exclude={"seen"})[0] == ["unseen"]


def test_an_exclusion_set_larger_than_the_profile_cap_is_honoured(index):
    """The exclusion set is the whole history and is never truncated.

    The profile caps its vector load; sharing that one bounded fetch
    would start recommending watched files to anyone past the cap.
    """
    index.add_many("seen", 2500, "a", cos_start=0.99, step=-0.0001)
    watched = {f"seen{i}" for i in range(2500)}
    index.add_file("unseen", "a", cosines=[0.50])

    assert _top(exclude=watched, limit=5)[0] == ["unseen"]


# ---------------------------------------------------------------------------
# Channel isolation
# ---------------------------------------------------------------------------


def test_another_embedding_type_in_the_same_table_is_not_scored(index):
    """``vec_text`` holds several types and a vector query knows none.

    Measured on the real index, ``whisper`` is 65.7% of that table.
    """
    index.add_file("kw", "a", cosines=[0.70], channel="tfidf_keywords")
    index.add_file("chunk", "a", cosines=[0.95], channel="text_content")

    assert _top(channel="text_content")[0] == ["chunk"]
    assert _top(channel="tfidf_keywords")[0] == ["kw"]


def test_clip_and_text_channels_use_their_own_tables(index):
    index.add_file("visual", "a", cosines=[0.80], channel="clip_thumbnail")
    index.add_file("textual", "a", cosines=[0.80], channel="tfidf_keywords")

    assert _top(channel="clip_thumbnail")[0] == ["visual"]
    assert _top(channel="tfidf_keywords")[0] == ["textual"]


# ---------------------------------------------------------------------------
# Many lanes at once
# ---------------------------------------------------------------------------


def test_every_lane_is_scored_independently(index):
    other = np.zeros(DIM, dtype=np.float32)
    other[2] = 1.0
    index.add_file("near_q", "a", cosines=[0.90])
    index.add_file(
        "near_other", "a",
        vectors=[_at_cosine(0.90, axis=2, off_axis=3)],
    )

    lanes = _top(centroids=[_query(), other])

    assert lanes[0] == ["near_q"]
    assert lanes[1] == ["near_other"]


def test_no_lanes_returns_no_results(index):
    index.add_file("a1", "a", cosines=[0.9])

    assert _top(centroids=[]) == []


# ---------------------------------------------------------------------------
# Empty, malformed and boundary inputs
# ---------------------------------------------------------------------------


def test_an_empty_channel_returns_nothing(index):
    index.add_file("only_visual", "a", cosines=[0.9], channel="clip_thumbnail")

    assert _top(channel="tfidf_keywords")[0] == []


def test_an_empty_index_returns_nothing(index):
    assert _top()[0] == []
    assert len(retrieval.load_candidates(drive="a", channel="text_content")) == 0


def test_results_are_ordered_by_descending_score(index):
    index.add_file("mid", "a", cosines=[0.70])
    index.add_file("best", "a", cosines=[0.95])
    index.add_file("worst", "a", cosines=[0.50])

    candidates = retrieval.load_candidates(drive="a", channel="clip_thumbnail")
    lane = retrieval.score_lanes(
        candidates, [_query()], exclude_file_ids=set(), limit=10,
    )[0]
    scores = [s for _, s in lane]

    assert scores == sorted(scores, reverse=True)


def test_limit_is_respected(index):
    index.add_many("a", 20, "a", cos_start=0.90, step=-0.001)

    assert len(_top(limit=7)[0]) == 7


def test_drive_and_channel_must_be_given(index):
    """A missed call site is a TypeError, not a whole-library read."""
    with pytest.raises(TypeError):
        retrieval.load_candidates(channel="clip_thumbnail")
    with pytest.raises(TypeError):
        retrieval.load_candidates(drive="a")


def test_an_unknown_channel_is_rejected(index):
    with pytest.raises(ValueError):
        retrieval.load_candidates(drive="a", channel="not_a_channel")


def test_a_row_of_the_wrong_width_is_reported_not_reshaped(index):
    """sqlite-vec validated width on the way in; reading blobs directly
    means validating it on the way out, or the failure surfaces later
    with nothing pointing at the bad row."""
    index.add_file("fine", "a", cosines=[0.9])
    with index.engine.begin() as conn:
        conn.execute(
            text("INSERT INTO vec_clip VALUES ('e_bad_clip_thumbnail_0', :v)"),
            {"v": np.ones(DIM + 3, dtype=np.float32).tobytes()},
        )
    session = index._maker()
    session.add(Embedding(
        id="e_bad_clip_thumbnail_0", file_id="bad",
        embedding_type="clip_thumbnail", vector_table="vec_clip",
        content_preview="x",
    ))
    session.add(IndexedFile(
        file_id="bad", drive="a", filename="bad.mp4", file_path="/bad.mp4",
        file_type="video", mime_type="video/mp4", file_size=1, active=True,
    ))
    session.commit()
    session.close()

    with pytest.raises(ValueError, match="width"):
        retrieval.load_candidates(drive="a", channel="clip_thumbnail")


def test_a_centroid_of_the_wrong_width_is_rejected(index):
    index.add_file("a1", "a", cosines=[0.9])
    candidates = retrieval.load_candidates(drive="a", channel="clip_thumbnail")

    with pytest.raises(ValueError, match="wide"):
        retrieval.score_lanes(
            candidates, [np.ones(DIM + 2, dtype=np.float32)],
            exclude_file_ids=set(), limit=5,
        )


def test_every_row_of_a_large_drive_is_loaded(index):
    """A drive holds far more rows than a neighbour search ever fetched."""
    total = 1007
    index.add_many("a", total, "a", cos_start=0.90, step=0.0)

    assert len(retrieval.load_candidates(drive="a", channel="clip_thumbnail")) \
        == total
