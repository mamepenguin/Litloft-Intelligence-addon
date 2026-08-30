"""Pickup's candidate generation, held to the guarantees the feed needs.

The feed asks a different question from the file-detail "similar files"
section, and the two need opposite answers when the evidence is weak.
The section is right to say nothing rather than something thin; the feed
exists to produce quantity, and a lane that goes silent is a lane that
contributes nothing to the interleave.

So this path does not reuse ``find_similar``. What it must guarantee:

- a drive is a security boundary, and the restriction is applied
  *before* any scoring decision, never after one,
- a neighbourhood whose scores are all alike still yields candidates,
- files the viewer has already opened never appear, no matter how many
  of them there are,
- one embedding type's vectors are never scored as another's.
"""

from __future__ import annotations

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

DIM = 4


def _vec() -> np.ndarray:
    v = np.ones(DIM, dtype=np.float32)
    return v / np.linalg.norm(v)


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
                    f"CREATE TABLE {table} (embedding_id TEXT PRIMARY KEY, "
                    "vector BLOB, distance REAL)"
                ))

    def add_file(
        self,
        file_id: str,
        drive: str,
        *,
        distance: float | None = None,
        channel: str = "clip_thumbnail",
        active: bool = True,
        file_type: str = "video",
    ):
        """Register a file and give it one embedding at ``distance``."""
        session = self._maker()
        try:
            session.add(IndexedFile(
                file_id=file_id, drive=drive, filename=f"{file_id}.mp4",
                file_path=f"/drives/{drive}/{file_id}.mp4",
                file_type=file_type, mime_type="video/mp4",
                file_size=1, active=active,
            ))
            if distance is not None:
                session.add(Embedding(
                    id=f"e_{file_id}", file_id=file_id,
                    embedding_type=channel,
                    vector_table=retrieval.vector_table_for(channel),
                    content_preview="x",
                ))
            session.commit()
        finally:
            session.close()

        if distance is None:
            return
        table = retrieval.vector_table_for(channel)
        with self.engine.begin() as conn:
            conn.execute(
                text(f"INSERT INTO {table} VALUES (:i, :v, :d)"),
                {"i": f"e_{file_id}", "v": _vec().tobytes(), "d": distance},
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
    # sqlite-vec's MATCH is not available here; order by the stored
    # distance column instead, which is what MATCH would have produced.
    monkeypatch.setattr(
        retrieval, "sql_text",
        lambda q: text(q.replace("WHERE vector MATCH :vec", "WHERE :vec IS NOT NULL")),
    )
    return idx


def _neighbours(drive="a", channel="clip_thumbnail", exclude=(), limit=10):
    return retrieval.centroid_neighbours(
        _vec(),
        channel=channel,
        drive=drive,
        exclude_file_ids=set(exclude),
        limit=limit,
    )


# ---------------------------------------------------------------------------
# The drive boundary
# ---------------------------------------------------------------------------


def test_a_nearer_neighbour_in_another_drive_does_not_displace_this_drive(index):
    """The regression test for scoring before scoping.

    ``find_similar`` ranks and prunes across the whole index and only
    then cuts the survivors down to the requested drive, so a file whose
    nearest neighbours live elsewhere can come back empty while perfectly
    good in-drive candidates sat two ranks lower. Here the other drive's
    files are strictly closer, and must change nothing.
    """
    for i in range(5):
        index.add_file(f"b{i}", "b", distance=0.01 + i * 0.001)
    index.add_file("a1", "a", distance=0.5)
    index.add_file("a2", "a", distance=0.6)

    got = [file_id for file_id, _ in _neighbours(drive="a")]

    assert got == ["a1", "a2"]


def test_fetch_widens_until_in_drive_neighbours_survive(index):
    """A drive in the minority still gets its neighbours found."""
    for i in range(300):
        index.add_file(f"b{i}", "b", distance=0.01 + i * 0.0001)
    index.add_file("a1", "a", distance=0.9)

    assert [f for f, _ in _neighbours(drive="a", limit=1)] == ["a1"]


def test_inactive_files_are_not_candidates(index):
    index.add_file("gone", "a", distance=0.1, active=False)
    index.add_file("here", "a", distance=0.5)

    assert [f for f, _ in _neighbours()] == ["here"]


# ---------------------------------------------------------------------------
# Weak evidence must still produce candidates
# ---------------------------------------------------------------------------


def test_a_flat_neighbourhood_still_yields_candidates(index):
    """Scores alike to three decimal places are normal for a cluster.

    ``find_similar`` reads that shape as "this embedding cannot tell
    these files apart" and discards every candidate — twice, once by the
    gap check and once by the coefficient of variation. A cluster
    centroid sits among files that genuinely resemble each other, so for
    the feed that shape is the expected one, not a warning.
    """
    for i in range(10):
        index.add_file(f"a{i}", "a", distance=0.700 + i * 0.0001)

    assert len(_neighbours(limit=10)) == 10


def test_candidates_below_the_top_score_are_kept(index):
    """No margin cutoff.

    ``find_similar`` keeps only what scores within 0.05 of the best
    match, which is the single largest reason the old Pickup could not
    fill twelve slots.
    """
    index.add_file("close", "a", distance=0.1)
    index.add_file("middling", "a", distance=0.8)
    index.add_file("distant", "a", distance=1.0)

    got = [f for f, _ in _neighbours()]

    assert got == ["close", "middling", "distant"]


def test_candidates_below_the_floor_are_dropped(index):
    """One low floor survives, to keep the tail from going arbitrary."""
    index.add_file("good", "a", distance=0.2)
    index.add_file("noise", "a", distance=1.9)

    assert [f for f, _ in _neighbours()] == ["good"]


def test_near_identical_matches_are_dropped(index):
    """A duplicate or trivial embedding matches everything."""
    index.add_file("dupe", "a", distance=0.0)
    index.add_file("real", "a", distance=0.5)

    assert [f for f, _ in _neighbours()] == ["real"]


# ---------------------------------------------------------------------------
# The exclusion set
# ---------------------------------------------------------------------------


def test_watched_files_are_excluded(index):
    index.add_file("seen", "a", distance=0.1)
    index.add_file("unseen", "a", distance=0.5)

    assert [f for f, _ in _neighbours(exclude={"seen"})] == ["unseen"]


def test_an_exclusion_set_larger_than_the_profile_cap_is_honoured(index):
    """The exclusion set is the whole history and is never truncated.

    The profile caps its vector load at 2000 files; sharing that one
    bounded fetch would start recommending watched files to anyone past
    the cap. Well beyond it here, and every one of them must stay out.
    """
    watched = {f"seen{i}" for i in range(2500)}
    for i, file_id in enumerate(sorted(watched)):
        index.add_file(file_id, "a", distance=0.01 + i * 0.0001)
    index.add_file("unseen", "a", distance=0.9)

    assert [f for f, _ in _neighbours(exclude=watched, limit=5)] == ["unseen"]


# ---------------------------------------------------------------------------
# Channel isolation
# ---------------------------------------------------------------------------


def test_another_embedding_type_in_the_same_table_is_not_scored(index):
    """``vec_text`` holds several types and MATCH knows none of them.

    Measured on a real index, a ``tfidf_keywords`` query came back 85%
    ``whisper`` rows, each scored as though it were the type asked for.
    """
    index.add_file("kw", "a", distance=0.5, channel="tfidf_keywords")
    index.add_file("chunk", "a", distance=0.1, channel="text_content")

    got = [f for f, _ in _neighbours(channel="tfidf_keywords")]

    assert got == ["kw"]


def test_clip_and_text_channels_use_their_own_tables(index):
    index.add_file("visual", "a", distance=0.2, channel="clip_thumbnail")
    index.add_file("textual", "a", distance=0.2, channel="tfidf_keywords")

    assert [f for f, _ in _neighbours(channel="clip_thumbnail")] == ["visual"]
    assert [f for f, _ in _neighbours(channel="tfidf_keywords")] == ["textual"]


# ---------------------------------------------------------------------------
# Empty and malformed paths
# ---------------------------------------------------------------------------


def test_an_empty_channel_returns_nothing(index):
    index.add_file("only_visual", "a", distance=0.2, channel="clip_thumbnail")

    assert _neighbours(channel="tfidf_keywords") == []


def test_an_empty_index_returns_nothing(index):
    assert _neighbours() == []


def test_results_are_ordered_by_descending_score(index):
    index.add_file("mid", "a", distance=0.6)
    index.add_file("best", "a", distance=0.2)
    index.add_file("worst", "a", distance=1.0)

    scores = [s for _, s in _neighbours()]

    assert scores == sorted(scores, reverse=True)


def test_limit_is_respected(index):
    for i in range(20):
        index.add_file(f"a{i}", "a", distance=0.3 + i * 0.001)

    assert len(_neighbours(limit=7)) == 7


def test_drive_must_be_given(index):
    """A missed call site is a TypeError, not a silent whole-library read."""
    with pytest.raises(TypeError):
        retrieval.centroid_neighbours(
            _vec(), channel="clip_thumbnail",
            exclude_file_ids=set(), limit=5,
        )


def test_an_unknown_channel_is_rejected(index):
    with pytest.raises(ValueError):
        _neighbours(channel="not_a_channel")


# ---------------------------------------------------------------------------
# The fetch schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base", [1, 10, 100, 256, 1000, 5000])
def test_the_fetch_schedule_always_reaches_the_ceiling(base):
    """Widening must not bottom out below the cap it declares.

    Multiplying a small base by the widening factors stops well short of
    ``_FETCH_MAX``, and the loop would then give up while the index
    still had rows to give. That matters more here than for a
    file-seeded neighbour search: the query vector is built from the
    viewer's own history and every one of those files is excluded, so
    pages of unusable rows are the normal case.
    """
    schedule = retrieval._fetch_schedule(base)

    assert schedule == sorted(schedule)
    assert len(set(schedule)) == len(schedule)
    assert schedule[-1] == retrieval._FETCH_MAX
    assert all(size <= retrieval._FETCH_MAX for size in schedule)
