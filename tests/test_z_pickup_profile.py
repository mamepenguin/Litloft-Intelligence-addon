"""The taste profile behind the Pickup feed.

The old Pickup asked "what resembles the last five files I watched",
which meant a binge decided everything that followed it. The profile
here is built from the viewer's *whole* history: recency is a weight on
an interest, never a filter on which interests exist.

Two properties carry that intent and are easy to lose:

- clustering sees every entry, so an interest from years ago still
  exists as a cluster;
- weights are normalised onto a bounded range, so that cluster still
  gets turns. Exponential decay alone does not make an old interest
  quiet — past about eighteen months it deletes it.
"""

from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.models import Base, Embedding, IndexedFile  # noqa: E402
from app.pickup import profile  # noqa: E402

DIM = 8


def _unit(*components: float) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for i, c in enumerate(components):
        v[i] = c
    norm = np.linalg.norm(v)
    return v / norm if norm else v


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


#: Phase 3 owns the real constant; the weights only matter here in terms
#: of how much of a feed of this size a lane ends up filling.
_FEED_DEPTH = 300


def _feed_share(lane, lanes) -> float:
    """Fraction of the interleave this lane takes at any prefix length.

    ``key(c, j) = j / weight(c)`` gives each lane a share of the output
    proportional to its weight, at every depth.
    """
    return lane.weight / sum(other.weight for other in lanes)


class _Library:
    """A Litloft DB with watch history, plus a search index with vectors."""

    def __init__(self, tmp_path):
        self.litloft_engine = create_engine(
            f"sqlite:///{tmp_path / 'litloft.db'}",
            connect_args={"check_same_thread": False},
        )
        self._litloft_maker = sessionmaker(
            bind=self.litloft_engine, expire_on_commit=False
        )
        with self.litloft_engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE files (id TEXT PRIMARY KEY, drive TEXT NOT NULL, "
                "deleted_at TEXT, missing_since TEXT)"
            ))
            conn.execute(text(
                "CREATE TABLE watch_history (viewer_id TEXT NOT NULL, "
                "file_id TEXT NOT NULL, last_played_at TEXT NOT NULL)"
            ))

        self.engine = create_engine(
            f"sqlite:///{tmp_path / 'search.db'}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self._maker = sessionmaker(bind=self.engine, expire_on_commit=False)

    def watch(
        self,
        file_id: str,
        *,
        viewer: str = "v1",
        drive: str = "a",
        age_days: float = 0.0,
        deleted: bool = False,
    ):
        played = _now() - timedelta(days=age_days)
        with self.litloft_engine.begin() as conn:
            conn.execute(
                text("INSERT OR IGNORE INTO files VALUES (:i, :d, :del, NULL)"),
                {"i": file_id, "d": drive, "del": "2026-01-01" if deleted else None},
            )
            conn.execute(
                text("INSERT INTO watch_history VALUES (:v, :f, :t)"),
                {"v": viewer, "f": file_id, "t": played.isoformat(sep=" ")},
            )

    def index(
        self,
        file_id: str,
        vectors: list[np.ndarray],
        *,
        channel: str = "clip_thumbnail",
        drive: str = "a",
    ):
        session = self._maker()
        try:
            if not session.get(IndexedFile, file_id):
                session.add(IndexedFile(
                    file_id=file_id, drive=drive, filename=f"{file_id}.mp4",
                    file_path=f"/{file_id}.mp4", file_type="video",
                    mime_type="video/mp4", file_size=1, active=True,
                ))
            for i, vec in enumerate(vectors):
                session.add(Embedding(
                    id=f"e_{file_id}_{channel}_{i}", file_id=file_id,
                    embedding_type=channel,
                    vector_table=(
                        "vec_clip" if channel == "clip_thumbnail" else "vec_text"
                    ),
                    content_preview="x",
                ))
            session.commit()
        finally:
            session.close()
        self._vectors.setdefault(channel, {})[file_id] = vectors

    _vectors: dict

    @contextmanager
    def litloft_db(self):
        session = self._litloft_maker()
        try:
            yield session
        finally:
            session.close()

    @contextmanager
    def search_db(self):
        session = self._maker()
        try:
            yield session
        finally:
            session.close()


@pytest.fixture()
def lib(monkeypatch, tmp_path):
    library = _Library(tmp_path)
    library._vectors = {}
    monkeypatch.setattr(profile, "get_litloft_db", library.litloft_db)
    monkeypatch.setattr(profile, "get_search_db_read", library.search_db)
    # The vector blobs live in sqlite-vec virtual tables, which are not
    # available here. Serve them from what the fixture recorded.
    def _load(embedding_ids, table):
        out = {}
        for channel, files in library._vectors.items():
            for file_id, vectors in files.items():
                for i, vec in enumerate(vectors):
                    key = f"e_{file_id}_{channel}_{i}"
                    if key in embedding_ids:
                        out[key] = vec
        return out
    monkeypatch.setattr(profile, "_load_vectors", _load)
    return library


# ---------------------------------------------------------------------------
# Reading the history: two reads, not one
# ---------------------------------------------------------------------------


def test_a_quiet_viewer_is_not_starved_by_a_busy_one(lib):
    """The regression test for taking LIMIT before grouping by viewer.

    The old worker read the drive's 50 most recent rows and *then* split
    them by viewer, so whoever had been active most recently consumed
    the whole budget and everyone else got a handful or nothing.
    """
    for i in range(200):
        lib.watch(f"loud{i}", viewer="loud", age_days=i * 0.001)
    for i in range(5):
        lib.watch(f"quiet{i}", viewer="quiet", age_days=30 + i)

    got = profile.profile_history("a", "quiet")

    assert {w.file_id for w in got} == {f"quiet{i}" for i in range(5)}


def test_the_exclusion_set_is_not_capped(lib):
    """Two reads with different jobs, and only one of them has a cap.

    Sharing one bounded fetch would start recommending watched files to
    any viewer past the cap — the feed's central promise, broken by an
    optimisation belonging to the other read.
    """
    total = profile.PROFILE_VECTOR_CAP + 300
    for i in range(total):
        lib.watch(f"f{i}", age_days=i * 0.01)

    assert len(profile.watched_file_ids("a", "v1")) == total
    assert len(profile.profile_history("a", "v1")) == profile.PROFILE_VECTOR_CAP


def test_the_profile_keeps_the_most_recent_entries(lib):
    for i in range(10):
        lib.watch(f"f{i}", age_days=i)

    got = profile.profile_history("a", "v1", cap=3)

    assert [w.file_id for w in got] == ["f0", "f1", "f2"]


def test_another_drive_is_not_read(lib):
    lib.watch("mine", drive="a")
    lib.watch("theirs", drive="b")

    assert {w.file_id for w in profile.profile_history("a", "v1")} == {"mine"}
    assert profile.watched_file_ids("a", "v1") == {"mine"}


def test_deleted_files_stay_in_the_exclusion_set(lib):
    """A file in the trash is still a file this viewer has opened."""
    lib.watch("trashed", deleted=True)

    assert profile.watched_file_ids("a", "v1") == {"trashed"}
    assert profile.profile_history("a", "v1") == []


def test_viewers_are_listed_per_drive(lib):
    lib.watch("f1", viewer="v1", drive="a")
    lib.watch("f2", viewer="v2", drive="a")
    lib.watch("f3", viewer="v3", drive="b")

    assert sorted(profile.viewer_ids("a")) == ["v1", "v2"]


# ---------------------------------------------------------------------------
# Representative vectors
# ---------------------------------------------------------------------------


def test_a_files_chunks_are_averaged_into_one_vector(lib):
    lib.index("doc", [_unit(1, 0), _unit(0, 1)], channel="text_content")

    got = profile.representative_vectors(["doc"], "text_content")

    assert set(got) == {"doc"}
    assert np.allclose(got["doc"], _unit(1, 1))


def test_a_file_without_the_channel_contributes_nothing(lib):
    lib.index("visual", [_unit(1, 0)], channel="clip_thumbnail")

    assert profile.representative_vectors(["visual"], "text_content") == {}


def test_metadata_is_not_a_profile_channel(lib):
    """A filename embedding names nothing for "IMG_1234.jpg" or a UUID.

    ``find_similar`` refuses it as a second opinion for that reason, and
    a feed has no user in the loop to notice when it is wrong.
    """
    assert "metadata" not in profile.CHANNELS


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected"),
    [(0, 1), (1, 1), (11, 1), (12, 2), (18, 3), (50, 5), (200, 8), (2000, 8)],
)
def test_cluster_count_is_bounded_and_continuous(n, expected):
    """K steps 1 -> 2 at the threshold, never 1 -> 3.

    Three clusters over twelve files is noise, and the discontinuity
    would make the feed's shape jump on the twelfth watched file.
    """
    assert profile.choose_k(n) == expected


def test_separated_interests_become_separate_clusters(lib):
    for i in range(8):
        lib.watch(f"cats{i}", age_days=i)
        lib.index(f"cats{i}", [_unit(1, 0, 0)])
    for i in range(8):
        lib.watch(f"trains{i}", age_days=i)
        lib.index(f"trains{i}", [_unit(0, 1, 0)])

    lanes = profile.build_lanes(profile.profile_history("a", "v1"), key="a:v1")

    assert len(lanes) == 2
    assert sorted(lane.member_count for lane in lanes) == [8, 8]


def test_clustering_is_deterministic(lib):
    for i in range(30):
        lib.watch(f"f{i}", age_days=i)
        lib.index(f"f{i}", [_unit(1, i % 5, (i % 3) * 0.5)])

    history = profile.profile_history("a", "v1")
    first = profile.build_lanes(history, key="a:v1")
    second = profile.build_lanes(history, key="a:v1")

    assert [(l.cluster_id, l.member_count) for l in first] == \
           [(l.cluster_id, l.member_count) for l in second]


def test_a_lone_stray_file_does_not_get_its_own_lane(lib):
    """Every lane is guaranteed a share of the feed, so a cluster of one
    would buy one stray file's neighbourhood real airtime."""
    for i in range(15):
        lib.watch(f"main{i}", age_days=i)
        lib.index(f"main{i}", [_unit(1, 0.01 * i, 0)])
    lib.watch("stray", age_days=1)
    lib.index("stray", [_unit(0, 0, 1)])

    lanes = profile.build_lanes(profile.profile_history("a", "v1"), key="a:v1")

    assert all(lane.member_count > 1 for lane in lanes)
    assert sum(lane.member_count for lane in lanes) == 16


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def test_an_old_interest_keeps_a_usable_share_of_the_turns(lib):
    """The reason the raw weight is not the interleave weight.

    Exponential decay past about eighteen months does not make a cluster
    quiet, it removes it: the raw weights differ by four orders of
    magnitude, and ``key = j / weight`` would place the old lane's first
    item thousands of positions down a feed that holds a few hundred.
    """
    for i in range(10):
        lib.watch(f"new{i}", age_days=i * 0.1)
        lib.index(f"new{i}", [_unit(1, 0, 0)])
    for i in range(10):
        lib.watch(f"old{i}", age_days=350 + i * 0.1)
        lib.index(f"old{i}", [_unit(0, 1, 0)])

    lanes = profile.build_lanes(profile.profile_history("a", "v1"), key="a:v1")
    by_weight = sorted(lanes, key=lambda l: l.weight)
    quiet, loud = by_weight[0], by_weight[-1]

    # The raw weights are what would have been used, and at the far edge
    # of the window they are already an order of magnitude down.
    assert quiet.raw_weight / loud.raw_weight < 0.1

    # Asserting the normalised ratio against W_MIN would prove nothing:
    # _normalise maps the minimum onto W_MIN by construction, so that
    # comparison is a constant against itself and holds for any value of
    # it. What has to hold is the consequence — that the old interest
    # still fills a usable part of the feed.
    assert _feed_share(quiet, lanes) >= 0.15


def test_no_lane_falls_below_the_floor(lib):
    for group, age in (("a", 0.0), ("b", 120.0), ("c", 240.0), ("d", 355.0)):
        for i in range(6):
            lib.watch(f"{group}{i}", age_days=age + i)
            lib.index(f"{group}{i}", [_unit(*(1 if x == ord(group) % 7 else 0
                                              for x in range(DIM)))])

    lanes = profile.build_lanes(profile.profile_history("a", "v1"), key="a:v1")
    quietest = min(lanes, key=lambda l: l.weight)

    assert len(lanes) >= 3
    # Again the requirement, not the definition: whatever the spread of
    # ages, the quietest lane still places a usable number of items in a
    # feed of _FEED_DEPTH.
    assert _feed_share(quietest, lanes) * _FEED_DEPTH >= 25


def test_a_binge_does_not_outweigh_a_small_interest_linearly():
    """Forty episodes must not count as eight times a five-file interest.

    Stated on the weight itself rather than through a clustering run:
    how k-means happens to split a binge is a separate question, and one
    the profile deliberately does not try to answer.
    """
    binge = profile.raw_weight([0.0] * 40)
    small = profile.raw_weight([0.0] * 5)

    assert binge / small < (40 / 5) / 2


def test_weight_still_rises_with_mass():
    assert profile.raw_weight([0.0] * 40) > profile.raw_weight([0.0] * 5)


def test_a_single_lane_gets_the_full_weight(lib):
    for i in range(5):
        lib.watch(f"f{i}", age_days=i)
        lib.index(f"f{i}", [_unit(1, 0, 0)])

    lanes = profile.build_lanes(profile.profile_history("a", "v1"), key="a:v1")

    assert len(lanes) == 1
    assert lanes[0].weight == pytest.approx(1.0)


def test_decay_is_measured_from_the_half_life(lib):
    one = profile.decay(0.0)
    half = profile.decay(profile.HALF_LIFE_DAYS)
    quarter = profile.decay(profile.HALF_LIFE_DAYS * 2)

    assert one == pytest.approx(1.0)
    assert half == pytest.approx(0.5)
    assert quarter == pytest.approx(0.25)


def test_weights_are_normalised_across_channels_together(lib):
    """One interleave means one scale.

    Normalising per channel would let a lane's share depend on which
    other lanes happened to share its embedding type.
    """
    for i in range(6):
        lib.watch(f"vis{i}", age_days=i)
        lib.index(f"vis{i}", [_unit(1, 0, 0)], channel="clip_thumbnail")
    for i in range(6):
        lib.watch(f"txt{i}", age_days=300 + i)
        lib.index(f"txt{i}", [_unit(0, 1, 0)], channel="tfidf_keywords")

    lanes = profile.build_lanes(profile.profile_history("a", "v1"), key="a:v1")
    channels = {lane.channel for lane in lanes}

    assert channels == {"clip_thumbnail", "tfidf_keywords"}
    assert min(l.weight for l in lanes) >= profile.W_MIN - 1e-9
    # Exactly one lane sits at the top of the shared scale.
    assert sum(1 for l in lanes if l.weight == pytest.approx(1.0)) == 1


def test_an_empty_history_produces_no_lanes(lib):
    assert profile.build_lanes([], key="a:v1") == []


def test_unparseable_timestamps_are_skipped_not_treated_as_now(lib):
    """Fabricating a recency would promote a broken row above real ones."""
    lib.watch("good", age_days=1)
    with lib.litloft_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO files VALUES ('bad', 'a', NULL, NULL)"
        ))
        conn.execute(text(
            "INSERT INTO watch_history VALUES ('v1', 'bad', 'not-a-date')"
        ))

    got = profile.profile_history("a", "v1")

    assert [w.file_id for w in got] == ["good"]


def test_an_aware_timestamp_is_read_as_utc(lib):
    """Legacy rows can carry a ``+00:00`` suffix; the column is naive."""
    with lib.litloft_engine.begin() as conn:
        conn.execute(text("INSERT INTO files VALUES ('f', 'a', NULL, NULL)"))
        conn.execute(text(
            "INSERT INTO watch_history VALUES ('v1', 'f', :t)"
        ), {"t": datetime.now(UTC).isoformat()})

    got = profile.profile_history("a", "v1")

    assert len(got) == 1
    assert got[0].last_played_at.tzinfo is None
    assert profile.decay(profile.age_days(got[0].last_played_at)) > 0.99


def test_raw_weight_uses_log_compression():
    assert profile.raw_weight([0.0] * 40) == pytest.approx(math.log1p(40.0))


# ---------------------------------------------------------------------------
# Loading vectors
# ---------------------------------------------------------------------------


def test_every_requested_vector_is_loaded(monkeypatch, tmp_path):
    """One statement per id, and none of them dropped.

    Batching this looks like the obvious optimisation and is the
    opposite: sqlite-vec scans the virtual table for every ``IN``, so
    grouping ids multiplies scans. Measured numbers are in
    ``app.pickup.retrieval``.
    """
    engine = create_engine(
        f"sqlite:///{tmp_path / 'vec.db'}",
        connect_args={"check_same_thread": False},
    )
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE vec_text (embedding_id TEXT PRIMARY KEY, vector BLOB)"
        ))
    total = 1007
    with engine.begin() as conn:
        for i in range(total):
            conn.execute(
                text("INSERT INTO vec_text VALUES (:i, :v)"),
                {"i": f"e{i}", "v": _unit(1, i % 3).tobytes()},
            )
    monkeypatch.setattr(
        "app.database.get_search_engine", lambda: engine, raising=False
    )

    got = profile._load_vectors([f"e{i}" for i in range(total)], "vec_text")

    assert len(got) == total
    assert np.allclose(got["e2"], _unit(1, 2))


def test_loading_no_vectors_touches_nothing():
    assert profile._load_vectors([], "vec_text") == {}


# ---------------------------------------------------------------------------
# The window, and what it is honest about
# ---------------------------------------------------------------------------


def test_history_outside_the_window_is_not_profiled(lib):
    lib.watch("inside", age_days=profile.PROFILE_WINDOW_DAYS - 1)
    lib.watch("outside", age_days=profile.PROFILE_WINDOW_DAYS + 1)

    assert [w.file_id for w in profile.profile_history("a", "v1")] == ["inside"]


def test_the_window_does_not_reach_the_exclusion_set(lib):
    """The two reads have different jobs and different bounds.

    A file watched two years ago is outside the profile but is still a
    file this viewer has opened, and recommending it back would be the
    feed breaking its one promise.
    """
    lib.watch("ancient", age_days=profile.PROFILE_WINDOW_DAYS * 2)

    assert profile.profile_history("a", "v1") == []
    assert profile.watched_file_ids("a", "v1") == {"ancient"}


def test_a_future_timestamp_is_dropped_not_treated_as_newest(lib):
    """Clock skew must not promote a row above every genuine one."""
    lib.watch("real", age_days=1)
    lib.watch("skewed", age_days=-30)

    assert [w.file_id for w in profile.profile_history("a", "v1")] == ["real"]


def test_slight_skew_inside_tolerance_is_kept(lib):
    lib.watch("barely", age_days=-0.5)

    assert [w.file_id for w in profile.profile_history("a", "v1")] == ["barely"]


# ---------------------------------------------------------------------------
# Weights: a floor, not a stretch
# ---------------------------------------------------------------------------


def test_near_ties_stay_near_ties():
    """Min-max normalisation turned a one percent spread into 4 : 2.5 : 1.

    Three interests watched in the same week are equally interesting,
    and the weighting has to be able to say so.
    """
    weights = profile._normalise([2.40, 2.39, 2.38])

    assert weights[0] == pytest.approx(1.0)
    assert min(weights) > 0.98


def test_the_floor_still_holds_for_a_genuine_gap():
    weights = profile._normalise([2.40, 0.001])

    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(profile.W_MIN)


def test_equal_weights_stay_equal():
    assert profile._normalise([1.5, 1.5, 1.5]) == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# A centroid with no direction is not a lane
# ---------------------------------------------------------------------------


def test_a_cluster_that_cancels_out_is_not_given_a_lane(lib):
    """Its mean has no direction, so it cannot be a query.

    Sent to the retrieval side it would sit equidistant from everything
    and score a flat 0.5 — above the floor — filling its lane with files
    related to nothing.
    """
    assert profile._centroid(np.stack([_unit(1, 0), -_unit(1, 0)])) is None


def test_a_normal_cluster_still_has_a_direction():
    got = profile._centroid(np.stack([_unit(1, 0), _unit(1, 0.1)]))

    assert got is not None
    assert np.linalg.norm(got) == pytest.approx(1.0, abs=1e-6)


def test_chunks_are_reduced_per_file_not_held_all_at_once(lib, monkeypatch):
    """The cap is on files; ``text_content`` averages 53 chunks each.

    Holding every chunk would keep six figures of vectors resident at
    the supported profile size, beside the drive-wide candidate
    matrices the worker already carries.
    """
    for i in range(6):
        lib.watch(f"doc{i}")
        lib.index(f"doc{i}", [_unit(1, 0), _unit(0, 1)], channel="text_content")

    live = []
    original = profile._load_vectors

    def counting(embedding_ids, table):
        live.append(len(embedding_ids))
        return original(embedding_ids, table)

    monkeypatch.setattr(profile, "_load_vectors", counting)
    got = profile.representative_vectors(
        [f"doc{i}" for i in range(6)], "text_content",
    )

    assert len(got) == 6
    # One file's chunks per call, never all eighteen at once.
    assert max(live) == 2
    assert len(live) == 6


def test_the_reduced_vector_is_still_the_normalised_mean(lib):
    lib.watch("doc")
    lib.index("doc", [_unit(1, 0), _unit(0, 1)], channel="text_content")

    got = profile.representative_vectors(["doc"], "text_content")

    assert np.allclose(got["doc"], _unit(1, 1))
