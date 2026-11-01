"""SPEC-ADDON-041: ``GET /search/scene-segments`` over HTTP, against a real search DB
with sqlite-vec loaded.

Frames are unit vectors, so a frame placed at score ``s`` is at L2 distance
``sqrt(2 (1 - s))`` from the query and scores ``s`` back.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import text

import app.database as database
import app.workers.embedder  # noqa: F401  (so the stand-in reaches every importer)
from app.models import Embedding, IndexedFile
from tests.search_endpoint_db import (  # noqa: F401
    DRIVE,
    OTHER,
    api,
    patch_everywhere,
    search_index,
    statements,
)
from tests.vec_partition_db import at_distance, unit

PATH = "/search/scene-segments"
SEGMENT_KEYS = {"file_id", "filename", "start", "end", "peak", "score"}
MANIFEST = Path(__file__).resolve().parents[1] / "manifest.json"


def _distance(score: float) -> float:
    return float(np.sqrt(2.0 * (1.0 - score)))


def _ok(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()


def _seg(body) -> list[tuple]:
    return [(s["file_id"], s["start"], s["end"], s["peak"]) for s in body["segments"]]


def _ids(body) -> list[str]:
    return [s["file_id"] for s in body["segments"]]


class World:
    """Rows are buffered and written together, so a test can build thousands."""

    def __init__(self, index):
        self.index = index
        self.dim = index.dim("vec_clip")
        self._files: dict[str, dict] = {}
        self._rows: list[tuple] = []
        self._n = 0

    def file(self, file_id, *, drive=DRIVE, active=True, duration=None):
        self._files[file_id] = {"drive": drive, "active": active, "duration": duration}

    def frame(self, file_id, t, score=None, *, etype="clip", drive=DRIVE, active=True):
        if file_id not in self._files:
            self.file(file_id, drive=drive, active=active)
        self._n += 1
        self._rows.append((f"e{self._n}", file_id, t, score, etype))

    def pad(self, count, prefix="pad", score=0.1):
        for i in range(count):
            self.frame(f"{prefix}{i}", 5.0, score)

    def commit(self):
        index = self.index
        with database.get_search_db() as s:
            s.add_all([
                IndexedFile(
                    file_id=f, drive=v["drive"], filename=f"{f}.mp4",
                    file_path=f"/drives/{v['drive']}/{f}.mp4", file_type="video",
                    mime_type="video/mp4", file_size=1, duration=v["duration"],
                    active=v["active"],
                )
                for f, v in self._files.items()
            ])
            s.add_all([
                Embedding(
                    id=e, file_id=f, embedding_type=etype, vector_table="vec_clip",
                    timestamp_start=t, content_preview="x",
                )
                for e, f, t, _, etype in self._rows
            ])
        vec = [
            {
                "i": e, "t": etype,
                "v": at_distance(self.dim, _distance(score), n + 1)
                .astype(np.float32).tobytes(),
            }
            for n, (e, _, _, score, etype) in enumerate(self._rows) if score is not None
        ]
        if vec:
            with index.engine.begin() as conn:
                conn.execute(
                    text("INSERT INTO vec_clip(embedding_id, embedding_type, vector) "
                         "VALUES (:i, :t, :v)"),
                    vec,
                )
        self._rows.clear()
        self._files.clear()


@pytest.fixture()
def world(search_index, monkeypatch):  # noqa: F811
    w = World(search_index)
    w.asked = []

    def _embed(*args, **kwargs):
        w.asked.append((args, kwargs))
        return unit(w.dim, 0)

    patch_everywhere(monkeypatch, "embed_text_clip", _embed)
    return w


def _get(api, world, params=None, drive=DRIVE):
    world.commit()
    return api(PATH, {"q": "cat"} if params is None else params, drive=drive)


class TestNormalFlow:
    def test_spec_addon_041_builds_segments_from_the_top_frames_with_start_end_peak_score(
        self, api, world,
    ):
        world.file("A", duration=100.0)
        world.frame("A", 10.0, 0.93)
        world.frame("A", 25.0, 0.20)
        world.frame("A", 40.0, 0.95)
        world.frame("A", 85.0, 0.90)
        world.frame("A", 130.0, None)
        world.frame("A", 176.0, 0.97)
        world.pad(28)

        body = _ok(_get(api, world))

        assert body == {
            "query": "cat",
            "segments": [
                {"file_id": "A", "filename": "A.mp4", "start": 176.0, "end": 206.0,
                 "peak": 176.0, "score": pytest.approx(0.97, abs=1e-4)},
                {"file_id": "A", "filename": "A.mp4", "start": 10.0, "end": 100.0,
                 "peak": 40.0, "score": pytest.approx(0.95, abs=1e-4)},
            ],
            "partial": False,
        }
        for s in body["segments"]:
            assert set(s) == SEGMENT_KEYS
            assert s["score"] == round(s["score"], 4)

    def test_spec_addon_041_q_is_stripped_echoed_and_embedded_stripped(self, api, world):
        world.frame("A", 10.0, 0.9)

        body = _ok(_get(api, world, {"q": "  a red car \n"}))

        assert body["query"] == "a red car"
        assert world.asked
        assert all(
            "a red car" in (list(a) + list(k.values())) for a, k in world.asked
        )

    def test_spec_addon_041_a_negative_cosine_still_returns_a_segment(self, api, world):
        world.frame("A", 10.0, -0.5)

        body = _ok(_get(api, world))

        assert _seg(body) == [("A", 10.0, 40.0, 10.0)]
        assert body["segments"][0]["score"] < 0


class TestThreshold:
    @pytest.mark.parametrize(
        "frames,expected",
        [
            (1, 1), (10, 1), (11, 2), (20, 2), (21, 3), (100, 10), (101, 11),
            (250, 25), (310, 30),
        ],
    )
    def test_spec_addon_041_keeps_about_the_top_percent_of_frames_across_at_most_30_files(
        self, api, world, frames, expected,
    ):
        for i in range(frames):
            world.frame(f"f{i:03d}", 10.0, 0.9 - 0.001 * i)

        body = _ok(_get(api, world, {"q": "cat", "limit": 50}))

        assert _ids(body) == [f"f{i:03d}" for i in range(expected)]

    def test_spec_addon_041_frames_tied_with_the_threshold_are_all_kept(self, api, world):
        world.frame("a", 0.0, 0.9)
        world.frame("b", 0.0, 0.9)
        world.frame("c", 0.0, 0.9)
        world.frame("d", 0.0, 0.5)

        body = _ok(_get(api, world))

        assert _ids(body) == ["a", "b", "c"]

    def test_spec_addon_041_only_frames_of_this_drive_active_with_a_timestamp_and_type_clip_count(
        self, api, world,
    ):
        for i in range(10):
            world.frame(f"own{i}", 10.0, 0.5 - 0.01 * i)
        for i in range(20):
            world.frame(f"other{i}", 10.0, 0.99, drive=OTHER)
            world.frame(f"gone{i}", 10.0, 0.98, active=False)
        for i in range(5):
            world.frame("own0", None, 0.97)
            world.frame(f"thumb{i}", 10.0, 0.96, etype="clip_thumbnail")

        body = _ok(_get(api, world))

        assert _ids(body) == ["own0"]
        assert _seg(body) == [("own0", 10.0, 40.0, 10.0)]

    def test_spec_addon_041_another_drive_sees_only_its_own_frames(self, api, world):
        world.frame("mine", 10.0, 0.9)
        world.frame("theirs", 20.0, 0.5, drive=OTHER)

        body = _ok(_get(api, world, drive=OTHER))

        assert _seg(body) == [("theirs", 20.0, 50.0, 20.0)]


class TestSegments:
    @pytest.mark.parametrize(
        "times,expected",
        [
            pytest.param([0.0, 45.0], [(0.0, 75.0, 0.0)], id="gap-45-joins"),
            pytest.param([0.0, 46.0], [(0.0, 30.0, 0.0), (46.0, 76.0, 46.0)], id="gap-46-splits"),
            pytest.param([0.0, 45.0, 90.0], [(0.0, 120.0, 0.0)], id="chain-joins-by-previous"),
        ],
    )
    def test_spec_addon_041_seeds_join_when_each_is_within_45_seconds_of_the_previous(
        self, api, world, times, expected,
    ):
        for t in times:
            world.frame("A", t, 0.9)

        body = _ok(_get(api, world))

        assert [(s["start"], s["end"], s["peak"]) for s in body["segments"]] == expected

    @pytest.mark.parametrize(
        "extras,duration,end",
        [
            pytest.param([], None, 130.0, id="last-plus-30"),
            pytest.param([], 105.0, 105.0, id="duration-above-last"),
            pytest.param([], 100.0, 130.0, id="duration-equal-ignored"),
            pytest.param([], 50.0, 130.0, id="duration-below-ignored"),
            pytest.param([(110.0, True)], None, 110.0, id="next-frame-with-vector"),
            pytest.param([(110.0, False)], None, 110.0, id="next-frame-without-vector"),
            pytest.param([(140.0, True)], None, 130.0, id="next-frame-beyond-30"),
            pytest.param([(140.0, True), (125.0, False)], None, 125.0, id="nearest-next-frame"),
            pytest.param([(100.0, False)], None, 130.0, id="same-time-is-not-later"),
            pytest.param([(90.0, True)], None, 130.0, id="earlier-frame-ignored"),
            pytest.param([(110.0, True)], 120.0, 110.0, id="next-frame-beats-duration"),
            pytest.param([(110.0, True)], 105.0, 105.0, id="duration-beats-next-frame"),
        ],
    )
    def test_spec_addon_041_end_is_the_smallest_of_next_frame_last_plus_30_and_duration(
        self, api, world, extras, duration, end,
    ):
        world.file("A", duration=duration)
        world.frame("A", 100.0, 0.9)
        for t, has_vector in extras:
            world.frame("A", t, 0.1 if has_vector else None)

        body = _ok(_get(api, world))

        assert _seg(body) == [("A", 100.0, end, 100.0)]

    def test_spec_addon_041_peak_is_the_earliest_frame_on_a_score_tie(self, api, world):
        world.frame("A", 10.0, 0.9)
        world.frame("A", 20.0, 0.9)

        body = _ok(_get(api, world))

        assert _seg(body) == [("A", 10.0, 50.0, 10.0)]

    def test_spec_addon_041_frames_without_a_vector_never_make_a_segment(self, api, world):
        for i in range(5):
            world.frame("A", 10.0 * i, None)

        body = _ok(_get(api, world))

        assert body == {"query": "cat", "segments": [], "partial": False}


class TestOrdering:
    def _layout(self, world):
        world.frame("A", 0.0, 0.99)
        world.frame("A", 100.0, 0.98)
        world.frame("A", 200.0, 0.97)
        world.frame("B", 0.0, 0.985)
        world.pad(30)

    def test_spec_addon_041_orders_by_score_and_a_file_contributes_at_most_two(
        self, api, world,
    ):
        self._layout(world)

        body = _ok(_get(api, world))

        assert [(s["file_id"], s["start"]) for s in body["segments"]] == [
            ("A", 0.0), ("B", 0.0), ("A", 100.0),
        ]

    @pytest.mark.parametrize("limit,count", [(1, 1), (2, 2), (3, 3), (50, 3)])
    def test_spec_addon_041_returns_at_most_limit_and_skipped_segments_do_not_use_it_up(
        self, api, world, limit, count,
    ):
        self._layout(world)

        body = _ok(_get(api, world, {"q": "cat", "limit": limit}))

        assert len(body["segments"]) == count

    def test_spec_addon_041_default_limit_is_20(self, api, world):
        for i in range(30):
            world.frame(f"f{i:02d}", 10.0, 0.9 - 0.001 * i)
        for i in range(300):
            world.frame(f"z{i}", 10.0, 0.1)

        assert len(_ok(_get(api, world))["segments"]) == 20

    def test_spec_addon_041_equal_scores_order_by_file_id_then_start(self, api, world):
        world.frame("b", 50.0, 0.9)
        world.frame("a", 100.0, 0.9)
        world.frame("a", 0.0, 0.9)

        body = _ok(_get(api, world))

        assert [(s["file_id"], s["start"]) for s in body["segments"]] == [
            ("a", 0.0), ("a", 100.0), ("b", 50.0),
        ]

    def test_spec_addon_041_orders_by_the_unrounded_score(self, api, world):
        world.frame("a", 0.0, 0.95001)
        world.frame("z", 0.0, 0.95004)
        world.pad(9)

        body = _ok(_get(api, world))

        assert _ids(body) == ["z", "a"]
        assert [s["score"] for s in body["segments"]] == [0.95, 0.95]


class TestPartial:
    def _crowd(self, world, count):
        for i in range(count):
            world.frame(f"crowd{i}", 10.0, 0.99, drive=OTHER)

    def test_spec_addon_041_other_drives_filling_the_knn_leave_no_seed_and_partial_true(
        self, api, world,
    ):
        self._crowd(world, 4096)
        for i in range(3):
            world.frame(f"own{i}", 10.0, 0.5)

        body = _ok(_get(api, world))

        assert body == {"query": "cat", "segments": [], "partial": True}

    def test_spec_addon_041_enough_seeds_after_a_full_knn_is_not_partial(self, api, world):
        self._crowd(world, 4090)
        for i in range(10):
            world.frame(f"own{i}", 10.0, 0.5 - 0.01 * i)

        body = _ok(_get(api, world))

        assert body["partial"] is False
        assert _ids(body) == ["own0"]

    def test_spec_addon_041_fewer_seeds_than_required_after_a_full_knn_is_partial(
        self, api, world,
    ):
        self._crowd(world, 4094)
        for i in range(30):
            world.frame(f"own{i:02d}", 10.0, 0.5 - 0.001 * i)

        body = _ok(_get(api, world))

        assert body["partial"] is True
        assert _ids(body) == ["own00", "own01"]

    def test_spec_addon_041_exactly_required_seeds_after_a_full_knn_is_not_partial(
        self, api, world,
    ):
        self._crowd(world, 4093)
        for i in range(30):
            world.frame(f"own{i:02d}", 10.0, 0.5 - 0.001 * i)

        body = _ok(_get(api, world))

        assert body["partial"] is False
        assert _ids(body) == ["own00", "own01", "own02"]


class TestEmpty:
    def test_spec_addon_041_a_drive_without_frames_answers_empty_and_never_embeds(
        self, api, world,
    ):
        world.frame("theirs", 10.0, 0.9, drive=OTHER)

        body = _ok(_get(api, world))

        assert body == {"query": "cat", "segments": [], "partial": False}
        assert world.asked == []

    def test_spec_addon_041_frames_present_but_none_has_a_vector_is_empty_not_partial(
        self, api, world,
    ):
        world.frame("A", 10.0, None)

        body = _ok(_get(api, world))

        assert body == {"query": "cat", "segments": [], "partial": False}


class TestParameters:
    @pytest.mark.parametrize(
        "params",
        [
            pytest.param({}, id="no-q"),
            pytest.param({"q": ""}, id="empty-q"),
            pytest.param({"q": " \t "}, id="blank-q"),
            pytest.param({"q": "x" * 201}, id="q-201"),
            pytest.param({"q": "cat", "limit": 0}, id="limit-0"),
            pytest.param({"q": "cat", "limit": 51}, id="limit-51"),
            pytest.param({"q": "cat", "limit": "many"}, id="limit-not-int"),
        ],
    )
    def test_spec_addon_041_invalid_parameters_are_422_and_not_logged(
        self, api, world, caplog, params,
    ):
        world.frame("A", 10.0, 0.9)
        caplog.set_level(logging.DEBUG)

        response = _get(api, world, params)

        assert response.status_code == 422, response.text
        assert not [
            r for r in caplog.records
            if r.name.startswith("app") and r.levelno >= logging.WARNING
        ]

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param({"q": "c"}, id="q-1"),
            pytest.param({"q": "c" * 200}, id="q-200"),
            pytest.param({"q": " " + "c" * 200 + " "}, id="q-200-after-strip"),
            pytest.param({"q": "cat", "limit": 1}, id="limit-1"),
            pytest.param({"q": "cat", "limit": 50}, id="limit-50"),
        ],
    )
    def test_spec_addon_041_boundary_parameters_are_accepted(self, api, world, params):
        world.frame("A", 10.0, 0.9)

        assert _get(api, world, params).status_code == 200

    def test_spec_addon_041_missing_drive_header_is_400(self, api, world):
        assert _get(api, world, drive=None).status_code == 400


class TestFailures:
    def test_spec_addon_041_encoder_failure_is_503_embedding_unavailable_and_logged(
        self, api, world, monkeypatch, caplog,
    ):
        world.frame("A", 10.0, 0.9)

        def _broken(*_a, **_k):
            raise RuntimeError("model failed to load")

        patch_everywhere(monkeypatch, "embed_text_clip", _broken)
        caplog.set_level(logging.DEBUG)

        response = _get(api, world)

        assert response.status_code == 503
        assert response.json()["detail"] == "Embedding unavailable"
        assert [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_spec_addon_041_a_table_read_failure_is_500_search_failed_and_logged(
        self, api, world, search_index, caplog,  # noqa: F811
    ):
        world.frame("A", 10.0, 0.9)
        world.commit()
        search_index.sql("DROP TABLE vec_clip")
        caplog.set_level(logging.DEBUG)

        response = api(PATH, {"q": "cat"})

        assert response.status_code == 500
        assert response.json()["detail"] == "Search failed"
        assert [r for r in caplog.records if r.levelno >= logging.ERROR]


class TestSideEffects:
    def test_spec_addon_041_writes_nothing_and_reads_nothing_on_the_event_loop(
        self, api, world, statements,  # noqa: F811
    ):
        world.frame("A", 10.0, 0.9)
        world.commit()
        statements.reset()

        body = _ok(api(PATH, {"q": "cat"}))

        assert len(body["segments"]) == 1
        assert statements.statements
        assert statements.writes() == []
        assert statements.on_loop == []


def test_spec_addon_041_manifest_declares_the_route_with_the_search_gate_and_30s_timeout():
    routes = json.loads(MANIFEST.read_text(encoding="utf-8"))["proxy"]["routes"]

    (route,) = [r for r in routes if r["path"] == PATH]

    assert route["methods"] == ["GET"]
    assert route["pre_check"] == {"type": "addon_feature", "feature": "search"}
    assert route["timeout"] == 30
