"""SPEC-ADDON-033: ``GET /search/scenes`` over HTTP, against a real search DB
with sqlite-vec loaded.

Vectors are unit vectors unless a test says otherwise, so a scene at L2
distance ``d`` from the query has cosine ``1 - d^2 / 2``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

import app.workers.embedder  # noqa: F401  (so the stand-in reaches every importer)
from tests.search_endpoint_db import (  # noqa: F401
    DRIVE,
    OTHER,
    api,
    patch_everywhere,
    search_index,
    seed,
    statements,
    visual,
)
from tests.vec_partition_db import at_distance, unit

PATH = "/search/scenes"
RESULT_KEYS = {
    "file_id", "filename", "start_time", "end_time", "scene_label", "visible_text", "score",
}


def _cos(d: float):
    return pytest.approx(1.0 - d * d / 2.0, abs=1e-4)


def _ok(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture()
def dim(search_index):  # noqa: F811
    return search_index.dim("vec_text")


@pytest.fixture()
def embedded(monkeypatch, dim):
    asked: list[tuple] = []

    def _embed_query(*args, **kwargs):
        asked.append((args, kwargs))
        return unit(dim, 0)

    patch_everywhere(monkeypatch, "embed_query", _embed_query)
    return asked


class Scenes:
    def __init__(self, seed, dim):
        self.seed = seed
        self.dim = dim
        self._axis = 0
        self._files: set[str] = set()

    def video(self, file_id, *, drive=DRIVE, active=True, filename=None):
        if file_id not in self._files:
            self.seed.file(
                file_id, filename or f"{file_id}.mp4", drive=drive, active=active,
                mime="video/mp4",
            )
            self.seed.run(file_id)
            self._files.add(file_id)
        return f"vvr_{file_id}"

    def add(
        self, file_id, distance, start, end=None, *, label="a label", visible=None,
        drive=DRIVE, active=True, run_id=None, axis=None, scale=1.0,
    ):
        run = run_id or self.video(file_id, drive=drive, active=active)
        scene_id = self.seed.scene(run, start, end, label=label, visible=visible)
        if axis is None:
            self._axis += 1
            axis = self._axis
        self.seed.scene_embedding(
            file_id, scene_id, scale * at_distance(self.dim, distance, axis),
        )
        return scene_id


@pytest.fixture()
def scenes(seed, dim, visual, embedded):
    visual(enabled=True)
    return Scenes(seed, dim)


def _ranked(body) -> list[tuple]:
    return [(r["file_id"], r["start_time"], r["score"]) for r in body["results"]]


class TestRanking:
    def test_spec_addon_033_ranks_the_drives_scenes_by_cosine_highest_first(self, api, scenes):
        scenes.add("vid-a", 0.9, 30.0, 40.0, label="far", visible="EXIT")
        scenes.add("vid-a", 0.2, 10.0, 20.0, label="near", visible=None)
        scenes.add("vid-b", 0.5, 5.0, None, label="middle", visible="sign")

        body = _ok(api(PATH, {"q": "a red car"}))

        assert set(body) == {"query", "results"}
        assert body["query"] == "a red car"
        assert body["results"] == [
            {"file_id": "vid-a", "filename": "vid-a.mp4", "start_time": 10.0,
             "end_time": 20.0, "scene_label": "near", "visible_text": None,
             "score": _cos(0.2)},
            {"file_id": "vid-b", "filename": "vid-b.mp4", "start_time": 5.0,
             "end_time": None, "scene_label": "middle", "visible_text": "sign",
             "score": _cos(0.5)},
            {"file_id": "vid-a", "filename": "vid-a.mp4", "start_time": 30.0,
             "end_time": 40.0, "scene_label": "far", "visible_text": "EXIT",
             "score": _cos(0.9)},
        ]
        for result in body["results"]:
            assert set(result) == RESULT_KEYS

    def test_spec_addon_033_score_is_cosine_even_for_a_vector_that_is_not_unit_length(
        self, api, scenes,
    ):
        scenes.add("vid-a", 0.6, 1.0, scale=3.0)

        (result,) = _ok(api(PATH, {"q": "cat"}))["results"]

        assert result["score"] == _cos(0.6)

    def test_spec_addon_033_score_is_rounded_to_four_places(self, api, scenes):
        scenes.add("vid-a", 0.123456, 1.0)

        (result,) = _ok(api(PATH, {"q": "cat"}))["results"]

        assert result["score"] == _cos(0.123456)
        assert result["score"] == round(result["score"], 4)
        assert result["score"] != 1.0 - 0.123456 ** 2 / 2.0

    def test_spec_addon_033_no_score_threshold_even_for_a_negative_cosine(self, api, scenes):
        scenes.add("vid-a", 1.9, 1.0)

        (result,) = _ok(api(PATH, {"q": "cat"}))["results"]

        assert result["score"] == _cos(1.9)
        assert result["score"] < 0

    def test_spec_addon_033_ties_break_by_file_id_then_start_time(self, api, scenes):
        scenes.add("vid-b", 0.4, 1.0, axis=7)
        scenes.add("vid-a", 0.4, 9.0, axis=7)
        scenes.add("vid-a", 0.4, 3.0, axis=7)

        body = _ok(api(PATH, {"q": "cat"}))

        assert _ranked(body) == [
            ("vid-a", 3.0, _cos(0.4)), ("vid-a", 9.0, _cos(0.4)), ("vid-b", 1.0, _cos(0.4)),
        ]

    @pytest.mark.parametrize("limit,count", [(None, 20), (1, 1), (7, 7), (50, 50)])
    def test_spec_addon_033_returns_at_most_limit_nearest_with_default_20(
        self, api, scenes, limit, count,
    ):
        distances = [0.01 * (i + 1) for i in range(60)]
        for i, d in enumerate(reversed(distances)):
            scenes.add(f"vid-{i % 3}", d, float(i))

        params = {"q": "cat"} if limit is None else {"q": "cat", "limit": limit}
        body = _ok(api(PATH, params))

        assert [r["score"] for r in body["results"]] == [_cos(d) for d in distances[:count]]


class TestCandidates:
    def test_spec_addon_033_nearer_rows_outside_the_drive_do_not_crowd_out_its_scenes(
        self, api, scenes, seed, dim,
    ):
        mine = [scenes.add("mine", d, float(i)) for i, d in enumerate((0.6, 0.7, 0.8))]
        for i in range(60):
            scenes.add(f"theirs-{i % 4}", 0.05, float(i), drive=OTHER)
        for i in range(30):
            scenes.add(f"gone-{i % 3}", 0.05, float(i), active=False)
        seed.run("mine", "vvr_mine_old", active=False)
        for i in range(30):
            scenes.add("mine", 0.05, 100.0 + i, run_id="vvr_mine_old")
        for i in range(30):
            emb = f"whisper_mine_{i}"
            seed.index.add_embedding(emb, "mine", "whisper", "vec_text")
            seed.index.put_vector("vec_text", emb, at_distance(dim, 0.05, 200 + i),
                                  partition="whisper")

        body = _ok(api(PATH, {"q": "cat", "limit": 3}))

        assert _ranked(body) == [
            ("mine", 0.0, _cos(0.6)), ("mine", 1.0, _cos(0.7)), ("mine", 2.0, _cos(0.8)),
        ]
        assert len(mine) == 3

    def test_spec_addon_033_another_drive_sees_only_its_own_scenes(self, api, scenes):
        scenes.add("mine", 0.3, 1.0)
        scenes.add("theirs", 0.2, 2.0, drive=OTHER)

        body = _ok(api(PATH, {"q": "cat"}, drive=OTHER))

        assert [r["file_id"] for r in body["results"]] == ["theirs"]

    def test_spec_addon_033_an_embedding_of_another_type_never_scores_a_scene(
        self, api, scenes, seed, dim,
    ):
        scene_id = scenes.add("vid-a", 0.8, 1.0)
        impostor = f"vvs_{scene_id}_bbbbbbbb"
        seed.index.add_embedding(impostor, "vid-a", "whisper", "vec_text")
        seed.index.put_vector("vec_text", impostor, at_distance(dim, 0.1, 300),
                              partition="whisper")

        body = _ok(api(PATH, {"q": "cat"}))

        assert _ranked(body) == [("vid-a", 1.0, _cos(0.8))]

    def test_spec_addon_033_a_scene_with_several_embeddings_is_listed_once_at_its_best(
        self, api, scenes, seed, dim,
    ):
        scene_id = scenes.add("vid-a", 0.7, 1.0)
        seed.scene_embedding("vid-a", scene_id, at_distance(dim, 0.3, 400), suffix="cccccccc")
        seed.scene_embedding("vid-a", scene_id, at_distance(dim, 0.9, 401), suffix="dddddddd")

        body = _ok(api(PATH, {"q": "cat"}))

        assert _ranked(body) == [("vid-a", 1.0, _cos(0.3))]

    def test_spec_addon_033_scenes_without_an_embedding_or_vector_are_skipped(
        self, api, scenes, seed,
    ):
        run = scenes.video("vid-a")
        no_vector = seed.scene(run, 5.0, label="no vector")
        seed.scene_embedding("vid-a", no_vector, None)
        seed.scene(run, 9.0, label="no embedding")
        scenes.add("vid-a", 0.5, 1.0, label="ranked")

        body = _ok(api(PATH, {"q": "cat"}))

        assert [r["scene_label"] for r in body["results"]] == ["ranked"]

    def test_spec_addon_033_a_drive_without_scene_embeddings_is_an_empty_list(
        self, api, scenes, seed,
    ):
        run = scenes.video("vid-a")
        seed.scene(run, 1.0, label="unembedded")

        body = _ok(api(PATH, {"q": "cat"}))

        assert body == {"query": "cat", "results": []}


class TestQuery:
    def test_spec_addon_033_q_is_stripped_echoed_and_embedded_stripped(
        self, api, scenes, embedded,
    ):
        scenes.add("vid-a", 0.2, 1.0)

        body = _ok(api(PATH, {"q": "  a red car \n"}))

        assert body["query"] == "a red car"
        assert embedded, "the query must be embedded"
        assert all(
            "a red car" in (list(args) + list(kwargs.values())) for args, kwargs in embedded
        )

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param({}, id="no-q"),
            pytest.param({"q": ""}, id="empty-q"),
            pytest.param({"q": " \t "}, id="blank-q"),
            pytest.param({"q": "x" * 1001}, id="q-1001"),
            pytest.param({"q": "cat", "limit": 0}, id="limit-0"),
            pytest.param({"q": "cat", "limit": 51}, id="limit-51"),
            pytest.param({"q": "cat", "limit": "many"}, id="limit-not-int"),
            pytest.param({"q": "cat", "limit": 2.5}, id="limit-fraction"),
        ],
    )
    def test_spec_addon_033_invalid_parameters_are_422_and_not_logged(
        self, api, scenes, caplog, params,
    ):
        caplog.set_level(logging.DEBUG)
        response = api(PATH, params)

        assert response.status_code == 422, response.text
        assert not [
            r for r in caplog.records
            if r.name.startswith("app") and r.levelno >= logging.WARNING
        ]

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param({"q": "c"}, id="q-1"),
            pytest.param({"q": "c" * 1000}, id="q-1000"),
            pytest.param({"q": " " + "c" * 1000 + " "}, id="q-1000-after-strip"),
            pytest.param({"q": "cat", "limit": 1}, id="limit-1"),
            pytest.param({"q": "cat", "limit": 50}, id="limit-50"),
        ],
    )
    def test_spec_addon_033_boundary_parameters_are_accepted(self, api, scenes, params):
        assert api(PATH, params).status_code == 200

    def test_spec_addon_033_missing_drive_header_is_400(self, api, scenes):
        assert api(PATH, {"q": "cat"}, drive=None).status_code == 400


class TestVisualIndexGate:
    @pytest.mark.parametrize(
        "enabled,policy",
        [
            pytest.param(False, None, id="feature-off"),
            pytest.param(True, {DRIVE: False}, id="drive-policy-off"),
            pytest.param(True, {DRIVE: RuntimeError("policy unavailable")},
                         id="drive-policy-unreadable"),
        ],
    )
    def test_spec_addon_033_closed_visual_index_is_404_and_shows_no_scene(
        self, api, scenes, visual, enabled, policy,
    ):
        scenes.add("vid-a", 0.2, 1.0, label="secret label")
        assert _ok(api(PATH, {"q": "cat"}))["results"]
        visual(enabled=enabled, policy=policy)

        response = api(PATH, {"q": "cat"})

        assert response.status_code == 404
        assert "secret label" not in response.text

    def test_spec_addon_033_another_drives_policy_does_not_close_this_drive(
        self, api, scenes, visual,
    ):
        scenes.add("vid-a", 0.2, 1.0)
        visual(enabled=True, policy={OTHER: False})

        assert len(_ok(api(PATH, {"q": "cat"}))["results"]) == 1


class TestFailures:
    def test_spec_addon_033_embedder_failure_is_503_and_logged_as_a_warning_with_its_type(
        self, api, scenes, monkeypatch, caplog,
    ):
        scenes.add("vid-a", 0.2, 1.0)

        class EmbedderBroke(RuntimeError):
            pass

        def _broken(*_args, **_kwargs):
            raise EmbedderBroke("model failed to load")

        patch_everywhere(monkeypatch, "embed_query", _broken)
        caplog.set_level(logging.DEBUG)

        response = api(PATH, {"q": "cat"})

        assert response.status_code == 503
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "EmbedderBroke" in r.getMessage()
            or (r.exc_info and r.exc_info[0] is EmbedderBroke)
            for r in warnings
        ), [r.getMessage() for r in caplog.records]

    def test_spec_addon_033_a_vector_read_failure_is_500_search_failed_and_logged(
        self, api, scenes, seed, caplog,
    ):
        scenes.add("vid-a", 0.2, 1.0)
        seed.index.sql("DROP TABLE vec_text")
        caplog.set_level(logging.DEBUG)

        response = api(PATH, {"q": "cat"})

        assert response.status_code == 500
        assert response.json()["detail"] == "Search failed"
        assert [r for r in caplog.records if r.levelno >= logging.ERROR]


class TestSideEffects:
    def test_spec_addon_033_writes_nothing_and_reads_nothing_on_the_event_loop(
        self, api, scenes, statements,
    ):
        scenes.add("vid-a", 0.2, 1.0)
        statements.reset()

        body = _ok(api(PATH, {"q": "cat"}))

        assert len(body["results"]) == 1
        assert statements.statements, "the endpoint must read the search DB"
        assert statements.writes() == []
        assert statements.on_loop == []


def test_spec_addon_033_query_vector_is_compared_not_the_query_text(api, scenes, monkeypatch, dim):
    scenes.add("vid-a", 0.2, 1.0, axis=1)
    scenes.add("vid-b", 0.2, 1.0, axis=2)
    toward_b = at_distance(dim, 0.2, 2)
    patch_everywhere(monkeypatch, "embed_query", lambda *_a, **_k: np.asarray(toward_b))

    body = _ok(api(PATH, {"q": "cat"}))

    assert body["results"][0]["file_id"] == "vid-b"
    assert body["results"][0]["score"] == _cos(0.0)
