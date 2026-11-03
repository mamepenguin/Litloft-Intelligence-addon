"""SPEC-ADDON-055: ``GET /search/images`` ranks the drive's representative frames
exactly, first 2000, no margin cutoff.

Frames are unit vectors, so a frame placed at score ``s`` is at L2 distance
``sqrt(2 (1 - s))`` from the query and scores ``s`` back.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import text

import app.database as database
from app import search as search_mod
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

PATH = "/search/images"
MANIFEST = Path(__file__).resolve().parents[1] / "manifest.json"


def _distance(score: float) -> float:
    return float(np.sqrt(2.0 * (1.0 - score)))


def _ok(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()


def _ids(body) -> list[str]:
    return [r["file_id"] for r in body["results"]]


class World:
    """Files and their frames, buffered so a test can build thousands."""

    def __init__(self, index):
        self.index = index
        self.dim = index.dim("vec_clip")
        self.scores: dict[str, float] = {}
        self._files: dict[str, dict] = {}
        self._rows: list[tuple] = []
        self._n = 0

    def file(self, file_id, *, drive=DRIVE, active=True, mime="image/jpeg"):
        file_type = mime.split("/")[0]
        if file_type not in ("video", "image", "audio"):
            file_type = "document"
        self._files[file_id] = {
            "drive": drive, "active": active, "mime": mime, "file_type": file_type,
        }

    def frame(self, file_id, score, *, etype="clip_thumbnail", **file_kw):
        if file_id not in self._files:
            self.file(file_id, **file_kw)
        self.scores[file_id] = score
        self._n += 1
        self._rows.append((f"e{self._n}", file_id, etype, score))

    def commit(self):
        files, rows = self._files, self._rows
        with database.get_search_db() as s:
            s.add_all([
                IndexedFile(
                    file_id=f, drive=v["drive"], filename=f"{f}.bin",
                    file_path=f"/drives/{v['drive']}/{f}.bin",
                    file_type=v["file_type"], mime_type=v["mime"], file_size=1,
                    active=v["active"],
                )
                for f, v in files.items()
            ])
            s.add_all([
                Embedding(
                    id=e, file_id=f, embedding_type=t, vector_table="vec_clip",
                    content_preview="x",
                )
                for e, f, t, _ in rows
            ])
        if rows:
            with self.index.engine.begin() as conn:
                conn.execute(
                    text("INSERT INTO vec_clip(embedding_id, embedding_type, vector) "
                         "VALUES (:i, :t, :v)"),
                    [
                        {
                            "i": e, "t": t,
                            "v": at_distance(self.dim, _distance(s), n)
                            .astype(np.float32).tobytes(),
                        }
                        for n, (e, _, t, s) in enumerate(rows)
                    ],
                )
        self._rows.clear()


@pytest.fixture()
def world(search_index):  # noqa: F811
    return World(search_index)


@pytest.fixture()
def query_vector(world, monkeypatch):
    dim = world.dim
    text_dim = world.index.dim("vec_text")
    patch_everywhere(monkeypatch, "embed_text_clip", lambda *_a, **_k: unit(dim, 0))
    patch_everywhere(monkeypatch, "embed_query", lambda *_a, **_k: unit(text_dim, 0))


@pytest.fixture()
def hydration(monkeypatch):
    """The core's bulk hydration, answering with every id it was asked for."""
    calls: list[list[str]] = []

    async def _hydrate(ids):
        calls.append(list(ids))
        return {fid: {"id": fid} for fid in ids}

    patch_everywhere(monkeypatch, "hydrate_files", _hydrate)
    return calls


def _get(api, world, params=None, drive=DRIVE):
    world.commit()
    return api(PATH, {"q": "cat"} if params is None else params, drive=drive)


# ---------------------------------------------------------------------------
# SPEC-ADDON-055 I4: the drive's own top 2000 files by one frame's cosine
# ---------------------------------------------------------------------------


class TestRanking:
    def test_spec_addon_055_i4_returns_only_active_files_of_the_request_drive(
        self, api, world, query_vector, hydration,
    ):
        world.frame("own", 0.80)
        world.frame("foreign", 0.95, drive=OTHER)
        world.frame("gone", 0.90, active=False)

        assert _ids(_ok(_get(api, world))) == ["own"]

    def test_spec_addon_055_i4_requires_a_clip_thumbnail_frame(
        self, api, world, query_vector, hydration,
    ):
        world.frame("own", 0.80)
        world.frame("scene", 0.95, etype="clip")

        assert _ids(_ok(_get(api, world))) == ["own"]

    @pytest.mark.parametrize(
        "kind, expected",
        [
            pytest.param("image", ["img"], id="image"),
            pytest.param("video", ["vid"], id="video"),
        ],
    )
    def test_spec_addon_055_i4_applies_the_requested_type(
        self, api, world, query_vector, hydration, kind, expected,
    ):
        world.frame("img", 0.80, mime="image/jpeg")
        world.frame("vid", 0.90, mime="video/mp4")

        assert _ids(_ok(_get(api, world, {"q": "cat", "type": kind}))) == expected

    def test_spec_addon_055_i4_drops_frames_below_the_floor(
        self, api, world, query_vector, hydration, monkeypatch, make_settings,
    ):
        search = search_mod.settings.search
        patch_everywhere(monkeypatch, "settings", make_settings(
            search=dataclasses.replace(search, min_score_clip_thumbnail=0.50),
        ))
        world.frame("kept", 0.51)
        world.frame("below", 0.49)

        assert _ids(_ok(_get(api, world))) == ["kept"]

    def test_spec_addon_055_i4_has_no_margin_cutoff(
        self, api, world, query_vector, hydration,
    ):
        world.frame("near", 0.90)
        world.frame("far", 0.20)

        assert _ids(_ok(_get(api, world))) == ["near", "far"]

    def test_spec_addon_055_i4_orders_by_descending_cosine_then_file_id(
        self, api, world, query_vector, hydration,
    ):
        world.frame("b", 0.90)
        world.frame("a", 0.90)
        world.frame("m", 0.80)

        assert _ids(_ok(_get(api, world))) == ["a", "b", "m"]

    def test_spec_addon_055_i4_ranks_exactly_past_4096_nearer_frames_of_other_drives(
        self, api, world, query_vector, hydration,
    ):
        for i in range(4100):
            world.frame(f"crowd{i:04d}", 0.99, drive=OTHER)
        for fid, score in (("own0", 0.60), ("own1", 0.50), ("own2", 0.40)):
            world.frame(fid, score)

        assert _ids(_ok(_get(api, world))) == ["own0", "own1", "own2"]

    def test_spec_addon_055_i4_keeps_the_top_2000_in_order(
        self, api, world, query_vector, hydration,
    ):
        for i in range(2010):
            world.frame(f"f{i:04d}", 0.99 - 0.0001 * i)

        assert _ids(_ok(_get(api, world))) == [f"f{i:04d}" for i in range(2000)]


# ---------------------------------------------------------------------------
# SPEC-ADDON-055 I5: the result /search would build for that frame alone
# ---------------------------------------------------------------------------


def test_spec_addon_055_i5_each_result_is_the_precision_search_result_for_that_frame(
    api, world, query_vector, hydration,
):
    for fid, score in (("a", 0.92), ("b", 0.91), ("c", 0.95)):
        world.frame(fid, score)
    world.commit()

    images = _ok(api(PATH, {"q": "cat"}))
    search = _ok(api("/search", {"q": "cat", "limit": 100}))
    by_id = {r["file_id"]: r for r in search["results"]}

    assert [r["file_id"] for r in images["results"]] == ["c", "a", "b"]
    for item in images["results"]:
        assert item == by_id[item["file_id"]]
        assert item["match_types"] == ["clip_thumbnail"]
        (segment,) = item["segments"]
        assert segment["time_range"] is None
        (match,) = segment["matches"]
        assert match["type"] == "clip_thumbnail"
        assert match["score"] == pytest.approx(world.scores[item["file_id"]], abs=5e-5)


def test_spec_addon_055_i5_score_is_alpha_times_the_thumbnail_weight_times_the_cosine(
    api, world, query_vector, hydration,
):
    world.frame("a", 0.80)
    config = search_mod.settings.search

    (item,) = _ok(_get(api, world))["results"]

    expected = config.alpha * config.type_weight_clip_thumbnail * world.scores["a"]
    assert item["score"] == pytest.approx(expected, abs=5e-5)


# ---------------------------------------------------------------------------
# Failure cases: 503 rather than snapshot fields
# ---------------------------------------------------------------------------


def test_spec_addon_055_an_embedding_failure_answers_503(
    api, world, monkeypatch,
):
    world.frame("a", 0.90)

    def _broken(*_a, **_k):
        raise RuntimeError("model failed to load")

    patch_everywhere(monkeypatch, "embed_text_clip", _broken)

    response = _get(api, world)

    assert response.status_code == 503


def test_spec_addon_055_a_hydration_batch_that_returns_nothing_answers_503(
    api, world, query_vector, monkeypatch,
):
    world.frame("a", 0.90)

    async def _empty(_ids):
        return {}

    patch_everywhere(monkeypatch, "hydrate_files", _empty)

    response = _get(api, world)

    assert response.status_code == 503


def test_spec_addon_055_hydrates_in_batches_of_at_most_500(
    api, world, query_vector, hydration,
):
    for i in range(2010):
        world.frame(f"f{i:04d}", 0.99 - 0.0001 * i)

    assert len(_ok(_get(api, world))["results"]) == 2000

    assert len(hydration) >= 4
    assert max(len(call) for call in hydration) <= 500
    assert sum(len(call) for call in hydration) == 2000


# ---------------------------------------------------------------------------
# SPEC-ADDON-055 I14: neither endpoint writes
# ---------------------------------------------------------------------------


def test_spec_addon_055_i14_neither_endpoint_writes_to_the_search_db(
    api, world, query_vector, hydration, statements,  # noqa: F811
):
    world.frame("a", 0.90)
    world.commit()
    statements.reset()

    assert _ids(_ok(api(PATH, {"q": "cat"}))) == ["a"]
    assert _ids(_ok(api("/search", {"q": "cat"}))) == ["a"]

    assert statements.statements
    assert statements.writes() == []
    assert statements.on_loop == []


# ---------------------------------------------------------------------------
# Manifest: the same gate, filter and timeout as the other search routes
# ---------------------------------------------------------------------------


def _route(path: str) -> dict:
    routes = json.loads(MANIFEST.read_text(encoding="utf-8"))["proxy"]["routes"]
    (route,) = [r for r in routes if r["path"] == path]
    return route


def test_spec_addon_055_manifest_declares_search_images_like_search():
    route = _route(PATH)

    assert route["methods"] == ["GET"]
    assert route["pre_check"] == {"type": "addon_feature", "feature": "search"}
    assert route["response_filter"] == {
        "type": "current_drive_only",
        "array_path": "results",
        "drive_field": "drive",
    }
    assert route["timeout"] == 30


def test_spec_addon_046_manifest_gives_search_a_30s_timeout():
    assert _route("/search")["timeout"] == 30
