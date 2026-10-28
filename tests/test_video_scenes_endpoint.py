"""SPEC-ADDON-031: ``GET /files/{file_id}/scenes`` over HTTP."""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import IndexedFile, VideoVisualRun, VideoVisualScene
from app.workers.video_visual_selection import Candidate, select_candidates

LOFT_MIME = "application/vnd.litloft.loft+json"
DRIVE = "family"
MANIFEST = Path(__file__).resolve().parent.parent / "manifest.json"


def _unit_vec(seed: int, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _basis(i: int, dim: int = 16) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


DURATIONS: dict[str, float | None] = {
    "vid-long": 7200.0,
    "vid-nodur": None,
    "vid-short": 600.0,
    "vid-empty": 600.0,
}

CANDIDATES: dict[str, list[Candidate]] = {
    "vid-long": [Candidate(f"L{i}", i * 36.0, _unit_vec(i)) for i in range(200)],
    "vid-nodur": [Candidate(f"N{i}", i * 15.0, _unit_vec(1000 + i)) for i in range(40)],
    # The last one lies past the probed duration, as a CLIP candidate stamped
    # beyond the container's duration can.
    "vid-short": [
        Candidate("S0", 100.0, _basis(0)),
        Candidate("S1", 300.0, _basis(1)),
        Candidate("S2", 600.5, _basis(2)),
    ],
    "vid-empty": [],
}


def _patch_everywhere(monkeypatch, name: str, value) -> None:
    for mod_name, module in list(sys.modules.items()):
        if module is None or not (mod_name == "app" or mod_name.startswith("app.")):
            continue
        if name in getattr(module, "__dict__", {}):
            monkeypatch.setattr(module, name, value)


@pytest.fixture()
def env(monkeypatch, tmp_path):
    from app.main import app

    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    def _video(file_id, duration, mime="video/mp4"):
        return IndexedFile(
            file_id=file_id, drive=DRIVE, filename=f"{file_id}.mp4",
            file_path=f"/drives/{DRIVE}/{file_id}.mp4", file_type="video",
            mime_type=mime, file_size=1000, active=True, duration=duration,
        )

    with Session() as s:
        s.add_all([_video(fid, dur) for fid, dur in DURATIONS.items()])
        s.add(_video("loft-1", None, mime=LOFT_MIME))
        s.add(IndexedFile(
            file_id="img-1", drive=DRIVE, filename="cat.jpg",
            file_path=f"/drives/{DRIVE}/cat.jpg", file_type="image",
            mime_type="image/jpeg", file_size=1000, active=True,
        ))
        s.commit()

    @contextmanager
    def _read():
        session = Session()
        try:
            yield session
        finally:
            session.rollback()
            session.close()

    @contextmanager
    def _write():
        session = Session()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    _patch_everywhere(monkeypatch, "get_search_db_read", _read)
    _patch_everywhere(monkeypatch, "get_search_db", _write)

    loaded: list[str] = []

    def _load_candidates(file_id):
        loaded.append(file_id)
        return list(CANDIDATES.get(file_id, [])), DURATIONS.get(file_id)

    _patch_everywhere(monkeypatch, "_load_candidates", _load_candidates)

    excerpt_calls: list[tuple] = []

    def _excerpt(file_id, start_time, end_time):
        excerpt_calls.append((file_id, start_time, end_time))
        return "" if start_time == 300.0 else f"near {start_time}"

    _patch_everywhere(monkeypatch, "_select_transcript_excerpt", _excerpt)

    client = TestClient(app)

    def get(file_id, params=None, drive=DRIVE):
        return client.get(
            f"/files/{file_id}/scenes",
            params=params or {},
            headers={"X-Lit-Drive": drive},
        )

    return {
        "app": app,
        "get": get,
        "Session": Session,
        "excerpt_calls": excerpt_calls,
        "loaded": loaded,
    }


def _expected_starts(candidates, duration):
    picked = select_candidates(candidates, duration_seconds=duration)
    return sorted(c.timestamp for c in picked)


def _all(_ts):
    return True


WINDOW_ROWS = [
    pytest.param("vid-long", {}, None, _all, 7200.0, id="neither-dur-known"),
    pytest.param("vid-nodur", {}, None, _all, None, id="neither-dur-null"),
    pytest.param(
        "vid-long", {"start": 3600}, {"start": 3600, "end": 7200},
        lambda ts: 3600 <= ts < 7200, 3600.0, id="start-only-dur-known",
    ),
    pytest.param(
        "vid-nodur", {"start": 150}, {"start": 150, "end": None},
        lambda ts: 150 <= ts, None, id="start-only-dur-null",
    ),
    pytest.param(
        "vid-long", {"end": 1800}, {"start": 0, "end": 1800},
        lambda ts: ts < 1800, 1800.0, id="end-only",
    ),
    pytest.param(
        "vid-nodur", {"end": 300}, {"start": 0, "end": 300},
        lambda ts: ts < 300, 300.0, id="end-only-dur-null",
    ),
    pytest.param(
        "vid-long", {"start": 1000, "end": 4600}, {"start": 1000, "end": 4600},
        lambda ts: 1000 <= ts < 4600, 3600.0, id="both",
    ),
]


class TestWindowTable:
    @pytest.mark.parametrize("file_id,params,window,keep,duration", WINDOW_ROWS)
    def test_spec_addon_031_window_candidates_and_budget(
        self, env, file_id, params, window, keep, duration,
    ):
        response = env["get"](file_id, params)
        assert response.status_code == 200, response.text
        body = response.json()

        kept = [c for c in CANDIDATES[file_id] if keep(c.timestamp)]
        starts = [s["start"] for s in body["scenes"]]

        assert body["file_id"] == file_id
        assert body["duration"] == DURATIONS[file_id]
        assert body["window"] == window
        assert body["candidate_count"] == len(kept)
        assert starts == _expected_starts(kept, duration)
        assert len(starts) <= min(24, len(kept))
        if window is not None:
            for ts in starts:
                assert ts >= window["start"]
                if window["end"] is not None:
                    assert ts < window["end"]

    @pytest.mark.parametrize("file_id,params,window,keep,duration", WINDOW_ROWS)
    def test_spec_addon_031_scene_end_is_next_start_then_window_or_duration(
        self, env, file_id, params, window, keep, duration,
    ):
        response = env["get"](file_id, params)
        assert response.status_code == 200, response.text
        scenes = response.json()["scenes"]
        assert scenes, "the row must select at least one scene"
        for current, following in zip(scenes, scenes[1:]):
            assert current["end"] == following["start"]
        last = scenes[-1]
        if window is not None:
            assert last["end"] == window["end"]
        else:
            dur = DURATIONS[file_id]
            assert last["end"] == (dur if dur is not None and dur > last["start"] else None)

    def test_spec_addon_031_a_budget_of_24_is_reached_on_a_long_video(self, env):
        response = env["get"]("vid-long")
        assert response.status_code == 200, response.text
        assert len(response.json()["scenes"]) == 24

    def test_spec_addon_031_last_scene_past_the_duration_has_no_end(self, env):
        response = env["get"]("vid-short")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [s["start"] for s in body["scenes"]] == [100.0, 300.0, 600.5]
        assert [s["end"] for s in body["scenes"]] == [300.0, 600.5, None]

    @pytest.mark.parametrize(
        "params,window",
        [
            pytest.param({"start": 7200}, {"start": 7200, "end": 7200}, id="start-at-dur"),
            pytest.param({"start": 8000}, {"start": 8000, "end": 7200}, id="start-past-dur"),
            pytest.param(
                {"start": 8000, "end": 9000}, {"start": 8000, "end": 9000},
                id="both-past-dur",
            ),
        ],
    )
    def test_spec_addon_031_window_at_or_past_the_duration_is_empty_not_an_error(
        self, env, params, window,
    ):
        response = env["get"]("vid-long", params)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["scenes"] == []
        assert body["candidate_count"] == 0
        assert body["window"] == window

    def test_spec_addon_031_no_candidates_is_an_empty_list(self, env):
        response = env["get"]("vid-empty")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["scenes"] == []
        assert body["candidate_count"] == 0

    def test_spec_addon_031_same_input_gives_same_output(self, env):
        first = env["get"]("vid-long", {"start": 500, "end": 5000})
        second = env["get"]("vid-long", {"start": 500, "end": 5000})
        assert first.status_code == second.status_code == 200
        assert first.json()["scenes"]
        assert first.json() == second.json()


class TestExcerpt:
    def test_spec_addon_031_excerpt_is_taken_at_each_scene_start_and_empty_is_null(self, env):
        response = env["get"]("vid-short")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [s["transcript_excerpt"] for s in body["scenes"]] == [
            "near 100.0", None, "near 600.5",
        ]
        assert sorted(env["excerpt_calls"]) == [
            ("vid-short", 100.0, None),
            ("vid-short", 300.0, None),
            ("vid-short", 600.5, None),
        ]

    def test_spec_addon_031_response_has_exactly_the_specified_keys(self, env):
        body = env["get"]("vid-short").json()
        assert set(body) == {"file_id", "duration", "window", "candidate_count", "scenes"}
        for scene in body["scenes"]:
            assert set(scene) == {"start", "end", "transcript_excerpt"}


class TestRejections:
    @pytest.mark.parametrize(
        "params",
        [
            pytest.param({"start": -1}, id="negative-start"),
            pytest.param({"end": -0.5}, id="negative-end"),
            pytest.param({"start": "nan"}, id="nan-start"),
            pytest.param({"end": "inf"}, id="inf-end"),
            pytest.param({"start": "-inf"}, id="minus-inf-start"),
            pytest.param({"start": 10, "end": 10}, id="end-equals-start"),
            pytest.param({"start": 10, "end": 5}, id="end-before-start"),
        ],
    )
    def test_spec_addon_031_invalid_window_is_422(self, env, params):
        response = env["get"]("vid-long", params)
        assert response.status_code == 422, response.text

    def test_spec_addon_031_zero_start_and_tiny_window_are_accepted(self, env):
        assert env["get"]("vid-long", {"start": 0, "end": 0.001}).status_code == 200

    @pytest.mark.parametrize("file_id", ["img-1", "loft-1"])
    def test_spec_addon_031_not_a_video_is_400(self, env, file_id):
        response = env["get"](file_id)
        assert response.status_code == 400
        assert response.json()["detail"] == "Not a video file"

    def test_spec_addon_031_file_in_another_drive_is_404_and_reveals_nothing(self, env):
        assert env["get"]("vid-short").status_code == 200
        response = env["get"]("vid-short", drive="other")
        assert response.status_code == 404
        assert DRIVE not in response.text
        assert "vid-short.mp4" not in response.text
        assert "scenes" not in response.json()

    def test_spec_addon_031_unindexed_file_is_404(self, env):
        response = env["get"]("ghost")
        assert response.status_code == 404
        assert response.json()["detail"] == "File not indexed"


class TestSideEffects:
    def test_spec_addon_031_makes_no_llm_call_and_leaves_visual_index_untouched(
        self, env, monkeypatch,
    ):
        def _no_llm(*_args, **_kwargs):
            raise AssertionError("the scenes endpoint must not resolve an LLM")

        _patch_everywhere(monkeypatch, "resolve", _no_llm)

        with env["Session"]() as s:
            s.add(VideoVisualRun(
                id="vvr_keep", file_id="vid-long", status="succeeded", is_active=True,
                requested_by="manual", priority=100, vision_model="llava:13b",
                pipeline_version=1, candidate_fingerprint="fp",
            ))
            s.commit()

        response = env["get"]("vid-long")
        assert response.status_code == 200, response.text

        with env["Session"]() as s:
            assert s.scalar(select(func.count()).select_from(VideoVisualRun)) == 1
            assert s.scalar(select(func.count()).select_from(VideoVisualScene)) == 0
            run = s.get(VideoVisualRun, "vvr_keep")
            assert (run.status, run.is_active) == ("succeeded", True)


class TestRouteDeclaration:
    def test_spec_addon_031_handler_is_sync_so_it_runs_in_the_thread_pool(self, env):
        routes = [
            r for r in env["app"].routes
            if getattr(r, "path", None) == "/files/{file_id}/scenes"
            and "GET" in getattr(r, "methods", set())
        ]
        assert len(routes) == 1
        assert not asyncio.iscoroutinefunction(routes[0].endpoint)

    def test_spec_addon_031_manifest_declares_the_route(self):
        manifest = json.loads(MANIFEST.read_text())
        entries = [
            r for r in manifest["proxy"]["routes"]
            if r.get("path") == "/files/{file_id}/scenes"
        ]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["methods"] == ["GET"]
        assert entry["pre_check"] == {"type": "file_access", "param": "file_id"}
        assert "drive_optional" not in entry
        assert "addon_feature" not in entry
        assert entry["timeout"] == 30

