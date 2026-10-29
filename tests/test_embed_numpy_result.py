"""SPEC-ADDON-034: the scene and image-description workers store what a real
embedder returns, a numpy array of one row per passage, and store nothing for
an empty result."""

from __future__ import annotations

import logging

import numpy as np
import pytest

import app.config as config
import app.database as database
from app.models import VideoVisualRun, VideoVisualScene
from tests.vec_partition_db import search_index, unit  # noqa: F401


def _install_embedder(monkeypatch, fn) -> None:
    import app.workers.embedder as embedder
    from app.workers import video_visual, vision

    monkeypatch.setattr(embedder, "embed_passages", fn)
    for module in (vision, video_visual):
        monkeypatch.setattr(module, "embed_passages", fn, raising=False)
    monkeypatch.setattr(config, "validate_file_path", lambda _path: True)


def _array_embedder(dim: int):
    calls: list[list[str]] = []

    def _embed(texts, *args, **kwargs):
        calls.append(list(texts))
        return np.stack([unit(dim, i + 1) for i, _ in enumerate(texts)])

    _embed.calls = calls
    return _embed


def _run_vision(index) -> str:
    from app.workers import vision

    index.add_file("img1", mime_type="image/jpeg")
    vision._embed_and_store("img1", "A black cat sleeping on a sofa")
    return "img1"


def _run_scene(index) -> str:
    from app.workers import video_visual

    index.add_file("vid1", mime_type="video/mp4", duration=600.0)
    with database.get_search_db() as s:
        s.add(VideoVisualRun(
            id="vvr_1", file_id="vid1", status="running", is_active=False,
            requested_by="manual", priority=100, vision_model="llava:13b",
            pipeline_version=2, candidate_fingerprint="fp", selected_count=1,
        ))
        scene = VideoVisualScene(
            run_id="vvr_1", ordering=0, clip_embedding_id="c0", start_time=5.0,
            status="pending",
        )
        s.add(scene)
        s.flush()
        scene_id = scene.id
    video_visual._embed_scene("vid1", scene_id, "Chicken marinade added", "", 5.0, None)
    return "vid1"


WORKERS = [
    pytest.param(_run_vision, "vision_description", id="vision_description"),
    pytest.param(_run_scene, "video_visual_scene", id="video_visual_scene"),
]


@pytest.mark.parametrize(("run", "etype"), WORKERS)
def test_spec_addon_034_a_one_row_array_is_stored_with_its_vector(
    search_index, monkeypatch, run, etype,
):
    dim = search_index.dim("vec_text")
    embed = _array_embedder(dim)
    _install_embedder(monkeypatch, embed)

    file_id = run(search_index)

    assert len(embed.calls) == 1 and len(embed.calls[0]) == 1
    rows = search_index.embeddings(file_id)
    assert [(t, tb) for _, t, tb in rows] == [(etype, "vec_text")]
    emb_id = rows[0][0]
    keyed = search_index.keyed("vec_text")
    assert keyed[emb_id][0] == etype
    assert keyed[emb_id][1] == unit(dim, 1).astype(np.float32).tobytes()


def _empty_result(dim: int):
    return {
        "none": lambda texts, *a, **k: None,
        "zero_rows": lambda texts, *a, **k: np.empty((0, dim), dtype=np.float32),
    }


@pytest.mark.parametrize("result", ["none", "zero_rows"])
@pytest.mark.parametrize(("run", "etype"), WORKERS)
def test_spec_addon_034_an_empty_result_stores_nothing(
    search_index, monkeypatch, run, etype, result,
):
    dim = search_index.dim("vec_text")
    _install_embedder(monkeypatch, _empty_result(dim)[result])

    file_id = run(search_index)

    assert search_index.embeddings(file_id) == []
    assert search_index.keyed("vec_text") == {}


@pytest.mark.parametrize(("run", "etype"), WORKERS)
def test_spec_addon_034_a_failed_vec_insert_still_commits_the_embedding_row(
    search_index, monkeypatch, caplog, run, etype,
):
    _install_embedder(monkeypatch, _array_embedder(search_index.dim("vec_text")))
    # A missing vec_text table makes only the vector insert fail.
    search_index.sql("DROP TABLE vec_text")

    with caplog.at_level(logging.WARNING):
        file_id = run(search_index)

    assert [(t, tb) for _, t, tb in search_index.embeddings(file_id)] == [
        (etype, "vec_text"),
    ]
    assert any(r.levelno == logging.WARNING for r in caplog.records)
