"""SPEC-ADDON-078 states and item 7: a YouTube ``.loft`` row is reset, queued and
purged by the sites that already serve a native video, with no change to them.

The rows are a ``.loft`` with scene rows (``clip``) and a thumbnail row
(``clip_thumbnail``) in a real search DB.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import numpy as np
import pytest

import app.database as database
from app.frame_cache import frame_cache_dir
from app.models import IndexedFile
from app.search import _frame_url
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.youtube_frame_doubles import LOFT_MIME

DRIVE = "home"
FILE = "ytloft000001"


def _seed(index, *, clip_indexed=True):
    index.add_file(FILE, drive=DRIVE, mime_type=LOFT_MIME, duration=120.0)
    dim = index.dim("vec_clip")
    for n in range(3):
        emb = f"clip_{FILE}_{n}"
        index.add_embedding(emb, FILE, "clip", "vec_clip")
        index.put_vector("vec_clip", emb, np.eye(dim, dtype=np.float32)[n], "clip")
    thumb = f"clipt_{FILE}"
    index.add_embedding(thumb, FILE, "clip_thumbnail", "vec_clip")
    index.put_vector("vec_clip", thumb, np.eye(dim, dtype=np.float32)[5], "clip_thumbnail")
    with database.get_search_db() as s:
        row = s.query(IndexedFile).filter_by(file_id=FILE).one()
        row.clip_indexed = clip_indexed
        row.clip_thumbnail_indexed = True


def _flags():
    with database.get_search_db_read() as s:
        row = s.query(IndexedFile).filter_by(file_id=FILE).one()
        return row.clip_indexed, row.clip_thumbnail_indexed


def _types(index):
    return sorted(t for _, t, _ in index.embeddings(FILE))


def test_a_vector_dimension_change_drops_the_scene_rows_and_reopens_the_task(
    search_index, monkeypatch,
):
    """The migration resets ``clip_indexed`` and deletes ``clip`` rows whatever the mime."""
    _seed(search_index)
    monkeypatch.setattr(
        database, "_get_clip_embedding_dim", lambda: search_index.dim("vec_clip") + 1,
    )

    with search_index.engine.connect() as conn:
        database._migrate_vec_clip_if_needed(conn)

    assert _types(search_index) == ["clip_thumbnail"]
    assert _flags()[0] is False


def test_the_policy_purge_removes_the_rows_and_the_frames(search_index):
    """``purge_drive`` removes a ``.loft``'s rows, row and cached frames like a video's."""
    from app import purge

    _seed(search_index)
    frames = frame_cache_dir(FILE)
    frames.mkdir(parents=True, exist_ok=True)
    (frames / "1000.large.webp").write_bytes(b"frame")

    assert purge.purge_drive(DRIVE) == 1

    assert search_index.embeddings(FILE) == []
    assert not frames.exists()
    with database.get_search_db_read() as s:
        assert s.query(IndexedFile).filter_by(file_id=FILE).count() == 0


async def test_a_reindex_of_the_clip_task_reopens_it_for_a_loft(search_index):
    """``POST /files/{id}/reindex`` flips ``clip_indexed`` and queues the task."""
    from app.dependencies import get_index_manager
    from app.main import app

    class Manager:
        def __init__(self):
            self.queued: list[tuple[str, str]] = []

        def is_queued(self, file_id, task):
            return False

        async def enqueue_task_for_file(self, file_id, task):
            self.queued.append((file_id, task))

    _seed(search_index)
    manager = Manager()
    app.dependency_overrides[get_index_manager] = lambda: manager
    try:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://intelligence",
        ) as client:
            response = await client.post(
                f"/files/{FILE}/reindex", json={"tasks": ["clip"]},
                headers={"X-Lit-Drive": DRIVE},
            )
    finally:
        app.dependency_overrides.pop(get_index_manager, None)

    assert response.status_code == 200, response.text
    assert response.json()["tasks_reset"] == ["clip"]
    assert manager.queued == [(FILE, "clip")]
    assert _flags() == (False, True)


async def test_resolving_a_failed_clip_job_closes_both_legs_of_a_loft(search_index):
    """The Failed Jobs ``resolve`` action closes ``clip_indexed`` as for a video."""
    from app.main import app

    _seed(search_index, clip_indexed=False)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://intelligence",
    ) as client:
        response = await client.post(
            "/admin/failed-jobs/resolve", json={"file_id": FILE, "job_kind": "clip"},
        )

    assert response.status_code == 200, response.text
    assert _flags() == (True, True)


class TestNoFrameUrlForALoftPassage:
    @pytest.mark.parametrize("kind", ["utterance", "scene", "screen_text"])
    def test_a_loft_passage_has_no_frame_url(self, kind):
        """SPEC-ADDON-078 I10: ``frame_url`` stays unset for every ``.loft`` passage."""
        loft = SimpleNamespace(file_id=FILE, file_type="video", mime_type=LOFT_MIME)

        assert _frame_url(loft, kind, 30.0) is None

    def test_a_native_video_passage_still_has_one(self):
        """The check above would pass for any input if the helper never built a URL."""
        native = SimpleNamespace(file_id="nat000000001", file_type="video", mime_type="video/mp4")

        assert _frame_url(native, "scene", 30.0) is not None
