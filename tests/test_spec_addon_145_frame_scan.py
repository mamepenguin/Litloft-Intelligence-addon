"""SPEC-ADDON-145 item 1 and item 5, SPEC-ADDON-111 I7 and I8 as revised: the frame
scan keeps its 1 fps hashes and writes no change frame.

The scan runs ``index_frame_scan`` over the gradient video of the SPEC-ADDON-111 scan
tests, with the CLIP model replaced by a stand-in that counts its calls.
"""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import text

import app.database as database
from app.frame_cache import frame_cache_dir
from app.models import IndexedFile
from tests.ocr_fixtures import pillow_real  # noqa: F401 - the usefixtures marker resolves it
from tests.search_endpoint_db import patch_everywhere
from tests.test_spec_addon_111_frame_scan_task import media, scan  # noqa: F401 - fixtures
from tests.vec_partition_db import search_index, start  # noqa: F401 - fixture

pytestmark = pytest.mark.usefixtures("pillow_real")

LOFT_MIME = "application/vnd.litloft.loft+json"


def _old_change_frame(index, file_id: str, second: int) -> str:
    emb_id = f"chg_{file_id}_{second}"
    index.add_embedding(emb_id, file_id, "clip_change", "vec_clip_change")
    index.put_vector(
        "vec_clip_change", emb_id, np.ones(index.dim("vec_clip_change"), dtype=np.float32),
        partition="clip_change",
    )
    return emb_id


def _scene_candidate(index, file_id: str, emb_id: str) -> None:
    index.add_embedding(emb_id, file_id, "clip", "vec_clip")
    index.put_vector(
        "vec_clip", emb_id, np.ones(index.dim("vec_clip"), dtype=np.float32), partition="clip",
    )


def test_spec_addon_145_item_1_the_scan_stores_its_hashes_and_writes_no_change_frame(
    scan, media,  # noqa: F811
):
    scan.video(media["video"])
    _old_change_frame(scan.index, "vid1", 3)
    _scene_candidate(scan.index, "vid1", "clip_vid1_1")

    assert scan.run() is True

    assert scan.flag() == 1
    version, seconds, hashes, sizes = scan.scan_row()
    assert 39 <= seconds <= 41
    assert len(hashes) == seconds * 32
    assert scan.embedded == 0
    assert scan.changes() == []
    assert scan.index.vectors("vec_clip_change") == {}
    assert [e[0] for e in scan.index.embeddings("vid1")] == ["clip_vid1_1"]
    assert set(scan.index.vectors("vec_clip")) == {"clip_vid1_1"}
    assert not frame_cache_dir("vid1").exists() or list(frame_cache_dir("vid1").iterdir()) == []


def test_spec_addon_145_revised_111_i7_the_scan_runs_and_sets_its_flag_without_the_clip_model(
    scan, media, monkeypatch,  # noqa: F811
):
    scan.video(media["video"])
    _old_change_frame(scan.index, "vid1", 3)
    attempts: list[int] = []

    def cannot_load(*_args, **_kwargs):
        attempts.append(1)
        raise RuntimeError("the CLIP weights are not on disk")

    patch_everywhere(monkeypatch, "load_image_model", cannot_load)

    assert scan.run() is True

    assert attempts == []
    assert scan.embedded == 0
    assert scan.flag() == 1
    assert scan.scan_row() is not None
    assert scan.changes() == []
    assert scan.index.vectors("vec_clip_change") == {}


def test_spec_addon_145_item_5_a_change_table_of_another_dimension_is_recreated_and_no_flag_moves(
    search_index,  # noqa: F811
):
    """SPEC-ADDON-111 I8 as revised: the table is rebuilt, empty, and nothing is rescanned."""
    for file_id, mime in (("native", "video/mp4"), ("youtube", LOFT_MIME)):
        with database.get_search_db() as s:
            s.add(IndexedFile(
                file_id=file_id, drive="family", filename=file_id,
                file_path=f"/drives/family/{file_id}", file_type="video", mime_type=mime,
                file_size=1, active=True, frame_scan_indexed=True,
            ))
    search_index.sql("UPDATE indexed_files SET loft_source = 'youtube' WHERE file_id = 'youtube'")
    dim = search_index.dim("vec_clip")
    search_index.sql("DROP TABLE vec_clip_change")
    search_index.sql(
        "CREATE VIRTUAL TABLE vec_clip_change USING vec0(embedding_id TEXT PRIMARY KEY, "
        "embedding_type text partition key, vector float[8])"
    )
    search_index.add_embedding("chg_native_3", "native", "clip_change", "vec_clip_change")
    search_index.put_vector("vec_clip_change", "chg_native_3", np.ones(8),
                            partition="clip_change")

    start()

    assert search_index.dim("vec_clip_change") == dim
    assert search_index.vectors("vec_clip_change") == {}
    with search_index.engine.connect() as conn:
        flags = dict(conn.execute(text(
            "SELECT file_id, frame_scan_indexed FROM indexed_files"
        )).fetchall())
    assert flags == {"native": 1, "youtube": 1}
