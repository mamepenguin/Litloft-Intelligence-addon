"""SPEC-ADDON-146 items 1-3, 7, 8, 9, I8, I10, I11 and I13: the ``loft_candidates_done``
flag, its migrations, and which ``.loft``s a fetch is queued for.
"""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import text

import app.database as database
from app.models import IndexedFile
from tests.ocr_fixtures import (  # noqa: F401 - fixtures
    SearchDB,
    build_search_db,
    patch_features,
    pillow_real,
    seed_indexed_file,
)
from tests.test_spec_addon_117_loft_media_queue import _manager, _queued, _reindex
from tests.vec_partition_db import search_index, start  # noqa: F401 - fixture

LOFT_MIME = "application/vnd.litloft.loft+json"
FETCH = "loft_media"
FLAG = "loft_candidates_done"
ALL_DONE = dict(
    metadata_indexed=True,
    clip_indexed=True,
    clip_thumbnail_indexed=True,
    whisper_indexed=True,
    text_indexed=True,
    tfidf_keywords_indexed=True,
    frame_scan_indexed=True,
    frame_ocr_read=True,
    frame_ocr_indexed=True,
)


# ---------------------------------------------------------------------------
# Item 1, I10, I13: the column and the migration that adds it
# ---------------------------------------------------------------------------


def _add(file_id: str, mime: str, source: str | None = None, failures: int = 0) -> None:
    with database.get_search_db() as s:
        s.add(IndexedFile(
            file_id=file_id, drive="family", filename=file_id,
            file_path=f"/drives/family/{file_id}", file_type="video", mime_type=mime,
            file_size=1, active=True,
        ))
    with database._search_engine.begin() as conn:
        conn.execute(text(
            "UPDATE indexed_files SET loft_source = :s, loft_fetch_failures = :n "
            "WHERE file_id = :f"
        ), {"s": source, "n": failures, "f": file_id})


def _flags(engine) -> dict[str, int]:
    with engine.connect() as conn:
        return dict(conn.execute(text(f"SELECT file_id, {FLAG} FROM indexed_files")).fetchall())


def _drop_flag_column(engine) -> None:
    with engine.begin() as conn:
        columns = [r[1] for r in conn.execute(text("PRAGMA table_info(indexed_files)"))]
        if FLAG not in columns:
            return
        for (name,) in conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            f"AND tbl_name = 'indexed_files' AND sql LIKE '%{FLAG}%'"
        )).fetchall():
            conn.execute(text(f"DROP INDEX {name}"))
        conn.execute(text(f"ALTER TABLE indexed_files DROP COLUMN {FLAG}"))


def test_spec_addon_146_i10_the_column_migration_closes_only_other_and_capped_lofts(
    search_index,  # noqa: F811
):
    _add("native", "video/mp4")
    _add("youtube", LOFT_MIME, "youtube")
    _add("youtube_two_failures", LOFT_MIME, "youtube", 2)
    _add("capped", LOFT_MIME, "youtube", 3)
    _add("capped_unclassified", LOFT_MIME, None, 4)
    _add("other", LOFT_MIME, "other")
    _add("unclassified", LOFT_MIME, None)
    _drop_flag_column(search_index.engine)

    start()
    start()

    assert _flags(search_index.engine) == {
        "native": 0, "youtube": 0, "youtube_two_failures": 0, "capped": 1,
        "capped_unclassified": 1, "other": 1, "unclassified": 0,
    }


def test_spec_addon_146_i13_a_row_written_without_the_column_still_inserts(
    search_index,  # noqa: F811
):
    """An older build inserts rows that do not name the column."""
    assert FLAG in [
        r[1] for r in search_index.engine.connect().execute(
            text("PRAGMA table_info(indexed_files)")
        )
    ]
    search_index.sql(
        "INSERT INTO indexed_files(file_id, drive, filename, file_path, file_type, "
        "mime_type, file_size, active, title, description, tags_text, indexed_at, "
        "updated_at) VALUES ('old', 'family', 'a.loft', '/drives/family/a.loft', 'video', "
        ":m, 1, 1, '', '', '', '2026-11-19 10:00:00', '2026-11-19 10:00:00')",
        m=LOFT_MIME,
    )

    assert _flags(search_index.engine)["old"] == 0


# ---------------------------------------------------------------------------
# Item 8, I11: the vec_clip dimension migration
# ---------------------------------------------------------------------------


def test_spec_addon_146_i11_the_vec_clip_dimension_migration_reopens_youtube_and_web_lofts_under_the_cap(
    search_index, monkeypatch,  # noqa: F811
):
    _add("youtube", LOFT_MIME, "youtube")
    _add("web", LOFT_MIME, "web")
    _add("capped", LOFT_MIME, "youtube", 3)
    _add("capped_web", LOFT_MIME, "web", 3)
    _add("other", LOFT_MIME, "other")
    search_index.sql(f"UPDATE indexed_files SET {FLAG} = 1")
    dim = search_index.dim("vec_clip")
    for file_id in ("youtube", "web", "capped", "capped_web"):
        search_index.add_embedding(f"clip_{file_id}_1", file_id, "clip", "vec_clip")
        search_index.put_vector("vec_clip", f"clip_{file_id}_1", np.ones(dim), "clip")
    monkeypatch.setattr(database, "_get_clip_embedding_dim", lambda: dim + 1)

    with search_index.engine.connect() as conn:
        database._migrate_vec_clip_if_needed(conn)

    assert _flags(search_index.engine) == {
        "youtube": 0, "web": 0, "capped": 1, "capped_web": 1, "other": 1,
    }


# ---------------------------------------------------------------------------
# Items 2-3, 9, I8: the sweep and prioritize
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path, monkeypatch) -> SearchDB:
    return build_search_db(tmp_path, monkeypatch)


def _loft(db, tmp_path, file_id: str, source: str | None, *, failures: int = 0,
          candidates: bool = False, **flags) -> None:
    path = tmp_path / f"{file_id}.loft"
    path.write_text('{"provider": "youtube", "url": "https://www.youtube.com/watch?v=x"}')
    seed_indexed_file(db, file_id, LOFT_MIME, path, file_type="video", **{**ALL_DONE, **flags})
    db.execute(
        f"UPDATE indexed_files SET loft_source = :s, loft_fetch_failures = :n, {FLAG} = :c "
        "WHERE file_id = :f",
        s=source, n=failures, c=1 if candidates else 0, f=file_id,
    )


@pytest.mark.usefixtures("pillow_real")
async def test_spec_addon_146_i8_a_fetched_youtube_loft_without_candidates_is_queued_as_backfill(
    db, tmp_path, monkeypatch,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _loft(db, tmp_path, "youtube", "youtube")
    _loft(db, tmp_path, "web", "web")
    _loft(db, tmp_path, "unclassified", None)
    _loft(db, tmp_path, "done", "youtube", candidates=True)
    _loft(db, tmp_path, "capped", "youtube", failures=3)
    _loft(db, tmp_path, "other", "other", candidates=True)
    seed_indexed_file(db, "native", "video/mp4", tmp_path / "native.mp4",
                      file_type="video", **ALL_DONE)
    manager = _manager()

    await manager._resume_incomplete()

    assert sorted(await _queued(manager, FETCH)) == [
        ("unclassified", -10), ("web", -10), ("youtube", -10),
    ]
    assert await _queued(manager, "frame_scan") == []
    assert await _queued(manager, "clip") == []


@pytest.mark.usefixtures("pillow_real")
async def test_spec_addon_146_i8_the_sweep_closes_the_flag_of_a_loft_of_source_other(
    db, tmp_path, monkeypatch,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _loft(db, tmp_path, "other", "other", frame_scan_indexed=False)
    _loft(db, tmp_path, "youtube", "youtube", candidates=True)
    manager = _manager()

    await manager._resume_incomplete()

    assert bool(db.flag("other", FLAG)) is True
    assert await _queued(manager, FETCH) == []


@pytest.mark.usefixtures("pillow_real")
async def test_spec_addon_146_i8_prioritize_queues_a_fetch_at_100_when_only_the_candidates_are_wanted(
    db, tmp_path, monkeypatch,
):
    """The frame text is read but not indexed, which alone queues the rebuild-only task."""
    patch_features(monkeypatch, video_frame_ocr="manual")
    _loft(db, tmp_path, "yt1", "youtube", frame_ocr_read=True, frame_ocr_indexed=False)
    _loft(db, tmp_path, "web1", "web", frame_ocr_read=True, frame_ocr_indexed=False)
    manager = _manager()

    await manager.prioritize("yt1")
    await manager.prioritize("web1")

    assert await _queued(manager, FETCH) == [("yt1", 100), ("web1", 100)]
    assert await _queued(manager, "frame_ocr") == [("yt1", 100), ("web1", 100)]


# ---------------------------------------------------------------------------
# Item 7, I7: a CLIP regenerate reopens the flag
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("pillow_real")
@pytest.mark.parametrize("source", ["youtube", "other", None])
async def test_spec_addon_146_i7_a_clip_regenerate_of_a_loft_sets_the_flag_false(
    db, tmp_path, monkeypatch, source,
):
    patch_features(monkeypatch, video_frame_ocr="on_index")
    _loft(db, tmp_path, "loft1", source, failures=3, candidates=True)
    manager = _manager()

    await _reindex(manager, monkeypatch, "loft1", "whisper")
    after_whisper = bool(db.flag("loft1", FLAG))
    await _reindex(manager, monkeypatch, "loft1", "clip")

    assert after_whisper is True
    assert bool(db.flag("loft1", FLAG)) is False
    assert db.flag("loft1", "loft_source") is None
    assert await _queued(manager, FETCH) == [("loft1", 100)]
