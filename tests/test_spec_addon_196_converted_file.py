"""SPEC-ADDON-196: a ``.loft`` converted into the media file it points at.

Intelligence syncs the file's mime through ``handle_files_moved`` (the
``files-moved`` webhook and reconcile's drift repair). When the stored mime was
the ``.loft`` mime and core's is not, the sync reopens exactly the stages the
``.loft`` closed without storing anything, keeps every stage that has content,
and queues the reopened ones as it would for a native video. A ``.loft`` task
still running at the sync writes nothing.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

import app.indexer  # noqa: E402,F401 - imports its writers before fixtures patch the database module
from app.config import FeaturesConfig  # noqa: E402
from app.database import Base  # noqa: E402
from app.models import (  # noqa: E402
    Embedding,
    IndexedFile,
    TranscriptChunk,
    VideoFrameScan,
    VideoFrameText,
    VideoVisualRun,
    VideoVisualScene,
)
from tests.vec_partition_db import search_index  # noqa: E402,F401 - fixture
from tests.youtube_frame_doubles import (  # noqa: E402,F401 - fixtures
    needs_ffmpeg,
    youtube,
)

LOFT_MIME = "application/vnd.litloft.loft+json"
VIDEO_MIME = "video/mp4"
DRIVE = "drive1"
FILE_ID = "f1"
TITLE = "Title"
INFO_SKIP = "loft task skipped, no longer a .loft (file {file_id})"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def search_engine(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_files "
            "USING fts5(file_id UNINDEXED, filename, title, description, tags_text)"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_files_word "
            "USING fts5(file_id UNINDEXED, filename, title, description, tags_text)"
        ))
    return engine


@pytest.fixture()
def litloft_engine(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'litloft.db'}",
        connect_args={"check_same_thread": False},
    )
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE files ("
            "  id TEXT PRIMARY KEY,"
            "  drive TEXT NOT NULL,"
            "  file_path TEXT NOT NULL,"
            "  filename TEXT NOT NULL,"
            "  title TEXT,"
            "  folder_path TEXT,"
            "  description TEXT,"
            "  file_size INTEGER,"
            "  file_type TEXT,"
            "  mime_type TEXT,"
            "  duration REAL,"
            "  thumbnail_path TEXT,"
            "  deleted_at TEXT,"
            "  missing_since TEXT"
            ")"
        ))
    return engine


@pytest.fixture()
def dbs(monkeypatch, search_engine, litloft_engine):
    SearchSession = sessionmaker(bind=search_engine, expire_on_commit=False)
    LitloftSession = sessionmaker(bind=litloft_engine, expire_on_commit=False)

    @contextmanager
    def _get_search_db():
        s = SearchSession()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    @contextmanager
    def _get_search_db_read():
        s = SearchSession()
        try:
            yield s
        finally:
            s.close()

    @contextmanager
    def _get_litloft_db():
        s = LitloftSession()
        try:
            yield s
        finally:
            s.close()

    monkeypatch.setattr("app.database.get_search_db", _get_search_db)
    monkeypatch.setattr("app.indexer.get_search_db", _get_search_db)
    monkeypatch.setattr("app.database.get_search_db_read", _get_search_db_read)
    monkeypatch.setattr("app.indexer.get_search_db_read", _get_search_db_read)
    monkeypatch.setattr("app.database.get_litloft_db", _get_litloft_db)
    monkeypatch.setattr("app.indexer.get_litloft_db", _get_litloft_db)
    monkeypatch.setattr(
        "app.indexer.resolve_file_path",
        lambda drive, relative: f"/drives/{drive}/{relative}",
    )
    return SearchSession, LitloftSession


@pytest.fixture()
def policy_on(monkeypatch):
    async def _enabled(_drive, _feature):
        return True

    monkeypatch.setattr("app.policy_client.is_feature_enabled", _enabled)


@pytest.fixture()
def manager():
    from app.indexer import IndexManager

    m = IndexManager.__new__(IndexManager)
    m.queued = []

    async def _enqueue(task):
        m.queued.append(task)

    m._enqueue = _enqueue
    return m


def _seed_indexed(SearchSession, **overrides):
    values = dict(
        file_id=FILE_ID,
        drive=DRIVE,
        filename=f"{TITLE}.loft",
        file_path=f"/drives/{DRIVE}/{TITLE}.loft",
        file_type="video",
        mime_type=LOFT_MIME,
        file_size=1,
        active=True,
        metadata_indexed=True,
        clip_indexed=True,
        clip_thumbnail_indexed=True,
        whisper_indexed=True,
        text_indexed=True,
        ocr_indexed=True,
        page_ocr_indexed=True,
        tfidf_keywords_indexed=True,
        frame_scan_indexed=True,
        frame_ocr_read=True,
        frame_ocr_indexed=True,
        frame_ocr_rule=1,
        title=TITLE,
        description="",
        tags_text="",
        loft_source="youtube",
        loft_fetch_failures=2,
        loft_candidates_done=False,
        idf_terms_indexed=True,
        idf_terms_gen=1,
    )
    values.update(overrides)
    s = SearchSession()
    try:
        s.add(IndexedFile(**values))
        s.commit()
    finally:
        s.close()


def _seed_core(LitloftSession, **overrides):
    values = dict(
        id=FILE_ID,
        drive=DRIVE,
        file_path=f"{TITLE}.mp4",
        filename=f"{TITLE}.mp4",
        title=TITLE,
        folder_path="",
        description="",
        file_size=1,
        file_type="video",
        mime_type=VIDEO_MIME,
        duration=None,
        thumbnail_path=None,
        deleted_at=None,
        missing_since=None,
    )
    values.update(overrides)
    s = LitloftSession()
    try:
        s.execute(text(
            "INSERT INTO files (id, drive, file_path, filename, title, folder_path, "
            "description, file_size, file_type, mime_type, duration, "
            "thumbnail_path, deleted_at, missing_since) "
            "VALUES (:id, :drive, :file_path, :filename, :title, :folder_path, "
            ":description, :file_size, :file_type, :mime_type, :duration, "
            ":thumbnail_path, :deleted_at, :missing_since)"
        ), values)
        s.commit()
    finally:
        s.close()


def _add_transcript(SearchSession, file_id=FILE_ID):
    s = SearchSession()
    try:
        s.add(TranscriptChunk(
            file_id=file_id, chunk_index=0, text="spoken", language="en",
            timestamp_start=0.0, timestamp_end=1.0,
        ))
        s.commit()
    finally:
        s.close()


def _add_scan(SearchSession, file_id=FILE_ID):
    s = SearchSession()
    try:
        s.add(VideoFrameScan(
            file_id=file_id, seconds=1, hashes=b"\x00" * 32, sizes=b"\x00",
        ))
        s.commit()
    finally:
        s.close()


def _add_frame_text(SearchSession, file_id=FILE_ID):
    s = SearchSession()
    try:
        s.add(VideoFrameText(
            file_id=file_id, chunk_index=0, start_time=0.0, end_time=1.0,
            text="on screen",
        ))
        s.commit()
    finally:
        s.close()


def _add_embedding(SearchSession, file_id=FILE_ID, embedding_type="clip", suffix="0"):
    s = SearchSession()
    try:
        s.add(Embedding(
            id=f"{embedding_type}_{file_id}_{suffix}"[:32],
            file_id=file_id,
            embedding_type=embedding_type,
            vector_table="vec_clip",
            content_preview="x",
        ))
        s.commit()
    finally:
        s.close()


def _add_visual_scene(SearchSession, file_id=FILE_ID):
    s = SearchSession()
    try:
        s.add(VideoVisualRun(
            id="run1", file_id=file_id, status="succeeded", is_active=True,
            requested_by="manual", priority=0, vision_model="test",
            pipeline_version=1, candidate_fingerprint="fp",
            selected_count=1, completed_count=1, succeeded_count=1,
            failed_count=0, created_at=datetime.now(UTC),
        ))
        s.add(VideoVisualScene(
            run_id="run1", ordering=0, clip_embedding_id="clip_f1_0",
            start_time=0.0, end_time=1.0, status="succeeded",
        ))
        s.commit()
    finally:
        s.close()


def _value(SearchSession, column, file_id=FILE_ID):
    s = SearchSession()
    try:
        return getattr(
            s.query(IndexedFile).filter_by(file_id=file_id).one(), column,
        )
    finally:
        s.close()


def _count(SearchSession, table, file_id=FILE_ID):
    s = SearchSession()
    try:
        return s.execute(
            text(f"SELECT COUNT(*) FROM {table} WHERE file_id = :f"),
            {"f": file_id},
        ).scalar_one()
    finally:
        s.close()


def _sync(manager, file_ids=(FILE_ID,)):
    import asyncio

    asyncio.run(manager.handle_files_moved(list(file_ids)))


# ---------------------------------------------------------------------------
# I29: the reopen
# ---------------------------------------------------------------------------


def test_spec_addon_196_i29_a_conversion_reopens_exactly_the_empty_stages(
    dbs, policy_on, manager, caplog,
):
    SearchSession, LitloftSession = dbs
    _seed_indexed(SearchSession)
    _seed_core(LitloftSession)

    _sync(manager)

    assert _value(SearchSession, "mime_type") == VIDEO_MIME
    assert _value(SearchSession, "whisper_indexed") is False
    assert _value(SearchSession, "frame_scan_indexed") is False
    assert _value(SearchSession, "frame_ocr_read") is False
    assert _value(SearchSession, "frame_ocr_indexed") is False
    assert _value(SearchSession, "clip_indexed") is False
    # The sync's own fields and every other flag are as they were.
    assert _value(SearchSession, "metadata_indexed") is True
    assert _value(SearchSession, "text_indexed") is True
    assert _value(SearchSession, "clip_thumbnail_indexed") is True
    assert _value(SearchSession, "tfidf_keywords_indexed") is True
    assert _value(SearchSession, "ocr_indexed") is True
    assert _value(SearchSession, "page_ocr_indexed") is True
    assert _value(SearchSession, "transcript_corrected") is False
    assert _value(SearchSession, "picture_text_corrected") is False
    assert _value(SearchSession, "loft_source") == "youtube"
    assert _value(SearchSession, "loft_fetch_failures") == 2
    assert _value(SearchSession, "loft_candidates_done") is False
    assert _value(SearchSession, "filename") == f"{TITLE}.mp4"
    assert _value(SearchSession, "file_path") == f"/drives/{DRIVE}/{TITLE}.mp4"


def test_spec_addon_196_i29_a_stage_already_false_stays_false(dbs, policy_on, manager):
    SearchSession, LitloftSession = dbs
    _seed_indexed(
        SearchSession,
        whisper_indexed=False,
        frame_scan_indexed=False,
        frame_ocr_read=False,
        frame_ocr_indexed=False,
        clip_indexed=False,
    )
    _seed_core(LitloftSession)

    _sync(manager)

    for column in (
        "whisper_indexed", "frame_scan_indexed", "frame_ocr_read",
        "frame_ocr_indexed", "clip_indexed",
    ):
        assert _value(SearchSession, column) is False


# ---------------------------------------------------------------------------
# I30: content is kept, corrections are kept
# ---------------------------------------------------------------------------


def test_spec_addon_196_i30_a_stage_with_content_keeps_its_flag_rows_and_vectors(
    dbs, policy_on, manager,
):
    SearchSession, LitloftSession = dbs
    _seed_indexed(SearchSession)
    _seed_core(LitloftSession)
    _add_transcript(SearchSession)
    _add_scan(SearchSession)
    _add_frame_text(SearchSession)
    _add_embedding(SearchSession, embedding_type="clip", suffix="0")
    _add_embedding(SearchSession, embedding_type="clip_thumbnail", suffix="t")
    _add_embedding(SearchSession, embedding_type="whisper", suffix="w")
    _add_visual_scene(SearchSession)

    _sync(manager)

    for column in (
        "whisper_indexed", "frame_scan_indexed", "frame_ocr_read",
        "frame_ocr_indexed", "clip_indexed", "clip_thumbnail_indexed",
    ):
        assert _value(SearchSession, column) is True
    assert _count(SearchSession, "transcript_chunks") == 1
    assert _count(SearchSession, "video_frame_scans") == 1
    assert _count(SearchSession, "video_frame_texts") == 1
    assert _count(SearchSession, "embeddings") == 3
    s = SearchSession()
    try:
        types = sorted(
            r[0] for r in s.execute(text(
                "SELECT embedding_type FROM embeddings WHERE file_id = :f"
            ), {"f": FILE_ID})
        )
        scenes = s.execute(text(
            "SELECT COUNT(*) FROM video_visual_scenes v JOIN video_visual_runs r "
            "ON r.id = v.run_id WHERE r.file_id = :f"
        ), {"f": FILE_ID}).scalar_one()
    finally:
        s.close()
    assert types == ["clip", "clip_thumbnail", "whisper"]
    assert scenes == 1


def test_spec_addon_196_i30_a_corrected_kind_is_never_reopened(dbs, policy_on, manager):
    SearchSession, LitloftSession = dbs
    _seed_indexed(
        SearchSession,
        transcript_corrected=True,
        picture_text_corrected=True,
    )
    _seed_core(LitloftSession)

    _sync(manager)

    assert _value(SearchSession, "whisper_indexed") is True
    assert _value(SearchSession, "frame_ocr_read") is True
    assert _value(SearchSession, "frame_ocr_indexed") is True
    # The kinds that hold no corrected text still reopen.
    assert _value(SearchSession, "frame_scan_indexed") is False
    assert _value(SearchSession, "clip_indexed") is False


# ---------------------------------------------------------------------------
# I31: no reopen on any other change, the sync's other effects
# ---------------------------------------------------------------------------


_MIME_TABLE = [
    pytest.param(
        "video/mp4", "audio/mp4", "Title.m4a", id="other-mime-change",
    ),
    pytest.param(LOFT_MIME, LOFT_MIME, "Renamed.loft", id="loft-rename"),
    pytest.param("video/mp4", "video/mp4", "Title.mp4", id="no-change"),
]


@pytest.mark.parametrize(("stored_mime", "core_mime", "core_filename"), _MIME_TABLE)
def test_spec_addon_196_i31_other_mime_changes_reopen_nothing(
    dbs, policy_on, manager, stored_mime, core_mime, core_filename,
):
    SearchSession, LitloftSession = dbs
    _seed_indexed(SearchSession, mime_type=stored_mime)
    _seed_core(LitloftSession, filename=core_filename, file_path=core_filename,
               mime_type=core_mime)

    _sync(manager)

    for column in (
        "whisper_indexed", "frame_scan_indexed", "frame_ocr_read",
        "frame_ocr_indexed", "clip_indexed", "clip_thumbnail_indexed",
        "metadata_indexed", "text_indexed",
    ):
        assert _value(SearchSession, column) is True
    assert _value(SearchSession, "mime_type") == core_mime
    assert _value(SearchSession, "filename") == core_filename


def test_spec_addon_196_i31_a_filename_change_still_marks_the_idf_terms_stale(
    dbs, policy_on, manager,
):
    SearchSession, LitloftSession = dbs
    _seed_indexed(SearchSession)
    _seed_core(LitloftSession)

    _sync(manager)

    assert _value(SearchSession, "idf_terms_indexed") is False
    assert _value(SearchSession, "idf_terms_gen") == 2


def test_spec_addon_196_i31_the_thumbnail_leg_repair_and_its_enqueue_are_unchanged(
    dbs, policy_on, manager, monkeypatch, tmp_path,
):
    from app import indexer

    SearchSession, LitloftSession = dbs
    thumbnails = tmp_path / "thumbnails"
    (thumbnails / DRIVE).mkdir(parents=True)
    (thumbnails / DRIVE / f"{FILE_ID}.jpg").write_bytes(b"jpeg")
    monkeypatch.setenv("HOMEVAULT_THUMBNAILS_DIR", str(thumbnails))
    _seed_indexed(SearchSession, thumbnail_path=DRIVE + f"/{FILE_ID}.jpg")
    _seed_core(
        LitloftSession, file_path=f"{TITLE}.mp4",
        thumbnail_path=DRIVE + f"/{FILE_ID}.jpg",
    )

    _sync(manager)

    assert _value(SearchSession, "clip_thumbnail_indexed") is False
    assert [t.task_type for t in manager.queued] == [indexer.TaskType.CLIP]
    assert [t.file_id for t in manager.queued] == [FILE_ID]


def test_spec_addon_196_i31_a_sync_that_raises_reopens_nothing_and_the_next_try_does(
    dbs, policy_on, manager, monkeypatch,
):
    from app import indexer

    SearchSession, LitloftSession = dbs
    _seed_indexed(SearchSession)
    _seed_core(LitloftSession)
    real_upsert = indexer.upsert_fts_file
    calls = []

    def _fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("fts write failed")
        return real_upsert(*args, **kwargs)

    monkeypatch.setattr(indexer, "upsert_fts_file", _fail_once)

    with pytest.raises(RuntimeError):
        _sync(manager)

    assert _value(SearchSession, "mime_type") == LOFT_MIME
    assert _value(SearchSession, "whisper_indexed") is True
    assert _value(SearchSession, "clip_indexed") is True

    _sync(manager)

    assert _value(SearchSession, "mime_type") == VIDEO_MIME
    assert _value(SearchSession, "whisper_indexed") is False
    assert _value(SearchSession, "clip_indexed") is False


# ---------------------------------------------------------------------------
# I29 through reconcile's drift repair
# ---------------------------------------------------------------------------


@pytest.fixture()
def reconcile_manager(manager, monkeypatch):
    manager._reset_loft_refs_with_new_vtt = lambda: None
    return manager


def test_spec_addon_196_i29_reconcile_drift_repair_reopens_the_empty_stages(
    dbs, policy_on, reconcile_manager,
):
    SearchSession, LitloftSession = dbs
    # A same-name replace emits no files.moved: the mime is the only drift.
    _seed_indexed(
        SearchSession,
        filename=f"{TITLE}.mp4",
        file_path=f"/drives/{DRIVE}/{TITLE}.mp4",
    )
    _seed_core(LitloftSession)

    import asyncio
    result = asyncio.run(reconcile_manager.reconcile())

    assert result.get("drift_repaired") == 1
    assert _value(SearchSession, "mime_type") == VIDEO_MIME
    assert _value(SearchSession, "whisper_indexed") is False
    assert _value(SearchSession, "frame_scan_indexed") is False
    assert _value(SearchSession, "clip_indexed") is False


def test_spec_addon_196_i31_reconcile_reopens_once_the_policy_turns_on(
    dbs, monkeypatch, reconcile_manager,
):
    SearchSession, LitloftSession = dbs
    _seed_indexed(SearchSession)
    _seed_core(LitloftSession)

    async def _off(_drive, _feature):
        return False

    monkeypatch.setattr("app.policy_client.is_feature_enabled", _off)
    import asyncio
    asyncio.run(reconcile_manager.reconcile())
    assert _value(SearchSession, "mime_type") == LOFT_MIME
    assert _value(SearchSession, "whisper_indexed") is True

    async def _on(_drive, _feature):
        return True

    monkeypatch.setattr("app.policy_client.is_feature_enabled", _on)
    asyncio.run(reconcile_manager.reconcile())
    assert _value(SearchSession, "mime_type") == VIDEO_MIME
    assert _value(SearchSession, "whisper_indexed") is False


# ---------------------------------------------------------------------------
# I32: the reopened stages are queued by the resume path
# ---------------------------------------------------------------------------


def test_spec_addon_196_i32_reopened_stages_are_queued_as_for_a_native_video(
    dbs, policy_on, manager, monkeypatch, make_settings,
):
    from app import indexer

    SearchSession, LitloftSession = dbs
    _seed_indexed(SearchSession)
    _seed_core(LitloftSession)
    monkeypatch.setattr(
        indexer, "settings",
        make_settings(features=FeaturesConfig(video_frame_ocr="on_index")),
    )
    _sync(manager)

    import asyncio
    asyncio.run(manager._resume_incomplete())

    types = [t.task_type for t in manager.queued]
    assert indexer.TaskType.WHISPER in types
    assert indexer.TaskType.CLIP in types
    assert indexer.TaskType.FRAME_SCAN in types
    assert indexer.TaskType.FRAME_OCR in types
    assert indexer.TaskType.LOFT_MEDIA not in types
    assert {t.file_id for t in manager.queued} == {FILE_ID}


# ---------------------------------------------------------------------------
# I37: a .loft task whose writes come after the sync
# ---------------------------------------------------------------------------


@pytest.fixture()
def loft_world(search_index, monkeypatch, tmp_path, make_settings):
    from tests.loft_media_world import LoftWorld

    return LoftWorld(search_index, None, monkeypatch, tmp_path, make_settings)


def test_spec_addon_196_i37_a_loft_fetch_after_the_sync_writes_nothing(
    loft_world, monkeypatch, caplog,
):
    from app.workers import loft_media

    world = loft_world
    world.add_loft(
        FILE_ID,
        frame_scan_indexed=True,
        frame_ocr_read=True,
        frame_ocr_indexed=True,
        loft_candidates_done=True,
    )

    def _read_loft(_path, _drive):
        # The sync lands after the reference was read, before the first write.
        world.set(
            FILE_ID, mime_type=VIDEO_MIME, filename=f"{TITLE}.mp4",
            file_path=str(world.drive_dir / f"{TITLE}.mp4"),
        )
        return SimpleNamespace(provider="youtube", url="https://example.test/v")

    monkeypatch.setattr(loft_media.youtube_source, "read_loft", _read_loft)

    from tests.loft_media_world import fetch
    with caplog.at_level(logging.INFO):
        fetch(FILE_ID)

    assert world.source(FILE_ID) is None
    assert world.flags(FILE_ID) == (True, True, True)
    assert bool(world.value(FILE_ID, "loft_candidates_done")) is True
    assert world.failures(FILE_ID) == 0
    assert world.value(FILE_ID, "duration") is None
    assert world.scan_rows(FILE_ID) == 0
    assert world.fetch_dir_files() == []
    messages = [r.getMessage() for r in caplog.records]
    assert messages.count(INFO_SKIP.format(file_id=FILE_ID)) == 1


def test_spec_addon_196_i37_the_adjacent_vtt_store_after_the_sync_writes_nothing(
    search_index, tmp_path, monkeypatch, caplog, make_settings,
):
    import asyncio

    from app.workers import whisper

    folder = tmp_path / "drive"
    folder.mkdir()
    loft = folder / f"{TITLE}.loft"
    loft.write_text("{}", encoding="utf-8")
    (folder / f"{TITLE}.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nthe transcript of the talk\n",
        encoding="utf-8",
    )
    search_index.add_file(FILE_ID, drive=DRIVE, mime_type=LOFT_MIME,
                          file_path=str(loft))
    search_index.sql(
        "UPDATE indexed_files SET whisper_indexed = 0 WHERE file_id = :f",
        f=FILE_ID,
    )

    def _sync_then_no_temp(_file_path):
        # The .loft branch was chosen by the mime read at the top of the task;
        # the sync lands before the store's transaction.
        search_index.sql(
            "UPDATE indexed_files SET mime_type = :m, filename = :n, "
            "file_path = :p WHERE file_id = :f",
            m=VIDEO_MIME, n=f"{TITLE}.mp4", p=str(folder / f"{TITLE}.mp4"),
            f=FILE_ID,
        )
        return None

    monkeypatch.setattr(whisper, "_loft_stt_temp_path", _sync_then_no_temp)
    monkeypatch.setattr(whisper, "settings", make_settings())
    text_dim = search_index.dim("vec_text")
    monkeypatch.setattr(
        whisper, "embed_passages",
        lambda texts: [
            np.ones(text_dim, dtype=np.float32) for _ in texts
        ],
    )
    emitted = []
    monkeypatch.setattr(
        whisper, "emit_processing_updated",
        lambda drive, ids: emitted.append((drive, list(ids))),
    )

    with caplog.at_level(logging.INFO):
        asyncio.run(whisper.index_whisper(FILE_ID))

    assert search_index.embeddings(FILE_ID) == []
    assert _count_scalar(search_index, "transcript_chunks") == 0
    assert bool(_index_value(search_index, "whisper_indexed")) is False
    assert emitted == []
    messages = [r.getMessage() for r in caplog.records]
    assert messages.count(INFO_SKIP.format(file_id=FILE_ID)) == 1


def test_spec_addon_196_i37_the_consumed_stt_temp_mark_after_the_sync_writes_nothing(
    search_index, tmp_path, monkeypatch,
):
    from app.workers import whisper

    search_index.add_file(
        FILE_ID, drive=DRIVE, mime_type=LOFT_MIME,
        file_path=str(tmp_path / f"{TITLE}.loft"),
    )
    search_index.sql(
        "UPDATE indexed_files SET mime_type = :m WHERE file_id = :f",
        m=VIDEO_MIME, f=FILE_ID,
    )
    emitted = []
    monkeypatch.setattr(
        whisper, "emit_processing_updated",
        lambda drive, ids: emitted.append((drive, list(ids))),
    )

    whisper._mark_loft_stt_temp_consumed(FILE_ID)

    assert bool(_index_value(search_index, "whisper_indexed")) is False
    assert emitted == []


def _count_scalar(index, table) -> int:
    with index.engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()


def _index_value(index, column, file_id=FILE_ID):
    with index.engine.connect() as conn:
        return conn.execute(
            text(f"SELECT {column} FROM indexed_files WHERE file_id = :f"),
            {"f": file_id},
        ).scalar_one()


# ---------------------------------------------------------------------------
# I37: a conversion during the slow steps of a fetch, before each store
# ---------------------------------------------------------------------------

VIDEO_SECONDS = 40.0


@pytest.fixture(scope="module")
def loft_video(tmp_path_factory):
    from tests.test_spec_addon_111_frame_scan_task import _encode

    return _encode(tmp_path_factory.mktemp("loft-converted") / "rendition.mp4")


@pytest.fixture()
def real_pillow(monkeypatch):
    """Real Pillow, in ``sys.modules`` and wherever an app module holds it."""
    from tests.search_endpoint_db import patch_everywhere
    from tests.youtube_frame_doubles import real_pil_modules

    real = real_pil_modules()
    for name, module in real.items():
        monkeypatch.setitem(sys.modules, name, module)
    patch_everywhere(monkeypatch, "Image", real["PIL.Image"])
    return real["PIL.Image"]


@pytest.fixture()
def fetch_world(search_index, youtube, monkeypatch, tmp_path, make_settings,
                real_pillow):
    from tests.loft_media_world import LoftWorld

    return LoftWorld(search_index, youtube, monkeypatch, tmp_path, make_settings)


@pytest.fixture()
def fresh_loft_state():
    from app.workers import loft_media

    loft_media.reset_worker_state()
    yield
    loft_media.reset_worker_state()


def _sync_to_video(world):
    """The conversion's ``files.moved`` sync: the row is now the media file."""
    world.set(
        FILE_ID, mime_type=VIDEO_MIME, filename=f"{TITLE}.mp4",
        file_path=str(world.drive_dir / f"{TITLE}.mp4"),
    )


def _answer(world, video, *, board=None):
    """The lookup's answer: one downloadable rendition, and maybe a storyboard."""
    from tests.youtube_frame_doubles import SIGNED_TOKEN, describe, video_format

    path = "/v/rendition"
    world.youtube.server.files[path] = video.read_bytes()
    formats = [video_format(
        "rendition", 720, url=f"{world.youtube.base}{path}?sig={SIGNED_TOKEN}",
    )]
    if board is not None:
        formats.append(board)
    world.youtube.set_ytdlp(describe=describe(formats, duration=VIDEO_SECONDS))


def _board(world, pil):
    from tests.test_spec_addon_078_youtube_storyboard import storyboard_fps
    from tests.youtube_frame_doubles import storyboard_format

    world.youtube.server.files["/sb/sheet.png"] = _sheet(pil)
    return storyboard_format(
        world.youtube.base, [("sheet.png", VIDEO_SECONDS)],
        rows=3, columns=3, tile=(320, 180),
        fps=storyboard_fps(VIDEO_SECONDS),
    )


def _sheet(png) -> bytes:
    import io

    image = png.new("RGB", (960, 540), (30, 60, 90))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _assert_fetch_wrote_nothing(world, caplog):
    assert world.failures(FILE_ID) == 0
    assert world.fetch_dir_files() == []
    messages = [r.getMessage() for r in caplog.records]
    assert messages.count(INFO_SKIP.format(file_id=FILE_ID)) == 1


@needs_ffmpeg
def test_spec_addon_196_i37_the_storyboard_store_after_the_sync_writes_nothing(
    fetch_world, loft_video, real_pillow, monkeypatch, caplog, fresh_loft_state,
):
    from app.workers import loft_media
    from tests.loft_media_world import fetch

    world = fetch_world
    _answer(world, loft_video, board=_board(world, real_pillow))
    world.add_loft(FILE_ID, duration=VIDEO_SECONDS, loft_candidates_done=True)

    real_lookup = loft_media.youtube_source.run_lookup

    def _lookup_then_sync(*args, **kwargs):
        answer = real_lookup(*args, **kwargs)
        _sync_to_video(world)
        return answer

    monkeypatch.setattr(loft_media.youtube_source, "run_lookup", _lookup_then_sync)

    with caplog.at_level(logging.INFO):
        fetch(FILE_ID)

    assert world.index.embeddings(FILE_ID) == []
    assert bool(world.value(FILE_ID, "clip_indexed")) is False
    assert world.scan_rows(FILE_ID) == 0
    _assert_fetch_wrote_nothing(world, caplog)


@needs_ffmpeg
def test_spec_addon_196_i37_the_scan_store_after_the_sync_writes_nothing(
    fetch_world, loft_video, real_pillow, monkeypatch, caplog, fresh_loft_state,
):
    from app.workers import loft_media
    from tests.loft_media_world import fetch

    world = fetch_world
    _answer(world, loft_video)
    world.add_loft(FILE_ID, loft_candidates_done=True)

    real_download = loft_media.youtube_source.download_rendition

    def _download_then_sync(*args, **kwargs):
        outcome = real_download(*args, **kwargs)
        _sync_to_video(world)
        return outcome

    monkeypatch.setattr(
        loft_media.youtube_source, "download_rendition", _download_then_sync,
    )

    with caplog.at_level(logging.INFO):
        fetch(FILE_ID)

    assert world.scan_rows(FILE_ID) == 0
    assert world.flags(FILE_ID)[0] is False
    _assert_fetch_wrote_nothing(world, caplog)


@needs_ffmpeg
def test_spec_addon_196_i37_the_frame_read_store_after_the_sync_writes_nothing(
    fetch_world, loft_video, real_pillow, monkeypatch, caplog, fresh_loft_state,
):
    from app.workers import loft_media
    from tests.loft_media_world import fetch

    world = fetch_world
    world.configure(frame_ocr="on_index")
    _answer(world, loft_video)
    world.add_loft(
        FILE_ID,
        loft_candidates_done=True,
        frame_scan_indexed=True,
        frame_ocr_read=False,
        frame_ocr_indexed=False,
    )

    real_download = loft_media.youtube_source.download_rendition

    def _download_then_sync(*args, **kwargs):
        outcome = real_download(*args, **kwargs)
        _sync_to_video(world)
        return outcome

    monkeypatch.setattr(
        loft_media.youtube_source, "download_rendition", _download_then_sync,
    )

    with caplog.at_level(logging.INFO):
        fetch(FILE_ID)

    assert world.frame_texts(FILE_ID) == []
    assert world.flags(FILE_ID) == (True, False, False)
    _assert_fetch_wrote_nothing(world, caplog)


# ---------------------------------------------------------------------------
# I37: the discarded transcription's JobRecord is finished
# ---------------------------------------------------------------------------


class _LoftProvider:
    """A provider that converts the file while its transcription is in flight."""

    name = "fake"

    def __init__(self, sync):
        self.capabilities = SimpleNamespace(
            handles_own_retry=True,
            sends_audio_offhost=False,
            supports_word_timestamps=True,
        )
        self._sync = sync

    async def transcribe(self, *_args, **_kwargs):
        self._sync()
        return [SimpleNamespace(
            text="hello world", start=0.0, end=1.0, language="en",
            words=[
                SimpleNamespace(text="hello", start=0.0, end=0.5, speaker_id=None),
                SimpleNamespace(text="world", start=0.5, end=1.0, speaker_id=None),
            ],
        )]


def _job_rows(index):
    with index.engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(text(
            "SELECT status, error_class, completed_at FROM job_records ORDER BY id"
        ))]


def test_spec_addon_196_i37_a_discarded_stt_transcription_finishes_its_job(
    search_index, tmp_path, monkeypatch, make_settings,
):
    import asyncio

    from app.workers import whisper

    folder = tmp_path / "drive"
    folder.mkdir()
    loft = folder / f"{TITLE}.loft"
    loft.write_text("{}", encoding="utf-8")
    (folder / f"{TITLE}{whisper.LOFT_STT_TEMP_SUFFIX}").write_bytes(b"audio")
    search_index.add_file(
        FILE_ID, drive=DRIVE, mime_type=LOFT_MIME, file_path=str(loft),
    )
    search_index.sql(
        "UPDATE indexed_files SET whisper_indexed = 0 WHERE file_id = :f",
        f=FILE_ID,
    )

    def _sync():
        search_index.sql(
            "UPDATE indexed_files SET mime_type = :m, filename = :n, "
            "file_path = :p WHERE file_id = :f",
            m=VIDEO_MIME, n=f"{TITLE}.mp4", p=str(folder / f"{TITLE}.mp4"),
            f=FILE_ID,
        )

    async def _enabled(_drive, _feature):
        return True

    monkeypatch.setattr("app.policy_client.is_feature_enabled", _enabled)
    monkeypatch.setattr(whisper, "settings", make_settings())
    monkeypatch.setattr(whisper, "validate_file_path", lambda _path: True)
    monkeypatch.setattr(whisper, "get_provider", lambda _name: _LoftProvider(_sync))
    monkeypatch.setattr(whisper, "emit_processing_updated", lambda *a, **k: None)
    text_dim = search_index.dim("vec_text")
    monkeypatch.setattr(
        whisper, "embed_passages",
        lambda texts: [np.ones(text_dim, dtype=np.float32) for _ in texts],
    )

    assert asyncio.run(whisper.index_whisper(FILE_ID)) is False

    rows = _job_rows(search_index)
    assert len(rows) == 1
    assert rows[0]["status"] == "skipped"
    assert rows[0]["completed_at"] is not None
