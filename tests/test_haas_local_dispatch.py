"""SPEC-ADDON-039: the ``haas_local`` provider through the transcription dispatch.

Real ``get_provider`` and ``index_whisper``; only the recognizer, model fetch,
VAD and ``ffmpeg`` are stand-ins (see ``haas_fakes``).
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

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

from app.database import Base  # noqa: E402
from app.models import IndexedFile, JobRecord  # noqa: E402
from tests.haas_fakes import haas  # noqa: E402,F401  (fixture)

FILE_ID = "f00000000001"


@pytest.fixture()
def session_factory(tmp_path, monkeypatch):
    eng = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(eng)
    with eng.begin() as conn:
        for table in ("fts_transcripts", "fts_transcripts_word"):
            conn.execute(text(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                "file_id TEXT, chunk_index INTEGER, text TEXT)"
            ))
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS vec_text ("
            "embedding_id TEXT PRIMARY KEY, vector BLOB)"
        ))
    Session = sessionmaker(bind=eng, expire_on_commit=False)

    @contextmanager
    def _get_search_db():
        s = Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    monkeypatch.setattr("app.database.get_search_db", _get_search_db)
    monkeypatch.setattr("app.workers.whisper.get_search_db", _get_search_db)
    s = Session()
    s.add(IndexedFile(
        file_id=FILE_ID,
        drive="d1",
        filename="x.mp3",
        file_path="/drives/d1/x.mp3",
        file_type="video",
        mime_type="video/mp4",
        file_size=1000,
        active=True,
        metadata_indexed=True,
        clip_indexed=False,
        whisper_indexed=False,
        text_indexed=False,
        title="x",
        description="",
        tags_text="",
    ))
    s.commit()
    s.close()
    return Session


async def _dispatch(haas, policy):
    from app.workers import whisper as whisper_module

    haas.fresh_module()
    with (
        patch.object(whisper_module, "validate_file_path", return_value=True),
        patch("app.policy_client.is_feature_enabled", new=policy),
    ):
        return await whisper_module.index_whisper(FILE_ID)


def _state(Session):
    s = Session()
    try:
        file = s.query(IndexedFile).filter_by(file_id=FILE_ID).one()
        records = s.query(JobRecord).filter_by(file_id=FILE_ID).all()
        return file.whisper_indexed, records
    finally:
        s.close()


def _policy(cloud_allowed=True):
    async def answer(drive, feature, **kw):
        if feature == "transcription_cloud":
            return cloud_allowed
        return True

    return AsyncMock(side_effect=answer)


async def test_spec_addon_039_i4_no_audio_is_zero_segment_success_with_provider(
    haas, session_factory
):
    haas.ffmpeg_mode("exit1")
    ok = await _dispatch(haas, _policy())

    indexed, records = _state(session_factory)
    assert ok is True
    assert indexed is True
    (record,) = records
    assert record.status == "succeeded"
    assert record.provider == "haas_local"
    assert haas.hf_calls == []


async def test_spec_addon_039_i7_failure_leaves_file_unindexed(
    haas, session_factory
):
    haas.build_exc = RuntimeError("bad onnx")
    ok = await _dispatch(haas, _policy())

    indexed, records = _state(session_factory)
    assert ok is False
    assert indexed is False
    (record,) = records
    assert record.status == "failed"
    assert record.error_class == "FatalError"
    assert record.provider == "haas_local"


async def test_spec_addon_039_i7_fetch_failure_leaves_file_unindexed(
    haas, session_factory
):
    haas.net_error = ConnectionError("down")

    async def instant(_delay, *a, **k):
        return None

    with pytest.MonkeyPatch.context() as m:
        m.setattr(asyncio, "sleep", instant)
        ok = await _dispatch(haas, _policy())

    indexed, records = _state(session_factory)
    assert ok is False
    assert indexed is False
    (record,) = records
    assert record.status == "failed"
    assert record.error_class == "TransientError"


async def test_spec_addon_039_i6_drive_with_cloud_off_uses_haas_local(
    haas, session_factory
):
    haas.ffmpeg_mode("exit1")
    policy = _policy(cloud_allowed=False)
    with patch(
        "app.workers.transcription.whisper_local.WhisperLocalProvider",
        side_effect=AssertionError("whisper_local must not be constructed"),
    ):
        ok = await _dispatch(haas, policy)

    _indexed, records = _state(session_factory)
    assert ok is True
    assert [r.provider for r in records] == ["haas_local"]
    asked = [c.args[1] for c in policy.await_args_list]
    assert "transcription_cloud" not in asked
    assert "index" in asked


def test_spec_addon_039_i6_capability_is_local():
    from app.workers.transcription import build_inner_provider

    assert build_inner_provider("haas_local").capabilities.sends_audio_offhost is False
    assert build_inner_provider("haas_local").capabilities.max_input_bytes == (
        50 * 1024 * 1024
    )
