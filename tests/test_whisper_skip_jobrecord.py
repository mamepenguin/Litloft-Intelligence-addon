"""Phase 2F: silent-skip path now writes a JobRecord row."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stub heavy ML deps before importing whisper.py.
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
from app.workers.transcription import FatalError, ProviderCapabilities  # noqa: E402


@pytest.fixture()
def search_engine(tmp_path):
    db_path = tmp_path / "search.db"
    eng = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(eng)
    with eng.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS fts_transcripts ("
            "  file_id TEXT, chunk_index INTEGER, text TEXT)"
        ))
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS fts_transcripts_word ("
            "  file_id TEXT, chunk_index INTEGER, text TEXT)"
        ))
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS vec_text ("
            "  embedding_id TEXT PRIMARY KEY, vector BLOB)"
        ))
    return eng


@pytest.fixture()
def patched_db(monkeypatch, search_engine):
    Session = sessionmaker(bind=search_engine, expire_on_commit=False)

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
    return Session


def _seed_file(
    Session,
    *,
    file_id: str,
    mime_type: str,
    file_path: str | None = None,
) -> None:
    s = Session()
    s.add(IndexedFile(
        file_id=file_id,
        drive="d1",
        filename=f"{file_id}.bin",
        file_path=file_path or f"/drives/d1/{file_id}.bin",
        file_type="other",
        mime_type=mime_type,
        file_size=1000,
        active=True,
        metadata_indexed=True,
        clip_indexed=False,
        whisper_indexed=False,
        text_indexed=False,
        title=file_id,
        description="",
        tags_text="",
    ))
    s.commit()
    s.close()


@pytest.mark.asyncio
async def test_unsupported_mime_writes_skipped_jobrecord_and_logs(
    patched_db, caplog
) -> None:
    """Phase 2F: a non-transcribable mime must (a) flip
    whisper_indexed=True, (b) emit one INFO log, and (c) leave a
    JobRecord row with status='skipped' and
    error_class='UnsupportedMimeType'."""
    import logging

    from app.workers import whisper as whisper_module

    _seed_file(patched_db, file_id="f00000000010", mime_type="image/jpeg")

    caplog.set_level(logging.INFO, logger="app.workers.whisper")
    result = await whisper_module.index_whisper("f00000000010")
    assert result is True

    s = patched_db()
    try:
        file = s.query(IndexedFile).filter_by(file_id="f00000000010").one()
        assert file.whisper_indexed is True
        records = s.query(JobRecord).filter_by(file_id="f00000000010").all()
    finally:
        s.close()

    assert len(records) == 1
    rec = records[0]
    assert rec.status == "skipped"
    assert rec.error_class == "UnsupportedMimeType"
    assert "image/jpeg" in (rec.error_message or "")
    assert rec.provider is None
    assert rec.completed_at is not None

    info_msgs = [
        rec.message for rec in caplog.records
        if rec.levelno == logging.INFO
    ]
    assert any(
        "skipped for transcription" in m and "image/jpeg" in m
        for m in info_msgs
    ), f"expected skip INFO log, got {info_msgs!r}"


@pytest.mark.asyncio
async def test_audio_mp4_is_transcribable(patched_db, monkeypatch) -> None:
    """Phase 2F regression guard: ``audio/mp4`` (the IANA MIME for
    .m4a) must hit the provider dispatch path, not the silent-skip
    branch. We assert by checking that the function reaches the
    file-path validator (the next gate after the mime check) — which
    we monkeypatch to fail fast and prove control flow."""
    from app.workers import whisper as whisper_module

    _seed_file(patched_db, file_id="f00000000011", mime_type="audio/mp4")

    sentinel = {"validate_called": False}

    def fake_validate(_path: str) -> bool:
        sentinel["validate_called"] = True
        return False  # bail before provider lookup

    monkeypatch.setattr(
        whisper_module, "validate_file_path", fake_validate
    )

    result = await whisper_module.index_whisper("f00000000011")
    # validate_file_path returned False → index_whisper returns False
    assert result is False
    assert sentinel["validate_called"] is True

    # No JobRecord written because we bailed before any provider step.
    s = patched_db()
    try:
        records = s.query(JobRecord).filter_by(file_id="f00000000011").all()
    finally:
        s.close()
    assert records == []


@pytest.mark.asyncio
async def test_audio_opus_is_transcribable(patched_db, monkeypatch) -> None:
    """Same regression guard for the new ``audio/opus`` registration."""
    from app.workers import whisper as whisper_module

    _seed_file(patched_db, file_id="f00000000012", mime_type="audio/opus")

    monkeypatch.setattr(
        whisper_module, "validate_file_path", lambda _p: False
    )

    result = await whisper_module.index_whisper("f00000000012")
    assert result is False
    s = patched_db()
    try:
        records = s.query(JobRecord).filter_by(file_id="f00000000012").all()
    finally:
        s.close()
    # Reached past the mime check: no skipped row written either.
    assert records == []


@pytest.mark.asyncio
async def test_loft_temp_audio_uses_provider_path_and_deletes_on_success(
    patched_db, tmp_path, monkeypatch
) -> None:
    from app.workers import whisper as whisper_module

    loft = tmp_path / "video.loft"
    temp_audio = tmp_path / "video.stt_temp.m4a"
    loft.write_text("{}", encoding="utf-8")
    temp_audio.write_bytes(b"audio")
    _seed_file(
        patched_db,
        file_id="f00000000013",
        mime_type=whisper_module.LOFT_MIME,
        file_path=str(loft),
    )

    provider = MagicMock()
    provider.name = "fake"
    provider.capabilities = ProviderCapabilities(
        sends_audio_offhost=False,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
    )
    provider.transcribe = AsyncMock(return_value=[])

    with (
        patch.object(whisper_module, "get_provider", return_value=provider),
        patch.object(whisper_module, "validate_file_path", return_value=True),
        patch(
            "app.policy_client.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
    ):
        result = await whisper_module.index_whisper("f00000000013")

    assert result is True
    provider.transcribe.assert_awaited_once()
    assert provider.transcribe.await_args.args[0] == str(temp_audio)
    assert not temp_audio.exists()


@pytest.mark.asyncio
async def test_loft_temp_audio_deletes_on_provider_failure(
    patched_db, tmp_path
) -> None:
    from app.workers import whisper as whisper_module

    loft = tmp_path / "broken.loft"
    temp_audio = tmp_path / "broken.stt_temp.m4a"
    loft.write_text("{}", encoding="utf-8")
    temp_audio.write_bytes(b"audio")
    _seed_file(
        patched_db,
        file_id="f00000000014",
        mime_type=whisper_module.LOFT_MIME,
        file_path=str(loft),
    )

    provider = MagicMock()
    provider.name = "fake"
    provider.capabilities = ProviderCapabilities(
        sends_audio_offhost=False,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
    )
    provider.transcribe = AsyncMock(side_effect=FatalError("boom"))

    with (
        patch.object(whisper_module, "get_provider", return_value=provider),
        patch.object(whisper_module, "validate_file_path", return_value=True),
        patch(
            "app.policy_client.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
    ):
        result = await whisper_module.index_whisper("f00000000014")

    assert result is False
    assert not temp_audio.exists()
    s = patched_db()
    try:
        file = s.query(IndexedFile).filter_by(file_id="f00000000014").one()
        assert file.whisper_indexed is True
    finally:
        s.close()
