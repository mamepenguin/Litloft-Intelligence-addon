"""Per-drive purge tests for the video_visual_index feature.

Mirrors ``test_vision_purge.py``. When ``addons.intelligence.video_visual_index``
flips off for a drive (umbrella ``index`` stays on), only the drive's
``video_visual_runs`` (scenes cascade via FK) and ``video_visual_scene``
embeddings must be removed — unrelated CLIP/transcript/summary data and
the shared frame cache survive untouched.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

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

from contextlib import contextmanager

from sqlalchemy import create_engine, event  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.database import Base  # noqa: E402
from app.models import Embedding, IndexedFile, VideoVisualRun, VideoVisualScene  # noqa: E402

pytest.importorskip(
    "app.purge", reason="video_visual purge helpers not yet implemented",
)

from app.purge import (  # noqa: E402
    purge_disabled_video_visual_drives,
    purge_video_visual_for_drive,
)


@pytest.fixture()
def search_db(monkeypatch, tmp_path):
    db_path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False},
    )
    # Production enables this via app.database._enable_wal_mode's connect
    # event so FK ON DELETE CASCADE (video_visual_scenes -> ...runs)
    # actually fires; a bare create_engine defaults to OFF.
    event.listen(
        engine, "connect",
        lambda dbapi_conn, _: dbapi_conn.execute("PRAGMA foreign_keys=ON"),
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def _get_search_db():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr("app.database.get_search_db", _get_search_db)
    return engine, Session


def _seed_file_with_run(Session, *, file_id: str, drive: str) -> None:
    with Session() as s:
        s.add(
            IndexedFile(
                file_id=file_id, drive=drive, filename=f"{file_id}.mp4",
                file_path=f"/drives/{drive}/{file_id}.mp4", file_type="video",
                mime_type="video/mp4", file_size=1000, active=True,
            )
        )
        s.add(
            VideoVisualRun(
                id=f"vvr_{file_id}", file_id=file_id, status="succeeded", is_active=True,
                requested_by="manual", priority=100, vision_model="llava:13b",
                pipeline_version=1, candidate_fingerprint="fp",
                selected_count=1, completed_count=1, succeeded_count=1, failed_count=0,
            )
        )
        s.commit()
        scene = VideoVisualScene(
            run_id=f"vvr_{file_id}", ordering=0, clip_embedding_id="c0",
            start_time=1.0, status="succeeded", visual_description="x",
        )
        s.add(scene)
        s.flush()
        s.add(
            Embedding(
                id=f"vvs_{scene.id}_aaaaaaaa",
                file_id=file_id,
                embedding_type="video_visual_scene",
                content_preview="x",
                vector_table="vec_text",
            )
        )
        # Unrelated embedding type that must survive the purge.
        s.add(
            Embedding(
                id=f"clip_{file_id}_bbbbbbbb",
                file_id=file_id,
                embedding_type="clip",
                content_preview="frame",
                vector_table="vec_clip",
            )
        )
        s.commit()


class TestPurgeVideoVisualForDrive:
    def test_removes_runs_scenes_and_embeddings_for_drive(self, search_db):
        engine, Session = search_db
        _seed_file_with_run(Session, file_id="vid-a", drive="family")

        touched = purge_video_visual_for_drive("family")
        assert touched == 1

        with Session() as s:
            assert s.query(VideoVisualRun).filter_by(file_id="vid-a").count() == 0
            assert s.query(VideoVisualScene).count() == 0  # cascaded via FK
            assert (
                s.query(Embedding)
                .filter_by(file_id="vid-a", embedding_type="video_visual_scene")
                .count()
                == 0
            )
            # Unrelated CLIP embedding survives.
            assert (
                s.query(Embedding)
                .filter_by(file_id="vid-a", embedding_type="clip")
                .count()
                == 1
            )
            # IndexedFile row itself is untouched (partial purge, not
            # a full file purge).
            assert s.query(IndexedFile).filter_by(file_id="vid-a").count() == 1

    def test_other_drives_are_untouched(self, search_db):
        _, Session = search_db
        _seed_file_with_run(Session, file_id="vid-a", drive="family")
        _seed_file_with_run(Session, file_id="vid-b", drive="private")

        purge_video_visual_for_drive("family")

        with Session() as s:
            assert s.query(VideoVisualRun).filter_by(file_id="vid-b").count() == 1
            assert (
                s.query(Embedding)
                .filter_by(file_id="vid-b", embedding_type="video_visual_scene")
                .count()
                == 1
            )

    def test_no_data_returns_zero(self, search_db):
        assert purge_video_visual_for_drive("empty-drive") == 0


class TestPurgeDisabledVideoVisualDrives:
    @pytest.mark.asyncio
    async def test_sweeps_only_policy_off_drives(self, search_db, monkeypatch):
        _, Session = search_db
        _seed_file_with_run(Session, file_id="vid-a", drive="family")
        _seed_file_with_run(Session, file_id="vid-b", drive="private")

        async def _is_enabled(drive: str, feature: str) -> bool:
            return drive != "private"

        monkeypatch.setattr(
            "app.policy_client.is_feature_enabled", AsyncMock(side_effect=_is_enabled)
        )

        results = await purge_disabled_video_visual_drives()
        assert results == {"private": 1}

        with Session() as s:
            assert s.query(VideoVisualRun).filter_by(file_id="vid-a").count() == 1
            assert s.query(VideoVisualRun).filter_by(file_id="vid-b").count() == 0

    @pytest.mark.asyncio
    async def test_policy_lookup_failure_skips_drive(self, search_db, monkeypatch):
        _, Session = search_db
        _seed_file_with_run(Session, file_id="vid-a", drive="family")

        monkeypatch.setattr(
            "app.policy_client.is_feature_enabled",
            AsyncMock(side_effect=RuntimeError("boom")),
        )

        results = await purge_disabled_video_visual_drives()
        assert results == {}

        with Session() as s:
            assert s.query(VideoVisualRun).filter_by(file_id="vid-a").count() == 1
