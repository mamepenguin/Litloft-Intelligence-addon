"""A real search DB, seeded row by row, behind the addon's HTTP app, for the
``/search/exact`` and ``/search/scenes`` endpoint tests.

The DB is the one ``init_search_db`` builds (``tests.vec_partition_db``), so the
FTS5 mirrors and the partitioned ``vec_text`` table are the production ones.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime

import numpy as np
import pytest
from sqlalchemy import event

import app.database as database
from app.config import FeaturesConfig, LLMConfig
from app.models import Embedding, IndexedFile, VideoVisualRun, VideoVisualScene
from tests.vec_partition_db import search_index  # noqa: F401

DRIVE = "family"
OTHER = "elsewhere"
EPUB = "application/epub+zip"


def patch_everywhere(monkeypatch, name: str, value) -> None:
    for mod_name, module in list(sys.modules.items()):
        if module is None or not (mod_name == "app" or mod_name.startswith("app.")):
            continue
        if name in getattr(module, "__dict__", {}):
            monkeypatch.setattr(module, name, value)


class Seed:
    def __init__(self, index):
        self.index = index
        self._ordering: dict[str, int] = {}

    def file(
        self, file_id: str, filename: str | None = None, *, drive: str = DRIVE,
        active: bool = True, mime: str = "text/markdown",
    ) -> None:
        file_type = mime.split("/")[0]
        if file_type not in ("video", "image", "audio"):
            file_type = "document"
        name = filename or f"{file_id}.md"
        with database.get_search_db() as s:
            s.add(IndexedFile(
                file_id=file_id, drive=drive, filename=name,
                file_path=f"/drives/{drive}/{name}", file_type=file_type,
                mime_type=mime, file_size=1, active=active,
            ))

    def text_chunk(self, file_id: str, chunk_index: int, body: str, page: str = "") -> None:
        self.index.sql(
            "INSERT INTO fts_text_content(file_id, chunk_index, page, text) "
            "VALUES (:f, :c, :p, :t)",
            f=file_id, c=str(chunk_index), p=page, t=body,
        )

    def section(self, file_id: str, page: int, title: str) -> None:
        self.index.sql(
            "INSERT INTO document_sections(file_id, page, title) VALUES (:f, :p, :t)",
            f=file_id, p=page, t=title,
        )

    def transcript(
        self, file_id: str, chunk_index: int, body: str,
        times: tuple[float, float] | None = None,
    ) -> None:
        self.index.sql(
            "INSERT INTO fts_transcripts(file_id, chunk_index, text) VALUES (:f, :c, :t)",
            f=file_id, c=str(chunk_index), t=body,
        )
        if times is not None:
            self.index.sql(
                "INSERT INTO transcript_chunks(file_id, chunk_index, text, language, "
                "timestamp_start, timestamp_end, created_at) "
                "VALUES (:f, :c, :t, 'en', :s, :e, :now)",
                f=file_id, c=chunk_index, t=body, s=times[0], e=times[1],
                now=datetime.now(UTC),
            )

    def file_metadata(self, file_id: str, description: str, tags: str) -> None:
        self.index.sql(
            "INSERT INTO fts_files(file_id, filename, title, description, tags_text) "
            "VALUES (:f, :f, '', :d, :t)",
            f=file_id, d=description, t=tags,
        )

    def run(self, file_id: str, run_id: str | None = None, *, active: bool = True) -> str:
        rid = run_id or f"vvr_{file_id}"
        with database.get_search_db() as s:
            s.add(VideoVisualRun(
                id=rid, file_id=file_id, status="succeeded", is_active=active,
                requested_by="manual", priority=100, vision_model="llava:13b",
                pipeline_version=2, candidate_fingerprint="fp",
                selected_count=1, completed_count=1, succeeded_count=1, failed_count=0,
            ))
        return rid

    def scene(
        self, run_id: str, start: float, end: float | None = None, *,
        label: str | None = None, visible: str | None = None,
        excerpt: str | None = None, description: str | None = None,
        ordering: int | None = None,
    ) -> int:
        if ordering is None:
            ordering = self._ordering.get(run_id, 0)
        self._ordering[run_id] = max(self._ordering.get(run_id, 0), ordering) + 1
        with database.get_search_db() as s:
            row = VideoVisualScene(
                run_id=run_id, ordering=ordering, clip_embedding_id=f"c{ordering}",
                start_time=start, end_time=end, status="succeeded",
                scene_label=label, visible_text=visible, transcript_excerpt=excerpt,
                visual_description=description, completed_at=datetime.now(UTC),
            )
            s.add(row)
            s.flush()
            return row.id

    def scene_embedding(
        self, file_id: str, scene_id: int, vector: np.ndarray | None, *,
        suffix: str = "aaaaaaaa", embedding_type: str = "video_visual_scene",
    ) -> str:
        emb_id = f"vvs_{scene_id}_{suffix}"
        self.index.add_embedding(emb_id, file_id, embedding_type, "vec_text")
        if vector is not None:
            self.index.put_vector("vec_text", emb_id, vector, partition=embedding_type)
        return emb_id


@pytest.fixture()
def seed(search_index):  # noqa: F811
    return Seed(search_index)


@pytest.fixture()
def visual(monkeypatch, make_settings):
    """Sets the Visual Index switch and the drive policy the handlers consult.

    ``policy`` maps a drive to True, False, or an exception the lookup raises;
    a drive not listed is allowed. Every other policy feature is allowed.
    """

    def _set(enabled: bool = True, policy: dict | None = None):
        settings = make_settings(
            features=FeaturesConfig(  # type: ignore[call-arg]
                video_visual_index="manual" if enabled else "false",
            ),
            llm=LLMConfig(
                provider="openai_compatible", base_url="http://test",
                model="gemma2:27b", vision_model="llava:13b",
            ),
        )
        patch_everywhere(monkeypatch, "settings", settings)
        rules = policy or {}
        asked: list[tuple] = []

        async def _is_feature_enabled(*args, **kwargs):
            asked.append((args, kwargs))
            values = list(args) + list(kwargs.values())
            drive = values[0] if values else None
            if "video_visual_index" not in values:
                return True
            verdict = rules.get(drive, True)
            if isinstance(verdict, BaseException):
                raise verdict
            return verdict

        patch_everywhere(monkeypatch, "is_feature_enabled", _is_feature_enabled)
        return asked

    return _set


@pytest.fixture()
def api(search_index, monkeypatch):  # noqa: F811
    from fastapi.testclient import TestClient

    from app.main import app

    def _no_llm(*_args, **_kwargs):
        raise AssertionError("a search endpoint must not resolve an LLM")

    patch_everywhere(monkeypatch, "resolve", _no_llm)
    client = TestClient(app)

    def get(path, params=None, drive: str | None = DRIVE):
        headers = {} if drive is None else {"X-Lit-Drive": drive}
        return client.get(path, params=params or {}, headers=headers)

    return get


class StatementLog:
    """Every statement the search engine runs, and whether it ran on an event loop."""

    def __init__(self):
        self.statements: list[str] = []
        self.on_loop: list[str] = []

    def _before(self, _conn, _cursor, statement, _params, _context, _many):
        self.statements.append(statement)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self.on_loop.append(statement)

    def reset(self) -> None:
        self.statements.clear()
        self.on_loop.clear()

    def writes(self) -> list[str]:
        verbs = ("INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE", "DROP", "ALTER")
        return [s for s in self.statements if s.lstrip().upper().startswith(verbs)]


@pytest.fixture()
def statements(search_index):  # noqa: F811
    log = StatementLog()
    engine = database._search_engine
    event.listen(engine, "before_cursor_execute", log._before)
    try:
        yield log
    finally:
        event.remove(engine, "before_cursor_execute", log._before)

