"""``GET /similar/{file_id}`` answers only about the drive it was asked about.

The host proxy's ``file_access`` pre_check establishes that the caller may
reach the file's drive. It does not establish that the file is in the drive
the request named — that half belongs to this addon, as
``refine._fetch_indexed_file``'s docstring states for its own route.

``find_similar`` restricts the results to the requested drive but looks its
source row up without one, so the source has to be checked here. What a
missing check yields is not an unreachable file: it is this drive's files
ranked against another drive's file, plus that file's keyword bag returned
verbatim in ``source_keywords``.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from fastapi import HTTPException  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

import app.database as database_mod  # noqa: E402
from app import search as search_mod  # noqa: E402
from app.models import Base, Embedding, IndexedFile  # noqa: E402
from app.routers.similar import similar_files_endpoint  # noqa: E402

HOME_DRIVE = "home"
OTHER_DRIVE = "elsewhere"


@pytest.fixture()
def index(monkeypatch, tmp_path):
    """Two drives, one indexed file each, and one neighbour to rank."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine, expire_on_commit=False)

    rows = [
        ("home_src", HOME_DRIVE, "miso fermenting"),
        ("home_neighbour", HOME_DRIVE, "fermenting koji"),
        ("other_src", OTHER_DRIVE, "tax return 2026"),
    ]
    session = maker()
    try:
        for file_id, drive, bag in rows:
            session.add(IndexedFile(
                file_id=file_id, drive=drive, filename=f"{file_id}.mp4",
                file_path=f"/drives/{file_id}.mp4", file_type="video",
                mime_type="video/mp4", file_size=1, active=True,
            ))
            session.add(Embedding(
                id=f"tk_{file_id}", file_id=file_id,
                embedding_type="tfidf_keywords",
                vector_table="vec_text", content_preview=bag,
            ))
        session.commit()
    finally:
        session.close()

    @contextmanager
    def _session():
        s = maker()
        try:
            yield s
        finally:
            s.close()

    # The handler opens ``get_search_db``; ``find_similar`` opens
    # ``get_search_db_read``. Both have to reach this database.
    monkeypatch.setattr(database_mod, "get_search_db", _session)
    monkeypatch.setattr(search_mod, "get_search_db_read", _session)

    # Pin what the embedding legs return so the drive check is the only
    # variable. The neighbour is in HOME_DRIVE, as a real result would be.
    def _legs(file_id, embedding_type, limit, drive):
        return [{
            "file_id": "home_neighbour", "drive": HOME_DRIVE,
            "filename": "home_neighbour.mp4", "file_type": "video",
            "mime_type": "video/mp4", "score": 0.8,
        }]

    monkeypatch.setattr(search_mod, "_find_similar_by_embedding", _legs)
    search_mod.invalidate_similar_cache()
    return maker


@pytest.mark.asyncio
async def test_a_file_in_the_requested_drive_still_answers(index):
    """Positive control.

    Without it, an assertion that rejected *everything* would look like a
    fix: the cross-drive test below would pass for the wrong reason.
    """
    result = await similar_files_endpoint(
        file_id="home_src", limit=6, drive=HOME_DRIVE,
    )

    assert [r.file_id for r in result.results] == ["home_neighbour"]
    assert [kw.word for kw in result.source_keywords] == [
        "miso", "fermenting",
    ]


@pytest.mark.asyncio
async def test_a_file_in_another_drive_is_not_found(index):
    """404, not 403 — the same answer an unknown file id gets, so the
    response does not say which file ids exist in the other drive."""
    with pytest.raises(HTTPException) as exc:
        await similar_files_endpoint(
            file_id="other_src", limit=6, drive=HOME_DRIVE,
        )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_the_other_drives_keywords_do_not_come_back(index):
    """The payload, not just the status code.

    ``source_keywords`` is the source file's own keyword bag, so a check
    that raised after building the response would still have leaked it.
    """
    with pytest.raises(HTTPException) as exc:
        await similar_files_endpoint(
            file_id="other_src", limit=6, drive=HOME_DRIVE,
        )

    assert "tax" not in str(exc.value.detail)


@pytest.mark.asyncio
async def test_a_file_that_is_not_indexed_yet_answers_empty(index):
    """Unchanged by the drive check, and deliberately not a 404.

    A file the core knows about reaches this route before indexing does.
    Raising would make the caller retry and then report the section
    unavailable, where an empty answer renders nothing and resolves itself.
    """
    result = await similar_files_endpoint(
        file_id="never_indexed", limit=6, drive=HOME_DRIVE,
    )

    assert result.results == []
    assert result.source_keywords == []
