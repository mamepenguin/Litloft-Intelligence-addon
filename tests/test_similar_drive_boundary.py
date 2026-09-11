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
        ("other_neighbour", OTHER_DRIVE, "fermenting koji"),
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

    # Both the handler and ``find_similar`` read through the read session;
    # they resolve it from different modules, so both are pointed here.
    # ``get_search_db`` — the write session — is deliberately NOT patched:
    # ``test_the_write_lock_is_not_taken`` relies on it staying unusable.
    monkeypatch.setattr(database_mod, "get_search_db_read", _session)
    monkeypatch.setattr(search_mod, "get_search_db_read", _session)

    # Stand-in for the embedding legs. It applies the drive filter the real
    # ``_find_similar_by_embedding`` applies in SQL, rather than ignoring the
    # argument: a stub that returns this drive's neighbour whatever it is
    # asked makes the results-side half of the boundary impossible to fail.
    def _legs(file_id, embedding_type, limit, drive):
        candidates = [
            ("home_neighbour", HOME_DRIVE),
            ("other_neighbour", OTHER_DRIVE),
        ]
        return [
            {
                "file_id": fid, "drive": d, "filename": f"{fid}.mp4",
                "file_type": "video", "mime_type": "video/mp4", "score": 0.8,
            }
            for fid, d in candidates
            if drive is None or d == drive
        ]

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


@pytest.mark.asyncio
async def test_another_drives_candidate_is_not_ranked(index):
    """The results-side half of the boundary, which was already working.

    ``find_similar`` passes the request drive down to each embedding leg. With
    that argument dropped, the ranking would carry files out of every drive in
    the index — and the host's ``current_drive_only`` response filter would
    strip them in production, so nothing here would notice unless this asserts
    it.
    """
    result = await similar_files_endpoint(
        file_id="home_src", limit=6, drive=HOME_DRIVE,
    )

    assert [r.file_id for r in result.results] == ["home_neighbour"]


@pytest.mark.asyncio
async def test_the_check_runs_before_any_work(index, monkeypatch):
    """Ordering, not just outcome.

    ``find_similar`` consults its in-memory LRU before it looks the source row
    up, so a check placed after the call would let the cross-drive ranking be
    computed, the other drive's keyword bag be loaded, and the whole result be
    written into the cache — and only then raise. Both arrangements answer 404,
    so the status code cannot tell them apart. This can.
    """
    def _explode(*args, **kwargs):
        raise AssertionError("find_similar ran before the drive check")

    monkeypatch.setattr("app.routers.similar.find_similar", _explode)

    with pytest.raises(HTTPException) as exc:
        await similar_files_endpoint(
            file_id="other_src", limit=6, drive=HOME_DRIVE,
        )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_the_write_lock_is_not_taken(index):
    """The lookup uses the read session, and the fixture proves it by omission.

    ``get_search_db`` takes a blocking ``threading.Lock``. This handler is
    ``async def``, so waiting on it stops the addon's whole event loop — refine
    holds the same lock across a WhisperX alignment. The fixture patches only
    the read session, so a handler that reached for the write one would open
    the real database instead of the fixture's and find no rows.
    """
    result = await similar_files_endpoint(
        file_id="home_src", limit=6, drive=HOME_DRIVE,
    )

    assert [r.file_id for r in result.results] == ["home_neighbour"]


@pytest.mark.asyncio
async def test_an_inactive_source_in_another_drive_answers_empty(index):
    """The ``active`` clause decides how wide the deliberate opening is.

    Both queries filter on ``active``, so a soft-deleted or missing row is
    invisible to the guard *and* to ``find_similar``: it falls into the
    not-indexed opening rather than past the guard. The legs would return this
    drive's neighbour for any id, so an empty answer says they were never
    reached.
    """
    from app.models import IndexedFile

    session = index()
    try:
        row = session.query(IndexedFile).filter(
            IndexedFile.file_id == "other_src"
        ).one()
        row.active = False
        session.commit()
    finally:
        session.close()

    result = await similar_files_endpoint(
        file_id="other_src", limit=6, drive=HOME_DRIVE,
    )

    assert result.results == []
    assert result.source_keywords == []
