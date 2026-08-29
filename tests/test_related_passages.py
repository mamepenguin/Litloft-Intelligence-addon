"""Passage-level links between a file and the sources you vouched for.

Spec ``2026-08-29-related-passages.md`` §3.1 / §4 / §5.6.

The tests that matter here are the ones that keep an unverified passage
out of the results, and the one that pins the display string to the
string that actually produced the score.

``vec_text`` is a sqlite-vec virtual table. Stage 2 only ever JOINs it,
so a plain table with the same columns satisfies every query except the
stage-1 KNN — that one gets its own test, skipped where the extension
cannot be loaded.
"""

from __future__ import annotations

import sqlite3
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import httpx
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

import numpy as np  # noqa: E402
from sqlalchemy import create_engine, event, text  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app import passages  # noqa: E402
from app.rag import retriever  # noqa: E402

_DIM = 4


def _unit(*values: float) -> np.ndarray:
    vec = np.array(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _axis(i: int) -> np.ndarray:
    return _unit(*[1.0 if j == i else 0.0 for j in range(_DIM)])


def _near(i: int, tilt: float = 0.2) -> np.ndarray:
    """A vector ~0.98 from ``_axis(i)``: related, not the same text.

    Fixtures must not use identical vectors. A cosine of 1.0 now means
    "the same passage twice", which the near-duplicate guard drops on
    purpose — boilerplate shared between two files would otherwise set
    the bar every real pair is measured against.
    """
    values = [0.0] * _DIM
    values[i] = 1.0
    values[(i + 1) % _DIM] = tilt
    return _unit(*values)


def _vec_extension_loads(db_path: str) -> bool:
    """sqlite-vec is present in the Docker test image, not on every host."""
    try:
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        conn.load_extension(passages_vec_path())
        conn.close()
        return True
    except Exception:
        return False


def passages_vec_path() -> str:
    from app.database import _SQLITE_VEC_PATH

    return _SQLITE_VEC_PATH


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    """A small search DB wired into ``app.passages.get_search_engine``."""
    db_path = tmp_path / "search.db"
    real_vec = _vec_extension_loads(str(db_path))

    # The production code runs its DB stages in ``asyncio.to_thread``,
    # so the test engine must tolerate a connection crossing threads.
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    if real_vec:
        def _load(dbapi_conn, _):
            dbapi_conn.enable_load_extension(True)
            dbapi_conn.load_extension(passages_vec_path())
            dbapi_conn.enable_load_extension(False)

        event.listen(engine, "connect", _load)

    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE indexed_files ("
            "file_id TEXT PRIMARY KEY, drive TEXT NOT NULL, "
            "filename TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1)"
        ))
        conn.execute(text(
            "CREATE TABLE embeddings ("
            "id TEXT PRIMARY KEY, file_id TEXT NOT NULL, "
            "embedding_type TEXT NOT NULL, content_preview TEXT NOT NULL, "
            "vector_table TEXT NOT NULL, page INTEGER, chunk_index INTEGER, "
            "timestamp_start REAL, timestamp_end REAL)"
        ))
        conn.execute(text(
            "CREATE TABLE transcript_chunks ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, file_id TEXT NOT NULL, "
            "chunk_index INTEGER NOT NULL, text TEXT NOT NULL, "
            "timestamp_start REAL NOT NULL, timestamp_end REAL NOT NULL)"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE fts_text_content "
            "USING fts5(file_id, chunk_index, page, text, tokenize='trigram')"
        ))
        if real_vec:
            conn.execute(text(
                "CREATE VIRTUAL TABLE vec_text USING vec0("
                f"embedding_id TEXT PRIMARY KEY, vector float[{_DIM}])"
            ))
        else:
            conn.execute(text(
                "CREATE TABLE vec_text ("
                "embedding_id TEXT PRIMARY KEY, vector BLOB NOT NULL)"
            ))

    monkeypatch.setattr(passages, "get_search_engine", lambda: engine)
    engine.real_vec = real_vec  # type: ignore[attr-defined]
    return engine


def _add_file(engine, file_id: str, drive: str = "d1", filename: str | None = None):
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO indexed_files (file_id, drive, filename) "
                "VALUES (:f, :d, :n)"
            ),
            {"f": file_id, "d": drive, "n": filename or f"{file_id}.md"},
        )


def _add_text_chunk(
    engine, file_id: str, chunk_index: int, body: str, vector: np.ndarray,
    *, page: int | None = None, index_fts: bool = True,
):
    embedding_id = f"txt_{file_id}_{chunk_index}"
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO embeddings (id, file_id, embedding_type, "
                "content_preview, vector_table, page, chunk_index) "
                "VALUES (:id, :f, 'text_content', :p, 'vec_text', :page, :ci)"
            ),
            {
                "id": embedding_id, "f": file_id, "p": body[:200],
                "page": page, "ci": chunk_index,
            },
        )
        conn.execute(
            text("INSERT INTO vec_text(embedding_id, vector) VALUES (:id, :v)"),
            {"id": embedding_id, "v": vector.tobytes()},
        )
        if index_fts:
            conn.execute(
                text(
                    "INSERT INTO fts_text_content(file_id, chunk_index, page, text) "
                    "VALUES (:f, :ci, :page, :t)"
                ),
                {
                    "f": file_id, "ci": str(chunk_index),
                    "page": "" if page is None else str(page), "t": body,
                },
            )
    return embedding_id


def _add_transcript_chunk(
    engine, file_id: str, chunk_index: int, body: str, vector: np.ndarray,
    *, start: float, end: float,
):
    embedding_id = f"wh_{file_id}_{chunk_index}"
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO embeddings (id, file_id, embedding_type, "
                "content_preview, vector_table, timestamp_start, timestamp_end) "
                "VALUES (:id, :f, 'whisper', :p, 'vec_text', :s, :e)"
            ),
            {"id": embedding_id, "f": file_id, "p": body[:200], "s": start, "e": end},
        )
        conn.execute(
            text("INSERT INTO vec_text(embedding_id, vector) VALUES (:id, :v)"),
            {"id": embedding_id, "v": vector.tobytes()},
        )
        conn.execute(
            text(
                "INSERT INTO transcript_chunks "
                "(file_id, chunk_index, text, timestamp_start, timestamp_end) "
                "VALUES (:f, :ci, :t, :s, :e)"
            ),
            {"f": file_id, "ci": chunk_index, "t": body, "s": start, "e": end},
        )
    return embedding_id


@contextmanager
def _internal_api(monkeypatch, handler):
    orig = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return orig(*args, **kwargs)

    monkeypatch.setattr(retriever.httpx, "AsyncClient", _factory)
    yield


def _verified(*file_ids: str):
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"accessible": list(file_ids), "trust_filtered": True}
        )
    return handler


def _settings_with(**overrides):
    """A settings stand-in with the related-passages knobs overridden.

    The config dataclasses are frozen, so a test tunes them by swapping
    the whole object rather than mutating a field.
    """
    import dataclasses
    from types import SimpleNamespace

    return SimpleNamespace(
        related_passages=dataclasses.replace(
            passages.settings.related_passages, **overrides
        )
    )


def _permissive(monkeypatch):
    """Turn the outlier gate off for tests that are not about the gate.

    The gate compares a pair against the spread of the whole request, so
    it needs a realistic field of noise to mean anything. Fixtures that
    exercise plumbing — dedupe, limits, text resolution, trust — hold
    two or three vectors, where z is undefined or unreachable. Those
    tests pin behaviour that must hold whatever the gate decides.
    """
    monkeypatch.setattr(
        passages, "settings", _settings_with(min_z=0.0, small_sample_z=0.0)
    )


def _nearest(monkeypatch, *file_ids: str):
    """Pin stage 1 so the logic tests do not need sqlite-vec."""
    monkeypatch.setattr(
        passages, "_nearest_files", lambda *a, **kw: list(file_ids)
    )


# ---------------------------------------------------------------------------
# The pair itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pairs_the_two_passages_verbatim(search_db, monkeypatch):
    """What is displayed is the text whose vector produced the score."""
    source_body = "A chunk about chunk sizes, long enough to be prose. " * 3
    other_body = "At 400 characters a paragraph gets split across two. " * 3

    _add_file(search_db, "src")
    _add_file(search_db, "note", filename="rag-design-notes.md")
    _add_text_chunk(search_db, "src", 0, source_body, _axis(0))
    _add_text_chunk(search_db, "note", 7, other_body, _near(0), page=3)

    _permissive(monkeypatch)
    _nearest(monkeypatch, "note")
    with _internal_api(monkeypatch, _verified("note")):
        pairs = await passages.find_related_passages(
            file_id="src", drive="d1", credential=None
        )

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.text == source_body
    assert pair.other_text == other_body
    assert pair.other_file_id == "note"
    assert pair.other_filename == "rag-design-notes.md"
    assert pair.other_page == 3
    assert pair.score == pytest.approx(0.98, abs=0.01)


@pytest.mark.asyncio
async def test_transcript_passages_carry_their_timestamp(search_db, monkeypatch):
    _add_file(search_db, "src")
    _add_file(search_db, "talk", filename="asr-workshop.mp4")
    _add_text_chunk(search_db, "src", 0, "Forced alignment rebuilds words. " * 4, _axis(1))
    _add_transcript_chunk(
        search_db, "talk", 2, "so the words are realigned, not interpolated. " * 3,
        _near(1), start=724.0, end=751.5,
    )

    _permissive(monkeypatch)
    _nearest(monkeypatch, "talk")
    with _internal_api(monkeypatch, _verified("talk")):
        pairs = await passages.find_related_passages(
            file_id="src", drive="d1", credential=None
        )

    assert len(pairs) == 1
    assert pairs[0].other_timestamp == pytest.approx(724.0)
    assert "realigned" in pairs[0].other_text


# ---------------------------------------------------------------------------
# Keeping unvouched and unreachable passages out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unverified_candidate_never_appears(search_db, monkeypatch):
    shared = _unit(1, 0, 0, 0)
    _add_file(search_db, "src")
    _add_file(search_db, "clip")
    _add_text_chunk(search_db, "src", 0, "Body text about a topic. " * 4, shared)
    _add_text_chunk(search_db, "clip", 0, "Someone else's prose. " * 4, shared)

    _nearest(monkeypatch, "clip")
    # Core answers that nothing in the candidate set is verified.
    with _internal_api(monkeypatch, _verified()):
        pairs = await passages.find_related_passages(
            file_id="src", drive="d1", credential=None
        )

    assert pairs == []


@pytest.mark.asyncio
async def test_fails_closed_when_core_did_not_confirm_the_trust_filter(
    search_db, monkeypatch
):
    """An older core drops the unknown field and answers with everything."""
    shared = _unit(1, 0, 0, 0)
    _add_file(search_db, "src")
    _add_file(search_db, "clip")
    _add_text_chunk(search_db, "src", 0, "Body text about a topic. " * 4, shared)
    _add_text_chunk(search_db, "clip", 0, "Someone else's prose. " * 4, shared)

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"accessible": ["clip"]})

    _nearest(monkeypatch, "clip")
    with _internal_api(monkeypatch, handler):
        pairs = await passages.find_related_passages(
            file_id="src", drive="d1", credential=None
        )

    assert pairs == []


@pytest.mark.asyncio
async def test_chunk_without_resolvable_text_is_dropped(search_db, monkeypatch):
    """No FTS row means no verbatim text, and a preview is not a substitute."""
    shared = _unit(1, 0, 0, 0)
    _add_file(search_db, "src")
    _add_file(search_db, "note")
    _add_text_chunk(search_db, "src", 0, "Body text about a topic. " * 4, shared)
    _add_text_chunk(
        search_db, "note", 0, "Its full text was never indexed. " * 4, shared,
        index_fts=False,
    )

    _nearest(monkeypatch, "note")
    with _internal_api(monkeypatch, _verified("note")):
        pairs = await passages.find_related_passages(
            file_id="src", drive="d1", credential=None
        )

    assert pairs == []


@pytest.mark.asyncio
async def test_unrelated_passages_produce_nothing(search_db, monkeypatch):
    _add_file(search_db, "src")
    _add_file(search_db, "note")
    _add_text_chunk(search_db, "src", 0, "Body about one topic. " * 4, _unit(1, 0, 0, 0))
    _add_text_chunk(search_db, "note", 0, "Body about another. " * 4, _unit(0, 0, 0, 1))

    _nearest(monkeypatch, "note")
    with _internal_api(monkeypatch, _verified("note")):
        pairs = await passages.find_related_passages(
            file_id="src", drive="d1", credential=None
        )

    assert pairs == []


@pytest.mark.asyncio
async def test_unindexed_file_asks_core_nothing(search_db, monkeypatch):
    """A file with no chunks has no centroid, so there is nothing to ask."""
    _add_file(search_db, "src")

    called = False

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"accessible": [], "trust_filtered": True})

    with _internal_api(monkeypatch, handler):
        pairs = await passages.find_related_passages(
            file_id="src", drive="d1", credential=None
        )

    assert pairs == []
    assert called is False


# ---------------------------------------------------------------------------
# Ranking and spread
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_row_per_source_passage_and_per_other_file(search_db, monkeypatch):
    """Five rows about the same paragraph would crowd out everything else."""
    _add_file(search_db, "src")
    _add_file(search_db, "note")
    _add_text_chunk(search_db, "src", 0, "First source paragraph. " * 4, _axis(0))
    _add_text_chunk(search_db, "src", 1, "Second source paragraph. " * 4, _axis(1))
    _add_text_chunk(search_db, "note", 0, "First note paragraph. " * 4, _near(0))
    _add_text_chunk(search_db, "note", 1, "Second note paragraph. " * 4, _near(0, 0.22))

    _permissive(monkeypatch)
    _nearest(monkeypatch, "note")
    with _internal_api(monkeypatch, _verified("note")):
        pairs = await passages.find_related_passages(
            file_id="src", drive="d1", credential=None
        )

    assert len(pairs) == 1
    assert pairs[0].other_file_id == "note"


@pytest.mark.asyncio
async def test_limit_caps_the_rows(search_db, monkeypatch):
    _add_file(search_db, "src")
    for i in range(4):
        _add_text_chunk(
            search_db, "src", i, f"Source paragraph {i}. " * 4, _axis(i),
        )
        _add_file(search_db, f"note{i}")
        _add_text_chunk(
            search_db, f"note{i}", 0, f"Note paragraph {i}. " * 4, _near(i),
        )

    notes = [f"note{i}" for i in range(4)]
    _permissive(monkeypatch)
    _nearest(monkeypatch, *notes)
    with _internal_api(monkeypatch, _verified(*notes)):
        pairs = await passages.find_related_passages(
            file_id="src", drive="d1", credential=None, limit=2
        )

    assert len(pairs) == 2
    assert pairs[0].score >= pairs[1].score


# ---------------------------------------------------------------------------
# Stage 1, against the real vector index
# ---------------------------------------------------------------------------


def test_nearest_files_stays_inside_the_drive_and_skips_the_source(search_db):
    if not getattr(search_db, "real_vec", False):
        pytest.skip("sqlite-vec extension not loadable here")

    shared = _unit(1, 0, 0, 0)
    _add_file(search_db, "src", drive="d1")
    _add_file(search_db, "same-drive", drive="d1")
    _add_file(search_db, "other-drive", drive="d2")
    _add_text_chunk(search_db, "src", 0, "Body. " * 20, shared)
    _add_text_chunk(search_db, "same-drive", 0, "Body. " * 20, shared)
    _add_text_chunk(search_db, "other-drive", 0, "Body. " * 20, shared)

    found = passages._nearest_files(shared, file_id="src", drive="d1")

    assert found == ["same-drive"]


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------


class _FakeRequest:
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}


@contextmanager
def _indexed_files(monkeypatch, rows: list[tuple[str, str]]):
    """Patch the router's file lookup with an in-memory table."""
    import app.database as database
    from app.models import IndexedFile

    from sqlalchemy.orm import sessionmaker

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    IndexedFile.__table__.create(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    for file_id, drive in rows:
        session.add(
            IndexedFile(
                file_id=file_id, drive=drive, filename=f"{file_id}.md",
                file_path=f"/{file_id}.md", file_type="document",
                mime_type="text/markdown", file_size=1, active=True,
            )
        )
    session.commit()

    @contextmanager
    def _read():
        yield session

    monkeypatch.setattr(database, "get_search_db_read", _read)
    try:
        yield
    finally:
        session.close()


@pytest.mark.asyncio
async def test_endpoint_404s_for_a_file_in_another_drive(monkeypatch):
    from fastapi import HTTPException

    from app.routers.passages import related_passages_endpoint

    with _indexed_files(monkeypatch, [("elsewhere", "d2")]):
        with pytest.raises(HTTPException) as excinfo:
            await related_passages_endpoint(
                file_id="elsewhere", request=_FakeRequest(), drive="d1"
            )

    # 404, not 403: the caller must not learn the file exists.
    assert excinfo.value.status_code == 404


@pytest.mark.asyncio
async def test_endpoint_is_empty_not_broken_before_the_addon_indexes(monkeypatch):
    """Core discovers a file before the addon reaches it.

    During that window the honest answer is the same as for an indexed
    file with no matches. An error would make the section show a failure
    every time a viewer opened something newly added.
    """
    from app.routers.passages import related_passages_endpoint

    with _indexed_files(monkeypatch, []):
        response = await related_passages_endpoint(
            file_id="not-indexed-yet", request=_FakeRequest(), drive="d1"
        )

    assert response.results == []


@pytest.mark.asyncio
async def test_endpoint_maps_a_pair_to_both_sides(monkeypatch):
    from app.routers import passages as router_module

    async def _fake(**kwargs):
        return [
            passages.PassagePair(
                text="my paragraph",
                page=None,
                timestamp=None,
                other_file_id="note",
                other_drive="d1",
                other_filename="notes.md",
                other_text="their paragraph",
                other_page=4,
                other_timestamp=None,
                score=0.912345,
                overlap=["対角線", "回転"],
            )
        ]

    monkeypatch.setattr(router_module, "find_related_passages", _fake)

    with _indexed_files(monkeypatch, [("src", "d1")]):
        response = await related_passages_endpoint_call()

    assert len(response.results) == 1
    item = response.results[0]
    assert item.source.text == "my paragraph"
    assert item.match.text == "their paragraph"
    assert item.match.page == 4
    assert item.file_id == "note"
    assert item.drive == "d1"
    assert item.filename == "notes.md"
    assert item.score == pytest.approx(0.9123)
    assert item.overlap == ["対角線", "回転"]


async def related_passages_endpoint_call():
    from app.routers.passages import related_passages_endpoint

    return await related_passages_endpoint(
        file_id="src", request=_FakeRequest(), drive="d1"
    )


# ---------------------------------------------------------------------------
# Regressions from review (2026-08-29)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_verified_file_below_a_run_of_unverified_ones_survives(
    search_db, monkeypatch
):
    """The candidate cap lands after the trust filter, not before it.

    Capping first lets a run of unverified neighbours at the top empty
    the list while verified files sit just below the cut.
    """
    _add_file(search_db, "src")
    _add_text_chunk(search_db, "src", 0, "Body about a topic. " * 4, _axis(0))

    nearest = []
    for i in range(passages.settings.related_passages.candidate_files + 5):
        name = f"clip{i}"
        _add_file(search_db, name)
        _add_text_chunk(search_db, name, 0, f"Clip prose {i}. " * 4, _near(0))
        nearest.append(name)
    _add_file(search_db, "note", filename="notes.md")
    _add_text_chunk(search_db, "note", 0, "The vouched-for passage. " * 4, _near(0))
    nearest.append("note")

    _permissive(monkeypatch)
    _nearest(monkeypatch, *nearest)
    with _internal_api(monkeypatch, _verified("note")):
        pairs = await passages.find_related_passages(
            file_id="src", drive="d1", credential=None
        )

    assert [p.other_file_id for p in pairs] == ["note"]


@pytest.mark.asyncio
async def test_a_late_source_passage_can_still_match(search_db, monkeypatch):
    """Sampling spreads across the file; it never keeps only the opening."""
    monkeypatch.setattr(
        passages, "settings",
        _settings_with(max_source_chunks=2),
    )

    _add_file(search_db, "src")
    for i in range(4):
        _add_text_chunk(
            search_db, "src", i, f"Source paragraph {i}. " * 4, _axis(i),
        )
    _add_file(search_db, "note", filename="notes.md")
    # Matches source paragraph 2 only — a paragraph a prefix cap of two
    # would never have looked at.
    _add_text_chunk(
        search_db, "note", 0, "The passage the middle echoes. " * 4, _near(2),
    )

    _permissive(monkeypatch)
    _nearest(monkeypatch, "note")
    with _internal_api(monkeypatch, _verified("note")):
        pairs = await passages.find_related_passages(
            file_id="src", drive="d1", credential=None
        )

    assert len(pairs) == 1
    assert "Source paragraph 2" in pairs[0].text


def test_sampling_spans_the_file_instead_of_taking_its_opening():
    assert passages._sample(list(range(10)), 3) == [0, 3, 6]
    assert passages._sample(list(range(3)), 10) == [0, 1, 2]


def test_knn_budget_covers_the_source_own_chunks(search_db, monkeypatch):
    """A file's chunks are the nearest things to its own centroid.

    With a budget that ignores them, ``k`` is spent on rows the
    ``!= :self`` predicate then removes, and a long document finds no
    neighbours at all.
    """
    if not getattr(search_db, "real_vec", False):
        pytest.skip("sqlite-vec extension not loadable here")

    monkeypatch.setattr(passages, "_KNN_POOL", 2)
    shared = _unit(1, 0, 0, 0)

    _add_file(search_db, "src")
    for i in range(6):
        _add_text_chunk(search_db, "src", i, f"Source paragraph {i}. " * 4, shared)
    _add_file(search_db, "note", filename="notes.md")
    _add_text_chunk(search_db, "note", 0, "A neighbour. " * 4, shared)

    starved = passages._nearest_files(
        shared, file_id="src", drive="d1", source_rows=0
    )
    budgeted = passages._nearest_files(
        shared, file_id="src", drive="d1", source_rows=passages._passage_count("src")
    )

    # The starved call is what the budget exists to prevent; the point of
    # the test is that the budgeted one finds the neighbour regardless.
    assert budgeted == ["note"]
    assert len(starved) <= len(budgeted)


# ---------------------------------------------------------------------------
# Gating (Phase 4): absolute cosine is a weak signal, so the gate is the
# one search._find_similar_by_embedding already arrived at.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_same_text_twice_is_a_duplicate_not_a_connection(
    search_db, monkeypatch
):
    """Boilerplate shared between two files would set the bar for everything.

    Two passages at cosine ~1.0 are the same text. Kept, it becomes the
    top score, and the margin cutoff then measures every real pair
    against it — which is how a frontmatter block came to be the only
    row a whole drive produced.
    """
    same = _axis(0)
    _add_file(search_db, "src")
    _add_file(search_db, "note")
    _add_text_chunk(search_db, "src", 0, "A shared boilerplate header. " * 4, same)
    _add_text_chunk(search_db, "note", 0, "A shared boilerplate header. " * 4, same)

    _nearest(monkeypatch, "note")
    with _internal_api(monkeypatch, _verified("note")):
        pairs = await passages.find_related_passages(
            file_id="src", drive="d1", credential=None
        )

    assert pairs == []


@pytest.mark.asyncio
async def test_frontmatter_is_not_a_passage(search_db, monkeypatch):
    """Two files' YAML blocks resemble each other far more than their prose."""
    _add_file(search_db, "src")
    _add_file(search_db, "note")
    _add_text_chunk(
        search_db, "src", 0,
        "---\norigin: source_capture\nsource_file_ids:\n- hh_45I_sgMVC\ntags: []\n",
        _axis(0),
    )
    _add_text_chunk(
        search_db, "note", 0,
        "---\norigin: source_capture\nsource_file_ids:\n- QQ_18B_qwPLZ\ntags: []\n",
        _near(0),
    )

    _nearest(monkeypatch, "note")
    with _internal_api(monkeypatch, _verified("note")):
        pairs = await passages.find_related_passages(
            file_id="src", drive="d1", credential=None
        )

    assert pairs == []


@pytest.mark.asyncio
async def test_fragments_match_everything_so_they_are_not_passages(
    search_db, monkeypatch
):
    """An ellipsis has no content to be similar *about*."""
    _add_file(search_db, "src")
    _add_file(search_db, "note")
    _add_text_chunk(search_db, "src", 0, "A real paragraph of prose. " * 4, _axis(0))
    _add_text_chunk(search_db, "note", 0, "……", _near(0))

    _nearest(monkeypatch, "note")
    with _internal_api(monkeypatch, _verified("note")):
        pairs = await passages.find_related_passages(
            file_id="src", drive="d1", credential=None
        )

    assert pairs == []


def _chunk(vector: np.ndarray, file_id: str) -> "passages._Chunk":
    return passages._Chunk(
        embedding_id=f"e_{file_id}",
        file_id=file_id,
        embedding_type="text_content",
        chunk_index=0,
        timestamp_start=None,
        page=None,
        preview="A paragraph long enough not to count as a fragment. " * 2,
        vector=vector,
    )


def _field(*cosines: float) -> tuple[list, list]:
    """One source passage and candidates at exactly the given cosines.

    ``c_k = cos(t)·e0 + sin(t)·e_k`` dots with ``e0`` to exactly cos(t),
    so a test can state the score distribution it means to describe
    instead of hoping vectors land near it.
    """
    dim = len(cosines) + 1
    source = [_chunk(_basis(0, dim), "src")]
    candidates = []
    for k, cos in enumerate(cosines):
        vec = np.zeros(dim, dtype=np.float32)
        vec[0] = cos
        vec[k + 1] = float(np.sqrt(max(0.0, 1.0 - cos * cos)))
        candidates.append(_chunk(vec, f"other{k}"))
    return source, candidates


def _basis(i: int, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[i] = 1.0
    return vec


def test_an_outlier_survives_a_field_of_noise():
    """Real matches stand out from the request; near misses do not.

    The numbers are the measured shape of a real request: unrelated
    passages bunched around 0.78-0.86, one genuine match at 0.93.
    """
    noise = [0.78 + (i % 9) * 0.01 for i in range(49)]
    source, candidates = _field(0.93, *noise)

    pairs = passages._rank_pairs(source, candidates, limit=5)

    assert [c.file_id for _, _, c in pairs] == ["other0"]


def test_a_field_with_no_standout_yields_nothing():
    """Everything equally close means the embedding cannot discriminate.

    This is also the honest answer when a whole candidate set genuinely
    is uniformly related — an empty section beats five arbitrary rows,
    because a reader shown spurious links learns to ignore the real ones.
    """
    source, candidates = _field(*[0.80 + (i % 7) * 0.01 for i in range(50)])

    assert passages._rank_pairs(source, candidates, limit=5) == []


def test_the_floor_still_applies_when_there_is_no_distribution():
    """A single pair has no spread to be an outlier against."""
    source, candidates = _field(0.93)
    assert len(passages._rank_pairs(source, candidates, limit=5)) == 1

    source, candidates = _field(0.42)
    assert passages._rank_pairs(source, candidates, limit=5) == []


@pytest.mark.asyncio
async def test_a_flat_field_of_candidates_yields_nothing(search_db, monkeypatch):
    """No standout means the embedding is not discriminating here."""
    _add_file(search_db, "src")
    _add_text_chunk(search_db, "src", 0, "The source paragraph. " * 4, _axis(0))

    names = []
    for i in range(6):
        name = f"flat{i}"
        _add_file(search_db, name)
        # All ~0.80, within a hair of each other.
        _add_text_chunk(
            search_db, name, 0, f"Loosely related prose {i}. " * 4,
            _unit(1, 0.75 + i * 0.0001, 0, 0),
        )
        names.append(name)

    _nearest(monkeypatch, *names)
    with _internal_api(monkeypatch, _verified(*names)):
        pairs = await passages.find_related_passages(
            file_id="src", drive="d1", credential=None
        )

    assert pairs == []


def test_a_duplicate_does_not_drag_the_bar_over_a_real_match():
    """The distribution has to describe the noise, not include the outlier.

    A pair at 1.0 is excluded from the results, but if it is left in the
    statistics first it lifts the mean and the spread — and the raised
    bar then rejects the genuine match it was supposed to be measured
    against.
    """
    noise = [0.78 + (i % 9) * 0.01 for i in range(48)]
    source, candidates = _field(1.0, 0.93, *noise)

    pairs = passages._rank_pairs(source, candidates, limit=5)

    # other0 is the duplicate (dropped), other1 the real match.
    assert [c.file_id for _, _, c in pairs] == ["other1"]


def test_a_small_field_can_still_produce_a_match():
    """A drive too small for a z of 3 must not be permanently empty.

    With n values the largest possible z is (n-1)/sqrt(n), so on a
    handful of pairs the configured bar is unreachable and every pair
    would be rejected however good it is.
    """
    source, candidates = _field(0.93, 0.78, 0.79, 0.80, 0.81)

    pairs = passages._rank_pairs(source, candidates, limit=5)

    assert [c.file_id for _, _, c in pairs] == ["other0"]


def test_a_small_flat_field_still_produces_nothing():
    """Reachability is not a licence to return the merely-largest value."""
    source, candidates = _field(0.80, 0.81, 0.82, 0.83, 0.84, 0.85)

    assert passages._rank_pairs(source, candidates, limit=5) == []


def _recurring_field(boilerplate_files: int) -> tuple[list, list]:
    """Two source passages: one that recurs across files, one that does not.

    The recurring passage scores *higher* than the real one, which is
    what makes it a threshold-proof problem — the boilerplate on a real
    library measured 0.98 against genuine subject matter at 0.93.
    """
    noise = 40
    dim = 2 + boilerplate_files + 1 + noise
    source = [
        _chunk(_basis(0, dim), "src"),  # the sign-off
        _chunk(_basis(1, dim), "src"),  # actual subject matter
    ]

    candidates = []
    axis = 2
    for k in range(boilerplate_files):
        vec = np.zeros(dim, dtype=np.float32)
        vec[0] = 0.97
        vec[axis] = float(np.sqrt(1 - 0.97**2))
        candidates.append(_chunk(vec, f"boiler{k}"))
        axis += 1

    real = np.zeros(dim, dtype=np.float32)
    real[1] = 0.93
    real[axis] = float(np.sqrt(1 - 0.93**2))
    candidates.append(_chunk(real, "real-note"))
    axis += 1

    for k in range(noise):
        vec = np.zeros(dim, dtype=np.float32)
        vec[0] = 0.1
        vec[axis] = float(np.sqrt(1 - 0.01))
        candidates.append(_chunk(vec, f"noise{k}"))
        axis += 1

    return source, candidates


def test_a_passage_that_recurs_across_files_is_boilerplate():
    """A sign-off is not what the file is about, however well it scores."""
    source, candidates = _recurring_field(boilerplate_files=3)

    pairs = passages._rank_pairs(source, candidates, limit=5)

    assert [c.file_id for _, _, c in pairs] == ["real-note"]


def test_a_passage_shared_with_one_other_file_is_kept():
    """Two episodes of a series legitimately cover the same ground.

    The cap has to sit above that: on a measured sample every passage
    matching two files was real content, and every passage matching
    three was a sign-off.
    """
    source, candidates = _recurring_field(boilerplate_files=2)

    pairs = passages._rank_pairs(source, candidates, limit=5)

    found = {c.file_id for _, _, c in pairs}
    assert "real-note" in found
    assert found & {"boiler0", "boiler1"}


def test_verbatim_copies_still_count_as_recurrence():
    """The copies that are never shown are the strongest evidence.

    A scripted sign-off is the likeliest thing in a library to appear
    word for word in several files. Those copies are dropped from the
    results as duplicates — but if they were also dropped from the
    count, two of them could hide and let the third pass the cap alone.
    """
    source, candidates = _recurring_field(boilerplate_files=3)
    # Two of the three are now verbatim rather than merely close.
    for k in (0, 1):
        exact = np.zeros(len(candidates[k].vector), dtype=np.float32)
        exact[0] = 1.0
        candidates[k] = _chunk(exact, f"boiler{k}")

    pairs = passages._rank_pairs(source, candidates, limit=5)

    assert [c.file_id for _, _, c in pairs] == ["real-note"]
