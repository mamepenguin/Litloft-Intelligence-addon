"""SPEC-ADDON-019: refine keeps the transcript keyword index equal to the stored chunks."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import IndexedFile, TranscriptChunk, TranscriptWord
from app.workers import refine
from tests.llm_helpers import resolved_with

FILE = "f00000000001"
OTHER = "f00000000002"

_FTS5_DDL = (
    "CREATE VIRTUAL TABLE fts_files USING fts5(file_id, filename, title, description, "
    "tags_text, tokenize='trigram')",
    "CREATE VIRTUAL TABLE fts_files_word USING fts5(file_id, filename, title, description, "
    "tags_text, tokenize=\"unicode61 remove_diacritics 2\")",
    "CREATE VIRTUAL TABLE fts_transcripts USING fts5(file_id, chunk_index, text, "
    "tokenize='trigram')",
    "CREATE VIRTUAL TABLE fts_transcripts_word USING fts5(file_id, chunk_index, text, "
    "tokenize=\"unicode61 remove_diacritics 2\")",
    "CREATE VIRTUAL TABLE fts_text_content USING fts5(file_id, chunk_index, page, text, "
    "tokenize='trigram')",
    "CREATE VIRTUAL TABLE fts_text_content_word USING fts5(file_id, chunk_index, page, text, "
    "tokenize=\"unicode61 remove_diacritics 2\")",
    "CREATE VIRTUAL TABLE fts_retrieval_keywords USING fts5(file_id, keywords, "
    "tokenize='trigram')",
)

# Plain tables whose CHECK rejects a marker word, so the keyword write itself can be made
# to raise without knowing which helper performs it.
_FAILING_TRANSCRIPT_DDL = (
    "CREATE TABLE fts_transcripts (file_id TEXT, chunk_index TEXT, "
    "text TEXT CHECK (text NOT LIKE '%BOOM%'))",
    "CREATE TABLE fts_transcripts_word (file_id TEXT, chunk_index TEXT, "
    "text TEXT CHECK (text NOT LIKE '%BOOM%'))",
)

_UNTOUCHED_TABLES = (
    "fts_files",
    "fts_files_word",
    "fts_text_content",
    "fts_text_content_word",
    "fts_retrieval_keywords",
)


def _make_db(tmp_path, transcript_ddl=None):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for ddl in _FTS5_DDL:
            if transcript_ddl and "fts_transcripts" in ddl.split()[3]:
                continue
            conn.execute(text(ddl))
        for ddl in transcript_ddl or ():
            conn.execute(text(ddl))
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def _seed_file(s, file_id, chunks, *, words=True):
    """Seed a transcribed file: chunk rows, their word rows, and the keyword rows
    the transcription path writes (one per chunk, chunk_index as a decimal string)."""
    s.add(IndexedFile(
        file_id=file_id, drive="d", filename=f"{file_id}.mp4",
        file_path=f"/nowhere/{file_id}.mp4", file_type="video", mime_type="video/mp4",
        file_size=1, active=True,
    ))
    for cid, idx, body, start, end in chunks:
        s.add(TranscriptChunk(
            id=cid, file_id=file_id, chunk_index=idx, text=body, language="en",
            timestamp_start=start, timestamp_end=end, created_at=datetime.now(UTC),
        ))
        if words:
            _add_words(s, file_id, body, start, end)
    s.flush()
    for _cid, idx, body, _start, _end in chunks:
        if not body.strip():
            continue
        for table in ("fts_transcripts", "fts_transcripts_word"):
            s.execute(
                text(f"INSERT INTO {table}(file_id, chunk_index, text) VALUES (:f, :i, :t)"),
                {"f": file_id, "i": str(idx), "t": body},
            )


def _add_words(s, file_id, body, start, end):
    tokens = body.split()
    if not tokens:
        return
    step = (end - start) / len(tokens)
    for n, tok in enumerate(tokens):
        s.add(TranscriptWord(
            file_id=file_id, text=tok, language="en",
            timestamp_start=start + n * step, timestamp_end=start + (n + 1) * step,
        ))


def _seed_other_tables(s):
    s.execute(text(
        "INSERT INTO fts_files(file_id, filename, title, description, tags_text) "
        "VALUES (:f, 'talk.mp4', 'cooper netties talk', '', '')"
    ), {"f": FILE})
    s.execute(text(
        "INSERT INTO fts_files_word(file_id, filename, title, description, tags_text) "
        "VALUES (:f, 'talk.mp4', 'cooper netties talk', '', '')"
    ), {"f": FILE})
    s.execute(text(
        "INSERT INTO fts_text_content(file_id, chunk_index, page, text) "
        "VALUES (:f, '0', '1', 'cooper netties page')"
    ), {"f": FILE})
    s.execute(text(
        "INSERT INTO fts_text_content_word(file_id, chunk_index, page, text) "
        "VALUES (:f, '0', '1', 'cooper netties page')"
    ), {"f": FILE})
    s.execute(text(
        "INSERT INTO fts_retrieval_keywords(file_id, keywords) VALUES (:f, 'cooper netties')"
    ), {"f": FILE})


def _db_factory(Session):
    @contextmanager
    def _db():
        session = Session()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    return _db


def _fake_realign(session, file_id, chunk_start, chunk_end, refined_text, *args, **kwargs):
    """Stands in for a successful forced alignment: the span's words become the refined words."""
    session.query(TranscriptWord).filter(
        TranscriptWord.file_id == file_id,
        TranscriptWord.timestamp_start >= chunk_start,
        TranscriptWord.timestamp_start < chunk_end,
    ).delete(synchronize_session=False)
    _add_words(session, file_id, refined_text, chunk_start, chunk_end)
    session.flush()
    return len(refined_text.split())


def _patch_job(monkeypatch, Session, *, realign=_fake_realign, rechunk=None):
    events = AsyncMock()
    monkeypatch.setattr(refine, "get_search_db", _db_factory(Session))
    monkeypatch.setattr(refine, "_emit_ws_event", events)
    monkeypatch.setattr(refine, "realign_words_for_chunk", realign)
    monkeypatch.setattr(refine, "recompute_chunk_embeddings", AsyncMock())
    if rechunk is not None:
        monkeypatch.setattr(refine, "rechunk_from_words", rechunk)
    monkeypatch.setattr(refine.aligner, "job_scope", nullcontext)
    return events


def _llm(*windows):
    client = MagicMock()
    client.enabled = True
    client.generate_json = AsyncMock(side_effect=[
        [{"id": cid, "text_refined": body} for cid, body in window] for window in windows
    ])
    return resolved_with(client, model="text-model")


def _chunks(Session, file_id=FILE):
    with Session() as s:
        return [
            (r.chunk_index, r.text, r.timestamp_start, r.timestamp_end)
            for r in s.query(TranscriptChunk)
            .filter(TranscriptChunk.file_id == file_id)
            .order_by(TranscriptChunk.chunk_index)
        ]


def _keyword_rows(Session, table, file_id=FILE):
    with Session() as s:
        return sorted(
            tuple(r) for r in s.execute(
                text(f"SELECT file_id, chunk_index, text FROM {table} WHERE file_id = :f"),
                {"f": file_id},
            )
        )


def _expected_rows(Session, file_id=FILE):
    return sorted(
        (file_id, str(idx), body)
        for idx, body, _s, _e in _chunks(Session, file_id)
        if body.strip()
    )


def _table_dump(Session, table):
    with Session() as s:
        return sorted(tuple(r) for r in s.execute(text(f"SELECT * FROM {table}")))


def _search(Session, table, term):
    """Keyword hits for a term, each resolved against the file's current chunk rows the way
    the transcript readers resolve chunk_index."""
    with Session() as s:
        hits = s.execute(
            text(f"SELECT file_id, chunk_index FROM {table} WHERE {table} MATCH :q"),
            {"q": f'"{term}"'},
        ).all()
        resolved = []
        for file_id, idx in hits:
            row = (
                s.query(TranscriptChunk)
                .filter(
                    TranscriptChunk.file_id == file_id,
                    TranscriptChunk.chunk_index == int(idx),
                )
                .one_or_none()
            )
            resolved.append((
                file_id,
                None if row is None else (row.text, row.timestamp_start, row.timestamp_end),
            ))
        return resolved


def _failed(events):
    return any("intelligence.refine.failed" in repr(c) for c in events.call_args_list)


PRE_REFINE = [
    (1, 0, "we deployed it on cooper netties", 0.0, 4.0),
    (2, 1, "the cluster had three nodes", 4.0, 8.0),
    (3, 2, "and it worked fine", 8.0, 12.0),
]
REFINED = [
    (1, "We deployed it on Kubernetes."),
    (2, "The cluster had 3 nodes."),
    (3, "And it worked fine."),
]


@pytest.fixture()
def rechunked_db(tmp_path, monkeypatch):
    engine, Session = _make_db(tmp_path)
    with Session() as s:
        _seed_file(s, FILE, PRE_REFINE)
        _seed_file(s, OTHER, [(10, 0, "cooper netties elsewhere", 0.0, 3.0)])
        _seed_other_tables(s)
        s.commit()
    return Session


@pytest.mark.asyncio
async def test_spec_addon_019_after_rechunk_keyword_rows_equal_the_stored_chunks(
    rechunked_db, monkeypatch
):
    Session = rechunked_db
    other_before = _keyword_rows(Session, "fts_transcripts", OTHER)
    other_word_before = _keyword_rows(Session, "fts_transcripts_word", OTHER)
    untouched_before = {t: _table_dump(Session, t) for t in _UNTOUCHED_TABLES}
    _patch_job(monkeypatch, Session)

    await refine._run_refine_job(FILE, "job1", [1, 2, 3], _llm(REFINED))

    expected = _expected_rows(Session)
    assert expected
    assert _keyword_rows(Session, "fts_transcripts") == expected
    assert _keyword_rows(Session, "fts_transcripts_word") == expected
    assert _keyword_rows(Session, "fts_transcripts", OTHER) == other_before
    assert _keyword_rows(Session, "fts_transcripts_word", OTHER) == other_word_before
    assert {t: _table_dump(Session, t) for t in _UNTOUCHED_TABLES} == untouched_before


@pytest.mark.asyncio
@pytest.mark.parametrize("table", ["fts_transcripts", "fts_transcripts_word"])
async def test_spec_addon_019_a_word_refine_wrote_is_found_in_the_chunk_that_holds_it(
    rechunked_db, monkeypatch, table
):
    Session = rechunked_db
    _patch_job(monkeypatch, Session)

    await refine._run_refine_job(FILE, "job1", [1, 2, 3], _llm(REFINED))

    holders = [
        (body, start, end)
        for _idx, body, start, end in _chunks(Session)
        if "Kubernetes" in body
    ]
    assert len(holders) == 1
    hits = [h for h in _search(Session, table, "Kubernetes") if h[0] == FILE]
    assert hits == [(FILE, holders[0])]


@pytest.mark.asyncio
@pytest.mark.parametrize("table", ["fts_transcripts", "fts_transcripts_word"])
async def test_spec_addon_019_a_word_refine_removed_no_longer_finds_the_file(
    rechunked_db, monkeypatch, table
):
    Session = rechunked_db
    _patch_job(monkeypatch, Session)

    await refine._run_refine_job(FILE, "job1", [1, 2, 3], _llm(REFINED))

    assert not any("netties" in body for _i, body, _s, _e in _chunks(Session))
    assert [f for f, _ in _search(Session, table, "netties") if f == FILE] == []
    assert [f for f, _ in _search(Session, table, "netties") if f == OTHER] == [OTHER]


@pytest.mark.asyncio
async def test_spec_addon_019_without_rechunk_keyword_rows_hold_refined_text_minus_blank_chunks(
    tmp_path, monkeypatch
):
    _engine, Session = _make_db(tmp_path)
    with Session() as s:
        _seed_file(s, FILE, [
            (1, 0, "we deployed it on cooper netties", 0.0, 4.0),
            (2, 1, "   ", 4.0, 5.0),
            (3, 2, "and it worked fine", 5.0, 9.0),
        ], words=False)
        s.commit()
    _patch_job(monkeypatch, Session, realign=lambda *a, **k: 0)

    await refine._run_refine_job(
        FILE, "job1", [1, 2, 3],
        _llm([(1, "We deployed it on Kubernetes."), (2, "   "), (3, "And it worked fine.")]),
    )

    expected = [
        (FILE, "0", "We deployed it on Kubernetes."),
        (FILE, "2", "And it worked fine."),
    ]
    assert _expected_rows(Session) == expected
    assert _keyword_rows(Session, "fts_transcripts") == expected
    assert _keyword_rows(Session, "fts_transcripts_word") == expected


@pytest.mark.asyncio
async def test_spec_addon_019_chunk_index_is_written_as_its_decimal_string(
    tmp_path, monkeypatch
):
    _engine, Session = _make_db(tmp_path)
    with Session() as s:
        _seed_file(s, FILE, [(1, 12, "only cooper netties chunk", 0.0, 3.0)], words=False)
        s.commit()
    _patch_job(monkeypatch, Session, realign=lambda *a, **k: 0)

    await refine._run_refine_job(
        FILE, "job1", [1], _llm([(1, "Only Kubernetes chunk.")])
    )

    expected = [(FILE, "12", "Only Kubernetes chunk.")]
    assert _keyword_rows(Session, "fts_transcripts") == expected
    assert _keyword_rows(Session, "fts_transcripts_word") == expected


@pytest.mark.asyncio
async def test_spec_addon_019_a_job_that_refines_nothing_leaves_keyword_rows_alone(
    tmp_path, monkeypatch
):
    _engine, Session = _make_db(tmp_path)
    with Session() as s:
        _seed_file(s, FILE, PRE_REFINE)
        s.execute(text(
            "INSERT INTO fts_transcripts(file_id, chunk_index, text) "
            "VALUES (:f, '7', 'a stale row no chunk carries')"
        ), {"f": FILE})
        s.commit()
    before = (
        _keyword_rows(Session, "fts_transcripts"),
        _keyword_rows(Session, "fts_transcripts_word"),
    )
    _patch_job(monkeypatch, Session)

    await refine._run_refine_job(FILE, "job1", [1, 2, 3], _llm([]))

    assert (
        _keyword_rows(Session, "fts_transcripts"),
        _keyword_rows(Session, "fts_transcripts_word"),
    ) == before


TWELVE = [(i, i - 1, f"segment {i} said cooper netties", float(i), float(i) + 1.0)
          for i in range(1, 13)]
FIRST_WINDOW = [(i, f"Segment {i} said Kubernetes.") for i in range(1, 11)]


@pytest.mark.asyncio
async def test_spec_addon_019_a_failing_keyword_write_rolls_back_its_window_and_keeps_earlier_ones(
    tmp_path, monkeypatch
):
    _engine, Session = _make_db(tmp_path, transcript_ddl=_FAILING_TRANSCRIPT_DDL)
    with Session() as s:
        _seed_file(s, FILE, TWELVE)
        s.commit()
    events = _patch_job(monkeypatch, Session, rechunk=lambda *a, **k: [])

    await refine._run_refine_job(
        FILE, "job1", [c[0] for c in TWELVE],
        _llm(FIRST_WINDOW, [(11, "BOOM segment 11."), (12, "Segment 12 said Kubernetes.")]),
    )

    chunks = {idx: body for idx, body, _s, _e in _chunks(Session)}
    assert [chunks[i - 1] for i, _ in FIRST_WINDOW] == [b for _, b in FIRST_WINDOW]
    assert chunks[10] == "segment 11 said cooper netties"
    assert chunks[11] == "segment 12 said cooper netties"
    with Session() as s:
        late_words = sorted(
            w.text for w in s.query(TranscriptWord).filter(
                TranscriptWord.file_id == FILE, TranscriptWord.timestamp_start >= 11.0
            )
        )
    assert late_words == sorted(
        "segment 11 said cooper netties segment 12 said cooper netties".split()
    )
    expected = _expected_rows(Session)
    assert _keyword_rows(Session, "fts_transcripts") == expected
    assert _keyword_rows(Session, "fts_transcripts_word") == expected
    assert _failed(events)


@pytest.mark.asyncio
async def test_spec_addon_019_a_failing_rechunk_leaves_keyword_rows_equal_to_the_last_window(
    rechunked_db, monkeypatch
):
    Session = rechunked_db
    real_rechunk = refine.rechunk_from_words

    def _rechunk_then_raise(*args, **kwargs):
        real_rechunk(*args, **kwargs)
        raise RuntimeError("rechunk write failed")

    events = _patch_job(monkeypatch, Session, rechunk=_rechunk_then_raise)

    await refine._run_refine_job(FILE, "job1", [1, 2, 3], _llm(REFINED))

    assert [(idx, body) for idx, body, _s, _e in _chunks(Session)] == [
        (0, "We deployed it on Kubernetes."),
        (1, "The cluster had 3 nodes."),
        (2, "And it worked fine."),
    ]
    expected = _expected_rows(Session)
    assert _keyword_rows(Session, "fts_transcripts") == expected
    assert _keyword_rows(Session, "fts_transcripts_word") == expected
    assert _failed(events)
