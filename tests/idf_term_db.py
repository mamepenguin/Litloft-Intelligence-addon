"""A real search DB for the SPEC-ADDON-021 tests, behind the app's own sessions."""

from __future__ import annotations

import math
import multiprocessing
import re
import time
from collections import Counter

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

from app import database, dependencies, tfidf
from app.database import Base
from app.models import IndexedFile, TranscriptChunk
from tests import idf_backfill_api as api
from tests.idf_doubles import WordTokenizer

_FTS = (
    ("fts_files", "file_id, filename, title, description, tags_text"),
    ("fts_files_word", "file_id, filename, title, description, tags_text"),
    ("fts_transcripts", "file_id, chunk_index, text"),
    ("fts_transcripts_word", "file_id, chunk_index, text"),
    ("fts_text_content", "file_id, chunk_index, page, text"),
    ("fts_text_content_word", "file_id, chunk_index, page, text"),
)

LINE = re.compile(
    r"IDF term backfill: written=(\d+) skipped=(\d+) failed=(\d+) "
    r"held=(\d+) remaining=(\d+)"
)


class TermIndex:
    def __init__(self, tmp_path, *, fts: bool = True):
        self.engine = create_engine(
            f"sqlite:///{tmp_path / 'search.db'}",
            connect_args={"check_same_thread": False},
        )
        event.listen(self.engine, "connect", database._enable_wal_mode)
        Base.metadata.create_all(self.engine)
        if fts:
            with self.engine.begin() as conn:
                for name, cols in _FTS:
                    conn.execute(text(f"CREATE VIRTUAL TABLE {name} USING fts5({cols})"))
                conn.execute(text(
                    "CREATE TABLE vec_text (embedding_id TEXT PRIMARY KEY, vector BLOB)"
                ))
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def add_file(
        self,
        file_id: str,
        *,
        drive: str = "a",
        filename: str | None = None,
        transcript: str | None = None,
        active: bool = True,
        indexed: bool | None = None,
        gen: int | None = None,
    ) -> None:
        filename = filename or f"{file_id}.mp4"
        with self.Session() as s:
            s.add(IndexedFile(
                file_id=file_id, drive=drive, filename=filename,
                file_path=f"/drives/{drive}/{filename}",
                file_type="video", mime_type="video/mp4", file_size=1, active=active,
            ))
            if transcript is not None:
                s.add(TranscriptChunk(
                    file_id=file_id, chunk_index=0, text=transcript,
                    timestamp_start=0.0, timestamp_end=1.0,
                ))
            s.commit()
        if indexed is not None or gen is not None:
            self.set_state(file_id, indexed=bool(indexed), gen=gen or 0)

    def set_state(self, file_id: str, *, indexed: bool, gen: int) -> None:
        self.sql(
            "UPDATE indexed_files SET idf_terms_indexed = :i, idf_terms_gen = :g "
            "WHERE file_id = :f",
            i=indexed, g=gen, f=file_id,
        )

    def put_terms(self, file_id: str, terms) -> None:
        for term in terms:
            self.sql(
                "INSERT INTO idf_doc_terms (file_id, term) VALUES (:f, :t)",
                f=file_id, t=term,
            )

    def sql(self, statement: str, **params) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(statement), params)

    def rows(self, statement: str, **params) -> list[tuple]:
        with self.engine.connect() as conn:
            return [tuple(r) for r in conn.execute(text(statement), params)]

    def state(self, file_id: str) -> tuple[bool, int]:
        with self.engine.connect() as conn:
            row = conn.execute(text(
                "SELECT idf_terms_indexed, idf_terms_gen FROM indexed_files "
                "WHERE file_id = :f"
            ), {"f": file_id}).one()
        return bool(row[0]), int(row[1])

    def terms(self, file_id: str) -> set[str]:
        with self.engine.connect() as conn:
            return {
                r[0] for r in conn.execute(
                    text("SELECT term FROM idf_doc_terms WHERE file_id = :f"),
                    {"f": file_id},
                )
            }

    def term_set(self, file_id: str) -> set[str]:
        """The term set the spec defines, from the file's current inputs."""
        with self.Session() as s:
            row = s.query(IndexedFile).filter_by(file_id=file_id).one()
            transcript = tfidf._get_transcript_text(s, file_id)
            return set(tfidf._document_tokens(row.filename, transcript, WordTokenizer()))

    def expected_idf(self, drive: str) -> tuple[dict[str, float], int]:
        """Tokenize every active file of ``drive`` and count document frequencies."""
        with self.Session() as s:
            ids = [
                r.file_id for r in
                s.query(IndexedFile).filter_by(drive=drive, active=True).all()
            ]
        sets = [t for t in (self.term_set(fid) for fid in ids) if t]
        df = Counter(term for terms in sets for term in terms)
        n = len(sets)
        return {term: math.log(n / c) + 1.0 for term, c in df.items()}, n


def counts(caplog) -> list[dict[str, int]]:
    keys = ("written", "skipped", "failed", "held", "remaining")
    return [
        dict(zip(keys, map(int, m.groups())))
        for r in caplog.records
        if (m := LINE.search(r.getMessage()))
    ]


def wait_for(predicate, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def no_children(timeout: float = 5.0) -> bool:
    return wait_for(lambda: not multiprocessing.active_children(), timeout)


@pytest.fixture()
def idx(tmp_path, monkeypatch):
    from app.indexer import IndexManager

    index = TermIndex(tmp_path)
    monkeypatch.setattr(database, "_search_engine", index.engine)
    monkeypatch.setattr(database, "_SearchSession", index.Session)
    monkeypatch.setattr(dependencies, "_index_manager", IndexManager())
    monkeypatch.setattr(tfidf, "_TOKENIZER_FACTORY", WordTokenizer)
    monkeypatch.setenv("IDF_TEST_GATE_DIR", str(tmp_path))
    tfidf.reset_corpus_idf_cache()
    try:
        api.reset_state()
    except AttributeError:
        pass
    yield index
    tfidf.reset_corpus_idf_cache()
    try:
        api.reset_state()
    except AttributeError:
        pass
    for child in multiprocessing.active_children():
        child.terminate()
    index.engine.dispose()
