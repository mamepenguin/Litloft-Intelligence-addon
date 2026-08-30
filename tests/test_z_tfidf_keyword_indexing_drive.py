"""The keyword embedding written at index time is weighted per drive.

``_index_tfidf_keywords`` runs at the end of transcription and stores
one TF-IDF keyword vector per file. The weights come from corpus
statistics, so they must be the statistics of the file's own drive:
otherwise what another drive contains decides which words end up
searchable here.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
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

from app.database import Base  # noqa: E402
from app.models import Embedding, IndexedFile  # noqa: E402
from app.workers import whisper as whisper_mod  # noqa: E402

TRANSCRIPT = ["今日はカレーを作ります。" * 3]


@pytest.fixture()
def indexed(monkeypatch, tmp_path):
    """One indexed video in drive ``b``, with the DB and embedder stubbed."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS vec_text ("
            "  embedding_id TEXT PRIMARY KEY, vector BLOB)"
        ))
    maker = sessionmaker(bind=engine, expire_on_commit=False)

    session = maker()
    session.add(IndexedFile(
        file_id="f1", drive="b", filename="カレー.mp4",
        file_path="/drives/b/カレー.mp4", file_type="video",
        mime_type="video/mp4", file_size=1, active=True,
    ))
    session.commit()
    session.close()

    @contextmanager
    def _db():
        s = maker()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    monkeypatch.setattr(whisper_mod, "get_search_db", _db)
    monkeypatch.setattr(whisper_mod, "get_search_db_read", _db)
    monkeypatch.setattr(
        whisper_mod, "embed_passages",
        lambda passages: [np.ones(4, dtype=np.float32)],
    )
    return maker


def _capture_keyword_calls(monkeypatch):
    """Record every ``extract_top_keywords`` call, returning one keyword."""
    import app.tfidf as tfidf_mod

    calls: list[dict] = []

    def _extract(text_arg, filename, *, drive, k=30):
        calls.append({"filename": filename, "drive": drive})
        return ["カレー"]

    monkeypatch.setattr(tfidf_mod, "extract_top_keywords", _extract)
    return calls


def test_keywords_are_weighted_by_the_files_own_drive(indexed, monkeypatch):
    calls = _capture_keyword_calls(monkeypatch)

    whisper_mod._index_tfidf_keywords("f1", TRANSCRIPT)

    assert calls == [{"filename": "カレー.mp4", "drive": "b"}]


def test_the_embedding_is_stored_and_the_leg_is_closed(indexed, monkeypatch):
    _capture_keyword_calls(monkeypatch)

    whisper_mod._index_tfidf_keywords("f1", TRANSCRIPT)

    session = indexed()
    try:
        rows = (
            session.query(Embedding)
            .filter_by(file_id="f1", embedding_type="tfidf_keywords")
            .all()
        )
        file_row = session.query(IndexedFile).filter_by(file_id="f1").one()
        assert len(rows) == 1
        assert file_row.tfidf_keywords_indexed is True
    finally:
        session.close()


def test_a_file_that_left_the_index_is_closed_without_keywords(
    indexed, monkeypatch
):
    """No row means no drive, and a drive-less corpus lookup is not a thing."""
    calls = _capture_keyword_calls(monkeypatch)

    whisper_mod._index_tfidf_keywords("gone", TRANSCRIPT)

    assert calls == []
    session = indexed()
    try:
        assert (
            session.query(Embedding).filter_by(file_id="gone").count() == 0
        )
    finally:
        session.close()
