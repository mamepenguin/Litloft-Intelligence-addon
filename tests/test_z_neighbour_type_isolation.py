"""A neighbour lookup must return the channel it was asked for.

``vec_text`` holds whisper, text_content, metadata and tfidf_keywords
side by side; ``vec_clip`` holds scene frames beside representative
ones. ``MATCH`` ranks across all of them, so without a filter the
"TF-IDF second opinion" behind every video and audio answer is mostly
whisper chunks wearing a TF-IDF label, and the representative-frame
route that spec 2026-05-02 chose over scene CLIP silently gets scene
CLIP anyway.

Measured on a real index before the filter: a tfidf_keywords query came
back 49 tfidf rows against 326 whisper, and a clip_thumbnail query 238
representative against 146 scene.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pytest

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app import search as search_mod  # noqa: E402
from app.models import Base, Embedding, IndexedFile  # noqa: E402

DIM = 8


def _vec(seed: float) -> np.ndarray:
    v = np.full(DIM, seed, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture()
def index(monkeypatch, tmp_path):
    """A vec_text stand-in holding two channels at the same distances."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine, expire_on_commit=False)

    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE vec_text (embedding_id TEXT PRIMARY KEY, "
            "vector BLOB, distance REAL)"
        ))

    rows = []
    s = maker()
    try:
        for i, (fid, etype) in enumerate([
            ("src", "tfidf_keywords"),
            ("whisper_a", "whisper"),
            ("whisper_b", "whisper"),
            ("whisper_c", "whisper"),
            ("kw_a", "tfidf_keywords"),
        ]):
            s.add(IndexedFile(
                file_id=fid, drive="d", filename=f"{fid}.mp4",
                file_path=f"/drives/{fid}.mp4", file_type="video",
                mime_type="video/mp4", file_size=1, active=True,
            ))
            eid = f"e_{fid}"
            s.add(Embedding(
                id=eid, file_id=fid, embedding_type=etype,
                vector_table="vec_text", content_preview="x",
            ))
            # The impostors sit *closer* than the real match, which is
            # what makes an unfiltered fetch return only impostors.
            rows.append((eid, _vec(1.0).tobytes(), 0.0 + i * 0.1))
        s.commit()
    finally:
        s.close()

    with engine.begin() as conn:
        for eid, blob, dist in rows:
            conn.execute(
                text("INSERT INTO vec_text VALUES (:i, :v, :d)"),
                {"i": eid, "v": blob, "d": dist},
            )

    @contextmanager
    def _read():
        sess = maker()
        try:
            yield sess
        finally:
            sess.close()

    monkeypatch.setattr(search_mod, "get_search_engine", lambda: engine)
    monkeypatch.setattr(search_mod, "get_search_db_read", _read)
    # sqlite-vec's MATCH is not available here; order by the stored
    # distance column instead, which is what MATCH would have produced.
    monkeypatch.setattr(
        search_mod, "sql_text",
        lambda q: text(q.replace("WHERE vector MATCH :vec", "WHERE :vec IS NOT NULL")),
    )
    return engine


def test_only_the_requested_channel_is_scored(index):
    """Whisper rows must not be returned as TF-IDF evidence."""
    results = search_mod._find_similar_by_embedding(
        "src", "tfidf_keywords", 5, None,
    )

    assert [r["file_id"] for r in results] == ["kw_a"]


def test_a_channel_with_no_other_member_returns_nothing(index):
    """Better an empty second opinion than three impostors."""
    results = search_mod._find_similar_by_embedding(
        "whisper_a", "metadata", 5, None,
    )

    assert results == []
