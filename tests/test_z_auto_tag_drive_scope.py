"""Auto-tag candidates must be shaped by one drive only.

A drive is a security boundary: a viewer who can open drive A may have
no access to drive B at all. Three of the candidate channels consult
files *other* than the one being tagged —

- the tag vocabulary handed to the CLIP zero-shot scorer,
- the k-NN vote over visually similar already-tagged files,
- the corpus statistics behind TF-IDF keyword extraction —

so each of them can only read what belongs to the drive the file is
in. Nothing drive B contains may decide which words and tag names
surface while tagging a file in drive A.
"""

from __future__ import annotations

import logging
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

from app import tfidf as tfidf_mod  # noqa: E402
from app.models import Base, Embedding, IndexedFile  # noqa: E402
from app.workers import clip_concepts, tag_knn  # noqa: E402

DIM = 4


def _vec() -> np.ndarray:
    v = np.ones(DIM, dtype=np.float32)
    return v / np.linalg.norm(v)


class _Index:
    """A two-drive index: a search DB, a vec_clip stand-in, a Litloft DB."""

    def __init__(self, tmp_path):
        self.engine = create_engine(
            f"sqlite:///{tmp_path / 'search.db'}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self._maker = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE vec_clip (embedding_id TEXT PRIMARY KEY, "
                "vector BLOB, distance REAL)"
            ))

        self.litloft_engine = create_engine(
            f"sqlite:///{tmp_path / 'litloft.db'}",
            connect_args={"check_same_thread": False},
        )
        self._litloft_maker = sessionmaker(
            bind=self.litloft_engine, expire_on_commit=False
        )
        with self.litloft_engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE tags (id INTEGER PRIMARY KEY, "
                "name TEXT NOT NULL, drive TEXT NOT NULL)"
            ))
            conn.execute(text(
                "CREATE TABLE file_tags (file_id TEXT NOT NULL, "
                "tag_id INTEGER NOT NULL)"
            ))
        self._next_tag_id = 1

    def add_file(self, file_id: str, drive: str, *, distance: float | None):
        """Register a file, and give it one CLIP frame at ``distance``."""
        session = self._maker()
        try:
            session.add(IndexedFile(
                file_id=file_id, drive=drive, filename=f"{file_id}.mp4",
                file_path=f"/drives/{drive}/{file_id}.mp4", file_type="video",
                mime_type="video/mp4", file_size=1, active=True,
            ))
            if distance is not None:
                session.add(Embedding(
                    id=f"e_{file_id}", file_id=file_id, embedding_type="clip",
                    vector_table="vec_clip", content_preview="x",
                ))
            session.commit()
        finally:
            session.close()

        if distance is None:
            return
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO vec_clip VALUES (:i, :v, :d)"),
                {"i": f"e_{file_id}", "v": _vec().tobytes(), "d": distance},
            )

    def tag(self, file_id: str, drive: str, name: str):
        with self.litloft_engine.begin() as conn:
            tag_id = self._next_tag_id
            self._next_tag_id += 1
            conn.execute(
                text("INSERT INTO tags VALUES (:i, :n, :d)"),
                {"i": tag_id, "n": name, "d": drive},
            )
            conn.execute(
                text("INSERT INTO file_tags VALUES (:f, :t)"),
                {"f": file_id, "t": tag_id},
            )

    @contextmanager
    def search_db(self):
        session = self._maker()
        try:
            yield session
        finally:
            session.close()

    @contextmanager
    def litloft_db(self):
        session = self._litloft_maker()
        try:
            yield session
        finally:
            session.close()


@pytest.fixture()
def index(monkeypatch, tmp_path):
    idx = _Index(tmp_path)
    monkeypatch.setattr(tag_knn, "get_search_engine", lambda: idx.engine)
    monkeypatch.setattr(tag_knn, "get_search_db_read", idx.search_db)
    monkeypatch.setattr(tag_knn, "get_litloft_db", idx.litloft_db)
    monkeypatch.setattr(clip_concepts, "get_litloft_db", idx.litloft_db)
    monkeypatch.setattr(tfidf_mod, "get_search_db_read", idx.search_db)
    # sqlite-vec's MATCH is not available here; order by the stored
    # distance column instead, which is what MATCH would have produced.
    monkeypatch.setattr(
        tag_knn, "sql_text",
        lambda q: text(q.replace("WHERE vector MATCH :vec", "WHERE :vec IS NOT NULL")),
    )
    return idx


# ---------------------------------------------------------------------------
# k-NN neighbours
# ---------------------------------------------------------------------------


def test_a_closer_neighbour_in_another_drive_contributes_nothing(index):
    """The nearest file wins the vote only if it is in the same drive.

    The other drive's file carries a tag row labelled with *this*
    drive, which is what core's tags-table migration produces for a
    library that had tags before drives were partitioned. The tag-side
    filter cannot see through that, so the neighbour filter is what
    keeps the vote inside the boundary.
    """
    index.add_file("src", "a", distance=None)
    index.add_file("near_b", "b", distance=0.01)
    index.add_file("far_a", "a", distance=0.9)
    index.tag("near_b", "a", "leaked-from-b")
    index.tag("far_a", "a", "shared-in-a")

    result = tag_knn.recommend_tags_by_similarity(
        "src", drive="a", min_support=1, vectors=[_vec()]
    )

    assert [name for name, _ in result] == ["shared-in-a"]


def test_fetch_widens_until_in_drive_neighbours_survive(index):
    """A drive in the minority still gets its neighbours found."""
    index.add_file("src", "a", distance=None)
    # More other-drive frames than the first KNN page holds, all closer
    # than anything in the drive being tagged.
    for i in range(120):
        index.add_file(f"b{i}", "b", distance=0.01 + i * 0.0001)
    index.add_file("far_a", "a", distance=0.9)
    index.tag("far_a", "a", "shared-in-a")

    result = tag_knn.recommend_tags_by_similarity(
        "src", drive="a", k_neighbors=2, min_support=1, vectors=[_vec()]
    )

    assert [name for name, _ in result] == ["shared-in-a"]


def test_the_source_file_is_never_its_own_neighbour(index):
    index.add_file("src", "a", distance=0.0)
    index.tag("src", "a", "self-tag")

    result = tag_knn.recommend_tags_by_similarity(
        "src", drive="a", min_support=1, vectors=[_vec()]
    )

    assert result == []


def test_tag_lookup_ignores_rows_from_another_drive(index, caplog):
    """The tag-side check is the second layer behind the neighbour filter.

    A dropped row means core's tag rows disagree with the files they
    hang off about which drive they are in, so it is reported rather
    than swallowed — otherwise suggestions just go quiet.
    """
    index.add_file("f1", "a", distance=0.1)
    index.tag("f1", "a", "in-drive")
    index.tag("f1", "b", "out-of-drive")

    tag_knn._drift_reported.discard("a")
    with caplog.at_level(logging.WARNING, logger="app.workers.tag_knn"):
        tags = tag_knn._load_tags_for_files(["f1"], "a")
        first = caplog.text
        tag_knn._load_tags_for_files(["f1"], "a")

    assert tags == {"f1": ["in-drive"]}
    assert "labelled for another drive" in first
    # A property of the library, so it is not repeated per tagged file.
    assert caplog.text == first
    tag_knn._drift_reported.discard("a")


# ---------------------------------------------------------------------------
# CLIP concept vocabulary
# ---------------------------------------------------------------------------


def test_tag_vocabulary_holds_only_the_drives_own_tags(index):
    index.add_file("f1", "a", distance=None)
    index.tag("f1", "a", "料理")
    index.add_file("f2", "b", distance=None)
    index.tag("f2", "b", "確定申告")

    assert clip_concepts.load_user_tags("a") == ["料理"]
    assert clip_concepts.load_user_tags("b") == ["確定申告"]


def test_concept_cache_is_keyed_by_drive(
    index, monkeypatch, tmp_path, make_settings
):
    """Tagging drive A first must not leave A's tags in B's vocabulary."""
    import json

    preset = tmp_path / "concepts.json"
    preset.write_text(json.dumps({"x": ["preset"]}), encoding="utf-8")

    index.add_file("f1", "a", distance=None)
    index.tag("f1", "a", "料理")
    index.add_file("f2", "b", distance=None)
    index.tag("f2", "b", "確定申告")

    clip_concepts.reset_cache()
    monkeypatch.setattr(clip_concepts, "settings", make_settings())
    monkeypatch.setattr(
        clip_concepts, "_encode_concepts",
        lambda names: {n: _vec() for n in names},
    )

    vocab_a = clip_concepts.get_concept_embeddings(drive="a", preset_path=preset)
    vocab_b = clip_concepts.get_concept_embeddings(drive="b", preset_path=preset)

    assert sorted(vocab_a) == ["preset", "料理"]
    assert sorted(vocab_b) == ["preset", "確定申告"]
    clip_concepts.reset_cache()


# ---------------------------------------------------------------------------
# TF-IDF corpus statistics
# ---------------------------------------------------------------------------


def test_document_frequency_counts_only_the_drives_own_files(index, monkeypatch):
    """``min_doc_freq`` must not be satisfied by files in another drive."""
    index.add_file("a1", "a", distance=None)
    for i in range(3):
        index.add_file(f"b{i}", "b", distance=None)

    # "共通語" is the whole of drive a, but one file in three in drive b.
    monkeypatch.setattr(
        tfidf_mod,
        "_tokenize_file",
        lambda fid, fn, tok: ["共通語"] if fid in ("a1", "b0") else ["他"],
    )
    tfidf_mod.reset_corpus_idf_cache()

    idf_a, n_docs_a = tfidf_mod._build_corpus_idf("a")
    idf_b, n_docs_b = tfidf_mod._build_corpus_idf("b")

    assert n_docs_a == 1
    assert n_docs_b == 3
    # Ubiquitous in a, rare in b: neither drive's weight for the word
    # can be carried into the other.
    assert idf_a["共通語"] == pytest.approx(1.0)
    assert idf_b["共通語"] > idf_a["共通語"]
    tfidf_mod.reset_corpus_idf_cache()
