"""A similar-files result says which words it shares with the source.

The words come from the keyword bag each transcribed file already
stores at index time (``Embedding.content_preview`` of the
``tfidf_keywords`` row), so answering costs one indexed lookup and no
Janome. The per-word TF-IDF scores are not stored, so none are
reported — only the words, in the source's own ranking order.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app import search as search_mod  # noqa: E402
from app.models import Base, Embedding, IndexedFile  # noqa: E402


def _bag(*words: str) -> str:
    return " ".join(words)


@pytest.fixture()
def index(monkeypatch, tmp_path):
    """A library of transcribed videos, each with its keyword bag."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine, expire_on_commit=False)

    bags = {
        # Ranked by TF-IDF at index time; the stored order is that ranking.
        "src": _bag("味噌", "発酵", "麹", "保存"),
        "kw_hit": _bag("保存", "発酵", "冷凍"),
        "vis_hit": _bag("麹", "味噌", "撮影"),
        "no_overlap": _bag("登山", "テント"),
        "empty_bag": "",
    }

    s = maker()
    try:
        for fid, preview in bags.items():
            s.add(IndexedFile(
                file_id=fid, drive="d", filename=f"{fid}.mp4",
                file_path=f"/drives/{fid}.mp4", file_type="video",
                mime_type="video/mp4", file_size=1, active=True,
            ))
            s.add(Embedding(
                id=f"tk_{fid}", file_id=fid,
                embedding_type="tfidf_keywords",
                vector_table="vec_text", content_preview=preview,
            ))
        # A document has no keyword bag at all.
        s.add(IndexedFile(
            file_id="doc", drive="d", filename="doc.md",
            file_path="/drives/doc.md", file_type="document",
            mime_type="text/markdown", file_size=1, active=True,
        ))
        s.commit()
    finally:
        s.close()

    @contextmanager
    def _read():
        sess = maker()
        try:
            yield sess
        finally:
            sess.close()

    monkeypatch.setattr(search_mod, "get_search_db_read", _read)
    search_mod.invalidate_similar_cache()
    return maker


def _result(file_id: str, score: float) -> dict:
    return {
        "file_id": file_id,
        "drive": "d",
        "filename": f"{file_id}.mp4",
        "file_type": "video",
        "mime_type": "video/mp4",
        "score": score,
    }


def _legs(monkeypatch, primary: list[dict], secondary: list[dict]) -> None:
    """Pin what each embedding leg returns, so the merge is the only variable."""
    def _fake(file_id, embedding_type, limit, drive):
        return secondary if embedding_type == "tfidf_keywords" else primary

    monkeypatch.setattr(search_mod, "_find_similar_by_embedding", _fake)


def test_shared_words_follow_the_source_ranking(index, monkeypatch):
    """The order is the source's TF-IDF order, not the candidate's."""
    _legs(monkeypatch, [], [_result("kw_hit", 0.8)])

    result = search_mod.find_similar("src", limit=6, drive="d")

    assert [r.file_id for r in result.results] == ["kw_hit"]
    assert [kw["word"] for kw in result.results[0].shared_keywords] == [
        "発酵", "保存",
    ]


def test_no_score_is_invented(index, monkeypatch):
    """Per-word TF-IDF is not stored, so the payload carries words only."""
    _legs(monkeypatch, [], [_result("kw_hit", 0.8)])

    result = search_mod.find_similar("src", limit=6, drive="d")

    assert result.results[0].shared_keywords[0] == {"word": "発酵"}
    assert [kw["word"] for kw in result.source_keywords] == [
        "味噌", "発酵", "麹", "保存",
    ]


def test_a_visual_only_match_still_reports_its_overlap(index, monkeypatch):
    """Sharing words is a fact about the pair, not about the leg that found it."""
    _legs(monkeypatch, [_result("vis_hit", 0.9)], [])

    result = search_mod.find_similar("src", limit=6, drive="d")

    assert result.results[0].match_type == "clip_thumbnail"
    assert [kw["word"] for kw in result.results[0].shared_keywords] == [
        "味噌", "麹",
    ]


def test_a_candidate_sharing_nothing_reports_nothing(index, monkeypatch):
    """No overlap is reported as no words, never as a weak guess."""
    _legs(monkeypatch, [_result("no_overlap", 0.9)], [_result("empty_bag", 0.8)])

    result = search_mod.find_similar("src", limit=6, drive="d")

    assert {r.file_id for r in result.results} == {"no_overlap", "empty_bag"}
    assert all(r.shared_keywords == () for r in result.results)


def test_a_source_without_a_keyword_bag_reports_nothing(index, monkeypatch):
    """Documents are never keyword-indexed; the lookup must stay quiet."""
    _legs(monkeypatch, [_result("kw_hit", 0.9)], [])

    result = search_mod.find_similar("doc", limit=6, drive="d")

    assert result.source_keywords == ()
    assert result.results[0].shared_keywords == ()


def test_a_truncated_bag_drops_its_final_word(index, monkeypatch):
    """content_preview is capped at 200 characters, which can cut a word."""
    long_bag = "あ" * 196 + " 発酵子"  # 200 chars: the tail word is half of one
    assert len(long_bag) == search_mod._KEYWORD_PREVIEW_MAX

    session = index()
    try:
        emb = session.query(Embedding).filter_by(file_id="kw_hit").one()
        emb.content_preview = long_bag
        session.commit()
    finally:
        session.close()

    assert search_mod._parse_keyword_preview(long_bag) == ["あ" * 196]
    # A bag that stopped short of the cap keeps every word.
    assert search_mod._parse_keyword_preview("発酵 保存") == ["発酵", "保存"]

    _legs(monkeypatch, [], [_result("kw_hit", 0.8)])
    result = search_mod.find_similar("src", limit=6, drive="d")

    assert result.results[0].shared_keywords == ()


def test_the_word_list_is_capped(index, monkeypatch):
    """A pair that shares its whole bag still reports a bounded list."""
    words = [f"w{i}" for i in range(30)]
    session = index()
    try:
        for fid in ("src", "kw_hit"):
            emb = session.query(Embedding).filter_by(file_id=fid).one()
            emb.content_preview = " ".join(words)
        session.commit()
    finally:
        session.close()

    _legs(monkeypatch, [], [_result("kw_hit", 0.8)])
    result = search_mod.find_similar("src", limit=6, drive="d")

    shared = result.results[0].shared_keywords
    assert len(shared) == search_mod._SHARED_KEYWORDS_MAX
    assert [kw["word"] for kw in shared] == words[:search_mod._SHARED_KEYWORDS_MAX]
