"""A video's frame text is a ``screen_text`` passage, not a ``text`` one.

SPEC-CORE-048 C1/C2 and SPEC-ADDON-066 I10: the passage's start and end are the
stored segment's, its place is the chunk index, ``frame_url`` is at
``floor(start)``, a chunk with no stored segment makes no passage, and search by
image does not return videos.
"""

from __future__ import annotations

import numpy as np
import pytest

from app import search as search_mod
from app.config import SearchConfig
from app.search import MatchInfo
from tests.search_endpoint_db import (  # noqa: F401
    DRIVE,
    Seed,
    api,
    patch_everywhere,
    search_index,
)
from tests.vec_partition_db import at_distance, unit

FRAME_URL = "/api/addons/intelligence/files/{file_id}/frame?t={second}"


def _file(index, file_id: str, *, file_type: str = "document", mime: str = "text/markdown") -> None:
    from app.models import IndexedFile
    import app.database as database

    name = f"{file_id}.bin"
    with database.get_search_db() as session:
        session.add(IndexedFile(
            file_id=file_id, drive=DRIVE, filename=name,
            file_path=f"/drives/{DRIVE}/{name}", file_type=file_type,
            mime_type=mime, file_size=1, active=True,
        ))


def _distance(score: float) -> float:
    return float(np.sqrt(2.0 * (1.0 - score)))


def _vector(
    index, file_id: str, *, etype: str = "text_content", score: float = 0.90,
    axis: int = 2, table: str = "vec_text", preview: str = "warehouse preview",
    page: int | None = None, chunk_index: int | None = None,
) -> None:
    from app.models import Embedding
    import app.database as database

    emb_id = f"{file_id[:6]}-{etype[:8]}-{axis}"
    with database.get_search_db() as session:
        session.add(Embedding(
            id=emb_id, file_id=file_id, embedding_type=etype,
            vector_table=table, content_preview=preview,
            page=page, chunk_index=chunk_index,
        ))
    index.put_vector(
        table, emb_id, at_distance(index.dim(table), _distance(score), axis),
        partition=etype,
    )


def _stored_segment(
    seed, file_id: str, chunk_index: int, start: float, end: float, text: str,
) -> None:
    seed.index.sql(
        "INSERT INTO video_frame_texts"
        "(file_id, chunk_index, start_time, end_time, text) "
        "VALUES(:f, :c, :s, :e, :t)",
        f=file_id, c=chunk_index, s=start, e=end, t=text,
    )


LOOSE_SEARCH = dict(
    score_gap_threshold=0.0,
    score_cutoff_ratio=0.0,
    score_cutoff_margin=10.0,
    min_score_text=0.0,
    min_score_clip=0.0,
    min_score_clip_thumbnail=0.0,
)


@pytest.fixture()
def thresholds(monkeypatch, make_settings):
    patch_everywhere(
        monkeypatch, "settings",
        make_settings(search=SearchConfig(**LOOSE_SEARCH)),
    )


@pytest.fixture()
def embedded(search_index, monkeypatch):  # noqa: F811
    patch_everywhere(
        monkeypatch, "embed_query",
        lambda *_a, **_k: unit(search_index.dim("vec_text"), 0),
    )
    patch_everywhere(
        monkeypatch, "embed_text_clip",
        lambda *_a, **_k: unit(search_index.dim("vec_clip"), 0),
    )


@pytest.fixture()
def ready(thresholds, embedded):
    """Thresholds loose and a fixed query vector."""


@pytest.fixture()
def no_hydration(monkeypatch):
    async def _none(_ids):
        return {}

    patch_everywhere(monkeypatch, "hydrate_files", _none)


def _ok(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()


def _passages(body: dict, file_id: str) -> list[dict]:
    (hit,) = [r for r in body["results"] if r["file_id"] == file_id]
    return hit["passages"]


def test_spec_addon_066_i10_a_video_text_content_match_is_screen_text():
    """SPEC-CORE-048 C1: the kind is decided by the file type, not page."""
    match = MatchInfo(
        match_type="text_content", text="warehouse", score=0.9,
        chunk_index=3, channel="keyword",
    )

    assert search_mod._passage_kind(match, "video/mp4", "video") == "screen_text"


def test_spec_addon_066_i10_a_screen_text_passage_takes_the_stored_segment(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    """SPEC-CORE-048 C1/C2: start, end, chunk place, frame and link."""
    seed = Seed(search_index)
    _file(search_index, "vid", file_type="video", mime="video/mp4")
    seed.text_chunk("vid", 3, "warehouse on screen", page="")
    _stored_segment(seed, "vid", 3, 12.9, 20.0, "warehouse on screen")

    body = _ok(api("/search", {"q": "warehouse", "passages": "true"}))

    (passage,) = _passages(body, "vid")
    assert passage["kind"] == "screen_text"
    assert (passage["start"], passage["end"]) == (12.9, 20.0)
    assert passage["chunk_index"] == 3
    assert passage["page"] is None
    assert passage["section"] is None
    assert passage["frame_url"] == FRAME_URL.format(file_id="vid", second=12)
    assert passage["text"] == "warehouse on screen"
    assert passage["keyword"] is True


def test_spec_addon_066_i10_a_chunk_with_no_stored_segment_makes_no_passage(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    """SPEC-CORE-048 C1: no row in video_frame_texts means no place."""
    seed = Seed(search_index)
    _file(search_index, "vid", file_type="video", mime="video/mp4")
    seed.text_chunk("vid", 3, "warehouse on screen", page="")

    body = _ok(api("/search", {"q": "warehouse", "passages": "true"}))

    assert _passages(body, "vid") == []


def test_spec_addon_066_i10_a_vector_and_keyword_match_of_one_chunk_are_one_passage(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    """SPEC-CORE-048 C1 / I3: the chunk index is the passage's place."""
    seed = Seed(search_index)
    _file(search_index, "vid", file_type="video", mime="video/mp4")
    seed.text_chunk("vid", 3, "warehouse on screen", page="")
    _vector(search_index, "vid", chunk_index=3)
    _stored_segment(seed, "vid", 3, 12.9, 20.0, "warehouse on screen")

    body = _ok(api("/search", {"q": "warehouse", "passages": "true"}))

    (passage,) = _passages(body, "vid")
    assert passage["kind"] == "screen_text"
    assert passage["keyword"] is True


def test_spec_core_048_c3_search_by_image_leaves_out_videos(
    search_index, ready,  # noqa: F811
):
    """SPEC-CORE-048 C3: the image query does not return a video's text rows."""
    seed = Seed(search_index)
    _file(search_index, "vid", file_type="video", mime="video/mp4")
    seed.text_chunk("vid", 0, "warehouse")
    _vector(search_index, "vid", chunk_index=0)
    _file(search_index, "note")
    seed.text_chunk("note", 0, "warehouse")
    _vector(search_index, "note", chunk_index=0, axis=3)

    results = search_mod.search_documents(
        unit(search_index.dim("vec_text"), 0), DRIVE,
    )

    assert [result.file_id for result in results] == ["note"]
