"""SPEC-ADDON-048: ``passages=true`` makes ``/search`` describe the places it found.

A passage is one distinct (kind, place) among a hit's matches, with its own
score, excerpt and locator. Without the parameter the answer is the one it was
before the parameter existed; the other search routes never carry a passage.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import app.database as database
from app import search as search_mod
from app.config import SearchConfig
from app.models import Embedding, IndexedFile
from tests.matching_core_double import MatchingCore, install
from tests.search_endpoint_db import (  # noqa: F401
    DRIVE,
    Seed,
    api,
    patch_everywhere,
    search_index,
)
from tests.vec_partition_db import at_distance, unit

EPUB_MIME = "application/epub+zip"
LOFT_MIME = "application/vnd.litloft.loft+json"
FRAME_URL = "/api/addons/intelligence/files/{file_id}/frame?t={second}"

HIT_FIELDS = {
    "file_id", "drive", "filename", "file_type", "score", "match_types",
    "segments", "file",
}
SEGMENT_FIELDS = {"time_range", "matches"}
MATCH_FIELDS = {"type", "text", "score", "page", "section", "section_title"}
PASSAGE_FIELDS = {
    "kind", "start", "end", "page", "section", "section_title", "chunk_index",
    "text", "score", "keyword", "highlight_terms", "frame_url",
}


# ---------------------------------------------------------------------------
# Seeding and reading
# ---------------------------------------------------------------------------


def _ok(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()


def _hit(body: dict, file_id: str) -> dict:
    hits = [r for r in body["results"] if r["file_id"] == file_id]
    assert len(hits) == 1, body
    return hits[0]


def _passages(body: dict, file_id: str) -> list[dict]:
    return _hit(body, file_id)["passages"]


def _matches(body: dict, file_id: str, match_type: str) -> list[dict]:
    return [
        match
        for segment in _hit(body, file_id)["segments"]
        for match in segment["matches"]
        if match["type"] == match_type
    ]


def _file(
    index, file_id: str, *, file_type: str = "document",
    mime: str = "text/markdown", drive: str = DRIVE,
) -> None:
    name = f"{file_id}.bin"
    with database.get_search_db() as session:
        session.add(IndexedFile(
            file_id=file_id, drive=drive, filename=name,
            file_path=f"/drives/{drive}/{name}", file_type=file_type,
            mime_type=mime, file_size=1, active=True,
        ))


def _distance(score: float) -> float:
    return float(np.sqrt(2.0 * (1.0 - score)))


def _vector(
    index, file_id: str, *, etype: str, score: float, axis: int,
    table: str = "vec_text", preview: str = "stored preview",
    start: float | None = None, end: float | None = None,
    page: int | None = None, chunk_index: int | None = None,
) -> None:
    emb_id = f"{file_id[:6]}-{etype[:8]}-{axis}"
    with database.get_search_db() as session:
        session.add(Embedding(
            id=emb_id, file_id=file_id, embedding_type=etype,
            vector_table=table, content_preview=preview,
            timestamp_start=start, timestamp_end=end,
            page=page, chunk_index=chunk_index,
        ))
    index.put_vector(
        table, emb_id,
        at_distance(index.dim(table), _distance(score), axis),
        partition=etype,
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
    """A seeded match is a hit: no gap check, no margin and no absolute floor."""
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
    """Thresholds loose and a fixed query vector, so one seeded match is one hit."""


@pytest.fixture()
def no_hydration(monkeypatch):
    async def _none(_ids):
        return {}

    patch_everywhere(monkeypatch, "hydrate_files", _none)


@pytest.fixture()
def hydration(monkeypatch):
    async def _ids(ids):
        return {file_id: {"id": file_id} for file_id in ids}

    patch_everywhere(monkeypatch, "hydrate_files", _ids)


@pytest.fixture()
def core(monkeypatch):
    return install(monkeypatch, MatchingCore())


# ---------------------------------------------------------------------------
# SPEC-ADDON-048 I1: no parameter, no passages key; the other routes never
# ---------------------------------------------------------------------------


def _i1_world(seed, index) -> None:
    _file(index, "doc", mime="application/pdf")
    seed.text_chunk("doc", 0, "warehouse invoice", page="3")
    seed.file_metadata("doc", "warehouse", "")
    _file(index, "pic", file_type="image", mime="image/jpeg")
    _vector(
        index, "pic", etype="clip_thumbnail", table="vec_clip", score=0.90, axis=1,
    )


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"q": "warehouse"}, id="absent"),
        pytest.param({"q": "warehouse", "passages": "false"}, id="false"),
    ],
)
def test_spec_addon_048_i1_search_without_the_parameter_has_no_passages_key(
    api, search_index, ready, no_hydration, params,  # noqa: F811
):
    _i1_world(Seed(search_index), search_index)

    body = _ok(api("/search", params))

    assert body["results"]
    for hit in body["results"]:
        assert set(hit) == HIT_FIELDS
        assert hit["file"] is None
        for segment in hit["segments"]:
            assert set(segment) == SEGMENT_FIELDS
            for match in segment["matches"]:
                assert set(match) == MATCH_FIELDS


def test_spec_addon_048_i1_passages_false_is_the_same_bytes_as_no_parameter(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    _i1_world(Seed(search_index), search_index)

    absent = api("/search", {"q": "warehouse"})
    explicit = api("/search", {"q": "warehouse", "passages": "false"})

    assert absent.status_code == explicit.status_code == 200
    assert absent.content == explicit.content


def test_spec_addon_048_i1_search_images_never_carries_passages(
    api, search_index, ready, hydration,  # noqa: F811
):
    _i1_world(Seed(search_index), search_index)

    body = _ok(api("/search/images", {"q": "warehouse", "passages": "true"}))

    hit = _hit(body, "pic")
    assert set(hit) == HIT_FIELDS
    (segment,) = hit["segments"]
    (match,) = segment["matches"]
    assert match["page"] is None
    assert match["section"] is None
    assert match["section_title"] is None


def test_spec_addon_048_i1_search_compare_never_carries_passages(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    _i1_world(Seed(search_index), search_index)

    body = _ok(api("/search/compare", {"q": "warehouse", "passages": "true"}))

    for answer in ("rrf", "cosine", "rrf_no_cutoff", "cosine_no_cutoff"):
        assert body[answer]["results"], answer
        for hit in body[answer]["results"]:
            assert set(hit) == HIT_FIELDS
            for segment in hit["segments"]:
                assert set(segment) == SEGMENT_FIELDS
                for match in segment["matches"]:
                    assert set(match) == MATCH_FIELDS


# ---------------------------------------------------------------------------
# SPEC-ADDON-048 I2: the rest of the answer does not move
# ---------------------------------------------------------------------------


def _i2_world(seed, index) -> None:
    _file(index, "vid", file_type="video", mime="video/mp4")
    seed.transcript("vid", 0, "warehouse rail", times=(10.0, 20.0))
    _vector(
        index, "vid", etype="whisper", score=0.93, axis=2,
        start=10.0, end=20.0, preview="warehouse rail",
    )
    _vector(
        index, "vid", etype="clip_thumbnail", table="vec_clip", score=0.90, axis=3,
    )

    _file(index, "book", mime=EPUB_MIME)
    seed.text_chunk("book", 0, "warehouse chapter", page="3")
    seed.section("book", 3, "The Warehouse")

    _file(index, "pdf", mime="application/pdf")
    seed.text_chunk("pdf", 0, "warehouse page", page="7")

    _file(index, "note")
    seed.text_chunk("note", 0, "warehouse note")
    seed.file_metadata("note", "warehouse note", "warehouse")


def test_spec_addon_048_i2_the_answer_apart_from_passages_is_unchanged(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    _i2_world(Seed(search_index), search_index)
    params = {"q": "warehouse", "limit": 100, "include_scene_clip": "true"}

    plain = _ok(api("/search", params))
    with_passages = _ok(api("/search", {**params, "passages": "true"}))

    for hit in with_passages["results"]:
        assert isinstance(hit["passages"], list)
        del hit["passages"]
    assert with_passages == plain


def test_spec_addon_047_i3_a_filtered_search_lists_the_passages_of_the_filtered_hits(
    api, search_index, ready, core,  # noqa: F811
):
    seed = Seed(search_index)
    for file_id, start in (("inside", 10.0), ("outside", 20.0)):
        _file(search_index, file_id, file_type="video", mime="video/mp4")
        seed.transcript(file_id, 0, "warehouse", times=(start, start + 5.0))
    core.file_ids = ["inside"]

    body = _ok(api("/search", {
        "q": "warehouse",
        "filter": json.dumps({"tags": ["x"]}),
        "passages": "true",
    }))

    assert [hit["file_id"] for hit in body["results"]] == ["inside"]
    (passage,) = _passages(body, "inside")
    assert passage["kind"] == "utterance"
    assert passage["start"] == 10.0


# ---------------------------------------------------------------------------
# SPEC-ADDON-048 I3: one entry per (kind, place), and the table is closed
# ---------------------------------------------------------------------------


def test_spec_addon_048_i3_a_vector_and_a_keyword_match_of_one_transcript_stretch_are_one_passage(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    seed = Seed(search_index)
    _file(search_index, "vid", file_type="video", mime="video/mp4")
    seed.transcript("vid", 3, "warehouse rail", times=(10.0, 20.0))
    _vector(
        search_index, "vid", etype="whisper", score=0.93, axis=2,
        start=10.0, end=20.0,
    )

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    (passage,) = _passages(body, "vid")
    assert set(passage) == PASSAGE_FIELDS
    assert passage["kind"] == "utterance"
    assert (passage["start"], passage["end"]) == (10.0, 20.0)
    assert passage["keyword"] is True


def test_spec_addon_048_i3_a_vector_and_a_keyword_match_of_one_text_chunk_are_one_passage(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    seed = Seed(search_index)
    _file(search_index, "pdf", mime="application/pdf")
    seed.text_chunk("pdf", 4, "warehouse invoice", page="2")
    _vector(
        search_index, "pdf", etype="text_content", score=0.90, axis=2,
        page=2, chunk_index=4,
    )

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    (passage,) = _passages(body, "pdf")
    assert passage["kind"] == "page"
    assert passage["page"] == 2
    assert passage["chunk_index"] == 4
    assert passage["keyword"] is True


def test_spec_addon_048_i3_two_places_of_one_file_are_two_passages(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    seed = Seed(search_index)
    _file(search_index, "vid", file_type="video", mime="video/mp4")
    seed.transcript("vid", 0, "warehouse first", times=(10.0, 20.0))
    seed.transcript("vid", 5, "warehouse second", times=(90.0, 100.0))

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    passages = _passages(body, "vid")
    assert [passage["kind"] for passage in passages] == ["utterance", "utterance"]
    assert sorted(passage["start"] for passage in passages) == [10.0, 90.0]


def _whisper_match(seed, index) -> None:
    _file(index, "f", file_type="video", mime="video/mp4")
    _vector(index, "f", etype="whisper", score=0.93, axis=2, start=10.0, end=20.0)


def _transcript_match(seed, index) -> None:
    _file(index, "f", file_type="video", mime="video/mp4")
    seed.transcript("f", 0, "warehouse", times=(10.0, 20.0))


def _scene_match(seed, index) -> None:
    _file(index, "f", file_type="video", mime="video/mp4")
    _vector(
        index, "f", etype="clip", table="vec_clip", score=0.90, axis=2,
        start=30.0, end=60.0,
    )


def _epub_match(seed, index) -> None:
    _file(index, "f", mime=EPUB_MIME)
    seed.text_chunk("f", 0, "warehouse chapter", page="3")
    seed.section("f", 3, "The Warehouse")


def _pdf_match(seed, index) -> None:
    _file(index, "f", mime="application/pdf")
    seed.text_chunk("f", 0, "warehouse page", page="7")


def _picture_match(seed, index) -> None:
    _file(index, "f", file_type="image", mime="image/jpeg")
    seed.text_chunk("f", 0, "warehouse sign")


def _text_match(seed, index) -> None:
    _file(index, "f")
    seed.text_chunk("f", 0, "warehouse note")


KIND_TABLE = [
    pytest.param("utterance", _whisper_match, {}, id="whisper-vector"),
    pytest.param("utterance", _transcript_match, {}, id="transcript-keyword"),
    pytest.param(
        "scene", _scene_match, {"include_scene_clip": "true"}, id="scene-clip",
    ),
    pytest.param("section", _epub_match, {}, id="epub-chunk"),
    pytest.param("page", _pdf_match, {}, id="paged-chunk"),
    pytest.param("picture_text", _picture_match, {}, id="picture-text"),
    pytest.param("text", _text_match, {}, id="plain-text"),
]


@pytest.mark.parametrize("kind, seed_one, extra", KIND_TABLE)
def test_spec_addon_048_i3_each_match_type_in_the_table_makes_its_kind(
    api, search_index, ready, no_hydration, kind, seed_one, extra,  # noqa: F811
):
    seed_one(Seed(search_index), search_index)

    body = _ok(api("/search", {"passages": "true", "q": "warehouse", **extra}))

    assert [passage["kind"] for passage in _passages(body, "f")] == [kind]


def _metadata_only(seed, index) -> None:
    _file(index, "f")
    seed.file_metadata("f", "warehouse", "")


def _clip_thumbnail_only(seed, index) -> None:
    _file(index, "f", file_type="image", mime="image/jpeg")
    _vector(
        index, "f", etype="clip_thumbnail", table="vec_clip", score=0.90, axis=1,
    )


def _embedding_only(etype: str):
    def _seed_one(seed, index) -> None:
        _file(index, "f")
        _vector(index, "f", etype=etype, score=0.90, axis=2)

    return _seed_one


def _retrieval_keywords_only(seed, index) -> None:
    _file(index, "f")
    index.sql(
        "INSERT INTO fts_retrieval_keywords(file_id, keywords) VALUES (:f, :k)",
        f="f", k="warehouse",
    )


CLOSED_TABLE = [
    pytest.param(_metadata_only, id="metadata-keyword"),
    pytest.param(_clip_thumbnail_only, id="clip-thumbnail"),
    pytest.param(_retrieval_keywords_only, id="retrieval-keywords"),
    pytest.param(_embedding_only("tfidf_keywords"), id="tfidf-keywords"),
    pytest.param(_embedding_only("vision_description"), id="vision-description"),
    pytest.param(_embedding_only("blip_caption"), id="blip-caption"),
]


@pytest.mark.parametrize("seed_one", CLOSED_TABLE)
def test_spec_addon_048_i3_a_match_type_outside_the_table_makes_no_passage(
    api, search_index, ready, no_hydration, seed_one,  # noqa: F811
):
    seed_one(Seed(search_index), search_index)

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    assert _hit(body, "f")["passages"] == []


def test_spec_addon_048_i3_an_untimed_transcript_match_makes_no_passage(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    seed = Seed(search_index)
    _file(search_index, "vid", file_type="video", mime="video/mp4")
    seed.transcript("vid", 0, "warehouse", times=None)

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    assert _hit(body, "vid")["passages"] == []


def _keyword_chunk_without_index(seed, index) -> None:
    _file(index, "f")
    index.sql(
        "INSERT INTO fts_text_content(file_id, chunk_index, page, text) "
        "VALUES ('f', 'unknown', '', 'warehouse')"
    )


def _vector_chunk_without_index(seed, index) -> None:
    _file(index, "f")
    _vector(index, "f", etype="text_content", score=0.90, axis=2, chunk_index=None)


@pytest.mark.parametrize(
    "seed_one",
    [
        pytest.param(_keyword_chunk_without_index, id="keyword"),
        pytest.param(_vector_chunk_without_index, id="vector"),
    ],
)
def test_spec_addon_048_i3_a_text_match_without_a_chunk_index_makes_no_passage(
    api, search_index, ready, no_hydration, seed_one,  # noqa: F811
):
    seed_one(Seed(search_index), search_index)

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    assert _hit(body, "f")["passages"] == []


# ---------------------------------------------------------------------------
# SPEC-ADDON-048 I15: section, page and chunk_index
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title, expected_title",
    [
        pytest.param("The Warehouse", "The Warehouse", id="titled"),
        pytest.param(None, None, id="untitled"),
    ],
)
def test_spec_addon_048_i15_an_epub_chunk_is_a_section_passage(
    api, search_index, ready, no_hydration, title, expected_title,  # noqa: F811
):
    seed = Seed(search_index)
    _file(search_index, "book", mime=EPUB_MIME)
    seed.text_chunk("book", 6, "warehouse chapter", page="3")
    if title is not None:
        seed.section("book", 3, title)

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    (passage,) = _passages(body, "book")
    assert passage["kind"] == "section"
    assert passage["section"] == 3
    assert passage["section_title"] == expected_title
    assert passage["page"] is None
    assert passage["chunk_index"] == 6


def test_spec_addon_048_i15_a_paged_chunk_outside_an_epub_is_a_page_passage(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    seed = Seed(search_index)
    _file(search_index, "pdf", mime="application/pdf")
    seed.text_chunk("pdf", 4, "warehouse page", page="7")

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    (passage,) = _passages(body, "pdf")
    assert passage["kind"] == "page"
    assert passage["page"] == 7
    assert passage["section"] is None
    assert passage["section_title"] is None
    assert passage["chunk_index"] == 4


@pytest.mark.parametrize(
    "file_type, mime, kind",
    [
        pytest.param("image", "image/jpeg", "picture_text", id="picture-text"),
        pytest.param("document", "text/markdown", "text", id="text"),
    ],
)
def test_spec_addon_048_i15_picture_text_and_text_carry_the_text_chunk_index(
    api, search_index, ready, no_hydration, file_type, mime, kind,  # noqa: F811
):
    seed = Seed(search_index)
    _file(search_index, "f", file_type=file_type, mime=mime)
    seed.text_chunk("f", 9, "warehouse")

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    (passage,) = _passages(body, "f")
    assert passage["kind"] == kind
    assert passage["chunk_index"] == 9
    assert passage["page"] is None
    assert passage["section"] is None
    assert passage["section_title"] is None


@pytest.mark.parametrize(
    "seed_one, expected",
    [
        pytest.param(lambda seed, index: (
            _file(index, "vid", file_type="video", mime="video/mp4"),
            seed.transcript("vid", 3, "warehouse", times=(10.0, 20.0)),
            _vector(
                index, "vid", etype="whisper", score=0.93, axis=2,
                start=10.0, end=20.0,
            ),
        ), 3, id="keyword-supplied"),
        pytest.param(lambda seed, index: (
            _file(index, "vid", file_type="video", mime="video/mp4"),
            _vector(
                index, "vid", etype="whisper", score=0.93, axis=2,
                start=10.0, end=20.0,
            ),
        ), None, id="vector-only"),
    ],
)
def test_spec_addon_048_i15_an_utterance_carries_only_a_keyword_transcript_chunk_index(
    api, search_index, ready, no_hydration, seed_one, expected,  # noqa: F811
):
    seed_one(Seed(search_index), search_index)

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    (passage,) = _passages(body, "vid")
    assert passage["kind"] == "utterance"
    assert passage["chunk_index"] == expected


# ---------------------------------------------------------------------------
# SPEC-ADDON-048 I4: the score and the order
# ---------------------------------------------------------------------------


def test_spec_addon_048_i4_a_passage_score_is_alpha_times_vector_plus_keyword(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    seed = Seed(search_index)
    _file(search_index, "vid", file_type="video", mime="video/mp4")
    seed.transcript("vid", 3, "warehouse rail", times=(10.0, 20.0))
    _vector(
        search_index, "vid", etype="whisper", score=0.93, axis=2,
        start=10.0, end=20.0,
    )

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    alpha = search_mod.settings.search.alpha
    (vector,) = _matches(body, "vid", "whisper")
    (keyword,) = _matches(body, "vid", "transcript")
    (passage,) = _passages(body, "vid")
    expected = alpha * vector["score"] + (1.0 - alpha) * keyword["score"]
    assert passage["score"] == pytest.approx(expected, abs=1e-3)


def test_spec_addon_048_i4_a_vector_score_is_weighted_as_the_file_score_weights_it(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    _file(search_index, "pdf", mime="application/pdf")
    _vector(
        search_index, "pdf", etype="text_content", score=0.90, axis=2,
        page=2, chunk_index=0,
    )

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    config = search_mod.settings.search
    (match,) = _matches(body, "pdf", "text_content")
    (passage,) = _passages(body, "pdf")
    expected = config.alpha * config.type_weight_text_content * match["score"]
    assert passage["score"] == pytest.approx(expected, abs=1e-3)


def test_spec_addon_048_i4_a_passage_takes_the_best_of_its_own_matches(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    _file(search_index, "vid", file_type="video", mime="video/mp4")
    _vector(
        search_index, "vid", etype="whisper", score=0.95, axis=2,
        start=10.0, end=20.0,
    )
    _vector(
        search_index, "vid", etype="whisper", score=0.88, axis=3,
        start=10.0, end=20.0,
    )

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    alpha = search_mod.settings.search.alpha
    best = max(match["score"] for match in _matches(body, "vid", "whisper"))
    (passage,) = _passages(body, "vid")
    assert passage["score"] == pytest.approx(alpha * best, abs=1e-3)


def test_spec_addon_048_i4_passages_are_sorted_by_score_descending(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    _file(search_index, "vid", file_type="video", mime="video/mp4")
    _vector(
        search_index, "vid", etype="whisper", score=0.94, axis=2,
        start=100.0, end=110.0,
    )
    _vector(
        search_index, "vid", etype="whisper", score=0.89, axis=3,
        start=10.0, end=20.0,
    )

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    passages = _passages(body, "vid")
    assert [passage["start"] for passage in passages] == [100.0, 10.0]
    assert [passage["score"] for passage in passages] == sorted(
        (passage["score"] for passage in passages), reverse=True,
    )


def test_spec_addon_048_i4_equal_scores_are_ordered_by_start(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    _file(search_index, "vid", file_type="video", mime="video/mp4")
    _vector(
        search_index, "vid", etype="whisper", score=0.90, axis=2,
        start=30.0, end=40.0,
    )
    _vector(
        search_index, "vid", etype="whisper", score=0.90, axis=3,
        start=10.0, end=20.0,
    )

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    assert [passage["start"] for passage in _passages(body, "vid")] == [10.0, 30.0]


def test_spec_addon_048_i4_equal_scores_are_ordered_by_chunk_index(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    _file(search_index, "note")
    _vector(
        search_index, "note", etype="text_content", score=0.90, axis=2,
        chunk_index=2,
    )
    _vector(
        search_index, "note", etype="text_content", score=0.90, axis=3,
        chunk_index=1,
    )

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    assert [
        passage["chunk_index"] for passage in _passages(body, "note")
    ] == [1, 2]


# ---------------------------------------------------------------------------
# SPEC-ADDON-048 I5: keyword, highlight terms and the stored preview
# ---------------------------------------------------------------------------


HIGHLIGHT_TERMS = [
    pytest.param("warehouse", ["warehouse"], id="ascii"),
    pytest.param("みかん", ["みかん", "ミカン"], id="hiragana"),
    pytest.param("ミカン", ["みかん", "ミカン"], id="katakana"),
    pytest.param(
        "warehouse そうこ", ["warehouse", "そうこ", "ソウコ"], id="two-terms",
    ),
    pytest.param("みかん ミカン", ["みかん", "ミカン"], id="kana-variants-deduped"),
    pytest.param("hello ばなな", ["hello", "ばなな", "バナナ"], id="mixed-script"),
    pytest.param(
        "みかん先輩", ["みかん先輩", "ミカン先輩"], id="term-with-both-kana",
    ),
]


@pytest.mark.parametrize("query, expected", HIGHLIGHT_TERMS)
def test_spec_addon_048_i5_highlight_terms_are_the_query_terms_with_their_kana_variants(
    api, search_index, ready, no_hydration, query, expected,  # noqa: F811
):
    seed = Seed(search_index)
    _file(search_index, "f")
    seed.text_chunk("f", 0, query)

    body = _ok(api("/search", {"passages": "true", "q": query}))

    (passage,) = _passages(body, "f")
    assert passage["keyword"] is True
    assert sorted(passage["highlight_terms"]) == expected


def test_spec_addon_048_i5_a_vector_only_passage_is_not_keyword_and_keeps_the_stored_preview(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    _file(search_index, "note")
    _vector(
        search_index, "note", etype="text_content", score=0.90, axis=2,
        chunk_index=0, preview="the stored content preview",
    )

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    (passage,) = _passages(body, "note")
    assert passage["keyword"] is False
    assert passage["highlight_terms"] == []
    assert passage["text"] == "the stored content preview"


def test_spec_addon_048_i5_a_keyword_match_elsewhere_in_the_file_does_not_mark_another_passage(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    seed = Seed(search_index)
    _file(search_index, "vid", file_type="video", mime="video/mp4")
    seed.transcript("vid", 0, "warehouse", times=(10.0, 20.0))
    _vector(
        search_index, "vid", etype="text_content", score=0.90, axis=2,
        chunk_index=5, preview="unrelated body",
    )

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    by_kind = {passage["kind"]: passage for passage in _passages(body, "vid")}
    assert by_kind["utterance"]["keyword"] is True
    assert by_kind["text"]["keyword"] is False
    assert by_kind["text"]["highlight_terms"] == []


# ---------------------------------------------------------------------------
# SPEC-ADDON-048 I6: the excerpt window
# ---------------------------------------------------------------------------


SHORT = "the warehouse is here"
INSIDE = "a" * 150 + "warehouse" + "b" * 250
STRADDLING = "a" * 195 + "warehouse" + "b" * 300
PAST = "a" * 300 + "warehouse" + "b" * 300
TAIL = "a" * 300 + "warehouse"
TWO_TERMS = "a" * 250 + "rail" + "b" * 146 + "warehouse" + "c" * 100

EXCERPTS = [
    pytest.param("warehouse", SHORT, SHORT, id="stored-text-is-short"),
    pytest.param(
        "warehouse", INSIDE, INSIDE[:200], id="wholly-inside-the-first-200",
    ),
    pytest.param(
        "warehouse", STRADDLING,
        "…" + STRADDLING[195 - 40:195 - 40 + 200] + "…",
        id="straddles-the-200th-character",
    ),
    pytest.param(
        "warehouse", PAST,
        "…" + PAST[300 - 40:300 - 40 + 200] + "…",
        id="window-around-an-occurrence-past-the-first-200",
    ),
    pytest.param(
        "warehouse", TAIL,
        "…" + TAIL[300 - 40:],
        id="window-reaches-the-end",
    ),
    pytest.param(
        "warehouse rail", TWO_TERMS,
        "…" + TWO_TERMS[250 - 40:250 - 40 + 200] + "…",
        id="window-starts-at-the-earliest-term",
    ),
]


@pytest.mark.parametrize("query, stored, expected", EXCERPTS)
def test_spec_addon_048_i6_a_keyword_passage_excerpt_is_the_window_around_its_match(
    api, search_index, ready, no_hydration, query, stored, expected,  # noqa: F811
):
    seed = Seed(search_index)
    _file(search_index, "note")
    seed.text_chunk("note", 0, stored)

    body = _ok(api("/search", {"passages": "true", "q": query}))

    (passage,) = _passages(body, "note")
    assert passage["keyword"] is True
    assert passage["text"] == expected


# ---------------------------------------------------------------------------
# SPEC-ADDON-048 I7: frame_url, and scenes only with the scene toggle
# ---------------------------------------------------------------------------


def test_spec_addon_048_i7_a_video_utterance_carries_its_frame_at_the_floored_start(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    seed = Seed(search_index)
    _file(search_index, "vid", file_type="video", mime="video/mp4")
    seed.transcript("vid", 0, "warehouse", times=(12.9, 20.0))

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    (passage,) = _passages(body, "vid")
    assert passage["kind"] == "utterance"
    assert passage["frame_url"] == FRAME_URL.format(file_id="vid", second=12)


def test_spec_addon_048_i7_a_scene_passage_carries_its_frame_and_the_stored_end(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    _file(search_index, "vid", file_type="video", mime="video/mp4")
    _vector(
        search_index, "vid", etype="clip", table="vec_clip", score=0.90, axis=2,
        start=30.7, end=60.7,
    )

    body = _ok(api("/search", {"passages": "true", "q": "warehouse", "include_scene_clip": "true"}))

    (passage,) = _passages(body, "vid")
    assert passage["kind"] == "scene"
    assert (passage["start"], passage["end"]) == (30.7, 60.7)
    assert passage["chunk_index"] is None
    assert passage["text"] == ""
    assert passage["frame_url"] == FRAME_URL.format(file_id="vid", second=30)


def _audio_utterance(seed, index) -> None:
    _file(index, "f", file_type="audio", mime="audio/mpeg")
    seed.transcript("f", 0, "warehouse", times=(10.0, 20.0))


def _loft_utterance(seed, index) -> None:
    _file(index, "f", file_type="video", mime=LOFT_MIME)
    seed.transcript("f", 0, "warehouse", times=(10.0, 20.0))


def _page(seed, index) -> None:
    _file(index, "f", mime="application/pdf")
    seed.text_chunk("f", 0, "warehouse", page="4")


def _section(seed, index) -> None:
    _file(index, "f", mime=EPUB_MIME)
    seed.text_chunk("f", 0, "warehouse", page="4")
    seed.section("f", 4, "Four")


def _picture_text(seed, index) -> None:
    _file(index, "f", file_type="image", mime="image/jpeg")
    seed.text_chunk("f", 0, "warehouse")


def _plain_text(seed, index) -> None:
    _file(index, "f")
    seed.text_chunk("f", 0, "warehouse")


NO_FRAME = [
    pytest.param(_audio_utterance, id="audio-utterance"),
    pytest.param(_loft_utterance, id="loft-reference"),
    pytest.param(_page, id="page"),
    pytest.param(_section, id="section"),
    pytest.param(_picture_text, id="picture-text"),
    pytest.param(_plain_text, id="text"),
]


@pytest.mark.parametrize("seed_one", NO_FRAME)
def test_spec_addon_048_i7_only_utterances_and_scenes_of_a_video_carry_a_frame_url(
    api, search_index, ready, no_hydration, seed_one,  # noqa: F811
):
    seed_one(Seed(search_index), search_index)

    body = _ok(api("/search", {"passages": "true", "q": "warehouse"}))

    (passage,) = _passages(body, "f")
    assert passage["frame_url"] is None


@pytest.mark.parametrize(
    "toggle, expected",
    [
        pytest.param("false", ["utterance"], id="without-scene-search"),
        pytest.param("true", ["scene", "utterance"], id="with-scene-search"),
    ],
)
def test_spec_addon_048_i7_a_scene_passage_exists_only_with_include_scene_clip(
    api, search_index, ready, no_hydration, toggle, expected,  # noqa: F811
):
    seed = Seed(search_index)
    _file(search_index, "vid", file_type="video", mime="video/mp4")
    seed.transcript("vid", 0, "warehouse", times=(10.0, 20.0))
    _vector(
        search_index, "vid", etype="clip", table="vec_clip", score=0.90, axis=2,
        start=30.0, end=60.0,
    )

    body = _ok(api("/search", {
        "passages": "true",
        "q": "warehouse", "include_scene_clip": toggle,
    }))

    kinds = sorted(passage["kind"] for passage in _passages(body, "vid"))
    assert kinds == expected


# ---------------------------------------------------------------------------
# SPEC-ADDON-048 I25: MatchInfo.chunk_index stays a text-chunk index
# ---------------------------------------------------------------------------


def test_spec_addon_048_i25_matchinfo_chunk_index_never_holds_a_transcript_chunk_index(
    search_index, ready,  # noqa: F811
):
    seed = Seed(search_index)
    _file(search_index, "vid", file_type="video", mime="video/mp4")
    seed.transcript("vid", 7, "warehouse", times=(10.0, 20.0))

    response = search_mod.search("warehouse", drive=DRIVE, limit=100)

    (result,) = [r for r in response.results if r.file_id == "vid"]
    transcript_matches = [
        match
        for segment in result.segments
        for match in segment.matches
        if match.match_type == "transcript"
    ]
    assert transcript_matches
    assert all(match.chunk_index is None for match in transcript_matches)


def test_spec_addon_048_i25_a_text_content_match_keeps_its_text_chunk_index(
    search_index, ready,  # noqa: F811
):
    seed = Seed(search_index)
    _file(search_index, "note")
    seed.text_chunk("note", 4, "warehouse note")

    response = search_mod.search("warehouse", drive=DRIVE, limit=100)

    (result,) = [r for r in response.results if r.file_id == "note"]
    text_matches = [
        match
        for segment in result.segments
        for match in segment.matches
        if match.match_type == "text_content"
    ]
    assert [match.chunk_index for match in text_matches] == [4]
