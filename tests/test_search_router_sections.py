"""Wire shape of an EPUB match in the search response."""

from __future__ import annotations

import pytest

from app.routers import search as search_router
from app.search import MatchInfo, SearchResponse, SearchResult, SegmentGroup


@pytest.fixture(autouse=True)
def no_hydration(monkeypatch):
    async def _hydrate(_ids):
        return {}

    monkeypatch.setattr(search_router, "hydrate_files", _hydrate)


def _response(mime: str, pages: list[int | None], titles=()) -> SearchResponse:
    return SearchResponse(
        results=(
            SearchResult(
                file_id="f1", drive="d1", filename="f1", file_type="document",
                score=1.0, match_types=("text_content",),
                segments=(
                    SegmentGroup(
                        time_range=None,
                        matches=tuple(
                            MatchInfo(match_type="text_content", text=f"t{p}",
                                      score=0.5, page=p)
                            for p in pages
                        ),
                    ),
                ),
                mime_type=mime,
                section_titles=titles,
            ),
        ),
        total=1, indexed_files=1, service_version="test",
    )


def _wire(model) -> list[tuple]:
    return [
        (m.text, m.page, m.section, m.section_title)
        for m in model.results[0].segments[0].matches
    ]


@pytest.mark.parametrize(
    ("mime", "pages", "expected"),
    [
        pytest.param(
            "application/epub+zip", [3, 4, None],
            [("t3", None, 3, "Chapter Three"), ("t4", None, 4, None),
             ("tNone", None, None, None)],
            id="epub-section-not-page",
        ),
        pytest.param(
            "application/pdf", [3], [("t3", 3, None, None)],
            id="pdf-page-not-section",
        ),
    ],
)
async def test_segment_match_wire_location(mime, pages, expected) -> None:
    model = await search_router._to_response_model(
        _response(mime, pages, titles=((3, "Chapter Three"),)),
    )

    assert _wire(model) == expected
