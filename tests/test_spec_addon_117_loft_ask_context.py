"""SPEC-ADDON-117 item 1.5, I4: the drive-wide Ask reads a YouTube ``.loft``'s
on-screen text under the rule a native video follows.

Search passages and find already read any video row's frame text; what the fetch
adds there is the rows themselves.
"""

from __future__ import annotations

import pytest

from app.config import RagConfig
from app.rag.context import build_file_context
from app.rag.retriever import RetrievedFile
from app.search import MatchInfo, SegmentGroup
from tests.search_endpoint_db import DRIVE, Seed, search_index  # noqa: F401 - fixture

LOFT_MIME = "application/vnd.litloft.loft+json"
ON_SCREEN = "salt and sugar on screen"


def _video_with_frame_text(index, file_id: str, mime: str, source: str | None = None) -> None:
    seed = Seed(index)
    seed.file(file_id, f"{file_id}.bin", mime=mime)
    index.sql(
        "UPDATE indexed_files SET file_type = 'video', frame_ocr_read = 1, "
        "frame_ocr_indexed = 1 WHERE file_id = :f",
        f=file_id,
    )
    if mime == LOFT_MIME:
        index.sql(
            "UPDATE indexed_files SET loft_source = :s WHERE file_id = :f",
            s=source, f=file_id,
        )
    seed.text_chunk(file_id, 0, ON_SCREEN)
    index.sql(
        "INSERT INTO video_frame_texts(file_id, chunk_index, start_time, end_time, text) "
        "VALUES (:f, 0, 10.0, 20.0, :t)",
        f=file_id, t=ON_SCREEN,
    )


def _candidate(file_id: str, mime: str, file_type: str) -> RetrievedFile:
    match = MatchInfo(
        match_type="text_content", text="salt", score=0.9, chunk_index=0, channel="keyword",
    )
    return RetrievedFile(
        file_id=file_id, drive=DRIVE, filename=f"{file_id}.bin", file_type=file_type,
        title="Cooking", description="", score=0.9, match_types=("text_content",),
        segments=(SegmentGroup(time_range=(10.0, 20.0), matches=(match,)),),
        mime_type=mime,
    )


@pytest.mark.parametrize("loft_file_type", ["video", "other"])
@pytest.mark.parametrize("loft_source", ["youtube", "web"])
def test_spec_addon_117_i4_an_analysable_lofts_frame_text_reaches_the_drive_wide_ask_as_a_native_videos(
    search_index, loft_file_type, loft_source,  # noqa: F811
):
    _video_with_frame_text(search_index, "native", "video/mp4")
    _video_with_frame_text(search_index, "ytloft", LOFT_MIME, loft_source)

    native = build_file_context(_candidate("native", "video/mp4", "video"), RagConfig())
    loft = build_file_context(_candidate("ytloft", LOFT_MIME, loft_file_type), RagConfig())

    native_snippets = [(s.source, s.text, s.location) for s in native.snippets]
    assert any(ON_SCREEN in text for _, text, _ in native_snippets)
    assert [(s.source, s.text, s.location) for s in loft.snippets] == native_snippets
