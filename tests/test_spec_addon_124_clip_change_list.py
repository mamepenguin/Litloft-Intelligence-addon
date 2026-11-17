"""SPEC-ADDON-124 I6: ``GET /files/{id}/clip-timestamps`` lists the file's stored
change frames beside its candidate frames, over a real search DB."""

from __future__ import annotations

import app.database as database
from app.models import Embedding, IndexedFile
from tests.search_endpoint_db import DRIVE, api  # noqa: F401 - fixture
from tests.vec_partition_db import search_index  # noqa: F401 - fixture

CANDIDATES = [0.0, 5.0, 10.0, 15.0, 20.0]
CHANGES = [3.0, 12.0, 17.5]


def _file(file_id: str) -> None:
    with database.get_search_db() as s:
        s.add(IndexedFile(
            file_id=file_id, drive=DRIVE, filename=f"{file_id}.mp4",
            file_path=f"/drives/{DRIVE}/{file_id}.mp4", file_type="video",
            mime_type="video/mp4", file_size=1, duration=30.0, active=True,
        ))


def _rows(file_id: str, rows: list[tuple[str, str, float, str]]) -> None:
    """``rows`` are ``(embedding_type, vector_table, second, preview)``, written in the
    given order so a test can store them out of time order."""
    with database.get_search_db() as s:
        s.add_all([
            Embedding(
                id=f"{etype}_{table}_{file_id}_{n}", file_id=file_id,
                embedding_type=etype, vector_table=table, timestamp_start=t,
                content_preview=preview,
            )
            for n, (etype, table, t, preview) in enumerate(rows)
        ])


def _candidates(file_id: str) -> None:
    _rows(file_id, [("clip", "vec_clip", t, f"Frame at {t:g}s") for t in CANDIDATES])


def _changes(file_id: str, seconds: list[float]) -> None:
    _rows(file_id, [
        ("clip_change", "vec_clip_change", t, f"Frame at {t:g}s")
        for t in reversed(seconds)
    ])


def _get(api, file_id: str) -> dict:
    response = api(f"/files/{file_id}/clip-timestamps", drive=DRIVE)
    assert response.status_code == 200, response.text
    return response.json()


def test_spec_addon_124_changes_are_the_files_change_rows_ascending_and_timestamps_are_unchanged(
    api, search_index,
):
    _file("plain")
    _candidates("plain")
    _file("scanned")
    _candidates("scanned")
    _changes("scanned", CHANGES)
    _rows("scanned", [("whisper", "vec_text", 7.0, "spoken")])
    _file("neighbour")
    _candidates("neighbour")
    _changes("neighbour", [8.0, 9.0])

    plain = _get(api, "plain")
    scanned = _get(api, "scanned")

    assert scanned.get("changes") == [
        {"start": t, "content_preview": f"Frame at {t:g}s"} for t in CHANGES
    ]
    assert scanned["timestamps"] == plain["timestamps"]
    assert [row["start"] for row in scanned["timestamps"]] == CANDIDATES


def test_spec_addon_124_changes_is_an_empty_list_for_a_file_with_no_change_rows(
    api, search_index,
):
    _file("unscanned")
    _candidates("unscanned")

    body = _get(api, "unscanned")

    assert "changes" in body
    assert body["changes"] == []
    assert [row["start"] for row in body["timestamps"]] == CANDIDATES


def test_spec_addon_124_changes_needs_both_the_type_and_the_vector_table(
    api, search_index,
):
    _file("decoys")
    _candidates("decoys")
    _changes("decoys", [12.0])
    _rows("decoys", [
        ("clip_change", "vec_clip", 4.0, "Frame at 4s"),
        ("clip", "vec_clip_change", 6.0, "Frame at 6s"),
    ])

    body = _get(api, "decoys")

    assert body.get("changes") == [{"start": 12.0, "content_preview": "Frame at 12s"}]


def test_spec_addon_124_a_file_with_change_frames_and_no_candidates_still_lists_its_changes(
    api, search_index,
):
    _file("changes_only")
    _changes("changes_only", CHANGES)

    body = _get(api, "changes_only")

    assert body["timestamps"] == []
    assert [row["start"] for row in body.get("changes", [])] == CHANGES
