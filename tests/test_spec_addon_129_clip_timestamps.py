"""SPEC-ADDON-129 I3: ``GET /files/{id}/clip-timestamps`` answers the file's candidate
frames alone, over a real search DB."""

from __future__ import annotations

import pytest

import app.database as database
from app.models import Embedding, IndexedFile
from tests.search_endpoint_db import DRIVE, OTHER, api  # noqa: F401 - fixture
from tests.vec_partition_db import search_index  # noqa: F401 - fixture

CANDIDATES = [0.0, 5.0, 10.0, 15.0, 20.0]
STORED_ORDER = [15.0, 0.0, 20.0, 5.0, 10.0]
CHANGES = [3.0, 12.0, 17.5]


def _file(file_id: str, drive: str = DRIVE) -> None:
    with database.get_search_db() as s:
        s.add(IndexedFile(
            file_id=file_id, drive=drive, filename=f"{file_id}.mp4",
            file_path=f"/drives/{drive}/{file_id}.mp4", file_type="video",
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


def _candidates(file_id: str, seconds: list[float] = STORED_ORDER) -> None:
    _rows(file_id, [("clip", "vec_clip", t, f"Frame at {t:g}s") for t in seconds])


def _changes(file_id: str, seconds: list[float] = CHANGES) -> None:
    _rows(file_id, [
        ("clip_change", "vec_clip_change", t, f"Change at {t:g}s")
        for t in reversed(seconds)
    ])


def _get(api, file_id: str) -> dict:
    response = api(f"/files/{file_id}/clip-timestamps", drive=DRIVE)
    assert response.status_code == 200, response.text
    return response.json()


EXPECTED = [{"start": t, "content_preview": f"Frame at {t:g}s"} for t in CANDIDATES]


@pytest.mark.parametrize("with_changes", [False, True], ids=["no_change_rows", "change_rows"])
def test_spec_addon_129_i3_answer_is_exactly_file_id_drive_and_the_clip_rows_ascending(
    api, search_index, with_changes,
):
    _file("video")
    _candidates("video")
    if with_changes:
        _changes("video")
    _rows("video", [("whisper", "vec_text", 7.0, "spoken")])
    _file("neighbour")
    _candidates("neighbour", [1.0, 2.0])
    _changes("neighbour", [8.0, 9.0])

    body = _get(api, "video")

    assert set(body) == {"file_id", "drive", "timestamps"}
    assert body["file_id"] == "video"
    assert body["drive"] == DRIVE
    assert body["timestamps"] == EXPECTED


def test_spec_addon_129_i3_change_rows_do_not_change_the_answer(
    api, search_index,
):
    _file("plain")
    _candidates("plain")
    _file("scanned")
    _candidates("scanned")
    _changes("scanned")

    plain = _get(api, "plain")
    scanned = _get(api, "scanned")

    assert {k: v for k, v in scanned.items() if k != "file_id"} == {
        k: v for k, v in plain.items() if k != "file_id"
    }
    assert set(scanned) == {"file_id", "drive", "timestamps"}


def test_spec_addon_129_i3_a_file_with_only_change_rows_answers_no_timestamps(
    api, search_index,
):
    _file("changes_only")
    _changes("changes_only")

    body = _get(api, "changes_only")

    assert body == {"file_id": "changes_only", "drive": DRIVE, "timestamps": []}


def test_spec_addon_129_i3_without_x_lit_drive_is_400(api, search_index):
    _file("video")
    _candidates("video")

    response = api("/files/video/clip-timestamps", drive=None)

    assert response.status_code == 400


def test_spec_addon_129_i3_a_file_not_indexed_on_the_requested_drive_is_404(
    api, search_index,
):
    _file("elsewhere", drive=OTHER)
    _candidates("elsewhere")

    response = api("/files/elsewhere/clip-timestamps", drive=DRIVE)

    assert response.status_code == 404
