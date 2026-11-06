"""The picture OCR worker keeps where each line sits (SPEC-ADDON-075 I1-I4).

SPEC-ID: SPEC-ADDON-075

The engine is the fake from ``tests.ocr_fixtures``; these tests hold what the
worker does with the boxes and scores it answers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.ocr_fixtures import (
    ABSENT,
    FakeEngine,
    SearchDB,
    build_search_db,
    huge_pixel_png,
    install,
    jpeg_with_exif_orientation_6,
    oversize_png,
    pillow_real,  # noqa: F401 - the usefixtures marker resolves it
    picture,
    seed_indexed_file,
    seed_ocr_rows,
)

pytestmark = pytest.mark.usefixtures("pillow_real")

ALL_DONE = dict(
    metadata_indexed=True,
    clip_indexed=True,
    clip_thumbnail_indexed=True,
    whisper_indexed=True,
    text_indexed=True,
    tfidf_keywords_indexed=True,
)


@pytest.fixture()
def search_db(tmp_path, monkeypatch) -> SearchDB:
    return build_search_db(tmp_path, monkeypatch)


def run_worker(harness, file_id: str) -> bool:
    return harness.module.index_ocr(file_id)


def _stored_lines(db: SearchDB, file_id: str) -> list[dict] | None:
    rows = db.rows(
        "SELECT lines_json FROM picture_lines WHERE file_id = :f", f=file_id
    )
    return json.loads(rows[0][0]) if rows else None


def _seed_lines(db: SearchDB, file_id: str, lines: list[dict]) -> None:
    db.execute(
        "INSERT INTO picture_lines(file_id, lines_json, updated_at) "
        "VALUES(:f, :j, CURRENT_TIMESTAMP)",
        f=file_id,
        j=json.dumps(lines),
    )


def _raw_lines_row(db: SearchDB, file_id: str) -> tuple | None:
    rows = db.rows(
        "SELECT lines_json, updated_at FROM picture_lines WHERE file_id = :f",
        f=file_id,
    )
    return rows[0] if rows else None


def _picture_text(db: SearchDB, file_id: str) -> str | None:
    rows = db.rows(
        "SELECT text FROM picture_texts WHERE file_id = :f", f=file_id
    )
    return rows[0][0] if rows else None


def _normalised(box: list[list[float]], width: int, height: int) -> list[float]:
    flat = [number for point in box for number in point]
    return [
        min(1.0, max(0.0, value / (width if index % 2 == 0 else height)))
        for index, value in enumerate(flat)
    ]


def _text_rows(db: SearchDB, file_id: str) -> tuple[list[tuple], list[tuple]]:
    return (
        db.ocr_fts(file_id),
        db.ocr_embeddings(file_id),
    )


# ---------------------------------------------------------------------------
# I1 - the row written by a successful run
# ---------------------------------------------------------------------------


def test_spec_addon_075_i1_one_row_holds_each_line_text_quad_and_score(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-075 I1: one row, engine order, normalised and clamped quads."""
    path = picture(tmp_path / "sign.png", size=(40, 20))
    seed_indexed_file(search_db, "pic1", "image/png", path, **ALL_DONE)
    lines = ["first line", "second line", "third line"]
    boxes = [
        [[2, 4], [18, 4], [18, 10], [2, 10]],
        [[-5, 1], [50, 2], [39, 19], [0, 25]],
        [[1, 2], [2, 3], [3, 4], [4, 5]],
    ]
    scores = [0.9, 0.5, 0.25]
    harness = install(monkeypatch, lines=lines, boxes=boxes, scores=scores)

    assert run_worker(harness, "pic1") is True

    stored = _stored_lines(search_db, "pic1")
    assert [line["text"] for line in stored] == lines
    assert [line["quad"] for line in stored] == [
        _normalised(box, 40, 20) for box in boxes
    ]
    assert [line["score"] for line in stored] == scores
    joined = "\n".join(lines)
    assert _picture_text(search_db, "pic1") == joined
    assert search_db.ocr_fts("pic1") == [(0, "", joined)]
    assert [row[2] for row in search_db.ocr_embeddings("pic1")] == [joined]


def test_spec_addon_075_i1_an_earlier_row_is_replaced_and_another_picture_is_untouched(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-075 I1: no earlier line survives; another file's row does."""
    mine = picture(tmp_path / "mine.png", size=(40, 20))
    other = picture(tmp_path / "other.png", size=(40, 20))
    seed_indexed_file(search_db, "pic1", "image/png", mine, **ALL_DONE)
    seed_indexed_file(search_db, "pic2", "image/png", other, **ALL_DONE)
    earlier = [{"text": "old", "quad": [0, 0, 1, 0, 1, 1, 0, 1], "score": 0.1}]
    _seed_lines(search_db, "pic1", earlier)
    _seed_lines(search_db, "pic2", [{"text": "theirs", "quad": [0, 0, 0, 0, 0, 0, 0, 0], "score": None}])
    before_other = _raw_lines_row(search_db, "pic2")
    boxes = [[[0, 0], [40, 0], [40, 20], [0, 20]]]
    harness = install(monkeypatch, lines=["fresh"], boxes=boxes)

    run_worker(harness, "pic1")

    assert [line["text"] for line in _stored_lines(search_db, "pic1")] == ["fresh"]
    assert _raw_lines_row(search_db, "pic2") == before_other


def test_spec_addon_075_i1_an_exif_rotated_picture_is_divided_by_the_oriented_size(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-075 I1: the quad's fractions are of the oriented picture."""
    path = jpeg_with_exif_orientation_6(tmp_path / "rotated.jpg")
    seed_indexed_file(search_db, "pic1", "image/jpeg", path, **ALL_DONE)
    box = [[5, 10], [15, 30], [10, 35], [0, 5]]
    harness = install(monkeypatch, lines=["sideways"], boxes=[box])

    run_worker(harness, "pic1")

    assert harness.engine.seen[0].size == (20, 40)
    (line,) = _stored_lines(search_db, "pic1")
    assert line["quad"] == pytest.approx(_normalised(box, 20, 40), abs=1e-9)
    assert line["score"] is None


def test_spec_addon_075_i1_a_failed_lines_insert_rolls_back_the_whole_write(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-075 I1: the lines share the text write's transaction."""
    path = picture(tmp_path / "pic.png", size=(40, 20))
    seed_indexed_file(search_db, "pic1", "image/png", path, **ALL_DONE)
    seed_ocr_rows(search_db, "pic1", ("kept",))
    before = _text_rows(search_db, "pic1")
    search_db.execute(
        "CREATE TRIGGER fail_picture_lines_insert BEFORE INSERT ON picture_lines "
        "BEGIN SELECT RAISE(ABORT, 'lines insert failed'); END"
    )
    boxes = [[[0, 0], [40, 0], [40, 20], [0, 20]]]
    harness = install(monkeypatch, lines=["fresh"], boxes=boxes)

    with pytest.raises(Exception):
        run_worker(harness, "pic1")

    assert _text_rows(search_db, "pic1") == before
    assert _picture_text(search_db, "pic1") is None
    assert bool(search_db.flag("pic1", "ocr_indexed")) is False
    assert _stored_lines(search_db, "pic1") is None


# ---------------------------------------------------------------------------
# I2 - reached but the lines cannot be kept
# ---------------------------------------------------------------------------


def _garbage(path: Path) -> Path:
    path.write_bytes(b"this is not any picture format" * 8)
    return path


def _no_text_boxes(lines: list[str]) -> list[list[list[int]]]:
    return [
        [[index, 0], [index + 1, 0], [index + 1, 1], [index, 1]]
        for index in range(len(lines))
    ]


@pytest.mark.parametrize(
    "case,builder,lines,boxes",
    [
        ("no-text", picture, [], ABSENT),
        ("undecodable", _garbage, ["never reached"], ABSENT),
        (
            "over-pixel-cap",
            huge_pixel_png,
            ["never reached"],
            ABSENT,
        ),
        (
            "over-byte-cap",
            oversize_png,
            ["never reached"],
            ABSENT,
        ),
        ("no-boxes", picture, ["text"], ABSENT),
        ("too-few-boxes", picture, ["one", "two"], [[[0, 0], [1, 0], [1, 1], [0, 1]]]),
        (
            "three-point-box",
            picture,
            ["text"],
            [[[0, 0], [1, 0], [0, 1]]],
        ),
        (
            "non-finite-box",
            picture,
            ["text"],
            [[[0, 0], [float("inf"), 0], [1, 1], [0, 1]]],
        ),
    ],
    ids=[
        "no-text",
        "undecodable",
        "over-pixel-cap",
        "over-byte-cap",
        "no-boxes",
        "too-few-boxes",
        "three-point-box",
        "non-finite-box",
    ],
)
def test_spec_addon_075_i2_a_result_without_usable_boxes_keeps_no_lines(
    search_db, tmp_path, monkeypatch, case, builder, lines, boxes
):
    """SPEC-ADDON-075 I2: no lines, the earlier row removed, the text as today."""
    invalid_path = builder(tmp_path / f"{case}-invalid.png")
    control_path = builder(tmp_path / f"{case}-control.png")
    seed_indexed_file(search_db, "pic1", "image/png", invalid_path, **ALL_DONE)
    seed_indexed_file(search_db, "pic2", "image/png", control_path, **ALL_DONE)
    for file_id in ("pic1", "pic2"):
        _seed_lines(
            search_db,
            file_id,
            [{"text": "earlier", "quad": [0, 0, 1, 0, 1, 1, 0, 1], "score": 0.7}],
        )

    invalid = install(monkeypatch, lines=lines, boxes=boxes)
    run_worker(invalid, "pic1")
    control = install(monkeypatch, lines=lines)
    run_worker(control, "pic2")

    assert _stored_lines(search_db, "pic1") is None
    assert _stored_lines(search_db, "pic2") is None
    assert _picture_text(search_db, "pic1") == _picture_text(search_db, "pic2")
    assert search_db.ocr_fts("pic1") == search_db.ocr_fts("pic2")
    assert [row[2] for row in search_db.ocr_embeddings("pic1")] == [
        row[2] for row in search_db.ocr_embeddings("pic2")
    ]
    assert bool(search_db.flag("pic1", "ocr_indexed")) is True


@pytest.mark.parametrize(
    "scores,label",
    [
        (ABSENT, "absent"),
        ([0.5], "short"),
        ([0.5, float("nan"), 0.5], "nan"),
        ([0.5, float("inf"), 0.5], "inf"),
    ],
    ids=["absent", "short", "nan", "inf"],
)
def test_spec_addon_075_i2_an_unusable_scores_list_null_every_score(
    search_db, tmp_path, monkeypatch, scores, label
):
    """SPEC-ADDON-075 I2: the lines are kept; every score is null."""
    lines = ["one", "two", "three"]
    path = picture(tmp_path / f"scores-{label}.png", size=(40, 20))
    seed_indexed_file(search_db, "pic1", "image/png", path, **ALL_DONE)
    harness = install(
        monkeypatch, lines=lines, boxes=_no_text_boxes(lines), scores=scores
    )

    run_worker(harness, "pic1")

    stored = _stored_lines(search_db, "pic1")
    assert [line["text"] for line in stored] == lines
    assert [line["score"] for line in stored] == [None, None, None]
    assert _picture_text(search_db, "pic1") == "\n".join(lines)


# ---------------------------------------------------------------------------
# I3, I4 - a discarded or failed write changes no line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        "unreachable",
        "failed-embedding",
        "missing-engine",
        "corrected-without-redo",
        "confirmed-redo-with-no-text",
        "left-the-index",
    ],
)
def test_spec_addon_075_i3_i4_a_discarded_write_leaves_the_lines_byte_for_byte(
    search_db, tmp_path, monkeypatch, case
):
    """SPEC-ADDON-075 I3/I4: only the text-storing branches may touch the row."""
    path = (
        tmp_path / "gone.png"
        if case == "unreachable"
        else picture(tmp_path / "pic.png", size=(40, 20))
    )
    flags = dict(ALL_DONE)
    if case in ("corrected-without-redo", "confirmed-redo-with-no-text"):
        flags["picture_text_corrected"] = True
        if case == "confirmed-redo-with-no-text":
            flags["picture_text_redo"] = True
    seed_indexed_file(search_db, "pic1", "image/png", path, **flags)
    seed_ocr_rows(search_db, "pic1", ("kept",))
    _seed_lines(
        search_db,
        "pic1",
        [{"text": "kept line", "quad": [0, 0, 1, 0, 1, 1, 0, 1], "score": 0.4}],
    )
    before = _raw_lines_row(search_db, "pic1")
    before_text = _text_rows(search_db, "pic1")

    if case == "unreachable":
        harness = install(monkeypatch, lines=["new"], boxes=_no_text_boxes(["new"]))
    elif case == "failed-embedding":
        harness = install(
            monkeypatch,
            lines=["new"],
            boxes=_no_text_boxes(["new"]),
            embed_error=RuntimeError("embed down"),
        )
    elif case == "missing-engine":
        harness = install(monkeypatch, engine_error=RuntimeError("models missing"))
    elif case == "confirmed-redo-with-no-text":
        harness = install(monkeypatch, lines=[], boxes=ABSENT)
    elif case == "left-the-index":
        engine = FakeEngine(
            ("late text",), boxes=_no_text_boxes(["late text"])
        )
        engine.on_call = lambda image: search_db.execute(
            "UPDATE indexed_files SET active = 0 WHERE file_id = 'pic1'"
        )
        harness = install(monkeypatch, engine=engine)
    else:
        harness = install(
            monkeypatch, lines=["new"], boxes=_no_text_boxes(["new"])
        )

    assert run_worker(harness, "pic1") is False

    assert _raw_lines_row(search_db, "pic1") == before
    assert _text_rows(search_db, "pic1") == before_text


def test_spec_addon_075_i4_a_confirmed_redo_that_stores_text_replaces_the_lines(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-075 I4: a storing confirmed redo is a lines -> lines move."""
    path = picture(tmp_path / "pic.png", size=(40, 20))
    seed_indexed_file(
        search_db,
        "pic1",
        "image/png",
        path,
        picture_text_corrected=True,
        picture_text_redo=True,
        **ALL_DONE,
    )
    seed_ocr_rows(search_db, "pic1", ("corrected",))
    _seed_lines(
        search_db,
        "pic1",
        [{"text": "old", "quad": [0, 0, 1, 0, 1, 1, 0, 1], "score": 0.4}],
    )
    boxes = [[[0, 0], [40, 0], [40, 20], [0, 20]]]
    harness = install(monkeypatch, lines=["fresh"], boxes=boxes, scores=[0.8])

    assert run_worker(harness, "pic1") is True

    (line,) = _stored_lines(search_db, "pic1")
    assert line == {
        "text": "fresh",
        "quad": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        "score": 0.8,
    }
    assert _picture_text(search_db, "pic1") == "fresh"
