"""SPEC-ADDON-193 I7: the one-time loft source reclassification on start.

Every ``.loft`` row of source ``other`` is reopened once, in a transaction of its
own, marked in ``index_meta``; a failure rolls the rows and the marker back,
logs one WARNING and leaves the next start to try again.
"""

from __future__ import annotations

import logging

from sqlalchemy import create_engine, text
from tests.test_frame_page_ocr_schema import init_database  # noqa: F401 - fixture

LOFT_MIME = "application/vnd.litloft.loft+json"
MARKER = "loft_source_web_reclassified"
FLAGS = ("frame_scan_indexed", "frame_ocr_read", "frame_ocr_indexed",
         "loft_candidates_done")
COLUMNS = (
    "file_id", "drive", "filename", "file_path", "file_type", "mime_type",
    "file_size", "active", "loft_source", "loft_fetch_failures", *FLAGS,
)


def _insert(engine, file_id: str, *, mime: str = LOFT_MIME, source=None,
            failures: int = 0, open_flags: bool = False) -> None:
    flags = 1 if open_flags else 0
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO indexed_files(file_id, drive, filename, file_path, "
                "file_type, mime_type, file_size, active, title, description, "
                "tags_text, indexed_at, updated_at, loft_source, "
                "loft_fetch_failures, frame_scan_indexed, frame_ocr_read, "
                "frame_ocr_indexed, loft_candidates_done) "
                "VALUES (:f, 'family', :f, :path, 'video', :mime, 1, 1, '', '', '', "
                "'2026-11-28 10:00:00', '2026-11-28 10:00:00', :source, "
                ":failures, :flags, :flags, :flags, :flags)"
            ),
            {
                "f": file_id, "path": f"/drives/family/{file_id}.loft",
                "mime": mime, "source": source, "failures": failures,
                "flags": flags,
            },
        )


def _engine(path):
    return create_engine(f"sqlite:///{path}")


def _rows(engine) -> dict[str, tuple]:
    with engine.connect() as conn:
        return {
            row[0]: tuple(row)
            for row in conn.execute(text(
                f"SELECT {', '.join(COLUMNS)} FROM indexed_files ORDER BY file_id"
            ))
        }


def _marker(engine) -> str | None:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT value FROM index_meta WHERE key = :k"), {"k": MARKER},
        ).fetchone()
    return None if row is None else row[0]


def _clear_marker(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM index_meta WHERE key = :k"), {"k": MARKER})


def _seed(engine) -> None:
    _insert(engine, "other", source="other", failures=2, open_flags=True)
    _insert(engine, "youtube", source="youtube", failures=1)
    _insert(engine, "web", source="web", failures=1)
    _insert(engine, "unclassified", source=None)
    _insert(engine, "native", mime="video/mp4", source=None)


def _find(row: tuple, column: str):
    return row[COLUMNS.index(column)]


def _other_is_reopened(row: tuple) -> bool:
    return (
        _find(row, "loft_source") is None
        and _find(row, "loft_fetch_failures") == 0
        and all(not _find(row, flag) for flag in FLAGS)
    )


def test_spec_addon_193_i7_the_start_reopens_every_other_loft_and_marks_it(
    init_database,
):
    start, path = init_database
    start()
    engine = _engine(path)
    _seed(engine)
    _clear_marker(engine)
    before = _rows(engine)

    start()

    after = _rows(engine)
    assert _other_is_reopened(after["other"]), after["other"]
    for file_id in ("youtube", "web", "unclassified", "native"):
        assert after[file_id] == before[file_id], file_id
    assert _marker(engine) is not None
    engine.dispose()


def test_spec_addon_193_i7_a_present_marker_changes_nothing_on_the_next_start(
    init_database,
):
    start, path = init_database
    start()
    engine = _engine(path)
    _seed(engine)
    _clear_marker(engine)
    start()
    _insert(engine, "other2", source="other", failures=2, open_flags=True)
    before = _rows(engine)

    start()

    assert _rows(engine) == before
    assert _marker(engine) is not None
    engine.dispose()


def test_spec_addon_193_i7_a_failed_reclassification_rolls_back_logs_and_retries(
    init_database, caplog,
):
    start, path = init_database
    start()
    engine = _engine(path)
    _seed(engine)
    _clear_marker(engine)
    before = _rows(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TRIGGER refuse_marker BEFORE INSERT ON index_meta "
            f"WHEN NEW.key = '{MARKER}' "
            "BEGIN SELECT RAISE(ABORT, 'refused'); END"
        ))

    with caplog.at_level(logging.WARNING):
        start()

    failed = _rows(engine)
    assert failed == before
    assert _marker(engine) is None
    lines = [
        record.getMessage() for record in caplog.records
        if "loft source reclassification failed" in record.getMessage()
    ]
    assert len(lines) == 1
    assert lines[0].startswith("loft source reclassification failed (")
    assert lines[0].endswith(")")
    with engine.connect() as conn:
        tables = {row[0] for row in conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ))}
    assert "vec_text" in tables

    with engine.begin() as conn:
        conn.execute(text("DROP TRIGGER refuse_marker"))
    start()

    assert _other_is_reopened(_rows(engine)["other"])
    assert _marker(engine) is not None
    engine.dispose()


def test_spec_addon_193_i7_only_loft_rows_of_source_other_are_reopened(
    init_database,
):
    start, path = init_database
    start()
    engine = _engine(path)
    _insert(engine, "other", source="other", failures=1, open_flags=True)
    _insert(engine, "other_native", mime="video/mp4", source="other",
            failures=1, open_flags=True)
    _clear_marker(engine)

    start()

    rows = _rows(engine)
    assert _other_is_reopened(rows["other"])
    assert not _other_is_reopened(rows["other_native"])
    engine.dispose()
