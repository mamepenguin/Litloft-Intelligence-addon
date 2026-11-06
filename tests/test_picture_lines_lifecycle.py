"""Corrections, restores, rebuilds, a model purge and the table (SPEC-ADDON-075 I3).

SPEC-ID: SPEC-ADDON-075

The operations here must leave the picture's lines exactly as the OCR worker
wrote them, because only the text-storing branches of that write may change
them.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from sqlalchemy import create_engine, text

from app.config import ModelConfig
from tests.extraction_fixtures import (
    HEADERS,
    SearchDB,
    api,
    build_search_db,
    search_db,
    seed_picture,
)

PICTURE = "lines0000001"


def _seed_lines(db: SearchDB, file_id: str, lines: list[dict]) -> None:
    db.execute(
        "INSERT INTO picture_lines(file_id, lines_json, updated_at) "
        "VALUES(:f, :j, CURRENT_TIMESTAMP)",
        f=file_id,
        j=json.dumps(lines),
    )


def _raw_row(db: SearchDB, file_id: str) -> tuple | None:
    rows = db.rows(
        "SELECT lines_json, updated_at FROM picture_lines WHERE file_id = :f",
        f=file_id,
    )
    return rows[0] if rows else None


def _machine_lines() -> list[dict]:
    return [
        {"text": "alpha beta", "quad": [0, 0, 0.5, 0, 0.5, 0.5, 0, 0.5], "score": 0.9},
        {"text": "gamma", "quad": [0.5, 0.5, 1, 0.5, 1, 1, 0.5, 1], "score": 0.8},
    ]


def _seed_picture_with_lines(db: SearchDB) -> tuple:
    seed_picture(db, PICTURE, ["alpha beta\ngamma"])
    _seed_lines(db, PICTURE, _machine_lines())
    return _raw_row(db, PICTURE)


def _stub_followup(monkeypatch, db: SearchDB) -> None:
    import numpy as np

    from app.workers import followup

    monkeypatch.setattr(
        "app.workers.embedder.embed_passages",
        lambda texts: [np.zeros(4, dtype=np.float32) for _ in texts],
    )
    monkeypatch.setattr(followup, "post_addon_event_sync", lambda *a, **k: True)


def _drain() -> None:
    from app.workers import followup

    while (item_id := followup.next_due_item_id()) is not None:
        followup.process_item(item_id)


@pytest.mark.parametrize("operation", ["correction", "restore", "rebuild"])
def test_spec_addon_075_i3_a_correction_restore_or_rebuild_leaves_the_lines_row(
    search_db, api, monkeypatch, operation
):
    """SPEC-ADDON-075 I3: these writes never touch the picture's lines."""
    client, _ = api
    before = _seed_picture_with_lines(search_db)
    assert before is not None

    assert client.get(f"/files/{PICTURE}/picture-text", headers=HEADERS).status_code == 200
    if operation in ("correction", "restore"):
        response = client.put(
            f"/files/{PICTURE}/picture-text",
            json={"text": "fixed text", "expected_text": "alpha beta\ngamma"},
            headers=HEADERS,
        )
        assert response.status_code == 200, response.text

    if operation == "restore":
        from app import addon_events

        async def _no_event(*_args, **_kwargs):
            return True

        monkeypatch.setattr(addon_events, "post_addon_event", _no_event)
        response = client.post(
            f"/files/{PICTURE}/picture-text/restore", headers=HEADERS
        )
        assert response.status_code == 200, response.text

    if operation == "rebuild":
        from app import extraction

        with search_db.Session() as session:
            extraction.queue_item(
                session, PICTURE, extraction.KIND_PICTURE, "rebuild"
            )
            session.commit()
        _stub_followup(monkeypatch, search_db)
        _drain()

    assert _raw_row(search_db, PICTURE) == before


def test_spec_addon_075_i3_the_kept_original_of_a_correction_holds_no_boxes(
    search_db, api
):
    """SPEC-ADDON-075 I3: the original is the text only."""
    client, _ = api
    _seed_picture_with_lines(search_db)
    assert client.get(f"/files/{PICTURE}/picture-text", headers=HEADERS).status_code == 200

    response = client.put(
        f"/files/{PICTURE}/picture-text",
        json={"text": "fixed text", "expected_text": "alpha beta\ngamma"},
        headers=HEADERS,
    )

    assert response.status_code == 200, response.text
    (original,) = search_db.rows(
        "SELECT text, chunks_json, words_json FROM kept_originals "
        "WHERE file_id = :f AND kind = 'picture_text'",
        f=PICTURE,
    )
    assert original == ("alpha beta\ngamma", None, None)


def test_spec_addon_075_i3_a_text_embedding_model_purge_leaves_the_lines_row(
    tmp_path, monkeypatch
):
    """SPEC-ADDON-075 I3: the model purge removes vectors and text, not lines."""
    import app.config as config

    db = build_search_db(tmp_path, monkeypatch)
    seed_picture(db, PICTURE, ["alpha beta\ngamma"])
    _seed_lines(db, PICTURE, _machine_lines())
    db.execute(
        "INSERT INTO picture_texts(file_id, text, updated_at) "
        "VALUES(:f, :t, CURRENT_TIMESTAMP)",
        f=PICTURE,
        t="alpha beta\ngamma",
    )
    before = _raw_row(db, PICTURE)
    db.execute("CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS detailed_summary_citations ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT, file_id TEXT NOT NULL, "
        "  section_path TEXT NOT NULL)"
    )
    db.execute(
        "INSERT INTO index_meta(key, value) VALUES('text_embedding_model', 'old-model')"
    )
    settings = dataclasses.replace(
        config.settings, models=ModelConfig(text_embedding="a-different-model")
    )
    monkeypatch.setattr(config, "settings", settings)

    from app.database import _migrate_vec_text_if_needed

    with db.engine.connect() as conn:
        _migrate_vec_text_if_needed(conn)
        conn.commit()

    assert _raw_row(db, PICTURE) == before
    assert db.rows(
        "SELECT 1 FROM picture_texts WHERE file_id = :f", f=PICTURE
    )


def test_spec_addon_075_the_picture_lines_table_is_added_by_create_all(tmp_path):
    """SPEC-ADDON-075 Data: an existing DB gains the table without a migration."""
    from app.database import Base
    from app.models import PictureLines

    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE indexed_files (file_id TEXT PRIMARY KEY)"
        ))

    Base.metadata.create_all(engine)

    with engine.connect() as conn:
        columns = [
            row[1]
            for row in conn.execute(text("PRAGMA table_info(picture_lines)")).fetchall()
        ]
    assert PictureLines.__tablename__ == "picture_lines"
    assert set(columns) == {"file_id", "lines_json", "updated_at"}
    engine.dispose()
