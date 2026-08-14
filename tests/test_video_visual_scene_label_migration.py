"""Migration coverage for the concise Visual Index scene label."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

for _mod in (
    "PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers",
    "faster_whisper", "onnxruntime", "transformers", "janome",
    "janome.tokenizer", "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine, text  # noqa: E402


def _legacy_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE video_visual_scenes ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "visual_description TEXT"
            ")"
        ))
        conn.execute(text(
            "INSERT INTO video_visual_scenes (visual_description) "
            "VALUES ('Legacy verbose description')"
        ))
    return engine


def _columns(engine) -> set[str]:
    with engine.connect() as conn:
        return {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(video_visual_scenes)"))
        }


def test_migration_adds_nullable_scene_label_and_preserves_legacy_data(tmp_path):
    from app.database import _migrate_video_visual_scenes_if_needed

    engine = _legacy_engine(tmp_path)
    with engine.begin() as conn:
        _migrate_video_visual_scenes_if_needed(conn)

    assert "scene_label" in _columns(engine)
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT scene_label, visual_description FROM video_visual_scenes"
        )).one()
    assert row[0] is None
    assert row[1] == "Legacy verbose description"


def test_migration_is_idempotent(tmp_path):
    from app.database import _migrate_video_visual_scenes_if_needed

    engine = _legacy_engine(tmp_path)
    with engine.begin() as conn:
        _migrate_video_visual_scenes_if_needed(conn)
        _migrate_video_visual_scenes_if_needed(conn)

    assert "scene_label" in _columns(engine)


def test_migration_is_safe_before_table_registration(tmp_path):
    from app.database import _migrate_video_visual_scenes_if_needed

    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    with engine.begin() as conn:
        _migrate_video_visual_scenes_if_needed(conn)

    assert _columns(engine) == set()
