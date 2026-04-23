"""One-shot LOFT MIME backfill test.

Commit ffdc0f7 renamed the LoftRef MIME from
``application/vnd.homevault.link+json`` to
``application/vnd.litloft.loft+json`` in every code-side constant but
did not migrate the intelligence DB cache. Rows indexed before that
commit kept the old MIME, so ``_classify_file_type`` returned None
for them and every downstream consumer (regenerate precheck,
classify_detailed_missing_reason, RAG retriever) treated them as
unsupported.

These tests confirm the startup migration:
- rewrites ``application/vnd.homevault.link+json`` to the current MIME
- leaves every other MIME untouched
- is idempotent across restarts
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.database import Base  # noqa: E402
from app.models import IndexedFile  # noqa: E402,F401


LEGACY_MIME = "application/vnd.homevault.link+json"
CURRENT_MIME = "application/vnd.litloft.loft+json"


@pytest.fixture()
def engine_with_mixed_mimes(tmp_path):
    db_path = tmp_path / "search.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine, expire_on_commit=False)
    s = Session()
    try:
        s.add_all([
            IndexedFile(
                file_id="legacy001",
                drive="d", filename="a.loft",
                file_path="/d/a.loft",
                file_type="other",
                mime_type=LEGACY_MIME,
                file_size=100,
                active=True,
            ),
            IndexedFile(
                file_id="legacy002",
                drive="d", filename="b.loft",
                file_path="/d/b.loft",
                file_type="other",
                mime_type=LEGACY_MIME,
                file_size=100,
                active=True,
            ),
            IndexedFile(
                file_id="current001",
                drive="d", filename="c.loft",
                file_path="/d/c.loft",
                file_type="other",
                mime_type=CURRENT_MIME,
                file_size=100,
                active=True,
            ),
            IndexedFile(
                file_id="unrelated001",
                drive="d", filename="movie.mp4",
                file_path="/d/movie.mp4",
                file_type="video",
                mime_type="video/mp4",
                file_size=100,
                active=True,
            ),
        ])
        s.commit()
    finally:
        s.close()
    return engine


def _mime_counts(engine) -> dict[str, int]:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT mime_type, COUNT(*) FROM indexed_files GROUP BY mime_type"
        )).fetchall()
    return {r[0]: r[1] for r in rows}


def test_migration_rewrites_legacy_mime(engine_with_mixed_mimes):
    """Legacy HvLink MIME rows are rewritten to the current LoftRef MIME."""
    with patch("app.database._search_engine", engine_with_mixed_mimes):
        from app.database import _migrate_loft_mime_legacy_to_current
        _migrate_loft_mime_legacy_to_current()

    counts = _mime_counts(engine_with_mixed_mimes)
    assert counts.get(LEGACY_MIME, 0) == 0
    assert counts[CURRENT_MIME] == 3


def test_migration_leaves_unrelated_mimes_alone(engine_with_mixed_mimes):
    with patch("app.database._search_engine", engine_with_mixed_mimes):
        from app.database import _migrate_loft_mime_legacy_to_current
        _migrate_loft_mime_legacy_to_current()

    counts = _mime_counts(engine_with_mixed_mimes)
    assert counts["video/mp4"] == 1


def test_migration_is_idempotent(engine_with_mixed_mimes):
    with patch("app.database._search_engine", engine_with_mixed_mimes):
        from app.database import _migrate_loft_mime_legacy_to_current
        _migrate_loft_mime_legacy_to_current()
        _migrate_loft_mime_legacy_to_current()

    counts = _mime_counts(engine_with_mixed_mimes)
    assert counts.get(LEGACY_MIME, 0) == 0
    assert counts[CURRENT_MIME] == 3


def test_migration_noop_when_no_legacy_rows(tmp_path):
    db_path = tmp_path / "search.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine, expire_on_commit=False)
    s = Session()
    try:
        s.add(
            IndexedFile(
                file_id="current-only",
                drive="d", filename="a.loft",
                file_path="/d/a.loft",
                file_type="other",
                mime_type=CURRENT_MIME,
                file_size=100,
                active=True,
            )
        )
        s.commit()
    finally:
        s.close()

    with patch("app.database._search_engine", engine):
        from app.database import _migrate_loft_mime_legacy_to_current
        # Should not raise.
        _migrate_loft_mime_legacy_to_current()

    counts = _mime_counts(engine)
    assert counts == {CURRENT_MIME: 1}
