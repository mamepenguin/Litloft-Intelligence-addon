"""Tests for GET /files/{file_id}/chunks/{chunk_id}/excerpt.

Covers:

* Transcript chunks return the chunk text + ±100 chars of neighbour
  context plus ``start_time`` / ``end_time`` (``page`` null).
* Document chunks resolve against the ``fts_text_content`` FTS5 virtual
  table, return ``page`` (when stored), and leave the timestamps null.
* Context composition trims neighbours to 100 chars and inserts ellipses
  only when the neighbour text was actually truncated.
* Bad chunk_id formats (no colon / unknown prefix / non-integer index)
  return 400.
* Missing file / missing chunk / cross-drive file all return 404.
* Per-drive policy (``features.detailed_summaries == "false"``) gates
  the endpoint with a 404, matching the host's addon_feature pre_check.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

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

from app.config import FeaturesConfig  # noqa: E402
from app.database import Base  # noqa: E402
from app.models import IndexedFile, TranscriptChunk  # noqa: E402
from app.routers.files import (  # noqa: E402
    _compose_excerpt,
    _parse_chunk_id,
    get_chunk_excerpt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    """Real SQLite engine with the tables the endpoint touches.

    sqlite-vec isn't available in the vanilla test image, so we skip
    ``_create_vec_tables`` entirely and create only the FTS5 mirror
    we actually query (``fts_text_content``) plus the ORM tables.
    """
    db_path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_text_content "
            "USING fts5(file_id, chunk_index, page, text, "
            "tokenize='trigram')"
        ))

    # Seed one file in 'drive1' and one in 'other' so cross-drive tests
    # have something to probe.
    seed = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        seed.add_all([
            IndexedFile(
                file_id="abc123",
                drive="drive1",
                filename="lecture.mp4",
                file_path="/drives/drive1/lecture.mp4",
                file_type="video",
                mime_type="video/mp4",
                file_size=100,
                active=True,
            ),
            IndexedFile(
                file_id="doc456",
                drive="drive1",
                filename="paper.pdf",
                file_path="/drives/drive1/paper.pdf",
                file_type="document",
                mime_type="application/pdf",
                file_size=200,
                active=True,
            ),
            IndexedFile(
                file_id="elsewhere",
                drive="other",
                filename="x.mp4",
                file_path="/drives/other/x.mp4",
                file_type="video",
                mime_type="video/mp4",
                file_size=10,
                active=True,
            ),
        ])
        seed.commit()
    finally:
        seed.close()

    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def _get_search_db():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # The endpoint imports get_search_db lazily inside the function
    # body (mirrors the pattern in app.routers.files for its other
    # routes), so patching app.database is sufficient — there is no
    # module-level binding in app.routers.files to override.
    monkeypatch.setattr("app.database.get_search_db", _get_search_db)
    return engine, Session


@pytest.fixture()
def feature_enabled(monkeypatch, make_settings):
    """Turn detailed_summaries on so the endpoint's gate passes."""
    settings = make_settings(
        features=FeaturesConfig(detailed_summaries="manual"),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.files.settings", settings)
    return settings


@pytest.fixture()
def feature_disabled(monkeypatch, make_settings):
    """Turn detailed_summaries off so the endpoint 404s."""
    settings = make_settings(
        features=FeaturesConfig(detailed_summaries="false"),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.files.settings", settings)
    return settings


def _seed_transcript_chunks(engine, file_id: str, chunks: list[tuple[int, str, float, float]]) -> None:
    """Insert ``(chunk_index, text, start, end)`` rows for ``file_id``."""
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        for idx, txt, start, end in chunks:
            session.add(TranscriptChunk(
                file_id=file_id,
                chunk_index=idx,
                text=txt,
                language="en",
                timestamp_start=start,
                timestamp_end=end,
            ))
        session.commit()
    finally:
        session.close()


def _seed_document_chunks(engine, file_id: str, chunks: list[tuple[int, str, int | None]]) -> None:
    """Insert ``(chunk_index, text, page)`` rows into fts_text_content."""
    with engine.begin() as conn:
        for idx, txt, page in chunks:
            conn.execute(text(
                "INSERT INTO fts_text_content(file_id, chunk_index, page, text) "
                "VALUES(:fid, :idx, :page, :text)"
            ), {
                "fid": file_id,
                "idx": str(idx),
                "page": str(page) if page is not None else "",
                "text": txt,
            })


# ---------------------------------------------------------------------------
# _parse_chunk_id — pure function
# ---------------------------------------------------------------------------


class TestParseChunkId:
    """Format validation of the ``<source>:<index>`` identifier."""

    def test_transcript_prefix_parses(self):
        assert _parse_chunk_id("transcript:5") == ("transcript", 5)

    def test_document_prefix_parses(self):
        assert _parse_chunk_id("document:12") == ("document", 12)

    def test_missing_colon_rejects(self):
        with pytest.raises(HTTPException) as exc:
            _parse_chunk_id("transcript5")
        assert exc.value.status_code == 400

    def test_empty_string_rejects(self):
        with pytest.raises(HTTPException) as exc:
            _parse_chunk_id("")
        assert exc.value.status_code == 400

    def test_unknown_prefix_rejects(self):
        with pytest.raises(HTTPException) as exc:
            _parse_chunk_id("caption:3")
        assert exc.value.status_code == 400

    def test_non_integer_index_rejects(self):
        with pytest.raises(HTTPException) as exc:
            _parse_chunk_id("transcript:abc")
        assert exc.value.status_code == 400

    def test_negative_index_rejects(self):
        with pytest.raises(HTTPException) as exc:
            _parse_chunk_id("transcript:-1")
        assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# _compose_excerpt — pure function
# ---------------------------------------------------------------------------


class TestComposeExcerpt:
    """Context windowing around the target chunk.

    The helper returns a ``(prefix, target, suffix)`` triple so the
    UI can render the target as a visual highlight against muted
    neighbour context. Concatenating the three strings must still
    reproduce the single-line layout the popover relied on before
    the split.
    """

    def test_returns_target_when_no_neighbours(self):
        assert _compose_excerpt("target", None, None) == ("", "target", "")

    def test_short_neighbours_included_verbatim(self):
        prefix, target, suffix = _compose_excerpt("target", "prev", "next")
        assert prefix == "prev "
        assert target == "target"
        assert suffix == " next"
        # Concatenation still matches the legacy single-line layout.
        assert prefix + target + suffix == "prev target next"

    def test_long_prev_truncated_with_leading_ellipsis(self):
        prev = "A" * 250
        prefix, target, suffix = _compose_excerpt(
            "target", prev, None, context_chars=100
        )
        # Tail(100) preceded by "… " so the UI can render a visual
        # truncation marker on the leading edge.
        assert prefix.startswith("… ")
        assert "A" * 100 in prefix
        # Target stays isolated and unmodified.
        assert target == "target"
        assert suffix == ""

    def test_long_next_truncated_with_trailing_ellipsis(self):
        nxt = "B" * 250
        prefix, target, suffix = _compose_excerpt(
            "target", None, nxt, context_chars=100
        )
        assert prefix == ""
        assert target == "target"
        assert suffix.endswith(" …")
        assert "B" * 100 in suffix

    def test_empty_neighbour_skipped(self):
        assert _compose_excerpt("target", "", None) == ("", "target", "")

    def test_both_neighbours_truncated(self):
        prev = "P" * 250
        nxt = "N" * 250
        prefix, target, suffix = _compose_excerpt(
            "target", prev, nxt, context_chars=100
        )
        assert prefix.startswith("… ")
        assert suffix.endswith(" …")
        assert target == "target"


# ---------------------------------------------------------------------------
# Transcript chunks
# ---------------------------------------------------------------------------


class TestTranscriptExcerpt:
    """End-to-end behaviour for ``transcript:N`` chunk_ids."""

    @pytest.mark.asyncio
    async def test_returns_text_with_timestamps(self, search_db, feature_enabled):
        engine, _ = search_db
        _seed_transcript_chunks(engine, "abc123", [
            (4, "Earlier sentence here.", 30.0, 40.0),
            (5, "The target sentence.", 40.5, 50.0),
            (6, "Following sentence here.", 50.5, 60.0),
        ])

        result = await get_chunk_excerpt("abc123", "transcript:5", "drive1")

        assert result.chunk_id == "transcript:5"
        assert result.file_id == "abc123"
        assert result.page is None
        assert result.start_time == pytest.approx(40.5)
        assert result.end_time == pytest.approx(50.0)
        assert result.target == "The target sentence."
        # Neighbours are short → no ellipsis markers; prefix / suffix
        # retain their space separators so concatenation is seamless.
        assert result.prefix == "Earlier sentence here. "
        assert result.suffix == " Following sentence here."
        assert (
            result.prefix + result.target + result.suffix
            == "Earlier sentence here. The target sentence. Following sentence here."
        )

    @pytest.mark.asyncio
    async def test_context_window_trimmed(self, search_db, feature_enabled):
        engine, _ = search_db
        _seed_transcript_chunks(engine, "abc123", [
            (0, "X" * 500, 0.0, 10.0),
            (1, "Target.", 10.5, 20.0),
            (2, "Y" * 500, 20.5, 30.0),
        ])

        result = await get_chunk_excerpt("abc123", "transcript:1", "drive1")

        # ±100 chars each side; target stays isolated and verbatim.
        assert result.target == "Target."
        assert result.prefix.startswith("…")  # leading ellipsis from prev truncation
        assert result.suffix.endswith("…")    # trailing ellipsis from next truncation
        # The full 500-char neighbours must NOT be present verbatim.
        assert "X" * 500 not in result.prefix
        assert "Y" * 500 not in result.suffix

    @pytest.mark.asyncio
    async def test_edge_chunk_only_has_forward_context(self, search_db, feature_enabled):
        engine, _ = search_db
        _seed_transcript_chunks(engine, "abc123", [
            (0, "First chunk.", 0.0, 5.0),
            (1, "Second chunk.", 5.5, 10.0),
        ])

        result = await get_chunk_excerpt("abc123", "transcript:0", "drive1")
        assert result.prefix == ""
        assert result.target == "First chunk."
        assert result.suffix == " Second chunk."

    @pytest.mark.asyncio
    async def test_nonexistent_chunk_404(self, search_db, feature_enabled):
        engine, _ = search_db
        _seed_transcript_chunks(engine, "abc123", [
            (0, "Only chunk.", 0.0, 5.0),
        ])

        with pytest.raises(HTTPException) as exc:
            await get_chunk_excerpt("abc123", "transcript:99", "drive1")
        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# Document chunks
# ---------------------------------------------------------------------------


class TestDocumentExcerpt:
    """End-to-end behaviour for ``document:N`` chunk_ids."""

    @pytest.mark.asyncio
    async def test_returns_text_with_page(self, search_db, feature_enabled):
        engine, _ = search_db
        _seed_document_chunks(engine, "doc456", [
            (0, "Intro paragraph.", 1),
            (1, "The target paragraph.", 2),
            (2, "Conclusion paragraph.", 2),
        ])

        result = await get_chunk_excerpt("doc456", "document:1", "drive1")

        assert result.chunk_id == "document:1"
        assert result.file_id == "doc456"
        assert result.start_time is None
        assert result.end_time is None
        assert result.page == 2
        assert result.target == "The target paragraph."
        assert result.prefix == "Intro paragraph. "
        assert result.suffix == " Conclusion paragraph."

    @pytest.mark.asyncio
    async def test_page_null_when_extractor_did_not_provide(self, search_db, feature_enabled):
        engine, _ = search_db
        _seed_document_chunks(engine, "doc456", [
            (0, "Plain text chunk.", None),
        ])

        result = await get_chunk_excerpt("doc456", "document:0", "drive1")
        assert result.page is None

    @pytest.mark.asyncio
    async def test_document_context_window_trimmed(self, search_db, feature_enabled):
        engine, _ = search_db
        _seed_document_chunks(engine, "doc456", [
            (0, "P" * 400, 1),
            (1, "target body", 1),
            (2, "N" * 400, 1),
        ])

        result = await get_chunk_excerpt("doc456", "document:1", "drive1")

        assert result.target == "target body"
        assert result.prefix.startswith("…")
        assert result.suffix.endswith("…")
        assert "P" * 400 not in result.prefix
        assert "N" * 400 not in result.suffix

    @pytest.mark.asyncio
    async def test_nonexistent_document_chunk_404(self, search_db, feature_enabled):
        engine, _ = search_db
        _seed_document_chunks(engine, "doc456", [
            (0, "Only chunk.", 1),
        ])

        with pytest.raises(HTTPException) as exc:
            await get_chunk_excerpt("doc456", "document:99", "drive1")
        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# Access control + feature gating
# ---------------------------------------------------------------------------


class TestAccessControl:
    """Per-file / per-drive / per-policy gating of the endpoint."""

    @pytest.mark.asyncio
    async def test_invalid_chunk_id_format_400(self, search_db, feature_enabled):
        with pytest.raises(HTTPException) as exc:
            await get_chunk_excerpt("abc123", "not_a_valid_id", "drive1")
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_file_id_404(self, search_db, feature_enabled):
        with pytest.raises(HTTPException) as exc:
            await get_chunk_excerpt("ghost", "transcript:0", "drive1")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_cross_drive_file_404(self, search_db, feature_enabled):
        # 'elsewhere' exists but lives in drive 'other'; asking as 'drive1'
        # must surface as 404 — not 403 — so we never leak existence.
        with pytest.raises(HTTPException) as exc:
            await get_chunk_excerpt("elsewhere", "transcript:0", "drive1")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_feature_disabled_404(self, search_db, feature_disabled):
        # Even when the file & chunk exist, a drive with
        # detailed_summaries=false must receive a 404.
        engine, _ = search_db
        _seed_transcript_chunks(engine, "abc123", [
            (0, "Present but unreachable.", 0.0, 5.0),
        ])

        with pytest.raises(HTTPException) as exc:
            await get_chunk_excerpt("abc123", "transcript:0", "drive1")
        assert exc.value.status_code == 404
