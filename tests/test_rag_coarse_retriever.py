"""Tests for app.rag.coarse_retriever (Stage 1 shortlist).

These exercise the SQL + scoring path with mocked DB / engine + a
stubbed embedder so no real model load happens.
"""

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

# Stub heavy ML/image deps before importing the module — coarse_retriever
# transitively pulls app.search → app.workers.embedder.
for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "numpy",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.rag import coarse_retriever as cr_mod  # noqa: E402
from app.rag.coarse_retriever import ShortlistResult, coarse_retrieve  # noqa: E402


def _stub_embed_query(*_args, **_kwargs):
    """Embed_query stub: returns a MagicMock that has a tobytes() method."""
    vec = MagicMock()
    vec.tobytes.return_value = b"\x00" * 16
    return vec


@pytest.fixture(autouse=True)
def _patch_embed_query(monkeypatch):
    monkeypatch.setattr(cr_mod, "embed_query", _stub_embed_query)


@pytest.fixture(autouse=True)
def _patch_l2_to_cosine(monkeypatch):
    """Make distance->cosine deterministic for test assertions."""
    monkeypatch.setattr(
        cr_mod, "_l2_to_cosine_similarity", lambda d: 1.0 - d
    )


def _patch_db(monkeypatch, *, vec_rows, drive_count):
    """Wire the engine + session mocks the SQL path expects."""

    # vec_text/embeddings/indexed_files JOIN response.
    engine_conn = MagicMock()
    engine_conn.execute.return_value.fetchall.return_value = vec_rows

    @contextmanager
    def _engine_connect():
        yield engine_conn

    engine = MagicMock()
    engine.connect = _engine_connect
    monkeypatch.setattr(cr_mod, "get_search_engine", lambda: engine)

    # Drive file count session.
    session = MagicMock()
    session.execute.return_value.fetchone.return_value = (drive_count,)

    @contextmanager
    def _get_search_db():
        yield session

    monkeypatch.setattr(cr_mod, "get_search_db", _get_search_db)
    return engine_conn, session


class TestCoarseRetrieve:
    @pytest.mark.asyncio
    async def test_returns_top_k_sorted_by_distance(self, monkeypatch):
        # Closer (smaller distance) ranks higher.
        rows = [
            ("file_b", 0.1),
            ("file_a", 0.4),
            ("file_c", 0.7),
        ]
        _patch_db(monkeypatch, vec_rows=rows, drive_count=200)

        result = await coarse_retrieve("query", drive="Videos", top_k=20)

        assert isinstance(result, ShortlistResult)
        assert result.file_ids == ("file_b", "file_a", "file_c")
        # Cosine = 1 - distance under our stub.
        assert result.scores[0] == pytest.approx(0.9)
        assert result.top_score == pytest.approx(0.9)
        assert result.drive_file_count == 200

    @pytest.mark.asyncio
    async def test_deduplicates_per_file_keeping_best(self, monkeypatch):
        rows = [
            ("file_a", 0.5),
            ("file_a", 0.2),  # better — should win
            ("file_b", 0.3),
        ]
        _patch_db(monkeypatch, vec_rows=rows, drive_count=100)

        result = await coarse_retrieve("query", drive="Videos", top_k=20)

        assert result.file_ids == ("file_a", "file_b")
        assert result.scores[0] == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_empty_match_set_returns_empty_with_top_score_zero(
        self, monkeypatch
    ):
        _patch_db(monkeypatch, vec_rows=[], drive_count=42)

        result = await coarse_retrieve("query", drive="Videos", top_k=20)

        assert result.file_ids == ()
        assert result.scores == ()
        assert result.top_score == 0.0
        assert result.drive_file_count == 42

    @pytest.mark.asyncio
    async def test_drive_file_count_reflects_session_response(
        self, monkeypatch
    ):
        _patch_db(monkeypatch, vec_rows=[("f1", 0.1)], drive_count=5)

        result = await coarse_retrieve("q", drive="Photos", top_k=10)

        assert result.drive_file_count == 5

    @pytest.mark.asyncio
    async def test_passes_drive_and_overfetch_to_query(self, monkeypatch):
        engine_conn, _ = _patch_db(
            monkeypatch, vec_rows=[("f1", 0.1)], drive_count=100
        )

        await coarse_retrieve("q", drive="MyDrive", top_k=7)

        # The SQL query carries drive + over_fetch as bound parameters.
        # over_fetch is max(top_k * 50, 500) to defend against sqlite-vec's
        # global top-K behaviour drowning out target-drive metadata rows
        # with closer non-metadata / wrong-drive vectors.
        first_call = engine_conn.execute.call_args_list[0]
        params = first_call.args[1] if len(first_call.args) > 1 else first_call.kwargs
        assert params["drive"] == "MyDrive"
        assert params["over_fetch"] == max(7 * 50, 500)

    @pytest.mark.asyncio
    async def test_trims_to_top_k_after_dedup(self, monkeypatch):
        # Over-fetch returns 5 distinct files; top_k=2 trims to the closest.
        rows = [
            ("file_a", 0.10),
            ("file_b", 0.20),
            ("file_c", 0.30),
            ("file_d", 0.40),
            ("file_e", 0.50),
        ]
        _patch_db(monkeypatch, vec_rows=rows, drive_count=200)

        result = await coarse_retrieve("q", drive="Videos", top_k=2)

        assert result.file_ids == ("file_a", "file_b")
        assert len(result.scores) == 2

    @pytest.mark.asyncio
    async def test_overfetch_isolates_target_drive_metadata(self, monkeypatch):
        # Simulate the multi-drive × multi-channel scenario: the SQL
        # post-filter has already dropped wrong-drive / wrong-type rows,
        # but only because the over-fetch was wide enough to include the
        # target-drive metadata neighbours. Verify the resulting shortlist
        # is the correctly filtered, ordered, trimmed pool.
        # This stub represents what reaches the JOIN's WHERE clause:
        # only target-drive metadata rows survive.
        rows = [
            ("target_metadata_a", 0.05),
            ("target_metadata_b", 0.12),
        ]
        _patch_db(monkeypatch, vec_rows=rows, drive_count=300)

        result = await coarse_retrieve("q", drive="target", top_k=20)

        assert result.file_ids == ("target_metadata_a", "target_metadata_b")
        # top_score uses the closest distance.
        assert result.top_score == pytest.approx(0.95)
