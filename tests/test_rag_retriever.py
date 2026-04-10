"""Tests for app.rag.retriever module.

Covers retrieve_candidates: wraps app.search.search(), filters file_ids
via the host's Internal API (/api/internal/filter-file-ids), and
enriches results with IndexedFile metadata.

Network calls are mocked — tests never touch real HTTP or databases.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub out heavy ML/image dependencies before importing app modules.
# The retriever imports from app.search, which transitively pulls
# torch/PIL/etc. — the same stubbing pattern used in test_webhook.
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

from app.rag.retriever import RetrievedFile, retrieve_candidates  # noqa: E402
from app.search import (  # noqa: E402
    MatchInfo,
    SearchResponse,
    SearchResult,
    SegmentGroup,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_search_result(
    file_id: str,
    drive: str = "Videos",
    filename: str = "clip.mp4",
    file_type: str = "video",
    score: float = 0.9,
) -> SearchResult:
    """Build a minimal SearchResult for mocking app.search.search()."""
    match = MatchInfo(
        match_type="transcript",
        text="Some spoken text",
        score=score,
        timestamp_start=10.0,
        timestamp_end=20.0,
    )
    segment = SegmentGroup(time_range=(10.0, 20.0), matches=(match,))
    return SearchResult(
        file_id=file_id,
        drive=drive,
        filename=filename,
        file_type=file_type,
        score=score,
        match_types=("transcript",),
        segments=(segment,),
    )


def _make_search_response(results: list[SearchResult]) -> SearchResponse:
    return SearchResponse(
        results=tuple(results),
        total=len(results),
        indexed_files=100,
        service_version="0.1.0",
    )


def _indexed_file_stub(
    file_id: str,
    title: str = "Sample Title",
    description: str = "Sample Description",
) -> dict:
    return {
        "file_id": file_id,
        "title": title,
        "description": description,
    }


# ---------------------------------------------------------------------------
# retrieve_candidates
# ---------------------------------------------------------------------------


class TestRetrieveCandidates:
    """End-to-end behaviour of retrieve_candidates with mocked collaborators."""

    @pytest.mark.asyncio
    async def test_returns_top_k_results_from_search(self, monkeypatch):
        """T1: search() returns N results; retriever returns all up to top_k."""
        results = [
            _make_search_result(f"f{i}", score=0.9 - i * 0.1)
            for i in range(5)
        ]

        monkeypatch.setattr(
            "app.rag.retriever.search",
            MagicMock(return_value=_make_search_response(results)),
        )
        # All files accessible -> identity filter.
        monkeypatch.setattr(
            "app.rag.retriever._filter_file_ids_via_internal_api",
            AsyncMock(side_effect=lambda file_ids, hv_token: set(file_ids)),
        )
        monkeypatch.setattr(
            "app.rag.retriever._get_indexed_files_meta",
            lambda fids: {fid: _indexed_file_stub(fid) for fid in fids},
        )

        retrieved = await retrieve_candidates(
            query="test",
            top_k=5,
            hv_token="token-abc",
        )

        assert len(retrieved) == 5
        assert [r.file_id for r in retrieved] == ["f0", "f1", "f2", "f3", "f4"]
        assert all(isinstance(r, RetrievedFile) for r in retrieved)

    @pytest.mark.asyncio
    async def test_forwards_top_k_to_search(self, monkeypatch):
        """The top_k argument must be passed into app.search.search() as limit."""
        search_spy = MagicMock(return_value=_make_search_response([]))
        monkeypatch.setattr("app.rag.retriever.search", search_spy)
        monkeypatch.setattr(
            "app.rag.retriever._filter_file_ids_via_internal_api",
            AsyncMock(return_value=set()),
        )
        monkeypatch.setattr(
            "app.rag.retriever._get_indexed_files_meta",
            lambda fids: {},
        )

        await retrieve_candidates(
            query="q", top_k=3, hv_token=None
        )

        # search() was called with limit=top_k. Accept either kwarg or
        # positional since the wrapper's signature is flexible.
        kwargs = search_spy.call_args.kwargs
        args = search_spy.call_args.args
        assert kwargs.get("limit") == 3 or (len(args) >= 2 and args[1] == 3)

    @pytest.mark.asyncio
    async def test_forwards_filters_to_search(self, monkeypatch):
        """file_type and drive filters should be passed through."""
        search_spy = MagicMock(return_value=_make_search_response([]))
        monkeypatch.setattr("app.rag.retriever.search", search_spy)
        monkeypatch.setattr(
            "app.rag.retriever._filter_file_ids_via_internal_api",
            AsyncMock(return_value=set()),
        )
        monkeypatch.setattr(
            "app.rag.retriever._get_indexed_files_meta",
            lambda fids: {},
        )

        await retrieve_candidates(
            query="q",
            top_k=5,
            hv_token=None,
            file_type="document",
            drive="Docs",
        )

        kwargs = search_spy.call_args.kwargs
        assert kwargs.get("file_type") == "document"
        assert kwargs.get("drive") == "Docs"

    @pytest.mark.asyncio
    async def test_filters_inaccessible_files(self, monkeypatch):
        """T2: file_ids not in the allowed set from Internal API are dropped."""
        results = [
            _make_search_result("allowed-1"),
            _make_search_result("forbidden-1"),
            _make_search_result("allowed-2"),
            _make_search_result("forbidden-2"),
        ]
        monkeypatch.setattr(
            "app.rag.retriever.search",
            MagicMock(return_value=_make_search_response(results)),
        )
        # Only half the files are accessible.
        monkeypatch.setattr(
            "app.rag.retriever._filter_file_ids_via_internal_api",
            AsyncMock(return_value={"allowed-1", "allowed-2"}),
        )
        monkeypatch.setattr(
            "app.rag.retriever._get_indexed_files_meta",
            lambda fids: {fid: _indexed_file_stub(fid) for fid in fids},
        )

        retrieved = await retrieve_candidates(
            query="test",
            top_k=10,
            hv_token="valid-token",
        )

        file_ids = [r.file_id for r in retrieved]
        assert "allowed-1" in file_ids
        assert "allowed-2" in file_ids
        assert "forbidden-1" not in file_ids
        assert "forbidden-2" not in file_ids
        assert len(retrieved) == 2

    @pytest.mark.asyncio
    async def test_internal_api_called_with_token(self, monkeypatch):
        """The caller's X-HV-Token should be forwarded to the Internal API."""
        filter_spy = AsyncMock(return_value={"f1"})

        monkeypatch.setattr(
            "app.rag.retriever.search",
            MagicMock(return_value=_make_search_response([
                _make_search_result("f1"),
            ])),
        )
        monkeypatch.setattr(
            "app.rag.retriever._filter_file_ids_via_internal_api", filter_spy
        )
        monkeypatch.setattr(
            "app.rag.retriever._get_indexed_files_meta",
            lambda fids: {fid: _indexed_file_stub(fid) for fid in fids},
        )

        await retrieve_candidates(
            query="test",
            top_k=5,
            hv_token="my-secret-token",
        )

        # The hv_token kwarg (or positional) should match.
        call_kwargs = filter_spy.call_args.kwargs
        assert call_kwargs.get("hv_token") == "my-secret-token" or (
            "my-secret-token" in filter_spy.call_args.args
        )

    @pytest.mark.asyncio
    async def test_handles_none_hv_token(self, monkeypatch):
        """T3: hv_token=None must not crash; access filter still runs."""
        filter_spy = AsyncMock(return_value={"f1"})
        monkeypatch.setattr(
            "app.rag.retriever.search",
            MagicMock(return_value=_make_search_response([
                _make_search_result("f1"),
            ])),
        )
        monkeypatch.setattr(
            "app.rag.retriever._filter_file_ids_via_internal_api", filter_spy
        )
        monkeypatch.setattr(
            "app.rag.retriever._get_indexed_files_meta",
            lambda fids: {fid: _indexed_file_stub(fid) for fid in fids},
        )

        retrieved = await retrieve_candidates(
            query="test",
            top_k=5,
            hv_token=None,
        )

        # The filter function should be invoked exactly once even when
        # token is None (it decides whether to skip or fall back).
        assert filter_spy.await_count == 1
        assert len(retrieved) == 1

    @pytest.mark.asyncio
    async def test_empty_search_returns_empty(self, monkeypatch):
        """T4: when search() yields nothing, retriever short-circuits to []."""
        monkeypatch.setattr(
            "app.rag.retriever.search",
            MagicMock(return_value=_make_search_response([])),
        )
        # The filter and meta lookups should not matter; but if called,
        # they should return empty collections.
        filter_spy = AsyncMock(return_value=set())
        monkeypatch.setattr(
            "app.rag.retriever._filter_file_ids_via_internal_api", filter_spy
        )
        monkeypatch.setattr(
            "app.rag.retriever._get_indexed_files_meta",
            lambda fids: {},
        )

        retrieved = await retrieve_candidates(
            query="q", top_k=5, hv_token="token"
        )

        assert retrieved == []

    @pytest.mark.asyncio
    async def test_attaches_indexed_file_metadata(self, monkeypatch):
        """T5: RetrievedFile should carry title/description from IndexedFile."""
        monkeypatch.setattr(
            "app.rag.retriever.search",
            MagicMock(return_value=_make_search_response([
                _make_search_result("f1"),
            ])),
        )
        monkeypatch.setattr(
            "app.rag.retriever._filter_file_ids_via_internal_api",
            AsyncMock(return_value={"f1"}),
        )
        monkeypatch.setattr(
            "app.rag.retriever._get_indexed_files_meta",
            lambda fids: {
                "f1": {
                    "file_id": "f1",
                    "title": "My Great Video",
                    "description": "A long description",
                }
            },
        )

        retrieved = await retrieve_candidates(
            query="test", top_k=5, hv_token="token"
        )

        assert len(retrieved) == 1
        rf = retrieved[0]
        assert rf.title == "My Great Video"
        assert rf.description == "A long description"
        # Basic fields from SearchResult must also be present.
        assert rf.file_id == "f1"
        assert rf.drive == "Videos"
        assert rf.filename == "clip.mp4"
        assert rf.file_type == "video"
        assert rf.score == pytest.approx(0.9)
        assert "transcript" in rf.match_types
        assert len(rf.segments) >= 1

    @pytest.mark.asyncio
    async def test_title_and_description_can_be_none(self, monkeypatch):
        """Missing metadata should not crash; None passes through."""
        monkeypatch.setattr(
            "app.rag.retriever.search",
            MagicMock(return_value=_make_search_response([
                _make_search_result("f1"),
            ])),
        )
        monkeypatch.setattr(
            "app.rag.retriever._filter_file_ids_via_internal_api",
            AsyncMock(return_value={"f1"}),
        )
        # Meta lookup returns empty dict -> no title/description.
        monkeypatch.setattr(
            "app.rag.retriever._get_indexed_files_meta",
            lambda fids: {},
        )

        retrieved = await retrieve_candidates(
            query="test", top_k=5, hv_token="token"
        )

        assert len(retrieved) == 1
        rf = retrieved[0]
        # The spec types these as `str | None`; None is an acceptable
        # fallback when no IndexedFile row exists.
        assert rf.title in (None, "", "Sample Title")
        assert rf.description in (None, "", "Sample Description")


# ---------------------------------------------------------------------------
# RetrievedFile dataclass
# ---------------------------------------------------------------------------


class TestRetrievedFileDataclass:
    """RetrievedFile is a frozen dataclass with the expected fields."""

    def test_fields_present(self):
        rf = RetrievedFile(
            file_id="f1",
            drive="Videos",
            filename="a.mp4",
            file_type="video",
            title="T",
            description="D",
            score=0.8,
            match_types=("transcript",),
            segments=(),
        )
        assert rf.file_id == "f1"
        assert rf.drive == "Videos"
        assert rf.filename == "a.mp4"
        assert rf.file_type == "video"
        assert rf.title == "T"
        assert rf.description == "D"
        assert rf.score == 0.8
        assert rf.match_types == ("transcript",)
        assert rf.segments == ()

    def test_is_frozen(self):
        rf = RetrievedFile(
            file_id="f1",
            drive="Videos",
            filename="a.mp4",
            file_type="video",
            title=None,
            description=None,
            score=0.8,
            match_types=(),
            segments=(),
        )
        with pytest.raises(Exception):
            rf.file_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Contract tests: _filter_file_ids_via_internal_api against the real
# Internal API response shape. These do NOT monkeypatch the helper
# itself — they mock the HTTP layer (httpx.AsyncClient.post) so we
# exercise the full parse path. Without these, a JSON-key mismatch
# like the one that shipped in the initial RAG implementation would
# silently pass every unit test.
# ---------------------------------------------------------------------------


class TestFilterFileIdsContract:
    """Exercise _filter_file_ids_via_internal_api against real-ish HTTP."""

    @pytest.mark.asyncio
    async def test_parses_accessible_key(self, monkeypatch):
        """The host's Internal API returns {"accessible": [...]}.

        This is the **canonical** response shape per
        backend/app/routers/internal.py::filter_file_ids. The retriever
        MUST read the ``accessible`` key; an earlier implementation read
        ``allowed`` and silently dropped every candidate.
        """
        from app.rag.retriever import _filter_file_ids_via_internal_api

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json = MagicMock(
            return_value={"accessible": ["f1", "f2"]}
        )

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                self.cookies_captured = kwargs.get("cookies", {})

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

            async def post(self, url, json=None):
                _FakeClient.last_url = url
                _FakeClient.last_json = json
                return fake_response

        monkeypatch.setattr(
            "app.rag.retriever.httpx.AsyncClient", _FakeClient
        )
        monkeypatch.setenv(
            "HOMEVAULT_INTERNAL_API_URL",
            "http://backend:8000/api/internal",
        )

        result = await _filter_file_ids_via_internal_api(
            file_ids=["f1", "f2", "f3"],
            hv_token="jwt-value",
        )

        assert result == {"f1", "f2"}
        assert _FakeClient.last_url == (
            "http://backend:8000/api/internal/filter-file-ids"
        )
        assert _FakeClient.last_json == {"file_ids": ["f1", "f2", "f3"]}

    @pytest.mark.asyncio
    async def test_forwards_access_token_cookie(self, monkeypatch):
        """Auth must be transmitted as an ``access_token`` cookie.

        The host's ``get_unlocked_groups`` dependency reads the
        ``access_token`` cookie via ``request.cookies.get`` — NOT a
        custom header. The retriever must configure httpx.AsyncClient
        with ``cookies={"access_token": <jwt>}``.
        """
        from app.rag.retriever import _filter_file_ids_via_internal_api

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json = MagicMock(
            return_value={"accessible": ["f1"]}
        )
        captured: dict = {}

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                captured["cookies"] = kwargs.get("cookies")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

            async def post(self, url, json=None):
                return fake_response

        monkeypatch.setattr(
            "app.rag.retriever.httpx.AsyncClient", _FakeClient
        )

        await _filter_file_ids_via_internal_api(
            file_ids=["f1"],
            hv_token="my-jwt",
        )

        assert captured["cookies"] == {"access_token": "my-jwt"}

    @pytest.mark.asyncio
    async def test_no_cookie_when_token_none(self, monkeypatch):
        """When ``hv_token`` is None, the cookie dict must be empty.

        An empty cookies dict causes the host's ``get_unlocked_groups``
        dependency to return ``[]`` — i.e., the caller is treated as
        unauthenticated and only fully-public drives survive the filter.
        """
        from app.rag.retriever import _filter_file_ids_via_internal_api

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json = MagicMock(return_value={"accessible": []})
        captured: dict = {}

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                captured["cookies"] = kwargs.get("cookies")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

            async def post(self, url, json=None):
                return fake_response

        monkeypatch.setattr(
            "app.rag.retriever.httpx.AsyncClient", _FakeClient
        )

        result = await _filter_file_ids_via_internal_api(
            file_ids=["f1"],
            hv_token=None,
        )

        assert captured["cookies"] == {}
        assert result == set()

    @pytest.mark.asyncio
    async def test_accepts_legacy_allowed_key_with_warning(
        self, monkeypatch, caplog
    ):
        """Legacy hosts returning ``{"allowed": ...}`` should still work.

        A host deployed before the key was standardized may still
        return the old ``allowed`` key. The retriever must accept it
        and log a deprecation warning so operators notice.
        """
        import logging as _logging
        from app.rag.retriever import _filter_file_ids_via_internal_api

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json = MagicMock(
            return_value={"allowed": ["f1"]}
        )

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

            async def post(self, url, json=None):
                return fake_response

        monkeypatch.setattr(
            "app.rag.retriever.httpx.AsyncClient", _FakeClient
        )

        with caplog.at_level(_logging.WARNING, logger="app.rag.retriever"):
            result = await _filter_file_ids_via_internal_api(
                file_ids=["f1"],
                hv_token="t",
            )

        assert result == {"f1"}
        assert any(
            "legacy 'allowed'" in rec.message for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_unknown_response_shape_fails_closed(self, monkeypatch):
        """Any unexpected response shape must return an empty set."""
        from app.rag.retriever import _filter_file_ids_via_internal_api

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json = MagicMock(return_value={"unexpected": "shape"})

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

            async def post(self, url, json=None):
                return fake_response

        monkeypatch.setattr(
            "app.rag.retriever.httpx.AsyncClient", _FakeClient
        )

        result = await _filter_file_ids_via_internal_api(
            file_ids=["f1"],
            hv_token="t",
        )

        assert result == set()
