"""Ask draws citations only from sources the viewer has vouched for.

Spec: docs/superpowers/specs/2026-08-29-web-clip-promotion.md §6.

The filter rides on the access call that already exists rather than opening
a second path into core's schema, so these tests pin the request body the
retriever actually sends.

Ordinary search does not use this path and keeps returning unverified files:
a clip stays findable, it just stops acting as evidence.
"""

import sys
from unittest.mock import MagicMock

import httpx
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
    "numpy",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.credentials import CallerCredential  # noqa: E402
from app.rag import retriever  # noqa: E402


def _install_transport(monkeypatch, handler):
    orig = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return orig(*args, **kwargs)

    monkeypatch.setattr(retriever.httpx, "AsyncClient", _factory)


@pytest.mark.asyncio
async def test_grounding_requests_verified_by_default(monkeypatch):
    """Forgetting the argument must give the safer behaviour, not the looser."""
    received: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        received["body"] = req.read().decode()
        received["url"] = str(req.url)
        return httpx.Response(200, json={"accessible": ["f1"], "trust_filtered": True})

    _install_transport(monkeypatch, handler)
    allowed = await retriever._filter_file_ids_via_internal_api(
        file_ids=["f1", "f2"], credential=None
    )

    assert received["url"].endswith("/filter-file-ids")
    assert '"trust_tier": "verified"' in received["body"] or (
        '"trust_tier":"verified"' in received["body"]
    )
    assert allowed == {"f1"}


@pytest.mark.asyncio
async def test_fails_closed_when_host_did_not_apply_the_filter(monkeypatch):
    """An older core drops the unknown field and answers with everything.

    Reading that back as verified would let Ask cite unvouched sources, so a
    requested-but-unconfirmed filter is treated like any other failure on
    this path.
    """
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"accessible": ["f1", "f2"]})

    _install_transport(monkeypatch, handler)
    allowed = await retriever._filter_file_ids_via_internal_api(
        file_ids=["f1", "f2"], credential=None
    )

    assert allowed == set()


@pytest.mark.asyncio
async def test_missing_marker_is_fine_when_no_filter_was_requested(monkeypatch):
    """Find and other unfiltered callers must not be broken by the check."""
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"accessible": ["f1", "f2"]})

    _install_transport(monkeypatch, handler)
    allowed = await retriever._filter_file_ids_via_internal_api(
        file_ids=["f1", "f2"], credential=None, trust_tier=None
    )

    assert allowed == {"f1", "f2"}


@pytest.mark.asyncio
async def test_trust_filter_can_be_disabled_explicitly(monkeypatch):
    received: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        received["body"] = req.read().decode()
        return httpx.Response(200, json={"accessible": []})

    _install_transport(monkeypatch, handler)
    await retriever._filter_file_ids_via_internal_api(
        file_ids=["f1"], credential=None, trust_tier=None
    )

    assert "trust_tier" not in received["body"]


@pytest.mark.asyncio
async def test_unverified_files_are_dropped_from_candidates(monkeypatch):
    """The host does the narrowing; the retriever must honour the result."""
    def handler(req: httpx.Request) -> httpx.Response:
        # Core answers with only the verified subset.
        return httpx.Response(
            200,
            json={"accessible": ["verified-file"], "trust_filtered": True},
        )

    _install_transport(monkeypatch, handler)
    allowed = await retriever._filter_file_ids_via_internal_api(
        file_ids=["verified-file", "fresh-clip"],
        credential=CallerCredential(cookie="access_token=t"),
    )

    assert allowed == {"verified-file"}


@pytest.mark.asyncio
async def test_still_fails_closed_on_transport_error(monkeypatch):
    """Adding the trust field must not weaken the existing access posture."""
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    _install_transport(monkeypatch, handler)
    allowed = await retriever._filter_file_ids_via_internal_api(
        file_ids=["f1"], credential=None
    )

    assert allowed == set()


@pytest.mark.asyncio
async def test_credential_is_still_forwarded(monkeypatch):
    received: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        received["cookie"] = req.headers.get("Cookie")
        return httpx.Response(200, json={"accessible": []})

    _install_transport(monkeypatch, handler)
    await retriever._filter_file_ids_via_internal_api(
        file_ids=["f1"],
        credential=CallerCredential(cookie="access_token=my-secret-token"),
    )

    assert received["cookie"] == "access_token=my-secret-token"


@pytest.mark.asyncio
async def test_find_keeps_unverified_files_visible(monkeypatch):
    """Find presents files; it is not a citation surface.

    Ask must not quote an unvouched source back as evidence, but Find
    answering "which of my files are about X" has to keep showing clips — they
    are findable, just not evidence. The two Stage D calls opt out explicitly.
    """
    import inspect

    from app.rag import service

    source = inspect.getsource(service.find_files)
    # Both Stage D branches (per-term expansion and the single-query path).
    assert source.count("trust_tier=None") == 2


@pytest.mark.asyncio
async def test_retrieval_entry_points_default_to_verified():
    """A caller that forgets the argument gets the safer behaviour."""
    import inspect

    from app.rag.retriever import retrieve_candidates, retrieve_with_keywords

    for fn in (retrieve_candidates, retrieve_with_keywords):
        default = inspect.signature(fn).parameters["trust_tier"].default
        assert default == "verified", fn.__name__


def test_pool_widens_only_when_a_trust_filter_is_active():
    """Find and every pre-existing caller must keep their exact behaviour."""
    from app.rag.retriever import _search_pool_size

    assert _search_pool_size(10, None) == 10
    assert _search_pool_size(10, "verified") == 40


def test_pool_growth_is_capped():
    """A large top_k must not turn into a pathologically wide scan."""
    from app.rag.retriever import _TRUST_OVERSAMPLE_MAX, _search_pool_size

    assert _search_pool_size(1000, "verified") == _TRUST_OVERSAMPLE_MAX


@pytest.mark.asyncio
async def test_verified_source_survives_higher_ranked_clips(monkeypatch):
    """The filter runs after ranking, so the pool must be drawn wider.

    With a pool of exactly top_k, unverified clips that outrank a verified
    source consume the whole budget and are then discarded, leaving Ask with
    nothing to cite even though an eligible source existed.
    """
    from unittest.mock import AsyncMock

    from app.rag.retriever import retrieve_with_keywords

    ranked = [f"clip{i}" for i in range(6)] + ["verified-source"]
    captured: dict = {}

    def _fake_search(keywords, **kwargs):
        captured["limit"] = kwargs["limit"]
        response = MagicMock()
        response.results = [
            _stub_result(fid) for fid in ranked[: kwargs["limit"]]
        ]
        return response

    monkeypatch.setattr("app.rag.retriever.search", _fake_search)
    monkeypatch.setattr(
        "app.rag.retriever._filter_file_ids_via_internal_api",
        AsyncMock(side_effect=lambda file_ids, credential, **kw: (
            {f for f in file_ids if f == "verified-source"}
        )),
    )
    monkeypatch.setattr(
        "app.rag.retriever._get_indexed_files_meta",
        lambda fids: {fid: {} for fid in fids},
    )

    retrieved = await retrieve_with_keywords(
        keywords="q", top_k=3, credential=None
    )

    assert captured["limit"] == 12  # 3 * oversample factor
    assert [r.file_id for r in retrieved] == ["verified-source"]


def _stub_result(file_id: str):
    result = MagicMock()
    result.file_id = file_id
    result.score = 1.0
    result.segments = []
    return result


@pytest.mark.asyncio
async def test_widens_until_the_budget_is_filled(monkeypatch):
    """A fixed multiplier alone starves a query whose eligible hits rank low.

    Here the first verified source sits at rank 21 with top_k=5, so the
    initial pool of 20 contains none. The loop must widen rather than return
    empty while citable sources exist.
    """
    from unittest.mock import AsyncMock

    from app.rag.retriever import retrieve_with_keywords

    ranked = [f"clip{i}" for i in range(20)] + [f"verified{i}" for i in range(5)]
    limits: list[int] = []

    def _fake_search(keywords, **kwargs):
        limits.append(kwargs["limit"])
        response = MagicMock()
        response.results = [
            _stub_result(fid) for fid in ranked[: kwargs["limit"]]
        ]
        return response

    monkeypatch.setattr("app.rag.retriever.search", _fake_search)
    monkeypatch.setattr(
        "app.rag.retriever._filter_file_ids_via_internal_api",
        AsyncMock(side_effect=lambda file_ids, credential, **kw: (
            {f for f in file_ids if f.startswith("verified")}
        )),
    )
    monkeypatch.setattr(
        "app.rag.retriever._get_indexed_files_meta",
        lambda fids: {fid: {} for fid in fids},
    )

    retrieved = await retrieve_with_keywords(
        keywords="q", top_k=5, credential=None
    )

    assert limits[0] == 20          # 5 * oversample factor
    assert len(limits) > 1          # widened rather than giving up
    assert len(retrieved) == 5
    assert all(r.file_id.startswith("verified") for r in retrieved)


@pytest.mark.asyncio
async def test_stops_widening_when_the_index_is_exhausted(monkeypatch):
    """A short result set means there is nothing more to find."""
    from unittest.mock import AsyncMock

    from app.rag.retriever import retrieve_with_keywords

    limits: list[int] = []

    def _fake_search(keywords, **kwargs):
        limits.append(kwargs["limit"])
        response = MagicMock()
        response.results = [_stub_result("clip0"), _stub_result("verified0")]
        return response

    monkeypatch.setattr("app.rag.retriever.search", _fake_search)
    monkeypatch.setattr(
        "app.rag.retriever._filter_file_ids_via_internal_api",
        AsyncMock(side_effect=lambda file_ids, credential, **kw: {"verified0"}),
    )
    monkeypatch.setattr(
        "app.rag.retriever._get_indexed_files_meta",
        lambda fids: {fid: {} for fid in fids},
    )

    retrieved = await retrieve_with_keywords(
        keywords="q", top_k=5, credential=None
    )

    assert len(limits) == 1  # no pointless re-query
    assert [r.file_id for r in retrieved] == ["verified0"]
