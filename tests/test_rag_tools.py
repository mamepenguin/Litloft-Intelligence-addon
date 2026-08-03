"""Unit tests for the Phase 1.B agentic tool wrappers.

Focus: pure-Python plumbing (token budget, ToolContext, schemas,
chunk slicing, token-cap truncation). The HTTP-hitting paths
(``search_files`` end-to-end, ``get_file_detail`` triple-fetch) are
covered with mocked ``retrieve_candidates`` / ``httpx.AsyncClient`` /
``get_search_db_read``.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# Heavyweight-dep stubs (mirrors other test files).
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

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _allow_all_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the access gate a no-op for tests that don't override it.

    The wrappers call ``ensure_access(file_ids, credential)`` which in
    production hits the host's filter-file-ids endpoint. For unit
    tests we treat every requested ID as allowed; tests that exercise
    the denial path patch ``ensure_access`` explicitly.
    """

    async def _allow(file_ids, credential=None):
        return {fid for fid in file_ids if isinstance(fid, str)}

    monkeypatch.setattr(
        "app.rag.tools.get_file_detail.ensure_access", _allow
    )
    monkeypatch.setattr(
        "app.rag.tools.get_file_chunks.ensure_access", _allow
    )
    monkeypatch.setattr(
        "app.rag.tools.get_related_files.ensure_access", _allow
    )

from app.rag.tools._access import (  # noqa: E402
    ALLOW_LIST_CAP,
    is_valid_file_id,
    is_valid_kind,
)
from app.rag.tools.budget import (  # noqa: E402
    DEFAULT_PER_CALL_TOKEN_CAP,
    estimate_payload_tokens,
    estimate_tokens,
    fits_per_call_cap,
    remaining_budget,
)
from app.rag.tools.context import ToolContext, ToolResultEnvelope  # noqa: E402
from app.rag.tools.get_file_chunks import (  # noqa: E402
    _ChunkRow,
    _apply_token_cap,
    _clip_range,
    _split_text_into_chunks,
    get_file_chunks,
)
from app.rag.tools.schemas import TOOL_NAMES, TOOL_SCHEMAS  # noqa: E402


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


def test_estimate_tokens_none_and_empty() -> None:
    assert estimate_tokens(None) == 0
    assert estimate_tokens("") == 0


def test_estimate_tokens_grows_with_length() -> None:
    short = estimate_tokens("abc")
    long = estimate_tokens("abc" * 100)
    assert long > short


def test_estimate_tokens_cjk_one_per_char_roughly() -> None:
    # 10 CJK chars ≈ at least 10 tokens (the heuristic counts every
    # CJK char as one token).
    assert estimate_tokens("こんにちは世界です。") >= 10


def test_estimate_payload_tokens_dict() -> None:
    n = estimate_payload_tokens({"key": "value", "list": [1, 2, 3]})
    assert n > 0


def test_estimate_payload_tokens_none_is_zero() -> None:
    assert estimate_payload_tokens(None) == 0


def test_fits_per_call_cap() -> None:
    assert fits_per_call_cap(100)
    assert fits_per_call_cap(DEFAULT_PER_CALL_TOKEN_CAP)
    assert not fits_per_call_cap(DEFAULT_PER_CALL_TOKEN_CAP + 1)
    assert fits_per_call_cap(100, cap=200)
    assert not fits_per_call_cap(300, cap=200)


def test_remaining_budget_clamps_to_zero() -> None:
    assert remaining_budget(50, 100) == 50
    assert remaining_budget(150, 100) == 0


# ---------------------------------------------------------------------------
# _access helpers
# ---------------------------------------------------------------------------


def test_is_valid_file_id_accepts_typical_shapes() -> None:
    assert is_valid_file_id("abc12345")
    assert is_valid_file_id("AaZz09_-")
    assert is_valid_file_id("x" * 64)


def test_is_valid_file_id_rejects_bad_shapes() -> None:
    assert not is_valid_file_id("")
    assert not is_valid_file_id("../etc")
    assert not is_valid_file_id("a/b")
    assert not is_valid_file_id("a?b")
    assert not is_valid_file_id("a b")
    assert not is_valid_file_id("x" * 65)
    assert not is_valid_file_id(123)  # type: ignore[arg-type]


def test_is_valid_kind() -> None:
    assert is_valid_kind("cite")
    assert is_valid_kind("see_also")
    assert is_valid_kind("a1-2.b")
    assert not is_valid_kind("Cite")
    assert not is_valid_kind("1foo")
    assert not is_valid_kind("")
    assert not is_valid_kind(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ToolContext
# ---------------------------------------------------------------------------


def test_tool_context_accumulates_file_ids() -> None:
    ctx = ToolContext(drive="x")
    ctx.register_file_ids(["a", "b", "a"])
    assert ctx.tool_returned_file_ids == {"a", "b"}
    ctx.register_file_ids(["", None, "c"])  # type: ignore[list-item]
    assert ctx.tool_returned_file_ids == {"a", "b", "c"}


def test_tool_context_register_tool_call_order() -> None:
    ctx = ToolContext()
    ctx.register_tool_call("search_files")
    ctx.register_tool_call("get_file_chunks")
    assert ctx.tool_calls == ["search_files", "get_file_chunks"]


def test_tool_context_register_result_tokens() -> None:
    ctx = ToolContext()
    ctx.register_result_tokens(100)
    ctx.register_result_tokens(50)
    ctx.register_result_tokens(0)  # ignored
    ctx.register_result_tokens(-5)  # ignored
    assert ctx.cumulative_tokens == 150


def test_tool_context_is_allowed_citation() -> None:
    ctx = ToolContext()
    ctx.register_file_ids(["a", "b"])
    assert ctx.is_allowed_citation("a")
    assert not ctx.is_allowed_citation("z")


def test_tool_context_allow_list_cap() -> None:
    ctx = ToolContext()
    ctx.register_file_ids([f"id{i}" for i in range(ALLOW_LIST_CAP + 50)])
    assert len(ctx.tool_returned_file_ids) == ALLOW_LIST_CAP


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------


def test_tool_names_match_schema_names() -> None:
    assert TOOL_NAMES == (
        "search_files",
        "get_file_detail",
        "get_file_chunks",
        "get_related_files",
    )


def test_each_schema_has_function_and_parameters() -> None:
    for s in TOOL_SCHEMAS:
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "description" in s["function"]
        assert "parameters" in s["function"]
        params = s["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params


def test_get_file_chunks_schema_modes_enum() -> None:
    chunks_schema = next(
        s for s in TOOL_SCHEMAS if s["function"]["name"] == "get_file_chunks"
    )
    enum = chunks_schema["function"]["parameters"]["properties"]["mode"]["enum"]
    assert enum == ["summary", "full"]


# ---------------------------------------------------------------------------
# get_file_chunks: helper-level
# ---------------------------------------------------------------------------


def test_split_text_into_chunks_empty() -> None:
    assert _split_text_into_chunks("") == []


def test_split_text_into_chunks_produces_sequential_ids() -> None:
    text = "x" * 2500
    chunks = _split_text_into_chunks(text)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert sum(len(c.text) for c in chunks) == len(text)


def test_clip_range_within_bounds() -> None:
    chunks = [_ChunkRow(i, f"chunk {i}", f"t{i}") for i in range(10)]
    sliced, truncated, warn = _clip_range(chunks, (2, 5))
    assert [c.chunk_id for c in sliced] == [2, 3, 4, 5]
    assert truncated is False
    assert warn is None


def test_clip_range_caps_at_50() -> None:
    chunks = [_ChunkRow(i, f"chunk {i}", "x") for i in range(100)]
    sliced, truncated, warn = _clip_range(chunks, (0, 70))
    assert len(sliced) == 50
    assert truncated is True
    assert warn is not None and "50" in warn


def test_clip_range_clamps_out_of_bounds() -> None:
    chunks = [_ChunkRow(i, f"chunk {i}", "x") for i in range(5)]
    sliced, _, _ = _clip_range(chunks, (-10, 999))
    assert [c.chunk_id for c in sliced] == [0, 1, 2, 3, 4]


def test_apply_token_cap_no_op_for_small_payloads() -> None:
    rows = [{"chunk_id": 0, "location": "0:00", "text": "small"}]
    out, truncated, warn = _apply_token_cap(rows)
    assert out == rows
    assert truncated is False
    assert warn is None


def test_apply_token_cap_truncates_oversize() -> None:
    big_text = "字" * 20000  # well over the per-call cap
    rows = [{"chunk_id": i, "location": "x", "text": big_text} for i in range(5)]
    out, truncated, warn = _apply_token_cap(rows)
    assert truncated is True
    assert warn is not None and "token cap" in warn
    assert 0 < len(out) < 5


# ---------------------------------------------------------------------------
# get_file_chunks: tool entry point (mocked DB)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_file_chunks_summary_mode_default() -> None:
    rows = [_ChunkRow(i, f"{i}:00", f"text {i}") for i in range(3)]
    with patch(
        "app.rag.tools.get_file_chunks._load_transcript_chunks",
        return_value=(rows, False),
    ):
        ctx = ToolContext()
        env = await get_file_chunks(
            context=ctx, file_id="f", type="transcript"
        )
    assert env.payload["mode"] == "summary"
    assert len(env.payload["chunks"]) == 3
    assert all("preview" in c for c in env.payload["chunks"])
    assert ctx.tool_calls == ["get_file_chunks"]
    assert "f" in ctx.tool_returned_file_ids


@pytest.mark.asyncio
async def test_get_file_chunks_full_requires_range() -> None:
    rows = [_ChunkRow(0, "0:00", "x")]
    with patch(
        "app.rag.tools.get_file_chunks._load_transcript_chunks",
        return_value=(rows, False),
    ):
        ctx = ToolContext()
        env = await get_file_chunks(
            context=ctx, file_id="f", type="transcript", mode="full"
        )
    assert "error" in env.payload
    assert env.warning is not None


@pytest.mark.asyncio
async def test_get_file_chunks_full_returns_text_in_range() -> None:
    rows = [_ChunkRow(i, f"{i}:00", f"text {i}") for i in range(5)]
    with patch(
        "app.rag.tools.get_file_chunks._load_transcript_chunks",
        return_value=(rows, False),
    ):
        env = await get_file_chunks(
            context=ToolContext(),
            file_id="f",
            type="transcript",
            mode="full",
            range=[1, 3],
        )
    assert env.payload["mode"] == "full"
    chunks_out = env.payload["chunks"]
    assert [c["chunk_id"] for c in chunks_out] == [1, 2, 3]
    assert all("text" in c for c in chunks_out)


@pytest.mark.asyncio
async def test_get_file_chunks_per_chunk_truncation_warning() -> None:
    rows = [_ChunkRow(0, "0:00", "x" * 50)]
    with patch(
        "app.rag.tools.get_file_chunks._load_transcript_chunks",
        return_value=(rows, True),
    ):
        env = await get_file_chunks(
            context=ToolContext(), file_id="f", type="transcript"
        )
    assert env.payload.get("truncated") is True
    assert any("per-chunk" in w for w in env.payload.get("warnings", []))


@pytest.mark.asyncio
async def test_get_file_chunks_access_denied_returns_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM-supplied file_id outside the viewer's access must 'not_found'."""

    async def _deny(file_ids, credential=None):
        return set()

    monkeypatch.setattr(
        "app.rag.tools.get_file_chunks.ensure_access", _deny
    )
    env = await get_file_chunks(
        context=ToolContext(), file_id="forbidden", type="transcript"
    )
    assert env.payload["error"] == "not_found"
    assert env.warning == "file not found"


@pytest.mark.asyncio
async def test_get_file_chunks_text_fetch_5xx_marks_internal_api_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fail-loud: 5xx on the text-content endpoint surfaces as a clear
    error envelope rather than 'no body' which is indistinguishable
    from a legitimately empty file."""
    import httpx as _httpx

    from app.rag.tools.get_file_chunks import _TextFetchError

    async def _raise(_id):
        raise _TextFetchError("internal API call failed: HTTPStatusError")

    monkeypatch.setattr(
        "app.rag.tools.get_file_chunks._load_text_chunks", _raise
    )
    env = await get_file_chunks(
        context=ToolContext(), file_id="f", type="text"
    )
    assert env.payload.get("error") == "internal_api_failed"
    assert env.warning is not None
    # The LLM must learn not to retry this file.
    assert "should not retry" in env.warning


@pytest.mark.asyncio
async def test_get_file_chunks_invalid_type() -> None:
    env = await get_file_chunks(
        context=ToolContext(), file_id="f", type="bogus"
    )
    assert env.payload == {"error": "type must be 'transcript' or 'text'"}


@pytest.mark.asyncio
async def test_get_file_chunks_invalid_file_id() -> None:
    env = await get_file_chunks(
        context=ToolContext(), file_id="", type="transcript"
    )
    assert "invalid" in (env.warning or "")


@pytest.mark.asyncio
async def test_get_file_chunks_text_mode_fetches_from_core() -> None:
    sample = "line 1\n" * 200  # ~1400 chars, two synthetic chunks
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = sample
    mock_response.raise_for_status = MagicMock()

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.get = AsyncMock(return_value=mock_response)

    with patch(
        "app.rag.tools.get_file_chunks.httpx.AsyncClient",
        return_value=client,
    ):
        env = await get_file_chunks(
            context=ToolContext(), file_id="f", type="text"
        )
    assert env.payload["type"] == "text"
    assert env.payload["total_chunks"] >= 1


# ---------------------------------------------------------------------------
# search_files: mocked retrieval path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_files_aggregates_to_file_rows() -> None:
    from app.rag.retriever import RetrievedFile
    from app.rag.tools.search_files import search_files

    fake_results = [
        RetrievedFile(
            file_id="a",
            drive="d",
            filename="a.mp4",
            file_type="video",
            title="A",
            description=None,
            score=0.9,
            match_types=("transcript",),
            segments=(),
            mime_type="video/mp4",
        ),
        RetrievedFile(
            file_id="b",
            drive="d",
            filename="b.md",
            file_type="document",
            title="B",
            description=None,
            score=0.5,
            match_types=("text_content",),
            segments=(),
            mime_type="text/markdown",
        ),
    ]

    with patch(
        "app.rag.tools.search_files.retrieve_with_keywords",
        AsyncMock(return_value=fake_results),
    ), patch(
        "app.rag.tools.search_files._transcript_bearing_file_ids",
        return_value={"a"},
    ):
        ctx = ToolContext(drive="d")
        env = await search_files(context=ctx, query="q", top_k=10)

    rows = env.payload["files"]
    assert [r["file_id"] for r in rows] == ["a", "b"]
    # has_transcript now comes from the DB check, not match_types.
    assert rows[0]["has_transcript"] is True
    assert rows[1]["has_transcript"] is False
    assert ctx.tool_returned_file_ids == {"a", "b"}
    assert ctx.tool_calls == ["search_files"]


# ---------------------------------------------------------------------------
# get_related_files
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_file_detail_access_denied_returns_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.rag.tools.get_file_detail import get_file_detail

    async def _deny(file_ids, credential=None):
        return set()

    monkeypatch.setattr(
        "app.rag.tools.get_file_detail.ensure_access", _deny
    )
    env = await get_file_detail(context=ToolContext(), file_id="x")
    assert env.payload == {"file_id": "x", "error": "not_found"}


@pytest.mark.asyncio
async def test_get_file_detail_rejects_invalid_file_id() -> None:
    from app.rag.tools.get_file_detail import get_file_detail

    env = await get_file_detail(
        context=ToolContext(), file_id="../etc/passwd"
    )
    assert env.payload["error"] == "invalid file_id"


@pytest.mark.asyncio
async def test_get_file_detail_rejects_cross_drive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive-equality check rejects rows whose host-reported drive
    does not match ``context.drive`` even if the access filter
    erroneously allowed the ID."""
    from app.rag.tools.get_file_detail import get_file_detail

    basic = {
        "id": "F",
        "drive": "secret_drive",
        "filename": "x.md",
        "file_type": "document",
        "folder_path": "",
        "thumbnail_path": None,
        "updated_at": None,
    }

    with patch(
        "app.rag.tools.get_file_detail._fetch_basic_metadata",
        AsyncMock(return_value=basic),
    ), patch(
        "app.rag.tools.get_file_detail.hydrate_files",
        AsyncMock(return_value={}),
    ), patch(
        "app.rag.tools.get_file_detail._fetch_relations_summary",
        AsyncMock(return_value={}),
    ), patch(
        "app.rag.tools.get_file_detail._count_transcript_chunks",
        return_value=0,
    ):
        env = await get_file_detail(
            context=ToolContext(drive="public_drive"), file_id="F"
        )
    assert env.payload["error"] == "not_found"


@pytest.mark.asyncio
async def test_get_file_detail_drops_unknown_hydrate_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defence against Tier 3 leakage if the host's bulk-hydrate
    response ever grows new fields."""
    from app.rag.tools.get_file_detail import get_file_detail

    basic = {
        "id": "F",
        "drive": "d",
        "filename": "x.md",
        "file_type": "document",
        "folder_path": "",
        "thumbnail_path": None,
        "updated_at": None,
    }
    hydrated = {
        "F": {
            "title": "OK",
            "tags": ["a"],
            "mime_type": "text/markdown",
            "auto_tags_suggested": ["leak"],
            "detailed_summary": "should not leak",
        }
    }
    with patch(
        "app.rag.tools.get_file_detail._fetch_basic_metadata",
        AsyncMock(return_value=basic),
    ), patch(
        "app.rag.tools.get_file_detail.hydrate_files",
        AsyncMock(return_value=hydrated),
    ), patch(
        "app.rag.tools.get_file_detail._fetch_relations_summary",
        AsyncMock(return_value={}),
    ), patch(
        "app.rag.tools.get_file_detail._count_transcript_chunks",
        return_value=0,
    ):
        env = await get_file_detail(
            context=ToolContext(drive="d"), file_id="F"
        )
    payload = env.payload
    assert payload["title"] == "OK"
    assert payload["tags"] == ["a"]
    # New host fields must not propagate.
    assert "auto_tags_suggested" not in payload
    assert "detailed_summary" not in payload


@pytest.mark.asyncio
async def test_get_related_files_marks_direction() -> None:
    from app.rag.tools.get_related_files import get_related_files

    payload: list[dict[str, Any]] = [
        {"id": 1, "file_id_a": "F", "file_id_b": "X", "kind": "cite"},
        {"id": 2, "file_id_a": "Y", "file_id_b": "F", "kind": "see_also"},
    ]
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value=payload)
    mock_response.raise_for_status = MagicMock()
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.get = AsyncMock(return_value=mock_response)

    with patch(
        "app.rag.tools.get_related_files.httpx.AsyncClient",
        return_value=client,
    ):
        ctx = ToolContext()
        env = await get_related_files(context=ctx, file_id="F")

    relations = env.payload["relations"]
    directions = {r["file_id"]: r["direction"] for r in relations}
    assert directions == {"X": "outgoing", "Y": "incoming"}
    assert ctx.tool_returned_file_ids == {"X", "Y"}


@pytest.mark.asyncio
async def test_get_related_files_filters_cross_drive_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``file_relations`` can legitimately point at IDs in another
    drive; the wrapper must drop those before registering them as
    valid citations."""
    from app.rag.tools.get_related_files import get_related_files

    raw: list[dict[str, Any]] = [
        {"id": 1, "file_id_a": "F", "file_id_b": "ALLOWED", "kind": "cite"},
        {"id": 2, "file_id_a": "FORBIDDEN", "file_id_b": "F", "kind": "see"},
    ]

    async def _access(file_ids, credential=None):
        # Input access (file_id='F') goes through; output filtered.
        ids = set(file_ids)
        return ids & {"F", "ALLOWED"}

    monkeypatch.setattr(
        "app.rag.tools.get_related_files.ensure_access", _access
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value=raw)
    mock_response.raise_for_status = MagicMock()
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.get = AsyncMock(return_value=mock_response)

    with patch(
        "app.rag.tools.get_related_files.httpx.AsyncClient",
        return_value=client,
    ):
        ctx = ToolContext()
        env = await get_related_files(context=ctx, file_id="F")

    rel_ids = {r["file_id"] for r in env.payload["relations"]}
    assert rel_ids == {"ALLOWED"}
    assert "FORBIDDEN" not in ctx.tool_returned_file_ids


@pytest.mark.asyncio
async def test_get_related_files_rejects_bad_kind() -> None:
    from app.rag.tools.get_related_files import get_related_files

    env = await get_related_files(
        context=ToolContext(), file_id="F", kind="Invalid Kind!"
    )
    assert env.payload["error"] == "invalid kind"


@pytest.mark.asyncio
async def test_get_related_files_access_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.rag.tools.get_related_files import get_related_files

    async def _deny(file_ids, credential=None):
        return set()

    monkeypatch.setattr(
        "app.rag.tools.get_related_files.ensure_access", _deny
    )
    env = await get_related_files(context=ToolContext(), file_id="X")
    assert env.payload["error"] == "not_found"


@pytest.mark.asyncio
async def test_get_related_files_empty_on_http_error() -> None:
    import httpx as _httpx

    from app.rag.tools.get_related_files import get_related_files

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.get = AsyncMock(side_effect=_httpx.ConnectError("nope"))

    with patch(
        "app.rag.tools.get_related_files.httpx.AsyncClient",
        return_value=client,
    ):
        env = await get_related_files(context=ToolContext(), file_id="F")
    assert env.payload["relations"] == []
    assert env.warning is not None


# ---------------------------------------------------------------------------
# get_file_detail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_file_detail_404_returns_error() -> None:
    from app.rag.tools.get_file_detail import get_file_detail

    with patch(
        "app.rag.tools.get_file_detail._fetch_basic_metadata",
        AsyncMock(return_value=None),
    ), patch(
        "app.rag.tools.get_file_detail.hydrate_files",
        AsyncMock(return_value={}),
    ), patch(
        "app.rag.tools.get_file_detail._fetch_relations_summary",
        AsyncMock(return_value={}),
    ), patch(
        "app.rag.tools.get_file_detail._count_transcript_chunks",
        return_value=0,
    ):
        env = await get_file_detail(context=ToolContext(), file_id="F")
    assert env.payload["error"] == "not_found"


@pytest.mark.asyncio
async def test_get_file_detail_merges_sources() -> None:
    from app.rag.tools.get_file_detail import get_file_detail

    basic = {
        "id": "F",
        "drive": "d",
        "filename": "f.md",
        "file_type": "document",
        "folder_path": "docs",
        "thumbnail_path": None,
        "updated_at": None,
    }
    hydrated = {
        "F": {
            "title": "Friendly title",
            "tags": ["x", "y"],
            "mime_type": "text/markdown",
        }
    }
    relations_summary = {"cite": 2, "see_also": 1}

    with patch(
        "app.rag.tools.get_file_detail._fetch_basic_metadata",
        AsyncMock(return_value=basic),
    ), patch(
        "app.rag.tools.get_file_detail.hydrate_files",
        AsyncMock(return_value=hydrated),
    ), patch(
        "app.rag.tools.get_file_detail._fetch_relations_summary",
        AsyncMock(return_value=relations_summary),
    ), patch(
        "app.rag.tools.get_file_detail._count_transcript_chunks",
        return_value=3,
    ):
        ctx = ToolContext()
        env = await get_file_detail(context=ctx, file_id="F")
    payload = env.payload
    assert payload["title"] == "Friendly title"
    assert payload["tags"] == ["x", "y"]
    assert payload["mime"] == "text/markdown"
    assert payload["relations"] == relations_summary
    assert payload["chunk_count"] == 3
    assert payload["has_transcript"] is True
    # Tier 3 fields must NOT appear in the payload.
    assert "auto_tags" not in payload
    assert "detailed_summary" not in payload
    assert ctx.tool_returned_file_ids == {"F"}
