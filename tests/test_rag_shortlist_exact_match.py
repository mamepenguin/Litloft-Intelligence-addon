"""SPEC-ADDON-035: rare exact-match files join Ask's hierarchical shortlist.

The lookup runs against a real search DB built by ``init_search_db`` so the FTS5
tables, tokenizers and column layout are the production ones. The coarse
retrieval, the access filter, the clue generator and the retriever are doubles.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event
from sqlalchemy.exc import OperationalError

import app.database as database
from app.config import (
    HierarchicalRagConfig,
    LLMConfig,
    PersonalHistoryConfig,
    RagConfig,
    load_settings,
)
from app.rag.coarse_retriever import ShortlistResult
from app.rag.query_decomposer import DecomposedQuery, TimeRange
from app.rag.query_transform import StructuredQuery
from app.rag.retriever import RetrievedFile
from app.rag.service import _answer_question
from app.rag.service import _stream_answer as stream_answer
from app.search import MatchInfo, SegmentGroup
from tests.llm_helpers import bind_llm
from tests.search_endpoint_db import (  # noqa: F401
    DRIVE,
    OTHER,
    Seed,
    patch_everywhere,
    search_index,
    seed,
    statements,
)

TOKEN = "ヒーター"
OTHER_TOKEN = "ストーブ"
_UNSET = object()


def _retrieved(file_id: str) -> RetrievedFile:
    match = MatchInfo(
        match_type="transcript", text="snippet", score=0.9,
        timestamp_start=0.0, timestamp_end=10.0,
    )
    return RetrievedFile(
        file_id=file_id, drive=DRIVE, filename=f"{file_id}.mp4", file_type="video",
        title=f"Title {file_id}", description="", score=0.9,
        match_types=("transcript",),
        segments=(SegmentGroup(time_range=(0.0, 10.0), matches=(match,)),),
    )


def _llm():
    from unittest.mock import MagicMock

    client = MagicMock()
    client.enabled = True

    async def _stream(*_a, **_k):
        yield '{"answer": "ans", "citations": []}'

    client.generate_stream = _stream
    client.generate_json = AsyncMock(return_value=None)
    return client


@dataclass
class Outcome:
    events: list = field(default_factory=list)
    retrieve: AsyncMock | None = None
    access: AsyncMock | None = None
    coarse: AsyncMock | None = None
    clues: AsyncMock | None = None
    summary_ids: list = field(default_factory=list)
    response: object = None

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]

    def shortlist(self):
        return next(e for e in self.events if e.kind == "shortlist").data

    def scopes(self) -> list:
        return [c.kwargs.get("file_id_scope") for c in self.retrieve.await_args_list]

    def access_arg(self) -> list[str]:
        return list(self.access.await_args.args[0])


@pytest.fixture()
def ask(monkeypatch, make_settings, search_index):  # noqa: F811
    async def _run(
        *,
        keywords: str,
        coarse: tuple[str, ...] = ("c1", "c2"),
        cap=_UNSET,
        deny: tuple[str, ...] = (),
        drive_file_count: int = 100,
        top_score: float = 0.9,
        hierarchical: bool = True,
        history: list[str] | None = None,
        streaming: bool = True,
    ) -> Outcome:
        hier_kwargs = {} if cap is _UNSET else {"exact_match_max_files": cap}
        hier = HierarchicalRagConfig(
            enabled=hierarchical, coarse_top_k=5, coarse_score_threshold=0.3,
            min_drive_files_for_shortlist=10, **hier_kwargs,
        )
        rag = RagConfig(
            hierarchical=hier,
            personal_history=PersonalHistoryConfig(enabled=history is not None),
        )
        s = make_settings(
            features=type(make_settings().features)(rag=True),
            llm=LLMConfig(provider="openai_compatible", model="m"),
            rag=rag,
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        out = Outcome()
        scores = tuple(round(top_score - 0.01 * i, 4) for i in range(len(coarse)))
        shortlist = ShortlistResult(
            file_ids=coarse, scores=scores,
            top_score=top_score if coarse else 0.0,
            drive_file_count=drive_file_count,
        )
        out.coarse = AsyncMock(return_value=shortlist)
        monkeypatch.setattr("app.rag.service.coarse_retrieve", out.coarse)
        monkeypatch.setattr(
            "app.rag.service.transform_query_structured",
            AsyncMock(return_value=StructuredQuery(
                required=(), semantic=("kw",), raw_keywords=keywords,
            )),
        )
        out.access = AsyncMock(side_effect=lambda ids, _t: set(ids) - set(deny))
        monkeypatch.setattr("app.rag.service._filter_file_ids_via_internal_api", out.access)
        out.retrieve = AsyncMock(return_value=[_retrieved("r1"), _retrieved("r2")])
        monkeypatch.setattr("app.rag.service.retrieve_with_keywords", out.retrieve)
        monkeypatch.setattr(
            "app.rag.service.assemble_contexts", lambda cands, cfg, **_kw: [],
        )

        def _summaries(ids):
            out.summary_ids.append(list(ids))
            return {i: f"summary of {i}" for i in ids}

        monkeypatch.setattr("app.rag.service.fetch_long_summaries", _summaries)
        out.clues = AsyncMock(
            side_effect=lambda **kw: [kw["fallback_keywords"], "second clue"],
        )
        monkeypatch.setattr("app.rag.service.generate_clues", out.clues)
        bind_llm(monkeypatch, _llm)

        viewer = None
        if history is not None:
            viewer = "x" * 16
            monkeypatch.setattr(
                "app.rag.service.decompose_query",
                AsyncMock(return_value=DecomposedQuery(
                    raw_query="q",
                    time_range=TimeRange(
                        label="last_week",
                        after=datetime(2026, 4, 22, tzinfo=UTC),
                        before=datetime(2026, 4, 29, tzinfo=UTC),
                    ),
                    personal_scope="viewed", file_type_hint="video",
                    semantic_query="s",
                )),
            )
            monkeypatch.setattr(
                "app.rag.service.fetch_viewer_history",
                AsyncMock(return_value=history),
            )

        if streaming:
            async for ev in stream_answer(
                query="q", credential="t", drive=DRIVE, viewer_id=viewer,
            ):
                out.events.append(ev)
        else:
            out.response = await _answer_question(
                query="q", credential="t", drive=DRIVE, viewer_id=viewer,
            )
        return out

    return _run


def mention(seed: Seed, file_id: str, body: str = TOKEN, *, via: str = "transcript",
            drive: str = DRIVE, active: bool = True) -> None:
    seed.file(file_id, drive=drive, active=active)
    if via == "transcript":
        seed.transcript(file_id, 0, f"これは {body} の話")
    elif via == "text":
        seed.text_chunk(file_id, 0, f"これは {body} の話")
    elif via == "transcript_word":
        seed.index.sql(
            "INSERT INTO fts_transcripts_word(file_id, chunk_index, text) "
            "VALUES (:f, '0', :t)", f=file_id, t=f"this is {body} talk",
        )
    elif via == "text_word":
        seed.index.sql(
            "INSERT INTO fts_text_content_word(file_id, chunk_index, page, text) "
            "VALUES (:f, '0', '', :t)", f=file_id, t=f"this is {body} talk",
        )
    else:
        raise AssertionError(via)


def assert_adds_nothing(out: Outcome, coarse=("c1", "c2")) -> None:
    data = out.shortlist()
    assert data["exact_match_file_ids"] == []
    assert data["file_ids"] == list(coarse)
    assert out.access_arg() == list(coarse)
    assert out.scopes() and all(sc == list(coarse) for sc in out.scopes())


# --- normal flow -------------------------------------------------------------


async def test_spec_addon_035_a_rare_token_adds_its_files_after_the_coarse_files(ask, seed):
    mention(seed, "y_file")
    mention(seed, "x_file", via="text")
    out = await ask(keywords=f"{TOKEN} お勧め")

    assert out.access_arg() == ["c1", "c2", "x_file", "y_file"]
    assert out.access.await_count == 1
    assert out.scopes() and all(sc == ["c1", "c2", "x_file", "y_file"] for sc in out.scopes())
    data = out.shortlist()
    assert data["file_ids"] == ["c1", "c2", "x_file", "y_file"]
    assert data["exact_match_file_ids"] == ["x_file", "y_file"]
    assert data["top_score"] == pytest.approx(0.9)
    assert data["drive_file_count"] == 100


async def test_spec_addon_035_every_clue_search_gets_the_same_widened_scope(ask, seed):
    mention(seed, "x_file")
    out = await ask(keywords=TOKEN)

    clue_calls = [c for c in out.retrieve.await_args_list]
    assert {c.kwargs["keywords"] for c in clue_calls} == {TOKEN, "second clue"}
    assert all(c.kwargs["file_id_scope"] == ["c1", "c2", "x_file"] for c in clue_calls)


async def test_spec_addon_035_files_are_ordered_by_token_then_file_id_and_added_once(ask, seed):
    mention(seed, "z2")
    mention(seed, "z1")
    seed.transcript("z1", 1, f"{OTHER_TOKEN} もある")
    mention(seed, "a9", OTHER_TOKEN)
    out = await ask(keywords=f"{TOKEN} {OTHER_TOKEN}")

    assert out.shortlist()["exact_match_file_ids"] == ["z1", "z2", "a9"]


async def test_spec_addon_035_a_file_already_in_the_coarse_shortlist_is_not_repeated(ask, seed):
    mention(seed, "c1")
    mention(seed, "x_file")
    out = await ask(keywords=TOKEN)

    data = out.shortlist()
    assert data["file_ids"] == ["c1", "c2", "x_file"]
    assert data["exact_match_file_ids"] == ["x_file"]
    assert out.access_arg() == ["c1", "c2", "x_file"]


@pytest.mark.parametrize(
    ("via", "token"),
    [
        ("transcript", TOKEN),
        ("text", TOKEN),
        ("transcript_word", "Heater"),
        ("text_word", "Heater"),
    ],
)
async def test_spec_addon_035_each_of_the_four_tables_is_searched(ask, seed, via, token):
    mention(seed, "x_file", token, via=via)
    out = await ask(keywords=token)

    assert out.shortlist()["exact_match_file_ids"] == ["x_file"]


async def test_spec_addon_035_a_two_character_word_matches_through_the_word_tables(ask, seed):
    mention(seed, "x_file", "AI", via="text_word")
    out = await ask(keywords="AI")

    assert out.shortlist()["exact_match_file_ids"] == ["x_file"]


async def test_spec_addon_035_a_two_character_cjk_run_matches_nothing(ask, seed):
    mention(seed, "x_file", "電気毛布")
    out = await ask(keywords="電気")

    assert_adds_nothing(out)


# --- token cleaning ----------------------------------------------------------


async def test_spec_addon_035_double_quotes_are_removed_before_the_lookup(ask, seed):
    mention(seed, "x_file")
    out = await ask(keywords='"ヒーター"')
    assert out.shortlist()["exact_match_file_ids"] == ["x_file"]

    out = await ask(keywords='ヒー"ター')
    assert out.shortlist()["exact_match_file_ids"] == ["x_file"]


async def test_spec_addon_035_tokens_are_split_on_unicode_whitespace(ask, seed):
    mention(seed, "x_file")
    mention(seed, "y_file", OTHER_TOKEN)
    out = await ask(keywords=f"{TOKEN}　{OTHER_TOKEN}")

    assert out.shortlist()["exact_match_file_ids"] == ["x_file", "y_file"]


async def test_spec_addon_035_tokens_under_two_characters_after_cleaning_are_skipped(
    ask, seed, statements,  # noqa: F811
):
    mention(seed, "x_file", "A", via="transcript_word")
    mention(seed, "y_file", "b", via="text_word")
    statements.reset()
    out = await ask(keywords='A "b" "" " " ""　')

    assert_adds_nothing(out)
    assert not [s for s in statements.statements if "fts_" in s]


async def test_spec_addon_035_a_duplicate_token_is_looked_up_once(ask, seed, statements):  # noqa: F811
    mention(seed, "x_file")
    statements.reset()
    out = await ask(keywords=f'{TOKEN} "{TOKEN}" {TOKEN}')

    assert out.shortlist()["exact_match_file_ids"] == ["x_file"]
    lookups = [s for s in statements.statements if "fts_" in s]
    assert 1 <= len(lookups) <= 4


# --- the cap and what counts -------------------------------------------------


@pytest.mark.parametrize(("count", "added"), [(1, True), (3, True), (4, False)])
async def test_spec_addon_035_a_token_adds_files_only_up_to_the_cap(ask, seed, count, added):
    for i in range(count):
        mention(seed, f"f{i}")
    out = await ask(keywords=TOKEN, cap=3)

    if added:
        assert out.shortlist()["exact_match_file_ids"] == [f"f{i}" for i in range(count)]
    else:
        assert_adds_nothing(out)


async def test_spec_addon_035_a_token_with_no_file_adds_nothing(ask, seed):
    out = await ask(keywords=TOKEN)

    assert_adds_nothing(out)


@pytest.mark.parametrize(("count", "added"), [(10, True), (11, False)])
async def test_spec_addon_035_the_default_cap_is_ten(ask, seed, count, added):
    for i in range(count):
        mention(seed, f"f{i:02d}")
    out = await ask(keywords=TOKEN)

    if added:
        assert out.shortlist()["exact_match_file_ids"] == [f"f{i:02d}" for i in range(count)]
    else:
        assert_adds_nothing(out)


@pytest.mark.parametrize(("cap", "added"), [(3, False), (4, True)])
async def test_spec_addon_035_the_count_is_taken_across_all_four_tables(ask, seed, cap, added):
    mention(seed, "f0", "Heater")
    mention(seed, "f1", "Heater", via="text")
    mention(seed, "f2", "Heater", via="transcript_word")
    mention(seed, "f3", "Heater", via="text_word")
    out = await ask(keywords="Heater", cap=cap)

    if added:
        assert out.shortlist()["exact_match_file_ids"] == ["f0", "f1", "f2", "f3"]
    else:
        assert_adds_nothing(out)


async def test_spec_addon_035_one_file_matching_in_several_tables_counts_once(ask, seed):
    mention(seed, "f0")
    seed.text_chunk("f0", 0, f"{TOKEN} again")
    seed.transcript("f0", 1, f"{TOKEN} more")
    mention(seed, "f1", via="text")
    out = await ask(keywords=TOKEN, cap=2)

    assert out.shortlist()["exact_match_file_ids"] == ["f0", "f1"]


async def test_spec_addon_035_an_over_cap_token_does_not_block_a_rare_one(ask, seed):
    for i in range(4):
        mention(seed, f"common{i}")
    mention(seed, "rare", OTHER_TOKEN)
    out = await ask(keywords=f"{TOKEN} {OTHER_TOKEN}", cap=3)

    assert out.shortlist()["exact_match_file_ids"] == ["rare"]


async def test_spec_addon_035_zero_cap_makes_no_lookup_and_changes_nothing(
    ask, seed, statements,  # noqa: F811
):
    mention(seed, "x_file")
    statements.reset()
    out = await ask(keywords=TOKEN, cap=0)

    assert not [s for s in statements.statements if "fts_" in s]
    assert_adds_nothing(out)
    assert out.summary_ids == [["c1", "c2"]]


# --- the drive boundary, active files and columns ----------------------------


async def test_spec_addon_035_other_drives_and_inactive_files_are_neither_added_nor_counted(
    ask, seed,
):
    mention(seed, "x_file")
    mention(seed, "y_file")
    mention(seed, "elsewhere_file", drive=OTHER)
    mention(seed, "gone_file", active=False)
    out = await ask(keywords=TOKEN, cap=2)

    assert out.shortlist()["exact_match_file_ids"] == ["x_file", "y_file"]
    assert "elsewhere_file" not in out.access_arg()
    assert "gone_file" not in out.access_arg()


async def test_spec_addon_035_only_the_text_column_is_matched(ask, seed):
    mention(seed, "has_text", "150 円")
    seed.file("id_150_file")
    seed.text_chunk("id_150_file", 150, "関係のない本文")
    seed.file("page_file")
    seed.text_chunk("page_file", 1, "関係のない本文", page="150")
    seed.file("chunk_file")
    seed.transcript("chunk_file", 150, "関係のない本文")
    out = await ask(keywords="150")

    assert out.shortlist()["exact_match_file_ids"] == ["has_text"]


async def test_spec_addon_035_filename_and_title_tables_are_not_searched(ask, seed):
    seed.file("title_only")
    seed.index.sql(
        "INSERT INTO fts_files(file_id, filename, title, description, tags_text) "
        "VALUES ('title_only', 'a.md', :t, '', '')", t=TOKEN,
    )
    seed.index.sql(
        "INSERT INTO fts_files_word(file_id, filename, title, description, tags_text) "
        "VALUES ('title_only', 'a.md', 'Heater', '', '')",
    )
    out = await ask(keywords=f"{TOKEN} Heater")

    assert_adds_nothing(out)


# --- personal history --------------------------------------------------------


async def test_spec_addon_035_with_a_personal_history_scope_only_files_inside_it_are_added(
    ask, seed,
):
    mention(seed, "inside")
    mention(seed, "outside")
    out = await ask(
        keywords=TOKEN, coarse=("c1", "c2", "c3"), history=["c2", "inside"],
    )

    assert out.access_arg() == ["c2", "inside"]
    data = out.shortlist()
    assert data["file_ids"] == ["c2", "inside"]
    assert data["exact_match_file_ids"] == ["inside"]
    assert all(sc == ["c2", "inside"] for sc in out.scopes())


# --- access control and the clue generator -----------------------------------


async def test_spec_addon_035_a_file_the_caller_cannot_access_appears_in_no_event(ask, seed):
    mention(seed, "open_file")
    mention(seed, "locked_file")
    out = await ask(keywords=TOKEN, deny=("locked_file",))

    assert out.shortlist()["exact_match_file_ids"] == ["open_file"]
    assert out.scopes() and all(sc == ["c1", "c2", "open_file"] for sc in out.scopes())
    for e in out.events:
        assert "locked_file" not in json.dumps(e.data, default=str)


async def test_spec_addon_035_the_clue_generator_sees_only_the_surviving_coarse_summaries(
    ask, seed,
):
    mention(seed, "x_file")
    out = await ask(keywords=TOKEN, coarse=("c1", "c2", "c3"), deny=("c2",))

    assert out.summary_ids == [["c1", "c3"]]
    assert out.clues.await_args.kwargs["summaries"] == ["summary of c1", "summary of c3"]
    data = out.shortlist()
    assert data["file_ids"] == ["c1", "c3", "x_file"]
    assert data["exact_match_file_ids"] == ["x_file"]
    assert data["top_score"] == pytest.approx(0.9)


async def test_spec_addon_035_top_score_is_the_best_surviving_coarse_score(ask, seed):
    mention(seed, "x_file")
    out = await ask(keywords=TOKEN, coarse=("c1", "c2"), deny=("c1",), top_score=0.9)

    data = out.shortlist()
    assert data["exact_match_file_ids"] == ["x_file"]
    assert data["top_score"] == pytest.approx(0.89)


async def test_spec_addon_035_with_no_surviving_coarse_file_the_access_bypass_runs_unscoped(
    ask, seed,
):
    mention(seed, "x_file")
    out = await ask(keywords=TOKEN, deny=("c1", "c2"))

    assert out.access.await_count == 1
    assert out.access_arg() == ["c1", "c2", "x_file"]
    assert "shortlist" not in out.kinds()
    assert "clues" not in out.kinds()
    assert out.scopes() == [None]


async def test_spec_addon_035_that_bypass_keeps_the_personal_history_scope(ask, seed):
    mention(seed, "inside")
    out = await ask(
        keywords=TOKEN, coarse=("c1", "c2"), history=["c1", "inside"], deny=("c1",),
    )

    assert "shortlist" not in out.kinds()
    assert out.scopes() == [["c1", "inside"]]


# --- bypasses that never reach the lookup ------------------------------------


@pytest.mark.parametrize(
    "case",
    ["small_drive", "empty_shortlist", "low_confidence", "disabled"],
)
async def test_spec_addon_035_bypass_branches_make_no_lookup_and_search_unscoped(
    ask, seed, statements, case,  # noqa: F811
):
    mention(seed, "x_file")
    statements.reset()
    kwargs = {
        "small_drive": {"drive_file_count": 3},
        "empty_shortlist": {"coarse": ()},
        "low_confidence": {"top_score": 0.1},
        "disabled": {"hierarchical": False},
    }[case]
    out = await ask(keywords=TOKEN, **kwargs)

    assert not [s for s in statements.statements if "fts_" in s]
    assert "shortlist" not in out.kinds()
    assert out.scopes() == [None]


# --- failures ----------------------------------------------------------------


def _fail_statements(*, table: str, token: str):
    def _boom(_conn, _cursor, statement, parameters, _context, _many):
        haystack = f"{statement} {parameters}"
        if table in haystack and token in haystack and "MATCH" in statement.upper():
            raise OperationalError(statement, parameters, Exception("database is locked"))

    engine = database._search_engine
    event.listen(engine, "before_cursor_execute", _boom)
    return lambda: event.remove(engine, "before_cursor_execute", _boom)


async def test_spec_addon_035_one_failing_table_voids_that_token_only_and_logs_a_warning(
    ask, seed, caplog,
):
    mention(seed, "x_file")
    mention(seed, "y_file", OTHER_TOKEN)
    undo = _fail_statements(table="fts_text_content_word", token=TOKEN)
    try:
        with caplog.at_level(logging.WARNING):
            out = await ask(keywords=f"{TOKEN} {OTHER_TOKEN}")
    finally:
        undo()

    assert out.shortlist()["exact_match_file_ids"] == ["y_file"]
    assert "error" not in out.kinds()
    assert [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and "fts_text_content_word" in r.getMessage() and TOKEN in r.getMessage()
    ]


async def test_spec_addon_035_a_lookup_that_fails_entirely_leaves_the_coarse_shortlist(
    ask, seed, monkeypatch, caplog,
):
    mention(seed, "x_file")

    def _down():
        raise RuntimeError("Search database not initialized")

    patch_everywhere(monkeypatch, "get_search_engine", _down)
    patch_everywhere(monkeypatch, "get_search_db_read", _down)
    with caplog.at_level(logging.WARNING):
        out = await ask(keywords=TOKEN)

    assert_adds_nothing(out)
    assert "error" not in out.kinds()
    assert [r for r in caplog.records if r.levelno == logging.WARNING]


# --- the event loop and logging ----------------------------------------------


async def test_spec_addon_035_the_lookup_never_runs_on_the_event_loop(ask, seed, statements):  # noqa: F811
    mention(seed, "x_file")
    statements.reset()
    out = await ask(keywords=TOKEN)

    assert out.shortlist()["exact_match_file_ids"] == ["x_file"]
    assert [s for s in statements.statements if "fts_" in s]
    assert statements.on_loop == []


async def test_spec_addon_035_a_debug_line_names_each_token_and_the_over_cap_ones(
    ask, seed, caplog,
):
    for i in range(4):
        mention(seed, f"common{i}")
    mention(seed, "rare", OTHER_TOKEN)
    with caplog.at_level(logging.DEBUG):
        await ask(keywords=f"{TOKEN} {OTHER_TOKEN}", cap=3)

    assert [
        r for r in caplog.records
        if r.levelno == logging.DEBUG
        and TOKEN in r.getMessage() and OTHER_TOKEN in r.getMessage()
    ]


# --- the non-streaming path --------------------------------------------------


async def test_spec_addon_035_the_non_streaming_answer_scopes_to_the_widened_shortlist(ask, seed):
    mention(seed, "x_file")
    out = await ask(keywords=TOKEN, streaming=False)

    assert out.scopes() and all(sc == ["c1", "c2", "x_file"] for sc in out.scopes())
    assert out.access_arg() == ["c1", "c2", "x_file"]


# --- data shape and configuration --------------------------------------------


def test_spec_addon_035_shortlist_result_defaults_to_no_exact_match_files():
    base = ShortlistResult(file_ids=("a",), scores=(0.5,), top_score=0.5, drive_file_count=9)
    assert base.exact_match_file_ids == ()
    widened = ShortlistResult(
        file_ids=("a",), scores=(0.5,), top_score=0.5, drive_file_count=9,
        exact_match_file_ids=("b",),
    )
    assert widened.exact_match_file_ids == ("b",)
    assert widened.file_ids == ("a",)


def _load(tmp_path, monkeypatch, hierarchical_yaml: str):
    config_file = tmp_path / "search-config.yml"
    config_file.write_text(f"rag:\n  hierarchical:\n{hierarchical_yaml}")
    monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
    monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config_file))
    return load_settings().rag.hierarchical


def test_spec_addon_035_the_default_cap_is_ten_in_the_config():
    assert HierarchicalRagConfig().exact_match_max_files == 10


@pytest.mark.parametrize(("yaml_value", "expected"), [("0", 0), ("7", 7), ("10", 10)])
def test_spec_addon_035_a_valid_cap_is_read_from_yaml(tmp_path, monkeypatch, caplog, yaml_value, expected):
    with caplog.at_level(logging.WARNING):
        h = _load(tmp_path, monkeypatch, f"    exact_match_max_files: {yaml_value}\n")

    assert h.exact_match_max_files == expected
    assert not [r for r in caplog.records if "exact_match_max_files" in r.getMessage()]


@pytest.mark.parametrize("yaml_line", ["    enabled: true\n", "    exact_match_max_files: null\n"])
def test_spec_addon_035_an_absent_or_null_cap_is_ten_without_a_log(
    tmp_path, monkeypatch, caplog, yaml_line,
):
    with caplog.at_level(logging.WARNING):
        h = _load(tmp_path, monkeypatch, yaml_line)

    assert h.exact_match_max_files == 10
    assert not [r for r in caplog.records if "exact_match_max_files" in r.getMessage()]


@pytest.mark.parametrize("yaml_value", ["-1", "2.5", '"5"', "true", "false", "[3]"])
def test_spec_addon_035_an_invalid_cap_falls_back_to_ten_with_one_warning(
    tmp_path, monkeypatch, caplog, yaml_value,
):
    with caplog.at_level(logging.WARNING):
        h = _load(tmp_path, monkeypatch, f"    exact_match_max_files: {yaml_value}\n")

    assert h.exact_match_max_files == 10
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "exact_match_max_files" in r.getMessage()
    ]
    assert len(warnings) == 1
