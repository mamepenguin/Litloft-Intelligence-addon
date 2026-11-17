"""SPEC-ADDON-128: a chunk whose realignment does not succeed is left exactly as it was;
only chunks whose refined text was applied count as refined (invariants I2-I7)."""

from __future__ import annotations

import json
import logging
import re

import pytest
from sqlalchemy import event

from app.models import KeptOriginal
from app.workers import aligner
from tests.refine_align_rig import (
    FILE,
    chunk_rows,
    install_whisperx,
    joined,
    make_db,
    overlaps,
    run_refine,
    seed_file,
    snapshot,
    squash,
    word_rows_in,
)

# Japanese chunks, 4 s each. The first is the spec's reproduction case.
JA_CHUNKS = [
    (1, 0, "にじゅうにんきた", 0.0, 4.0),
    (2, 1, "あしたははれです", 4.0, 8.0),
    (3, 2, "もうかえります", 8.0, 12.0),
]
REFINED_1 = "20人来た。"
REFINED_2 = "明日は晴れです"
REFINED_3 = "もう帰ります"
ALL_REFINED = {1: REFINED_1, 2: REFINED_2, 3: REFINED_3}
OLD_TEXT = {cid: body for cid, _i, body, _s, _e in JA_CHUNKS}
SPAN = {cid: (s, e) for cid, _i, _b, s, e in JA_CHUNKS}


@pytest.fixture(autouse=True)
def _clean_aligner():
    aligner.release_all()
    yield
    aligner.release_all()


def _setup(tmp_path, *, language="ja", chunks=JA_CHUNKS, **seed):
    _engine, Session = make_db(tmp_path)
    seed_file(Session, chunks, language=language, **seed)
    return _engine, Session


# --- Failure modes: every alignment outcome other than success leaves the file alone -------

FAILURES = {
    "no_model_for_language": dict(language="en", chunks=[
        (1, 0, "we deployed it on cooper netties", 0.0, 4.0),
        (2, 1, "the cluster had three nodes", 4.0, 8.0),
        (3, 2, "and it worked fine", 8.0, 12.0),
    ], whisper=dict(models=("ja",))),
    "audio_unreadable": dict(language="ja", chunks=JA_CHUNKS, whisper=dict(audio_ok=False)),
    "model_load_raises": dict(language="ja", chunks=JA_CHUNKS,
                              whisper=dict(load_error=RuntimeError("out of memory"))),
    "alignment_returns_nothing": dict(language="ja", chunks=JA_CHUNKS, whisper=dict(
        fail_texts=ALL_REFINED.values(), fail_mode="empty")),
    "alignment_raises": dict(language="ja", chunks=JA_CHUNKS, whisper=dict(
        fail_texts=ALL_REFINED.values(), fail_mode="raises")),
}
EN_REFINED = {1: "We deployed it on Kubernetes.", 2: "The cluster had 3 nodes.",
              3: "And it worked fine."}


def _failing_case(tmp_path, monkeypatch, name, **seed):
    case = FAILURES[name]
    _e, Session = _setup(tmp_path, language=case["language"], chunks=case["chunks"], **seed)
    fake = install_whisperx(monkeypatch, **case["whisper"])
    refined = EN_REFINED if case["language"] == "en" else ALL_REFINED
    return Session, fake, refined


@pytest.mark.asyncio
@pytest.mark.parametrize("name", FAILURES)
async def test_spec_addon_128_I3_I4_run_that_applies_nothing_leaves_every_table_unchanged(
    tmp_path, monkeypatch, name
):
    Session, _fake, refined = _failing_case(tmp_path, monkeypatch, name)
    before = snapshot(Session)

    run = await run_refine(Session, monkeypatch, refined)

    assert snapshot(Session) == before
    assert run.rechunk_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("name", FAILURES)
async def test_spec_addon_128_I2_unapplied_chunks_keep_their_pre_run_text(
    tmp_path, monkeypatch, name
):
    Session, _fake, refined = _failing_case(tmp_path, monkeypatch, name)
    expected = squash("".join(body for _c, _i, body, _s, _e in FAILURES[name]["chunks"]))

    await run_refine(Session, monkeypatch, refined)

    assert joined(Session) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("name", FAILURES)
async def test_spec_addon_128_failed_chunks_count_as_skipped_and_none_as_refined(
    tmp_path, monkeypatch, name
):
    Session, _fake, refined = _failing_case(tmp_path, monkeypatch, name)

    run = await run_refine(Session, monkeypatch, refined)

    assert run.completed["refined_count"] == 0
    assert run.completed["skipped_count"] == 3
    assert all(r["refined_at"] is None and r["model"] is None for r in chunk_rows(Session))


@pytest.mark.asyncio
async def test_spec_addon_128_run_that_applies_nothing_logs_one_warning_with_file_and_count(
    tmp_path, monkeypatch, caplog
):
    Session, _fake, refined = _failing_case(tmp_path, monkeypatch, "audio_unreadable")

    with caplog.at_level(logging.WARNING):
        await run_refine(Session, monkeypatch, refined)

    summary = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and FILE in r.getMessage()
        and re.search(r"(?<!\d)3(?!\d)", r.getMessage())
    ]
    assert summary, [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_spec_addon_128_I3_pre_refined_chunk_whose_realignment_fails_keeps_its_marks(
    tmp_path, monkeypatch
):
    _e, Session = _setup(tmp_path, refined=(2,))
    install_whisperx(monkeypatch, audio_ok=False)
    before = snapshot(Session)
    assert before["chunks"][1][5] is not None

    await run_refine(Session, monkeypatch, {2: REFINED_2}, model="newer-model")

    assert snapshot(Session) == before
    assert chunk_rows(Session)[1]["model"] == "old-model"


# --- Positive controls and the reproduction ------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["ja", "jpn"])
async def test_spec_addon_128_reproduction_refined_text_survives_when_words_realign(
    tmp_path, monkeypatch, language
):
    """The spec's case: 「にじゅうにんきた」 refined to 「20人来た。」. ``ja`` is the control that
    shows the harness aligns; ``jpn`` is the production code (SPEC-ADDON-127 + 128)."""
    _e, Session = _setup(tmp_path, language=language, chunks=JA_CHUNKS[:1])
    install_whisperx(monkeypatch)

    run = await run_refine(Session, monkeypatch, {1: REFINED_1}, model="m1")

    rows = chunk_rows(Session)
    assert [r["text"] for r in rows] == [REFINED_1]
    assert rows[0]["refined_at"] is not None and rows[0]["model"] == "m1"
    spoken = "".join(t for t, _s, _e in word_rows_in(Session, 0.0, 99.0))
    assert squash(spoken).rstrip("。") == "20人来た"
    assert run.completed["refined_count"] == 1


@pytest.mark.asyncio
async def test_spec_addon_128_failure_with_jpn_words_does_not_resurrect_the_old_text(
    tmp_path, monkeypatch
):
    """The defect as shipped: ``jpn`` file whose model lookup fails (here: the only model
    available is not Japanese) must read as the old text, not a mix."""
    _e, Session = _setup(tmp_path, language="jpn", chunks=JA_CHUNKS[:1])
    install_whisperx(monkeypatch, models=("en",))
    before = snapshot(Session)

    await run_refine(Session, monkeypatch, {1: REFINED_1})

    assert snapshot(Session) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["ja", "jpn"])
async def test_spec_addon_128_I2_every_applied_chunk_text_is_kept_and_marked(
    tmp_path, monkeypatch, language
):
    _e, Session = _setup(tmp_path, language=language)
    install_whisperx(monkeypatch)

    run = await run_refine(Session, monkeypatch, ALL_REFINED)

    assert joined(Session) == squash(REFINED_1 + REFINED_2 + REFINED_3)
    assert run.rechunk_calls
    assert run.completed["refined_count"] == 3 and run.completed["skipped_count"] == 0
    assert any(r["refined_at"] is not None for r in chunk_rows(Session))


LONG_CHUNKS = [
    (1, 0, "にじゅうにんきた", 0.0, 40.0),
    (2, 1, "あしたははれです", 40.0, 80.0),
    (3, 2, "もうかえります", 80.0, 120.0),
    (4, 3, "おやすみなさい", 120.0, 160.0),
]
LONG_SPAN = {cid: (s, e) for cid, _i, _b, s, e in LONG_CHUNKS}


# --- Mixed outcome: some chunks apply, some do not -----------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("fail_mode", ["empty", "raises"])
async def test_spec_addon_128_I2_I3_mixed_run_joins_refined_and_pre_run_text(
    tmp_path, monkeypatch, fail_mode
):
    _e, Session = _setup(tmp_path, language="jpn")
    install_whisperx(monkeypatch, fail_texts=[REFINED_2], fail_mode=fail_mode)
    words_2_before = word_rows_in(Session, *SPAN[2])

    run = await run_refine(Session, monkeypatch, ALL_REFINED)

    assert joined(Session) == squash(REFINED_1 + OLD_TEXT[2] + REFINED_3)
    assert word_rows_in(Session, *SPAN[2]) == words_2_before
    assert run.completed["refined_count"] == 2
    assert run.completed["skipped_count"] == 1


@pytest.mark.asyncio
async def test_spec_addon_128_I6_rebuilt_chunks_carry_marks_only_where_a_refined_chunk_was(
    tmp_path, monkeypatch
):
    """Chunk 1 was refined by an earlier run, chunk 2 is applied now, chunk 3 is not in the
    run, chunk 4 fails: a rebuilt chunk is marked only if it overlaps chunk 1 or chunk 2."""
    _e, Session = _setup(tmp_path, chunks=LONG_CHUNKS, refined=(1,))
    install_whisperx(monkeypatch, fail_texts=["おやすみ"], fail_mode="empty")
    refined_spans = [LONG_SPAN[1], LONG_SPAN[2]]

    await run_refine(Session, monkeypatch, {2: REFINED_2, 4: "おやすみ"}, model="m2")

    rows = chunk_rows(Session)
    for r in rows:
        touches = any(overlaps(r["start"], r["end"], a, b) for a, b in refined_spans)
        if r["refined_at"] is not None:
            assert touches, f"chunk {r['text']!r} {r['start']}-{r['end']} marked without overlap"
    tail = [r for r in rows if r["start"] >= LONG_SPAN[3][0]]
    assert tail, [(r["start"], r["end"]) for r in rows]
    assert all(r["refined_at"] is None for r in tail)


@pytest.mark.asyncio
async def test_spec_addon_128_I6_a_failed_neighbour_is_not_marked_by_an_applied_one(
    tmp_path, monkeypatch
):
    _e, Session = _setup(tmp_path, chunks=LONG_CHUNKS)
    install_whisperx(monkeypatch, fail_texts=[REFINED_2, REFINED_3, "おやすみ"],
                     fail_mode="empty")

    await run_refine(Session, monkeypatch, {**ALL_REFINED, 4: "おやすみ"}, model="m1")

    rows = chunk_rows(Session)
    assert any(r["refined_at"] is not None for r in rows)
    for r in rows:
        if r["refined_at"] is not None:
            assert overlaps(r["start"], r["end"], *LONG_SPAN[1]), r
    assert any(r["start"] >= LONG_SPAN[3][0] for r in rows)


# --- Kept original -------------------------------------------------------------------------

def _kept(Session):
    with Session() as s:
        rows = s.query(KeptOriginal).filter_by(file_id=FILE).all()
        return [(r.kind, r.trigger, r.chunks_json, r.words_json) for r in rows]


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


@pytest.mark.asyncio
async def test_spec_addon_128_I5_kept_original_is_the_pre_run_transcript_even_if_first_chunk_failed(
    tmp_path, monkeypatch
):
    _e, Session = _setup(tmp_path, language="jpn")
    install_whisperx(monkeypatch, fail_texts=[REFINED_1], fail_mode="empty")
    n_words = len(word_rows_in(Session, 0.0, 99.0))

    await run_refine(Session, monkeypatch, ALL_REFINED)

    kept = _kept(Session)
    assert len(kept) == 1
    kind, trigger, chunks_json, words_json = kept[0]
    assert (kind, trigger) == ("transcript", "refine")
    texts = set(_strings(json.loads(chunks_json)))
    assert set(OLD_TEXT.values()) <= texts
    assert not texts & {REFINED_1, REFINED_2, REFINED_3}
    assert len(json.loads(words_json)) == n_words


@pytest.mark.asyncio
async def test_spec_addon_128_I5_an_existing_kept_original_is_not_replaced(
    tmp_path, monkeypatch
):
    engine, Session = _setup(tmp_path, language="jpn")
    with Session() as s:
        s.add(KeptOriginal(file_id=FILE, kind="transcript", trigger="correction",
                           chunks_json='["sentinel"]', words_json="[]"))
        s.commit()
    install_whisperx(monkeypatch)

    await run_refine(Session, monkeypatch, ALL_REFINED)

    assert _kept(Session) == [("transcript", "correction", '["sentinel"]', "[]")]


@pytest.mark.asyncio
async def test_spec_addon_128_I5_kept_original_is_written_in_the_transaction_of_the_first_applied_chunk(
    tmp_path, monkeypatch
):
    engine, Session = _setup(tmp_path, language="jpn")
    install_whisperx(monkeypatch, fail_texts=[REFINED_1], fail_mode="empty")
    log: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _stmt(conn, cursor, statement, params, context, executemany):
        head = statement.lstrip().split("(")[0].strip().upper()
        if "KEPT_ORIGINALS" in head and head.startswith("INSERT"):
            log.append("kept")
        elif head.startswith(("UPDATE TRANSCRIPT_CHUNKS", "INSERT INTO TRANSCRIPT_CHUNKS",
                              "DELETE FROM TRANSCRIPT_CHUNKS")):
            log.append("chunk")

    @event.listens_for(engine, "commit")
    def _commit(conn):
        log.append("commit")

    await run_refine(Session, monkeypatch, ALL_REFINED)

    assert "kept" in log and "chunk" in log
    first, second = sorted((log.index("kept"), log.index("chunk")))
    assert "commit" not in log[first:second + 1], log


@pytest.mark.asyncio
async def test_spec_addon_128_I5_applied_run_writes_the_kept_original(tmp_path, monkeypatch):
    _e, Session = _setup(tmp_path, language="ja")
    install_whisperx(monkeypatch)

    await run_refine(Session, monkeypatch, ALL_REFINED)

    assert [(k, t) for k, t, _c, _w in _kept(Session)] == [("transcript", "refine")]


# --- .loft files (no word rows) ------------------------------------------------------------

@pytest.mark.asyncio
async def test_spec_addon_128_I7_chunks_without_word_rows_receive_refined_text_and_marks(
    tmp_path, monkeypatch
):
    _e, Session = _setup(tmp_path, language="ja", words=False)
    install_whisperx(monkeypatch, models=(), audio_ok=False)

    run = await run_refine(Session, monkeypatch, ALL_REFINED, model="loft-model")

    rows = chunk_rows(Session)
    assert [r["text"] for r in rows] == [REFINED_1, REFINED_2, REFINED_3]
    assert all(r["refined_at"] is not None and r["model"] == "loft-model" for r in rows)
    assert run.completed["refined_count"] == 3 and run.completed["skipped_count"] == 0
    assert [(k, t) for k, t, _c, _w in _kept(Session)] == [("transcript", "refine")]
