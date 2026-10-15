"""SPEC-ADDON-021: the backfill pass and the IDF built from stored term sets."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import pytest

from app import database, tfidf
from app.models import Embedding
from tests import idf_backfill_api as api
from tests.idf_doubles import ExplodingTokenizer, MarkerTokenizer
from tests.idf_term_db import counts, idx, no_children, wait_for  # noqa: F401

WORDS = "共通語 " * 10
RARE = "珍語 " * 10


def _run_in_thread(target):
    out: dict = {}

    def run():
        try:
            out["value"] = target()
        except BaseException as exc:  # noqa: BLE001
            out["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    return thread, out


def test_spec_addon_021_a_pass_writes_the_term_set_of_each_active_file(idx, caplog):
    idx.add_file("k1", transcript=WORDS)
    idx.add_file("k2", filename="題名.mp4")
    idx.add_file("k3", transcript="短い")
    idx.add_file("k4", transcript=WORDS, active=False)
    idx.set_state("k1", indexed=False, gen=4)

    with caplog.at_level(logging.INFO):
        api.run_pass()

    assert idx.terms("k1") == idx.term_set("k1") != set()
    assert idx.terms("k2") == idx.term_set("k2") != set()
    assert idx.terms("k3") == idx.term_set("k3")
    assert idx.state("k1") == (True, 4)
    assert idx.state("k2") == (True, 0)
    assert idx.state("k3") == (True, 0)
    assert idx.terms("k4") == set()
    assert idx.state("k4") == (False, 0)
    assert counts(caplog) == [
        {"written": 3, "skipped": 0, "failed": 0, "held": 0, "remaining": 0}
    ]


def test_spec_addon_021_a_rewritten_file_loses_the_terms_it_no_longer_has(idx):
    idx.add_file("k1", transcript=RARE)
    idx.put_terms("k1", {"旧語", "珍語"})

    api.run_pass()

    assert idx.terms("k1") == idx.term_set("k1")
    assert "旧語" not in idx.terms("k1")


def test_spec_addon_021_after_a_full_backfill_the_idf_equals_tokenizing_every_active_file(
    idx, monkeypatch,
):
    monkeypatch.setattr(tfidf, "_CORPUS_BATCH_FILES", 2)
    idx.add_file("a1", transcript=WORDS)
    idx.add_file("a2", transcript=WORDS)
    idx.add_file("a3", transcript=RARE)
    idx.add_file("a4", filename="題名.mp4")
    idx.add_file("a5", transcript="短い")
    idx.add_file("a6", transcript=RARE, active=False)
    idx.add_file("b1", drive="b", transcript=RARE)
    idx.add_file("b2", drive="b", transcript="別の話題 " * 10)

    api.run_pass()

    for drive in ("a", "b"):
        expected_idf, expected_n = idx.expected_idf(drive)
        idf, n_docs = tfidf._get_corpus_idf(drive)
        assert n_docs == expected_n
        assert idf == pytest.approx(expected_idf)
    assert idx.expected_idf("a")[1] == 4


def test_spec_addon_021_one_file_gives_every_term_an_idf_of_one(idx):
    idx.add_file("a1", transcript=WORDS)

    api.run_pass()

    idf, n_docs = tfidf._get_corpus_idf("a")
    assert n_docs == 1
    assert idf == pytest.approx({t: 1.0 for t in idx.term_set("a1")})


def test_spec_addon_021_a_cached_idf_is_dropped_when_a_pass_writes_its_drive(idx):
    idx.add_file("a1", transcript=WORDS)
    assert tfidf._get_corpus_idf("a") == ({}, 0)

    api.run_pass()

    idf, n_docs = tfidf._get_corpus_idf("a")
    assert (idf, n_docs) == (pytest.approx(idx.expected_idf("a")[0]), 1)


def test_spec_addon_021_a_build_in_flight_while_the_batch_is_written_does_not_cache_its_result(
    idx, monkeypatch,
):
    idx.add_file("a1", transcript=WORDS)
    real_build = tfidf._build_corpus_idf
    built = threading.Event()
    release = threading.Event()
    calls = []

    def build_then_wait(drive):
        calls.append(drive)
        result = real_build(drive)
        if len(calls) == 1:
            built.set()
            assert release.wait(timeout=10)
        return result

    monkeypatch.setattr(tfidf, "_build_corpus_idf", build_then_wait)
    owner, owner_out = _run_in_thread(lambda: tfidf._get_corpus_idf("a"))
    assert built.wait(timeout=10)

    api.run_pass()
    release.set()
    owner.join(timeout=10)

    assert owner_out["value"] == ({}, 0)
    idf, n_docs = tfidf._get_corpus_idf("a")
    assert n_docs == 1
    assert idf == pytest.approx(idx.expected_idf("a")[0])


def test_spec_addon_021_building_the_idf_tokenizes_nothing_and_takes_no_write_lock(
    idx, monkeypatch,
):
    idx.add_file("a1", transcript=WORDS)
    idx.add_file("a2", transcript=RARE)
    api.run_pass()
    expected = idx.expected_idf("a")

    def no_process(*_a, **_k):
        raise AssertionError("the build started a process")

    monkeypatch.setattr(tfidf, "_TOKENIZER_FACTORY", ExplodingTokenizer)
    monkeypatch.setattr(tfidf.multiprocessing, "get_context", no_process)
    with database._write_lock:
        builder, out = _run_in_thread(lambda: tfidf._build_corpus_idf("a"))
        builder.join(timeout=5)
        assert not builder.is_alive()

    assert "error" not in out, out.get("error")
    idf, n_docs = out["value"]
    assert n_docs == expected[1]
    assert idf == pytest.approx(expected[0])


def test_spec_addon_021_the_idf_skips_inactive_files_and_counts_a_reactivated_one_with_its_rows(
    idx,
):
    idx.add_file("a1", transcript=WORDS)
    idx.add_file("a2", transcript=RARE)
    api.run_pass()
    before = idx.terms("a2")

    idx.sql("UPDATE indexed_files SET active = 0 WHERE file_id = 'a2'")
    idf, n_docs = tfidf._build_corpus_idf("a")
    assert n_docs == 1
    assert set(idf) == idx.term_set("a1")

    idx.sql("UPDATE indexed_files SET active = 1 WHERE file_id = 'a2'")
    idf, n_docs = tfidf._build_corpus_idf("a")
    assert n_docs == 2
    assert idf == pytest.approx(idx.expected_idf("a")[0])
    assert idx.terms("a2") == before
    assert idx.state("a2") == (True, 0)


def test_spec_addon_021_rows_of_a_file_flagged_false_still_count_until_replaced(idx):
    idx.add_file("a1", transcript=WORDS)
    idx.add_file("a2", transcript=RARE)
    api.run_pass()
    idx.set_state("a2", indexed=False, gen=1)

    idf, n_docs = tfidf._build_corpus_idf("a")

    assert n_docs == 2
    assert idf == pytest.approx(idx.expected_idf("a")[0])


def test_spec_addon_021_a_file_without_rows_counts_toward_nothing(idx):
    idx.add_file("a1", transcript=WORDS)
    api.run_pass()
    idx.add_file("a2", transcript=RARE)

    idf, n_docs = tfidf._build_corpus_idf("a")

    assert n_docs == 1
    assert set(idf) == idx.term_set("a1")


def test_spec_addon_021_the_build_raises_when_its_aggregate_raises(idx):
    idx.add_file("a1", transcript=WORDS)
    api.run_pass()
    idx.sql("ALTER TABLE idf_doc_terms RENAME TO idf_doc_terms_gone")

    with pytest.raises(Exception):
        tfidf._build_corpus_idf("a")


def test_spec_addon_021_with_nothing_flagged_a_pass_spawns_nothing_and_logs_no_info_line(
    idx, monkeypatch, caplog,
):
    idx.add_file("a1", transcript=WORDS, indexed=True, gen=2)
    idx.add_file("a2", transcript=WORDS, active=False)
    spawned = []
    monkeypatch.setattr(
        tfidf.multiprocessing, "get_context", lambda *a, **k: spawned.append(a)
    )

    with caplog.at_level(logging.DEBUG):
        api.run_pass()

    assert spawned == []
    assert counts(caplog) == []
    assert not [r for r in caplog.records if r.levelno >= logging.INFO]
    assert idx.state("a2") == (False, 0)


@pytest.mark.parametrize("marker", ["CRASHWORD", "BOOMWORD", "SLOWWORD"])
def test_spec_addon_021_a_failing_batch_is_retried_one_file_at_a_time(
    idx, monkeypatch, caplog, marker,
):
    api.set_timings(monkeypatch, child_timeout=2.0)
    monkeypatch.setattr(tfidf, "_TOKENIZER_FACTORY", MarkerTokenizer)
    idx.add_file("k1", transcript=WORDS)
    idx.add_file("k2", transcript=f"{marker} " + RARE)
    idx.add_file("k3", transcript=RARE)
    idx.put_terms("k2", {"旧語"})
    idx.set_state("k2", indexed=False, gen=3)

    with caplog.at_level(logging.INFO):
        api.run_pass()

    assert idx.state("k1") == (True, 0)
    assert idx.state("k3") == (True, 0)
    assert idx.terms("k1") == idx.term_set("k1")
    assert idx.terms("k3") == idx.term_set("k3")
    assert idx.state("k2") == (False, 3)
    assert idx.terms("k2") == {"旧語"}
    assert any(
        r.levelno == logging.WARNING and "k2" in r.getMessage() for r in caplog.records
    )
    [line] = counts(caplog)
    assert (line["written"], line["skipped"], line["failed"], line["held"]) == (2, 0, 1, 0)
    assert no_children()


def test_spec_addon_021_a_file_that_failed_on_its_own_is_held_until_it_changes(
    idx, monkeypatch, caplog,
):
    monkeypatch.setattr(tfidf, "_TOKENIZER_FACTORY", MarkerTokenizer)
    idx.add_file("k1", transcript="BOOMWORD " + RARE)
    api.run_pass()
    idx.add_file("k2", transcript=WORDS)

    with caplog.at_level(logging.INFO):
        api.run_pass()
    assert counts(caplog)[-1] == {
        "written": 1, "skipped": 0, "failed": 0, "held": 1, "remaining": 0,
    }
    assert idx.state("k1") == (False, 0)

    idx.sql("UPDATE transcript_chunks SET text = :t WHERE file_id = 'k1'", t=RARE)
    idx.set_state("k1", indexed=False, gen=1)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        api.run_pass()

    assert counts(caplog) == [
        {"written": 1, "skipped": 0, "failed": 0, "held": 0, "remaining": 0}
    ]
    assert idx.state("k1") == (True, 1)
    assert idx.terms("k1") == idx.term_set("k1")


def test_spec_addon_021_a_write_that_raises_rolls_back_that_file_and_is_not_held(
    idx, caplog,
):
    idx.add_file("k1", transcript=WORDS)
    idx.add_file("k2", transcript=RARE)
    idx.add_file("k3", transcript=WORDS)
    idx.put_terms("k2", {"旧語"})
    idx.sql("CREATE TABLE fail_writes (on_ INTEGER)")
    idx.sql("INSERT INTO fail_writes VALUES (1)")
    idx.sql(
        "CREATE TRIGGER fail_k2 BEFORE INSERT ON idf_doc_terms "
        "WHEN NEW.file_id = 'k2' AND (SELECT on_ FROM fail_writes) = 1 "
        "BEGIN SELECT RAISE(ABORT, 'write failed'); END"
    )

    with caplog.at_level(logging.INFO):
        api.run_pass()

    assert idx.state("k1") == (True, 0)
    assert idx.state("k3") == (True, 0)
    assert idx.state("k2") == (False, 0)
    assert idx.terms("k2") == {"旧語"}
    assert counts(caplog)[-1]["failed"] == 1
    assert counts(caplog)[-1]["written"] == 2

    idx.sql("UPDATE fail_writes SET on_ = 0")
    api.run_pass()

    assert idx.state("k2") == (True, 0)
    assert idx.terms("k2") == idx.term_set("k2")


def _change_inputs(idx):
    idx.sql("UPDATE transcript_chunks SET text = :t WHERE file_id = 'k1'", t=RARE)
    idx.sql(
        "UPDATE indexed_files SET idf_terms_indexed = 0, "
        "idf_terms_gen = idf_terms_gen + 1 WHERE file_id = 'k1'"
    )


def _purge(idx):
    idx.sql("DELETE FROM transcript_chunks WHERE file_id = 'k1'")
    idx.sql("DELETE FROM indexed_files WHERE file_id = 'k1'")


def _deactivate(idx):
    idx.sql("UPDATE indexed_files SET active = 0 WHERE file_id = 'k1'")


@pytest.mark.parametrize("change", [_change_inputs, _purge, _deactivate])
def test_spec_addon_021_a_file_that_changes_between_read_and_write_is_skipped(
    idx, monkeypatch, caplog, tmp_path, change,
):
    monkeypatch.setattr(tfidf, "_TOKENIZER_FACTORY", MarkerTokenizer)
    idx.add_file("k1", transcript="GATEWORD " + WORDS)
    gate = Path(tmp_path)

    with caplog.at_level(logging.INFO):
        runner, out = _run_in_thread(api.run_pass)
        wait_for(lambda: (gate / "entered").exists() or not runner.is_alive())
        assert "error" not in out, out.get("error")
        assert (gate / "entered").exists()
        change(idx)
        (gate / "release").touch()
        runner.join(timeout=30)

    assert "error" not in out, out.get("error")
    assert counts(caplog)[-1]["skipped"] == 1
    assert counts(caplog)[-1]["written"] == 0
    assert idx.terms("k1") == set()
    if change is _purge:
        return
    assert idx.state("k1")[0] is False

    if change is _change_inputs:
        api.run_pass()
        assert idx.state("k1") == (True, 1)
        assert idx.terms("k1") == idx.term_set("k1")
        assert "GATEWORD" not in idx.terms("k1")


def test_spec_addon_021_the_backfill_leaves_embeddings_and_keyword_flags_alone(idx):
    idx.add_file("k1", transcript=WORDS)
    idx.sql("UPDATE indexed_files SET tfidf_keywords_indexed = 1 WHERE file_id = 'k1'")
    with idx.Session() as s:
        s.add(Embedding(
            id="e_k1", file_id="k1", embedding_type="tfidf_keywords",
            vector_table="vec_text", content_preview="x",
        ))
        s.commit()
    idx.sql("INSERT INTO vec_text VALUES ('e_k1', x'00')")
    before = (
        idx.rows("SELECT * FROM embeddings"),
        idx.rows("SELECT * FROM vec_text"),
    )

    api.run_pass()

    assert idx.state("k1")[0] is True
    assert idx.rows(
        "SELECT tfidf_keywords_indexed FROM indexed_files WHERE file_id = 'k1'"
    ) == [(1,)]
    assert (
        idx.rows("SELECT * FROM embeddings"),
        idx.rows("SELECT * FROM vec_text"),
    ) == before
