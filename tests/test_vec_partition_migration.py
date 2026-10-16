"""SPEC-ADDON-022: the vector tables are partitioned by embedding type, and an
existing database is migrated to that schema once, atomically, at startup."""

from __future__ import annotations

import collections
import logging
import re
import shutil
import subprocess
import sys
import textwrap

import pytest

from tests.vec_partition_db import (  # noqa: F401
    MIGRATION_LOG,
    VEC_TABLES,
    random_vector,
    run_partition_migration,
    search_index,
    start,
)

# (file_id, mime, [(embedding_id, embedding_type, vec table)])
FILES = [
    ("img1", "image/jpeg", [("ct_img1", "clip_thumbnail", "vec_clip")]),
    ("vid1", "video/mp4", [
        ("cs_vid1_0", "clip", "vec_clip"),
        ("cs_vid1_1", "clip", "vec_clip"),
        ("ct_vid1", "clip_thumbnail", "vec_clip"),
        ("wh_vid1_0", "whisper", "vec_text"),
        ("wh_vid1_1", "whisper", "vec_text"),
        ("tk_vid1", "tfidf_keywords", "vec_text"),
        ("meta_vid1", "metadata", "vec_text"),
    ]),
    ("doc1", "text/plain", [
        ("txt_doc1_0", "text_content", "vec_text"),
        ("meta_doc1", "metadata", "vec_text"),
    ]),
]
ORPHANS = [("orph_clip", "vec_clip"), ("orph_text", "vec_text")]

START_LINE = re.compile(r"vec partition migration: start tables=(.*) estimated_bytes=(\d+)")
DONE_LINE = re.compile(r"vec partition migration: done (.*) elapsed=")
NO_DISK = re.compile(
    r"vec partition migration: not enough free disk: free=(\d+) required=(\d+)"
)


def _seed(index, *, tables=VEC_TABLES, orphans=True, keyed=False) -> dict:
    """Files, embeddings and vec rows as a pre-upgrade index holds them.

    Returns {table: {embedding_id: (embedding_type, vector bytes)}} for the rows
    that have an embeddings row.
    """
    expected: dict[str, dict[str, tuple[str, bytes]]] = {t: {} for t in VEC_TABLES}
    seed = 0
    for file_id, mime, embeddings in FILES:
        index.add_file(file_id, mime_type=mime)
        for emb_id, etype, table in embeddings:
            if table not in tables:
                continue
            seed += 1
            vec = random_vector(index.dim(table), seed)
            index.add_embedding(emb_id, file_id, etype, table)
            index.put_vector(table, emb_id, vec, partition=etype if keyed else "omit")
            expected[table][emb_id] = (etype, vec.tobytes())
    if orphans:
        for emb_id, table in ORPHANS:
            if table in tables:
                seed += 1
                index.put_vector(
                    table, emb_id, random_vector(index.dim(table), seed),
                    partition="clip" if keyed else "omit",
                )
    return expected


def _estimated_bytes(index, expected) -> int:
    return sum(len(rows) * index.dim(t) * 4 for t, rows in expected.items() if rows)


def _messages(caplog, pattern) -> list[re.Match]:
    return [m for r in caplog.records if (m := pattern.search(r.getMessage()))]


def _migration_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if MIGRATION_LOG in r.getMessage()]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_spec_addon_022_a_fresh_database_has_both_tables_partitioned_by_type(search_index):
    for table in VEC_TABLES:
        assert search_index.is_partitioned(table), search_index.create_sql(table)
        assert re.search(r"vector\s+float\[\d+\]", search_index.create_sql(table))
    assert not any(name.endswith("_migrate") for name in search_index.table_names())


def test_spec_addon_022_a_model_migration_rebuilds_the_table_partitioned_and_copies_nothing(
    search_index, caplog,
):
    """A table dropped by the model migration comes back partitioned and empty;
    the partition migration then has nothing to do and says nothing."""
    search_index.downgrade("vec_text")
    _seed(search_index, tables=("vec_text",), orphans=False)
    search_index.sql(
        "UPDATE index_meta SET value = 'some-other/model' WHERE key = 'text_embedding_model'"
    )

    caplog.set_level(logging.INFO)
    start()

    assert search_index.is_partitioned("vec_text")
    assert search_index.vectors("vec_text") == {}
    assert _migration_records(caplog) == []


# ---------------------------------------------------------------------------
# I1: every backed row is copied once, byte-identical, keyed by its type
# ---------------------------------------------------------------------------


def test_spec_addon_022_migration_keys_every_row_by_its_embedding_type(search_index):
    for table in VEC_TABLES:
        search_index.downgrade(table)
    dims = {t: search_index.dim(t) for t in VEC_TABLES}
    expected = _seed(search_index)

    run_partition_migration()

    for table in VEC_TABLES:
        assert search_index.is_partitioned(table)
        assert search_index.dim(table) == dims[table]
        assert search_index.keyed(table) == expected[table]
    assert not any(name.endswith("_migrate") for name in search_index.table_names())


def test_spec_addon_022_after_migration_a_type_constrained_scan_finds_its_rows(search_index):
    for table in VEC_TABLES:
        search_index.downgrade(table)
    expected = _seed(search_index)

    run_partition_migration()

    probe = random_vector(search_index.dim("vec_clip"), 999)
    found = search_index.knn("vec_clip", probe, 50, "clip")
    assert sorted(found) == sorted(
        i for i, (t, _) in expected["vec_clip"].items() if t == "clip"
    )


# ---------------------------------------------------------------------------
# I2: once only, one table at a time, all or nothing
# ---------------------------------------------------------------------------


def test_spec_addon_022_a_start_on_partitioned_tables_changes_nothing_and_logs_nothing(
    search_index, caplog,
):
    for table in VEC_TABLES:
        search_index.downgrade(table)
    expected = _seed(search_index)
    run_partition_migration()

    caplog.clear()
    caplog.set_level(logging.INFO)
    start()

    for table in VEC_TABLES:
        assert search_index.keyed(table) == expected[table]
    assert _migration_records(caplog) == []


def test_spec_addon_022_only_the_unpartitioned_table_is_migrated(search_index, caplog):
    """vec_clip is already partitioned (a dimension migration rebuilt it): its rows,
    orphan included, stay exactly as they are."""
    search_index.downgrade("vec_text")
    _seed(search_index, tables=("vec_clip",), keyed=True)
    clip_before = search_index.keyed("vec_clip")
    assert "orph_clip" in clip_before
    text_expected = _seed_text_only(search_index)

    caplog.set_level(logging.INFO)
    run_partition_migration()

    assert search_index.keyed("vec_clip") == clip_before
    assert search_index.keyed("vec_text") == text_expected
    (start_line,) = _messages(caplog, START_LINE)
    assert "vec_text" in start_line.group(1)
    assert "vec_clip" not in start_line.group(1)


def _seed_text_only(index) -> dict:
    expected = {}
    for n, etype in enumerate(("whisper", "tfidf_keywords", "text_content")):
        file_id = f"t{n}"
        index.add_file(file_id)
        vec = random_vector(index.dim("vec_text"), 500 + n)
        index.add_embedding(f"e_{file_id}", file_id, etype, "vec_text")
        index.put_vector("vec_text", f"e_{file_id}", vec)
        expected[f"e_{file_id}"] = (etype, vec.tobytes())
    return expected


def test_spec_addon_022_a_failure_after_a_drop_leaves_both_tables_as_they_were(search_index):
    """An embedding type that vec0 cannot take as a partition value makes the copy
    back into the new vec_text fail, after the old vec_text was dropped."""
    for table in VEC_TABLES:
        search_index.downgrade(table)
    expected = _seed(search_index)
    old_sql = {t: search_index.create_sql(t) for t in VEC_TABLES}
    old_vectors = {t: search_index.vectors(t) for t in VEC_TABLES}
    search_index.sql("UPDATE embeddings SET embedding_type = x'00' WHERE id = 'wh_vid1_1'")

    with pytest.raises(Exception) as excinfo:
        start()

    assert "key type mismatch" in str(excinfo.value)
    for table in VEC_TABLES:
        assert search_index.create_sql(table) == old_sql[table]
        assert search_index.vectors(table) == old_vectors[table]
    assert not any(name.endswith("_migrate") for name in search_index.table_names())
    assert search_index.wal_bytes() == 0

    search_index.sql("UPDATE embeddings SET embedding_type = 'whisper' WHERE id = 'wh_vid1_1'")
    start()

    for table in VEC_TABLES:
        assert search_index.keyed(table) == expected[table]


def test_spec_addon_022_a_process_killed_mid_migration_leaves_both_tables_as_they_were(
    search_index,
):
    """The process dies as it creates the new vec_text, after the old one was
    dropped. The next start finds the pre-migration state and migrates it."""
    for table in VEC_TABLES:
        search_index.downgrade(table)
    expected = _seed(search_index)
    old_sql = {t: search_index.create_sql(t) for t in VEC_TABLES}
    old_vectors = {t: search_index.vectors(t) for t in VEC_TABLES}
    search_index.engine.dispose()

    script = textwrap.dedent(f"""
        import sys
        from pathlib import Path
        sys.path.insert(0, "/app")
        import tests.conftest
        import app.database as database
        from tests.vec_partition_db import Authorizer
        path = Path({str(search_index.path)!r})
        database._resolve_search_db_path = lambda: path
        auth = Authorizer()
        auth.deny_create_vtable = "vec_text"
        auth.exit_code = 37
        auth.install()
        database.init_search_db()
    """)
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd="/app", capture_output=True, text=True,
        timeout=300,
    )

    assert proc.returncode == 37, proc.stderr[-2000:]
    for table in VEC_TABLES:
        assert search_index.create_sql(table) == old_sql[table]
        assert search_index.vectors(table) == old_vectors[table]
    assert not any(name.endswith("_migrate") for name in search_index.table_names())

    start()

    for table in VEC_TABLES:
        assert search_index.keyed(table) == expected[table]


# ---------------------------------------------------------------------------
# Free disk
# ---------------------------------------------------------------------------


_Usage = collections.namedtuple("_Usage", "total used free")


def test_spec_addon_022_less_free_disk_than_four_times_the_vectors_refuses_and_writes_nothing(
    search_index, monkeypatch, caplog,
):
    for table in VEC_TABLES:
        search_index.downgrade(table)
    expected = _seed(search_index, orphans=False)
    free = 4 * _estimated_bytes(search_index, expected) - 1
    monkeypatch.setattr(shutil, "disk_usage", lambda path: _Usage(10**12, 0, free))
    old_sql = {t: search_index.create_sql(t) for t in VEC_TABLES}
    old_vectors = {t: search_index.vectors(t) for t in VEC_TABLES}

    caplog.set_level(logging.INFO)
    with pytest.raises(Exception):
        start()

    refusals = [
        r for r in caplog.records
        if r.levelno == logging.ERROR and NO_DISK.search(r.getMessage())
    ]
    assert len(refusals) == 1
    assert int(NO_DISK.search(refusals[0].getMessage()).group(1)) == free
    assert _messages(caplog, START_LINE) == []
    for table in VEC_TABLES:
        assert search_index.create_sql(table) == old_sql[table]
        assert search_index.vectors(table) == old_vectors[table]


def test_spec_addon_022_exactly_four_times_the_vectors_free_is_enough(
    search_index, monkeypatch, caplog,
):
    for table in VEC_TABLES:
        search_index.downgrade(table)
    expected = _seed(search_index, orphans=False)
    estimate = _estimated_bytes(search_index, expected)
    monkeypatch.setattr(
        shutil, "disk_usage", lambda path: _Usage(10**12, 0, 4 * estimate),
    )

    caplog.set_level(logging.INFO)
    start()

    for table in VEC_TABLES:
        assert search_index.keyed(table) == expected[table]
    (start_line,) = _messages(caplog, START_LINE)
    assert int(start_line.group(2)) == estimate


# ---------------------------------------------------------------------------
# I8: one line at the start, one at the end with the counts, and no WAL left
# ---------------------------------------------------------------------------


def test_spec_addon_022_migration_logs_one_start_and_one_done_line_with_counts(
    search_index, caplog,
):
    for table in VEC_TABLES:
        search_index.downgrade(table)
    expected = _seed(search_index)

    caplog.set_level(logging.INFO)
    run_partition_migration()

    starts = _messages(caplog, START_LINE)
    dones = _messages(caplog, DONE_LINE)
    assert len(starts) == 1 and len(dones) == 1
    assert "vec_clip" in starts[0].group(1) and "vec_text" in starts[0].group(1)
    done = dones[0].group(1)
    for table in VEC_TABLES:
        m = re.search(rf"{table} copied=(\d+) orphans_dropped=(\d+)", done)
        assert m, done
        assert int(m.group(1)) == len(expected[table])
        assert int(m.group(2)) == 1
    for record in _migration_records(caplog):
        assert record.levelno == logging.INFO


def test_spec_addon_022_no_wal_is_left_after_a_committed_migration(search_index):
    for table in VEC_TABLES:
        search_index.downgrade(table)
    _seed(search_index)
    search_index.engine.dispose()

    run_partition_migration()

    assert search_index.wal_bytes() == 0
