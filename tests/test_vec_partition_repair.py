"""SPEC-ADDON-022 I9: a vec row stored without its partition value (by addon code
older than the partitioned schema) is re-keyed from its embeddings row, or deleted
when it has none, and no vector is ever lost on the way."""

from __future__ import annotations

import logging

import pytest

from tests.vec_partition_db import (  # noqa: F401
    random_vector,
    search_index,
)


def _repair():
    from app.indexer import repair_unkeyed_vec_rows

    return repair_unkeyed_vec_rows()


def _seed_unkeyed(index, n: int, *, table: str = "vec_text", etype: str = "whisper",
                  prefix: str = "u") -> dict[str, tuple[str, bytes]]:
    dim = index.dim(table)
    index.add_file(f"{prefix}_file")
    expected = {}
    for i in range(n):
        emb_id = f"{prefix}{i:05d}"
        vec = random_vector(dim, 10_000 + i)
        index.add_embedding(emb_id, f"{prefix}_file", etype, table)
        index.put_vector(table, emb_id, vec)
        expected[emb_id] = (etype, vec.tobytes())
    return expected


def _info_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.INFO]


def test_spec_addon_022_unkeyed_rows_are_rekeyed_or_dropped(search_index, caplog):
    text_dim = search_index.dim("vec_text")
    clip_dim = search_index.dim("vec_clip")
    search_index.add_file("vid1")
    expected_text: dict[str, tuple[str, bytes]] = {}
    for n, etype in enumerate(
        ("whisper", "whisper", "tfidf_keywords", "text_content", "metadata")
    ):
        emb_id = f"t{n}"
        vec = random_vector(text_dim, n)
        search_index.add_embedding(emb_id, "vid1", etype, "vec_text")
        search_index.put_vector("vec_text", emb_id, vec)
        expected_text[emb_id] = (etype, vec.tobytes())
    for n in range(2):
        search_index.put_vector("vec_text", f"orphan{n}", random_vector(text_dim, 50 + n))
    keyed_vec = random_vector(clip_dim, 99)
    search_index.add_embedding("c_keyed", "vid1", "clip_thumbnail", "vec_clip")
    search_index.put_vector("vec_clip", "c_keyed", keyed_vec, partition="clip_thumbnail")

    caplog.set_level(logging.INFO)
    _repair()

    assert search_index.keyed("vec_text") == expected_text
    assert search_index.keyed("vec_clip") == {
        "c_keyed": ("clip_thumbnail", keyed_vec.tobytes())
    }
    probe = random_vector(text_dim, 2)
    assert search_index.knn("vec_text", probe, 1, "tfidf_keywords") == ["t2"]
    infos = _info_records(caplog)
    assert len(infos) == 1
    assert "5" in infos[0].getMessage() and "2" in infos[0].getMessage()


def test_spec_addon_022_repair_with_nothing_to_repair_changes_nothing_and_logs_nothing(
    search_index, caplog,
):
    dim = search_index.dim("vec_text")
    search_index.add_file("vid1")
    vec = random_vector(dim, 1)
    search_index.add_embedding("t0", "vid1", "whisper", "vec_text")
    search_index.put_vector("vec_text", "t0", vec, partition="whisper")

    caplog.set_level(logging.INFO)
    _repair()

    assert search_index.keyed("vec_text") == {"t0": ("whisper", vec.tobytes())}
    assert _info_records(caplog) == []


def test_spec_addon_022_repair_covers_more_rows_than_one_transaction(search_index):
    expected = _seed_unkeyed(search_index, 1203)

    _repair()

    assert search_index.keyed("vec_text") == expected


def test_spec_addon_022_a_repair_that_fails_loses_no_vector_and_the_next_one_finishes(
    search_index,
):
    """One embedding type vec0 cannot take as a partition value makes its re-insert
    fail. Whatever transaction it lands in, every vector is still there afterwards."""
    expected = _seed_unkeyed(search_index, 1203)
    bad = "u00700"
    search_index.sql("UPDATE embeddings SET embedding_type = x'00' WHERE id = :i", i=bad)

    try:
        _repair()
    except Exception:  # noqa: BLE001  raising is allowed; losing a vector is not
        pass

    after = search_index.keyed("vec_text")
    assert set(after) == set(expected)
    assert {i: v for i, (_, v) in after.items()} == {i: v for i, (_, v) in expected.items()}
    assert after[bad][0] is None
    assert all(t in (None, "whisper") for t, _ in after.values())

    search_index.sql("UPDATE embeddings SET embedding_type = 'whisper' WHERE id = :i", i=bad)
    _repair()

    assert search_index.keyed("vec_text") == expected


@pytest.mark.parametrize("table, etype", [("vec_clip", "clip"), ("vec_text", "metadata")])
def test_spec_addon_022_a_repaired_row_is_found_by_a_scan_of_its_type(
    search_index, table, etype,
):
    expected = _seed_unkeyed(search_index, 3, table=table, etype=etype)

    _repair()

    probe = random_vector(search_index.dim(table), 10_001)
    assert search_index.knn(table, probe, 10, etype)[0] == "u00001"
    assert sorted(search_index.knn(table, probe, 10, etype)) == sorted(expected)
