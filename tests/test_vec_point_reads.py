"""Reading stored vectors one id at a time: the helper agrees with an ``IN``
read on a real sqlite-vec database, and no source file reads a vec table's
vectors by ``IN`` or by joining on ``embedding_id`` again.

The source detector matches the written form of the read: a statement that
names a vec table (or one of the ``{table}``-style placeholders) and carries
``embedding_id IN``, or joins ``embeddings`` on ``embedding_id``, outside a
``DELETE``, an ``INSERT ... SELECT`` migration, a ``MATCH`` query or
``app/indexer.py``'s orphan scan. A different way of writing the same read
can pass it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from sqlalchemy import event, text

from tests.vec_partition_db import (  # noqa: F401
    random_vector,
    search_index,
)

APP_DIR = Path(__file__).resolve().parents[1] / "app"

_VEC_TABLE = re.compile(r"\bvec_(?:text|clip|clip_change)\b")
_VEC_PLACEHOLDER = re.compile(r"\{(?:table|SCENE_TABLE|CHANGE_TABLE)\}")
_IN_READ = re.compile(r"\bembedding_id\s+IN\b", re.IGNORECASE)


def _read_vectors():
    from app.vec_partition import read_vectors

    return read_vectors


# ---------------------------------------------------------------------------
# The helper, against a real vec0 table
# ---------------------------------------------------------------------------

PARTITIONS = ("one", "two")
_ROWS = 1205  # more than one sqlite-vec chunk (1024)


def _seed(index, table: str, n: int = _ROWS) -> list[str]:
    dim = index.dim(table)
    ids: list[str] = []
    rows = []
    for i in range(n):
        embedding_id = f"{table}_{i:05d}"
        ids.append(embedding_id)
        rows.append({
            "i": embedding_id,
            "t": PARTITIONS[i % len(PARTITIONS)],
            "v": random_vector(dim, i).tobytes(),
        })
    with index.engine.begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO {table}(embedding_id, embedding_type, vector) "
                "VALUES (:i, :t, :v)"
            ),
            rows,
        )
    return ids


def _in_read(conn, table: str, embedding_ids: list[str]) -> dict[str, bytes]:
    placeholders = ", ".join(f":p{i}" for i in range(len(embedding_ids)))
    rows = conn.execute(
        text(
            f"SELECT embedding_id, vector FROM {table} "
            f"WHERE embedding_id IN ({placeholders})"
        ),
        {f"p{i}": value for i, value in enumerate(embedding_ids)},
    ).fetchall()
    return {row[0]: bytes(row[1]) for row in rows}


@pytest.mark.parametrize("table", ["vec_text", "vec_clip", "vec_clip_change"])
def test_the_helper_returns_exactly_the_in_read_for_every_table(search_index, table):
    read_vectors = _read_vectors()
    ids = _seed(search_index, table)
    missing = ["missing_0", "missing_1"]
    duplicates = ids[:5]
    wanted = [*ids, *missing, *duplicates]

    with search_index.engine.connect() as conn:
        expected = _in_read(conn, table, wanted)
        got = read_vectors(conn, table, wanted)

    assert got == expected
    assert set(got) == set(ids)
    assert len(got) == len(ids)


def test_the_helper_never_issues_an_in_against_a_vec_table(search_index):
    read_vectors = _read_vectors()
    ids = _seed(search_index, "vec_text")
    engine = search_index.engine
    seen: list[str] = []

    def _before(_conn, _cursor, statement, _params, _context, _many):
        seen.append(statement)

    event.listen(engine, "before_cursor_execute", _before)
    try:
        with engine.connect() as conn:
            got = read_vectors(conn, "vec_text", [*ids, "missing"])
    finally:
        event.remove(engine, "before_cursor_execute", _before)

    vec_reads = [s for s in seen if _VEC_TABLE.search(s)]
    assert len(vec_reads) == len(ids) + 1
    assert got
    assert all(re.search(r"\bIN\b", s, re.IGNORECASE) is None for s in vec_reads)


def test_the_helper_rejects_a_table_that_is_not_a_vec_table(search_index):
    read_vectors = _read_vectors()

    with search_index.engine.connect() as conn:
        with pytest.raises(ValueError):
            read_vectors(conn, "embeddings", ["x"])


# ---------------------------------------------------------------------------
# The source detector
# ---------------------------------------------------------------------------


def _string_nodes(node):
    """The outermost strings of a subtree, in source order, as ``(line, text)``."""
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append("{" + ast.unparse(value.value) + "}")
        yield node.lineno, "".join(parts)
        return
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.lineno, node.value
        return
    for child in ast.iter_child_nodes(node):
        yield from _string_nodes(child)


def _is_vec_read(sql: str) -> bool:
    if not (_VEC_TABLE.search(sql) or _VEC_PLACEHOLDER.search(sql)):
        return False
    upper = sql.upper()
    start = upper.lstrip()
    if start.startswith(("DELETE", "INSERT")):
        return False
    if "MATCH" in upper:
        return False
    if _IN_READ.search(sql):
        return True
    return "JOIN EMBEDDINGS" in upper and "EMBEDDING_ID" in upper


def vec_read_sites(root: Path = APP_DIR) -> list[str]:
    sites: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "indexer.py":
            continue
        source = path.read_text(encoding="utf-8")
        for line, sql in _string_nodes(ast.parse(source, filename=str(path))):
            if _is_vec_read(sql):
                sites.append(
                    f"{path.relative_to(root)}:{line}: "
                    + " ".join(sql.split())[:120]
                )
    return sites


def test_no_source_reads_vec_vectors_by_in_or_by_a_join():
    sites = vec_read_sites()
    assert sites == [], "\n".join(sites)
