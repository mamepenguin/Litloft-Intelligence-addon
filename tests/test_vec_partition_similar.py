"""SPEC-ADDON-023: the similar-files KNN scans only the requested type's partition,
widens until it holds ``limit`` files the viewer can be shown, and runs the
quality filters on those files only.

Distances are L2 between unit vectors, so a candidate at distance ``d`` from the
source scores ``1 - d^2 / 2``. The drive's own candidates sit at 0.30-0.48
(scores 0.955-0.885), a range every quality filter passes on its own: each test
checks that baseline first, so a test cannot pass by returning nothing.
"""

from __future__ import annotations

import pytest

from app import search as search_mod
from tests.vec_partition_db import (  # noqa: F401
    at_distance,
    search_index,
    unit,
)

HOME = "home"
OTHER = "elsewhere"
LIMIT = 6
CANDIDATES = [0.30 + 0.02 * i for i in range(10)]

LEGS = [
    pytest.param("vec_clip", "clip_thumbnail", "clip", id="clip_thumbnail-vs-scene-clip"),
    pytest.param("vec_text", "tfidf_keywords", "whisper", id="tfidf_keywords-vs-whisper"),
]


def _source_and_candidates(index, table, etype, distances=CANDIDATES):
    index.add_many([("src", HOME, True, "e_src", etype, table, unit(index.dim(table), 0))])
    _candidates(index, table, etype, distances, "c")


def _candidates(index, table, etype, distances, prefix):
    dim = index.dim(table)
    index.add_many([
        (f"{prefix}{i}", HOME, True, f"e_{prefix}{i}", etype, table, at_distance(dim, d, i))
        for i, d in enumerate(distances)
    ])


def _noise(index, table, etype, n, *, drive=OTHER, active=True, prefix="n",
           near=0.05, far=0.25, rows_per_file=1):
    """``n`` rows nearer to the source than every candidate."""
    dim = index.dim(table)
    index.add_many([
        (
            f"{prefix}{i // rows_per_file}", drive, active, f"e_{prefix}{i}", etype, table,
            at_distance(dim, near + (far - near) * i / max(n, 1), 1000 + i),
        )
        for i in range(n)
    ])


def _similar(etype, limit=LIMIT, drive=HOME):
    return [
        (r["file_id"], round(r["score"], 6))
        for r in search_mod._find_similar_by_embedding("src", etype, limit, drive)
    ]


def _baseline(index, table, etype):
    _source_and_candidates(index, table, etype)
    before = _similar(etype)
    assert len(before) == LIMIT, before
    assert all(fid.startswith("c") for fid, _ in before)
    return before


# ---------------------------------------------------------------------------
# I4: the scan never scores another type's row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table, etype, impostor", LEGS)
def test_spec_addon_023_nearer_rows_of_another_type_do_not_crowd_out_the_requested_type(
    search_index, table, etype, impostor,
):
    """More rows of another type than the fetch cap, all nearer than every
    candidate: only a scan confined to the requested type reaches the candidates."""
    before = _baseline(search_index, table, etype)

    _noise(
        search_index, table, impostor, search_mod._NEIGHBOUR_FETCH_MAX + 10,
        drive=HOME, prefix="imp", rows_per_file=50,
    )

    assert _similar(etype) == before


# ---------------------------------------------------------------------------
# I6 / I11: rows the viewer cannot be shown neither stop the widening nor take
# part in the quality filters
# ---------------------------------------------------------------------------


NOISE = [
    pytest.param(dict(drive=OTHER, active=True), 40, id="other-drive-fills-first-fetch"),
    pytest.param(dict(drive=HOME, active=False), 40, id="inactive-fills-first-fetch"),
    pytest.param(dict(drive=OTHER, active=False), 40, id="inactive-other-drive"),
    pytest.param(dict(drive=OTHER, active=True, near=0.10, far=0.11), 1,
                 id="one-near-other-drive-neighbour"),
    pytest.param(dict(drive=OTHER, active=True, near=0.0, far=0.29), 400,
                 id="other-drive-fills-the-fourth-fetch"),
]


@pytest.mark.parametrize("table, etype, _impostor", LEGS)
@pytest.mark.parametrize("noise, n", NOISE)
def test_spec_addon_023_rows_the_viewer_cannot_see_do_not_change_the_result(
    search_index, table, etype, _impostor, noise, n,
):
    before = _baseline(search_index, table, etype)

    _noise(search_index, table, etype, n, **noise)

    assert _similar(etype) == before


def test_spec_addon_023_the_cap_ends_the_widening_and_what_was_found_is_returned(
    search_index,
):
    """A partition larger than the cap, where fewer than ``limit`` usable files lie
    within it: the files inside the cap are returned, the ones beyond it are not."""
    table, etype = "vec_clip", "clip_thumbnail"
    cap = search_mod._NEIGHBOUR_FETCH_MAX
    limit = 300
    assert (1 + limit + 20) * 16 > cap
    _source_and_candidates(search_index, table, etype, distances=[0.30, 0.32, 0.34])
    _noise(search_index, table, etype, cap - 1 - 3 - 5, near=0.02, far=0.29, prefix="a")
    _noise(search_index, table, etype, 20, near=0.35, far=0.36, prefix="b")
    _candidates(search_index, table, etype, [0.40, 0.42, 0.44], "far")

    result = {fid for fid, _ in _similar(etype, limit=limit)}

    assert result == {"c0", "c1", "c2"}


# ---------------------------------------------------------------------------
# I5: find_similar never returns the source, another drive or an inactive file
# ---------------------------------------------------------------------------


def test_spec_addon_023_find_similar_returns_only_the_drives_active_files(search_index):
    table, etype = "vec_clip", "clip_thumbnail"
    _source_and_candidates(search_index, table, etype)
    _noise(search_index, table, etype, 25, drive=OTHER, prefix="o")
    _noise(search_index, table, etype, 25, drive=HOME, active=False, prefix="i",
           near=0.06, far=0.26)
    search_index.sql(
        "UPDATE indexed_files SET mime_type = 'image/jpeg', file_type = 'image'"
    )
    search_mod.invalidate_similar_cache()

    result = search_mod.find_similar("src", limit=3, drive=HOME)

    ids = [r["file_id"] if isinstance(r, dict) else r.file_id for r in result.results]
    assert ids, "the drive holds ten similar files"
    assert set(ids) <= {f"c{i}" for i in range(len(CANDIDATES))}
