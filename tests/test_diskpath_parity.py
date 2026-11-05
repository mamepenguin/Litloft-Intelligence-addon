"""SPEC-ADDON-062 I16: the intelligence copy of the resolver returns, for each
case of the shared parity table, that table's expected result. Also pins the
resolver contract the spec states (exact wins, strict ambiguity, parent
resolution, no writes)."""

from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path

import pytest

from tests.nfd_fs import (
    form_path,
    make_file,
    require_distinguishing_filesystem,
)

TABLE = json.loads(
    (Path(__file__).parent / "fixtures" / "diskpath_parity.json").read_text()
)
NFC = lambda s: unicodedata.normalize("NFC", s)  # noqa: E731
NFD = lambda s: unicodedata.normalize("NFD", s)  # noqa: E731


@pytest.fixture()
def root(tmp_path):
    require_distinguishing_filesystem(tmp_path)
    return tmp_path


@pytest.mark.parametrize("case", TABLE["cases"], ids=[c["id"] for c in TABLE["cases"]])
def test_spec_addon_062_parity_table(root, case):
    from app import diskpath

    for spec in case["disk"]:
        make_file(root, spec)

    got = diskpath.resolve_on_disk(root, case["rel"])

    assert os.fspath(got) == os.fspath(root / form_path(case["expect"]))


def test_spec_addon_062_resolution_does_not_write_to_the_tree(root):
    from app import diskpath

    make_file(root, "nfd:がぎ/nfd:ぱぴ.mp4")
    before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))

    diskpath.resolve_on_disk(root, "がぎ/ぱぴ.mp4")
    diskpath.resolve_parent(root, "がぎ/new.mp4")

    after = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
    assert after == before


def test_spec_addon_062_resolve_parent_returns_real_parent_and_untouched_leaf(root):
    from app import diskpath

    make_file(root, "nfd:がぎ/other.mp4")

    got = diskpath.resolve_parent(root, "がぎ/ぱぴ.mp4")

    assert os.fspath(got) == os.fspath(root / NFD("がぎ") / NFC("ぱぴ.mp4"))


def test_spec_addon_062_strict_resolve_refuses_an_ambiguous_component(root):
    from app import diskpath

    # Two spellings, neither equal to the NFC string handed in, so no exact
    # entry exists and the match is ambiguous.
    mixed_a = NFD("ぱ") + NFC("ぴ")
    mixed_b = NFC("ぱ") + NFD("ぴ")
    (root / mixed_a).write_bytes(b"a")
    (root / mixed_b).write_bytes(b"b")

    with pytest.raises(diskpath.AmbiguousPath):
        diskpath.resolve_on_disk(root, NFC("ぱぴ"), strict=True)


def test_spec_addon_062_non_strict_resolve_of_an_ambiguous_component_does_not_raise(root):
    from app import diskpath

    (root / (NFD("ぱ") + NFC("ぴ"))).write_bytes(b"a")
    (root / (NFC("ぱ") + NFD("ぴ"))).write_bytes(b"b")

    diskpath.resolve_on_disk(root, NFC("ぱぴ"))


def test_spec_addon_062_strict_resolve_prefers_the_exact_entry_over_twins(root):
    from app import diskpath

    make_file(root, "nfc:ぱぴ.mp4")
    make_file(root, "nfd:ぱぴ.mp4")

    got = diskpath.resolve_on_disk(root, NFC("ぱぴ.mp4"), strict=True)

    assert os.fspath(got) == os.fspath(root / NFC("ぱぴ.mp4"))
