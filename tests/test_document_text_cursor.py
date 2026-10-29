"""SPEC-ADDON-036: ``start``, ``count``, ``offset``, ``max_chars`` and the ``next``
cursor of ``GET /files/{file_id}/document-text``."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.document_fixtures import DOCX, EPUB, make_docx
from tests.document_text_api import api, reader, search_index, seed  # noqa: F401
from tests.epub_fixtures import Item, make_epub, xhtml

SVG = '<svg xmlns="http://www.w3.org/2000/svg"><text>picture</text></svg>'


def _book(tmp_path: Path, *bodies: str | None) -> Path:
    """One spine item per body; ``None`` is an image item, which reads as empty."""
    items = [
        Item(f"c{i}", f"c{i}.svg", SVG, media_type="image/svg+xml") if body is None
        else Item(f"c{i}", f"c{i}.xhtml", xhtml(f"<p>{body}</p>"))
        for i, body in enumerate(bodies, start=1)
    ]
    return make_epub(tmp_path / "b.epub", items, spine=[i.id for i in items])


@pytest.fixture()
def three(reader, tmp_path):
    reader.register(
        "f1", _book(tmp_path, "alpha alpha alpha", "bravo bravo bravo", "charlie charlie"), EPUB,
    )
    texts = reader.full_texts("f1")
    assert all(len(t) >= 10 for t in texts.values()), texts
    return texts


def _pieces(body: dict) -> list[tuple[int, str]]:
    return [(u["number"], u["text"]) for u in body["units"]]


class TestMaxChars:
    def test_spec_addon_036_a_unit_that_would_pass_max_chars_is_cut_and_next_points_into_it(
        self, reader, three,
    ):
        body = reader.ok("f1", max_chars=len(three[1]) + 5)

        assert _pieces(body) == [(1, three[1]), (2, three[2][:5])]
        assert body["next"] == {"start": 2, "offset": 5}
        assert [u["truncated"] for u in body["units"]] == [False, False]

    def test_spec_addon_036_text_that_reaches_max_chars_exactly_ends_at_the_unit_boundary(
        self, reader, three,
    ):
        body = reader.ok("f1", max_chars=len(three[1]))

        assert _pieces(body) == [(1, three[1])]
        assert body["next"] == {"start": 2, "offset": 0}

    def test_spec_addon_036_the_next_offset_is_where_the_text_stopped_within_the_unit(
        self, reader, three,
    ):
        body = reader.ok("f1", start=2, offset=3, max_chars=4)

        assert _pieces(body) == [(2, three[2][3:7])]
        assert body["next"] == {"start": 2, "offset": 7}

    def test_spec_addon_036_default_max_chars_is_50000(self, reader, tmp_path):
        text = "".join(chr(0x61 + i % 26) for i in range(60_000))
        reader.register("f1", make_docx(tmp_path / "a.docx", paragraphs=(text,)), DOCX)

        body = reader.ok("f1")

        [unit] = body["units"]
        assert len(unit["text"]) == 50_000
        assert unit["text"] == text[:50_000]
        assert body["next"] == {"start": 1, "offset": 50_000}

    def test_spec_addon_036_max_chars_200000_returns_200000_characters(self, reader, tmp_path):
        text = "".join(chr(0x61 + i % 26) for i in range(250_000))
        reader.register("f1", make_docx(tmp_path / "a.docx", paragraphs=(text,)), DOCX)

        body = reader.ok("f1", max_chars=200_000)

        assert len(body["units"][0]["text"]) == 200_000
        assert body["next"] == {"start": 1, "offset": 200_000}

    def test_spec_addon_036_offsets_and_max_chars_count_code_points(self, reader, tmp_path):
        reader.register(
            "f1", make_docx(tmp_path / "a.docx", paragraphs=("\U00020bb7" * 3 + "abc",)), DOCX,
        )

        cut = reader.ok("f1", max_chars=2)
        rest = reader.ok("f1", offset=2, max_chars=4)

        assert cut["units"][0]["text"] == "\U00020bb7\U00020bb7"
        assert cut["next"] == {"start": 1, "offset": 2}
        assert rest["units"][0]["text"] == "\U00020bb7abc"


class TestOffsetAndRange:
    def test_spec_addon_036_offset_skips_characters_of_the_first_unit_only(self, reader, three):
        body = reader.ok("f1", start=2, offset=3, count=2)

        assert _pieces(body) == [(2, three[2][3:]), (3, three[3])]
        assert body["next"] is None

    @pytest.mark.parametrize("extra", [0, 1, 100], ids=["at-length", "one-past", "far-past"])
    def test_spec_addon_036_an_offset_at_or_past_the_first_unit_leaves_it_out_and_reads_on(
        self, reader, three, extra,
    ):
        body = reader.ok("f1", start=2, offset=len(three[2]) + extra, count=2)

        assert _pieces(body) == [(3, three[3])]
        assert body["next"] is None

    def test_spec_addon_036_a_left_out_unit_still_counts_toward_count(self, reader, three):
        body = reader.ok("f1", start=1, offset=len(three[1]), count=2)

        assert _pieces(body) == [(2, three[2])]
        assert body["next"] == {"start": 3, "offset": 0}

    def test_spec_addon_036_an_empty_unit_read_at_offset_0_is_returned_empty(
        self, reader, tmp_path,
    ):
        reader.register("f1", _book(tmp_path, "alpha alpha", None, "charlie charlie"), EPUB)

        alone = reader.ok("f1", start=2, count=1)
        among = reader.ok("f1", count=3)

        assert _pieces(alone) == [(2, "")]
        assert alone["next"] == {"start": 3, "offset": 0}
        assert [n for n, _ in _pieces(among)] == [1, 2, 3]
        assert _pieces(among)[1] == (2, "")

    def test_spec_addon_036_an_empty_unit_read_past_offset_0_is_left_out(
        self, reader, tmp_path,
    ):
        reader.register("f1", _book(tmp_path, "alpha alpha", None, "charlie charlie"), EPUB)
        third = reader.full_texts("f1")[3]

        body = reader.ok("f1", start=2, offset=1, count=2)

        assert _pieces(body) == [(3, third)]

    def test_spec_addon_036_count_bounds_the_units_examined(self, reader, three):
        body = reader.ok("f1", start=1, count=2)

        assert _pieces(body) == [(1, three[1]), (2, three[2])]
        assert body["next"] == {"start": 3, "offset": 0}

    def test_spec_addon_036_the_last_unit_in_full_ends_the_cursor(self, reader, three):
        body = reader.ok("f1", start=3)

        assert _pieces(body) == [(3, three[3])]
        assert body["next"] is None

    @pytest.mark.parametrize("start", [4, 50])
    def test_spec_addon_036_start_past_unit_count_is_an_empty_read(self, reader, three, start):
        body = reader.ok("f1", start=start)

        assert body["unit_count"] == 3
        assert body["units"] == []
        assert body["next"] is None

    def test_spec_addon_036_start_past_a_docx_single_unit_is_an_empty_read(self, reader, tmp_path):
        reader.register("f1", make_docx(tmp_path / "a.docx", paragraphs=("text",)), DOCX)

        body = reader.ok("f1", start=2)

        assert (body["unit_count"], body["units"], body["next"]) == (1, [], None)


class TestCursorWalk:
    @pytest.mark.parametrize("max_chars", [1, 4, 7, 50_000])
    @pytest.mark.parametrize("count", [1, 2, 10])
    def test_spec_addon_036_i2_i5_following_next_yields_every_unit_once_in_order(
        self, reader, tmp_path, max_chars, count,
    ):
        reader.register(
            "f1",
            _book(tmp_path, "alpha alpha", None, "charlie", "delta delta delta delta"),
            EPUB,
        )
        full = reader.full_texts("f1")

        returned: list[tuple[int, str]] = []
        cursor: dict = {"count": count, "max_chars": max_chars}
        for _ in range(1_000):
            body = reader.ok("f1", **cursor)
            assert sum(len(u["text"]) for u in body["units"]) <= max_chars
            assert all(1 <= u["number"] <= 4 for u in body["units"])
            returned.extend(_pieces(body))
            if body["next"] is None:
                break
            cursor = {**cursor, **body["next"]}
        else:
            pytest.fail("next never became null")

        numbers = [n for n, _ in returned]
        assert numbers == sorted(numbers)
        assert set(numbers) == set(full)
        joined = {n: "".join(t for m, t in returned if m == n) for n in full}
        assert joined == full

    @pytest.mark.parametrize("max_chars", [1_000, 2_499, 2_500])
    def test_spec_addon_036_i2_a_long_document_pages_by_characters(
        self, reader, tmp_path, max_chars,
    ):
        paragraphs = tuple(f"paragraph {i:03d} " + "x" * 40 for i in range(50))
        reader.register("f1", make_docx(tmp_path / "a.docx", paragraphs=paragraphs), DOCX)
        [whole] = reader.full_texts("f1").values()

        units = reader.walk("f1", max_chars=max_chars)

        assert {u["number"] for u in units} == {1}
        assert "".join(u["text"] for u in units) == whole
