"""Unit tests for ``ContentExtractor.chunk_text``.

Exposes the overlap-vs-separator pathology that caused HTML files to
explode into hundreds of single-character chunks (observed in
2026-05-12 reindex of citation-ui-mockup.html: 2687-char body
produced 195 chunks where ~10 was expected).

The bug: when ``rfind(separator, start, end)`` returns a position
within ``overlap`` chars of ``start``, the ``start = max(split_pos -
overlap, start + 1)`` fallback advances by 1 char per iteration and
re-discovers the same separator on the next loop, producing 1-char
tail chunks until ``start`` finally crosses the separator.
"""

from __future__ import annotations

from app.extractors.base import ContentExtractor


def test_chunk_text_short_text_returns_single_chunk() -> None:
    chunks = ContentExtractor.chunk_text("hello", max_size=100, overlap=20)
    assert chunks == ["hello"]


def test_chunk_text_empty_returns_empty() -> None:
    assert ContentExtractor.chunk_text("", max_size=100, overlap=20) == []
    assert ContentExtractor.chunk_text("   ", max_size=100, overlap=20) == []


def test_chunk_text_no_pathological_growth_for_2687_char_body() -> None:
    """Regression: real citation-ui-mockup.html section.

    The body has sparse ``\\n\\n`` separators that triggered the
    overlap-re-discovery pathology. The expected chunk count is
    roughly ``len(body) / (max_size - overlap)`` ≈ 8-12. The buggy
    algorithm produced 195.
    """
    # Reconstruct a body shape that triggers the bug: a long text with
    # ``\\n\\n`` separators spaced 200-300 chars apart, then a tail
    # paragraph close (within overlap) to a previous separator.
    paragraphs = [
        "あ" * 280,
        "い" * 240,
        "う" * 260,
        "え" * 220,
        "お" * 290,
        "か" * 250,
        "き" * 230,
        "く" * 270,
        "け" * 240,
        "こ" * 200,
    ]
    body = "\n\n".join(paragraphs)
    chunks = ContentExtractor.chunk_text(body, max_size=400, overlap=80)
    # The body is ~2500 chars. With max_size=400 / overlap=80, the
    # legitimate chunk count is in the single digits to low teens.
    assert len(chunks) < 30, f"chunk explosion: got {len(chunks)} chunks"
    # No 1-char chunks should appear (a clear sign of the bug).
    tiny = [c for c in chunks if len(c) < 10]
    assert tiny == [], f"unexpectedly tiny chunks: {tiny[:5]}"


def test_chunk_text_close_separator_does_not_loop() -> None:
    """Direct repro: separator within ``overlap`` distance of start.

    Even when the only separator in the window is right after
    ``start``, the algorithm must advance past it on the next
    iteration instead of re-discovering it. Symptom of the buggy
    behaviour: a sequence of chunks that are 1-char-shorter strict
    suffixes of each other (e.g. ``"short"``, ``"hort"``, ``"ort"``,
    ``"rt"``, ``"t"``). One overlap-driven near-prefix repeat is fine
    (e.g. "short" followed by "hort..."), three or more in a row is
    the suffix-loop bug.
    """
    body = "short\n\n" + ("x" * 1000)
    chunks = ContentExtractor.chunk_text(body, max_size=400, overlap=80)
    # 1006 chars / (400 - 80) ≈ 4 chunks. Tolerate up to 5.
    assert len(chunks) <= 5, f"chunk explosion: got {len(chunks)}: {chunks!r}"
    # No 3+ consecutive strict-suffix chunks of the same source string.
    suffix_streak = 0
    max_streak = 0
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        cur = chunks[i]
        if cur != prev and prev.endswith(cur):
            suffix_streak += 1
            max_streak = max(max_streak, suffix_streak)
        else:
            suffix_streak = 0
    assert max_streak < 2, f"suffix-loop detected ({max_streak + 1} chunks): {chunks!r}"


def test_chunk_text_sparse_separators_with_long_tail() -> None:
    """Citation-ui-mockup pathology: a separator followed by a long body
    of single-newline content (e.g. markdown tables) further than
    ``max_size`` away from the next ``\\n\\n``.

    The buggy algorithm re-finds the same ``\\n\\n`` on every
    iteration as ``start`` creeps forward by 1 char.
    """
    # 300 chars before the separator, then 1500 chars with only single
    # \n line breaks (table rows pattern).
    body = (
        ("p" * 300)
        + "\n\n"
        + "\n".join("row " + str(i) for i in range(150))
    )
    chunks = ContentExtractor.chunk_text(body, max_size=400, overlap=80)
    # Should be ~5-10 chunks, definitely not 50+.
    assert len(chunks) < 20, f"chunk explosion: got {len(chunks)}"


def test_chunk_text_preserves_full_content() -> None:
    """All input chars (excluding leading/trailing whitespace) appear in chunks.

    Note: with overlap the same chars appear in multiple chunks. The
    union of all chunks must cover the original body.
    """
    body = "Lorem ipsum dolor sit amet. " * 50  # ~1400 chars
    chunks = ContentExtractor.chunk_text(body, max_size=400, overlap=80)
    joined = " ".join(chunks)
    # Every word from the source must appear somewhere in the chunks.
    for token in {"Lorem", "ipsum", "dolor", "sit", "amet"}:
        assert token in joined


def test_chunk_text_overlap_zero_works() -> None:
    """overlap=0 produces non-overlapping disjoint chunks."""
    body = "a" * 1500
    chunks = ContentExtractor.chunk_text(body, max_size=400, overlap=0)
    # Reconstruct: should be exactly the body when concatenated
    assert "".join(chunks) == body
    assert len(chunks) == 4  # ceil(1500/400)


def test_chunk_text_chunks_under_max_size() -> None:
    body = "word " * 500  # ~2500 chars with many spaces
    chunks = ContentExtractor.chunk_text(body, max_size=400, overlap=80)
    for c in chunks:
        assert len(c) <= 400 + 5  # tolerance for strip boundaries
