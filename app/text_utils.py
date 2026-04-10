"""Shared text manipulation utilities.

Currently hosts the sentence-boundary trimmer used by both the
summaries worker (window sampling) and the RAG context builder
(segment excerpt trimming). Kept dependency-free so any module
can import it without pulling in search/DB code.
"""


# Sentence boundary characters used when trimming snippet edges.
# Keeps an extracted substring from starting or ending mid-sentence.
# Note: ASCII "." is deliberately excluded — it creates too many false
# positives on abbreviations (Mr., e.g., U.S., 3.14, etc.). Full stops
# in English are detected via "!" / "?" / "\n" which is good enough
# for the trimming heuristic.
_SENTENCE_BOUNDARY_CHARS: tuple[str, ...] = (
    "。", "．", "!", "?", "！", "？", "\n",
)


def trim_to_sentence_boundary(snippet: str) -> str:
    """Trim a snippet so both ends land on a sentence boundary.

    Drops the leading fragment up to the first boundary and the trailing
    fragment after the last boundary. The leading trim is skipped when
    it would remove more than half of the snippet — in that case we'd
    rather keep a mid-sentence opening than lose the majority of the
    extracted content.

    Args:
        snippet: Raw substring extracted from a larger text.

    Returns:
        The snippet with incomplete fragments at each edge removed.
    """
    if not snippet:
        return snippet

    # Find the first sentence boundary after the first char (leading trim).
    leading_cut = 0
    for i, char in enumerate(snippet):
        if char in _SENTENCE_BOUNDARY_CHARS:
            # Include the boundary char itself, cut after it.
            leading_cut = i + 1
            break

    # Guard: don't drop more than half the snippet chasing a boundary.
    # If the first boundary sits past the midpoint, the head content
    # outweighs the benefit of a clean sentence start, so we keep it.
    if leading_cut > len(snippet) // 2:
        leading_cut = 0

    # Find the last sentence boundary (trailing trim).
    trailing_cut = len(snippet)
    for i in range(len(snippet) - 1, -1, -1):
        if snippet[i] in _SENTENCE_BOUNDARY_CHARS:
            trailing_cut = i + 1
            break

    trimmed = snippet[leading_cut:trailing_cut].strip()
    return trimmed or snippet.strip()
