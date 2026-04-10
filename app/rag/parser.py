"""RAG response parser.

Validates the raw JSON the LLM returns from the answer prompt and
enforces the critical security invariant: citations referencing
file_ids that were not in the retrieved set are silently dropped.

LLMs hallucinate file IDs when they want to "sound authoritative";
without this check, the UI would render cards pointing at nothing
or (worse) at unrelated files the caller never saw.

hako ``RftwcVMgA0pWVBWMbN6An``.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Citation:
    """A validated LLM citation entry.

    ``relevance`` is always a float in [0.0, 1.0]; the parser clamps
    or drops any value the LLM returns outside that range.
    """

    file_id: str
    quote: str
    relevance: float


@dataclass(frozen=True)
class ParsedAnswer:
    """The parsed RAG answer with validated citations."""

    answer: str
    citations: tuple[Citation, ...]


def _coerce_relevance(value: object) -> float | None:
    """Coerce a raw relevance value to a clamped float, or None if unusable.

    Accepts:
    * int / float -> clamped to [0.0, 1.0]
    * numeric string -> parsed then clamped
    * anything else (None, dict, list, non-numeric string) -> None

    Returning None lets the caller decide whether to drop the citation
    or fall back to a default.
    """
    if isinstance(value, bool):
        # bool is a subclass of int; treat it as invalid here — the
        # LLM has no business returning booleans for relevance.
        return None
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        try:
            return max(0.0, min(1.0, float(value)))
        except ValueError:
            return None
    return None


def _parse_citation(
    entry: object,
    allowed_file_ids: frozenset[str],
) -> Citation | None:
    """Validate a single citation dict against the allowed file_id set.

    Returns None when:
    * the entry is not a dict,
    * ``file_id`` is missing or not a string,
    * ``file_id`` is not in ``allowed_file_ids``.

    Missing ``quote`` defaults to the empty string. An unparseable
    ``relevance`` defaults to ``0.0`` rather than dropping the whole
    citation — the file_id check is the real security gate.
    """
    if not isinstance(entry, dict):
        return None

    file_id = entry.get("file_id")
    if not isinstance(file_id, str) or not file_id:
        return None

    if file_id not in allowed_file_ids:
        # Log at DEBUG only — a noisy LLM could otherwise flood logs.
        logger.debug(
            "RAG parser dropped citation for unknown file_id"
        )
        return None

    quote_raw = entry.get("quote", "")
    quote = quote_raw if isinstance(quote_raw, str) else ""

    relevance = _coerce_relevance(entry.get("relevance"))
    if relevance is None:
        relevance = 0.0

    return Citation(file_id=file_id, quote=quote, relevance=relevance)


def parse_answer(
    raw: dict | list | None,
    allowed_file_ids: frozenset[str],
) -> ParsedAnswer | None:
    """Parse + validate the LLM JSON response.

    Args:
        raw: The JSON payload as returned by ``LLMClient.generate_json``
            — a dict in the happy path, possibly None or a list if the
            model misbehaves.
        allowed_file_ids: The set of file_ids the retriever actually
            returned. Any citation whose ``file_id`` is not in this
            set is dropped as a suspected hallucination.

    Returns:
        ``ParsedAnswer`` on success, or ``None`` if the response shape
        is fundamentally wrong (non-dict or missing the ``answer``
        field). Missing ``citations`` or a non-list ``citations`` value
        produces an empty citation tuple instead of a None return, so
        the caller can still render the answer text.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None

    answer = raw.get("answer")
    if not isinstance(answer, str):
        return None

    citations_raw = raw.get("citations", [])
    if not isinstance(citations_raw, list):
        citations_raw = []

    citations: list[Citation] = []
    for entry in citations_raw:
        parsed = _parse_citation(entry, allowed_file_ids)
        if parsed is not None:
            citations = [*citations, parsed]

    return ParsedAnswer(answer=answer, citations=tuple(citations))
