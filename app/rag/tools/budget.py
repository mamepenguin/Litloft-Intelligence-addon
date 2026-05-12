"""Token budget utilities for the agentic Ask loop (Phase 1.B).

Goals:
* Cheap, dependency-free token estimation that works the same way in
  prod containers and the eval harness. tiktoken would be ideal but
  is an extra dependency we have not adopted; the agentic loop must
  not require it just to enforce a budget.
* A single per-call cap so a runaway tool result cannot blow up the
  conversation in one go.
* A cumulative counter the loop can poll between iterations to decide
  whether to allow the next tool call or to force the final answer.

The estimator is deliberately conservative — it over-counts CJK text
by treating each character as roughly one token. For Latin text the
heuristic underestimates by a small factor; the agentic loop budgets
30% headroom (``context_window * 0.7``) which absorbs the noise.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Per-call hard cap (default 8K tokens). The spec wants 4K-10K range
# depending on the tool; individual tools may pass tighter caps when
# their output is small by construction.
DEFAULT_PER_CALL_TOKEN_CAP = 8000

# Optional dependency. Prefer tiktoken if available so the production
# container (which may install it for parity with cloud LLM SDKs)
# uses the accurate count; otherwise fall back to the heuristic.
try:  # pragma: no cover - import guard
    import tiktoken  # type: ignore[import-not-found]

    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - import guard
    _TIKTOKEN_ENC = None


_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿＀-￯]")


def estimate_tokens(text: str | bytes | None) -> int:
    """Approximate the token count of ``text``.

    The estimator never raises — None / non-string inputs return 0 so
    callers can pass arbitrary tool results without guarding the type.
    Empty string returns 0 (not 1) so per-call caps treat "no output"
    as cheap.
    """
    if text is None:
        return 0
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8", errors="replace")
        except Exception:
            return 0
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return 0

    if _TIKTOKEN_ENC is not None:  # pragma: no cover - accuracy depends on env
        try:
            return len(_TIKTOKEN_ENC.encode(text))
        except Exception as exc:
            logger.debug("tiktoken encode failed (%s); using heuristic", exc)

    # Heuristic: CJK chars ≈ 1 token each, the rest ≈ 1 token / 3.7 chars.
    cjk = sum(1 for ch in text if _CJK_RE.match(ch))
    other = len(text) - cjk
    return cjk + max(1, round(other / 3.7))


def estimate_payload_tokens(payload: Any) -> int:
    """Token estimate for a structured payload (dict / list / scalar).

    Serialises to compact JSON so the count reflects what the LLM
    actually sees on the wire. Falls back to ``str(payload)`` on any
    serialisation failure so a non-JSON object (e.g. a dataclass that
    forgot to provide ``__dict__``) is still bounded.
    """
    if payload is None:
        return 0
    try:
        as_text = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        as_text = str(payload)
    return estimate_tokens(as_text)


def fits_per_call_cap(token_count: int, cap: int | None = None) -> bool:
    """True if ``token_count`` is at or below the per-call cap."""
    effective = DEFAULT_PER_CALL_TOKEN_CAP if cap is None else cap
    return token_count <= effective


def remaining_budget(cumulative: int, total_budget: int) -> int:
    """Return how many tokens remain of the loop-level total budget.

    Clamps to zero — a negative remainder means the loop should stop
    issuing tool calls and force an answer, not "owe" tokens to the
    next iteration.
    """
    return max(0, total_budget - cumulative)


__all__ = [
    "DEFAULT_PER_CALL_TOKEN_CAP",
    "estimate_tokens",
    "estimate_payload_tokens",
    "fits_per_call_cap",
    "remaining_budget",
]
