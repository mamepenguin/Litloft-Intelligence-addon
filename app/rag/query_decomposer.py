"""Stage A: natural-language query → structured Ask query.

Spec: ``2026-04-26-intelligence-ask-personal-history-query.md`` §4.2.
"先週観た映画の中で SF っぽいのどれ？" decomposes into:

* ``time_range`` — half-open ``[after, before)`` over ``last_played_at``.
* ``personal_scope`` — ``viewed`` / ``not_viewed`` / ``none``.
* ``file_type_hint`` — ``video`` / ``audio`` / ``image`` / ``text`` /
  ``none``. Matches ``File.file_type`` so the retriever can reuse it.
* ``semantic_query`` — the residual concept (e.g. "SF") that survives
  after stripping out time / scope / file-type clues. Stage C expands
  it; Stages B+D use it as the query text.

The LLM only emits *symbolic* labels (``last_week``, ``viewed``, …);
the Python layer turns ``last_week`` into concrete datetimes against
the server's current time. This keeps language interpretation in the
LLM and time math out of the prompt — the spec's "LLM 出力を回答ソース
にしない" rule and the global "言語依存のロジックを書かない" rule pull
in the same direction here.

On any failure (LLM disabled, parse failure, schema mismatch) the
result is a ``DecomposedQuery`` with all fields set to "no signal":
empty time_range, ``personal_scope="none"``, ``file_type_hint="none"``,
``semantic_query=raw_query``. The caller can then run a legacy Ask
without special-casing the failure path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from app.dependencies import get_llm_client
from app.prompt_loader import render

logger = logging.getLogger(__name__)


# Same reasoning as ``query_transform`` / ``clue_generator``: small
# local models occasionally emit short reasoning prose before the
# JSON. 256 absorbs that without meaningfully changing latency.
_DECOMPOSE_MAX_TOKENS = 256


# Symbolic time-range labels accepted from the LLM. Anything outside
# this set is mapped to ``"none"`` server-side. Keeping the vocabulary
# explicit (not free-form date ranges) is the load-bearing trick: the
# LLM never has to know today's date to be useful.
_TIME_RANGE_LABELS = frozenset(
    {
        "today",
        "yesterday",
        "this_week",
        "last_week",
        "this_month",
        "last_month",
        "this_year",
        "last_year",
        "recent",
        "none",
    }
)

# Symbolic personal-scope labels. Mirrors the ``kind`` parameter of
# ``GET /api/internal/viewer-history`` plus an explicit "no signal"
# sentinel so the caller can short-circuit Stage B entirely.
_SCOPE_LABELS = frozenset({"viewed", "not_viewed", "none"})

# File-type hint vocabulary. Aligns with ``File.file_type`` values that
# the retriever already understands. ``"none"`` is the no-hint sentinel.
_FILE_TYPE_LABELS = frozenset({"video", "audio", "image", "text", "none"})


@dataclass(frozen=True)
class TimeRange:
    """Resolved half-open time window for the ``last_played_at`` filter.

    ``after`` is inclusive, ``before`` is exclusive — matches the
    Internal API contract so the two ends agree on which boundary
    semantics to use. Either side may be ``None`` for unbounded.
    ``label`` carries the symbolic form back to the caller for SSE
    surfacing ("先週観た 12 件から検索しています").
    """

    label: str  # one of _TIME_RANGE_LABELS
    after: datetime | None
    before: datetime | None

    @classmethod
    def empty(cls) -> "TimeRange":
        """No time signal — caller should not filter on last_played_at."""
        return cls(label="none", after=None, before=None)


@dataclass(frozen=True)
class DecomposedQuery:
    """Structured form of a natural-language Ask query.

    The "all-none" instance (``DecomposedQuery.passthrough(query)``) is
    the graceful-degradation result on any LLM failure: the caller can
    treat it as "no structured signal, run a legacy Ask".
    """

    raw_query: str
    time_range: TimeRange
    personal_scope: str  # one of _SCOPE_LABELS
    file_type_hint: str  # one of _FILE_TYPE_LABELS
    semantic_query: str  # residual concept; falls back to raw_query

    @classmethod
    def passthrough(cls, raw_query: str) -> "DecomposedQuery":
        """A no-signal decomposition. Used on every failure path."""
        return cls(
            raw_query=raw_query,
            time_range=TimeRange.empty(),
            personal_scope="none",
            file_type_hint="none",
            semantic_query=raw_query,
        )

    @property
    def has_personal_signal(self) -> bool:
        """True when the caller should engage Stage B (history filter)."""
        return self.personal_scope in {"viewed", "not_viewed"}


_SYSTEM_PROMPT = render("rag/query_decomposer_system.jinja2")


def _start_of_day(now: datetime) -> datetime:
    """Truncate to 00:00:00 of the same calendar day."""
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _start_of_week(now: datetime) -> datetime:
    """Monday 00:00 of the calendar week containing ``now``.

    Uses ISO-week semantics (Monday is day 0). The personal-history
    spec doesn't pin a week-start convention; Monday is the broadly
    expected default for "今週" / "先週" in Japanese contexts.
    """
    midnight = _start_of_day(now)
    return midnight - timedelta(days=midnight.weekday())


def _start_of_month(now: datetime) -> datetime:
    """First day of the calendar month at 00:00."""
    return _start_of_day(now).replace(day=1)


def _previous_month_start(now: datetime) -> datetime:
    """First day of the calendar month preceding ``now`` at 00:00.

    Plain (year, month - 1) arithmetic with January wraparound; we
    deliberately avoid ``relativedelta`` to keep the dep tree slim.
    """
    this_month_start = _start_of_month(now)
    if this_month_start.month == 1:
        return this_month_start.replace(
            year=this_month_start.year - 1, month=12
        )
    return this_month_start.replace(month=this_month_start.month - 1)


def _start_of_year(now: datetime) -> datetime:
    """January 1st at 00:00 of the year containing ``now``."""
    return _start_of_day(now).replace(month=1, day=1)


def _resolve_time_range(
    label: str,
    *,
    now: datetime,
    max_lookback_days: int,
    recent_days: int = 14,
) -> TimeRange:
    """Map a symbolic label to a concrete ``[after, before)`` window.

    All boundaries are anchored on ``now`` so the call is deterministic
    even when the underlying clock is mocked in tests. ``max_lookback_days``
    clips the lower bound on every label so a pathological "this_year"
    on a long-running deployment cannot retrieve a multi-year history;
    the spec's §11 explicitly calls this out as a worst-case to defend
    against.

    ``recent_days`` controls how the soft "recent" / "ちょっと前" label
    is materialised. 14 days is the spec's example interpretation
    (§5).
    """
    if label not in _TIME_RANGE_LABELS or label == "none":
        return TimeRange.empty()

    midnight = _start_of_day(now)
    floor = midnight - timedelta(days=max(max_lookback_days, 0))

    if label == "today":
        after, before = midnight, midnight + timedelta(days=1)
    elif label == "yesterday":
        after, before = midnight - timedelta(days=1), midnight
    elif label == "this_week":
        start = _start_of_week(now)
        after, before = start, start + timedelta(days=7)
    elif label == "last_week":
        this_week = _start_of_week(now)
        after, before = this_week - timedelta(days=7), this_week
    elif label == "this_month":
        start = _start_of_month(now)
        # Use "now" as the upper bound rather than next-month-start so
        # an in-progress month does not mark future days as "viewed";
        # the underlying watch_history rows will never have last_played_at
        # in the future, but staying consistent with the actual signal
        # makes the SSE event ("今月観た N 件") line up with the count
        # the user mentally expects on the day the question is asked.
        after, before = start, now
    elif label == "last_month":
        prev = _previous_month_start(now)
        after, before = prev, _start_of_month(now)
    elif label == "this_year":
        start = _start_of_year(now)
        after, before = start, now
    elif label == "last_year":
        this_year = _start_of_year(now)
        after, before = (
            this_year.replace(year=this_year.year - 1),
            this_year,
        )
    elif label == "recent":
        after, before = midnight - timedelta(days=recent_days), now
    else:  # pragma: no cover - guarded by the membership check above
        return TimeRange.empty()

    # Clip the lower bound. Equivalent to ``max(after, floor)`` but
    # explicit so the intent reads cleanly in a future code review:
    # we never look further back than ``max_lookback_days``.
    if after < floor:
        after = floor

    # Pathological case: the configured floor moved past the upper
    # bound. The window is empty, so emit ``empty()`` rather than a
    # malformed ``[after >= before)`` that would 400 the Internal API.
    if after >= before:
        return TimeRange.empty()

    return TimeRange(label=label, after=after, before=before)


def _normalise_label(value: Any, allowed: frozenset[str]) -> str:
    """Pull a string field out of an LLM JSON dict and validate it.

    Anything outside ``allowed`` (including non-string values) collapses
    to ``"none"`` — the spec's no-signal sentinel that downstream stages
    treat as "skip this dimension". This is the single chokepoint where
    we enforce the schema, so the caller can trust the returned labels.
    """
    if not isinstance(value, str):
        return "none"
    cleaned = value.strip().lower()
    if cleaned not in allowed:
        return "none"
    return cleaned


async def decompose_query(
    natural_query: str,
    *,
    now: datetime | None = None,
    max_lookback_days: int = 365,
    temperature: float | None = None,
) -> DecomposedQuery:
    """Decompose a natural-language Ask query into structured form.

    Args:
        natural_query: The raw user input. Caller has already enforced
            length bounds.
        now: Override for the "current time" used to resolve relative
            time labels. Defaults to ``datetime.now(UTC)``. Tests pin
            this to a specific instant; production leaves it None.
        max_lookback_days: Hard ceiling on how far back the resolved
            time range may extend. Spec defaults to 365.
        temperature: Optional LLM temperature override. None means use
            the provider default.

    Returns:
        A ``DecomposedQuery`` instance. Always returns a value — failures
        materialise as ``DecomposedQuery.passthrough(natural_query)``
        so the caller never has to handle ``None``. ``has_personal_signal``
        is the recommended way to decide whether to engage Stage B.
    """
    stripped = natural_query.strip()
    if not stripped:
        return DecomposedQuery.passthrough(natural_query)

    try:
        llm = get_llm_client()
    except RuntimeError:
        # Dependency container not initialised — fall through to legacy
        # Ask behaviour rather than 500-ing on a startup race.
        return DecomposedQuery.passthrough(natural_query)

    if not llm.enabled:
        return DecomposedQuery.passthrough(natural_query)

    user_prompt = f"<user_question>\n{stripped}\n</user_question>"
    raw = await llm.generate_json(
        _SYSTEM_PROMPT,
        user_prompt,
        max_tokens_override=_DECOMPOSE_MAX_TOKENS,
        temperature=temperature,
    )

    if not isinstance(raw, dict):
        logger.debug("Query decomposition returned non-dict, using passthrough")
        return DecomposedQuery.passthrough(natural_query)

    time_label = _normalise_label(raw.get("time_range"), _TIME_RANGE_LABELS)
    personal_scope = _normalise_label(raw.get("personal_scope"), _SCOPE_LABELS)
    file_type_hint = _normalise_label(
        raw.get("file_type_hint"), _FILE_TYPE_LABELS
    )

    semantic_raw = raw.get("semantic_query")
    if isinstance(semantic_raw, str):
        semantic = semantic_raw.strip()
    else:
        semantic = ""
    # Empty semantic_query is a legitimate signal ("今月観た動画" — no
    # concept filter, just the personal-history slice). But when the
    # whole decomposition collapsed (no time, no scope, no semantic)
    # the LLM did not actually understand the query; fall back to the
    # raw text so retrieval still has something to embed.
    if (
        not semantic
        and time_label == "none"
        and personal_scope == "none"
    ):
        semantic = stripped

    resolved_now = now if now is not None else datetime.now(UTC)
    time_range = _resolve_time_range(
        time_label,
        now=resolved_now,
        max_lookback_days=max_lookback_days,
    )

    return DecomposedQuery(
        raw_query=natural_query,
        time_range=time_range,
        personal_scope=personal_scope,
        file_type_hint=file_type_hint,
        semantic_query=semantic,
    )
