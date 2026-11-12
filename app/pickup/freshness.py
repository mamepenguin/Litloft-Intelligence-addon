"""How much a candidate's age costs it in the Pickup ranking.

The profile's own half-life (``app.pickup.profile.HALF_LIFE_DAYS``) makes
an *interest* go quiet. This module's is the other side of the same
question: two candidates that resemble the history equally are not equal
to a viewer who wants the new one first.
"""

from __future__ import annotations

from datetime import UTC, datetime

#: Provisional: the spec's ship bar decides the setting that ships, from the
#: replay's grid, which reports these two as the worker's constants. Until
#: that decision nothing may depend on the numbers themselves.
HALF_LIFE_DAYS = 30.0
FLOOR = 0.5

#: A ``created_at`` this far in the future is clock skew rather than a new
#: file. Further ahead is unreadable, and gets the floor like an old file:
#: clamping it to age zero would rank the broken row above every real one.
_FUTURE_TOLERANCE_DAYS = 1.0


def reference_now() -> datetime:
    """Naive UTC now, taken once per drive computation."""
    return datetime.now(UTC).replace(tzinfo=None)


def freshness(
    created_at: datetime | None,
    now: datetime,
    *,
    half_life_days: float = HALF_LIFE_DAYS,
    floor: float = FLOOR,
) -> float:
    """The multiplier a candidate's age earns it, in ``[floor, 1]``.

    ``created_at`` is naive UTC, or ``None`` when it could not be parsed.
    An age up to one day ahead of ``now`` is clock skew and counts as
    zero; further ahead is treated as unknown. Both unknown cases return
    ``floor`` exactly, so they tie with the oldest real file rather than
    outranking it.
    """
    if floor >= 1.0:
        return 1.0
    if created_at is None:
        return floor

    age = (now - created_at).total_seconds() / 86400.0
    if age < -_FUTURE_TOLERANCE_DAYS:
        return floor
    if age < 0.0:
        age = 0.0
    return floor + (1.0 - floor) * 0.5 ** (age / half_life_days)
