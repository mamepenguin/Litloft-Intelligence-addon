"""How much a candidate's age costs it in the Pickup ranking.

The profile's own half-life (``app.pickup.profile.HALF_LIFE_DAYS``) makes
an *interest* go quiet. This module's is the other side of the same
question: two candidates that resemble the history equally are not equal
to a viewer who wants the new one first.
"""

from __future__ import annotations

from datetime import UTC, datetime

#: Chosen by ``evaluate replay`` against a production copy (main drive, one
#: viewer, 55 cut points, 397 targets), recall@40 per setting:
#:
#:     half-life   floor 0   floor 0.5        no freshness   0.098
#:            7d     0.082       0.070        newest first   0.127
#:           30d     0.054       0.059
#:           90d     0.078       0.075
#:          365d     0.101       0.084
#:
#: Only 365d / 0 beats no freshness, and narrowly; hit@40 moves further
#: (0.364 to 0.473) and recall@12 from 0.041 to 0.060.
HALF_LIFE_DAYS = 365.0
FLOOR = 0.0

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
