"""Per-provider circuit breaker for transcription.

Trips when a provider returns ``RateLimitError`` (or other classified
failure) more than ``threshold`` times within ``window_s`` seconds.
Once tripped, the breaker reports "open" for ``pause_s`` seconds
regardless of subsequent recordings, after which it closes again
unless a fresh burst of failures retrips it.

Design notes
------------

* Counters are kept per ``provider_name`` so Deepgram outages do not
  pause ElevenLabs Scribe / Whisper API.
* The clock is injectable (``now`` constructor arg) so tests can run
  deterministically without ``time.monotonic`` wall-clock drift.
* The state is in-process. Restart of the intelligence container
  resets every counter — acceptable because the worst case is one
  more cycle of 429s before retripping.
* "Open after pause" is a soft auto-close: if the underlying problem
  persists, the next batch of jobs will hit the same wall and retrip
  the breaker. This is intentional — auto-close lets the system
  self-heal without operator action.

The class deliberately avoids ``asyncio`` primitives. Worker code
calls ``is_open(...)`` synchronously before launching coroutines and
``record_failure(...)`` from inside an ``except`` block, neither of
which need awaiting.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Final


# Spec defaults — tuned in 2026-05-07-cloud-transcription-providers.md
# §"閾値の根拠" (R2-4). 20/60s tolerates batch-indexing transient bursts
# while still tripping on wholesale provider outage. The 60s pause then
# gives the upstream rate-limit window time to reset.
_DEFAULT_THRESHOLD: Final[int] = 20
_DEFAULT_WINDOW_S: Final[int] = 60
_DEFAULT_PAUSE_S: Final[int] = 60


class ProviderCircuitBreaker:
    """Sliding-window failure counter with per-provider isolation.

    Args:
        threshold: Number of failures *tolerated* within the window
            before the breaker trips. The (threshold + 1)-th failure
            opens the breaker.
        window_s: Width of the sliding window in seconds.
        pause_s: How long the breaker stays open after tripping. Held
            even if the failure count drops below threshold; counters
            do not influence the breaker again until ``pause_s`` has
            elapsed.
        now: Optional clock function returning a monotonic timestamp.
            Tests inject a fake clock; production uses
            :func:`time.monotonic`.
    """

    def __init__(
        self,
        threshold: int = _DEFAULT_THRESHOLD,
        window_s: int = _DEFAULT_WINDOW_S,
        pause_s: int = _DEFAULT_PAUSE_S,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.threshold = threshold
        self.window_s = window_s
        self.pause_s = pause_s
        self._now = now if now is not None else time.monotonic
        # provider_name -> deque[timestamp]
        self._failure_counts: dict[str, deque[float]] = defaultdict(deque)
        # provider_name -> "breaker stays open until this timestamp"
        self._open_until: dict[str, float] = {}

    def record_failure(self, provider_name: str) -> None:
        """Record one failure timestamp for ``provider_name``.

        If the recorded failure pushes the within-window count past
        ``threshold``, also start a ``pause_s`` open period. Calling
        again during an active pause simply extends the underlying
        count (and therefore the breaker stays open at least until
        the existing ``_open_until`` expires).
        """
        timestamp = self._now()
        bucket = self._failure_counts[provider_name]
        bucket.append(timestamp)
        self._evict_stale(bucket, timestamp)

        if len(bucket) > self.threshold:
            existing = self._open_until.get(provider_name, 0.0)
            self._open_until[provider_name] = max(
                existing, timestamp + self.pause_s
            )

    def is_open(self, provider_name: str) -> bool:
        """Return True iff the breaker is currently open for the provider."""
        timestamp = self._now()

        # Active pause window forces "open" regardless of the rolling count.
        open_until = self._open_until.get(provider_name)
        if open_until is not None and timestamp < open_until:
            return True

        # Pause expired (or never set) — count freshness in the window
        # to detect a re-trip in progress without waiting for the next
        # ``record_failure`` call.
        bucket = self._failure_counts.get(provider_name)
        if bucket is None:
            return False
        self._evict_stale(bucket, timestamp)
        return len(bucket) > self.threshold

    def _evict_stale(self, bucket: deque[float], now: float) -> None:
        """Drop timestamps outside the rolling window from the left."""
        cutoff = now - self.window_s
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

    def reset(self) -> None:
        """Drop all counters and open-state — exposed for tests."""
        self._failure_counts.clear()
        self._open_until.clear()
