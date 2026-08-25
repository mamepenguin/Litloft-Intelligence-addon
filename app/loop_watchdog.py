"""Detects an event loop that has stopped making progress.

uvicorn runs this service on a single event-loop thread. Anything that
blocks that thread — a synchronous lock acquired while another task holds
it across an ``await``, a long CPU-bound call that was never offloaded —
takes down *every* endpoint at once while the process keeps looking
healthy from the outside: the container stays up, memory is flat, CPU is
zero. A deadlock in this shape once went unnoticed for twelve hours, and
diagnosing it required installing ``py-spy`` into the running container.

The watchdog closes that gap. A daemon thread asks the loop to touch a
timestamp every ``interval`` seconds; if the loop has not answered for
``threshold`` seconds, it dumps every thread's stack to the log. That is
the same evidence ``py-spy dump`` produces, captured while the stall is
still happening.

It never terminates the process. Blocking the loop for minutes is bad but
sometimes legitimate (a corpus IDF rebuild, forced alignment over a long
window), and killing a real indexing job to recover from a *suspected*
hang trades one outage for another. Recovery stays a human decision,
informed by the stack dump.
"""

import faulthandler
import io
import logging
import os
import threading
import time
from asyncio import AbstractEventLoop

logger = logging.getLogger(__name__)

# How often the watchdog pings the loop.
_INTERVAL_SECONDS = float(os.environ.get("INTELLIGENCE_WATCHDOG_INTERVAL", "10"))

# How long the loop may stay silent before we consider it stalled. Well
# above any legitimate blocking call, so a dump in the log is a finding
# rather than noise.
_THRESHOLD_SECONDS = float(os.environ.get("INTELLIGENCE_WATCHDOG_THRESHOLD", "120"))


class LoopWatchdog:
    """Watches one event loop from a daemon thread."""

    def __init__(
        self,
        loop: AbstractEventLoop,
        *,
        interval: float = _INTERVAL_SECONDS,
        threshold: float = _THRESHOLD_SECONDS,
    ) -> None:
        self._loop = loop
        self._interval = interval
        self._threshold = threshold
        self._last_seen = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # One dump per stall, not one per interval.
        self._reported = False

    # -- loop side ------------------------------------------------------

    def _touch(self) -> None:
        with self._lock:
            self._last_seen = time.monotonic()

    def _silent_for(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_seen

    # -- watchdog thread ------------------------------------------------

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._loop.call_soon_threadsafe(self._touch)
            except RuntimeError:
                # Loop already closed; nothing left to watch.
                return

            silent = self._silent_for()
            if silent < self._threshold:
                if self._reported:
                    logger.warning(
                        "Event loop responsive again after %.0fs", silent
                    )
                    self._reported = False
                continue

            if self._reported:
                continue
            self._reported = True
            logger.error(
                "Event loop has not run a callback for %.0fs — the loop "
                "thread is blocked and every endpoint is unreachable. "
                "Thread stacks follow.\n%s",
                silent,
                self._thread_dump(),
            )

    @staticmethod
    def _thread_dump() -> str:
        buf = io.StringIO()
        try:
            faulthandler.dump_traceback(file=buf, all_threads=True)
        except Exception as e:  # pragma: no cover - diagnostics must not raise
            return f"<stack dump unavailable: {e}>"
        return buf.getvalue()

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._touch()
        self._thread = threading.Thread(
            target=self._run, name="loop-watchdog", daemon=True
        )
        self._thread.start()
        logger.info(
            "Event loop watchdog started (interval=%.0fs, threshold=%.0fs)",
            self._interval,
            self._threshold,
        )

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=self._interval + 1)
