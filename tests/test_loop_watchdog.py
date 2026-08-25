"""The event-loop watchdog must notice a blocked loop and stay harmless."""

import asyncio
import logging
import threading
import time

import pytest

from app.loop_watchdog import LoopWatchdog


@pytest.fixture
def loop():
    """A real event loop running on its own thread."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_healthy_loop_produces_no_report(loop, caplog):
    watchdog = LoopWatchdog(loop, interval=0.02, threshold=0.5)
    with caplog.at_level(logging.ERROR):
        watchdog.start()
        time.sleep(0.3)
        watchdog.stop()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_blocked_loop_is_reported_with_a_stack_dump(loop, caplog):
    watchdog = LoopWatchdog(loop, interval=0.02, threshold=0.2)
    release = threading.Event()

    # Occupy the loop thread the way a blocking lock acquire would.
    loop.call_soon_threadsafe(lambda: release.wait(timeout=5))

    with caplog.at_level(logging.ERROR):
        watchdog.start()
        found = _wait_for(
            lambda: any(r.levelno >= logging.ERROR for r in caplog.records)
        )
        release.set()
        watchdog.stop()

    assert found, "watchdog did not report a blocked loop"
    message = caplog.records[0].getMessage()
    assert "not run a callback" in message
    # The dump is the point: it must name threads, not just complain.
    assert "Thread" in message


def test_report_is_emitted_once_per_stall(loop, caplog):
    watchdog = LoopWatchdog(loop, interval=0.02, threshold=0.2)
    release = threading.Event()
    loop.call_soon_threadsafe(lambda: release.wait(timeout=5))

    with caplog.at_level(logging.ERROR):
        watchdog.start()
        _wait_for(lambda: any(r.levelno >= logging.ERROR for r in caplog.records))
        time.sleep(0.3)  # several more intervals while still blocked
        release.set()
        watchdog.stop()

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1, "watchdog spammed the log while stalled"


def test_recovery_is_announced_and_rearms(loop, caplog):
    watchdog = LoopWatchdog(loop, interval=0.02, threshold=0.2)
    release = threading.Event()
    loop.call_soon_threadsafe(lambda: release.wait(timeout=5))

    with caplog.at_level(logging.WARNING):
        watchdog.start()
        _wait_for(lambda: any(r.levelno >= logging.ERROR for r in caplog.records))
        release.set()
        recovered = _wait_for(
            lambda: any("responsive again" in r.getMessage() for r in caplog.records)
        )
        watchdog.stop()

    assert recovered, "watchdog never reported the loop coming back"


def test_stop_is_idempotent_and_never_kills_the_process(loop):
    watchdog = LoopWatchdog(loop, interval=0.02, threshold=0.2)
    watchdog.start()
    watchdog.stop()
    watchdog.stop()  # must not raise
