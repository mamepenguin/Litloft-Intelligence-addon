"""SPEC-ADDON-076 I3: a waiting request whose deadline passes neither fails the
others nor ends the lookup they share.

Through the route the clause cannot be reached: a request's deadline is never
earlier than the 15 s limit of the lookup it joined. So the waiters are driven at
the function the route calls, with a deadline shorter than the lookup.
"""

from __future__ import annotations

import threading
import time
import uuid

import pytest
from fastapi import HTTPException

from app import youtube_source
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    YOUTUBE_URL,
    describe,
    needs_ffmpeg,
    youtube,
)

pytestmark = needs_ffmpeg


def test_a_waiter_that_times_out_leaves_the_lookup_to_the_others(youtube):
    """A gives up after 1 s; B, which joined at once, is served by the same lookup."""
    youtube.set_ytdlp(describe=describe([youtube.format("f720", 720)]), delay=3.0)
    fid = "yt" + uuid.uuid4().hex[:10]
    results: dict[str, object] = {}

    def wait(name: str, seconds: float) -> None:
        try:
            deadline = time.monotonic() + seconds
            results[name] = youtube_source._description(fid, YOUTUBE_URL, deadline)
        except HTTPException as exc:
            results[name] = exc.status_code

    threads = [
        threading.Thread(target=wait, args=("a", 1.0)),
        threading.Thread(target=wait, args=("b", 10.0)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results["a"] == 504
    assert isinstance(results["b"], dict)
    assert len(youtube.lookups()) == 1
    assert fid in youtube_source._descriptions
