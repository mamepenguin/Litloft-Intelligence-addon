"""Memory management utilities for the intelligence addon."""

import ctypes
import logging

logger = logging.getLogger(__name__)


def malloc_trim() -> None:
    """Return freed memory pages to the OS via glibc malloc_trim.

    Python's gc.collect() frees objects but glibc's allocator tends to
    hold pages in its own free-list rather than returning them to the OS.
    Calling malloc_trim(0) forces those pages back, lowering RSS after
    model unloads. Silent no-op on non-glibc platforms (musl, macOS).
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
