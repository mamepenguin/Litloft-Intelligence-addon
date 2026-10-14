"""Read-session doubles built from a test's write-session double."""

from __future__ import annotations

from contextlib import contextmanager


def read_session_from(write_factory):
    """A ``get_search_db_read`` stand-in over ``write_factory``'s sessions.

    Rolls back on exit, as the real read session never commits.
    """

    @contextmanager
    def _read():
        with write_factory() as session:
            try:
                yield session
            finally:
                session.rollback()

    return _read
