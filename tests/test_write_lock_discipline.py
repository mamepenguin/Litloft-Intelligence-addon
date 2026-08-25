"""Guards on the search-DB write lock.

``app.database._write_lock`` is a plain :class:`threading.Lock`, so
acquiring it is a *blocking* call even on the single uvicorn event-loop
thread. A coroutine that suspends while holding it can only be resumed by
that same loop, so the next task to enter ``get_search_db()`` blocks the
loop against a lock that will never be released — the addon stops
answering entirely and Docker sees a perfectly healthy process.

That has now bitten three times in different shapes (``/status`` running
sync queries on the loop, auto_tags' TF-IDF rebuild, and a
``files.moved`` / ``scan-complete`` webhook race holding the lock across
an HTTP round-trip), so the invariant is enforced here rather than
rediscovered.
"""

import ast
import pathlib
import threading

import pytest

APP_ROOT = pathlib.Path(__file__).resolve().parent.parent / "app"


def _entered_context_names(node: ast.With | ast.AsyncWith) -> set[str]:
    """Names of the context managers a ``with`` statement enters."""
    names: set[str] = set()
    for item in node.items:
        for sub in ast.walk(item.context_expr):
            if isinstance(sub, ast.Name):
                names.add(sub.id)
            elif isinstance(sub, ast.Attribute):
                names.add(sub.attr)
    return names


def _suspension_points(node: ast.With | ast.AsyncWith) -> list[int]:
    """Line numbers inside ``node`` where the coroutine can suspend.

    Nested function bodies are skipped: a coroutine *defined* inside the
    block does not run there.
    """
    lines: list[int] = []
    for stmt in node.body:
        for sub in ast.walk(stmt):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(sub, (ast.Await, ast.AsyncFor, ast.AsyncWith)):
                lines.append(sub.lineno)
    return sorted(set(lines))


def _offenders() -> list[str]:
    found: list[str] = []
    for path in sorted(APP_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.With, ast.AsyncWith)):
                continue
            if "get_search_db" not in _entered_context_names(node):
                continue
            suspensions = _suspension_points(node)
            if suspensions:
                found.append(
                    f"{path.relative_to(APP_ROOT.parent)}:{node.lineno} "
                    f"suspends at line(s) {suspensions}"
                )
    return found


def test_write_lock_is_never_held_across_a_suspension_point():
    offenders = _offenders()
    assert not offenders, (
        "A `with get_search_db()` block must not await — it holds a "
        "blocking threading.Lock and will wedge the event loop.\n"
        "Move the awaited work outside the block, or use "
        "get_search_db_read() if the block only reads.\n  "
        + "\n  ".join(offenders)
    )


def test_write_lock_acquire_times_out_instead_of_hanging():
    """A stuck holder must surface as an exception, not an infinite wait."""
    from app import database

    database._write_lock.acquire()
    try:
        with pytest.raises(database.WriteLockTimeout):
            with database._write_lock_held():
                pytest.fail("acquired a lock that was already held")
    finally:
        database._write_lock.release()

    # The lock is usable again once the holder lets go.
    with database._write_lock_held():
        assert database._write_lock.locked()
    assert not database._write_lock.locked()


@pytest.fixture(autouse=True)
def _short_timeout(monkeypatch):
    """Keep the timeout test fast without weakening the production default."""
    from app import database

    monkeypatch.setattr(database, "_WRITE_LOCK_TIMEOUT_SECONDS", 0.05)


def test_write_lock_is_released_when_the_body_raises():
    from app import database

    with pytest.raises(ValueError):
        with database._write_lock_held():
            raise ValueError("boom")
    assert not database._write_lock.locked()


def test_lock_is_not_reentrant_by_design():
    """Documents why a nested get_search_db() self-deadlocks.

    Kept as a test so nobody 'fixes' a deadlock by swapping in an RLock:
    that would paper over same-thread nesting while leaving the
    cross-task and ``to_thread`` cases broken.
    """
    from app import database

    assert isinstance(database._write_lock, type(threading.Lock()))
