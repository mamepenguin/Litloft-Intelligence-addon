"""No synchronous DB work may run on the indexer's event loop.

This is the third outage from one bug class, and the second shape of it.

* The first shape was `await` inside `with get_search_db()` — a
  `threading.Lock` held across a suspension point, which deadlocked the
  loop for 12 hours. `test_write_lock_discipline.py` guards that.
* The second shape is plain synchronous SQLAlchemy called straight from
  an `async def`. No `await` in the lock, so the first guard never saw
  it. `reconcile()` in that shape held the loop past the watchdog's
  121s threshold and every endpoint became unreachable.

Both shapes stall the same thing, so this file pins the broader
invariant: inside an `async def`, a helper that opens a DB session must
go through `asyncio.to_thread` (or an equivalent executor hop), never a
direct call.

Waiting on the write lock is itself the expensive part, so this applies
even to helpers whose query is O(1) — a bulk writer holding the lock
would otherwise stall every endpoint while a single-row lookup waits.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_INDEXER = (
    pathlib.Path(__file__).resolve().parent.parent / "app" / "indexer.py"
)

# Session factories. Calling one inside an async def means the caller
# is about to do blocking DB work on the loop.
_SESSION_FACTORIES = {
    "get_search_db",
    "get_search_db_read",
    "get_litloft_db",
}


def _module() -> ast.Module:
    return ast.parse(_INDEXER.read_text(), str(_INDEXER))


def _called_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class _CallGraph(ast.NodeVisitor):
    """Every function definition, what it calls, and whether it opens a session.

    Nested defs and methods are collected too, not just module level:
    the offload pattern puts the blocking body in a closure, and a
    closure invoked directly is exactly the regression to catch.

    Only `Call` nodes count. A name passed to ``asyncio.to_thread`` is
    an argument, not a call, which is what separates the fix from the
    bug.
    """

    def __init__(self) -> None:
        self.defs: dict[str, dict] = {}
        self.is_async: dict[str, bool] = {}
        self._stack: list[str] = []

    def _enter(self, node, is_async: bool) -> None:
        self.defs.setdefault(node.name, {"calls": set(), "opens": False})
        # A name defined both sync and async somewhere would be ambiguous;
        # treat it as async only if every definition is.
        self.is_async[node.name] = self.is_async.get(node.name, True) and is_async
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter(node, is_async=True)

    def visit_Call(self, node: ast.Call) -> None:
        name = _called_name(node)
        if self._stack and name:
            current = self.defs[self._stack[-1]]
            if name in _SESSION_FACTORIES:
                current["opens"] = True
            current["calls"].add(name)
        self.generic_visit(node)


def _imported_app_modules(tree: ast.Module) -> list[pathlib.Path]:
    """Addon modules `indexer.py` imports from, resolved to files.

    Workers like ``index_text_content`` open sessions of their own, and
    calling one straight from an async body blocks the loop exactly the
    same way. Their definitions live in another file, so the graph has
    to reach across the import to see them.
    """
    app_dir = _INDEXER.parent
    paths: list[pathlib.Path] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not node.module.startswith("app."):
            continue
        rel = node.module.split(".")[1:]
        candidate = app_dir.joinpath(*rel).with_suffix(".py")
        # A module imported from twice must not be audited twice, or the
        # same offender is reported once per import site.
        if candidate.is_file() and candidate not in paths:
            paths.append(candidate)
    return paths


def _sync_db_helpers(
    tree: ast.Module, extra_trees: list[ast.Module] | None = None
) -> set[str]:
    """Names that block when called, directly or through another helper.

    Async definitions are excluded: awaiting one is not what stalls the
    loop, and if it does block, its own body is flagged on its own.
    """
    graph = _CallGraph()
    graph.visit(tree)
    for extra in extra_trees or []:
        graph.visit(extra)

    blocking = {
        name
        for name, info in graph.defs.items()
        if info["opens"] and not graph.is_async.get(name, False)
    }

    # Propagate to callers until it settles: a helper that calls a
    # blocking helper blocks too (_purge_files -> _purge_file).
    changed = True
    while changed:
        changed = False
        for name, info in graph.defs.items():
            if name in blocking or graph.is_async.get(name, False):
                continue
            if info["calls"] & blocking:
                blocking.add(name)
                changed = True
    return blocking


class _Offenders(ast.NodeVisitor):
    """Collect blocking DB calls made directly from an async body.

    A nested `def` inside an async function is NOT flagged: that is the
    shape of the fix (define a sync closure, hand it to
    `asyncio.to_thread`). Only calls made while the innermost enclosing
    function is `async` count.
    """

    def __init__(self, blocking: set[str]) -> None:
        self.blocking = blocking
        self.current: str | None = None
        self.hits: list[tuple[str, str, int]] = []

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        prev, self.current = self.current, node.name
        self.generic_visit(node)
        self.current = prev

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Entering a sync function: anything inside is off the loop
        # already (it can only run via an executor).
        prev, self.current = self.current, None
        self.generic_visit(node)
        self.current = prev

    def visit_Call(self, node: ast.Call) -> None:
        name = _called_name(node)
        if self.current and name in self.blocking:
            self.hits.append((self.current, name, node.lineno))
        self.generic_visit(node)


def _offenders() -> list[tuple[str, str, str, int]]:
    """Blocking calls on the loop, in indexer.py and everything it imports.

    Auditing only indexer.py is not enough: an imported coroutine such
    as ``index_whisper`` is awaited straight from a worker, so a session
    opened in its body runs on the very same loop.
    """
    tree = _module()
    paths = _imported_app_modules(tree)
    imported = [ast.parse(p.read_text(), str(p)) for p in paths]
    blocking = _sync_db_helpers(tree, imported) | _SESSION_FACTORIES

    hits: list[tuple[str, str, str, int]] = []
    for label, subtree in [("indexer.py", tree)] + [
        (p.name, t) for p, t in zip(paths, imported)
    ]:
        visitor = _Offenders(blocking)
        visitor.visit(subtree)
        hits.extend((label, fn, called, line) for fn, called, line in visitor.hits)
    return hits


def test_no_sync_db_call_runs_on_the_event_loop():
    offenders = _offenders()
    assert not offenders, "\n".join(
        f"  async {fn}() calls {called}() directly at {where}:{line} — "
        "wrap it in asyncio.to_thread"
        for where, fn, called, line in offenders
    )


def test_guard_reaches_across_imports():
    """The import crawl must actually resolve worker modules.

    If it silently resolved nothing, the guard would still pass on a
    clean tree while covering none of the imported helpers — the
    failure mode that let the previous guard miss a whole shape.
    """
    resolved = _imported_app_modules(_module())
    names = {p.name for p in resolved}
    assert "metadata.py" in names, (
        f"worker modules not resolved from imports: {sorted(names)}"
    )

    imported = [
        ast.parse(p.read_text(), str(p)) for p in resolved
    ]
    blocking = _sync_db_helpers(_module(), imported)
    assert "index_text_content" in blocking, (
        "an imported helper that opens a session was not classified as "
        "blocking, so calling it on the loop would go unnoticed"
    )


# --- the detector itself ---------------------------------------------------
#
# A guard is worth exactly its coverage, and the last one silently
# missed a whole shape. These pin what this one does and does not catch.


def _detect(source: str) -> list[tuple[str, str, int]]:
    tree = ast.parse(source)
    visitor = _Offenders(_sync_db_helpers(tree) | _SESSION_FACTORIES)
    visitor.visit(tree)
    return visitor.hits


@pytest.mark.parametrize(
    "source",
    [
        # Session factory opened straight from an async body.
        "async def f():\n"
        "    with get_search_db() as s:\n"
        "        s.query(X).all()\n",
        # Read-only factory is just as blocking.
        "async def f():\n"
        "    with get_search_db_read() as s:\n"
        "        s.query(X).all()\n",
        # The core DB counts too.
        "async def f():\n"
        "    with get_litloft_db() as s:\n"
        "        s.query(X).all()\n",
        # Module-level helper that opens a session, called from async.
        "def _helper():\n"
        "    with get_search_db() as s:\n"
        "        return s.query(X).all()\n"
        "async def f():\n"
        "    return _helper()\n",
        # Method call spelling.
        "def _helper():\n"
        "    with get_search_db() as s:\n"
        "        return s\n"
        "class C:\n"
        "    async def f(self):\n"
        "        return _helper()\n",
        # Transitive: the helper never names a session factory itself,
        # it just calls one that does (_purge_files -> _purge_file).
        "def _one():\n"
        "    with get_search_db() as s:\n"
        "        return s\n"
        "def _many(ids):\n"
        "    for i in ids:\n"
        "        _one(i)\n"
        "async def f(ids):\n"
        "    return _many(ids)\n",
        # Two hops.
        "def _one():\n"
        "    with get_search_db() as s:\n"
        "        return s\n"
        "def _mid():\n"
        "    return _one()\n"
        "def _outer():\n"
        "    return _mid()\n"
        "async def f():\n"
        "    return _outer()\n",
        # The offload was removed: the closure is now called inline.
        "async def f():\n"
        "    def _inner():\n"
        "        with get_search_db() as s:\n"
        "            return s\n"
        "    return _inner()\n",
    ],
)
def test_detector_catches_blocking_shapes(source):
    assert _detect(source), "detector missed a blocking shape"


@pytest.mark.parametrize(
    "source",
    [
        # The fix: sync closure handed to a worker thread.
        "def _outer():\n"
        "    with get_search_db() as s:\n"
        "        return s\n"
        "async def f():\n"
        "    return await asyncio.to_thread(_outer)\n",
        # Nested closure defined inside the async function, run off-loop.
        "async def f():\n"
        "    def _inner():\n"
        "        with get_search_db() as s:\n"
        "            return s\n"
        "    return await asyncio.to_thread(_inner)\n",
        # Plain sync function is free to block; it is not on the loop.
        "def f():\n"
        "    with get_search_db() as s:\n"
        "        return s\n",
        # Unrelated call that merely shares a prefix.
        "async def f():\n"
        "    return other.get_search_db_stats()\n",
        # A helper that only *mentions* a factory in prose is not blocking.
        'def _helper():\n'
        '    """Unlike get_search_db(), this touches nothing."""\n'
        '    return 1\n'
        'async def f():\n'
        '    return _helper()\n',
        # Awaiting an async function that offloads correctly.
        "def _inner():\n"
        "    with get_search_db() as s:\n"
        "        return s\n"
        "async def b():\n"
        "    return await asyncio.to_thread(_inner)\n"
        "async def f():\n"
        "    return await b()\n",
        # Passing the helper as an argument is not calling it.
        "def _inner():\n"
        "    with get_search_db() as s:\n"
        "        return s\n"
        "async def f():\n"
        "    return await asyncio.to_thread(_inner, 1, key=2)\n",
    ],
)
def test_detector_allows_non_blocking_shapes(source):
    assert not _detect(source), "detector flagged a safe shape"
