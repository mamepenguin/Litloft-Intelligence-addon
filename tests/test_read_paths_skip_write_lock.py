"""A read request never waits on the search-DB write lock.

Refine holds ``get_search_db()``'s lock across a WhisperX alignment and
reconcile across its writes, so a read that takes the lock stalls the
screen that issued it for as long as the writer runs. Reads go through
``get_search_db_read()``, which takes no lock.
"""

from __future__ import annotations

import ast

import pytest

from tests.test_event_loop_discipline import (
    _APP,
    _Module,
    _module_name,
    _served_files,
)

Key = tuple[str, str]

_OFFLOADERS = {"to_thread", "run_in_threadpool", "run_in_executor"}

# POST routes that only read: the body carries the query.
_READ_POSTS = {"/ask", "/find"}


def _own_nodes(fn: ast.AST):
    todo = list(ast.iter_child_nodes(fn))
    while todo:
        node = todo.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            todo.extend(ast.iter_child_nodes(node))


def _resolve_ref(mod: _Module, defs: set[Key], expr: ast.AST) -> Key | None:
    if isinstance(expr, ast.Name):
        if (mod.name, expr.id) in defs:
            return (mod.name, expr.id)
        return mod.names.get(expr.id)
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        base = expr.value.id
        if base in ("self", "cls"):
            return (mod.name, expr.attr)
        if base in mod.modules:
            return (mod.modules[base], expr.attr)
    return None


def _callee_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _edges(mod: _Module, defs: set[Key], fn: ast.AST) -> tuple[bool, set[Key]]:
    """Whether ``fn`` opens a write session itself, and what it reaches.

    A function handed to ``to_thread`` still runs for this request, so it
    counts as reached.
    """
    opens = False
    reached: set[Key] = set()
    for node in _own_nodes(fn):
        if not isinstance(node, ast.Call):
            continue
        name = _callee_name(node)
        if name == "get_search_db":
            opens = True
            continue
        target = _resolve_ref(mod, defs, node.func)
        if target is not None:
            reached.add(target)
        if name in _OFFLOADERS:
            for arg in node.args:
                ref = _resolve_ref(mod, defs, arg)
                if ref is not None:
                    reached.add(ref)
    return opens, reached


def _is_read_route(fn: ast.AST) -> bool:
    for dec in getattr(fn, "decorator_list", []):
        if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)):
            continue
        if dec.func.attr == "get":
            return True
        path = dec.args[0].value if dec.args and isinstance(dec.args[0], ast.Constant) else None
        if dec.func.attr == "post" and path in _READ_POSTS:
            return True
    return False


def _read_routes_reaching_write_lock(
    sources: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    if sources is None:
        modules = [
            _Module(_module_name(p), ast.parse(p.read_text()), str(p.relative_to(_APP)))
            for p in _served_files()
        ]
    else:
        modules = [_Module(n, ast.parse(s), n) for n, s in sources.items()]
    known = {m.name for m in modules}
    for m in modules:
        m.bind_imports(known)

    fns: dict[Key, list[tuple[_Module, ast.AST]]] = {}
    for m in modules:
        for node in ast.walk(m.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fns.setdefault((m.name, node.name), []).append((m, node))
    defs = set(fns)

    opens: set[Key] = set()
    calls: dict[Key, set[Key]] = {}
    for key, entries in fns.items():
        calls[key] = set()
        for m, node in entries:
            o, r = _edges(m, defs, node)
            if o:
                opens.add(key)
            calls[key] |= r

    reaching = set(opens)
    changed = True
    while changed:
        changed = False
        for key, targets in calls.items():
            if key not in reaching and targets & reaching:
                reaching.add(key)
                changed = True

    return sorted(
        (m.label, node.name)
        for key, entries in fns.items()
        for m, node in entries
        if _is_read_route(node) and key in reaching
    )


def test_no_read_route_takes_the_write_lock():
    offenders = _read_routes_reaching_write_lock()
    assert not offenders, "\n".join(
        f"  {where}:{fn}() reaches get_search_db(); read with "
        "get_search_db_read()"
        for where, fn in offenders
    )


def test_guard_sees_read_routes():
    """A guard that recognised no route would pass on any tree."""
    found = sum(
        1
        for p in _served_files()
        for node in ast.walk(ast.parse(p.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _is_read_route(node)
    )
    assert found >= 20


_ROUTE = "@router.get('/x')\n"


@pytest.mark.parametrize(
    "sources",
    [
        {"app.m": _ROUTE + "def f():\n    with get_search_db() as s:\n        return s\n"},
        {"app.m": _ROUTE + "async def f():\n"
                  "    def _read():\n"
                  "        with get_search_db() as s:\n"
                  "            return s\n"
                  "    return await asyncio.to_thread(_read)\n"},
        {"app.h": "def helper():\n    with get_search_db() as s:\n        return s\n",
         "app.m": "from app.h import helper\n" + _ROUTE
                  + "async def f():\n    return await asyncio.to_thread(helper)\n"},
        {"app.h": "def inner():\n    with get_search_db() as s:\n        return s\n"
                  "def helper():\n    return inner()\n",
         "app.m": "import app.h as h\n" + _ROUTE + "def f():\n    return h.helper()\n"},
        {"app.m": "@router.post('/ask')\n"
                  "def f():\n    with get_search_db() as s:\n        return s\n"},
    ],
    ids=["direct", "offloaded-closure", "offloaded-import", "two-hops", "read-post"],
)
def test_detector_catches_a_read_reaching_the_lock(sources):
    assert _read_routes_reaching_write_lock(sources)


@pytest.mark.parametrize(
    "sources",
    [
        {"app.m": _ROUTE + "def f():\n    with get_search_db_read() as s:\n        return s\n"},
        {"app.m": "@router.post('/x')\n"
                  "def f():\n    with get_search_db() as s:\n        return s\n"},
    ],
    ids=["read-session", "a-write-route"],
)
def test_detector_allows_reads_and_writes_elsewhere(sources):
    assert not _read_routes_reaching_write_lock(sources)
