"""No synchronous DB work may run on the event loop.

The search-DB write lock is a ``threading.Lock``. Waiting for it from a
coroutine blocks the whole loop for as long as another thread holds it,
so even an O(1) lookup stalls every endpoint while a bulk writer runs.
Inside an ``async def``, a helper that opens a DB session must go
through ``asyncio.to_thread`` (or an equivalent executor hop), never a
direct call.

The await-inside-the-lock shape is guarded separately by
``test_write_lock_discipline.py``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_APP = pathlib.Path(__file__).resolve().parent.parent / "app"

# Dev-time harnesses run as their own process, never on the service loop.
_NOT_SERVED = {_APP / "evals"}

_SESSION_FACTORIES = {
    "get_search_db",
    "get_search_db_read",
    "get_litloft_db",
}

# Known offenders waiting for their own fix. An entry is (path relative to
# app/, async function). The list may only shrink: a new offender fails,
# and so does an entry that no longer offends.
_PENDING: set[tuple[str, str]] = {
    ("rag/clue_generator.py", "generate_clues"),
    ("rag/coarse_retriever.py", "coarse_retrieve"),
    ("rag/retriever.py", "retrieve_with_keywords"),
    ("rag/service.py", "_answer_question"),
    ("rag/service.py", "_run_hierarchical_retrieval"),
    ("rag/service.py", "_stream_answer"),
    ("routers/admin.py", "get_embedding_config"),
    ("routers/admin.py", "get_failed_jobs"),
    ("routers/admin.py", "reset_embedding_config"),
    ("routers/admin.py", "resolve_failed_job"),
    ("routers/admin.py", "update_embedding_config"),
    ("routers/chapter_suggestions.py", "_require_allowed"),
    ("routers/chapter_suggestions.py", "approve_chapter_suggestions"),
    ("routers/chapter_suggestions.py", "dismiss_chapter_suggestions"),
    ("routers/chapter_suggestions.py", "generate_chapter_suggestions"),
    ("routers/chapter_suggestions.py", "get_chapter_suggestions"),
    ("routers/files.py", "batch_suggested_tags"),
    ("routers/files.py", "dismiss_suggested_tags"),
    ("routers/files.py", "get_chunk_excerpt"),
    ("routers/files.py", "get_clip_timestamps"),
    ("routers/files.py", "get_frame"),
    ("routers/files.py", "get_index_details"),
    ("routers/files.py", "get_pdf_markdown"),
    ("routers/files.py", "get_subtitles_vtt"),
    ("routers/files.py", "get_suggested_tags"),
    ("routers/files.py", "get_transcript"),
    ("routers/files.py", "regenerate_suggested_tags"),
    ("routers/files.py", "reindex_file"),
    ("routers/pickup.py", "pickup_endpoint"),
    ("routers/refine.py", "_enqueue"),
    ("routers/refine.py", "refine_file"),
    ("routers/refine.py", "refine_folder"),
    ("routers/search.py", "search_compare_endpoint"),
    ("routers/similar.py", "debug_similar_endpoint"),
    ("routers/similar.py", "similar_files_endpoint"),
    ("routers/summaries.py", "batch_summaries"),
    ("routers/summaries.py", "delete_detailed_summary_route"),
    ("routers/summaries.py", "download_detailed_summary"),
    ("routers/summaries.py", "edit_detailed_summary_section"),
    ("routers/summaries.py", "edit_summary"),
    ("routers/summaries.py", "get_detailed_summary_citations"),
    ("routers/summaries.py", "get_detailed_summary_route"),
    ("routers/summaries.py", "get_summary"),
    ("routers/summaries.py", "regenerate_detailed_summary"),
    ("routers/summaries.py", "regenerate_summary"),
    ("routers/summaries.py", "revert_detailed_summary"),
    ("routers/summaries.py", "revert_summary"),
    ("routers/summaries.py", "start_detailed_summary"),
    ("routers/video_visual.py", "generate_visual_index"),
    ("routers/video_visual.py", "get_visual_index"),
    ("routers/video_visual.py", "retry_visual_index"),
    ("routers/vision.py", "delete_visual_description"),
    ("routers/vision.py", "generate_folder_visual_description"),
    ("routers/vision.py", "generate_visual_description"),
    ("routers/vision.py", "get_visual_description"),
}

Key = tuple[str, str]


def _module_name(path: pathlib.Path) -> str:
    rel = path.relative_to(_APP.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


class _Module:
    """One parsed source file and the names its imports bind."""

    def __init__(self, name: str, tree: ast.Module, label: str) -> None:
        self.name = name
        self.tree = tree
        self.label = label
        # alias -> module name, for `import app.x as m` / `from app import x`
        self.modules: dict[str, str] = {}
        # alias -> (module, attribute), for `from app.x import f as g`
        self.names: dict[str, Key] = {}

    def bind_imports(self, known: set[str]) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in known and a.asname:
                        self.modules[a.asname] = a.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.level:
                    continue
                for a in node.names:
                    alias = a.asname or a.name
                    sub = f"{node.module}.{a.name}"
                    if sub in known:
                        self.modules[alias] = sub
                    elif node.module in known:
                        self.names[alias] = (node.module, a.name)


class _Graph:
    """Every definition across modules, keyed by (module, name).

    Nested defs and methods are keyed by their bare name inside their
    module: the offload pattern puts the blocking body in a closure, and
    a closure called directly is exactly the regression to catch.
    """

    def __init__(self, modules: list[_Module]) -> None:
        self.modules = modules
        self.opens: dict[Key, bool] = {}
        self.is_async: dict[Key, bool] = {}
        for m in modules:
            self._collect_defs(m)
        self.blocking = self._blocking()

    def resolve(self, mod: _Module, call: ast.Call) -> Key | None:
        func = call.func
        if isinstance(func, ast.Name):
            if (mod.name, func.id) in self.is_async:
                return (mod.name, func.id)
            return mod.names.get(func.id)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            base = func.value.id
            if base in ("self", "cls"):
                return (mod.name, func.attr)
            if base in mod.modules:
                return (mod.modules[base], func.attr)
        return None

    def _collect_defs(self, mod: _Module) -> None:
        for node in ast.walk(mod.tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            key = (mod.name, node.name)
            # A name defined both sync and async in one module is treated
            # as async only if every definition is.
            is_async = isinstance(node, ast.AsyncFunctionDef)
            self.is_async[key] = self.is_async.get(key, True) and is_async
            self.opens[key] = self.opens.get(key, False) or any(
                isinstance(c, ast.Call) and _factory(c)
                for c in _own_nodes(node)
            )

    def _calls(self) -> dict[Key, set[Key]]:
        out: dict[Key, set[Key]] = {k: set() for k in self.is_async}
        for mod in self.modules:
            for node in ast.walk(mod.tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    key = (mod.name, node.name)
                    for c in _own_nodes(node):
                        if isinstance(c, ast.Call):
                            target = self.resolve(mod, c)
                            if target is not None:
                                out[key].add(target)
        return out

    def _blocking(self) -> set[Key]:
        calls = self._calls()
        blocking = {
            k for k, opens in self.opens.items()
            if opens and not self.is_async.get(k, False)
        }
        # A sync helper that calls a blocking helper blocks too.
        changed = True
        while changed:
            changed = False
            for k, targets in calls.items():
                if k in blocking or self.is_async.get(k, False):
                    continue
                if targets & blocking:
                    blocking.add(k)
                    changed = True
        return blocking


def _own_nodes(fn: ast.AST):
    """Nodes in a function's body, not descending into nested defs.

    A nested def is its own key; what it calls is attributed to it, and a
    call to it is what makes the outer function block.
    """
    todo = list(ast.iter_child_nodes(fn))
    while todo:
        node = todo.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            todo.extend(ast.iter_child_nodes(node))


def _factory(call: ast.Call) -> str | None:
    func = call.func
    name = func.id if isinstance(func, ast.Name) else (
        func.attr if isinstance(func, ast.Attribute) else None
    )
    return name if name in _SESSION_FACTORIES else None


def _offenders_in(graph: _Graph, mod: _Module) -> list[tuple[str, str, int]]:
    """Blocking calls made while the innermost enclosing function is async.

    A nested ``def`` or a ``lambda`` inside an async function is not on
    the loop: that is the shape of the fix, handed to ``to_thread``.
    """
    hits: list[tuple[str, str, int]] = []
    current: list[str | None] = [None]

    class V(ast.NodeVisitor):
        def visit_AsyncFunctionDef(self, node):
            current.append(node.name)
            self.generic_visit(node)
            current.pop()

        def _sync(self, node):
            current.append(None)
            self.generic_visit(node)
            current.pop()

        visit_FunctionDef = _sync
        visit_Lambda = _sync

        def visit_Call(self, node):
            if current[-1]:
                name = _factory(node)
                if name is None:
                    target = graph.resolve(mod, node)
                    if target in graph.blocking:
                        name = target[1]
                if name:
                    hits.append((current[-1], name, node.lineno))
            self.generic_visit(node)

    V().visit(mod.tree)
    return hits


def _detect_sources(sources: dict[str, str]) -> list[tuple[str, str, str, int]]:
    """Offenders across modules given as {module name: source}."""
    known = set(sources)
    modules = [
        _Module(name, ast.parse(src), name) for name, src in sources.items()
    ]
    for m in modules:
        m.bind_imports(known)
    graph = _Graph(modules)
    return [
        (m.label, fn, called, line)
        for m in modules
        for fn, called, line in _offenders_in(graph, m)
    ]


def _served_files() -> list[pathlib.Path]:
    return sorted(
        p for p in _APP.rglob("*.py")
        if not any(ex in p.parents for ex in _NOT_SERVED)
    )


def _app_offenders() -> list[tuple[str, str, str, int]]:
    paths = _served_files()
    modules = [
        _Module(
            _module_name(p),
            ast.parse(p.read_text(), str(p)),
            str(p.relative_to(_APP)),
        )
        for p in paths
    ]
    known = {m.name for m in modules} | {
        _module_name(p.parent / "__init__.py")
        for p in paths if (p.parent / "__init__.py").exists()
    }
    for m in modules:
        m.bind_imports(known)
    graph = _Graph(modules)
    return [
        (m.label, fn, called, line)
        for m in modules
        for fn, called, line in _offenders_in(graph, m)
    ]


def test_no_sync_db_call_runs_on_the_event_loop():
    offenders = _app_offenders()
    new = [o for o in offenders if (o[0], o[1]) not in _PENDING]
    assert not new, "\n".join(
        f"  async {fn}() calls {called}() directly at {where}:{line} — "
        "wrap it in asyncio.to_thread"
        for where, fn, called, line in new
    )


def test_pending_list_only_names_live_offenders():
    live = {(where, fn) for where, fn, _, _ in _app_offenders()}
    assert not (_PENDING - live), (
        f"no longer offending, remove from _PENDING: {sorted(_PENDING - live)}"
    )


def test_guard_covers_the_service_tree():
    """A guard that silently resolved nothing would pass on any tree."""
    labels = {str(p.relative_to(_APP)) for p in _served_files()}
    for expected in ("indexer.py", "workers/summaries.py", "routers/files.py"):
        assert expected in labels
    assert not any(label.startswith("evals/") for label in labels)


# --- the detector itself ---------------------------------------------------


def _detect(source: str) -> list[tuple[str, str, str, int]]:
    return _detect_sources({"app.m": source})


@pytest.mark.parametrize(
    "source",
    [
        "async def f():\n"
        "    with get_search_db() as s:\n"
        "        s.query(X).all()\n",
        "async def f():\n"
        "    with get_search_db_read() as s:\n"
        "        s.query(X).all()\n",
        "async def f():\n"
        "    with get_litloft_db() as s:\n"
        "        s.query(X).all()\n",
        "async def f():\n"
        "    with database.get_search_db() as s:\n"
        "        s.query(X).all()\n",
        "def _helper():\n"
        "    with get_search_db() as s:\n"
        "        return s.query(X).all()\n"
        "async def f():\n"
        "    return _helper()\n",
        "class C:\n"
        "    def _helper(self):\n"
        "        with get_search_db() as s:\n"
        "            return s\n"
        "    async def f(self):\n"
        "        return self._helper()\n",
        "def _one():\n"
        "    with get_search_db() as s:\n"
        "        return s\n"
        "def _many(ids):\n"
        "    for i in ids:\n"
        "        _one(i)\n"
        "async def f(ids):\n"
        "    return _many(ids)\n",
        "def _one():\n"
        "    with get_search_db() as s:\n"
        "        return s\n"
        "def _mid():\n"
        "    return _one()\n"
        "def _outer():\n"
        "    return _mid()\n"
        "async def f():\n"
        "    return _outer()\n",
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
    "sources",
    [
        {
            "app.helpers": "def helper():\n"
                           "    with get_search_db() as s:\n"
                           "        return s\n",
            "app.m": "from app.helpers import helper\n"
                     "async def f():\n"
                     "    return helper()\n",
        },
        {
            "app.helpers": "def helper():\n"
                           "    with get_search_db() as s:\n"
                           "        return s\n",
            "app.m": "from app.helpers import helper as h\n"
                     "async def f():\n"
                     "    return h()\n",
        },
        {
            "app.helpers": "def helper():\n"
                           "    with get_search_db() as s:\n"
                           "        return s\n",
            "app.m": "import app.helpers as hp\n"
                     "async def f():\n"
                     "    return hp.helper()\n",
        },
        {
            "app.helpers": "def helper():\n"
                           "    with get_search_db() as s:\n"
                           "        return s\n",
            "app.m": "async def f():\n"
                     "    from app.helpers import helper\n"
                     "    return helper()\n",
        },
    ],
    ids=["from-import", "aliased", "module-alias", "local-import"],
)
def test_detector_follows_imports(sources):
    assert _detect_sources(sources), "detector missed a helper in another module"


@pytest.mark.parametrize(
    "source",
    [
        "def _outer():\n"
        "    with get_search_db() as s:\n"
        "        return s\n"
        "async def f():\n"
        "    return await asyncio.to_thread(_outer)\n",
        "async def f():\n"
        "    def _inner():\n"
        "        with get_search_db() as s:\n"
        "            return s\n"
        "    return await asyncio.to_thread(_inner)\n",
        "def _outer(x):\n"
        "    with get_search_db() as s:\n"
        "        return s\n"
        "async def f():\n"
        "    return await asyncio.to_thread(lambda: _outer(1))\n",
        "def f():\n"
        "    with get_search_db() as s:\n"
        "        return s\n",
        "async def f():\n"
        "    return other.get_search_db_stats()\n",
        'def _helper():\n'
        '    """Unlike get_search_db(), this touches nothing."""\n'
        '    return 1\n'
        'async def f():\n'
        '    return _helper()\n',
        "def _inner():\n"
        "    with get_search_db() as s:\n"
        "        return s\n"
        "async def b():\n"
        "    return await asyncio.to_thread(_inner)\n"
        "async def f():\n"
        "    return await b()\n",
        "def _inner():\n"
        "    with get_search_db() as s:\n"
        "        return s\n"
        "async def f():\n"
        "    return await asyncio.to_thread(_inner, 1, key=2)\n",
        # A method of an unrelated object that shares a blocking name.
        "def search():\n"
        "    with get_search_db() as s:\n"
        "        return s\n"
        "async def f(text):\n"
        "    return re.search('x', text)\n",
    ],
)
def test_detector_allows_non_blocking_shapes(source):
    assert not _detect(source), "detector flagged a safe shape"


def test_detector_keeps_same_named_functions_in_other_modules_apart():
    sources = {
        "app.a": "def resolve():\n"
                 "    with get_search_db() as s:\n"
                 "        return s\n",
        "app.b": "async def resolve():\n"
                 "    return 1\n"
                 "async def f():\n"
                 "    return await resolve()\n",
    }
    assert not _detect_sources(sources)
