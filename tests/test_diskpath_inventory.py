"""SPEC-ADDON-062 I14: every join of a drive mount with a stored path in app/
is routed through the on-disk resolver or listed with a reason.

The pure join ``config.resolve_file_path`` is also checked by call site: its
result must be wrapped by ``resolve_stored_path``, so a caller cannot store or
open the unresolved spelling.
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"

STORED = {"file_path", "folder_path", "relative_path", "rel_path"}
RESOLVERS = {"resolve_on_disk", "resolve_parent", "resolve_stored_path"}
PURE_JOIN = "resolve_file_path"
DRIVE_PART = re.compile(r"drive_path|drive_root|drive_dir|base_dir|base_path|mount|\broot\b|_root\b")

# (path relative to the addon root, enclosing function) -> why the join is not a
# stored NFC name that has to be found on disk. An entry without a reason fails.
LISTING_DERIVED: dict[tuple[str, str], str] = {
    ("app/config.py", "resolve_file_path"): (
        "the pure join itself; every caller wraps it in resolve_stored_path"
    ),
}


def _is_fn(n: ast.AST) -> bool:
    return isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))


def _mentions_stored(node: ast.AST, via: set[str]) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in STORED:
            return True
        if isinstance(n, ast.Name) and (n.id in STORED or n.id in via):
            return True
    return False


def _stored_names(fn: ast.AST) -> set[str]:
    """Names assigned once, in this function, from an expression that mentions a stored path."""
    counts: dict[str, int] = {}
    carries: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            counts[name] = counts.get(name, 0) + 1
            if _mentions_stored(node.value, set()):
                carries.add(name)
    return {n for n in carries if counts[n] == 1}


def _drive_names(fn: ast.AST) -> set[str]:
    """Names assigned once from an expression that mentions a drive mount."""
    counts: dict[str, int] = {}
    carries: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            counts[name] = counts.get(name, 0) + 1
            if DRIVE_PART.search(ast.unparse(node.value)):
                carries.add(name)
    return {n for n in carries if counts[n] == 1}


def _join_parts(node: ast.AST) -> list[ast.AST] | None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return [node.left, node.right]
    if isinstance(node, ast.JoinedStr):
        return [v.value for v in node.values if isinstance(v, ast.FormattedValue)]
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "joinpath":
            return [f.value, *node.args]
        if isinstance(f, ast.Attribute) and f.attr == "join" and ast.unparse(f.value) in {"os.path", "posixpath"}:
            return list(node.args)
        if isinstance(f, (ast.Name, ast.Attribute)) and (getattr(f, "id", None) or f.attr) in {"Path", "PurePath"} and len(node.args) > 1:
            return list(node.args)
    return None


def _call_name(call: ast.Call) -> str:
    f = call.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""


def unrouted_joins(source: str) -> list[tuple[str, int]]:
    """(enclosing function, line) of each unrouted drive/stored join or bare pure-join call."""
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for p in ast.walk(tree):
        for ch in ast.iter_child_nodes(p):
            parents[ch] = p

    def owner_of(node: ast.AST) -> ast.AST:
        owner = node
        while owner in parents and not _is_fn(owner) and not isinstance(owner, ast.Module):
            owner = parents[owner]
        return owner

    def routed(node: ast.AST, names: set[str]) -> bool:
        up = node
        while up in parents:
            up = parents[up]
            if isinstance(up, ast.Call) and _call_name(up) in names:
                return True
        return False

    found: list[tuple[str, int]] = []
    for fn in [tree, *[n for n in ast.walk(tree) if _is_fn(n)]]:
        via = _stored_names(fn)
        drive_via = _drive_names(fn)
        name = "<module>" if isinstance(fn, ast.Module) else fn.name
        for node in ast.walk(fn):
            if owner_of(node) is not fn:
                continue
            if isinstance(node, ast.Call) and _call_name(node) == PURE_JOIN and not routed(node, {"resolve_stored_path"}):
                found.append((name, node.lineno))
            parts = _join_parts(node)
            if parts is None:
                continue
            has_stored = any(_mentions_stored(p, via) for p in parts)
            has_drive = any(
                DRIVE_PART.search(ast.unparse(p))
                or any(isinstance(n, ast.Name) and n.id in drive_via for n in ast.walk(p))
                for p in parts
            )
            if has_stored and has_drive and not routed(node, RESOLVERS):
                found.append((name, node.lineno))
    return sorted(set(found))


def test_the_detector_finds_a_seeded_unrouted_site_in_every_form():
    src = textwrap.dedent('''
        import os
        def a(mount, file):
            return mount / file.file_path
        def b(mount, file):
            return os.path.join(mount, file.file_path)
        def c(mount, file):
            return mount.joinpath(file.folder_path, "x")
        def d(mount, file):
            return Path(mount, file.file_path)
        def e(mount, file):
            return f"{mount}/{file.file_path}"
        def f(mount, file):
            rel = file.file_path
            return mount / rel
        def g(drive, rel_path):
            return resolve_file_path(drive, rel_path)
        def h(drive, file):
            mount = settings.drive_mounts.get(drive)
            return Path(mount) / file.file_path
    ''')
    assert [n for n, _ in unrouted_joins(src)] == ["a", "b", "c", "d", "e", "f", "g", "h"]


def test_the_detector_ignores_routed_and_unrelated_joins():
    src = textwrap.dedent('''
        def a(mount, file):
            return resolve_on_disk(mount, file.file_path)
        def b(drive, rel_path):
            return resolve_stored_path(resolve_file_path(drive, rel_path))
        def c(mount, name):
            return mount / name
        def d(file):
            return file.folder_path + "/" + file.filename
    ''')
    assert unrouted_joins(src) == []


def test_every_join_of_a_drive_mount_with_a_stored_path_is_routed_or_listed():
    for key, reason in LISTING_DERIVED.items():
        assert reason.strip(), f"{key} is listed without a reason"
    unrouted = []
    for path in sorted(APP.rglob("*.py")):
        rel = f"app/{path.relative_to(APP).as_posix()}"
        for func, line in unrouted_joins(path.read_text(encoding="utf-8")):
            if (rel, func) not in LISTING_DERIVED:
                unrouted.append(f"{rel}:{line} in {func}")
    assert unrouted == []
