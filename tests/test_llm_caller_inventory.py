"""Only the resolver builds LLM clients; nothing else reads the LLM config.

A caller that builds its own client, or reads ``settings.llm`` to choose a
model, sends without the per-drive ceiling. ``output_language`` stays a
global setting and is the one field anyone may read.
"""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"


def _sources():
    for path in sorted(APP.rglob("*.py")):
        if "evals" in path.relative_to(APP).parts:
            continue
        yield path.relative_to(APP.parent).as_posix(), ast.parse(path.read_text())


def _called_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _calls_to(names: set[str]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for rel, tree in _sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _called_name(node) in names:
                found.setdefault(_called_name(node), set()).add(rel)
    return found


def test_only_the_resolver_and_the_factory_construct_clients() -> None:
    assert _calls_to(
        {"create_llm_client", "LLMClient", "OllamaLLMClient", "get_llm_client"}
    ) == {
        "create_llm_client": {"app/llm_routing.py"},
        "LLMClient": {"app/llm.py"},
        "OllamaLLMClient": {"app/llm.py"},
    }


def test_nothing_reachable_bypasses_the_ceiling() -> None:
    assert _calls_to({"resolve_without_ceiling"}) == {}


def test_only_output_language_is_read_from_the_llm_settings() -> None:
    readers: set[str] = set()
    for rel, tree in _sources():
        parents = {
            child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
        }
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute) and node.attr == "llm"):
                continue
            base = node.value
            base_name = (
                base.id if isinstance(base, ast.Name)
                else base.attr if isinstance(base, ast.Attribute)
                else None
            )
            parent = parents.get(node)
            if base_name == "settings" and not (
                isinstance(parent, ast.Attribute) and parent.attr == "output_language"
            ):
                readers.add(rel)

    assert readers == {"app/llm_routing.py", "app/routers/admin.py"}


def test_only_known_modules_import_the_client_types() -> None:
    imported: dict[str, set[str]] = {}
    for rel, tree in _sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "app.llm":
                for alias in node.names:
                    if alias.name in {"LLMClient", "OllamaLLMClient", "create_llm_client"}:
                        imported.setdefault(alias.name, set()).add(rel)

    assert imported == {
        "create_llm_client": {"app/llm_routing.py"},
        "OllamaLLMClient": {"app/llm_routing.py"},
        # Type annotations only.
        "LLMClient": {
            "app/llm_routing.py",
            "app/rag/agentic.py",
            "app/workers/chapter_suggestions.py",
        },
    }


def test_nothing_reaches_into_the_resolver_or_the_eval_runner() -> None:
    reached: set[tuple[str, str]] = set()
    for rel, tree in _sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "app.evals" or module.startswith("app.evals."):
                    reached.add((rel, module))
                if module == "app.llm_routing":
                    reached.update(
                        (rel, alias.name) for alias in node.names if alias.name.startswith("_")
                    )
            elif isinstance(node, ast.Import):
                reached.update(
                    (rel, alias.name) for alias in node.names if alias.name.startswith("app.evals")
                )
            elif (
                isinstance(node, ast.Attribute)
                and node.attr.startswith("_")
                and isinstance(node.value, ast.Name)
                and node.value.id == "llm_routing"
                and rel != "app/llm_routing.py"
            ):
                reached.add((rel, node.attr))

    assert reached == set()
