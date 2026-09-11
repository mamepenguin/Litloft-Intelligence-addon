r"""The eval harnesses are excluded from the coverage denominator on the claim
that nothing this addon ships imports them. This holds that claim.

`.coveragerc` drops `app/evals/`, `app/evals_citations/` and
`app/evals_transcription/` from the measured population, and
`scripts/check-coverage-population.py` drops the same three from the walk it
compares the report against. Those two agree with each other by construction —
that is the point of declaring the list twice — and they go on agreeing however
the import graph changes. So neither can notice the day `app/routers/foo.py`
starts importing `app.evals.stages`: the exclusion stays consistent, the floor
stays green, and only the sentence justifying the exclusion becomes false.

That sentence is what this file tests. It is the reason the exclusion is
allowed to exist, so it is held by something that fails.

The scan is over the source, not over anything the coverage run produced. A
detector built from the report would inherit the same blindness for the same
reason.

One thing to know before trusting a grep instead of this: relative imports
(`from ..evals import config`) resolve against the importing module's package,
so what they mean depends on where the file sits, and a package's `__init__.py`
resolves one level differently from a plain module beside it. This package is
absolute-imports throughout — `grep -rnE '^\s*from \.+' app/` over the shipped
trees returns nothing — so that has never mattered here in practice. It is
worth stating which of those two facts is doing the work: **the scan below
handles relative imports because the guarantee has to hold for code nobody has
written yet, not because the tree contains any.** A gap in it would have been a
gap in the guarantee while the grep stayed at zero, which is exactly the kind of
defect that survives review.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app"

# The trees excluded from the denominator. Declared here, in `.coveragerc`, and
# in `scripts/check-coverage-population.py`. Written out rather than globbed:
# `app/evals*` would take in a fourth tree nobody decided to exclude.
EXCLUDED_TREES = ("evals", "evals_citations", "evals_transcription")


def _imports_of(source: str, module: str) -> set[str]:
    """Every `app.evals*` module `source` imports, as `module` would see it.

    `module` is the dotted name of the file being read (`app.routers.files`),
    which is what a relative import resolves against.
    """
    found: set[str] = set()
    excluded = {f"app.{name}" for name in EXCLUDED_TREES}

    def note(name: str) -> None:
        for target in excluded:
            if name == target or name.startswith(f"{target}."):
                found.add(name)

    # `parts[:-1]` for every module, including a package's `__init__.py`.
    # `_shipped_modules` names that one `app.rag.__init__`, so dropping the last
    # segment already yields its package, `app.rag` — the thing a relative
    # import resolves against. A special case here reads as though `__init__`
    # needed one and silently resolved `from ..evals` in `app/rag/__init__.py`
    # to `app.rag.evals`, which matches nothing.
    package = module.split(".")[:-1]

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                note(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                prefix = package[: len(package) - node.level + 1]
                stem = ".".join(prefix + ([node.module] if node.module else []))
            else:
                stem = node.module or ""
            note(stem)
            for alias in node.names:
                note(f"{stem}.{alias.name}" if stem else alias.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # `importlib.import_module("app.evals.stages")` and friends.
            note(node.value)
    return found


def _shipped_modules() -> list[tuple[str, Path]]:
    """Every `.py` under `app/` that is NOT in an excluded tree."""
    out = []
    for path in sorted(APP.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(APP)
        if rel.parts[0] in EXCLUDED_TREES:
            continue
        dotted = "app." + ".".join(rel.with_suffix("").parts)
        out.append((dotted, path))
    return out


def test_no_shipped_module_imports_an_eval_harness():
    offenders = {}
    for dotted, path in _shipped_modules():
        hits = _imports_of(path.read_text(encoding="utf-8"), dotted)
        if hits:
            offenders[str(path.relative_to(APP.parent))] = sorted(hits)

    assert offenders == {}, (
        "These modules ship and import an eval harness:\n"
        + "\n".join(f"  {k} -> {', '.join(v)}" for k, v in offenders.items())
        + "\n\nThe harnesses are excluded from the coverage denominator in"
        " `.coveragerc` on the claim that shipped code does not reach them."
        " Either drop the import, or stop excluding that tree and re-measure"
        " the floor."
    )


def test_the_scan_covers_the_modules_it_claims_to():
    """A scan that silently finds nothing would pass the test above.

    So: the excluded trees must still be there under the names the exclusion
    uses, and the scan must reach the module most obviously part of the
    service.
    """
    for name in EXCLUDED_TREES:
        tree = APP / name
        assert tree.is_dir(), f"app/{name}/ is gone — .coveragerc excludes a path that no longer exists"
        assert list(tree.rglob("*.py")), f"app/{name}/ holds no Python"

    scanned = {dotted for dotted, _ in _shipped_modules()}
    assert "app.main" in scanned
    assert not any(d.startswith(f"app.{n}") for d in scanned for n in EXCLUDED_TREES)


@pytest.mark.parametrize(
    "module, source",
    [
        # Absolute, where the importing module cannot matter.
        ("app.routers.files", "import app.evals.stages"),
        ("app.routers.files", "from app.evals import stages"),
        ("app.routers.files", "from app.evals.stages import run"),
        ("app.routers.files", "from app import evals"),
        ("app.routers.files", "import app.evals_citations.runner"),
        ("app.routers.files", "from app.evals_transcription import metrics"),
        ("app.routers.files", 'importlib.import_module("app.evals.stages")'),
        # Relative, from a plain module...
        ("app.routers.files", "from ..evals import stages"),
        ("app.routers.files", "from ..evals.stages import run"),
        ("app.config", "from .evals import stages"),
        # ...including with no module name at all, where the target is only in
        # `node.names`. `from . import service` is the ordinary way a barrel
        # `__init__.py` is written, so this is the common spelling on the axis
        # below rather than an exotic one.
        ("app.routers.files", "from .. import evals"),
        # ...and from a package's __init__.py, which resolves one level
        # differently. These are the cases the list first went without, and
        # `app/rag/__init__.py` with `from ..evals import config` is a working
        # import of a module that exists.
        ("app.__init__", "from .evals import config"),
        ("app.rag.__init__", "from ..evals import config"),
        ("app.rag.__init__", "from ..evals.stages import run"),
        ("app.routers.__init__", "from ..evals import config"),
        ("app.rag.tools.__init__", "from ...evals import config"),
        ("app.__init__", "from . import evals"),
    ],
)
def test_the_scan_detects_an_import(module, source):
    """Every cell of the matrix below, confirmed to be caught.

    Without this, the test above asserts that a function returns empty — which
    it would also do if it had stopped looking.

    **This is not "every spelling an import can take".** That phrasing invites
    the question "are you sure?" and cannot answer it, because the set of
    things a reader can imagine is not closed. What is closed, and what this
    holds, is narrower and checkable:

        no import of an eval harness that is visible in the source tree.

    The matrix has three axes and they are exhaustive for that claim:

    1. **Statement.** The grammar has exactly two import statements — read out
       of it rather than recalled: `[c.__name__ for c in ast.stmt.__subclasses__()
       if "Import" in c.__name__]` is `['Import', 'ImportFrom']`.
    2. **Shape of `ImportFrom`.** `node.level` is zero or not (absolute or
       relative), and `node.module` is present or not (`from ..evals import x`
       against `from .. import evals`) — and when it is absent the target name
       is only in `node.names`, which is a separate branch of the resolution.
    3. **Importing file.** A plain module or a package's `__init__.py`, because
       a relative import resolves against the importer's package and those two
       differ by one level.

    Three things deliberately outside it, so their absence is not a gap:

    - **Aliases.** `import app.evals.stages as s` is the `Import` cell already;
      the scan keys on `alias.name`, the module path, never on the bound name,
      so `as` cannot hide a target.
    - **Position.** Nested in a function, a class, a `try`, an `if` — covered by
      construction, since the scan is `ast.walk` over the whole module rather
      than a pass over top-level statements.
    - **Namespace packages.** A directory with no `__init__.py` is covered by
      construction too: `_shipped_modules` enumerates from paths, not from what
      is importable.

    And one thing genuinely out of scope, for a reason rather than by omission:
    a module name assembled at run time — `exec`, string concatenation, an
    `importlib` call on a computed name. Those are not spellings this refuses;
    they are names that do not exist until the process runs, so no reader of the
    source can see them either. `.coveragerc`'s exclusion rests on what the
    source says, and so does this.
    """
    assert _imports_of(source, module), f"not detected: {source!r}"


@pytest.mark.parametrize(
    "module, source",
    [
        ("app.routers.files", "from app.rag import service"),
        ("app.routers.files", "import app.search"),
        ("app.routers.files", "from app.evaluation import thing"),
        ("app.routers.files", 'log.info("evals are dev-time only")'),
    ],
)
def test_the_scan_does_not_fire_on_neighbours(module, source):
    """`app.evaluation` is not `app.evals`, and prose about evals is not an import."""
    assert not _imports_of(source, module), f"false positive: {source!r}"
