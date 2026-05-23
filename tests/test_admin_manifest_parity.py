"""Regression: every ``/admin/*`` route declared in
``app/routers/admin.py`` must be paired with a ``manifest.json`` route
entry that carries ``pre_check: {"type": "admin"}``.

Background: in Group B (spec 2026-05-20-gui-text-embedding-model) the
new ``/admin/embedding`` GET/PUT/DELETE handlers were initially shipped
without their manifest entry. The host addon proxy (`addon_proxy.py`
``_match_route``) 404s any unregistered route *before* evaluating
``pre_check``, so the endpoint was unreachable from the GUI
(fail-closed). A future contributor who adds the manifest entry without
``pre_check.type == "admin"`` would silently expose a privileged write
path. This parity test makes the next drift obvious instead of subtle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock


# Stub heavy ML deps before importing the addon app (mirrors
# test_admin_router.py / test_admin_embedding_endpoint.py).
for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.routers.admin import router  # noqa: E402


_ADDON_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_PATH = _ADDON_ROOT / "manifest.json"


def _load_admin_manifest_routes() -> dict[str, dict]:
    """Return ``{path: route_dict}`` for every ``/admin/*`` route
    declared in ``manifest.json``."""
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    routes = manifest.get("proxy", {}).get("routes", [])
    return {
        entry["path"]: entry
        for entry in routes
        if isinstance(entry, dict)
        and isinstance(entry.get("path"), str)
        and entry["path"].startswith("/admin/")
    }


def _router_admin_routes() -> list[tuple[str, frozenset[str]]]:
    """Return ``[(full_path, methods)]`` for every ``/admin/*`` route
    registered on the FastAPI router."""
    out: list[tuple[str, frozenset[str]]] = []
    for route in router.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if isinstance(path, str) and path.startswith("/admin/"):
            # Strip framework-internal methods (HEAD is auto-added by
            # FastAPI alongside GET); manifest declares user-facing ones.
            visible = {m for m in methods if m in {"GET", "PUT", "POST", "DELETE", "PATCH"}}
            out.append((path, frozenset(visible)))
    return out


def test_every_admin_route_has_admin_pre_check_in_manifest() -> None:
    manifest_routes = _load_admin_manifest_routes()
    missing: list[str] = []
    bad_pre_check: list[str] = []
    method_mismatch: list[str] = []

    for path, methods in _router_admin_routes():
        entry = manifest_routes.get(path)
        if entry is None:
            missing.append(path)
            continue
        pre_check = entry.get("pre_check") or {}
        if pre_check.get("type") != "admin":
            bad_pre_check.append(f"{path} (pre_check={pre_check!r})")
        declared = set(entry.get("methods") or [])
        if not methods.issubset(declared):
            method_mismatch.append(
                f"{path} router={sorted(methods)} manifest={sorted(declared)}"
            )

    problems = []
    if missing:
        problems.append(
            "manifest.json is missing entries for admin routes: "
            + ", ".join(sorted(missing))
        )
    if bad_pre_check:
        problems.append(
            "admin routes must carry pre_check.type == 'admin': "
            + ", ".join(sorted(bad_pre_check))
        )
    if method_mismatch:
        problems.append(
            "manifest methods do not cover the registered router methods: "
            + ", ".join(sorted(method_mismatch))
        )
    assert not problems, "; ".join(problems)


def test_admin_embedding_specifically_registered_with_admin_pre_check() -> None:
    # Locks the Group B finding in by name — independent of the
    # general loop above so a regression names the exact route.
    manifest_routes = _load_admin_manifest_routes()
    entry = manifest_routes.get("/admin/embedding")
    assert entry is not None, (
        "/admin/embedding must be declared in manifest.json so the host "
        "addon proxy can route GUI calls to it (spec §3.4)."
    )
    assert (entry.get("pre_check") or {}).get("type") == "admin", (
        "/admin/embedding manifest entry must carry "
        "pre_check.type == 'admin' (spec §2.1, parity with "
        "/admin/transcription)."
    )


# ---------------------------------------------------------------------------
# Spec 2026-05-24-intelligence-reindex-controls additions
# ---------------------------------------------------------------------------


def _load_all_manifest_routes() -> list[dict]:
    """Return every entry under ``proxy.routes`` so the death-confirm
    tests can inspect non-``/admin/*`` paths as well."""
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    routes = manifest.get("proxy", {}).get("routes", [])
    return [entry for entry in routes if isinstance(entry, dict)]


def _router_all_paths() -> set[str]:
    """Return the path string of every route declared in the admin
    router. Other routers are imported lazily by ``app.routers.*``; here
    we collect from the queue router too because the death-confirm
    targets ``/queue/reindex``."""
    out: set[str] = set()
    for route in router.routes:
        path = getattr(route, "path", None)
        if isinstance(path, str):
            out.add(path)

    # The queue router is what owned ``/queue/reindex`` historically;
    # importing it lets the death-confirm test verify the handler is
    # gone from the FastAPI surface too.
    import sys as _sys
    from unittest.mock import MagicMock as _MM
    for _m in (
        "PIL", "PIL.Image",
        "open_clip",
        "torch",
        "sentence_transformers",
        "faster_whisper",
        "onnxruntime",
        "transformers",
        "janome", "janome.tokenizer",
        "sqlite_vec",
    ):
        if _m not in _sys.modules:
            _sys.modules[_m] = _MM()

    from app.routers.queue import router as queue_router  # noqa: E402

    for route in queue_router.routes:
        path = getattr(route, "path", None)
        if isinstance(path, str):
            out.add(path)
    return out


def test_queue_reindex_route_is_removed_from_manifest() -> None:
    """Death-confirm: spec §1 deletes ``POST /queue/reindex`` entirely.

    The host's addon_proxy resolves manifest paths first; leaving the
    entry behind would let "let's restore the convenience" come back
    silently. This test fires the moment someone re-adds it."""
    entries = _load_all_manifest_routes()
    paths_with_post = [
        e["path"]
        for e in entries
        if isinstance(e.get("path"), str)
        and "POST" in (e.get("methods") or [])
    ]
    assert "/queue/reindex" not in paths_with_post, (
        "spec 2026-05-24-intelligence-reindex-controls §1 removed "
        "POST /queue/reindex permanently. Re-adding it re-introduces "
        "the global reset blast-radius bug "
        "(hako WmAMUDZSsMHlutJFKsyAe)."
    )


def test_queue_reindex_handler_is_removed_from_router() -> None:
    """Death-confirm: the handler itself must also be gone (spec §1
    deletes ``queue.queue_reindex``)."""
    router_paths = _router_all_paths()
    assert "/queue/reindex" not in router_paths, (
        "queue.queue_reindex handler must be removed alongside the "
        "manifest entry (spec §1). A re-introduced handler would be "
        "reachable as soon as the manifest entry is restored."
    )


def test_reindex_all_method_is_removed_from_index_manager() -> None:
    """Death-confirm: ``IndexManager.reindex_all`` was only ever called
    by the deleted HTTP handler (spec §1 architect review). The method
    body is removed so future contributors can't wire it up to a new
    endpoint without seeing the spec rationale first."""
    import sys as _sys
    from unittest.mock import MagicMock as _MM
    for _m in (
        "PIL", "PIL.Image",
        "open_clip",
        "torch",
        "sentence_transformers",
        "faster_whisper",
        "onnxruntime",
        "transformers",
        "janome", "janome.tokenizer",
        "sqlite_vec",
    ):
        if _m not in _sys.modules:
            _sys.modules[_m] = _MM()

    from app.indexer import IndexManager  # noqa: E402

    assert not hasattr(IndexManager, "reindex_all"), (
        "IndexManager.reindex_all must be removed (spec §1). The "
        "single caller (POST /queue/reindex) is gone; restoring this "
        "method makes the dangerous global reset a one-line "
        "endpoint away."
    )


# ---------------------------------------------------------------------------
# New routes added by spec §2.1 (file_access pre_check) and §2.2 (admin
# pre_check) must be reachable.
# ---------------------------------------------------------------------------


def test_admin_failed_jobs_manifest_entry_exists() -> None:
    """Spec §2.2: ``GET /admin/failed-jobs`` is gated by
    ``pre_check.type='admin'``. The general loop above validates
    every ``/admin/*`` route, but pinning the name here makes a
    regression message point at the right line in the spec."""
    manifest_routes = _load_admin_manifest_routes()
    entry = manifest_routes.get("/admin/failed-jobs")
    assert entry is not None, (
        "/admin/failed-jobs must be declared in manifest.json "
        "(spec 2026-05-24-intelligence-reindex-controls §2.2)."
    )
    assert "GET" in (entry.get("methods") or []), (
        "/admin/failed-jobs must expose GET (spec §2.2)."
    )
    assert (entry.get("pre_check") or {}).get("type") == "admin", (
        "/admin/failed-jobs must carry pre_check.type == 'admin' "
        "(spec §2.2, parity with /admin/embedding)."
    )


def test_files_reindex_manifest_entry_exists() -> None:
    """Spec §2.1: ``POST /files/{file_id}/reindex`` must carry
    ``pre_check.type='file_access'`` so cross-drive writes are blocked
    (Internal API policy R4 mirrored at the addon-proxy layer)."""
    entries = _load_all_manifest_routes()
    matching = [
        e for e in entries
        if e.get("path") == "/files/{file_id}/reindex"
        and "POST" in (e.get("methods") or [])
    ]
    assert matching, (
        "POST /files/{file_id}/reindex must be declared in "
        "manifest.json (spec 2026-05-24-intelligence-reindex-controls "
        "§2.1)."
    )
    entry = matching[0]
    pre = entry.get("pre_check") or {}
    assert pre.get("type") == "file_access", (
        "POST /files/{file_id}/reindex must carry "
        "pre_check.type == 'file_access' to block cross-drive writes "
        "(spec §2.1)."
    )
    assert pre.get("param") == "file_id", (
        "pre_check.param must be 'file_id' so the host proxy resolves "
        "the path parameter to the right access-control subject "
        "(spec §2.1)."
    )
