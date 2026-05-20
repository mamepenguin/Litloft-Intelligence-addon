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
