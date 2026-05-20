"""RED-phase tests for the admin ``/embedding`` endpoint (Plan Phase 3).

Spec: ``docs/superpowers/specs/2026-05-20-gui-text-embedding-model.md`` §3.4
Plan: ``...-plan.md`` Phase 3.

These tests are written **before** the endpoint exists, so the whole
file is expected to FAIL (RED): the ``/admin/embedding`` routes are
not registered yet, so every request 404s.

Conventions are copied verbatim from the existing
``/admin/transcription`` tests in ``test_admin_router.py``:

* the ``admin_app`` / ``client`` fixtures reload ``app.routers.admin``
  with ``INTELLIGENCE_DATA_DIR`` pointed at ``tmp_path`` so the
  embedding-overrides file lands in a throwaway dir;
* ``restart_pending`` is verified by monkeypatching the module-level
  ``_notify_core_restart_pending`` coroutine with an ``AsyncMock``
  (``_ok_notify``) and asserting ``core_notified == "ok"`` /
  ``.assert_awaited_once()`` — the SAME hook the transcription PUT /
  DELETE tests assert against;
* auth parity: the route is gated by the host addon proxy
  (``pre_check: {type: admin}``); the router itself implements no
  auth. Mirroring the transcription tests, every request here is
  sent WITHOUT an Authorization header and must still reach the
  handler. A 401/403 would mean the implementer wrongly invented a
  per-route auth scheme that ``/transcription`` does not have.

The recorded model lives in the search DB
(``index_meta['text_embedding_model']``). Group A exposes
``app.database._read_index_meta`` / ``_TEXT_EMBEDDING_MODEL_KEY``; the
GET handler is expected to read it through a read-only search session.
These tests stub ``app.database._read_index_meta`` (the documented
Group A reader) plus a fake ``get_search_db_read`` session so no real
search.db is required, and assert only the response shape.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub heavy ML deps before importing anything that pulls them in
# (mirrors test_admin_router.py / conftest).
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

from fastapi.testclient import TestClient

import app.config as config
from app.embedding_overrides import read_overrides, write_overrides
from app.workers.embedder import _MODEL_DIMS

# Baseline default from config.ModelConfig.text_embedding.
_BASELINE_MODEL = "intfloat/multilingual-e5-small"
_E5_BASE = "intfloat/multilingual-e5-base"      # 768, "multi"
_RURI_30 = "cl-nagoya/ruri-v3-30m"              # 256, "ja"


# ---------------------------------------------------------------------------
# Fixtures — copied from test_admin_router.py so the conventions match
# ---------------------------------------------------------------------------


@pytest.fixture()
def admin_app(tmp_path, monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path))
    # Reload the router module so the env var is picked up.
    sys.modules.pop("app.routers.admin", None)
    from fastapi import FastAPI

    from app.routers import admin
    app = FastAPI()
    app.include_router(admin.router)
    return app, tmp_path


@pytest.fixture()
def client(admin_app):
    app, _ = admin_app
    return TestClient(app)


def _ok_notify():
    """Identical to the transcription tests' restart_pending stub."""
    return AsyncMock(return_value="ok")


@pytest.fixture()
def stub_recorded(monkeypatch):
    """Stub the search-DB recorded-model read.

    Returns a setter ``set_recorded(value | None)``. The GET handler
    is expected to obtain ``index_meta['text_embedding_model']`` via
    the Group A reader ``app.database._read_index_meta`` inside a
    read-only search session. We patch both ``get_search_db_read``
    (so no real search.db is needed) and ``_read_index_meta`` (the
    documented Group A seam) at the ``app.database`` module boundary.
    """
    import app.database as database_mod

    state: dict[str, str | None] = {"recorded": None}

    @contextmanager
    def _fake_read_session():
        session = MagicMock()
        session.connection.return_value = MagicMock()
        yield session

    def _fake_read_index_meta(_conn, key):
        from app.database import _TEXT_EMBEDDING_MODEL_KEY

        if key == _TEXT_EMBEDDING_MODEL_KEY:
            return state["recorded"]
        return None

    monkeypatch.setattr(
        database_mod, "get_search_db_read", _fake_read_session, raising=False
    )
    monkeypatch.setattr(
        database_mod, "_read_index_meta", _fake_read_index_meta,
        raising=False,
    )

    def _set(value: str | None) -> None:
        state["recorded"] = value

    return _set


# ---------------------------------------------------------------------------
# 1. GET — effective model + catalog shape
# ---------------------------------------------------------------------------


def test_get_returns_effective_baseline_and_catalog(
    client, stub_recorded
) -> None:
    """GET with no overrides → effective == the search-config.yml
    baseline; ``catalog`` is non-empty and every entry has
    ``id/family/dim/weight`` with ``id`` a key of ``_MODEL_DIMS``,
    ``family`` in {"ja","multi"}, and ``dim == _MODEL_DIMS[id]``.

    Auth parity: no Authorization header is sent (just like every
    ``/admin/transcription`` test) and the call must still 200.
    """
    stub_recorded(None)
    response = client.get("/admin/embedding")
    assert response.status_code == 200
    body = response.json()

    assert body["effective"] == config.settings.models.text_embedding
    assert body["effective"] == _BASELINE_MODEL

    catalog = body["catalog"]
    assert isinstance(catalog, list)
    assert len(catalog) == len(_MODEL_DIMS)
    assert len(catalog) > 0

    seen_ids = set()
    for entry in catalog:
        assert set(("id", "family", "dim", "weight")) <= set(entry)
        assert entry["id"] in _MODEL_DIMS
        assert entry["dim"] == _MODEL_DIMS[entry["id"]]
        assert entry["family"] in {"ja", "multi"}
        # ruri == Japanese-specialised, everything else multilingual.
        expected_family = "ja" if "ruri" in entry["id"] else "multi"
        assert entry["family"] == expected_family
        assert entry["weight"] in {"light", "normal", "heavy"}
        seen_ids.add(entry["id"])

    assert seen_ids == set(_MODEL_DIMS)


def test_get_catalog_weight_is_dim_size_hint(client, stub_recorded) -> None:
    """``weight`` is a size hint derived from the embedding dim:
    dim<=384 "light", <=768 "normal", else "heavy"."""
    stub_recorded(None)
    body = client.get("/admin/embedding").json()
    by_id = {e["id"]: e for e in body["catalog"]}

    for model_id, dim in _MODEL_DIMS.items():
        if dim <= 384:
            expected = "light"
        elif dim <= 768:
            expected = "normal"
        else:
            expected = "heavy"
        assert by_id[model_id]["weight"] == expected, (
            f"{model_id} (dim {dim}) should be {expected!r}"
        )


def test_get_reflects_saved_override_before_restart(
    client, admin_app, stub_recorded
) -> None:
    """Read-after-write contract identical to the other admin GETs:
    a freshly saved override is the effective model immediately,
    before the container restart swaps the frozen
    ``config.settings``."""
    _app, data_dir = admin_app
    stub_recorded(None)
    write_overrides_path = write_overrides
    write_overrides_path(
        __import__(
            "app.embedding_overrides", fromlist=["EmbeddingOverrides"]
        ).EmbeddingOverrides(text_embedding=_E5_BASE),
        data_dir=data_dir,
    )
    body = client.get("/admin/embedding").json()
    assert body["effective"] == _E5_BASE


# ---------------------------------------------------------------------------
# 2. GET — recorded / reindex_pending
# ---------------------------------------------------------------------------


def test_get_recorded_null_and_no_reindex_when_unset(
    client, stub_recorded
) -> None:
    """index_meta unset → recorded is null and reindex_pending false
    (a fresh / never-indexed DB is not 'pending')."""
    stub_recorded(None)
    body = client.get("/admin/embedding").json()
    assert body["recorded"] is None
    assert body["reindex_pending"] is False


def test_get_reindex_pending_true_when_recorded_differs(
    client, stub_recorded
) -> None:
    """recorded set to a model different from the effective baseline
    → reindex_pending true (a restart will rebuild vec_text)."""
    stub_recorded(_E5_BASE)  # effective baseline is e5-small
    body = client.get("/admin/embedding").json()
    assert body["recorded"] == _E5_BASE
    assert body["effective"] == _BASELINE_MODEL
    assert body["reindex_pending"] is True


def test_get_reindex_pending_false_when_recorded_equals_effective(
    client, stub_recorded
) -> None:
    """recorded == effective → nothing to rebuild, reindex_pending
    false."""
    stub_recorded(_BASELINE_MODEL)
    body = client.get("/admin/embedding").json()
    assert body["recorded"] == _BASELINE_MODEL
    assert body["reindex_pending"] is False


# ---------------------------------------------------------------------------
# 3. PUT — allowlisted model persists + restart_pending touched
# ---------------------------------------------------------------------------


def test_put_persists_allowlisted_model_and_touches_restart(
    client, admin_app, monkeypatch, stub_recorded
) -> None:
    """A model in ``set(_MODEL_DIMS)`` → 200, override written
    (asserted via ``embedding_overrides.read_overrides``), and the
    SAME ``_notify_core_restart_pending`` hook the transcription PUT
    test asserts is awaited exactly once."""
    _app, data_dir = admin_app
    stub_recorded(None)
    from app.routers import admin as admin_module

    notify = _ok_notify()
    monkeypatch.setattr(
        admin_module, "_notify_core_restart_pending", notify
    )

    response = client.put(
        "/admin/embedding",
        json={"text_embedding": _E5_BASE},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["core_notified"] == "ok"
    notify.assert_awaited_once()

    persisted = read_overrides(data_dir)
    assert persisted is not None
    assert persisted.text_embedding == _E5_BASE


def test_put_response_reports_effective_recorded_reindex(
    client, admin_app, monkeypatch, stub_recorded
) -> None:
    """After a successful PUT the response surfaces the updated
    effective model and the reindex_pending flag (recorded is still
    the old/none value until the next restart rebuilds vec_text)."""
    _app, data_dir = admin_app
    stub_recorded(None)
    from app.routers import admin as admin_module

    monkeypatch.setattr(
        admin_module, "_notify_core_restart_pending", _ok_notify()
    )
    response = client.put(
        "/admin/embedding",
        json={"text_embedding": _RURI_30},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["effective"] == _RURI_30
    assert body["reindex_pending"] is True


# ---------------------------------------------------------------------------
# 4. PUT — model NOT in the allowlist → 422, nothing written/touched
# ---------------------------------------------------------------------------


def test_put_rejects_non_allowlisted_model(
    client, admin_app, monkeypatch, stub_recorded
) -> None:
    """A free-text / typo model → HTTP 422. NO override file is
    written and the restart_pending hook is NOT awaited (structural
    prevention of the silent dim-384 fallback incident, §2.1-4)."""
    _app, data_dir = admin_app
    stub_recorded(None)
    from app.routers import admin as admin_module

    notify = _ok_notify()
    monkeypatch.setattr(
        admin_module, "_notify_core_restart_pending", notify
    )

    response = client.put(
        "/admin/embedding",
        json={"text_embedding": "foo/bar-not-real"},
    )
    assert response.status_code == 422
    assert read_overrides(data_dir) is None
    notify.assert_not_awaited()


# ---------------------------------------------------------------------------
# 5. PUT — missing / empty text_embedding → 422
# ---------------------------------------------------------------------------


def test_put_missing_text_embedding_is_422(
    client, admin_app, monkeypatch, stub_recorded
) -> None:
    _app, data_dir = admin_app
    stub_recorded(None)
    from app.routers import admin as admin_module

    notify = _ok_notify()
    monkeypatch.setattr(
        admin_module, "_notify_core_restart_pending", notify
    )

    response = client.put("/admin/embedding", json={})
    assert response.status_code == 422
    assert read_overrides(data_dir) is None
    notify.assert_not_awaited()


def test_put_empty_text_embedding_is_422(
    client, admin_app, monkeypatch, stub_recorded
) -> None:
    _app, data_dir = admin_app
    stub_recorded(None)
    from app.routers import admin as admin_module

    notify = _ok_notify()
    monkeypatch.setattr(
        admin_module, "_notify_core_restart_pending", notify
    )

    response = client.put(
        "/admin/embedding",
        json={"text_embedding": ""},
    )
    assert response.status_code == 422
    assert read_overrides(data_dir) is None
    notify.assert_not_awaited()


# ---------------------------------------------------------------------------
# 6. DELETE — override removed + restart_pending touched
# ---------------------------------------------------------------------------


def test_delete_removes_override_and_touches_restart(
    client, admin_app, monkeypatch, stub_recorded
) -> None:
    """DELETE drops the embedding-overrides file (baseline takes back
    over on the next restart) and notifies the core via the SAME
    restart_pending hook the transcription DELETE test asserts."""
    _app, data_dir = admin_app
    stub_recorded(None)
    from app.embedding_overrides import EmbeddingOverrides

    write_overrides(
        EmbeddingOverrides(text_embedding=_E5_BASE),
        data_dir=data_dir,
    )
    assert read_overrides(data_dir) is not None

    from app.routers import admin as admin_module

    notify = _ok_notify()
    monkeypatch.setattr(
        admin_module, "_notify_core_restart_pending", notify
    )

    response = client.delete("/admin/embedding")
    assert response.status_code == 200
    assert read_overrides(data_dir) is None
    notify.assert_awaited_once()


# ---------------------------------------------------------------------------
# 7. Auth parity with /transcription
# ---------------------------------------------------------------------------


def test_auth_parity_no_authorization_header_required(
    client, admin_app, monkeypatch, stub_recorded
) -> None:
    """The ``/transcription`` endpoints implement NO in-router auth —
    the route is gated by the host addon proxy (``pre_check:
    {type: admin}``). The new ``/embedding`` routes must use the
    exact same model: every verb reachable WITHOUT an Authorization
    header (never 401/403), exactly like the existing transcription
    endpoint tests, which never send credentials.
    """
    _app, data_dir = admin_app
    stub_recorded(None)
    from app.routers import admin as admin_module

    monkeypatch.setattr(
        admin_module, "_notify_core_restart_pending", _ok_notify()
    )

    # No headers passed anywhere — mirrors test_admin_router.py.
    get_resp = client.get("/admin/embedding")
    assert get_resp.status_code not in (401, 403)
    assert get_resp.status_code == 200

    put_resp = client.put(
        "/admin/embedding", json={"text_embedding": _E5_BASE}
    )
    assert put_resp.status_code not in (401, 403)

    del_resp = client.delete("/admin/embedding")
    assert del_resp.status_code not in (401, 403)
