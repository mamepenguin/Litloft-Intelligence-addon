"""Status counts, admin settings and the config key for picture OCR (SPEC-ADDON-050)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from tests.ocr_fixtures import (
    SearchDB,
    build_search_db,
    patch_features,
    pillow_real,  # noqa: F401  - the usefixtures marker resolves it
    picture,
    seed_indexed_file,
)

pytestmark = pytest.mark.usefixtures("pillow_real")

ALL_DONE = dict(
    metadata_indexed=True,
    clip_indexed=True,
    clip_thumbnail_indexed=True,
    whisper_indexed=True,
    text_indexed=True,
    tfidf_keywords_indexed=True,
)


@pytest.fixture()
def search_db(tmp_path, monkeypatch) -> SearchDB:
    return build_search_db(tmp_path, monkeypatch)


def seed_mix(search_db, tmp_path) -> SearchDB:
    """One read picture, one pending picture, one picture-less document."""
    seed_indexed_file(
        search_db, "done1", "image/png", picture(tmp_path / "done1.png"), **ALL_DONE
    )
    search_db.set_flag("done1", "ocr_indexed", True)
    seed_indexed_file(
        search_db, "todo1", "image/jpeg", picture(tmp_path / "todo1.png"), **ALL_DONE
    )
    document = tmp_path / "notes.txt"
    document.write_text("plain text")
    seed_indexed_file(search_db, "doc1", "text/plain", document, **ALL_DONE)
    search_db.set_flag("doc1", "ocr_indexed", True)
    return search_db


def manage(search_db, monkeypatch):
    from app import main

    manager = main.IndexManager()
    monkeypatch.setattr(main.dependencies, "get_index_manager", lambda: manager)
    # An earlier test's lifespan can leave this worker bound to a disposed
    # search.db; /status would read through it.
    monkeypatch.setattr(main.dependencies, "_video_visual_worker", None)
    return manager, main


# ---------------------------------------------------------------------------
# I9, I10 - /status
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,indexed_ocr,pending_ocr,in_total",
    [("on_index", 1, 1, True), ("manual", 1, 0, False), ("false", 1, 0, False)],
)
def test_status_counts_pictures_by_the_mode(
    search_db, tmp_path, monkeypatch, mode, indexed_ocr, pending_ocr, in_total
):
    """SPEC-ADDON-050 I9: pending.ocr exists only while the mode is on_index."""
    seed_mix(search_db, tmp_path)
    patch_features(monkeypatch, image_ocr=mode)
    _manager, main = manage(search_db, monkeypatch)

    body = main.status_endpoint()

    assert body.features.image_ocr == mode
    assert body.indexed["ocr"] == indexed_ocr
    assert body.pending.get("ocr", 0) == pending_ocr
    other_pending = (
        body.pending["metadata"]
        + body.pending["clip"]
        + body.pending["whisper"]
        + body.pending["text"]
    )
    assert body.pending["total"] == other_pending + (pending_ocr if in_total else 0)


def test_the_queue_breakdown_reports_an_ocr_queue(search_db, tmp_path, monkeypatch):
    """SPEC-ADDON-050 I9: the dashboard sees the OCR queue like the others."""
    from app.indexer import TaskType

    seed_mix(search_db, tmp_path)
    manager, _main = manage(search_db, monkeypatch)

    assert manager.get_queue_breakdown()[TaskType.OCR.value] == {
        "waiting": 0,
        "processing": [],
    }


# ---------------------------------------------------------------------------
# I10 / settings - the admin surface and the config key
# ---------------------------------------------------------------------------


@pytest.fixture()
def admin_client(tmp_path, monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path))
    sys.modules.pop("app.routers.admin", None)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routers import admin

    app = FastAPI()
    app.include_router(admin.router)
    return TestClient(app), tmp_path


def test_admin_features_reads_and_writes_image_ocr(admin_client, monkeypatch):
    """SPEC-ADDON-050 I10: the mode is one more three-state feature setting."""
    client, data_dir = admin_client
    from app.routers import admin as admin_module

    asserted: list[str] = []

    async def _notify() -> str:
        asserted.append("notified")
        return "unconfigured"

    monkeypatch.setattr(admin_module, "_notify_core_restart_pending", _notify)

    baseline = client.get("/admin/features")
    assert baseline.status_code == 200
    assert baseline.json()["image_ocr"] == "on_index"

    saved = client.put("/admin/features", json={"image_ocr": "manual"})
    assert saved.status_code == 200
    assert saved.json()["restart_required"] is True
    assert asserted == ["notified"]

    from app.features_overrides import read_overrides

    overrides = read_overrides(data_dir=data_dir)
    assert overrides is not None
    assert overrides.image_ocr == "manual"
    assert client.get("/admin/features").json()["image_ocr"] == "manual"


def test_admin_features_rejects_a_mode_outside_the_three(admin_client):
    """SPEC-ADDON-050 I10: false / manual / on_index only."""
    client, _data_dir = admin_client

    response = client.put("/admin/features", json={"image_ocr": "sometimes"})

    assert response.status_code == 400
    assert "image_ocr" in response.json()["detail"]


def test_the_example_config_declares_image_ocr():
    """SPEC-ADDON-050 I10 / §1.7: the key is set in search-config.yml."""
    example = Path(__file__).resolve().parents[1] / "search-config.yml.example"
    config = yaml.safe_load(example.read_text())

    assert config["features"]["image_ocr"] == "on_index"
