"""Router endpoint tests for the vision_describe feature (RED phase).

The router lives at ``app.routers.vision`` and exposes:

* ``POST   /files/{file_id}/visual_description/generate`` — manual trigger
* ``GET    /files/{file_id}/visual_description``           — read current state
* ``DELETE /files/{file_id}/visual_description``            — clear
* ``POST   /folders/visual_description/generate``           — bulk enqueue

Access gates (in order):

1. ``features.vision_describe == "false"`` → 404 on all endpoints
2. ``llm.vision_model`` empty → 404 on generate, 200 with
   ``status="unsupported"`` on GET
3. per-drive policy OFF (``is_feature_enabled(drive, "vision_describe")``
   returns False) → 404 on all endpoints
4. file not in the requesting drive → 404 (cross-drive isolation)
5. file mime not image/* → 404 on generate
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import BackgroundTasks, HTTPException

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

from app.config import FeaturesConfig, LLMConfig  # noqa: E402

# RED phase: expected to fail at collection time until vision router exists.
pytest.importorskip(
    "app.routers.vision",
    reason="vision router not yet implemented (RED phase)",
)

from app.routers.vision import (  # noqa: E402
    delete_visual_description,
    generate_folder_visual_description,
    generate_visual_description,
    get_visual_description,
)


@pytest.fixture()
def feature_manual(monkeypatch, make_settings):
    settings = make_settings(
        features=FeaturesConfig(vision_describe="manual"),  # type: ignore[call-arg]
        llm=LLMConfig(
            provider="openai_compatible",
            base_url="http://test",
            model="gemma2:27b",
            vision_model="llava:13b",
        ),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.vision.settings", settings)

    llm_stub = MagicMock()
    llm_stub.enabled = True
    monkeypatch.setattr("app.routers.vision.get_llm_client", lambda: llm_stub)

    monkeypatch.setattr(
        "app.routers.vision.is_feature_enabled",
        AsyncMock(return_value=True),
    )
    return settings


@pytest.fixture()
def feature_off(monkeypatch, make_settings):
    settings = make_settings(
        features=FeaturesConfig(vision_describe="false"),  # type: ignore[call-arg]
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.vision.settings", settings)
    return settings


@pytest.fixture()
def stub_indexed_file(monkeypatch):
    """Stub get_search_db so file lookups return a deterministic row.

    Returns a callable so each test can tune the stored file shape
    (mime_type, drive, active, etc.).
    """
    state: dict = {"file": None}

    def _seed(
        *, file_id: str = "img-abc", drive: str = "family",
        mime_type: str = "image/jpeg", active: bool = True,
    ):
        state["file"] = MagicMock(
            file_id=file_id,
            drive=drive,
            mime_type=mime_type,
            filename="cat.jpg",
            active=active,
        )

    class _Ctx:
        def __enter__(self_inner):
            session = MagicMock()
            # Flexible enough to match both raw-SQL and ORM lookups.
            session.query.return_value.filter.return_value.first.return_value = (
                state["file"]
            )
            return session
        def __exit__(self_inner, *a):
            return False

    monkeypatch.setattr(
        "app.routers.vision.get_search_db", lambda: _Ctx()
    )
    return _seed


# ---------------------------------------------------------------------------
# POST /files/{file_id}/visual_description/generate
# ---------------------------------------------------------------------------


class TestGenerateVisualDescription:
    @pytest.mark.asyncio
    async def test_feature_off_returns_404(
        self, feature_off, stub_indexed_file,
    ):
        stub_indexed_file()
        with pytest.raises(HTTPException) as exc:
            await generate_visual_description(
                file_id="img-abc",
                background_tasks=BackgroundTasks(),
                drive="family",
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_non_image_returns_404(
        self, feature_manual, stub_indexed_file,
    ):
        stub_indexed_file(mime_type="video/mp4")
        with pytest.raises(HTTPException) as exc:
            await generate_visual_description(
                file_id="img-abc",
                background_tasks=BackgroundTasks(),
                drive="family",
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_policy_off_returns_404(
        self, feature_manual, stub_indexed_file, monkeypatch,
    ):
        stub_indexed_file()
        monkeypatch.setattr(
            "app.routers.vision.is_feature_enabled",
            AsyncMock(return_value=False),
        )
        with pytest.raises(HTTPException) as exc:
            await generate_visual_description(
                file_id="img-abc",
                background_tasks=BackgroundTasks(),
                drive="family",
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_unknown_file_returns_404(
        self, feature_manual, monkeypatch,
    ):
        class _Ctx:
            def __enter__(self_inner):
                session = MagicMock()
                session.query.return_value.filter.return_value.first.return_value = None
                return session
            def __exit__(self_inner, *a):
                return False

        monkeypatch.setattr(
            "app.routers.vision.get_search_db", lambda: _Ctx()
        )

        with pytest.raises(HTTPException) as exc:
            await generate_visual_description(
                file_id="nope",
                background_tasks=BackgroundTasks(),
                drive="family",
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_cross_drive_access_returns_404(
        self, feature_manual, stub_indexed_file,
    ):
        stub_indexed_file(drive="work")
        with pytest.raises(HTTPException) as exc:
            await generate_visual_description(
                file_id="img-abc",
                background_tasks=BackgroundTasks(),
                drive="family",
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_happy_path_schedules_background_task(
        self, feature_manual, stub_indexed_file, monkeypatch,
    ):
        stub_indexed_file()

        # Stub the worker entry point to avoid real enqueue plumbing.
        enqueue_mock = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "app.routers.vision.enqueue_visual_description",
            enqueue_mock,
            raising=False,
        )

        bg = BackgroundTasks()
        result = await generate_visual_description(
            file_id="img-abc", background_tasks=bg, drive="family",
        )

        # Response shape: accepted-style.
        status = getattr(result, "status", None)
        if status is None and isinstance(result, dict):
            status = result.get("status")
        assert status == "accepted"
        # One background task (or an eager enqueue) must have been
        # scheduled.
        assert len(bg.tasks) >= 1 or enqueue_mock.await_count >= 1


# ---------------------------------------------------------------------------
# GET /files/{file_id}/visual_description
# ---------------------------------------------------------------------------


class TestGetVisualDescription:
    @pytest.mark.asyncio
    async def test_feature_off_returns_404(
        self, feature_off, stub_indexed_file,
    ):
        stub_indexed_file()
        with pytest.raises(HTTPException) as exc:
            await get_visual_description(file_id="img-abc", drive="family")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_generated_row(
        self, feature_manual, stub_indexed_file, monkeypatch,
    ):
        stub_indexed_file()

        # Stub DB read to return a canned success row. Implementation
        # is expected to expose ``_fetch_visual_description(file_id)``
        # or similar; we patch the whole fetch surface.
        monkeypatch.setattr(
            "app.routers.vision._fetch_visual_description",
            lambda file_id: {
                "visual_description": "A cat on a mat.",
                "visual_description_status": "success",
                "visual_description_model": "llava:13b",
                "visual_description_generated_at": "2026-04-23T00:00:00Z",
            },
            raising=False,
        )

        result = await get_visual_description(
            file_id="img-abc", drive="family"
        )

        # Shape: dict or pydantic model carrying the four stored fields.
        def _field(name):
            return (
                getattr(result, name, None)
                if not isinstance(result, dict)
                else result.get(name)
            )

        assert _field("visual_description") == "A cat on a mat."
        assert _field("status") == "success" or _field(
            "visual_description_status"
        ) == "success"
        assert _field("model") == "llava:13b" or _field(
            "visual_description_model"
        ) == "llava:13b"

    @pytest.mark.asyncio
    async def test_policy_off_returns_404(
        self, feature_manual, stub_indexed_file, monkeypatch,
    ):
        stub_indexed_file()
        monkeypatch.setattr(
            "app.routers.vision.is_feature_enabled",
            AsyncMock(return_value=False),
        )
        with pytest.raises(HTTPException) as exc:
            await get_visual_description(file_id="img-abc", drive="family")
        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /files/{file_id}/visual_description
# ---------------------------------------------------------------------------


class TestDeleteVisualDescription:
    @pytest.mark.asyncio
    async def test_feature_off_returns_404(
        self, feature_off, stub_indexed_file,
    ):
        stub_indexed_file()
        with pytest.raises(HTTPException) as exc:
            await delete_visual_description(file_id="img-abc", drive="family")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_clears_columns_and_embedding(
        self, feature_manual, stub_indexed_file, monkeypatch,
    ):
        stub_indexed_file()

        clear_calls: list[str] = []

        def _clear(file_id: str) -> bool:
            clear_calls.append(file_id)
            return True

        monkeypatch.setattr(
            "app.routers.vision._clear_visual_description",
            _clear,
            raising=False,
        )

        result = await delete_visual_description(
            file_id="img-abc", drive="family"
        )
        assert clear_calls == ["img-abc"]
        # Response conveys ok-ness.
        status = getattr(result, "status", None)
        if status is None and isinstance(result, dict):
            status = result.get("status")
        assert status in ("ok", "deleted", "cleared")

    @pytest.mark.asyncio
    async def test_missing_row_returns_404(
        self, feature_manual, stub_indexed_file, monkeypatch,
    ):
        stub_indexed_file()
        monkeypatch.setattr(
            "app.routers.vision._clear_visual_description",
            lambda fid: False,
            raising=False,
        )
        with pytest.raises(HTTPException) as exc:
            await delete_visual_description(
                file_id="img-abc", drive="family"
            )
        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# POST /folders/visual_description/generate
# ---------------------------------------------------------------------------


class TestFolderBulkGenerate:
    @pytest.mark.asyncio
    async def test_feature_off_returns_404(self, feature_off):
        with pytest.raises(HTTPException) as exc:
            await generate_folder_visual_description(
                body={"drive": "family", "path": "photos/2024"},
                drive="family",
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_enqueues_image_files_in_folder(
        self, feature_manual, monkeypatch,
    ):
        # Only image files in the folder are returned by the finder.
        monkeypatch.setattr(
            "app.routers.vision.find_image_files_in_folder",
            MagicMock(return_value=["img-1", "img-2"]),
            raising=False,
        )
        enqueue_mock = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "app.routers.vision.enqueue_visual_description",
            enqueue_mock,
            raising=False,
        )

        result = await generate_folder_visual_description(
            body={"drive": "family", "path": "photos/2024"},
            drive="family",
        )

        assert enqueue_mock.await_count == 2
        queued = (
            getattr(result, "queued", None)
            if not isinstance(result, dict)
            else result.get("queued")
        )
        assert queued in (2, ["img-1", "img-2"])

    @pytest.mark.asyncio
    async def test_policy_off_returns_404(self, feature_manual, monkeypatch):
        monkeypatch.setattr(
            "app.routers.vision.is_feature_enabled",
            AsyncMock(return_value=False),
        )
        with pytest.raises(HTTPException) as exc:
            await generate_folder_visual_description(
                body={"drive": "private", "path": "photos"},
                drive="private",
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_body_drive_mismatch_rejected(
        self, feature_manual,
    ):
        """body.drive must match the X-HV-Drive header."""
        with pytest.raises(HTTPException) as exc:
            await generate_folder_visual_description(
                body={"drive": "other", "path": "photos"},
                drive="family",
            )
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_path",
        ["", "/etc", "../escape", "foo/../bar"],
    )
    async def test_rejects_invalid_paths(self, feature_manual, bad_path):
        with pytest.raises(HTTPException) as exc:
            await generate_folder_visual_description(
                body={"drive": "family", "path": bad_path},
                drive="family",
            )
        assert exc.value.status_code == 400
