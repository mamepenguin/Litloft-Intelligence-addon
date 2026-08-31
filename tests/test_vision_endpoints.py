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
from fastapi import HTTPException

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
                    drive="family",
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_happy_path_queues_the_file(
        self, feature_manual, stub_indexed_file, monkeypatch,
    ):
        stub_indexed_file()

        # Stub the worker entry point to avoid real enqueue plumbing.
        enqueue_mock = AsyncMock(return_value={"accepted": True, "reason": None})
        monkeypatch.setattr(
            "app.routers.vision.enqueue_visual_description",
            enqueue_mock,
            raising=False,
        )

        result = await generate_visual_description(
            file_id="img-abc", drive="family",
        )

        assert result["status"] == "accepted"
        assert enqueue_mock.await_count == 1

    @pytest.mark.asyncio
    async def test_the_button_overrides_stickiness(
        self, feature_manual, stub_indexed_file, monkeypatch,
    ):
        """Without manual=True the recovery path does not exist.

        The worker refuses a settled file on the automatic contract, so
        a request that does not say it is a person asking would be
        dropped — and, before this route awaited the answer, dropped
        silently.
        """
        stub_indexed_file()
        enqueue_mock = AsyncMock(return_value={"accepted": True, "reason": None})
        monkeypatch.setattr(
            "app.routers.vision.enqueue_visual_description",
            enqueue_mock,
            raising=False,
        )

        await generate_visual_description(file_id="img-abc", drive="family")

        assert enqueue_mock.await_args.kwargs.get("manual") is True

    @pytest.mark.asyncio
    async def test_work_already_on_its_way_is_not_an_error(
        self, feature_manual, stub_indexed_file, monkeypatch,
    ):
        """The caller wanted the file described; it is being described.

        Answering 409 would push the UI into an error state over an
        outcome that is exactly what was asked for.
        """
        stub_indexed_file()
        monkeypatch.setattr(
            "app.routers.vision.enqueue_visual_description",
            AsyncMock(
                return_value={"accepted": False, "reason": "already_queued"}
            ),
            raising=False,
        )

        result = await generate_visual_description(
            file_id="img-abc", drive="family",
        )
        assert result["status"] == "already_queued"

    @pytest.mark.asyncio
    async def test_a_declined_file_answers_409_with_the_reason(
        self, feature_manual, stub_indexed_file, monkeypatch,
    ):
        """Regression detector for the silent no-op.

        Reporting "accepted" for work that was dropped leaves the
        browser polling unchanged state until it gives up, with nothing
        to show the user.
        """
        stub_indexed_file()
        monkeypatch.setattr(
            "app.routers.vision.enqueue_visual_description",
            AsyncMock(return_value={"accepted": False, "reason": "policy_off"}),
            raising=False,
        )

        with pytest.raises(HTTPException) as excinfo:
            await generate_visual_description(
                file_id="img-abc", drive="family",
            )

        assert excinfo.value.status_code == 409
        assert excinfo.value.detail["reason"] == "policy_off"


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
    async def test_an_unconfigured_model_is_told_apart_from_a_measured_one(
        self, feature_manual, stub_indexed_file, monkeypatch, make_settings,
    ):
        """Two different situations wore the same label.

        With no vision model set there is nothing to retry — the fix is
        in the configuration. A stored verdict came from a real attempt
        and the user can run it again. The UI needs to tell them apart
        to know whether to offer the button.
        """
        stub_indexed_file()
        unconfigured = make_settings(
            features=FeaturesConfig(vision_describe="manual"),  # type: ignore[call-arg]
            llm=LLMConfig(
                provider="openai_compatible",
                base_url="http://test",
                model="gemma2:27b",
                vision_model="",
            ),
        )
        monkeypatch.setattr("app.config.settings", unconfigured)
        monkeypatch.setattr("app.routers.vision.settings", unconfigured)

        result = await get_visual_description(
            file_id="img-abc", drive="family"
        )
        assert result["status"] == "unsupported"
        assert result["reason"] == "not_configured"

    @pytest.mark.asyncio
    async def test_a_disabled_client_reads_as_not_configured(
        self, feature_manual, stub_indexed_file, monkeypatch,
    ):
        """A vision model set against a disabled client runs nothing.

        There is no work the user can trigger, so this belongs with
        "not configured" — the notice, without a retry button that
        would only queue something nobody would run.
        """
        stub_indexed_file()
        disabled = MagicMock()
        disabled.enabled = False
        monkeypatch.setattr(
            "app.routers.vision.get_llm_client", lambda: disabled
        )

        result = await get_visual_description(
            file_id="img-abc", drive="family"
        )
        assert result["status"] == "unsupported"
        assert result["reason"] == "not_configured"

        with pytest.raises(HTTPException) as exc:
            await generate_visual_description(
                file_id="img-abc", drive="family",
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_a_stored_reason_reaches_the_caller(
        self, feature_manual, stub_indexed_file, monkeypatch,
    ):
        stub_indexed_file()
        monkeypatch.setattr(
            "app.routers.vision._fetch_visual_description",
            lambda file_id: {
                "visual_description": None,
                "visual_description_status": "failed",
                "visual_description_model": "llava:13b",
                "visual_description_generated_at": None,
                "visual_description_error": "model_missing",
            },
            raising=False,
        )

        result = await get_visual_description(
            file_id="img-abc", drive="family"
        )
        assert result["status"] == "failed"
        assert result["reason"] == "model_missing"

    @pytest.mark.asyncio
    async def test_a_row_from_before_the_column_existed_has_no_reason(
        self, feature_manual, stub_indexed_file, monkeypatch,
    ):
        """No backfill, so the UI must cope with not being told why."""
        stub_indexed_file()
        monkeypatch.setattr(
            "app.routers.vision._fetch_visual_description",
            lambda file_id: {
                "visual_description": None,
                "visual_description_status": "failed",
                "visual_description_model": "llava:13b",
                "visual_description_generated_at": None,
                "visual_description_error": None,
            },
            raising=False,
        )

        result = await get_visual_description(
            file_id="img-abc", drive="family"
        )
        assert result["status"] == "failed"
        assert result["reason"] is None

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
                body={"drive": "family", "file_ids": ["img-1"]},
                drive="family",
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_enqueues_filtered_file_ids(
        self, feature_manual, monkeypatch,
    ):
        # The router narrows the supplied ids to active image rows in
        # the request drive; cross-drive / non-image / inactive ids are
        # silently dropped before enqueue.
        monkeypatch.setattr(
            "app.routers.vision.filter_image_file_ids",
            MagicMock(return_value=["img-1", "img-2"]),
            raising=False,
        )
        enqueue_mock = AsyncMock(return_value={"accepted": True, "reason": None})
        monkeypatch.setattr(
            "app.routers.vision.enqueue_visual_description",
            enqueue_mock,
            raising=False,
        )

        result = await generate_folder_visual_description(
            body={
                "drive": "family",
                "file_ids": ["img-1", "img-2", "video-3"],
            },
            drive="family",
        )

        assert enqueue_mock.await_count == 2
        # The folder button is a person pressing it, so it carries the
        # same authority as the single-file one — otherwise the images
        # the old guessing marked unsupported could only be recovered
        # one at a time.
        assert all(
            call.kwargs.get("manual") is True
            for call in enqueue_mock.await_args_list
        )
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
                body={"drive": "private", "file_ids": ["img-1"]},
                drive="private",
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_body_drive_mismatch_rejected(
        self, feature_manual,
    ):
        """body.drive must match the X-Lit-Drive header."""
        with pytest.raises(HTTPException) as exc:
            await generate_folder_visual_description(
                body={"drive": "other", "file_ids": ["img-1"]},
                drive="family",
            )
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_body",
        [
            {"drive": "family"},  # missing file_ids
            {"drive": "family", "file_ids": "img-1"},  # not a list
            {"drive": "family", "file_ids": [1, 2]},  # not strings
        ],
    )
    async def test_rejects_invalid_file_ids(self, feature_manual, bad_body):
        with pytest.raises(HTTPException) as exc:
            await generate_folder_visual_description(
                body=bad_body,
                drive="family",
            )
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_file_ids_returns_zero_queued(
        self, feature_manual, monkeypatch,
    ):
        """Empty list is well-formed; we just enqueue nothing."""
        monkeypatch.setattr(
            "app.routers.vision.filter_image_file_ids",
            MagicMock(return_value=[]),
            raising=False,
        )
        enqueue_mock = AsyncMock(return_value={"accepted": True, "reason": None})
        monkeypatch.setattr(
            "app.routers.vision.enqueue_visual_description",
            enqueue_mock,
            raising=False,
        )

        result = await generate_folder_visual_description(
            body={"drive": "family", "file_ids": []},
            drive="family",
        )

        assert enqueue_mock.await_count == 0
        queued = (
            getattr(result, "queued", None)
            if not isinstance(result, dict)
            else result.get("queued")
        )
        assert queued == 0
