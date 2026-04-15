"""RED-phase tests for the /refine/* router endpoints.

Spec: docs/superpowers/specs/2026-04-15-intelligence-transcript-refine.md

Endpoints under test (not yet implemented):

* ``POST /refine/files/{file_id}``
* ``POST /refine/files/{file_id}/revert``
* ``POST /refine/folders``

Access control, feature gating, per-drive policy, and 404 on unknown
file ids are all driver-level concerns the router must enforce before
touching the DB. The host's Generic Addon Proxy handles drive_access
separately — this test focuses on the router's own gates.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

# Stub ML deps so importing the router chain doesn't need torch et al.
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

# Import EXPECTED to fail in RED phase — drives collection failure so
# pytest reports the file as failing instead of skipped.
from app.routers.refine import (  # noqa: E402
    refine_file,
    refine_folder,
)


@pytest.fixture()
def feature_manual(monkeypatch, make_settings):
    """features.transcript_refine = 'manual' + LLM enabled."""
    base = make_settings()
    features = FeaturesConfig(
        indexing=base.features.indexing,
        search=base.features.search,
        auto_tags=base.features.auto_tags,
        summaries=base.features.summaries,
        rag=base.features.rag,
        # New field — expected to exist once the feature is implemented.
        transcript_refine="manual",  # type: ignore[call-arg]
    )
    settings = make_settings(
        features=features,
        llm=LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
        ),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.refine.settings", settings)

    llm_stub = MagicMock()
    llm_stub.enabled = True
    monkeypatch.setattr("app.routers.refine.get_llm_client", lambda: llm_stub)

    # Default: policy allows every drive. Individual tests override to
    # exercise the 403 / 404 paths. Async because the router awaits it.
    monkeypatch.setattr(
        "app.routers.refine.is_feature_enabled",
        AsyncMock(return_value=True),
    )
    return settings


@pytest.fixture()
def feature_off(monkeypatch, make_settings):
    base = make_settings()
    features = FeaturesConfig(
        indexing=base.features.indexing,
        search=base.features.search,
        auto_tags=base.features.auto_tags,
        summaries=base.features.summaries,
        rag=base.features.rag,
        transcript_refine="false",  # type: ignore[call-arg]
    )
    settings = make_settings(features=features)
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.refine.settings", settings)
    return settings


class TestRefineFileGating:
    @pytest.mark.asyncio
    async def test_feature_off_raises(self, feature_off):
        with pytest.raises(HTTPException) as exc:
            await refine_file(file_id="abc", drive="family")
        assert exc.value.status_code in (400, 403)

    @pytest.mark.asyncio
    async def test_unknown_file_id_returns_404(
        self, feature_manual, monkeypatch
    ):
        # File not present in indexed_files → 404.
        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = None

        class _Ctx:
            def __enter__(self):
                return session
            def __exit__(self, *a):
                return False

        monkeypatch.setattr(
            "app.routers.refine.get_search_db", lambda: _Ctx()
        )

        with pytest.raises(HTTPException) as exc:
            await refine_file(file_id="missing", drive="family")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_manual_mode_starts_job(self, feature_manual, monkeypatch):
        """Happy path returns a job_id and chunk_count."""
        session = MagicMock()
        # Indexed file exists on this drive.
        session.query.return_value.filter.return_value.first.return_value = (
            MagicMock(file_id="abc", drive="family", active=True)
        )
        # Chunk count for this file.
        session.query.return_value.filter.return_value.count.return_value = 47

        class _Ctx:
            def __enter__(self):
                return session
            def __exit__(self, *a):
                return False

        monkeypatch.setattr(
            "app.routers.refine.get_search_db", lambda: _Ctx()
        )
        # Stub the worker entry point so we don't spin up real asyncio tasks.
        start_mock = AsyncMock(return_value="job-123")
        monkeypatch.setattr(
            "app.routers.refine.start_refine_job", start_mock
        )

        resp = await refine_file(file_id="abc", drive="family")

        # Response shape from spec: { "job_id": ..., "chunk_count": 47 }
        assert getattr(resp, "job_id", None) == "job-123" or (
            isinstance(resp, dict) and resp.get("job_id") == "job-123"
        )
        chunk_count = getattr(resp, "chunk_count", None)
        if chunk_count is None and isinstance(resp, dict):
            chunk_count = resp.get("chunk_count")
        assert chunk_count == 47


class TestRefineFolder:
    @pytest.mark.asyncio
    async def test_folder_processes_transcript_bearing_files(
        self, feature_manual, monkeypatch
    ):
        # Two files in the folder have transcript_chunks; one does not.
        session = MagicMock()
        # query returns file_ids; shape is lightly checked — the impl
        # may switch between raw SQL and ORM. We patch the executor
        # directly so whatever shape wins, this test notices.
        monkeypatch.setattr(
            "app.routers.refine.find_transcript_files_in_folder",
            MagicMock(return_value=["f1", "f2"]),
        )
        start_mock = AsyncMock(side_effect=["job-a", "job-b"])
        monkeypatch.setattr(
            "app.routers.refine.start_refine_job", start_mock
        )

        class _Ctx:
            def __enter__(self):
                return session
            def __exit__(self, *a):
                return False

        monkeypatch.setattr(
            "app.routers.refine.get_search_db", lambda: _Ctx()
        )

        resp = await refine_folder(
            body={"drive": "family", "path": "videos/2024"},
            drive="family",
        )

        # Both files enqueued.
        assert start_mock.await_count == 2
        # Response lists both queued file ids.
        queued = getattr(resp, "queued", None)
        if queued is None and isinstance(resp, dict):
            queued = resp.get("queued")
        assert queued == 2 or queued == ["f1", "f2"]


class TestPerDrivePolicy:
    @pytest.mark.asyncio
    async def test_policy_off_for_drive_returns_403(
        self, feature_manual, monkeypatch
    ):
        """Per-drive policy ``transcript_refine: false`` must block the
        route regardless of the global ``features.transcript_refine``.
        """
        # Stub the policy client to deny this drive. AsyncMock because
        # is_feature_enabled is awaited in the route handler.
        monkeypatch.setattr(
            "app.routers.refine.is_feature_enabled",
            AsyncMock(return_value=False),
        )

        with pytest.raises(HTTPException) as exc:
            await refine_file(file_id="abc", drive="private")

        assert exc.value.status_code in (403, 404)


class TestCrossDriveIsolation:
    """A caller in drive A must not refine files that live in drive B."""

    @pytest.mark.asyncio
    async def test_other_drive_file_id_returns_404(
        self, feature_manual, monkeypatch
    ):
        # Simulate _fetch_indexed_file's drive filter: the file_id exists
        # but only in a different drive, so the query returns None.
        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = None

        class _Ctx:
            def __enter__(self):
                return session
            def __exit__(self, *a):
                return False

        monkeypatch.setattr(
            "app.routers.refine.get_search_db", lambda: _Ctx()
        )
        monkeypatch.setattr(
            "app.routers.refine.is_feature_enabled",
            AsyncMock(return_value=True),
        )

        with pytest.raises(HTTPException) as exc:
            # file_id belongs to 'other', caller header says 'family'
            await refine_file(file_id="other-drive-file", drive="family")

        assert exc.value.status_code == 404


class TestFolderPathValidation:
    """Folder endpoint must reject traversal / absolute / empty paths."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_path",
        ["", "/etc", "../escape", "foo/../bar", "a/b/../../c"],
    )
    async def test_rejects_invalid_paths(
        self, feature_manual, monkeypatch, bad_path
    ):
        monkeypatch.setattr(
            "app.routers.refine.is_feature_enabled",
            AsyncMock(return_value=True),
        )
        with pytest.raises(HTTPException) as exc:
            await refine_folder(
                body={"drive": "family", "path": bad_path},
                drive="family",
            )
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_body_drive_mismatch_rejected(
        self, feature_manual, monkeypatch
    ):
        """body.drive must match X-HV-Drive when both present."""
        monkeypatch.setattr(
            "app.routers.refine.is_feature_enabled",
            AsyncMock(return_value=True),
        )
        with pytest.raises(HTTPException) as exc:
            await refine_folder(
                body={"drive": "other", "path": "videos"},
                drive="family",
            )
        assert exc.value.status_code == 400


class TestFolderFileCap:
    @pytest.mark.asyncio
    async def test_rejects_over_cap(self, feature_manual, monkeypatch):
        """Folder with >MAX_FOLDER_FILES returns 413."""
        from app.routers.refine import MAX_FOLDER_FILES

        huge = [f"f{i}" for i in range(MAX_FOLDER_FILES + 5)]
        monkeypatch.setattr(
            "app.routers.refine.find_transcript_files_in_folder",
            MagicMock(return_value=huge),
        )
        monkeypatch.setattr(
            "app.routers.refine.is_feature_enabled",
            AsyncMock(return_value=True),
        )

        class _Ctx:
            def __enter__(self):
                return MagicMock()
            def __exit__(self, *a):
                return False

        monkeypatch.setattr(
            "app.routers.refine.get_search_db", lambda: _Ctx()
        )

        with pytest.raises(HTTPException) as exc:
            await refine_folder(
                body={"drive": "family", "path": "videos"},
                drive="family",
            )
        assert exc.value.status_code == 413


class TestIsFeatureEnabledAsync:
    """``is_feature_enabled`` must be awaitable and honour the policy client."""

    @pytest.mark.asyncio
    async def test_returns_false_when_feature_off(
        self, feature_off
    ):
        from app.workers.refine import is_feature_enabled as worker_fn
        result = await worker_fn("family")
        assert result is False

    @pytest.mark.asyncio
    async def test_delegates_to_policy_client(
        self, feature_manual, monkeypatch
    ):
        from app.workers import refine as worker_module

        policy_mock = AsyncMock(return_value=False)
        monkeypatch.setattr(
            "app.policy_client.is_feature_enabled",
            policy_mock,
        )
        result = await worker_module.is_feature_enabled("private")
        assert result is False
        policy_mock.assert_awaited_once_with("private", "transcript_refine")

    @pytest.mark.asyncio
    async def test_fails_open_on_policy_exception(
        self, feature_manual, monkeypatch
    ):
        from app.workers import refine as worker_module

        async def _boom(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "app.policy_client.is_feature_enabled", _boom
        )
        result = await worker_module.is_feature_enabled("family")
        assert result is True
