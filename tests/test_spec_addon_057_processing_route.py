"""SPEC-ADDON-057: ``POST /files/processing`` and its three stages.

The spec is the untracked ``.flash/spec.md`` working copy. SPEC-CORE-031
steps 2 and 3 define the request/answer envelope this addon fills;
SPEC-ADDON-057 defines the stages, the manifest declaration, the i18n keys
and the proxy-table exclusion.

These tests are RED until the addon route, the stage computation, the
manifest key and the message keys exist. Values come from the spec, never
from today's code.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Heavy ML deps are stubbed before importing anything that pulls them in.
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

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.database import Base  # noqa: E402
from app.models import IndexedFile, JobRecord, TranscriptChunk  # noqa: E402
from app.workers.whisper import LOFT_MIME  # noqa: E402
from tests.search_db_doubles import read_session_from  # noqa: E402


DRIVE = "d1"
OTHER_DRIVE = "d2"

SEARCH = {
    "key": "search",
    "i18n_key": "intelligence.processing.search",
    "required": True,
}
TEXT = {
    "key": "text",
    "i18n_key": "intelligence.processing.text",
    "required": False,
}
TRANSCRIPT = {
    "key": "transcript",
    "i18n_key": "intelligence.processing.transcript",
    "required": False,
}

VALID_STATES = {"pending", "running", "done", "failed"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    """A real search DB wired into the routes the handler may use."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def _write():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    read = read_session_from(_write)
    for target in ("app.database", "app.routers.files"):
        monkeypatch.setattr(f"{target}.get_search_db", _write, raising=False)
        monkeypatch.setattr(f"{target}.get_search_db_read", read, raising=False)
    return Session


@pytest.fixture()
def client(search_db):
    from app.routers import files

    app = FastAPI()
    app.include_router(files.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Row/job builders
# ---------------------------------------------------------------------------


def _file_type(mime_type: str) -> str:
    return mime_type.split("/")[0]


def _seed_file(
    Session,
    file_id: str,
    *,
    drive: str = DRIVE,
    active: bool = True,
    mime_type: str = "video/mp4",
    metadata_indexed: bool = False,
    text_indexed: bool = False,
    whisper_indexed: bool = False,
) -> None:
    session = Session()
    try:
        session.add(IndexedFile(
            file_id=file_id,
            drive=drive,
            filename=f"{file_id}.bin",
            file_path=f"/drives/{drive}/{file_id}.bin",
            file_type=_file_type(mime_type),
            mime_type=mime_type,
            file_size=1,
            active=active,
            metadata_indexed=metadata_indexed,
            text_indexed=text_indexed,
            whisper_indexed=whisper_indexed,
        ))
        session.commit()
    finally:
        session.close()


def _seed_job(
    Session,
    file_id: str,
    *,
    status: str,
    error_class: str | None = None,
    job_kind: str = "transcription",
    attempted_at: datetime | None = None,
) -> None:
    session = Session()
    try:
        session.add(JobRecord(
            file_id=file_id,
            job_kind=job_kind,
            provider="whisper_local",
            status=status,
            error_class=error_class,
            attempted_at=attempted_at or datetime.now(UTC),
            completed_at=(
                datetime.now(UTC) if status != "running" else None
            ),
        ))
        session.commit()
    finally:
        session.close()


def _seed_chunk(Session, file_id: str) -> None:
    session = Session()
    try:
        session.add(TranscriptChunk(
            file_id=file_id,
            chunk_index=0,
            text="a transcript chunk",
            language="en",
            timestamp_start=0.0,
            timestamp_end=1.0,
        ))
        session.commit()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _post(client, file_ids, *, drive: str = DRIVE):
    response = client.post(
        "/files/processing",
        json={"file_ids": list(file_ids)},
        headers={"X-Lit-Drive": drive},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "files" in payload, payload
    return payload["files"]


def _stages(files, file_id):
    assert file_id in files, f"{file_id} missing from {sorted(files)}"
    stages = files[file_id]["stages"]
    assert isinstance(stages, list)
    return stages


def _stage(stages, key):
    matches = [s for s in stages if s.get("key") == key]
    assert len(matches) == 1, f"expected one {key!r} stage, got {stages!r}"
    return matches[0]


def _assert_stage(stage, expected, state):
    assert stage["key"] == expected["key"]
    assert stage["i18n_key"] == expected["i18n_key"]
    assert stage["required"] is expected["required"]
    assert stage["state"] in VALID_STATES
    assert stage["state"] == state


# ---------------------------------------------------------------------------
# I5: search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "active, metadata_indexed, expected_state",
    [
        pytest.param(True, True, "done", id="active-metadata-indexed"),
        pytest.param(True, False, "pending", id="active-metadata-pending"),
        pytest.param(False, True, "pending", id="inactive-metadata-indexed"),
        pytest.param(False, False, "pending", id="inactive-metadata-pending"),
    ],
)
def test_spec_addon_057_i5_search_follows_the_active_row_flag(
    client, search_db, active, metadata_indexed, expected_state
) -> None:
    """SPEC-ADDON-057 I5: search is done only when the request-drive row is active with metadata_indexed."""
    _seed_file(
        search_db,
        "search-case",
        active=active,
        mime_type="image/jpeg",
        metadata_indexed=metadata_indexed,
    )

    stages = _stages(_post(client, ["search-case"]), "search-case")

    assert [s["key"] for s in stages] == ["search"]
    _assert_stage(_stage(stages, "search"), SEARCH, expected_state)


def test_spec_addon_057_i5_id_without_a_row_is_search_pending(
    client, search_db
) -> None:
    """SPEC-ADDON-057 I5: a requested id with no row still answers, with only search pending."""
    stages = _stages(_post(client, ["not-yet-here"]), "not-yet-here")

    assert [s["key"] for s in stages] == ["search"]
    _assert_stage(_stage(stages, "search"), SEARCH, "pending")


def test_spec_addon_057_i5_row_of_another_drive_is_left_out(
    client, search_db
) -> None:
    """SPEC-ADDON-057 I5: only the request drive's rows are answered."""
    _seed_file(search_db, "mine", drive=DRIVE, metadata_indexed=True)
    _seed_file(
        search_db,
        "theirs",
        drive=OTHER_DRIVE,
        metadata_indexed=True,
    )

    files = _post(client, ["mine", "theirs"], drive=DRIVE)

    assert "mine" in files
    assert "theirs" not in files


def test_spec_addon_057_i5_unrequested_ids_are_not_answered(
    client, search_db
) -> None:
    """SPEC-ADDON-057 I5: the answer covers the requested ids only."""
    _seed_file(search_db, "asked", metadata_indexed=True)
    _seed_file(search_db, "not-asked", metadata_indexed=True)

    files = _post(client, ["asked"])

    assert set(files) == {"asked"}


# ---------------------------------------------------------------------------
# I5: text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mime_type, text_indexed, active, expected_state",
    [
        pytest.param("text/plain", True, True, "done", id="text-plain-done"),
        pytest.param("text/plain", False, True, "pending", id="text-plain-pending"),
        pytest.param("application/pdf", True, True, "done", id="pdf-done"),
        pytest.param(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            False,
            True,
            "pending",
            id="docx-pending",
        ),
        pytest.param("text/html", False, True, "pending", id="html-pending"),
        pytest.param("text/plain", True, False, "done", id="inactive-text-row"),
        pytest.param("video/mp4", True, True, None, id="video-has-no-text-stage"),
        pytest.param("image/jpeg", True, True, None, id="image-has-no-text-stage"),
        pytest.param(LOFT_MIME, True, True, None, id="loft-has-no-text-stage"),
    ],
)
def test_spec_addon_057_i5_text_only_for_text_mimes(
    client, search_db, mime_type, text_indexed, active, expected_state
) -> None:
    """SPEC-ADDON-057 I5: text exists only for TEXT_MIMES and is never required."""
    _seed_file(
        search_db,
        "text-case",
        active=active,
        mime_type=mime_type,
        metadata_indexed=True,
        text_indexed=text_indexed,
    )

    stages = _stages(_post(client, ["text-case"]), "text-case")

    if expected_state is None:
        assert all(s["key"] != "text" for s in stages)
        return
    assert [s["key"] for s in stages] == ["search", "text"]
    _assert_stage(_stage(stages, "text"), TEXT, expected_state)


# ---------------------------------------------------------------------------
# I5: transcript
# ---------------------------------------------------------------------------


TRANSCRIPT_CASES = [
    pytest.param(
        dict(mime_type="video/mp4", whisper_indexed=False,
             jobs=[dict(status="running")]),
        "running",
        id="latest-job-running",
    ),
    pytest.param(
        dict(mime_type="video/mp4", whisper_indexed=False,
             jobs=[dict(status="failed", error_class="FatalError")]),
        "failed",
        id="latest-job-fatal",
    ),
    pytest.param(
        dict(mime_type="video/mp4", whisper_indexed=False,
             jobs=[dict(status="failed", error_class="TransientError")]),
        "pending",
        id="latest-job-transient-retried",
    ),
    pytest.param(
        dict(mime_type="video/mp4", whisper_indexed=False,
             jobs=[dict(status="failed", error_class="RateLimitError")]),
        "pending",
        id="latest-job-rate-limit-retried",
    ),
    pytest.param(
        dict(mime_type="video/mp4", whisper_indexed=False,
             jobs=[dict(status="failed", error_class="CircuitBreakerOpen")]),
        "pending",
        id="latest-job-circuit-breaker-retried",
    ),
    pytest.param(
        dict(mime_type="video/mp4", whisper_indexed=False,
             jobs=[dict(status="failed", error_class="ContainerRestart")]),
        "pending",
        id="latest-job-container-restart-retried",
    ),
    pytest.param(
        dict(mime_type="video/mp4", whisper_indexed=False,
             jobs=[dict(status="failed", error_class="UnsupportedMimeType")]),
        "pending",
        id="latest-job-unsupported-mime-retried",
    ),
    pytest.param(
        dict(mime_type="video/mp4", whisper_indexed=False,
             jobs=[dict(status="failed", error_class=None)]),
        "pending",
        id="latest-job-failed-without-class",
    ),
    pytest.param(
        dict(mime_type=LOFT_MIME, whisper_indexed=True,
             jobs=[dict(status="failed", error_class="RateLimitError")]),
        "failed",
        id="loft-temp-audio-consumed-latest-job-failed",
    ),
    pytest.param(
        dict(mime_type="video/mp4", whisper_indexed=True,
             jobs=[dict(status="failed", error_class="FatalError")]),
        "failed",
        id="whisper-indexed-latest-job-failed",
    ),
    pytest.param(
        dict(mime_type="video/mp4", whisper_indexed=True,
             jobs=[dict(status="succeeded")]),
        "done",
        id="whisper-indexed-latest-job-succeeded",
    ),
    pytest.param(
        dict(mime_type="video/mp4", whisper_indexed=True, jobs=[]),
        "done",
        id="whisper-indexed-without-a-job",
    ),
    pytest.param(
        dict(mime_type="audio/wav", whisper_indexed=True,
             jobs=[dict(status="succeeded")], chunks=False),
        "done",
        id="silent-file-whisper-indexed-without-chunks",
    ),
    pytest.param(
        dict(mime_type="video/mp4", whisper_indexed=False,
             jobs=[dict(status="succeeded")]),
        "pending",
        id="not-whisper-indexed-latest-job-succeeded",
    ),
    pytest.param(
        dict(mime_type="video/mp4", whisper_indexed=False, jobs=[]),
        "pending",
        id="not-whisper-indexed-without-a-job",
    ),
    pytest.param(
        dict(mime_type="video/mp4", whisper_indexed=False,
             jobs=[], chunks=True),
        "pending",
        id="chunks-alone-do-not-make-it-done",
    ),
    pytest.param(
        dict(
            mime_type="video/mp4",
            whisper_indexed=False,
            jobs=[
                dict(status="succeeded", attempted_offset_minutes=0),
                dict(
                    status="failed",
                    error_class="FatalError",
                    attempted_offset_minutes=-60,
                ),
            ],
        ),
        "failed",
        id="newest-id-wins-over-attempted-at",
    ),
    pytest.param(
        dict(
            mime_type="video/mp4",
            whisper_indexed=True,
            jobs=[
                dict(status="succeeded"),
                dict(status="failed", error_class="FatalError",
                     job_kind="embedding"),
            ],
        ),
        "done",
        id="non-transcription-job-ignored",
    ),
    pytest.param(
        dict(mime_type="audio/mpeg", whisper_indexed=True,
             jobs=[dict(status="succeeded")]),
        "done",
        id="audio-mime-transcript-done",
    ),
    pytest.param(
        dict(mime_type=LOFT_MIME, whisper_indexed=False, jobs=[]),
        "pending",
        id="loft-without-a-job-pending",
    ),
    pytest.param(
        dict(mime_type="image/jpeg", whisper_indexed=True,
             jobs=[dict(status="failed", error_class="FatalError")]),
        None,
        id="image-has-no-transcript-stage",
    ),
    pytest.param(
        dict(mime_type="application/pdf", whisper_indexed=True,
             jobs=[dict(status="failed", error_class="FatalError")]),
        None,
        id="text-mime-has-no-transcript-stage",
    ),
]


@pytest.mark.parametrize("case, expected_state", TRANSCRIPT_CASES)
def test_spec_addon_057_i5_transcript_states(
    client, search_db, case, expected_state
) -> None:
    """SPEC-ADDON-057 I5: transcript follows the newest transcription job by id and the whisper flag."""
    _seed_file(
        search_db,
        "transcript-case",
        mime_type=case["mime_type"],
        metadata_indexed=True,
        whisper_indexed=case["whisper_indexed"],
    )
    if case.get("chunks", True):
        _seed_chunk(search_db, "transcript-case")

    for job in case["jobs"]:
        offset = job.get("attempted_offset_minutes")
        attempted_at = (
            None if offset is None else datetime.now(UTC) + timedelta(minutes=offset)
        )
        _seed_job(
            search_db,
            "transcript-case",
            status=job["status"],
            error_class=job.get("error_class"),
            job_kind=job.get("job_kind", "transcription"),
            attempted_at=attempted_at,
        )

    stages = _stages(_post(client, ["transcript-case"]), "transcript-case")

    if expected_state is None:
        assert all(s["key"] != "transcript" for s in stages)
        return
    assert [s["key"] for s in stages] == ["search", "transcript"]
    _assert_stage(_stage(stages, "transcript"), TRANSCRIPT, expected_state)


def test_spec_addon_057_i5_transcript_absent_for_another_drive(
    client, search_db
) -> None:
    """SPEC-ADDON-057 I5: no stage is answered from another drive's row."""
    _seed_file(
        search_db,
        "elsewhere",
        drive=OTHER_DRIVE,
        mime_type="video/mp4",
        metadata_indexed=True,
        whisper_indexed=True,
    )

    files = _post(client, ["elsewhere"], drive=DRIVE)

    assert "elsewhere" not in files


# ---------------------------------------------------------------------------
# Contract (SPEC-CORE-031 steps 2 and 3) and I10
# ---------------------------------------------------------------------------


def test_spec_addon_057_core_contract_request_requires_drive_header(
    client, search_db
) -> None:
    """SPEC-ADDON-057 (SPEC-CORE-031 step 2): the call carries X-Lit-Drive; without it there is no drive context."""
    response = client.post(
        "/files/processing",
        json={"file_ids": ["x"]},
    )
    assert response.status_code == 400


def test_spec_addon_057_i10_processing_is_a_literal_path_not_a_file_id(
    client, search_db
) -> None:
    """SPEC-ADDON-057 I10: /files/processing must not be consumed by a /files/{file_id} route."""
    _seed_file(search_db, "ordinary", metadata_indexed=True)

    files = _post(client, ["ordinary"])

    assert "ordinary" in files


def test_spec_addon_057_i10_processing_route_precedes_file_id_routes() -> None:
    """SPEC-ADDON-057 I10: the literal path is declared before the /files/{file_id} routes."""
    from app.routers import files

    paths = [route.path for route in files.router.routes]
    assert "/files/processing" in paths, paths
    processing_at = paths.index("/files/processing")
    file_id_routes = [
        index
        for index, path in enumerate(paths)
        if path.startswith("/files/{file_id}")
    ]
    assert all(index > processing_at for index in file_id_routes), paths


def test_spec_addon_057_i10_manifest_declares_processing_status() -> None:
    """SPEC-ADDON-057 I10: the manifest declares the processing_status path."""
    manifest = json.loads(Path("manifest.json").read_text(encoding="utf-8"))

    assert manifest["processing_status"]["path"] == "/files/processing"


def test_spec_addon_057_i10_processing_route_is_absent_from_the_proxy_table() -> None:
    """SPEC-ADDON-057 I10: the browser-facing proxy must not list /files/processing."""
    manifest = json.loads(Path("manifest.json").read_text(encoding="utf-8"))

    proxy_paths = [
        route["path"] for route in manifest["proxy"]["routes"]
    ]
    assert "/files/processing" not in proxy_paths


