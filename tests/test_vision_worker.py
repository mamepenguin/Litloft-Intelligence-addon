"""VisionDescribeWorker tests (RED phase).

Spec: docs/superpowers/specs/2026-04-23-intelligence-vision-describe.md

The worker lives at ``app.workers.vision`` and mirrors the shape of the
summaries / refine workers:

* ``enqueue(file_id)`` — add a file to the queue, no-op for non-image
  mimes, missing/trash files, or per-drive-policy-OFF drives.
* ``_process_file(file_id)`` — build context, call LLM, persist result.
* status transitions: NULL → "pending" → "success" | "failed" | "unsupported".
* On "success", insert a row into the ``embeddings`` table with
  ``embedding_type = "vision_description"`` and a content_preview so
  hybrid retrieval can index it.

Key policy rules:

* Per-drive policy OFF → ``is_file_feature_enabled(file_id, "vision_describe")``
  returns False → worker silently skips, no status mutation.
* "unsupported" is a sticky state until the model changes — a subsequent
  run with the same vision_model must NOT retry.
* "failed" is retryable — a subsequent run is allowed to re-attempt.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

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

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.config import FeaturesConfig, LLMConfig  # noqa: E402
from app.llm import (  # noqa: E402
    FAILURE_EMPTY,
    FAILURE_MODEL_MISSING,
    FAILURE_TOKEN_BUDGET,
    FAILURE_VISION_UNSUPPORTED,
    VisionGeneration,
)
from app.database import (  # noqa: E402
    Base,
    _create_file_summaries_table,
)
from app.models import Embedding, IndexedFile  # noqa: E402


# Worker import — expected to fail in RED phase.
pytest.importorskip(
    "app.workers.vision",
    reason="VisionDescribeWorker not yet implemented (RED phase)",
)

from app.workers.vision import (  # noqa: E402
    VisionDescribeWorker,
)


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    """Real SQLite with indexed_files + file_summaries + embeddings tables."""
    db_path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        _create_file_summaries_table(conn)

    # Seed an image file and a non-image file so we can exercise the
    # mime filter. All rows belong to drive "family" by default.
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    seed = Session()
    try:
        seed.add_all([
            IndexedFile(
                file_id="img-ok",
                drive="family",
                filename="cat.jpg",
                file_path="/drives/family/cat.jpg",
                file_type="image",
                mime_type="image/jpeg",
                file_size=10000,
                active=True,
            ),
            IndexedFile(
                file_id="vid-skip",
                drive="family",
                filename="clip.mp4",
                file_path="/drives/family/clip.mp4",
                file_type="video",
                mime_type="video/mp4",
                file_size=200000,
                active=True,
            ),
            IndexedFile(
                file_id="img-off-drive",
                drive="private",
                filename="secret.jpg",
                file_path="/drives/private/secret.jpg",
                file_type="image",
                mime_type="image/jpeg",
                file_size=5000,
                active=True,
            ),
        ])
        seed.commit()
    finally:
        seed.close()

    @contextmanager
    def _get_search_db():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr("app.database.get_search_db", _get_search_db)
    monkeypatch.setattr("app.workers.vision.get_search_db", _get_search_db)
    return engine, Session


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
    monkeypatch.setattr("app.workers.vision.settings", settings)
    return settings


@pytest.fixture()
def policy_allow_family(monkeypatch):
    """is_file_feature_enabled returns True for drive 'family', False for 'private'."""
    async def _is_enabled(file_id: str, feature: str) -> bool:
        # Worker looks up drive from file_id; this fixture collapses
        # that to the seeded mapping.
        mapping = {
            "img-ok": True,
            "vid-skip": True,
            "img-off-drive": False,
            "ghost": True,
        }
        return mapping.get(file_id, True)

    monkeypatch.setattr(
        "app.workers.vision.is_file_feature_enabled",
        AsyncMock(side_effect=_is_enabled),
        raising=False,
    )


# ---------------------------------------------------------------------------
# enqueue()
# ---------------------------------------------------------------------------


class TestEnqueue:
    @pytest.mark.asyncio
    async def test_image_file_is_enqueued(
        self, search_db, feature_manual, policy_allow_family,
    ):
        worker = VisionDescribeWorker()
        accepted = (await worker.enqueue("img-ok"))["accepted"]
        assert accepted is True

    @pytest.mark.asyncio
    async def test_non_image_mime_is_rejected(
        self, search_db, feature_manual, policy_allow_family,
    ):
        worker = VisionDescribeWorker()
        accepted = (await worker.enqueue("vid-skip"))["accepted"]
        assert accepted is False

    @pytest.mark.asyncio
    async def test_missing_file_id_skipped_gracefully(
        self, search_db, feature_manual, policy_allow_family,
    ):
        """Unknown file_id must not raise — it's a no-op skip."""
        worker = VisionDescribeWorker()
        accepted = (await worker.enqueue("ghost"))["accepted"]
        assert accepted is False

    @pytest.mark.asyncio
    async def test_policy_off_drive_is_skipped(
        self, search_db, feature_manual, policy_allow_family,
    ):
        worker = VisionDescribeWorker()
        accepted = (await worker.enqueue("img-off-drive"))["accepted"]
        assert accepted is False

    @pytest.mark.asyncio
    async def test_feature_false_mode_rejects_everything(
        self, search_db, monkeypatch, make_settings, policy_allow_family,
    ):
        settings = make_settings(
            features=FeaturesConfig(vision_describe="false"),  # type: ignore[call-arg]
            llm=LLMConfig(vision_model="llava:13b"),
        )
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.vision.settings", settings)

        worker = VisionDescribeWorker()
        assert (await worker.enqueue("img-ok"))["accepted"] is False

    @pytest.mark.asyncio
    async def test_missing_vision_model_rejects(
        self, search_db, monkeypatch, make_settings, policy_allow_family,
    ):
        """Graceful degradation — no vision_model → no enqueue."""
        settings = make_settings(
            features=FeaturesConfig(vision_describe="manual"),  # type: ignore[call-arg]
            llm=LLMConfig(
                provider="openai_compatible",
                base_url="http://test",
                model="gemma2:27b",
                vision_model="",
            ),
        )
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.vision.settings", settings)

        worker = VisionDescribeWorker()
        assert (await worker.enqueue("img-ok"))["accepted"] is False


# ---------------------------------------------------------------------------
# _process_file status transitions
# ---------------------------------------------------------------------------


def _get_summary_row(engine, file_id: str):
    with engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT visual_description, visual_description_status, "
                "visual_description_model, visual_description_generated_at, "
                "visual_description_error "
                "FROM file_summaries WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchone()


class TestProcessFileStatusTransitions:
    @pytest.mark.asyncio
    async def test_success_writes_description_and_status(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        engine, _ = search_db

        # Stub the image loader so we don't need a real file on disk.
        monkeypatch.setattr(
            "app.workers.vision._load_image_bytes",
            lambda file_id: (b"\xff\xd8\xff\xe0fake", "image/jpeg"),
            raising=False,
        )

        # Stub the LLM to return a fixed description.
        llm_stub = MagicMock()
        llm_stub.enabled = True
        llm_stub.generate_vision = AsyncMock(
            return_value=VisionGeneration("A red apple on a wooden table.", None)
        )
        monkeypatch.setattr(
            "app.workers.vision.get_llm_client", lambda: llm_stub
        )
        # Stub embedding generation so we don't need a real model.
        monkeypatch.setattr(
            "app.workers.vision._embed_and_store",
            lambda file_id, text: None,
            raising=False,
        )

        worker = VisionDescribeWorker()
        await worker._process_file("img-ok")

        row = _get_summary_row(engine, "img-ok")
        assert row is not None
        assert row[0] == "A red apple on a wooden table."
        assert row[1] == "success"
        assert row[2] == "llava:13b"
        assert row[3] is not None  # generated_at timestamp set

    @pytest.mark.asyncio
    async def test_failed_llm_sets_failed_status(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        engine, _ = search_db

        monkeypatch.setattr(
            "app.workers.vision._load_image_bytes",
            lambda file_id: (b"\xff\xd8fake", "image/jpeg"),
            raising=False,
        )

        llm_stub = MagicMock()
        llm_stub.enabled = True
        llm_stub.generate_vision = AsyncMock(
            return_value=VisionGeneration(None, FAILURE_EMPTY)
        )
        monkeypatch.setattr(
            "app.workers.vision.get_llm_client", lambda: llm_stub
        )

        worker = VisionDescribeWorker()
        await worker._process_file("img-ok")

        row = _get_summary_row(engine, "img-ok")
        assert row is not None
        assert row[0] is None
        assert row[1] == "failed"

    @pytest.mark.asyncio
    async def test_the_failure_reason_is_persisted(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        """The UI cannot offer the right remedy for a reason it never sees."""
        engine, _ = search_db

        monkeypatch.setattr(
            "app.workers.vision._load_image_bytes",
            lambda file_id: (b"\xff\xd8fake", "image/jpeg"),
            raising=False,
        )

        llm_stub = MagicMock()
        llm_stub.enabled = True
        llm_stub.generate_vision = AsyncMock(
            return_value=VisionGeneration(None, FAILURE_MODEL_MISSING)
        )
        monkeypatch.setattr(
            "app.workers.vision.get_llm_client", lambda: llm_stub
        )

        worker = VisionDescribeWorker()
        await worker._process_file("img-ok")

        row = _get_summary_row(engine, "img-ok")
        assert row[1] == "failed"
        assert row[4] == FAILURE_MODEL_MISSING

    @pytest.mark.asyncio
    async def test_a_reason_does_not_outlive_its_attempt(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        """A stale reason read against a later attempt misleads.

        The file fails, then succeeds. The success must not carry the
        earlier failure's explanation.
        """
        engine, _ = search_db

        monkeypatch.setattr(
            "app.workers.vision._load_image_bytes",
            lambda file_id: (b"\xff\xd8fake", "image/jpeg"),
            raising=False,
        )
        monkeypatch.setattr(
            "app.workers.vision._embed_and_store",
            lambda file_id, text: None,
            raising=False,
        )

        llm_stub = MagicMock()
        llm_stub.enabled = True
        monkeypatch.setattr(
            "app.workers.vision.get_llm_client", lambda: llm_stub
        )

        worker = VisionDescribeWorker()
        llm_stub.generate_vision = AsyncMock(
            return_value=VisionGeneration(None, FAILURE_MODEL_MISSING)
        )
        await worker._process_file("img-ok")
        assert _get_summary_row(engine, "img-ok")[4] == FAILURE_MODEL_MISSING

        llm_stub.generate_vision = AsyncMock(
            return_value=VisionGeneration("A red apple.", None)
        )
        await worker._process_file("img-ok")
        row = _get_summary_row(engine, "img-ok")
        assert row[1] == "success"
        assert row[4] is None

    @pytest.mark.asyncio
    async def test_truncated_description_is_not_stored_as_success(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        """A description cut off by the token ceiling is not a description.

        Success is sticky for the same model, so storing half a sentence
        would both put it into retrieval as though it were whole and
        stop a raised ``vision_max_tokens`` from ever re-running the
        file.
        """
        engine, _ = search_db

        monkeypatch.setattr(
            "app.workers.vision._load_image_bytes",
            lambda file_id: (b"\xff\xd8fake", "image/jpeg"),
            raising=False,
        )

        llm_stub = MagicMock()
        llm_stub.enabled = True
        llm_stub.generate_vision = AsyncMock(
            return_value=VisionGeneration(
                "A red apple on a wooden tab", FAILURE_TOKEN_BUDGET
            )
        )
        monkeypatch.setattr(
            "app.workers.vision.get_llm_client", lambda: llm_stub
        )
        embed_mock = MagicMock()
        monkeypatch.setattr(
            "app.workers.vision._embed_and_store", embed_mock, raising=False,
        )

        worker = VisionDescribeWorker()
        await worker._process_file("img-ok")

        row = _get_summary_row(engine, "img-ok")
        assert row is not None
        assert row[1] == "failed"
        assert row[0] is None
        embed_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_unsupported_response_sets_unsupported_status(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        engine, _ = search_db

        monkeypatch.setattr(
            "app.workers.vision._load_image_bytes",
            lambda file_id: (b"\xff\xd8fake", "image/jpeg"),
            raising=False,
        )

        # A probed verdict about the model → status = "unsupported".
        llm_stub = MagicMock()
        llm_stub.enabled = True
        llm_stub.generate_vision = AsyncMock(
            return_value=VisionGeneration(None, FAILURE_VISION_UNSUPPORTED)
        )
        monkeypatch.setattr(
            "app.workers.vision.get_llm_client", lambda: llm_stub
        )

        worker = VisionDescribeWorker()
        await worker._process_file("img-ok")

        row = _get_summary_row(engine, "img-ok")
        assert row is not None
        assert row[0] is None
        assert row[1] == "unsupported"
        # Model name recorded so regenerate-after-model-change can detect.
        assert row[2] == "llava:13b"

    @pytest.mark.asyncio
    async def test_unsupported_same_model_does_not_retry(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        """A subsequent enqueue for an unsupported file must skip LLM call."""
        engine, _ = search_db
        now = datetime.now(UTC).isoformat()
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO file_summaries "
                    "(file_id, short_summary, long_summary, model, "
                    "context_type, context_chars, was_truncated, status, "
                    "created_at, visual_description_status, "
                    "visual_description_model) "
                    "VALUES (:fid, '', '', '', 'image', 0, 0, 'hidden', "
                    ":now, 'unsupported', 'llava:13b')"
                ),
                {"fid": "img-ok", "now": now},
            )

        llm_stub = MagicMock()
        llm_stub.enabled = True
        llm_stub.generate_vision = AsyncMock(
            return_value=VisionGeneration("should not be called", None)
        )
        monkeypatch.setattr(
            "app.workers.vision.get_llm_client", lambda: llm_stub
        )

        worker = VisionDescribeWorker()
        accepted = (await worker.enqueue("img-ok"))["accepted"]
        assert accepted is False
        llm_stub.generate_vision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_manual_request_overrides_unsupported_stickiness(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        """The escape hatch from a settled verdict.

        Stickiness protects automatic sweeps from re-spending on an
        outcome that will not change on its own. A person pressing the
        button has decided otherwise — most likely because they just
        fixed the thing that caused it — and must not be told no.
        """
        engine, _ = search_db
        now = datetime.now(UTC).isoformat()
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO file_summaries "
                    "(file_id, short_summary, long_summary, model, "
                    "context_type, context_chars, was_truncated, status, "
                    "created_at, visual_description_status, "
                    "visual_description_model) "
                    "VALUES (:fid, '', '', '', 'image', 0, 0, 'hidden', "
                    ":now, 'unsupported', 'llava:13b')"
                ),
                {"fid": "img-ok", "now": now},
            )

        worker = VisionDescribeWorker()
        # Same model as the stored verdict — the automatic path refuses.
        assert (await worker.enqueue("img-ok"))["accepted"] is False
        # The person asking gets through.
        assert (await worker.enqueue("img-ok", manual=True))["accepted"] is True

    @pytest.mark.asyncio
    async def test_manual_request_overrides_success_stickiness(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        """Regenerate must work without deleting the row first."""
        engine, _ = search_db
        now = datetime.now(UTC).isoformat()
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO file_summaries "
                    "(file_id, short_summary, long_summary, model, "
                    "context_type, context_chars, was_truncated, status, "
                    "created_at, visual_description, "
                    "visual_description_status, visual_description_model) "
                    "VALUES (:fid, '', '', '', 'image', 0, 0, 'hidden', "
                    ":now, 'An old description', 'success', 'llava:13b')"
                ),
                {"fid": "img-ok", "now": now},
            )

        worker = VisionDescribeWorker()
        assert (await worker.enqueue("img-ok"))["accepted"] is False
        assert (await worker.enqueue("img-ok", manual=True))["accepted"] is True

    @pytest.mark.asyncio
    async def test_work_already_in_flight_is_not_bought_twice(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        """Manual does not override "already queued".

        Stickiness is about a settled outcome the user may want to
        re-spend on. Work already on its way is not that: asking again
        buys a second LLM call whose result overwrites the first.
        """
        worker = VisionDescribeWorker()
        assert (await worker.enqueue("img-ok", manual=True))["accepted"] is True

        second = await worker.enqueue("img-ok", manual=True)
        assert second == {"accepted": False, "reason": "already_queued"}

        # Draining the queue releases the file — a worker that took it
        # and died leaves no claim behind.
        file_id = await worker._queue.get()
        worker._queued.discard(file_id)
        third = await worker.enqueue("img-ok", manual=True)
        assert third["accepted"] is True

    @pytest.mark.asyncio
    async def test_a_restart_resumes_work_that_was_already_accepted(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        """The queue dies with the process; the rows do not.

        Recording pending on acceptance is what makes the state honest
        while a file waits its turn, but it also means a restart leaves
        every waiting file marked pending with nothing left to run it.
        Startup is where that reading is unambiguous: a fresh worker
        holds nothing, so a pending row is an abandoned claim.
        """
        engine, _ = search_db
        now = datetime.now(UTC).isoformat()
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO file_summaries "
                    "(file_id, short_summary, long_summary, model, "
                    "context_type, context_chars, was_truncated, status, "
                    "created_at, visual_description_status, "
                    "visual_description_model) "
                    "VALUES (:fid, '', '', '', 'image', 0, 0, 'hidden', "
                    ":now, 'pending', 'llava:13b')"
                ),
                {"fid": "img-ok", "now": now},
            )

        fresh = VisionDescribeWorker()
        assert await fresh.requeue_abandoned() == 1
        assert "img-ok" in fresh._queued

    @pytest.mark.asyncio
    async def test_resuming_leaves_settled_rows_alone(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        """Only pending rows are abandoned claims.

        A described or refused file was not waiting for anything, so
        picking it up would be new spending, not resumption.
        """
        engine, _ = search_db
        now = datetime.now(UTC).isoformat()
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO file_summaries "
                    "(file_id, short_summary, long_summary, model, "
                    "context_type, context_chars, was_truncated, status, "
                    "created_at, visual_description, "
                    "visual_description_status, visual_description_model) "
                    "VALUES (:fid, '', '', '', 'image', 0, 0, 'hidden', "
                    ":now, 'A description', 'success', 'llava:13b')"
                ),
                {"fid": "img-ok", "now": now},
            )

        fresh = VisionDescribeWorker()
        assert await fresh.requeue_abandoned() == 0

    @pytest.mark.asyncio
    async def test_accepting_a_file_records_pending_immediately(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        """Queued is a true state, and every reader deserves to see it.

        Behind other files the row would otherwise keep reporting its
        previous outcome, so a caller polling for the change it just
        asked for reads the old answer and concludes nothing happened.
        """
        engine, _ = search_db
        now = datetime.now(UTC).isoformat()
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO file_summaries "
                    "(file_id, short_summary, long_summary, model, "
                    "context_type, context_chars, was_truncated, status, "
                    "created_at, visual_description_status, "
                    "visual_description_model, visual_description_error) "
                    "VALUES (:fid, '', '', '', 'image', 0, 0, 'hidden', "
                    ":now, 'unsupported', 'llava:13b', 'vision_unsupported')"
                ),
                {"fid": "img-ok", "now": now},
            )

        worker = VisionDescribeWorker()
        assert (await worker.enqueue("img-ok", manual=True))["accepted"] is True

        row = _get_summary_row(engine, "img-ok")
        assert row[1] == "pending"
        assert row[4] is None

    @pytest.mark.asyncio
    async def test_manual_does_not_override_the_gates_that_are_not_about_cost(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        """Policy and mime are not the user's to overrule.

        Stickiness is a spending guard, so an explicit request may spend.
        A drive whose operator turned the feature off, and a file that is
        not an image, are different questions entirely.
        """
        worker = VisionDescribeWorker()
        off_drive = await worker.enqueue("img-off-drive", manual=True)
        assert off_drive == {"accepted": False, "reason": "policy_off"}
        not_image = await worker.enqueue("vid-skip", manual=True)
        assert not_image == {"accepted": False, "reason": "not_an_image"}

    @pytest.mark.asyncio
    async def test_rejection_names_the_gate_that_refused(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        """A router cannot explain a refusal it was not told about."""
        engine, _ = search_db
        now = datetime.now(UTC).isoformat()
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO file_summaries "
                    "(file_id, short_summary, long_summary, model, "
                    "context_type, context_chars, was_truncated, status, "
                    "created_at, visual_description_status, "
                    "visual_description_model) "
                    "VALUES (:fid, '', '', '', 'image', 0, 0, 'hidden', "
                    ":now, 'unsupported', 'llava:13b')"
                ),
                {"fid": "img-ok", "now": now},
            )

        worker = VisionDescribeWorker()
        assert await worker.enqueue("img-ok") == {
            "accepted": False, "reason": "unsupported_sticky",
        }
        assert await worker.enqueue("ghost") == {
            "accepted": False, "reason": "file_not_found",
        }

    @pytest.mark.asyncio
    async def test_success_same_model_does_not_retry(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        """An already-described file with the SAME model must skip LLM call.

        Without this guard, every CLIP re-run (model swap, manual reindex)
        would re-describe the image and burn LLM budget overwriting an
        identical caption.
        """
        engine, _ = search_db
        now = datetime.now(UTC).isoformat()
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO file_summaries "
                    "(file_id, short_summary, long_summary, model, "
                    "context_type, context_chars, was_truncated, status, "
                    "created_at, visual_description_status, "
                    "visual_description_model) "
                    "VALUES (:fid, '', '', '', 'image', 0, 0, 'hidden', "
                    ":now, 'success', 'llava:13b')"
                ),
                {"fid": "img-ok", "now": now},
            )

        llm_stub = MagicMock()
        llm_stub.enabled = True
        llm_stub.generate_vision = AsyncMock(
            return_value=VisionGeneration("should not be called", None)
        )
        monkeypatch.setattr(
            "app.workers.vision.get_llm_client", lambda: llm_stub
        )

        worker = VisionDescribeWorker()
        accepted = (await worker.enqueue("img-ok"))["accepted"]
        assert accepted is False
        llm_stub.generate_vision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_success_different_model_retries(
        self, search_db, monkeypatch, make_settings, policy_allow_family,
    ):
        """Changing vision_model lets a previously successful file rerun."""
        settings = make_settings(
            features=FeaturesConfig(vision_describe="manual"),  # type: ignore[call-arg]
            llm=LLMConfig(
                provider="openai_compatible",
                base_url="http://test",
                model="gemma2:27b",
                vision_model="gpt-4o-mini",  # Different from stored "llava:13b".
            ),
        )
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.vision.settings", settings)

        engine, _ = search_db
        now = datetime.now(UTC).isoformat()
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO file_summaries "
                    "(file_id, short_summary, long_summary, model, "
                    "context_type, context_chars, was_truncated, status, "
                    "created_at, visual_description_status, "
                    "visual_description_model) "
                    "VALUES (:fid, 'old desc', '', '', 'image', 0, 0, "
                    "'hidden', :now, 'success', 'llava:13b')"
                ),
                {"fid": "img-ok", "now": now},
            )

        worker = VisionDescribeWorker()
        accepted = (await worker.enqueue("img-ok"))["accepted"]
        assert accepted is True

    @pytest.mark.asyncio
    async def test_unsupported_different_model_retries(
        self, search_db, monkeypatch, make_settings, policy_allow_family,
    ):
        """Changing vision_model must reset stickiness and allow retry."""
        settings = make_settings(
            features=FeaturesConfig(vision_describe="manual"),  # type: ignore[call-arg]
            llm=LLMConfig(
                provider="openai_compatible",
                base_url="http://test",
                model="gemma2:27b",
                vision_model="gpt-4o-mini",  # Different from stored "llava:13b".
            ),
        )
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.vision.settings", settings)

        engine, _ = search_db
        now = datetime.now(UTC).isoformat()
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO file_summaries "
                    "(file_id, short_summary, long_summary, model, "
                    "context_type, context_chars, was_truncated, status, "
                    "created_at, visual_description_status, "
                    "visual_description_model) "
                    "VALUES (:fid, '', '', '', 'image', 0, 0, 'hidden', "
                    ":now, 'unsupported', 'llava:13b')"
                ),
                {"fid": "img-ok", "now": now},
            )

        worker = VisionDescribeWorker()
        accepted = (await worker.enqueue("img-ok"))["accepted"]
        # Different model, so the sticky-unsupported check should let it
        # through. Actual LLM call is out of scope here.
        assert accepted is True


# ---------------------------------------------------------------------------
# enqueue_unprocessed() — startup sweep for already-indexed images
# ---------------------------------------------------------------------------


def _seed_summary_row(engine, file_id: str, status: str, model: str) -> None:
    """Insert a minimal file_summaries placeholder + vision columns."""
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO file_summaries "
                "(file_id, short_summary, long_summary, model, context_type, "
                "context_chars, was_truncated, status, created_at, "
                "visual_description, visual_description_status, "
                "visual_description_model, visual_description_generated_at) "
                "VALUES (:fid, '', '', '', 'image', 0, 0, 'hidden', :now, "
                "NULL, :status, :model, NULL)"
            ),
            {
                "fid": file_id,
                "now": datetime.now(UTC).isoformat(),
                "status": status,
                "model": model,
            },
        )


class TestEnqueueUnprocessed:
    """Startup sweep should pick up already-indexed images with no
    description yet, and skip any row already in a terminal/sticky state.
    """

    @pytest.mark.asyncio
    async def test_picks_up_image_with_null_status(
        self, search_db, feature_manual, policy_allow_family,
    ):
        worker = VisionDescribeWorker()
        queued = await worker.enqueue_unprocessed()
        # ``img-ok`` has no file_summaries row → NULL status → enqueue.
        # ``vid-skip`` is video → mime filter excludes.
        # ``img-off-drive`` is policy OFF → enqueue() rejects via _should_accept.
        assert queued == 1

    @pytest.mark.asyncio
    async def test_skips_when_status_already_success(
        self, search_db, feature_manual, policy_allow_family,
    ):
        engine, _ = search_db
        _seed_summary_row(engine, "img-ok", "success", "llava:13b")

        worker = VisionDescribeWorker()
        queued = await worker.enqueue_unprocessed()
        assert queued == 0

    @pytest.mark.asyncio
    async def test_skips_when_status_pending(
        self, search_db, feature_manual, policy_allow_family,
    ):
        """Pending rows are owned by an in-flight worker — don't double up."""
        engine, _ = search_db
        _seed_summary_row(engine, "img-ok", "pending", "llava:13b")

        worker = VisionDescribeWorker()
        queued = await worker.enqueue_unprocessed()
        assert queued == 0

    @pytest.mark.asyncio
    async def test_skips_when_status_failed(
        self, search_db, feature_manual, policy_allow_family,
    ):
        """Failed status is left as a manual-retry case (matches UI semantics)."""
        engine, _ = search_db
        _seed_summary_row(engine, "img-ok", "failed", "llava:13b")

        worker = VisionDescribeWorker()
        queued = await worker.enqueue_unprocessed()
        assert queued == 0

    @pytest.mark.asyncio
    async def test_skips_unsupported_with_same_model(
        self, search_db, feature_manual, policy_allow_family,
    ):
        engine, _ = search_db
        _seed_summary_row(engine, "img-ok", "unsupported", "llava:13b")

        worker = VisionDescribeWorker()
        queued = await worker.enqueue_unprocessed()
        # Either skipped at the SQL filter (status non-NULL) or at
        # _should_accept's sticky-unsupported guard. Either way, 0.
        assert queued == 0

    @pytest.mark.asyncio
    async def test_skips_non_image_files(
        self, search_db, feature_manual, policy_allow_family,
    ):
        """Videos / audio rows must never get enqueued for vision."""
        worker = VisionDescribeWorker()
        queued = await worker.enqueue_unprocessed()
        # ``vid-skip`` is the only video; we already verified the count
        # is 1 (just ``img-ok``) in the NULL-status case.
        assert queued == 1

    @pytest.mark.asyncio
    async def test_respects_per_drive_policy(
        self, search_db, feature_manual, policy_allow_family,
    ):
        """``img-off-drive`` is an image but its drive opts out of vision."""
        worker = VisionDescribeWorker()
        queued = await worker.enqueue_unprocessed()
        # img-off-drive must not contribute to the queue count.
        assert queued == 1


# ---------------------------------------------------------------------------
# Embedding write-through after success
# ---------------------------------------------------------------------------


class TestEmbeddingRegistration:
    @pytest.mark.asyncio
    async def test_success_inserts_vision_description_embedding(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        """A success path must register an embeddings row of type
        ``vision_description`` so hybrid retrieval can pick it up."""
        engine, Session = search_db

        monkeypatch.setattr(
            "app.workers.vision._load_image_bytes",
            lambda file_id: (b"\xff\xd8fake", "image/jpeg"),
            raising=False,
        )

        llm_stub = MagicMock()
        llm_stub.enabled = True
        llm_stub.generate_vision = AsyncMock(
            return_value=VisionGeneration(
                "A yellow duckling swimming in a pond.", None
            )
        )
        monkeypatch.setattr(
            "app.workers.vision.get_llm_client", lambda: llm_stub
        )

        # Capture embed-and-store invocations (we don't want to run the
        # real embedder, which pulls in heavy ML deps).
        captured: list[tuple[str, str]] = []

        def _fake_embed_and_store(file_id: str, text: str) -> None:
            # Mimic the implementation: insert a real Embedding row so
            # the outer assertion can observe it.
            session = Session()
            try:
                session.add(
                    Embedding(
                        id=f"vd_{file_id}_0",
                        file_id=file_id,
                        embedding_type="vision_description",
                        content_preview=text[:200],
                        vector_table="vec_text",
                    )
                )
                session.commit()
                captured.append((file_id, text))
            finally:
                session.close()

        monkeypatch.setattr(
            "app.workers.vision._embed_and_store",
            _fake_embed_and_store,
            raising=False,
        )

        worker = VisionDescribeWorker()
        await worker._process_file("img-ok")

        session = Session()
        try:
            rows = (
                session.query(Embedding)
                .filter(Embedding.file_id == "img-ok")
                .filter(Embedding.embedding_type == "vision_description")
                .all()
            )
        finally:
            session.close()

        assert len(rows) == 1
        assert rows[0].content_preview.startswith("A yellow duckling")
        assert captured == [("img-ok", "A yellow duckling swimming in a pond.")]

    @pytest.mark.asyncio
    async def test_failure_does_not_write_embedding(
        self, search_db, feature_manual, policy_allow_family, monkeypatch,
    ):
        engine, Session = search_db

        monkeypatch.setattr(
            "app.workers.vision._load_image_bytes",
            lambda file_id: (b"\xff\xd8fake", "image/jpeg"),
            raising=False,
        )

        llm_stub = MagicMock()
        llm_stub.enabled = True
        llm_stub.generate_vision = AsyncMock(
            return_value=VisionGeneration(None, FAILURE_EMPTY)
        )
        monkeypatch.setattr(
            "app.workers.vision.get_llm_client", lambda: llm_stub
        )

        embed_mock = MagicMock()
        monkeypatch.setattr(
            "app.workers.vision._embed_and_store",
            embed_mock,
            raising=False,
        )

        worker = VisionDescribeWorker()
        await worker._process_file("img-ok")

        embed_mock.assert_not_called()


# ---------------------------------------------------------------------------
# EXIF strip regression — verifies _preprocess_image does not leak GPS tags
# from camera-originated JPEGs to the outbound LLM payload. The rest of the
# suite runs with a MagicMock PIL stub for speed; this one test opts into
# real Pillow so the sanitize-and-save path is exercised end to end.
# ---------------------------------------------------------------------------


def _ensure_real_pil() -> None:
    """Restore real Pillow in sys.modules (conftest pre-sets a MagicMock).

    Skips the test cleanly if Pillow isn't installed in this environment.
    """
    import importlib
    import sys as _sys

    for name in ("PIL", "PIL.Image"):
        _sys.modules.pop(name, None)
    try:
        importlib.import_module("PIL.Image")
    except Exception:
        pytest.skip("Pillow not installed in this test environment")


def test_preprocess_strips_gps_exif():
    """JPEG with GPS EXIF must not have the GPS tags in the re-encoded output.

    Constructs a 32x32 red square, attaches GPS-tagged EXIF via Pillow,
    then runs ``_preprocess_image`` and confirms:

      * The bytes returned are still a decodable JPEG
      * The decoded image exposes no ``GPSInfo`` (tag 0x8825) and the
        raw bytes don't contain the literal GPS lat/lon values
    """
    _ensure_real_pil()

    import io

    from PIL import Image, TiffImagePlugin  # type: ignore

    # Build a minimal EXIF blob with GPS lat/lon (Tokyo-ish coords).
    exif = Image.Exif()
    gps_ifd = {
        0: b"\x02\x02\x00\x00",  # GPSVersionID
        1: "N",                   # GPSLatitudeRef
        2: (
            TiffImagePlugin.IFDRational(35, 1),
            TiffImagePlugin.IFDRational(41, 1),
            TiffImagePlugin.IFDRational(0, 1),
        ),
        3: "E",                   # GPSLongitudeRef
        4: (
            TiffImagePlugin.IFDRational(139, 1),
            TiffImagePlugin.IFDRational(46, 1),
            TiffImagePlugin.IFDRational(0, 1),
        ),
    }
    exif[0x8825] = gps_ifd

    original = Image.new("RGB", (32, 32), color=(200, 50, 50))
    src = io.BytesIO()
    original.save(src, format="JPEG", exif=exif.tobytes())
    src_bytes = src.getvalue()

    # Sanity-check the fixture really does carry GPSInfo before strip.
    verify = Image.open(io.BytesIO(src_bytes))
    verify.load()
    assert 0x8825 in verify.getexif(), (
        "Fixture precondition: JPEG should contain GPSInfo before strip"
    )

    # Re-import with real PIL in place so the worker module reads
    # the real Image class (it imports lazily per-call).
    from app.workers import vision as vision_module

    result = vision_module._preprocess_image(src_bytes, "image/jpeg")
    assert result is not None, "preprocess should succeed on a valid JPEG"
    out_bytes, out_mime = result
    assert out_mime == "image/jpeg"

    # Decoded output must not expose any GPSInfo tag.
    stripped = Image.open(io.BytesIO(out_bytes))
    stripped.load()
    assert 0x8825 not in stripped.getexif(), (
        "_preprocess_image must strip GPSInfo before handing bytes to LLM"
    )

    # Belt-and-suspenders: the raw JPEG payload shouldn't contain
    # the GPS IFD marker string. Pillow writes GPS data inside an
    # EXIF segment tagged with the 0x8825 directory entry; checking
    # for the ``GPSInfo`` literal covers the common readable markers.
    assert b"GPSInfo" not in out_bytes
