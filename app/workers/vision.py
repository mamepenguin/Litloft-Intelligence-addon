"""Vision-LLM image description worker.

Spec: docs/superpowers/specs/2026-04-23-intelligence-vision-describe.md

The worker reads an image file, hands its bytes to a vision-capable LLM
via :meth:`app.llm.LLMClient.generate_vision` (or the Ollama equivalent),
and persists the result in ``file_summaries.visual_description*``. On
success it also registers an embeddings row of type
``"vision_description"`` so hybrid retrieval can index the description.

Status lifecycle (per file_summaries row):

* NULL            → no attempt yet
* "pending"       → worker claimed the file, LLM call in flight
* "success"       → description persisted + embedding registered
* "failed"        → transient LLM / decode failure; retryable
* "unsupported"   → the model was measured, by capability probe, not to
                    accept image content. Sticky for this
                    ``visual_description_model`` on automatic paths;
                    an explicit user request overrides it.

A provider rejection on its own never reaches "unsupported": an absent
model and an unreadable image produce the same 400/404 as a model that
cannot see, so ``app.llm`` probes before it names the cause.

Non-goals in Phase 1 (see spec):

* Video frame extraction — CLIP already covers that channel.
* PDF page images — cost scales per-page.
* Writing ``files.description`` on the core side — we treat that
  column as user-curated.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text as sql_text
from sqlalchemy.exc import OperationalError

import app.config as config
from app.config import settings
from app.database import get_search_db
from app.llm import (
    FAILURE_REQUEST_FAILED,
    FAILURE_VISION_UNSUPPORTED,
    VisionGeneration,
)
from app.models import Embedding, IndexedFile
from app.policy_client import is_file_feature_enabled

logger = logging.getLogger(__name__)


# Defuse zip-bomb style PNG/JPEG inputs: Pillow raises
# DecompressionBombError when an image claims pixel counts beyond this cap.
# The limit is generous enough for reasonable phone/DSLR photos (50 MP) but
# refuses pathological crafted files that would exhaust RAM on decode.
try:
    from PIL import Image as _PILImage  # type: ignore

    _PILImage.MAX_IMAGE_PIXELS = 50_000_000
except Exception:
    # Pillow missing (pure unit-test stubs) — nothing to configure.
    pass


# Hard ceiling on raw file bytes. Even well-formed images larger than this
# indicate operator misconfiguration (RAW files, multi-hundred-MP stitched
# panoramas) and make preprocessing expensive; refuse early with a clean
# "failed" status instead of OOMing the worker.
_MAX_IMAGE_FILE_BYTES = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_content_preview(text: str) -> str:
    """Match the existing content_preview convention (first ~200 chars)."""
    return (text or "")[:200]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _is_image_mime(mime_type: str | None) -> bool:
    """Accept any ``image/*`` MIME. HEIC is handled via explicit conversion."""
    return bool(mime_type) and mime_type.lower().startswith("image/")


def get_llm_client() -> Any:
    """Thin indirection so worker tests can monkeypatch the LLM.

    Mirrors the pattern used by other workers (via
    ``app.dependencies.get_llm_client``) but kept local so tests don't
    need to initialise the full dependency graph.
    """
    from app.dependencies import get_llm_client as _get

    return _get()


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


_MAX_EDGE_PX = 1024
_JPEG_QUALITY = 85


def _preprocess_image(raw: bytes, mime_type: str) -> tuple[bytes, str] | None:
    """Downscale + re-encode so vision API payloads stay bounded.

    Falls back to returning the raw bytes when Pillow isn't importable
    (e.g. unit-test stubs) so call sites don't have to branch. HEIC is
    converted to JPEG via pillow-heif (same path as the host thumbnail
    pipeline). Returns ``None`` when decode fails altogether.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return (raw, mime_type)

    # Defensive guard: unit-test environments replace PIL with a
    # MagicMock, so Image.open(...).size is a MagicMock that fails to
    # unpack. Detect that shape and fall back to passing the raw bytes
    # through unchanged. Real Pillow installs expose a real class on
    # ``Image.Image`` so the attribute chain is trustworthy.
    if not hasattr(Image, "LANCZOS") or not isinstance(
        getattr(Image, "LANCZOS"), int
    ):
        return (raw, mime_type)

    try:
        if mime_type.lower() in ("image/heic", "image/heif"):
            try:
                import pillow_heif  # type: ignore

                pillow_heif.register_heif_opener()
            except Exception:
                # Without the opener we can't decode HEIC reliably.
                return None

        import io

        img = Image.open(io.BytesIO(raw))
        # ``img.load()`` is where Pillow actually decodes pixels and
        # where DecompressionBombError can fire — catch it (and other
        # PIL errors) as a failed decode rather than crashing the worker.
        img.load()
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        w, h = img.size
        long_edge = max(w, h)
        if long_edge > _MAX_EDGE_PX:
            scale = _MAX_EDGE_PX / float(long_edge)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img = img.resize(new_size, Image.LANCZOS)

        # Strip EXIF (GPS coords, camera serial, timestamps, thumbnails)
        # before handing the image to an external LLM. Home-photo corpora
        # routinely include geotags; we don't want to leak them to third-
        # party vision APIs. Rebuilding the image from raw pixels is the
        # most robust way to guarantee no info segments survive — this
        # also drops any HEIC-side metadata that pillow-heif surfaced
        # via ``img.info``.
        sanitized = Image.new(img.mode, img.size)
        sanitized.paste(img)

        buf = io.BytesIO()
        # Pass ``exif=b""`` as a belt-and-suspenders guard in case a
        # future Pillow version decides to carry EXIF across ``paste``.
        sanitized.save(buf, format="JPEG", quality=_JPEG_QUALITY, exif=b"")
        return (buf.getvalue(), "image/jpeg")
    except Exception as e:
        # Covers DecompressionBombError, UnidentifiedImageError, OSError,
        # and anything else PIL raises while decoding malicious or
        # malformed inputs. Returning None is interpreted by the worker
        # as a "failed" status — no crash, no retry storm.
        logger.warning("vision preprocess failed: %s", type(e).__name__)
        return None


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _load_image_bytes(file_id: str) -> tuple[bytes, str] | None:
    """Fetch raw image bytes + mime_type for ``file_id`` from disk.

    Looks the file up in the local index (for ``file_path`` + mime), then
    reads from the validated path. Returns ``None`` when the file is
    unknown, outside an allowed base dir, or unreadable. Callers treat
    that as a "failed" status outcome.

    Tests override this helper via monkeypatch so they never touch the
    filesystem.
    """
    try:
        with get_search_db() as session:
            row = (
                session.query(
                    IndexedFile.file_path,
                    IndexedFile.mime_type,
                    IndexedFile.file_size,
                )
                .filter(
                    IndexedFile.file_id == file_id,
                    IndexedFile.active.is_(True),
                )
                .first()
            )
    except OperationalError as e:
        # Missing table / schema drift in very narrow test harnesses.
        # Treat as "cannot load" rather than crashing the worker loop.
        logger.warning("vision: index lookup failed (%s)", type(e).__name__)
        return None
    if row is None:
        return None
    file_path = row[0]
    mime_type = row[1] or (mimetypes.guess_type(file_path)[0] or "")
    file_size = row[2] if row[2] is not None else 0

    # Refuse oversized files before touching the filesystem. A 200-MP RAW
    # would pass Pillow's pixel cap but still make the vision payload
    # hugely expensive; bailing here keeps the worker predictable.
    if file_size and file_size > _MAX_IMAGE_FILE_BYTES:
        logger.warning(
            "vision: file %s exceeds max bytes (%d > %d); refusing",
            file_id, file_size, _MAX_IMAGE_FILE_BYTES,
        )
        return None

    if not config.validate_file_path(file_path):
        return None
    try:
        with open(file_path, "rb") as f:
            return (f.read(), mime_type)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _ensure_summary_row(session: Any, file_id: str) -> None:
    """Insert a placeholder row so UPDATEs always land.

    ``file_summaries`` has NOT NULL columns that predate vision (short
    / long / model / context_type / context_chars / was_truncated /
    status / created_at). The vision feature only writes the four
    ``visual_description*`` columns, so we seed defaults for the legacy
    columns when no row exists — consistent with how auto_tags /
    summaries placeholders are handled elsewhere.
    """
    existing = session.execute(
        sql_text("SELECT 1 FROM file_summaries WHERE file_id = :fid"),
        {"fid": file_id},
    ).fetchone()
    if existing is not None:
        return
    session.execute(
        sql_text(
            "INSERT INTO file_summaries "
            "(file_id, short_summary, long_summary, model, context_type, "
            "context_chars, was_truncated, status, created_at) "
            "VALUES (:fid, '', '', '', 'image', 0, 0, 'hidden', :now)"
        ),
        {"fid": file_id, "now": _now_iso()},
    )


def _fetch_existing_vision(
    session: Any, file_id: str
) -> tuple[str | None, str | None] | None:
    """Return ``(status, model)`` for the current vision row, or None."""
    row = session.execute(
        sql_text(
            "SELECT visual_description_status, visual_description_model "
            "FROM file_summaries WHERE file_id = :fid"
        ),
        {"fid": file_id},
    ).fetchone()
    if row is None:
        return None
    return (row[0], row[1])


def _write_status(
    session: Any,
    file_id: str,
    *,
    description: str | None,
    status: str,
    model: str,
    generated_at: str | None,
    error: str | None = None,
) -> None:
    """Atomic UPDATE of the vision columns.

    ``error`` is written on every transition, cleared included, so a
    reason can never outlive the attempt that produced it and be read
    against a later one.
    """
    _ensure_summary_row(session, file_id)
    session.execute(
        sql_text(
            "UPDATE file_summaries SET "
            "visual_description = :desc, "
            "visual_description_status = :status, "
            "visual_description_model = :model, "
            "visual_description_generated_at = :gen_at, "
            "visual_description_error = :error "
            "WHERE file_id = :fid"
        ),
        {
            "fid": file_id,
            "desc": description,
            "status": status,
            "model": model,
            "gen_at": generated_at,
            "error": error,
        },
    )


def _mark_pending(session: Any, file_id: str) -> None:
    """Move the row to ``pending`` without touching the stored result.

    Only the status changes, and the reason that belonged to the last
    attempt is cleared with it. The description, its model and its
    timestamp are left alone: they still describe the last completed
    run, and accepting a file is not a reason to destroy what is
    already there — a bulk accept would otherwise empty every row it
    touched before a single LLM call had been made.
    """
    _ensure_summary_row(session, file_id)
    session.execute(
        sql_text(
            "UPDATE file_summaries SET "
            "visual_description_status = 'pending', "
            "visual_description_error = NULL "
            "WHERE file_id = :fid"
        ),
        {"fid": file_id},
    )


def _clear_vision_embeddings(session: Any, file_id: str) -> None:
    """Remove prior ``vision_description`` embeddings for this file.

    Regenerate replaces the description wholesale, so previous rows must
    disappear from both the relational ``embeddings`` table and any
    backing vec table. We resolve the IDs first so the vec-table delete
    is scoped precisely.
    """
    rows = (
        session.query(Embedding)
        .filter(
            Embedding.file_id == file_id,
            Embedding.embedding_type == "vision_description",
        )
        .all()
    )
    for emb in rows:
        # Canonical production value is "vec_text"; any non-canonical
        # name is silently skipped (embedding metadata row still drops
        # below). We narrow the vec DELETE exception to OperationalError
        # so genuine programming bugs surface instead of being swallowed.
        table = emb.vector_table or ""
        if table.startswith("vec_"):
            try:
                session.execute(
                    sql_text(f"DELETE FROM {table} WHERE embedding_id = :id"),
                    {"id": emb.id},
                )
            except OperationalError as e:
                # Missing vec table (e.g. narrow test harness that skipped
                # vec fixture creation). Metadata row still cleaned below.
                logger.warning(
                    "vision: vec delete failed for %s (%s)",
                    table, type(e).__name__,
                )
        session.delete(emb)


def _embed_and_store(file_id: str, description_text: str) -> None:
    """Embed ``description_text`` and register a ``vision_description`` row.

    Separated from the worker body so tests can stub it out without
    pulling the full embedder (sentence-transformers) into scope. The
    production implementation mirrors the refine worker's re-embed path.
    """
    if not description_text.strip():
        return

    try:
        from app.workers.embedder import embed_passages
    except Exception as e:
        logger.warning(
            "vision: embedder unavailable (%s); description saved without embedding", e
        )
        return

    try:
        vectors = embed_passages([description_text])
    except Exception as e:
        logger.warning("vision: embed_passages failed (%s)", type(e).__name__)
        return
    if not vectors:
        return
    vec = vectors[0]

    import uuid

    embedding_id = f"vd_{file_id}_{uuid.uuid4().hex[:8]}"
    with get_search_db() as session:
        # Replace any existing rows so regenerate doesn't leave stale
        # vectors pointing at an outdated description.
        _clear_vision_embeddings(session, file_id)
        session.add(
            Embedding(
                id=embedding_id,
                file_id=file_id,
                embedding_type="vision_description",
                content_preview=_image_content_preview(description_text),
                vector_table="vec_text",
            )
        )
        session.flush()
        try:
            session.execute(
                sql_text(
                    "INSERT INTO vec_text(embedding_id, vector) VALUES(:id, :vec)"
                ),
                {"id": embedding_id, "vec": vec.tobytes()},
            )
        except Exception as e:
            logger.warning(
                "vision: vec_text insert failed: %s", type(e).__name__,
            )


# ---------------------------------------------------------------------------
# WebSocket progress
# ---------------------------------------------------------------------------


async def _emit_ws_event(event: str, data: dict) -> None:
    """Best-effort progress event emission — mirrors refine worker."""
    logger.info("vision-event %s %s", event, data)
    base = os.environ.get(
        "HOMEVAULT_INTERNAL_API_URL", "http://backend:8000/api/internal"
    )
    url = f"{base}/addon-events"
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(url, json={"event": event, "data": data})
    except Exception:
        return


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class VisionDescribeWorker:
    """Async worker: accepts file_ids, produces LLM image descriptions.

    Structure mirrors :class:`app.workers.refine` — a simple FIFO queue
    drained one file at a time. Image-LLM calls are comparatively slow
    and network-bound, so concurrency is intentionally single-threaded:
    an operator running vision on hundreds of files should spread the
    load across time rather than starve text-channel LLM calls.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._processing: list[str] = []
        # Mirrors the queue's contents, because an asyncio.Queue cannot
        # be asked whether it holds something. Needed so a second
        # request for a file already waiting is recognised as the same
        # work rather than bought twice.
        self._queued: set[str] = set()

    def get_status(self) -> dict[str, object]:
        """Snapshot for /status: ``{waiting, processing}``."""
        return {
            "waiting": self._queue.qsize(),
            "processing": list(self._processing),
        }

    # -- Enqueue --------------------------------------------------------

    async def _should_accept(
        self, file_id: str, *, manual: bool = False
    ) -> tuple[bool, str | None]:
        """Shared gate: feature flag + vision_model + drive policy + mime + stickiness.

        Returns ``(accepted, reason)`` so a router can say which gate
        turned the request away instead of accepting work it discarded.

        ``manual`` skips the stickiness checks and nothing else. Those
        checks exist to stop automatic sweeps from re-spending on a
        settled outcome; a person asking for this file has already
        decided it is worth spending. Every other gate — feature,
        policy, mime, active, and work already in flight — still
        applies, because those are not about cost.
        """
        if not config.is_vision_describe_available(settings):
            return False, "feature_unavailable"

        # The worker task only runs when the LLM client is enabled
        # (app/main.py), so accepting work here in that state would
        # queue it for nobody: the row would be marked pending and stay
        # that way, and startup recovery lives behind the same gate and
        # would not free it either. Config can satisfy
        # is_vision_describe_available while the client is disabled —
        # provider "disabled", or an empty base_url / text model.
        try:
            llm_enabled = bool(getattr(get_llm_client(), "enabled", False))
        except Exception as e:
            logger.warning(
                "vision: LLM client unavailable (%s); refusing to enqueue %s",
                type(e).__name__, file_id,
            )
            return False, "llm_unavailable"
        if not llm_enabled:
            return False, "llm_unavailable"

        # Fail-closed on policy lookup failure: the file stays in the
        # queue mentally (caller can retry) and we never accidentally
        # run vision on a drive whose operator has opted out. Read paths
        # in the router remain fail-open since they only surface state.
        try:
            enabled = await is_file_feature_enabled(file_id, "vision_describe")
        except Exception as e:
            logger.warning(
                "vision: policy lookup failed for %s (%s); refusing "
                "to enqueue (fail-closed)",
                file_id, type(e).__name__,
            )
            return False, "policy_unavailable"
        if not enabled:
            return False, "policy_off"

        with get_search_db() as session:
            file_row = (
                session.query(IndexedFile)
                .filter(
                    IndexedFile.file_id == file_id,
                    IndexedFile.active.is_(True),
                )
                .first()
            )
            if file_row is None:
                return False, "file_not_found"
            if not _is_image_mime(file_row.mime_type):
                return False, "not_an_image"

            state = _fetch_existing_vision(session, file_id)

        # Already ours to do. Manual does not override this: asking
        # twice for the same work does not make it happen sooner, it
        # just buys a second LLM call whose result overwrites the
        # first. A ``pending`` row that this worker does not hold is a
        # different thing — a previous process died mid-flight — and
        # stays retryable.
        if file_id in self._queued or file_id in self._processing:
            return False, "already_queued"

        if state is not None and not manual:
            status, stored_model = state
            same_model = (stored_model or "") == (settings.llm.vision_model or "")
            # "success" is sticky for the SAME vision_model: re-running
            # would just overwrite an identical description and burn
            # LLM budget. Swap the model and we retry so the new model
            # gets a chance.
            if status == "success" and same_model:
                return False, "already_described"
            # "unsupported" is sticky only for the SAME vision_model. If
            # the operator swapped models we retry — the new model may
            # handle images even if the old one didn't.
            if status == "unsupported" and same_model:
                return False, "unsupported_sticky"

        return True, None

    async def enqueue(self, file_id: str, *, manual: bool = False) -> dict:
        """Queue ``file_id`` for vision description.

        Returns ``{"accepted": bool, "reason": str | None}``. The reason
        lets a router answer 409 with what actually happened rather than
        reporting acceptance for work that was dropped.

        ``manual=True`` is for an explicit user request and overrides
        the stickiness gates; automatic callers leave it False.
        """
        accepted, reason = await self._should_accept(file_id, manual=manual)
        if not accepted:
            return {"accepted": False, "reason": reason}
        # Record "pending" here rather than when processing starts, so
        # the state is true for everyone the moment it becomes true.
        # A file behind others in the queue would otherwise keep
        # reporting its previous outcome, and a caller polling for the
        # change it just asked for would read the old answer and
        # conclude nothing happened.
        with get_search_db() as session:
            _mark_pending(session, file_id)
        self._queued.add(file_id)
        await self._queue.put(file_id)
        return {"accepted": True, "reason": None}

    async def requeue_abandoned(self) -> int:
        """Re-queue rows left ``pending`` by a process that is gone.

        Acceptance records ``pending`` so the state is true the moment
        it becomes true, but the queue holding that work lives only as
        long as the process. A restart therefore leaves every waiting
        file marked pending with nothing left to run it — a whole bulk
        run stranded by one deploy.

        Called at startup, where the reading is unambiguous: this
        worker holds nothing yet, so a pending row can only be someone
        else's abandoned claim. It is deliberately not a periodic sweep,
        which would race the queue it is meant to repair.

        Runs regardless of feature mode. Resuming work already paid for
        is not the same decision as starting new work.
        """
        with get_search_db() as session:
            rows = session.execute(
                sql_text(
                    "SELECT f.file_id FROM indexed_files f "
                    "JOIN file_summaries s ON s.file_id = f.file_id "
                    "WHERE f.active = 1 "
                    "AND f.mime_type LIKE 'image/%' "
                    "AND s.visual_description_status = 'pending'"
                )
            ).fetchall()

        queued = 0
        for (file_id,) in rows:
            # manual=True: the row is already pending, which the
            # stickiness gates would otherwise read as nothing to do.
            if (await self.enqueue(file_id, manual=True))["accepted"]:
                queued += 1
        return queued

    async def enqueue_unprocessed(self) -> int:
        """Sweep already-indexed images that have no description yet.

        Mirrors :meth:`AutoTagsWorker.enqueue_unprocessed` so an operator
        flipping ``features.vision_describe`` to ``on_index`` after files
        are already CLIP-indexed gets coverage on startup. The on-the-fly
        on_index hook in ``IndexManager._clip_worker`` only fires for
        newly-indexed files; without this sweep, prior images would
        never auto-describe.

        Selects active image rows whose ``visual_description_status``
        is NULL (never attempted). Files with ``success`` /
        ``pending`` / ``unsupported`` / ``failed`` are skipped here —
        ``failed`` is left as a manual-retry case (matches the UI
        semantics in ``VisualDescriptionSection``) and ``pending`` /
        ``unsupported`` are owned by the worker's stickiness rules.

        Per-drive policy is enforced by ``enqueue`` via
        ``_should_accept`` so a drive whose ``vision_describe`` policy
        is OFF won't see files queued.

        Returns the number of files actually accepted onto the queue.
        """
        with get_search_db() as session:
            rows = session.execute(
                sql_text(
                    "SELECT f.file_id FROM indexed_files f "
                    "LEFT JOIN file_summaries s ON s.file_id = f.file_id "
                    "WHERE f.active = 1 "
                    "AND f.mime_type LIKE 'image/%' "
                    "AND s.visual_description_status IS NULL"
                )
            ).fetchall()

        queued = 0
        for (file_id,) in rows:
            if (await self.enqueue(file_id))["accepted"]:
                queued += 1
        return queued

    # -- Processing -----------------------------------------------------

    async def run(self) -> None:
        """Main loop — process one file at a time."""
        while True:
            try:
                file_id = await self._queue.get()
                self._queued.discard(file_id)
                self._processing.append(file_id)
                try:
                    await self._process_file(file_id)
                finally:
                    try:
                        self._processing.remove(file_id)
                    except ValueError:
                        pass
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Vision worker error: %s", type(e).__name__)

    async def _process_file(self, file_id: str) -> None:
        """Generate a description for ``file_id`` and persist the outcome.

        Status transitions (NULL → pending → success/failed/unsupported)
        are written at deterministic points so the UI can poll and see
        progress. A best-effort WS emission happens at start / end.
        """
        vision_model = settings.llm.vision_model or ""
        llm = None
        try:
            llm = get_llm_client()
        except Exception as e:
            logger.warning(
                "vision: LLM client unavailable (%s); aborting %s",
                type(e).__name__, file_id,
            )
            return
        if not getattr(llm, "enabled", False):
            return

        await _emit_ws_event(
            "intelligence.vision_describe.started",
            {"file_id": file_id, "model": vision_model},
        )

        # Mark pending so concurrent readers see "in flight" instead of
        # the previous state. Redundant when the file arrived through
        # ``enqueue``, which already did it, but ``_process_file`` is
        # also driven directly and must not depend on that.
        with get_search_db() as session:
            _mark_pending(session, file_id)

        loaded = _load_image_bytes(file_id)
        if loaded is None:
            with get_search_db() as session:
                _write_status(
                    session,
                    file_id,
                    description=None,
                    status="failed",
                    model=vision_model,
                    generated_at=None,
                    error="load",
                )
            await _emit_ws_event(
                "intelligence.vision_describe.failed",
                {"file_id": file_id, "reason": "load"},
            )
            return

        raw_bytes, mime_type = loaded
        preprocessed = _preprocess_image(raw_bytes, mime_type)
        if preprocessed is None:
            with get_search_db() as session:
                _write_status(
                    session,
                    file_id,
                    description=None,
                    status="failed",
                    model=vision_model,
                    generated_at=None,
                    error="decode",
                )
            await _emit_ws_event(
                "intelligence.vision_describe.failed",
                {"file_id": file_id, "reason": "decode"},
            )
            return

        image_bytes, image_mime = preprocessed

        try:
            result = await llm.generate_vision(
                image_bytes,
                image_mime,
                "Describe this image in detail.",
                output_language=settings.llm.output_language,
            )
        except Exception as e:
            logger.warning(
                "vision: LLM call raised for %s (%s)",
                file_id, type(e).__name__,
            )
            result = VisionGeneration(None, FAILURE_REQUEST_FAILED)

        # Only a measured verdict about the model is latched. Every
        # other reason clears on its own — the operator pulls the model,
        # or the next file is one the provider can read — so it stays a
        # retryable failure.
        if result.failure == FAILURE_VISION_UNSUPPORTED:
            with get_search_db() as session:
                _write_status(
                    session,
                    file_id,
                    description=None,
                    status="unsupported",
                    model=vision_model,
                    generated_at=None,
                    error=FAILURE_VISION_UNSUPPORTED,
                )
            await _emit_ws_event(
                "intelligence.vision_describe.unsupported",
                {"file_id": file_id, "model": vision_model},
            )
            return

        description = (result.text or "").strip()
        # Success needs both a description and a clean call. Text cut off
        # by the token ceiling arrives non-empty and would otherwise be
        # stored as success — which is sticky for this model, so raising
        # ``vision_max_tokens`` would never re-run the file, and half a
        # sentence would sit in retrieval as though it were the whole
        # description.
        if result.failure is not None or not description:
            with get_search_db() as session:
                _write_status(
                    session,
                    file_id,
                    description=None,
                    status="failed",
                    model=vision_model,
                    generated_at=None,
                    error=result.failure or FAILURE_REQUEST_FAILED,
                )
            await _emit_ws_event(
                "intelligence.vision_describe.failed",
                {
                    "file_id": file_id,
                    "reason": result.failure or FAILURE_REQUEST_FAILED,
                },
            )
            return

        generated_at = _now_iso()
        with get_search_db() as session:
            _write_status(
                session,
                file_id,
                description=description,
                status="success",
                model=vision_model,
                generated_at=generated_at,
            )

        # Embedding registration is best-effort — a failure here leaves
        # the description queryable via keyword search (fts) but not
        # dense retrieval; we still consider the overall outcome success.
        try:
            _embed_and_store(file_id, description)
        except Exception as e:
            logger.warning(
                "vision: embed_and_store failed for %s (%s)",
                file_id, type(e).__name__,
            )

        # Approach B parity: refresh the file's metadata embedding so
        # the new visual_description contributes to hierarchical RAG
        # Stage 1 ranking the same way ``long_summary`` does for
        # transcribable / textual files (see
        # ``app.workers.summaries._save_summary``). Failure here MUST
        # NOT roll back the durable description write — log and move
        # on; the next ``index_metadata_batch`` pass picks the new
        # text up.
        try:
            from app.workers.metadata import index_metadata_batch
            await asyncio.to_thread(index_metadata_batch, [file_id])
        except Exception as e:  # noqa: BLE001 — never fail vision describe
            logger.warning(
                "vision: metadata re-embed after describe failed for %s (%s)",
                file_id, type(e).__name__,
            )

        await _emit_ws_event(
            "intelligence.vision_describe.succeeded",
            {
                "file_id": file_id,
                "model": vision_model,
                "generated_at": generated_at,
            },
        )


__all__ = [
    "VisionDescribeWorker",
    "_clear_vision_embeddings",
    "_embed_and_store",
    "_ensure_summary_row",
    "_fetch_existing_vision",
    "_load_image_bytes",
    "_mark_pending",
    "_preprocess_image",
    "_write_status",
    "get_llm_client",
    "is_file_feature_enabled",
    "settings",
]
