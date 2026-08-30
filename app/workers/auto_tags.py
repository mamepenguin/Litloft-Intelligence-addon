"""Auto-tagging worker using CLIP zero-shot, TF-IDF, and optional LLM.

Generates tag suggestions through a layered pipeline:

1. Always: CLIP zero-shot scoring against a curated concept vocabulary
   (plus the drive's own tag history) and TF-IDF keyword extraction
   from transcript + filename.

2. If an LLM is configured: the CLIP/TF-IDF candidates are fed to the
   LLM as grounding context, and the LLM's refined tag list is the
   final output. The LLM sees the noisy raw candidates and can
   abstract/filter them — this is the "local candidates as grounding"
   pattern rather than a simple union.

3. If no LLM is configured: CLIP + TF-IDF candidates are merged
   directly (deduplicated, capped to 10) and saved as suggestions.

Runs as a dedicated async queue processing one file at a time.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import text as sql_text

from app.config import settings
from app.database import get_litloft_db, get_search_db
from app.llm import LLMClient
from app.models import Embedding, IndexedFile, TranscriptChunk
from app.prompt_loader import render
from app.tfidf import get_tfidf_keywords_for_file
from app.workers.clip_concepts import load_file_clip_vectors, score_file_concepts
from app.workers.tag_knn import recommend_tags_by_similarity

logger = logging.getLogger(__name__)

_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ja": "- Output tags in Japanese\n",
    "en": "- Tags must be in English\n",
}

_MAX_CONTEXT_CHARS = 2000
# Cap the number of CLIP / TF-IDF candidates offered to the LLM.  Too
# many and the prompt wastes tokens on noise; too few and genuine
# signals from the local pipeline disappear.
_MAX_CLIP_CANDIDATES = 10
_MAX_TFIDF_CANDIDATES = 10
_MAX_KNN_CANDIDATES = 10
_KNN_NEIGHBORS = 20
# When no LLM is configured we surface a compact tag list. Auto-tag UX
# expects roughly five to ten tags per file, matching the LLM prompt.
_LOCAL_ONLY_TAG_LIMIT = 10
# Minimum cosine similarity for a CLIP concept to count as a candidate.
# 0.25 matches the threshold used by the semantic-search pipeline.
_CLIP_THRESHOLD = 0.25


@dataclass(frozen=True)
class TagCandidates:
    """Local-only tag candidates from CLIP zero-shot, TF-IDF, and k-NN."""

    clip: list[str]
    tfidf: list[str]
    # k-NN recommendations come from tags on CLIP-similar files that
    # the user has already tagged. This channel is empty on cold
    # start but grows more valuable as the user keeps tagging.
    knn: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        # Default empty list without a mutable default argument.
        if self.knn is None:
            object.__setattr__(self, "knn", [])

    def has_any(self) -> bool:
        return bool(self.clip or self.tfidf or self.knn)

    def merged(self, limit: int) -> list[str]:
        """Merge candidate sources, dedupe case-insensitively, cap to ``limit``.

        Priority order:
          1. k-NN — tags the user has actually applied to similar files,
             the strongest signal once the library is non-empty.
          2. CLIP zero-shot — visually grounded concepts.
          3. TF-IDF — keywords mined from noisy transcripts; lowest
             confidence channel.
        """
        seen: set[str] = set()
        out: list[str] = []
        for src in (self.knn, self.clip, self.tfidf):
            for tag in src:
                key = tag.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(tag)
                if len(out) >= limit:
                    return out
        return out


def _build_system_prompt() -> str:
    """Build system prompt with language instruction based on config."""
    lang = settings.llm.output_language
    lang_line = _LANGUAGE_INSTRUCTIONS.get(
        lang,
        f"- Output tags in {lang}\n" if lang and lang != "auto" else "",
    )
    return render(
        "auto_tags/system.jinja2",
        language_instruction=lang_line,
    )


class AutoTagsWorker:
    """Async worker that processes auto-tagging requests via a queue."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        # File ids currently being tagged. List preserves order; LLM
        # workers process one file at a time so this stays a list of
        # length 0 or 1 in practice — leaving it as a list keeps the
        # /status surface symmetric with index_manager's per-task lists.
        self._processing: list[str] = []

    def get_status(self) -> dict[str, object]:
        """Snapshot for /status: ``{waiting, processing}``.

        ``waiting`` is the queue size (per-drive policy already
        filtered at enqueue time). ``processing`` is a copy of the
        in-flight file_id list.
        """
        return {
            "waiting": self._queue.qsize(),
            "processing": list(self._processing),
        }

    async def enqueue(self, file_id: str) -> None:
        """Add a file to the auto-tagging queue.

        Per-drive policy (``intelligence.auto_tags``) is checked here
        so files in opted-out drives never enter the queue. The check
        fails open so a transient internal-API failure won't suppress
        legitimate work.
        """
        from app.policy_client import is_file_feature_enabled
        if not await is_file_feature_enabled(file_id, "auto_tags"):
            return
        await self._queue.put(file_id)

    async def enqueue_unprocessed(self) -> int:
        """Find indexed files without suggested tags and enqueue them.

        Returns:
            Number of files queued (after per-drive policy filtering).
        """
        from app.policy_client import is_feature_enabled

        with get_search_db() as session:
            rows = session.execute(
                sql_text(
                    "SELECT f.file_id, f.drive FROM indexed_files f "
                    "WHERE f.active = 1 AND f.metadata_indexed = 1 "
                    "AND f.file_id NOT IN (SELECT file_id FROM suggested_tags)"
                )
            ).fetchall()

        # Cache per-drive decisions so we don't pay one HTTP round-trip
        # per file when many files share a drive.
        drive_allowed: dict[str, bool] = {}
        count = 0
        for file_id, drive in rows:
            allowed = drive_allowed.get(drive)
            if allowed is None:
                allowed = await is_feature_enabled(drive, "auto_tags")
                drive_allowed[drive] = allowed
            if not allowed:
                continue
            await self._queue.put(file_id)
            count += 1
        return count

    async def run(self) -> None:
        """Main worker loop. Processes one file at a time."""
        while True:
            try:
                file_id = await self._queue.get()
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
                logger.error("Auto-tags worker error: %s", e)

    async def _process_file(self, file_id: str) -> None:
        """Generate and persist auto-tag suggestions for one file.

        Unlike the original LLM-only flow, this runs the CLIP/TF-IDF
        candidate pipeline first and only falls through to the LLM
        when one is configured. When the LLM is disabled we still
        produce suggestions from the local candidates.
        """
        if settings.features.auto_tags == "false":
            return

        if _has_suggested_tags(file_id):
            return

        indexed_file = _get_indexed_file(file_id)
        if indexed_file is None:
            return

        existing_tags = _get_existing_tags(file_id)
        context_type = _classify_file_type(indexed_file["file_type"])

        t_start = time.perf_counter()

        # Phase 1: always collect local candidates.
        # CLIP/TF-IDF/k-NN scoring is CPU-bound and can take tens of
        # seconds (e.g. TF-IDF corpus IDF rebuild across every active
        # file). Offloaded to a thread so it doesn't block the event
        # loop that also serves every other request in this process.
        t_cand_start = time.perf_counter()
        candidates = await asyncio.to_thread(
            _generate_candidates,
            file_id,
            context_type,
            existing_tags,
            indexed_file["drive"],
        )
        t_cand = time.perf_counter() - t_cand_start

        # Phase 2: produce the final tag list.
        llm_enabled = self._llm_client.enabled
        t_llm = 0.0
        if llm_enabled:
            t_llm_start = time.perf_counter()
            tags, model_label = await self._tags_via_llm(
                indexed_file, context_type, existing_tags, candidates
            )
            t_llm = time.perf_counter() - t_llm_start
        else:
            tags = candidates.merged(_LOCAL_ONLY_TAG_LIMIT)
            model_label = "clip+tfidf"

        t_total = time.perf_counter() - t_start
        logger.debug(
            "PROFILE auto_tags file=%s type=%s total=%.3fs "
            "candidates=%.3fs llm=%.3fs llm_enabled=%s",
            file_id, context_type, t_total, t_cand, t_llm, llm_enabled,
        )

        if not tags:
            logger.info(
                "Auto-tags: no candidates for %s (llm=%s)",
                file_id, llm_enabled,
            )
            return

        filtered = _filter_tags(tags, existing_tags)
        if not filtered:
            logger.info("Auto-tags: all tags filtered out for %s", file_id)
            return

        _save_suggested_tags(
            file_id=file_id,
            tags=filtered,
            model=model_label,
            context_type=context_type,
        )
        logger.info(
            "Auto-tags: saved %d tags for %s (model=%s)",
            len(filtered), file_id, model_label,
        )

    async def _tags_via_llm(
        self,
        indexed_file: dict,
        context_type: str,
        existing_tags: list[str],
        candidates: TagCandidates,
    ) -> tuple[list[str], str]:
        """Run the LLM with candidates as grounding context.

        Returns the parsed tag list (empty on parse failure) and the
        ``model`` label to persist, which records both the grounding
        pipeline and the LLM so we can later tell which pipeline
        produced a given suggestion.
        """
        context = _build_context(indexed_file, context_type)
        user_prompt = _build_user_prompt(
            indexed_file, context_type, context, existing_tags, candidates
        )
        raw = await self._llm_client.generate_json(
            _build_system_prompt(), user_prompt
        )
        model_suffix = settings.llm.model or "llm"
        model_label = f"clip+tfidf+{model_suffix}"

        # With json_object mode the provider returns an object, so the
        # canonical shape is ``{"tags": [...]}``. Older / non-compliant
        # providers might still emit a raw list — accept that too for
        # backward compatibility.
        tag_list: list | None = None
        if isinstance(raw, list):
            tag_list = raw
        elif isinstance(raw, dict):
            candidate_value = raw.get("tags")
            if isinstance(candidate_value, list):
                tag_list = candidate_value

        if tag_list is None:
            logger.warning(
                "Auto-tags LLM returned unusable shape for %s "
                "(type=%s); falling back to local candidates",
                indexed_file["file_id"], type(raw).__name__,
            )
            return candidates.merged(_LOCAL_ONLY_TAG_LIMIT), model_label

        tags = [t for t in tag_list if isinstance(t, str) and t.strip()]
        return tags, model_label


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def _generate_candidates(
    file_id: str,
    context_type: str,
    existing_tags: list[str],
    drive: str,
) -> TagCandidates:
    """Collect tag candidates from all local signals within one drive.

    Runs three pipelines independently:
      - CLIP zero-shot against the curated concept vocabulary
      - TF-IDF keyword extraction from transcript + filename
      - k-NN recommendation from already-tagged similar files

    Every pipeline that consults other files — the tag vocabulary, the
    neighbor search, the corpus statistics behind TF-IDF — is confined
    to ``drive``, because a drive is a security boundary and a
    suggestion must never be shaped by a library the viewer may not
    even be able to see.

    Failures in any single pipeline are logged and treated as empty
    signals rather than aborting the whole run — e.g. a document has
    no CLIP embeddings, which is not an error.

    CLIP vectors are loaded once here and shared between the CLIP and
    k-NN pipelines (both need the same file's vec_clip rows) instead
    of each pipeline fetching them independently.
    """
    clip_vectors = (
        load_file_clip_vectors(file_id)
        if context_type in ("image", "video")
        else []
    )

    t0 = time.perf_counter()
    clip_tags = _safe_clip_candidates(
        file_id, context_type, existing_tags, drive, clip_vectors
    )
    t_clip = time.perf_counter() - t0

    t0 = time.perf_counter()
    tfidf_tags = _safe_tfidf_candidates(file_id, existing_tags, drive)
    t_tfidf = time.perf_counter() - t0

    t0 = time.perf_counter()
    knn_tags = _safe_knn_candidates(
        file_id, context_type, existing_tags, drive, clip_vectors
    )
    t_knn = time.perf_counter() - t0

    logger.debug(
        "PROFILE candidates file=%s type=%s clip=%.3fs(%d) tfidf=%.3fs(%d) knn=%.3fs(%d)",
        file_id, context_type,
        t_clip, len(clip_tags),
        t_tfidf, len(tfidf_tags),
        t_knn, len(knn_tags),
    )
    return TagCandidates(clip=clip_tags, tfidf=tfidf_tags, knn=knn_tags)


def _safe_clip_candidates(
    file_id: str,
    context_type: str,
    existing_tags: list[str],
    drive: str,
    vectors: list[np.ndarray] | None = None,
) -> list[str]:
    if context_type not in ("image", "video"):
        return []
    try:
        scored = score_file_concepts(
            file_id,
            drive=drive,
            threshold=_CLIP_THRESHOLD,
            top_k=_MAX_CLIP_CANDIDATES,
            vectors=vectors,
        )
    except Exception as e:
        logger.warning("CLIP candidate scoring failed for %s: %s", file_id, e)
        return []
    existing_lower = {t.lower() for t in existing_tags}
    return [name for name, _ in scored if name.lower() not in existing_lower]


def _safe_knn_candidates(
    file_id: str,
    context_type: str,
    existing_tags: list[str],
    drive: str,
    vectors: list[np.ndarray] | None = None,
) -> list[str]:
    """Recommend tags from CLIP-similar already-tagged files in this drive.

    Silently returns an empty list for file types without CLIP
    embeddings (documents, audio) since k-NN has nothing to compare
    against. Any runtime error is logged and downgraded to "no
    candidates" so a bad neighbor query doesn't crash the worker.
    """
    if context_type not in ("image", "video"):
        return []
    try:
        scored = recommend_tags_by_similarity(
            file_id,
            drive=drive,
            k_neighbors=_KNN_NEIGHBORS,
            top_tags=_MAX_KNN_CANDIDATES,
            vectors=vectors,
        )
    except Exception as e:
        logger.warning("k-NN tag recommendation failed for %s: %s", file_id, e)
        return []
    existing_lower = {t.lower() for t in existing_tags}
    return [name for name, _ in scored if name.lower() not in existing_lower]


def _safe_tfidf_candidates(
    file_id: str,
    existing_tags: list[str],
    drive: str,
) -> list[str]:
    try:
        rows = get_tfidf_keywords_for_file(
            file_id,
            drive=drive,
            k=_MAX_TFIDF_CANDIDATES,
            # Drop words that appear in only one file — almost always
            # a Whisper mis-transcription rather than a real topic.
            min_doc_freq=2,
        )
    except Exception as e:
        logger.warning("TF-IDF candidate extraction failed for %s: %s", file_id, e)
        return []
    existing_lower = {t.lower() for t in existing_tags}
    out: list[str] = []
    for row in rows:
        word = row.get("word") if isinstance(row, dict) else None
        if not isinstance(word, str) or not word:
            continue
        if word.lower() in existing_lower:
            continue
        out.append(word)
    return out


# ---------------------------------------------------------------------------
# Filtering + DB helpers
# ---------------------------------------------------------------------------


def _filter_tags(tags: list, existing_tags: list[str]) -> list[str]:
    """Normalize a raw tag list: strings only, non-empty, not already applied."""
    existing_lower = {t.lower() for t in existing_tags}
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        if not isinstance(t, str):
            continue
        trimmed = t.strip()
        if not trimmed:
            continue
        key = trimmed.lower()
        if key in existing_lower or key in seen:
            continue
        seen.add(key)
        out.append(trimmed)
    return out


def _has_suggested_tags(file_id: str) -> bool:
    """Check if suggested tags already exist for a file."""
    with get_search_db() as session:
        row = session.execute(
            sql_text("SELECT 1 FROM suggested_tags WHERE file_id = :fid"),
            {"fid": file_id},
        ).fetchone()
        return row is not None


def _get_indexed_file(file_id: str) -> dict | None:
    """Get indexed file info from search DB."""
    with get_search_db() as session:
        f = (
            session.query(IndexedFile)
            .filter(
                IndexedFile.file_id == file_id,
                IndexedFile.active.is_(True),
            )
            .first()
        )
        if f is None:
            return None
        return {
            "file_id": f.file_id,
            "drive": f.drive,
            "filename": f.filename,
            "file_type": f.file_type,
            "mime_type": f.mime_type,
            "title": f.title,
            "description": f.description,
            "tags_text": f.tags_text,
        }


def _get_existing_tags(file_id: str) -> list[str]:
    """Get existing tags from Litloft DB for a file."""
    try:
        with get_litloft_db() as session:
            rows = session.execute(
                sql_text(
                    "SELECT t.name FROM tags t "
                    "JOIN file_tags ft ON t.id = ft.tag_id "
                    "WHERE ft.file_id = :file_id"
                ),
                {"file_id": file_id},
            ).fetchall()
            return [row[0] for row in rows]
    except Exception:
        return []


def _classify_file_type(file_type: str) -> str:
    """Classify file type into context categories."""
    if file_type == "video":
        return "video"
    if file_type == "audio":
        return "audio"
    if file_type == "image":
        return "image"
    if file_type in ("document", "text"):
        return "document"
    return "other"


def _build_context(indexed_file: dict, context_type: str) -> str:
    """Build additional context (transcript/captions/text) based on file type."""
    file_id = indexed_file["file_id"]
    parts: list[str] = []

    if context_type in ("video", "audio"):
        transcript = _get_transcript_text(file_id)
        if transcript:
            parts = [*parts, f"Transcript:\n{transcript}"]
    elif context_type == "image":
        captions = _get_blip_captions(file_id)
        if captions:
            parts = [*parts, f"Image captions:\n{captions}"]
        # Vision LLM description is a richer, structured account of the
        # image than BLIP's short caption — feed it alongside so the
        # tag-generation LLM sees both the terse label and the detailed
        # scene. Spec: 2026-04-23-intelligence-vision-describe.md §非目標
        # ("auto_tags は既存経路を維持、vision 記述を入力として間接的に恩恵を受ける").
        vision_desc = _get_visual_description(file_id)
        if vision_desc:
            parts = [*parts, f"Visual description:\n{vision_desc}"]
    elif context_type == "document":
        text_content = _get_text_content(file_id)
        if text_content:
            parts = [*parts, f"Content:\n{text_content}"]

    return "\n".join(parts)


def _get_transcript_text(file_id: str) -> str:
    """Get transcript text for a file, truncated to max context chars."""
    with get_search_db() as session:
        chunks = (
            session.query(TranscriptChunk)
            .filter(TranscriptChunk.file_id == file_id)
            .order_by(TranscriptChunk.chunk_index)
            .all()
        )
        if not chunks:
            return ""
        text = " ".join(c.text for c in chunks)
        return text[:_MAX_CONTEXT_CHARS]


def _get_blip_captions(file_id: str) -> str:
    """Get BLIP captions from embedding content_preview."""
    with get_search_db() as session:
        embeddings = (
            session.query(Embedding)
            .filter(
                Embedding.file_id == file_id,
                Embedding.embedding_type == "blip_caption",
            )
            .all()
        )
        if not embeddings:
            return ""
        return " ".join(e.content_preview for e in embeddings if e.content_preview)


def _get_visual_description(file_id: str) -> str:
    """Return the stored vision LLM description for an image file.

    Only ``visual_description_status = 'success'`` rows contribute
    context — failed / pending / unsupported rows would inject noise or
    empty strings, so they're filtered out here rather than in the
    caller. Truncated at ``_MAX_CONTEXT_CHARS`` to match the transcript
    / text_content paths and avoid blowing the LLM prompt budget on
    verbose descriptions.
    """
    with get_search_db() as session:
        row = session.execute(
            sql_text(
                "SELECT visual_description FROM file_summaries "
                "WHERE file_id = :fid AND visual_description_status = 'success'"
            ),
            {"fid": file_id},
        ).fetchone()
    if row is None or not row[0]:
        return ""
    text = str(row[0])
    return text[:_MAX_CONTEXT_CHARS]


def _get_text_content(file_id: str) -> str:
    """Get text content chunks for a file, truncated to max context chars."""
    with get_search_db() as session:
        embeddings = (
            session.query(Embedding)
            .filter(
                Embedding.file_id == file_id,
                Embedding.embedding_type == "text_content",
            )
            .order_by(Embedding.timestamp_start.asc().nullsfirst())
            .all()
        )
        if not embeddings:
            return ""
        text = " ".join(e.content_preview for e in embeddings if e.content_preview)
        return text[:_MAX_CONTEXT_CHARS]


def _build_user_prompt(
    indexed_file: dict,
    context_type: str,
    context: str,
    existing_tags: list[str],
    candidates: TagCandidates | None = None,
) -> str:
    """Build the user prompt for LLM tag generation.

    When ``candidates`` is provided (non-empty CLIP or TF-IDF lists),
    the prompt includes a "Reference candidates" section so the LLM can use them
    as grounding signals. The instruction deliberately allows the LLM
    to override or ignore them — they're hints, not ground truth.
    """
    title = (
        indexed_file["title"]
        if indexed_file["title"]
        and indexed_file["title"] != indexed_file["filename"]
        else ""
    )

    candidate_block = ""
    if candidates is not None and candidates.has_any():
        candidate_lines: list[str] = ["", "[Reference candidates]"]
        # k-NN first because "similar tagged files" is the strongest
        # signal for "what the user considers relevant".
        if candidates.knn:
            candidate_lines.append(
                "Tags from similar files: " + ", ".join(candidates.knn)
            )
        if candidates.clip:
            candidate_lines.append(
                "Image analysis candidates: " + ", ".join(candidates.clip)
            )
        if candidates.tfidf:
            candidate_lines.append(
                "Keyword extraction candidates: " + ", ".join(candidates.tfidf)
            )
        candidate_lines.append(
            "Note: These candidates are for reference. Prefer more appropriate tags if available."
        )
        candidate_block = "\n".join(candidate_lines)

    tags_display = ", ".join(existing_tags) if existing_tags else "none"
    return render(
        "auto_tags/user.jinja2",
        filename=indexed_file["filename"],
        context_type=context_type,
        title=title,
        description=indexed_file["description"] or "",
        tags_text=indexed_file["tags_text"] or "",
        context=context or "",
        candidate_block=candidate_block,
        tags_display=tags_display,
    )


def _save_suggested_tags(
    file_id: str,
    tags: list[str],
    model: str,
    context_type: str,
) -> None:
    """Save suggested tags to the search database."""
    now = datetime.now(UTC).isoformat()
    tags_json = json.dumps(tags, ensure_ascii=False)

    with get_search_db() as session:
        session.execute(
            sql_text(
                "INSERT OR REPLACE INTO suggested_tags "
                "(file_id, tags, model, context_type, created_at, status) "
                "VALUES (:file_id, :tags, :model, :context_type, :created_at, 'pending')"
            ),
            {
                "file_id": file_id,
                "tags": tags_json,
                "model": model,
                "context_type": context_type,
                "created_at": now,
            },
        )
