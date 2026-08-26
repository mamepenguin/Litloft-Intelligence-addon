"""SIRA-style retrieval keyword expansion worker.

The LLM is shown a file's content and asked to predict the synonyms,
abbreviations and alternate names a user might search by. The predicted
keywords go into a dedicated FTS surface (``fts_retrieval_keywords``)
which file search and (later) Ask retrieval UNION into their FTS side.
The original document text is unchanged; this is purely a search-index
enrichment.

Why a separate worker (rather than extending auto_tags or summaries):

* Different output shape — keyword strings tuned for FTS recall, not
  display-quality tags or human-readable summaries.
* Different signal-vs-display contract — these keywords are tier-3 in
  the 3-tier model (hako ``SKiYgE6GtttlQW7fEwAY6``); they shape
  retrieval ranking but are never shown in citations. Mixing them
  with display-facing artefacts would erode that boundary.
* Different policy gate — runs on its own ``features.retrieval_keywords``
  policy so operators can enable/disable independently of summaries
  and auto_tags.

Failure modes (all fall through to a silent no-op for the file):

* File is not indexed / no usable content → skip.
* Per-drive policy is off / ``manual`` → skip via enqueue gate.
* LLM disabled / unavailable / throws → skip.
* LLM returns malformed JSON / empty keywords array → skip.
* Blocklist + rarity filter empties the result → skip.

A skip never marks the file as "tried", so a later regenerate or a
policy flip from manual to on_index can re-attempt the same file.

Spec: docs/superpowers/specs/2026-05-14-sira-retrieval-keywords.md.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from sqlalchemy import text as sql_text

from app.config import settings
from app.database import (
    get_search_db,
    get_search_db_read,
    upsert_retrieval_keywords,
)
from app.llm import LLMClient
from app.prompt_loader import render
from app.rag.keyword_filter import filter_keywords
from app.rag.rarity_filter import filter_clue_by_rarity
from app.workers.summaries import (
    _build_context,
    _classify_file_type,
    _get_indexed_file,
)

logger = logging.getLogger(__name__)

# Hard cap on input characters to the LLM. retrieval_keywords does NOT
# benefit from reading the entire document — the LLM only needs enough
# context to guess the file's topic and likely search terms. Cap matches
# the auto_tags worker's choice to keep latency predictable on local
# models. (Spec D4: "プロンプトで最大 N=20" applies to the OUTPUT count.)
_MAX_CONTEXT_CHARS = 8000

# Max tokens for the keyword-generation LLM call. The expected output
# is a small JSON object with a keyword array; 256 absorbs the JSON
# envelope + a generous keyword list without inviting drift.
_MAX_TOKENS = 256

# Upper bound on kept keywords after filtering. Anything past this is
# truncated — guards against an LLM that emits 100s of low-quality
# expansions and keeps the FTS row size bounded.
_MAX_KEPT_KEYWORDS = 20

# Context-type values handled here. Mirrors the summaries worker's
# classification so we share the same _build_context helper. Images
# (vision_describe context) are intentionally out of scope for Phase 1:
# their description is itself an AI artefact (tier 3), so generating
# keywords from a description would compound generations without a
# first-source anchor. Revisit if a real use-case appears.
_HANDLED_CONTEXT_TYPES = ("video", "audio", "document")

# Language instruction matching the auto_tags / summaries pattern. The
# system prompt template renders an empty string when the configured
# output_language is ``"auto"`` or an unrecognised code — gemma will
# then pick the language from the context tags, which is usually
# right but can mix ja/en on bilingual content. Explicit ja / en
# locks the output language.
_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ja": "- Output keywords in Japanese (proper nouns keep their original script)\n",
    "en": "- Output keywords in English (proper nouns keep their original script)\n",
}


class RetrievalKeywordsWorker:
    """Async worker that generates SIRA-style retrieval keywords.

    Lifecycle and queue semantics intentionally mirror
    ``AutoTagsWorker`` / ``SummariesWorker`` so operators have one
    mental model for the on_index post-task family.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        # One file id at a time in practice — keeps /status symmetric
        # with the other workers.
        self._processing: list[str] = []

    def get_status(self) -> dict[str, object]:
        """Snapshot for /status: ``{waiting, processing}``."""
        return {
            "waiting": self._queue.qsize(),
            "processing": list(self._processing),
        }

    async def enqueue(self, file_id: str) -> None:
        """Add a file to the retrieval-keywords queue.

        Per-drive policy (``intelligence.retrieval_keywords``) is
        checked here so files in opted-out drives never enter the
        queue. The check fails open so a transient internal-API
        failure won't suppress legitimate work.
        """
        from app.policy_client import is_file_feature_enabled

        if not await is_file_feature_enabled(file_id, "retrieval_keywords"):
            return
        await self._queue.put(file_id)

    async def enqueue_unprocessed(self) -> int:
        """Find indexed files without retrieval keywords and enqueue them.

        Returns the number of files queued after per-drive policy
        filtering. Honours the same conventions as
        ``AutoTagsWorker.enqueue_unprocessed`` (active + metadata
        indexed + no existing row, with per-drive policy lookup
        cached across files sharing a drive).

        Restricted to ``file_type IN ('video', 'audio', 'document',
        'text')`` — the same set ``SummariesWorker.enqueue_unprocessed``
        filters to — because ``_process_file`` only handles
        ``_HANDLED_CONTEXT_TYPES`` and returns without writing a
        ``retrieval_keywords`` row for anything else (e.g. images).
        Without this filter, files whose type is permanently
        unsupported never leave the "unprocessed" gap and get
        re-enqueued (and silently re-skipped) on every restart.
        """
        from app.policy_client import is_feature_enabled

        with get_search_db_read() as session:
            rows = session.execute(
                sql_text(
                    "SELECT f.file_id, f.drive FROM indexed_files f "
                    "WHERE f.active = 1 AND f.metadata_indexed = 1 "
                    "AND f.file_type IN ('video', 'audio', 'document', 'text') "
                    "AND f.file_id NOT IN "
                    "  (SELECT file_id FROM retrieval_keywords)"
                )
            ).fetchall()

        drive_allowed: dict[str, bool] = {}
        count = 0
        for file_id, drive in rows:
            allowed = drive_allowed.get(drive)
            if allowed is None:
                allowed = await is_feature_enabled(drive, "retrieval_keywords")
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
            except Exception as exc:  # noqa: BLE001 — worker must not die
                logger.error("Retrieval-keywords worker error: %s", exc)

    async def _process_file(self, file_id: str) -> None:
        """Generate and persist keyword expansions for one file."""
        if settings.features.retrieval_keywords == "false":
            return

        if not self._llm_client.enabled:
            return

        if _has_retrieval_keywords(file_id):
            return

        indexed_file = _get_indexed_file(file_id)
        if indexed_file is None:
            return

        context_type = _classify_file_type(
            indexed_file["file_type"],
            indexed_file.get("mime_type"),
        )
        if context_type not in _HANDLED_CONTEXT_TYPES:
            return

        context = _build_context(indexed_file, context_type)
        if not context:
            # Either no content (silent transcript / empty extractor)
            # or below summaries.min_context_chars — keyword expansion
            # from a near-empty source would be hallucination.
            return

        was_truncated = len(context) > _MAX_CONTEXT_CHARS
        if was_truncated:
            context = context[:_MAX_CONTEXT_CHARS]

        t_start = time.perf_counter()

        raw_keywords = await self._generate_keywords(
            indexed_file=indexed_file,
            context_type=context_type,
            context=context,
            was_truncated=was_truncated,
        )

        cleaned = _post_filter(raw_keywords)
        if not cleaned:
            logger.debug(
                "retrieval_keywords: no keywords survived filtering for %s",
                file_id,
            )
            return

        # SIRA-style category mapping: 'transcript' for audio-backed,
        # 'document' for text-backed. Mirrors the storage convention
        # of suggested_tags / file_summaries.
        stored_context_type = "transcript" if context_type in ("video", "audio") else "document"

        # ``settings.llm.model`` is the canonical model label across
        # provider implementations; OllamaLLMClient / OpenAICompatibleLLMClient
        # do not expose a uniform ``.model`` attribute, so read from
        # config — matches auto_tags / summaries convention.
        model_label = settings.llm.model or "unknown"
        with get_search_db() as session:
            upsert_retrieval_keywords(
                session,
                file_id=file_id,
                keywords=cleaned,
                model=model_label,
                context_type=stored_context_type,
            )

        logger.info(
            "retrieval_keywords: generated for %s (%d chars context, "
            "%d keywords kept, model=%s, elapsed=%.2fs)",
            file_id,
            len(context),
            len(cleaned.split()),
            model_label,
            time.perf_counter() - t_start,
        )

    async def _generate_keywords(
        self,
        *,
        indexed_file: dict,
        context_type: str,
        context: str,
        was_truncated: bool,
    ) -> list[str]:
        """Call the LLM and return its raw keyword candidates.

        Returns an empty list on any failure mode (disabled / throws /
        malformed JSON / missing keywords key). The caller treats an
        empty list as "skip this file" silently.
        """
        language_instruction = _LANGUAGE_INSTRUCTIONS.get(
            settings.llm.output_language, ""
        )
        system_prompt = render(
            "retrieval_keywords/system.jinja2",
            max_keywords=_MAX_KEPT_KEYWORDS,
            language_instruction=language_instruction,
        )
        user_prompt = render(
            "retrieval_keywords/user.jinja2",
            filename=indexed_file["filename"],
            context_type=context_type,
            title=indexed_file.get("title") or "",
            description=indexed_file.get("description") or "",
            was_truncated=was_truncated,
            context=context,
        )

        try:
            raw: Any = await self._llm_client.generate_json(
                system_prompt,
                user_prompt,
                max_tokens_override=_MAX_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "retrieval_keywords: LLM call failed for %s: %s",
                indexed_file["file_id"], exc,
            )
            return []

        if not isinstance(raw, dict):
            return []
        items = raw.get("keywords")
        if not isinstance(items, list):
            return []
        out: list[str] = []
        for entry in items:
            if isinstance(entry, str) and entry.strip():
                out.append(entry.strip())
        return out


def _post_filter(raw_keywords: list[str]) -> str:
    """Run static blocklist + DF rarity filter on the LLM output.

    Joins the raw keyword list with spaces (the canonical FTS form),
    drops blocklist tokens, drops corpus-common tokens, truncates to
    the configured cap, and returns the final whitespace-separated
    string. Empty when nothing survives.

    Both filters reuse the exact same modules that protect the Stage 2
    clue generator — there is no second copy of the blocklist or DF
    logic to maintain.
    """
    if not raw_keywords:
        return ""

    joined = " ".join(raw_keywords)

    # Static blocklist (question / file-type words).
    joined = filter_keywords(joined)
    if not joined:
        return ""

    # Corpus-DF rarity (drops particles / over-common nouns the LLM
    # leaked despite the "no document words" instruction).
    joined = filter_clue_by_rarity(joined)
    if not joined:
        return ""

    # Cap the final keyword count. Truncation is on whitespace-split
    # tokens; we never split a keyword in the middle. dict.fromkeys
    # preserves first-occurrence order while de-duplicating: gemma /
    # other small models routinely repeat the same noun phrase across
    # several keyword positions (observed: "クロノトリガー" x3 in a
    # single response) and the duplicates add nothing to FTS recall.
    tokens = list(dict.fromkeys(joined.split()))
    if len(tokens) > _MAX_KEPT_KEYWORDS:
        tokens = tokens[:_MAX_KEPT_KEYWORDS]
    return " ".join(tokens)


def _has_retrieval_keywords(file_id: str) -> bool:
    """Return True iff a retrieval_keywords row exists for ``file_id``.

    Any existing row (status 'generated' or 'hidden') counts as
    "tried" so we do not regenerate without an explicit user request.
    Regeneration goes through the dedicated endpoint, not the
    background queue.
    """
    with get_search_db_read() as session:
        row = session.execute(
            sql_text(
                "SELECT 1 FROM retrieval_keywords "
                "WHERE file_id = :fid LIMIT 1"
            ),
            {"fid": file_id},
        ).fetchone()
    return row is not None
