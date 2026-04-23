"""Pydantic models for request/response schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# --- Search ---


class SearchResultSegmentMatch(BaseModel):
    type: str
    text: str
    score: float
    page: int | None = None


class SearchResultSegment(BaseModel):
    time_range: list[float] | None = None
    matches: list[SearchResultSegmentMatch]


class SearchResultItem(BaseModel):
    file_id: str
    drive: str
    filename: str
    file_type: str
    score: float
    match_types: list[str]
    segments: list[SearchResultSegment]


class SearchResponseModel(BaseModel):
    results: list[SearchResultItem]
    total: int
    indexed_files: int
    service_version: str


class SourceCountsModel(BaseModel):
    text_vector: int
    clip_vector: int
    keyword: int
    transcript_keyword: int
    text_content_keyword: int = 0


class CompareResponseModel(BaseModel):
    rrf: SearchResponseModel
    cosine: SearchResponseModel
    rrf_no_cutoff: SearchResponseModel
    cosine_no_cutoff: SearchResponseModel
    source_counts: SourceCountsModel


# --- Status ---


class FeaturesStatus(BaseModel):
    indexing: bool
    search: bool
    auto_tags: str
    summaries: str
    rag: bool = False
    transcript_refine: str = "false"


class LLMStatus(BaseModel):
    provider: str
    model: str
    enabled: bool
    output_language: str


class StatusResponse(BaseModel):
    status: str
    indexed: dict[str, int]
    pending: dict[str, int]
    queue: dict[str, Any]
    models: dict[str, str]
    features: FeaturesStatus
    llm: LLMStatus


# --- Webhooks ---


class WebhookScanComplete(BaseModel):
    drive: str
    added: int = 0
    # ``removed`` kept for Litloft builds that predate the missing-files
    # feature. New builds send ``missing`` / ``recovered`` instead.
    removed: int = 0
    missing: int = 0
    recovered: int = 0


class WebhookFilesDeleted(BaseModel):
    file_ids: list[str] = Field(..., max_length=10000)
    type: str = "soft_delete"


class WebhookFilesRestored(BaseModel):
    file_ids: list[str] = Field(..., max_length=10000)


class WebhookFilesPurged(BaseModel):
    file_ids: list[str] = Field(..., max_length=10000)


class WebhookFilesMissing(BaseModel):
    file_ids: list[str] = Field(..., max_length=10000)


class WebhookFilesRecovered(BaseModel):
    file_ids: list[str] = Field(..., max_length=10000)


# --- Queue ---


class QueuePrioritize(BaseModel):
    file_id: str


class MessageResponse(BaseModel):
    status: str
    message: str


# --- Similar files ---


class KeywordScore(BaseModel):
    word: str
    score: float | None = None
    source_tfidf: float | None = None
    target_tfidf: float | None = None
    relevance: float | None = None


class SimilarFileItem(BaseModel):
    file_id: str
    drive: str
    filename: str
    file_type: str
    mime_type: str
    score: float
    match_type: str
    primary_score: float | None = None
    secondary_score: float | None = None
    shared_keywords: list[KeywordScore] = []


class SimilarFilesResponse(BaseModel):
    results: list[SimilarFileItem]
    source_keywords: list[KeywordScore] = []


# --- File inspection ---


class TranscriptChunkResponse(BaseModel):
    """One transcript chunk, possibly AI-refined.

    ``text_refined_at`` is populated when the chunk has been through
    the refine worker; None for unrefined chunks. Serialized as
    camelCase (``refinedAt``) to match the UI's ``TranscriptChunkItem``.
    """

    model_config = ConfigDict(populate_by_name=True)

    index: int
    text: str
    start: float
    end: float
    text_refined_at: datetime | None = Field(default=None, alias="refinedAt", serialization_alias="refinedAt")


class TranscriptResponse(BaseModel):
    file_id: str
    drive: str
    language: str
    chunks: list[TranscriptChunkResponse]


class IndexDetailEmbeddingItem(BaseModel):
    content_preview: str
    start: float | None = None
    end: float | None = None


class IndexDetailType(BaseModel):
    count: int
    items: list[IndexDetailEmbeddingItem]


class IndexDetailsResponse(BaseModel):
    file_id: str
    drive: str
    filename: str
    status: dict[str, bool]
    indexed_at: str
    embeddings: dict[str, IndexDetailType]


class ClipTimestampItem(BaseModel):
    start: float
    content_preview: str


class ClipTimestampsResponse(BaseModel):
    file_id: str
    drive: str
    timestamps: list[ClipTimestampItem]


# --- Suggested tags ---


class SuggestedTagsResponse(BaseModel):
    available: bool
    file_id: str | None = None
    tags: list[str] | None = None
    model: str | None = None
    status: str | None = None
    created_at: str | None = None


class BatchSuggestedTagsRequest(BaseModel):
    file_ids: list[str] = Field(..., max_length=500)


class BatchSuggestedTagsResponse(BaseModel):
    queued: int
    skipped: int


# --- Summaries ---


class SummaryResponse(BaseModel):
    available: bool
    file_id: str | None = None
    short_summary: str | None = None
    long_summary: str | None = None
    model: str | None = None
    context_type: str | None = None
    was_truncated: bool | None = None
    status: str | None = None
    created_at: str | None = None
    # Set to an ISO timestamp when the user has edited the summary;
    # NULL means the displayed text is the raw AI output.
    edited_at: str | None = None
    # True when a pre-edit AI snapshot is stored and revert is possible.
    has_original: bool = False
    # When available=False, explains why:
    #   "not_generated"        — file is eligible but user hasn't generated yet
    #   "insufficient_content" — transcript/text below min_context_chars
    #   "unsupported_type"     — image/archive/etc., can't be summarized
    #   "file_not_found"       — no indexed_files row
    #   None                   — feature disabled or summary hidden by user
    reason: str | None = None


class SummaryEditRequest(BaseModel):
    short_summary: str = Field(..., min_length=1, max_length=200)
    long_summary: str = Field(..., min_length=1, max_length=4000)


class BatchSummariesRequest(BaseModel):
    file_ids: list[str] = Field(..., max_length=500)


class BatchSummariesResponse(BaseModel):
    queued: int
    skipped: int


# --- Detailed (long-form Markdown) summary ---


class DetailedSummaryResponse(BaseModel):
    """Response for GET /files/{id}/summary/detailed.

    ``available`` is True only when ``status == "generated"`` — callers
    look at ``status`` for the finer-grained state of in-progress work.
    """

    available: bool
    file_id: str | None = None
    detailed_summary: str | None = None
    # "generating" | "generated" | "failed" — None when nothing has been
    # started yet, in which case ``reason`` explains eligibility.
    status: str | None = None
    model: str | None = None
    generated_at: str | None = None
    context_chars: int | None = None
    was_truncated: bool | None = None
    error: str | None = None
    # Same semantics as SummaryResponse.reason — populated when
    # ``available`` is False so the frontend can render the right state.
    reason: str | None = None
    # User-edit metadata. ``edited_at`` is an ISO timestamp when the
    # user has edited any section; ``has_original`` is True when a
    # pre-edit snapshot is stored and revert is possible.
    edited_at: str | None = None
    has_original: bool = False


class DetailedSummaryStartResponse(BaseModel):
    """Response for POST /files/{id}/summary/detailed."""

    status: str
    message: str


class DetailedSummaryCitationItem(BaseModel):
    """One row of the citations response.

    ``chunk_ids`` is already decoded from its JSON storage. ``top_score``
    is the cosine similarity of the best-matching chunk regardless of
    whether it cleared the threshold, so the UI can surface a ⚠ badge
    with the weak score when ``has_citation = False``.
    """

    section_path: str
    segment_type: str  # "bullet" | "paragraph"
    segment_text: str
    chunk_ids: list[str] = []
    top_score: float
    has_citation: bool


class DetailedSummaryCitationsResponse(BaseModel):
    """Response for GET /files/{id}/summary/detailed/citations."""

    file_id: str
    citations: list[DetailedSummaryCitationItem]


class ChunkExcerptResponse(BaseModel):
    """Response for GET /files/{id}/chunks/{chunk_id}/excerpt.

    ``chunk_id`` echoes the prefixed identifier stored in the citations
    table (``transcript:{idx}`` or ``document:{idx}``). The excerpt is
    split into ``prefix`` / ``target`` / ``suffix`` so the UI can
    visually highlight the cited chunk against its surrounding
    context. Concatenating the three strings reproduces the flat
    rendering used before this shape was introduced.

    * ``target`` — the cited chunk's own text, verbatim.
    * ``prefix`` / ``suffix`` — up to ±100 characters of neighbour
      context, already including their trailing / leading space
      separator and a ``"… "`` marker when truncated. Empty string
      when the chunk sits at the edge of the source.

    Transcript chunks populate ``start_time`` / ``end_time`` (float
    seconds) and leave ``page`` null. Document chunks do the inverse —
    ``page`` is the page number (when the source extractor produced
    one) while the timestamps are null.
    """

    chunk_id: str
    file_id: str
    prefix: str
    target: str
    suffix: str
    start_time: float | None = None
    end_time: float | None = None
    page: int | None = None


class DetailedSummaryEditRequest(BaseModel):
    """Request body for PUT /files/{id}/summary/detailed/section.

    ``section_heading`` is the H2 anchor text (``## 見出し`` without the
    leading hashes, e.g. ``"全体像"``). ``subsection_heading`` narrows
    the edit to a single ``### 見出し`` inside that H2 body; omit it
    to splice the whole H2 section.

    ``new_content`` is the full Markdown fragment that replaces the
    range — *including the heading line itself* so the user can rename
    the heading or restructure it with nested ``###`` subsections. The
    fragment is spliced verbatim; structural changes propagate on the
    next parse/render cycle.
    """

    section_heading: str = Field(..., min_length=1, max_length=200)
    subsection_heading: str | None = Field(
        default=None, max_length=200
    )
    new_content: str = Field(..., min_length=1, max_length=20000)


class DetailedSummaryRegenerateRequest(BaseModel):
    """Optional request body for POST /files/{id}/summary/detailed/regenerate.

    ``force=True`` suppresses the 409-Conflict when the current
    detailed summary has been user-edited. The frontend uses the
    unforced path first and re-submits after a confirmation dialog.
    """

    force: bool = False


# --- RAG (question answering) ---


class AskRequest(BaseModel):
    """Request body for POST /ask.

    Enforces a 1-1000 character range on the raw query as a basic
    DoS guard. The router applies an additional >=3 character
    post-strip check so "a  " does not reach the LLM.
    """

    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    file_type: str | None = None
    drive: str | None = None


class CitationModel(BaseModel):
    """A single citation backing one part of the RAG answer."""

    file_id: str
    drive: str
    filename: str
    file_type: str
    quote: str
    relevance: float
    segment_location: str | None = None  # e.g. "0:45" or "page 3"


class SourceModel(BaseModel):
    """A retrieved file that was passed to the LLM as context.

    Populated even when the LLM fails to produce a valid answer,
    so the UI can still show the user which files were considered.
    """

    file_id: str
    drive: str
    filename: str
    file_type: str
    score: float
    match_types: list[str]


class AnswerResponseModel(BaseModel):
    """Response body for POST /ask."""

    query: str
    answer: str | None
    citations: list[CitationModel]
    sources: list[SourceModel]
    retrieved_count: int
    took_ms: int
