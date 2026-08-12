"""Configuration for the semantic search service.

Reads settings from environment variables and search-config.yml.
All config values are immutable after initialization.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Deprecation date for the legacy ``indexing.whisper.*`` keys. After
# this date the shim is removed and the keys raise a ConfigError.
# Spec: 2026-05-07-cloud-transcription-providers.md "Deprecation
# timeline".
_LEGACY_WHISPER_REMOVAL_DATE = "2026-07-07"

# Once-per-process flag: log the deprecation warning only on the first
# parse call so a worker that re-parses (test harness, hot reload)
# does not spam the log.
_whisper_deprecation_logged: bool = False


@dataclass(frozen=True)
class ModelConfig:
    whisper: str = "openai/whisper-large-v3-turbo"
    text_embedding: str = "ibm-granite/granite-embedding-97m-multilingual-r2"
    clip: str = "llm-jp/waon-siglip2-base-patch16-256"
    blip: str = "Salesforce/blip-image-captioning-base"
    blip_max_tokens: int = 50  # max tokens for BLIP caption generation
    blip_num_beams: int = 1  # beam search width (1 = greedy, higher = better quality but slower)
    text_query_prefix: str = ""
    text_passage_prefix: str = ""


@dataclass(frozen=True)
class SearchConfig:
    # Candidate retrieval
    rrf_candidates: int = 50
    # Result limits
    default_limit: int = 20
    max_limit: int = 100
    # Pre-filter thresholds (exclude candidates below these before ranking)
    min_score_text: float = 0.85
    min_score_clip: float = 0.05
    # ``clip_thumbnail`` (representative 1-frame route from spec
    # 2026-05-02-thumbnail-clip-default-shallow-search.md) has a
    # different false-positive profile than scene CLIP: 1 vector per
    # file, ffmpeg ``thumbnail=300`` already filters black/blurred
    # frames. Worth a looser threshold and a stronger weight in its
    # own knob so tuning ``clip`` does not move ``clip_thumbnail``.
    min_score_clip_thumbnail: float = 0.05
    # Score gap analysis: if (top_score - mean_score) < this threshold,
    # the result set is "flat" (no standout match) and discarded entirely.
    score_gap_threshold: float = 0.02
    # Dynamic cutoff: discard results below (top_score - margin)
    score_cutoff_margin: float = 0.05
    # Combined score cutoff: discard results below (top_score * ratio)
    score_cutoff_ratio: float = 0.5
    # Cosine scoring weights
    # alpha: balance between vector similarity and keyword matching
    alpha: float = 0.7
    # Per-source type weights (applied to vector scores before combining)
    type_weight_metadata: float = 1.3
    type_weight_transcript: float = 1.0
    type_weight_text_content: float = 0.9
    type_weight_clip: float = 1.0
    # ``clip_thumbnail`` is the "video about X" main route, so it
    # ranks at parity with metadata-class signals rather than the
    # half-weight applied to scene CLIP.
    type_weight_clip_thumbnail: float = 1.0
    # Legacy (kept for config file compatibility, used by compare endpoint)
    rrf_k: int = 60
    rrf_weight_clip: float = 0.5
    rrf_weight_clip_thumbnail: float = 1.0


@dataclass(frozen=True)
class FrameExtractionConfig:
    scene_threshold: float = 0.3
    min_interval: int = 30
    max_frames: int = 500


@dataclass(frozen=True)
class WhisperIndexConfig:
    min_segment_duration: int = 10
    max_segment_duration: int = 20
    beam_size: int = 1
    batch_size: int = 0
    condition_on_previous_text: bool = True
    # Optional override for the Whisper ``initial_prompt`` (≤224 tokens).
    # When blank, ``app.workers.whisper_prompts`` supplies a per-language
    # default keyed off Whisper's detected language; set this only when
    # you need a domain-specific style hint (e.g. specialised
    # vocabulary). The override fully replaces the default — we do not
    # concatenate. Do not include filenames or curated vocabulary here;
    # put those in a dedicated glossary layer if needed.
    initial_prompt: str = ""
    # Segments with compression ratio above this threshold are discarded
    # (repetitive/looping output). Lower = stricter. 0 = disabled.
    compression_ratio_threshold: float = 2.0
    # Segments with no-speech probability above this are discarded.
    # Lower = more aggressive silence removal. 0 = disabled.
    no_speech_threshold: float = 0.45
    # Segments with average log probability below this are discarded.
    # 0 = disabled.
    log_prob_threshold: float = -1.0


@dataclass(frozen=True)
class WhisperLocalConfig:
    """Provider config for the local faster-whisper backend.

    Mirrors :class:`WhisperIndexConfig` for the new ``transcription``
    config tree introduced by spec
    ``2026-05-07-cloud-transcription-providers.md``. Fields stay
    1:1 with the legacy ``indexing.whisper.*`` keys so the
    backward-compat shim can copy values across without translation.
    """

    model: str = "openai/whisper-large-v3-turbo"
    initial_prompt: str = ""
    beam_size: int = 1
    batch_size: int = 0
    condition_on_previous_text: bool = True
    compression_ratio_threshold: float = 2.0
    no_speech_threshold: float = 0.45
    log_prob_threshold: float = -1.0
    # Same chunk-boundary knobs as WhisperIndexConfig — kept on the
    # provider sub-config so each provider can theoretically tune
    # independently in the future.
    min_segment_duration: int = 10
    max_segment_duration: int = 20


@dataclass(frozen=True)
class OpenAICompatibleProviderConfig:
    """OpenAI Whisper API + base_url override (Groq / Fireworks)."""

    base_url: str = "https://api.openai.com/v1"
    model: str = "whisper-1"
    timeout_s: int = 600
    # Per-provider override for the rate-limit circuit breaker
    # threshold. ``None`` keeps the spec default (20 failures /
    # 60 s window) — see
    # ``2026-05-07-cloud-transcription-providers.md`` R2-4.
    circuit_breaker_threshold: int | None = None


@dataclass(frozen=True)
class DeepgramProviderConfig:
    model: str = "nova-3"
    diarize: bool = True
    smart_format: bool = True
    detect_language: bool = True
    timeout_s: int = 600
    circuit_breaker_threshold: int | None = None


@dataclass(frozen=True)
class ElevenLabsScribeProviderConfig:
    model_id: str = "scribe_v2"
    diarize: bool = True
    no_verbatim: bool = False
    timeout_s: int = 600
    circuit_breaker_threshold: int | None = None


@dataclass(frozen=True)
class AssemblyAIProviderConfig:
    """AssemblyAI v2 API (httpx-based, no SDK).

    Defaults reflect spec ``2026-05-08-transcription-providers-phase-2a.md``:
    ``best`` (Universal-2) for highest-quality multi-language output;
    diarisation on by default to match Deepgram / ElevenLabs parity.
    Polling lives on this config (not the provider) so operators can
    tune cost vs. latency without code changes.
    """

    model: str = "best"  # "best" (Universal-2) or "nano"
    language_detection: bool = True
    speaker_labels: bool = True
    timeout_s: int = 1800
    poll_interval_s: int = 3
    circuit_breaker_threshold: int | None = None


@dataclass(frozen=True)
class GeminiProviderConfig:
    """Gemini File API + generate_content.

    ``upload_wait_sec`` covers the gap between ``client.files.upload``
    completing and the file becoming ``ACTIVE`` for inference; it is
    distinct from ``timeout_s`` which bounds the upload POST itself.
    Synthetic word timestamps are provider-internal (see spec
    §"Synthetic word 生成") so no knob is exposed here.
    """

    model: str = "gemini-2.5-flash"
    output_language: str = "ja"
    upload_wait_sec: int = 300
    timeout_s: int = 1800
    circuit_breaker_threshold: int | None = None


@dataclass(frozen=True)
class TranscriptionConfig:
    """Top-level transcription settings.

    Replaces the legacy ``indexing.whisper.*`` section. The ``provider``
    field selects which sub-config the worker reads at runtime. Phase
    1B will ship four concrete provider classes
    (``whisper_local`` / ``openai_compatible`` / ``deepgram`` /
    ``elevenlabs_scribe``); Phase 1A only defines the shape.
    """

    provider: str = "whisper_local"
    language_hint: str = ""
    hotwords: tuple[str, ...] = ()
    whisper_local: WhisperLocalConfig = field(default_factory=WhisperLocalConfig)
    openai_compatible: OpenAICompatibleProviderConfig = field(
        default_factory=OpenAICompatibleProviderConfig
    )
    deepgram: DeepgramProviderConfig = field(
        default_factory=DeepgramProviderConfig
    )
    elevenlabs_scribe: ElevenLabsScribeProviderConfig = field(
        default_factory=ElevenLabsScribeProviderConfig
    )
    assemblyai: AssemblyAIProviderConfig = field(
        default_factory=AssemblyAIProviderConfig
    )
    gemini: GeminiProviderConfig = field(
        default_factory=GeminiProviderConfig
    )


@dataclass(frozen=True)
class TextChunkingConfig:
    max_chunk_size: int = 400
    overlap: int = 80


@dataclass(frozen=True)
class IndexingConfig:
    reconciliation_interval: int = 3600
    frame_extraction: FrameExtractionConfig = field(
        default_factory=FrameExtractionConfig
    )
    whisper: WhisperIndexConfig = field(default_factory=WhisperIndexConfig)
    text_chunking: TextChunkingConfig = field(default_factory=TextChunkingConfig)


@dataclass(frozen=True)
class WorkerConfig:
    whisper_parallel: int = 1
    clip_parallel: int = 2
    metadata_batch_size: int = 32
    clip_frame_batch_size: int = 50


@dataclass(frozen=True)
class MemoryConfig:
    whisper_idle_unload: int = 300
    blip_idle_unload: int = 300  # seconds; 0 = never unload
    clip_concepts_idle_unload: int = 600  # seconds; 0 = never unload


@dataclass(frozen=True)
class FeaturesConfig:
    indexing: bool = True
    search: bool = True
    auto_tags: str = "manual"  # "false" | "manual" | "on_index"
    summaries: str = "manual"  # "false" | "manual" | "on_index"
    # Detailed (long-form, Markdown) summary modes:
    # - "false":    no generation
    # - "manual":   generated on user request (file-detail button)
    # - "on_index": generated automatically after short/long summary
    #               finishes for each newly indexed file. Cost scales
    #               linearly with new files per day — recommended only
    #               with cheap cloud LLMs (nano/mini tier) or patient
    #               local LLMs.
    detailed_summaries: str = "false"  # "false" | "manual" | "on_index"
    # RAG (question answering) is a simple on/off switch: there is no
    # index-time equivalent to "on_index" because RAG only runs in
    # response to user queries. Enabled by default; runtime still requires
    # an LLM provider, and file content is sent to that provider.
    rag: bool = True
    # Transcript AI refine ("false" | "manual" | "on_index"). Default off
    # since file contents are sent to the LLM API during refine.
    transcript_refine: str = "false"
    # Vision-LLM image description ("false" | "manual" | "on_index").
    # Default is manual; running it sends image bytes to the LLM API. Requires
    # ``llm.vision_model`` to be set or the feature is unavailable even
    # when this flag is "manual"/"on_index" (graceful degradation).
    vision_describe: str = "manual"
    # SIRA-style LLM-generated retrieval keywords per file
    # ("false" | "manual" | "on_index"). The LLM predicts synonyms,
    # abbreviations and alternate names users might search by; the
    # output goes into a dedicated FTS surface (fts_retrieval_keywords)
    # and improves file-search robustness against vocabulary gaps.
    # Default off — file content is sent to the LLM API. Spec:
    # docs/superpowers/specs/2026-05-14-sira-retrieval-keywords.md.
    retrieval_keywords: str = "false"
    # LLM-derived media chapter candidates ("false" | "manual" |
    # "on_index"). Default off because timestamped transcript content is
    # sent to the configured LLM provider.
    chapter_suggestions: str = "false"


@dataclass(frozen=True)
class HierarchicalRagConfig:
    """Hierarchical RAG (Stage 1 coarse-retrieval) configuration.

    Phase 1 (shadow mode) only logs the shortlist; Phase 2 actually
    scopes the chunk-level retrieval to the shortlist. See
    ``docs/superpowers/specs/2026-04-26-intelligence-ask-hierarchical-retrieval.md``.

    Defaults enable the shortlist path, while bypass thresholds err on
    the side of running legacy full-file retrieval when in doubt.
    """

    # Master switch. False keeps the legacy single-stage retrieval.
    enabled: bool = True
    # Top-K shortlist size: how many files survive Stage 1.
    coarse_top_k: int = 20
    # Below this top cosine similarity the Stage 1 result is considered
    # untrustworthy and we bypass scoping (run the legacy path).
    coarse_score_threshold: float = 0.3
    # Drives smaller than this skip Stage 1 entirely — file-level
    # narrowing has no statistical meaning on a tiny corpus.
    min_drive_files_for_shortlist: int = 50
    # When the scoped retrieval returns very few candidates, also run
    # an unscoped pass and merge so pinpoint factual queries still
    # have a fallback path. See spec §7.4.
    fallback_full_search: bool = True
    # Reserved for Phase 3 (multi-query clue generation). Unused in
    # Phases 1 / 2.
    clue_count: int = 3


@dataclass(frozen=True)
class PersonalHistoryConfig:
    """Personal-history scoping for Ask (Stage A/B/D).

    Spec: ``2026-04-26-intelligence-ask-personal-history-query.md``.
    When enabled, Ask decomposes the natural query into a structured
    form (time range + viewer scope), then narrows retrieval to the
    file_ids the caller has actually opened in the requested window.

    Defaults enable personal scoping, while the lookback ceiling defends
    against pathologically wide ``last_year`` resolutions on long-running
    deployments.
    """

    # Master switch. False keeps Ask viewer-agnostic (legacy behaviour).
    enabled: bool = True
    # Hard ceiling on Stage A's resolved time range. "ずっと前に観たやつ"
    # otherwise expands to a multi-year scan that defeats the point of
    # narrowing. 365 days matches a typical "this year" intuition.
    max_lookback_days: int = 365
    # When the history filter resolves to zero file_ids:
    #   "graceful" — drop the filter and run a normal Ask
    #   "strict"   — return "該当なし" without further retrieval
    fallback_when_empty: str = "graceful"

    def __post_init__(self) -> None:
        # Validate enums at construction so a typo'd YAML value
        # (``"graeful"``) fails loudly at startup rather than silently
        # taking the unknown branch later. ``object.__setattr__`` is
        # not needed because we're only asserting, not mutating.
        if self.fallback_when_empty not in {"graceful", "strict"}:
            raise ValueError(
                "personal_history.fallback_when_empty must be "
                "'graceful' or 'strict', got "
                f"{self.fallback_when_empty!r}"
            )
        if self.max_lookback_days < 1:
            raise ValueError(
                "personal_history.max_lookback_days must be >= 1, "
                f"got {self.max_lookback_days}"
            )


@dataclass(frozen=True)
class CategoryExpansionConfig:
    """Semantic category expansion for Ask (Stage C).

    Spec: ``2026-04-26-intelligence-ask-personal-history-query.md``.
    "SF っぽい" is rewritten into a small bag of bilingual surface
    forms (science fiction / 宇宙船 / ロボット / ...) so multi-query
    retrieval has something concrete to embed against.
    """

    # Master switch. False keeps the Stage A semantic_query verbatim.
    enabled: bool = False
    # Hard cap on the LLM-emitted expansion list. 8 is the sweet spot
    # observed in the original draft: enough breadth for "SF" /
    # "ホラー" style genre prompts, narrow enough that noise terms do
    # not dominate the multi-query fan-out budget.
    max_terms: int = 8


@dataclass(frozen=True)
class RagConfig:
    """RAG (question answering) configuration.

    Controls how many files are retrieved, how much context per file
    is extracted, the hard total-context cap, and the LLM max_tokens
    override for answer generation. Defaults are tuned for current Ask.
    """

    # Number of files retrieved and fed to the LLM as context.
    top_k: int = 5
    # Max characters of context extracted from each file
    # (from segment matches, not full transcripts).
    max_context_chars_per_file: int = 3500
    # Hard cap on total context size across all retrieved files.
    # When exceeded, lower-scoring files are dropped.
    max_total_context_chars: int = 17500
    # LLM max_tokens override for answer + citations generation.
    # The answer JSON includes both the prose answer and a citations
    # array, so 2048 gives room for detailed answers without truncation.
    max_tokens: int = 2048
    # How many characters around a timestamped transcript match
    # to include as "context window" (only when segments have times).
    transcript_window_seconds: float = 60.0
    # For document files: number of top vector-similar chunks to pull
    # into the LLM context, in addition to keyword-match chunks. This
    # rescues queries whose natural-language vocabulary doesn't overlap
    # with the target file's literal tokens — the file-level vector
    # channel already surfaced the right file, but keyword-driven chunk
    # selection can still miss the actual answer passage. 0 disables
    # the vector pass and restores keyword-only behaviour.
    document_vector_top_n: int = 6
    # For transcript files: number of top vector-similar / keyword-OR
    # chunks to pull in addition to the time-window segments. Smaller
    # than document_vector_top_n because Whisper chunks are short
    # (2-3 sentences) so more fit under the per-file budget.
    # 0 disables the additional passes.
    transcript_vector_top_n: int = 4
    # Hierarchical retrieval (Stage 1 coarse shortlist). Enabled by default;
    # set ``rag.hierarchical.enabled: false`` to force legacy retrieval.
    hierarchical: HierarchicalRagConfig = field(
        default_factory=HierarchicalRagConfig
    )
    # Personal-history scoping (Stage A/B/D). Enabled by default; set
    # ``rag.personal_history.enabled: false`` to force viewer-agnostic Ask.
    personal_history: PersonalHistoryConfig = field(
        default_factory=PersonalHistoryConfig
    )
    # Semantic category expansion (Stage C). Default disabled — opt-in
    # via ``rag.category_expansion.enabled: true`` in ``search-config.yml``.
    category_expansion: CategoryExpansionConfig = field(
        default_factory=CategoryExpansionConfig
    )


@dataclass(frozen=True)
class SummariesConfig:
    # Minimum context length (stripped). Files with less usable content
    # than this are skipped entirely — a tiny transcript like the word
    # "you" (Whisper false positive on silent content) would otherwise
    # cause the LLM to hallucinate a summary from the filename alone.
    min_context_chars: int = 50
    # Threshold: if total context <= max_context_chars, send full text.
    # Otherwise, sample windows from beginning/middle/end.
    max_context_chars: int = 8000
    # Characters per window when sampling (window_count windows total).
    window_chars: int = 2500
    # Number of windows to sample (first/middle/last; use odd numbers).
    window_count: int = 3
    # Detailed-summary threshold: full text is sent up to this length.
    # Higher than the short/long path because the detailed prompt's
    # "重要ポイントまとめ" table wants broad coverage, and detailed
    # generation is manual/on-demand so extra prefill cost is acceptable.
    detailed_max_context_chars: int = 24000
    # Detailed-summary fallback window count. Larger than the short path
    # to preserve coverage when the detailed threshold is exceeded.
    detailed_window_count: int = 5
    # Citation linking: cosine similarity threshold for "has_citation".
    # top-1 score >= this counts as a confirmed citation; below is
    # surfaced in the UI as a "no strong source" warning.
    citation_threshold: float = 0.55
    # Number of citation candidates retrieved per segment. The UI can
    # surface all of them; ``has_citation`` is driven by the top one.
    citation_top_k: int = 3
    # Hybrid retrieval: when True, dense-KNN top-N is reranked by a
    # BM25 (FTS5) pass over the same chunk pool. Keyword-aligned
    # candidates bubble up when the segment shares salient tokens with
    # the source (numbers, proper nouns, katakana). Dense cosine is
    # still used as ``top_score`` so ``citation_threshold`` keeps its
    # meaning. False restores the legacy dense-only behaviour.
    citation_hybrid_enabled: bool = True
    # Candidate pool size for hybrid retrieval — dense pulls this many
    # candidates, BM25 reorders within. Must be >= citation_top_k.
    citation_top_k_internal: int = 10
    # RRF fusion constant. Standard IR value is 60; smaller values
    # amplify the weight of top ranks (more "winner-takes-all").
    citation_rrf_k: int = 60
    # Section anchoring: hierarchical top-down narrowing. For every
    # prefix of a segment's ancestor_headings chain, pool the
    # embeddings of all segments under that prefix and find the
    # tightest transcript chunk range where that pool's content is
    # discussed. Deeper prefixes narrow within shallower prefixes'
    # ranges, and narrowing stops (inheriting the parent range) as
    # soon as the pool's top-1 match falls below ``citation_section_narrow_threshold``.
    # Naturally handles both focused bullets (narrowed deep) and
    # cross-cutting summaries like まとめ / 結論 (stop early at a wide
    # range or at full file). Disable to fall back to full-file
    # retrieval for every segment.
    citation_section_anchor_enabled: bool = True
    # Top-M chunks whose indices define a range at each narrowing
    # level. Higher values tolerate spread topics (wider range);
    # lower values insist on tighter focus.
    citation_section_range_top_m: int = 12
    # Score floor for continuing to narrow. When the pool's top-1
    # cosine similarity within the parent range is below this, we
    # stop and inherit the parent range. 0.5 is a moderate floor:
    # strong enough to avoid chasing noise, loose enough to let
    # loosely-related summaries pass through.
    citation_section_narrow_threshold: float = 0.5
    # Dense-cluster detection: after picking the top-M scoring chunks
    # for a section pool, split them into contiguous clusters by
    # chunk_index gap. A gap of more than this many positions starts
    # a new cluster. The largest cluster (by total score) defines the
    # section's range, so a few high-scoring outliers on the other
    # side of the file can't drag the range open. Increase on long
    # files where legitimate topics span wider spans.
    citation_section_cluster_gap: int = 5
    # Union the runner-up cluster with the primary when its total
    # score is this fraction of the primary's or higher. Catches
    # sections that legitimately reference two separate parts of the
    # video (e.g. a 結論 that revisits both an intro and a conclusion).
    # 0.8 is conservative: only near-tied runners-up get unioned.
    citation_section_cluster_union_ratio: float = 0.8
    # Discriminative scoring: subtract the highest sibling-section
    # cosine from the target section's cosine before top-M / cluster
    # detection. This handles the "whole video shares one topic"
    # case (e.g. a recipe video where every chunk mentions cooking
    # and every section's pool matches every chunk at ~0.9) by
    # ranking chunks on "how specifically does this match THIS
    # section vs any sibling section" rather than on absolute cosine.
    # Chunks that match a sibling section more strongly get filtered
    # out of the range.
    citation_section_discriminative_enabled: bool = True
    # Margin required above sibling cosines when discriminative is
    # enabled. A chunk is kept for this section only when
    # ``this_cos - max_sibling_cos >= disc_margin``. 0.01 is a small
    # margin that catches clear outliers while tolerating minor noise;
    # raise to be more strict about assigning chunks to sections.
    citation_section_disc_margin: float = 0.01
    # Monotonic DP alignment: for sibling section groups (2+ prefixes
    # with a shared parent), assign each chunk to exactly one section
    # via a Viterbi pass that forbids going backward through the
    # summary's section order. This formalises the user intuition
    # "summary sections appear in chronological order of the source"
    # and fixes cases where a section's pool matches broadly across
    # the video (e.g. a DQ-remake video's "近年のリメイクの流れ" H3
    # whose content overlaps the whole video). When disabled, or for
    # prefixes with fewer than two siblings that have usable pools,
    # the code falls back to the pool + cluster-detection path.
    citation_section_alignment_enabled: bool = True
    # DP's strict 1:1 chunk→section assignment creates hard borders;
    # a chunk on the boundary between two adjacent sections ends up
    # in exactly one section's range, even if it legitimately serves
    # both (e.g. the speaker's transition sentence). Expanding each
    # section's DP-assigned range by this many chunks on each side
    # creates a small overlap at the borders so the "transition
    # chunk" is reachable by either section's retrieval. 2 is a
    # gentle default; 0 disables smoothing (use strict DP boundaries).
    citation_section_boundary_margin: int = 2
    # Margin gate: when ``top1_score - top2_score`` is smaller than
    # this, flip ``has_citation`` to False because the top pick is
    # low-confidence (many chunks look comparably close). 0 disables
    # the gate. 0.05 is a mild default that demotes the worst
    # "misleadingly-confident" segments without punishing genuine
    # strong matches. Only active when top_score < margin_bypass_score.
    citation_margin_gate: float = 0.05
    # Bypass threshold: segments whose top_score >= this value skip
    # the margin gate entirely. A close runner-up is fine when the
    # leader is already strongly matched.
    citation_margin_bypass_score: float = 0.75
    # Compound-bullet multi-anchor retrieval. When a summary bullet
    # lists several sub-anchors — e.g. "洗って + 芯を切り落とし + 葉と
    # 芯を分けて + 千切り" (four kitchen operations) or "にんじん3本、
    # 手元分量でよい" (two facts) — a single top-1 retrieval is
    # under-determined: the dense embedding of the joined text blurs
    # across anchors and the retriever ends up snapping to a
    # neighbouring "theme" chunk that matches the compound text's
    # register (declarative summary) rather than the imperative
    # instructional chunks where each sub-anchor lives. Splitting the
    # bullet on CJK punctuation (、。・，) and running retrieval per
    # sub-segment recovers each anchor independently; results are then
    # unioned by max-score. 0-1 usable sub-segments falls back to the
    # single-embedding path. Table rows (cells != None) and paragraphs
    # are skipped — cell pooling and claim-vs-example are separate
    # concerns. False restores the legacy single-embedding behaviour.
    citation_multi_anchor_enabled: bool = True
    # Minimum character length for a sub-segment to be kept. Anchors
    # shorter than this (particles, single-word fragments like "水分"
    # after "絞って") tend to over-match; dropping them leaves the
    # parent bullet's full text as the sole anchor, which is safer
    # than running retrieval with an impoverished query.
    citation_multi_anchor_min_len: int = 4
    # Paragraph synthesis gate. When a paragraph-type segment's
    # citation chunks are scattered across the file — normalised
    # index-spread > this threshold — the LLM most likely synthesised
    # the claim from multiple parts of the document and no single
    # chunk is the source. Flip has_citation to False so the UI
    # doesn't imply single-source provenance. 0 disables the gate.
    #
    # Only applies to segment_type == "paragraph". Bullets (fact-level
    # items and table rows) are unaffected — those are almost always
    # single-source. The signal is purely the chunk-index distribution,
    # so it is language- and LLM-independent.
    citation_paragraph_spread_gate: float = 0.3
    # Files smaller than this many chunks skip the paragraph gate:
    # short files push most chunks together and normalised spread
    # loses its meaning (a 3-chunk song where the chorus recurs in
    # chunks 0 and 2 looks "scattered" but is really single-source).
    citation_paragraph_spread_min_chunks: int = 50


@dataclass(frozen=True)
class AgenticModelEntry:
    """Per-model capability + budget hint for the agentic Ask loop."""

    name: str
    context_window: int = 32768


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "disabled"  # "openai_compatible" | "ollama" | "disabled"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    max_tokens: int = 2048
    temperature: float = 0.3
    output_language: str = "auto"  # "auto" | "ja" | "en" | etc. — applies to auto_tags and summaries
    # Agentic Ask (Phase 1.C). ``agentic_mode="off"`` is the kill
    # switch; ``"auto"`` activates the loop iff the active model name
    # appears in ``agentic_models``. Unknown models always fall back
    # to legacy single-turn RAG.
    #
    # Default is "off": Phase 1.D testing on Apple Silicon (M-series,
    # 64GB unified memory) found local LLMs hit a prefill wall on the
    # 3rd loop iteration (~8K accumulated tool-result tokens):
    # Qwen3:8b, Qwen3.5:9b, and Qwen3.5:35b-a3b all failed to complete
    # within a 5-minute deadline. GPT-4o cloud completes the same
    # query in ~15s. Operators with cloud LLM keys (or a future
    # locally-runnable tool-capable model that doesn't choke on
    # prefill) can opt in via ``agentic_mode: "auto"`` plus an entry
    # in ``agentic_models``.
    agentic_mode: str = "off"  # "auto" | "off"
    agentic_min_capability: str = "tool_use_native"
    agentic_models: tuple[AgenticModelEntry, ...] = field(default_factory=tuple)
    # Retry behavior for transient failures (timeouts, 429, 5xx)
    retry_attempts: int = 3  # total attempts = 1 + retries
    retry_base_delay: float = 1.0  # seconds, doubled on each retry
    retry_max_delay: float = 30.0  # cap on individual backoff delay
    # Minimum interval between requests (0 = no rate limiting)
    min_request_interval_ms: int = 0
    # Hard per-request timeout. The OpenAI SDK defaults to ~600 seconds,
    # which is far longer than any reasonable home-LAN deployment wants
    # — a stuck upstream would otherwise keep the coroutine alive for
    # 10 minutes after the host proxy's 15s timeout has already given
    # up, and a user mashing "Regenerate" could accumulate dozens of
    # pending LLM calls. 90s is generous enough for local ollama on
    # mid-range hardware but forces clean failure on remote-API stalls.
    request_timeout_seconds: float = 90.0
    request_connect_timeout_seconds: float = 10.0
    # Vision-LLM settings (separate from text-mode so operators can mix
    # providers — e.g. gemma2 for text + llava for images). Empty
    # ``vision_model`` disables the vision feature regardless of
    # ``features.vision_describe``.
    vision_model: str = ""
    vision_max_tokens: int = 1024
    vision_temperature: float = 0.1


@dataclass(frozen=True)
class Settings:
    intelligence_data_dir: Path
    litloft_db_path: Path
    model_cache_dir: Path
    search_db_path: Path
    allowed_base_dirs: tuple[str, ...] = ("/drives/",)
    drive_mounts: dict[str, str] = field(default_factory=dict)
    models: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    workers: WorkerConfig = field(default_factory=WorkerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    summaries: SummariesConfig = field(default_factory=SummariesConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    service_version: str = "0.1.0"
    port: int = 8100


def _parse_nested(data: dict[str, Any], key: str, cls: type) -> Any:
    """Parse a nested config section into a frozen dataclass."""
    section = data.get(key, {})
    if not isinstance(section, dict):
        return cls()
    return cls(**{k: v for k, v in section.items() if k in cls.__dataclass_fields__})


_RAG_NESTED_KEYS = ("hierarchical", "personal_history", "category_expansion")


def _parse_rag(
    data: dict[str, Any],
    overrides: "RagOverrides | None" = None,  # type: ignore[name-defined]
) -> RagConfig:
    """Parse the rag config section with the nested sub-sections.

    Mirrors ``_parse_indexing`` so unknown top-level keys are dropped
    rather than blowing up dataclass construction, and the nested
    sub-sections (``hierarchical``, ``personal_history``,
    ``category_expansion``) hydrate via ``_parse_nested``.

    GUI overrides for ``personal_history.enabled`` and
    ``category_expansion.enabled`` are applied on top of the parsed
    section before sub-section dataclasses are built.
    """
    section_raw = data.get("rag", {})
    section: dict[str, Any] = (
        section_raw if isinstance(section_raw, dict) else {}
    )

    if overrides is not None:
        from app.rag_overrides import merge_into_rag_dict

        section = merge_into_rag_dict(section, overrides)

    flat_kwargs = {
        k: v
        for k, v in section.items()
        if k in RagConfig.__dataclass_fields__ and k not in _RAG_NESTED_KEYS
    }
    return RagConfig(
        **flat_kwargs,
        hierarchical=_parse_nested(section, "hierarchical", HierarchicalRagConfig),
        personal_history=_parse_nested(
            section, "personal_history", PersonalHistoryConfig
        ),
        category_expansion=_parse_nested(
            section, "category_expansion", CategoryExpansionConfig
        ),
    )


_WHISPER_LOCAL_SHIM_FIELDS = (
    "model",
    "initial_prompt",
    "beam_size",
    "batch_size",
    "condition_on_previous_text",
    "compression_ratio_threshold",
    "no_speech_threshold",
    "log_prob_threshold",
    "min_segment_duration",
    "max_segment_duration",
)


def _parse_transcription(data: dict[str, Any]) -> TranscriptionConfig:
    """Parse the ``transcription`` section with backward-compat shim.

    Resolution rules (spec §"旧キー → 新キーの後方互換 shim"):

    1. Read legacy ``indexing.whisper.*`` first.
    2. Read new ``transcription.whisper_local.*`` if present.
    3. New keys override old keys at field granularity.
    4. If only legacy keys are present, log a once-per-process
       deprecation warning naming the removal date.
    """
    global _whisper_deprecation_logged

    new_section_raw = data.get("transcription", {})
    new_section = new_section_raw if isinstance(new_section_raw, dict) else {}

    indexing_section = data.get("indexing", {})
    legacy_whisper_raw = (
        indexing_section.get("whisper", {})
        if isinstance(indexing_section, dict)
        else {}
    )
    legacy_whisper = (
        legacy_whisper_raw if isinstance(legacy_whisper_raw, dict) else {}
    )

    new_whisper_local_raw = new_section.get("whisper_local", {})
    new_whisper_local = (
        new_whisper_local_raw
        if isinstance(new_whisper_local_raw, dict)
        else {}
    )

    has_legacy = bool(legacy_whisper)
    has_new = bool(new_whisper_local)

    # Build whisper_local: legacy keys form the base; new keys override.
    merged_whisper_local: dict[str, Any] = {}
    for key in _WHISPER_LOCAL_SHIM_FIELDS:
        if key in legacy_whisper:
            merged_whisper_local[key] = legacy_whisper[key]
        if key in new_whisper_local:
            merged_whisper_local[key] = new_whisper_local[key]

    whisper_local = WhisperLocalConfig(**merged_whisper_local)

    if has_legacy and not has_new and not _whisper_deprecation_logged:
        logger.warning(
            "config: indexing.whisper.* is deprecated, will be removed "
            "%s. Move to transcription.whisper_local.*",
            _LEGACY_WHISPER_REMOVAL_DATE,
        )
        _whisper_deprecation_logged = True

    # Phase 2D: GUI-managed overrides live in
    # ``/intelligence-data/transcription-overrides.json`` and apply
    # only to the three top-level fields below. Provider sub-configs
    # (deepgram.model, openai_compatible.base_url, …) stay search-
    # config.yml's responsibility — the GUI does not edit them.
    from app.transcription_overrides import read_overrides

    overrides = read_overrides()
    if overrides is not None and overrides.provider is not None:
        provider_value = overrides.provider
    else:
        provider_value = new_section.get(
            "provider", TranscriptionConfig.provider
        )
    if overrides is not None and overrides.language_hint is not None:
        # Empty string is a meaningful "no hint" override; keep it.
        language_hint_value = overrides.language_hint
    else:
        language_hint_value = new_section.get(
            "language_hint", TranscriptionConfig.language_hint
        )
    if overrides is not None and overrides.hotwords is not None:
        hotwords_value = overrides.hotwords
    else:
        hotwords_value = tuple(new_section.get("hotwords", []) or [])

    return TranscriptionConfig(
        provider=provider_value,
        language_hint=language_hint_value,
        hotwords=hotwords_value,
        whisper_local=whisper_local,
        openai_compatible=_parse_nested(
            new_section, "openai_compatible", OpenAICompatibleProviderConfig
        ),
        deepgram=_parse_nested(
            new_section, "deepgram", DeepgramProviderConfig
        ),
        elevenlabs_scribe=_parse_nested(
            new_section, "elevenlabs_scribe", ElevenLabsScribeProviderConfig
        ),
        assemblyai=_parse_nested(
            new_section, "assemblyai", AssemblyAIProviderConfig
        ),
        gemini=_parse_nested(new_section, "gemini", GeminiProviderConfig),
    )


def _parse_indexing(data: dict[str, Any]) -> IndexingConfig:
    """Parse the indexing config section with nested sub-configs."""
    section = data.get("indexing", {})
    if not isinstance(section, dict):
        return IndexingConfig()

    return IndexingConfig(
        reconciliation_interval=section.get(
            "reconciliation_interval",
            IndexingConfig.reconciliation_interval,
        ),
        frame_extraction=_parse_nested(
            section, "frame_extraction", FrameExtractionConfig
        ),
        whisper=_parse_nested(section, "whisper", WhisperIndexConfig),
        text_chunking=_parse_nested(section, "text_chunking", TextChunkingConfig),
    )


def load_config_file(path: Path) -> dict[str, Any]:
    """Load YAML config file. Returns empty dict if file not found."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            result = yaml.safe_load(f)
            return result if isinstance(result, dict) else {}
    except (yaml.YAMLError, OSError) as e:
        print(f"Warning: Failed to load config file {path}: {e}")
        return {}


def load_settings() -> Settings:
    """Load settings from environment variables and config file.

    Environment variables take precedence over config file values.
    """
    import os

    intelligence_data_dir = Path(os.environ.get("INTELLIGENCE_DATA_DIR", "/intelligence-data"))
    litloft_db_path = Path(
        os.environ.get("HOMEVAULT_DB_PATH", "/data/litloft.db")
    )

    config_path = Path(
        os.environ.get("SEARCH_CONFIG_PATH", "/app/search-config.yml")
    )
    config_data = load_config_file(config_path)

    model_cache_dir = intelligence_data_dir / "models"
    search_db_path = intelligence_data_dir / "search.db"

    allowed_base_dirs_env = os.environ.get("ALLOWED_BASE_DIRS", "/drives/")
    allowed_base_dirs = tuple(d.strip() for d in allowed_base_dirs_env.split(",") if d.strip())

    # Parse DRIVE_MOUNTS: "動画=/drives/default,写真=/drives/photos"
    drive_mounts_env = os.environ.get("DRIVE_MOUNTS", "")
    drive_mounts: dict[str, str] = {}
    for entry in drive_mounts_env.split(","):
        entry = entry.strip()
        if "=" in entry:
            name, path = entry.split("=", 1)
            drive_mounts[name.strip()] = path.strip()

    # GUI overrides for features / llm / rag — see
    # ``app/{features,llm,rag}_overrides.py``. Each module reads a
    # tiny JSON file written by the admin GUI and silently no-ops
    # when the file is missing, so an unconfigured deployment keeps
    # using whatever ``search-config.yml`` ships.
    from app import embedding_overrides as _embedding_overrides
    from app import features_overrides as _features_overrides
    from app import llm_overrides as _llm_overrides
    from app import rag_overrides as _rag_overrides

    # Features (8 toggle gates)
    features_yaml_raw = config_data.get("features", {})
    features_yaml = features_yaml_raw if isinstance(features_yaml_raw, dict) else {}
    features_merged = _features_overrides.merge_into_dict(
        features_yaml, _features_overrides.read_overrides()
    )
    features_config = FeaturesConfig(
        **{
            k: v
            for k, v in features_merged.items()
            if k in FeaturesConfig.__dataclass_fields__
        }
    )

    # LLM (provider / base_url / model / output_language / vision_model
    # via GUI; api_key still env-only, sub-tuning still file-only)
    llm_yaml_raw = config_data.get("llm", {})
    llm_yaml = llm_yaml_raw if isinstance(llm_yaml_raw, dict) else {}
    llm_merged = _llm_overrides.merge_into_dict(
        llm_yaml, _llm_overrides.read_overrides()
    )
    llm_api_key_env = os.environ.get("LLM_API_KEY", "")
    if llm_api_key_env:
        # Env LLM_API_KEY wins over both the yaml field and any GUI
        # override path (secrets do not live in the data volume).
        llm_merged["api_key"] = llm_api_key_env
    # ``agentic_models`` arrives from yaml as a list of dicts; coerce
    # to a tuple of dataclasses before passing through so consumers
    # always see a typed sequence regardless of yaml authoring quirks.
    raw_agentic = llm_merged.get("agentic_models")
    if isinstance(raw_agentic, list):
        coerced: list[AgenticModelEntry] = []
        for entry in raw_agentic:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            context_window = entry.get("context_window", 32768)
            try:
                context_window_int = int(context_window)
            except (TypeError, ValueError):
                context_window_int = 32768
            coerced.append(
                AgenticModelEntry(
                    name=name.strip(),
                    context_window=max(1024, context_window_int),
                )
            )
        llm_merged["agentic_models"] = tuple(coerced)

    llm_config = LLMConfig(
        **{
            k: v
            for k, v in llm_merged.items()
            if k in LLMConfig.__dataclass_fields__
        }
    )

    # RAG (only ``personal_history.enabled`` and
    # ``category_expansion.enabled`` are GUI-overridable; everything
    # else stays file-only because it's tuned, not operator-facing.)
    rag_config = _parse_rag(config_data, _rag_overrides.read_overrides())

    # Models (only ``text_embedding`` is GUI-overridable; clip /
    # whisper / blip* / prefixes stay file-only). A model change
    # triggers a full vec_text rebuild at restart — see
    # ``database._migrate_vec_text_if_needed``. An allowlist-failing
    # override is dropped on read, so the yaml baseline (never a
    # silent dim-384 fallback) remains in effect (invariant §2.1-4).
    models_yaml_raw = config_data.get("models", {})
    models_yaml = (
        models_yaml_raw if isinstance(models_yaml_raw, dict) else {}
    )
    models_merged = _embedding_overrides.merge_into_dict(
        models_yaml, _embedding_overrides.read_overrides()
    )
    models_config = ModelConfig(
        **{
            k: v
            for k, v in models_merged.items()
            if k in ModelConfig.__dataclass_fields__
        }
    )

    return Settings(
        intelligence_data_dir=intelligence_data_dir,
        litloft_db_path=litloft_db_path,
        model_cache_dir=model_cache_dir,
        search_db_path=search_db_path,
        allowed_base_dirs=allowed_base_dirs,
        drive_mounts=drive_mounts,
        models=models_config,
        search=_parse_nested(config_data, "search", SearchConfig),
        indexing=_parse_indexing(config_data),
        workers=_parse_nested(config_data, "workers", WorkerConfig),
        memory=_parse_nested(config_data, "memory", MemoryConfig),
        features=features_config,
        llm=llm_config,
        summaries=_parse_nested(config_data, "summaries", SummariesConfig),
        rag=rag_config,
        transcription=_parse_transcription(config_data),
    )


def resolve_file_path(drive: str, relative_path: str) -> str | None:
    """Resolve a relative file path to an absolute path using drive mount mapping.

    Returns None if the drive has no mount configured.
    """
    mount = settings.drive_mounts.get(drive)
    if not mount:
        return None
    return str(Path(mount) / relative_path)


def validate_file_path(file_path: str) -> bool:
    """Validate that a file path resides within allowed base directories.

    Resolves symlinks before checking to prevent path traversal attacks.
    """
    import os

    real_path = os.path.realpath(file_path)
    return any(real_path.startswith(base) for base in settings.allowed_base_dirs)


def is_vision_describe_available(settings_obj: "Settings | None" = None) -> bool:
    """Return True iff the vision_describe feature is usable right now.

    Two independent gates: ``features.vision_describe`` must be set to a
    non-``"false"`` mode, AND ``llm.vision_model`` must be a non-empty,
    non-whitespace string (graceful degradation — an operator who set
    ``vision_describe: manual`` but forgot ``vision_model`` sees the
    feature silently disabled instead of producing confusing errors).
    """
    source = settings_obj if settings_obj is not None else settings
    mode = getattr(source.features, "vision_describe", "false")
    if mode == "false":
        return False
    model = getattr(source.llm, "vision_model", "") or ""
    return bool(model.strip())


# Module-level singleton, loaded once at import time
settings = load_settings()
