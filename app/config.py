"""Configuration for the semantic search service.

Reads settings from environment variables and search-config.yml.
All config values are immutable after initialization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    whisper: str = "openai/whisper-small"
    text_embedding: str = "intfloat/multilingual-e5-small"
    clip: str = "llm-jp/llm-jp-clip-vit-base-patch16"
    blip: str = ""  # empty = disabled. e.g. "Salesforce/blip-image-captioning-base"
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
    min_score_clip: float = 0.25
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
    # Legacy (kept for config file compatibility, used by compare endpoint)
    rrf_k: int = 60
    rrf_weight_clip: float = 0.5


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
    # Short example sentence (≤224 tokens) used to bias the decoder
    # toward a desired style — primarily punctuation insertion. Written
    # in the user's primary content language; an irrelevant prompt mostly
    # wastes token budget rather than corrupting output. Do not include
    # filenames or curated vocabulary here; put those in a dedicated
    # glossary layer if needed.
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


@dataclass(frozen=True)
class FeaturesConfig:
    indexing: bool = True
    search: bool = True
    auto_tags: str = "false"  # "false" | "manual" | "on_index"
    summaries: str = "false"  # "false" | "manual" | "on_index"
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
    # response to user queries. Default off for security — file content
    # (transcripts, captions, text) is sent to the LLM API.
    rag: bool = False
    # Transcript AI refine ("false" | "manual" | "on_index"). Default off
    # since file contents are sent to the LLM API during refine.
    transcript_refine: str = "false"
    # Vision-LLM image description ("false" | "manual" | "on_index").
    # Default off — enabling sends image bytes to the LLM API. Requires
    # ``llm.vision_model`` to be set or the feature is unavailable even
    # when this flag is "manual"/"on_index" (graceful degradation).
    vision_describe: str = "false"


@dataclass(frozen=True)
class HierarchicalRagConfig:
    """Hierarchical RAG (Stage 1 coarse-retrieval) configuration.

    Phase 1 (shadow mode) only logs the shortlist; Phase 2 actually
    scopes the chunk-level retrieval to the shortlist. See
    ``docs/superpowers/specs/2026-04-26-intelligence-ask-hierarchical-retrieval.md``.

    Defaults are conservative: hierarchical is OFF until an operator
    opts in, the bypass thresholds err on the side of running the
    legacy full-file retrieval when in doubt.
    """

    # Master switch. False keeps the legacy single-stage retrieval.
    enabled: bool = False
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
class RagConfig:
    """RAG (question answering) configuration.

    Controls how many files are retrieved, how much context per file
    is extracted, the hard total-context cap, and the LLM max_tokens
    override for answer generation. Defaults match spec Phase A.
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
    # Hierarchical retrieval (Stage 1 coarse shortlist). Default
    # disabled — opt-in via ``rag.hierarchical.enabled: true`` in
    # ``search-config.yml``.
    hierarchical: HierarchicalRagConfig = field(
        default_factory=HierarchicalRagConfig
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
class LLMConfig:
    provider: str = "disabled"  # "openai_compatible" | "ollama" | "disabled"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    max_tokens: int = 2048
    temperature: float = 0.3
    output_language: str = "auto"  # "auto" | "ja" | "en" | etc. — applies to auto_tags and summaries
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
    service_version: str = "0.1.0"
    port: int = 8100


def _parse_nested(data: dict[str, Any], key: str, cls: type) -> Any:
    """Parse a nested config section into a frozen dataclass."""
    section = data.get(key, {})
    if not isinstance(section, dict):
        return cls()
    return cls(**{k: v for k, v in section.items() if k in cls.__dataclass_fields__})


def _parse_rag(data: dict[str, Any]) -> RagConfig:
    """Parse the rag config section with the nested hierarchical block.

    Mirrors ``_parse_indexing`` so unknown top-level keys are dropped
    rather than blowing up dataclass construction, and the nested
    ``hierarchical`` sub-section hydrates via ``_parse_nested``.
    """
    section = data.get("rag", {})
    if not isinstance(section, dict):
        return RagConfig()

    flat_kwargs = {
        k: v
        for k, v in section.items()
        if k in RagConfig.__dataclass_fields__ and k != "hierarchical"
    }
    return RagConfig(
        **flat_kwargs,
        hierarchical=_parse_nested(section, "hierarchical", HierarchicalRagConfig),
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

    # Parse LLM config with env var override for api_key
    llm_config = _parse_nested(config_data, "llm", LLMConfig)
    llm_api_key_env = os.environ.get("LLM_API_KEY", "")
    if llm_api_key_env:
        # Rebuild with overridden api_key (frozen dataclass)
        llm_config = LLMConfig(
            provider=llm_config.provider,
            base_url=llm_config.base_url,
            api_key=llm_api_key_env,
            model=llm_config.model,
            max_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
            output_language=llm_config.output_language,
            retry_attempts=llm_config.retry_attempts,
            retry_base_delay=llm_config.retry_base_delay,
            retry_max_delay=llm_config.retry_max_delay,
            min_request_interval_ms=llm_config.min_request_interval_ms,
            request_timeout_seconds=llm_config.request_timeout_seconds,
            request_connect_timeout_seconds=llm_config.request_connect_timeout_seconds,
            vision_model=llm_config.vision_model,
            vision_max_tokens=llm_config.vision_max_tokens,
            vision_temperature=llm_config.vision_temperature,
        )

    return Settings(
        intelligence_data_dir=intelligence_data_dir,
        litloft_db_path=litloft_db_path,
        model_cache_dir=model_cache_dir,
        search_db_path=search_db_path,
        allowed_base_dirs=allowed_base_dirs,
        drive_mounts=drive_mounts,
        models=_parse_nested(config_data, "models", ModelConfig),
        search=_parse_nested(config_data, "search", SearchConfig),
        indexing=_parse_indexing(config_data),
        workers=_parse_nested(config_data, "workers", WorkerConfig),
        memory=_parse_nested(config_data, "memory", MemoryConfig),
        features=_parse_nested(config_data, "features", FeaturesConfig),
        llm=llm_config,
        summaries=_parse_nested(config_data, "summaries", SummariesConfig),
        rag=_parse_rag(config_data),
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
