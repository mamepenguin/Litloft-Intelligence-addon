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
    batch_size: int = 16
    condition_on_previous_text: bool = False
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
    max_chunk_size: int = 1000
    overlap: int = 200


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
    # RAG (question answering) is a simple on/off switch: there is no
    # index-time equivalent to "on_index" because RAG only runs in
    # response to user queries. Default off for security — file content
    # (transcripts, captions, text) is sent to the LLM API.
    rag: bool = False
    # Transcript AI refine ("false" | "manual" | "on_index"). Default off
    # since file contents are sent to the LLM API during refine.
    transcript_refine: str = "false"


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


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "disabled"  # "openai_compatible" | "disabled"
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


@dataclass(frozen=True)
class Settings:
    intelligence_data_dir: Path
    homevault_db_path: Path
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
    homevault_db_path = Path(
        os.environ.get("HOMEVAULT_DB_PATH", "/data/homevault.db")
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
        )

    return Settings(
        intelligence_data_dir=intelligence_data_dir,
        homevault_db_path=homevault_db_path,
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
        rag=_parse_nested(config_data, "rag", RagConfig),
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


# Module-level singleton, loaded once at import time
settings = load_settings()
