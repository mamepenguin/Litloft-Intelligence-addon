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


@dataclass(frozen=True)
class FeaturesConfig:
    indexing: bool = True
    search: bool = True
    auto_tags: bool = False


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "disabled"  # "openai_compatible" | "disabled"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    max_tokens: int = 256
    temperature: float = 0.3


@dataclass(frozen=True)
class Settings:
    search_data_dir: Path
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

    search_data_dir = Path(os.environ.get("SEARCH_DATA_DIR", "/search-data"))
    homevault_db_path = Path(
        os.environ.get("HOMEVAULT_DB_PATH", "/data/homevault.db")
    )

    config_path = Path(
        os.environ.get("SEARCH_CONFIG_PATH", "/app/search-config.yml")
    )
    config_data = load_config_file(config_path)

    model_cache_dir = search_data_dir / "models"
    search_db_path = search_data_dir / "search.db"

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
        )

    return Settings(
        search_data_dir=search_data_dir,
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
