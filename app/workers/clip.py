"""CLIP analysis worker for image and video frame embeddings.

Extracts visual features using CLIP (ViT-B/32 by default, configurable).
For videos, uses hybrid frame extraction: scene detection + minimum interval.
"""

import asyncio
import logging
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings, validate_file_path
from app.database import get_search_db, get_search_engine
from app.models import Embedding, IndexedFile
from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)

# Model state (lazy-loaded, thread-safe)
_lock = threading.Lock()
_model: object | None = None
_preprocess: object | None = None
_tokenizer: object | None = None
_loaded = False

# CLIP produces 512-dimensional vectors (ViT-B/32)
CLIP_DIM = 512

# Semaphore for parallel CLIP processing
_semaphore: asyncio.Semaphore | None = None

# Image file types that CLIP can process directly
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}

# Video types that need frame extraction
VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime", "video/x-matroska"}


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the CLIP processing semaphore."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.workers.clip_parallel)
    return _semaphore


def _ensure_loaded() -> tuple[object, object, object]:
    """Lazy-load the CLIP model on first use.

    Returns:
        Tuple of (model, preprocess, tokenizer).
    """
    global _model, _preprocess, _tokenizer, _loaded

    if _loaded and _model is not None:
        return _model, _preprocess, _tokenizer

    with _lock:
        if _loaded and _model is not None:
            return _model, _preprocess, _tokenizer

        try:
            import open_clip

            model_name = settings.models.clip
            cache_dir = str(settings.model_cache_dir)

            logger.info("Loading CLIP model: %s", model_name)

            # Map config model names to open_clip model/pretrained pairs
            clip_config = _resolve_clip_model(model_name)

            _model, _, _preprocess = open_clip.create_model_and_transforms(
                clip_config["model_name"],
                pretrained=clip_config["pretrained"],
                cache_dir=cache_dir,
            )
            _tokenizer = open_clip.get_tokenizer(clip_config["model_name"])

            _model.eval()
            _loaded = True
            logger.info("CLIP model loaded successfully")
            return _model, _preprocess, _tokenizer

        except Exception as e:
            logger.error("Failed to load CLIP model: %s", e)
            raise RuntimeError(f"CLIP model load failed: {e}") from e


def _resolve_clip_model(config_name: str) -> dict[str, str]:
    """Resolve config model name to open_clip model/pretrained pair.

    Args:
        config_name: Model name from search-config.yml.

    Returns:
        Dict with model_name and pretrained keys.
    """
    model_map = {
        "openai/clip-vit-b-32": {
            "model_name": "ViT-B-32",
            "pretrained": "openai",
        },
        "rinna/japanese-clip-vit-b-16": {
            "model_name": "ViT-B-16",
            "pretrained": "laion2b_s34b_b88k",
        },
    }

    resolved = model_map.get(config_name)
    if resolved is not None:
        return resolved

    # Default fallback
    return {"model_name": "ViT-B-32", "pretrained": "openai"}


def embed_image(image: Image.Image) -> np.ndarray:
    """Generate CLIP embedding for a single image.

    Args:
        image: PIL Image object.

    Returns:
        Normalized embedding vector of shape (CLIP_DIM,).
    """
    import torch

    model, preprocess, _ = _ensure_loaded()

    preprocessed = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        features = model.encode_image(preprocessed)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze().cpu().numpy().astype(np.float32)


def embed_text_clip(text: str) -> np.ndarray:
    """Generate CLIP text embedding for a search query.

    Args:
        text: Search query text.

    Returns:
        Normalized embedding vector of shape (CLIP_DIM,).
    """
    import torch

    model, _, tokenizer = _ensure_loaded()

    tokens = tokenizer([text])

    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze().cpu().numpy().astype(np.float32)


def _extract_frames_from_video(
    video_path: str, duration: float | None
) -> list[tuple[float, Path]]:
    """Extract key frames from a video using hybrid scene detection.

    Uses ffmpeg scene detection with minimum interval guarantee.

    Args:
        video_path: Path to the video file.
        duration: Video duration in seconds (None if unknown).

    Returns:
        List of (timestamp_seconds, frame_path) tuples.
    """
    frame_config = settings.indexing.frame_extraction

    with tempfile.TemporaryDirectory(prefix="clip_frames_") as tmpdir:
        tmp_path = Path(tmpdir)

        # Step 1: Scene detection frames
        scene_frames = _extract_scene_frames(
            video_path, tmp_path, frame_config.scene_threshold
        )

        # Step 2: Fill gaps with interval-based frames
        if duration is not None and duration > 0:
            all_frames = _fill_interval_gaps(
                video_path,
                tmp_path,
                scene_frames,
                duration,
                frame_config.min_interval,
                frame_config.max_frames,
            )
        else:
            all_frames = scene_frames

        # Limit total frames
        return all_frames[: frame_config.max_frames]


def _extract_scene_frames(
    video_path: str, output_dir: Path, threshold: float
) -> list[tuple[float, Path]]:
    """Extract frames at scene change points using ffmpeg.

    Args:
        video_path: Path to the video.
        output_dir: Directory to save extracted frames.
        threshold: Scene detection threshold (0.0-1.0).

    Returns:
        List of (timestamp, frame_path) tuples.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vf", f"select='gt(scene,{threshold})',scale=224:224",
                "-vsync", "vfp",
                "-frame_pts", "1",
                str(output_dir / "scene_%04d.jpg"),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            logger.warning("Scene detection failed: %s", result.stderr[:200])
            return []

        # Parse frame timestamps from showinfo or use frame numbers
        frames: list[tuple[float, Path]] = []
        frame_files = sorted(output_dir.glob("scene_*.jpg"))

        # Get timestamps using ffprobe
        for frame_file in frame_files:
            frame_num = int(frame_file.stem.split("_")[1])
            # Approximate timestamp from frame number
            frames = [*frames, (float(frame_num), frame_file)]

        return frames

    except (subprocess.TimeoutExpired, OSError) as e:
        logger.error("Frame extraction failed: %s", e)
        return []


def _fill_interval_gaps(
    video_path: str,
    output_dir: Path,
    existing_frames: list[tuple[float, Path]],
    duration: float,
    min_interval: int,
    max_frames: int,
) -> list[tuple[float, Path]]:
    """Fill gaps between scene frames with interval-based frames.

    Args:
        video_path: Path to the video.
        output_dir: Directory to save extracted frames.
        existing_frames: Already extracted scene frames.
        duration: Video duration in seconds.
        min_interval: Minimum seconds between frames.
        max_frames: Maximum total frames.

    Returns:
        Combined and sorted list of (timestamp, frame_path) tuples.
    """
    existing_times = {t for t, _ in existing_frames}

    # Generate interval timestamps that aren't covered by scene detection
    needed_times: list[float] = []
    current_time = 0.0

    while current_time < duration:
        # Check if any existing frame is within min_interval of this time
        is_covered = any(
            abs(current_time - et) < min_interval for et in existing_times
        )
        if not is_covered:
            needed_times = [*needed_times, current_time]
        current_time += min_interval

    if not needed_times:
        return existing_frames

    # Extract interval frames
    interval_frames: list[tuple[float, Path]] = []

    for idx, timestamp in enumerate(needed_times):
        if len(existing_frames) + len(interval_frames) >= max_frames:
            break

        frame_path = output_dir / f"interval_{idx:04d}.jpg"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-ss", str(timestamp),
                    "-i", video_path,
                    "-vf", "scale=224:224",
                    "-frames:v", "1",
                    "-q:v", "5",
                    str(frame_path),
                ],
                capture_output=True,
                timeout=30,
            )
            if frame_path.exists():
                interval_frames = [*interval_frames, (timestamp, frame_path)]
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("Interval frame extraction failed at %ss: %s", timestamp, e)

    combined = [*existing_frames, *interval_frames]
    return sorted(combined, key=lambda x: x[0])


async def index_clip(file_id: str) -> bool:
    """Index CLIP embeddings for a file (image or video).

    For images: single embedding per file.
    For videos: one embedding per extracted key frame.

    Args:
        file_id: The file ID to index.

    Returns:
        True if indexing succeeded.
    """
    sem = _get_semaphore()
    async with sem:
        return await asyncio.to_thread(_index_clip_sync, file_id)


def _index_clip_sync(file_id: str) -> bool:
    """Synchronous CLIP indexing implementation.

    Args:
        file_id: The file ID to index.

    Returns:
        True if indexing succeeded.
    """
    with get_search_db() as session:
        file = session.query(IndexedFile).filter_by(
            file_id=file_id, active=True
        ).first()

        if file is None:
            return False

        # Remove old CLIP embeddings
        _remove_clip_embeddings(session, file_id)

        if file.mime_type in IMAGE_TYPES:
            return _index_image_clip(session, file)
        elif file.mime_type in VIDEO_TYPES:
            return _index_video_clip(session, file)
        else:
            file.clip_indexed = True
            return True


def _index_image_clip(session: object, file: IndexedFile) -> bool:
    """Index CLIP embedding for a single image.

    Args:
        session: Database session.
        file: The indexed file record.

    Returns:
        True if successful.
    """
    try:
        if not validate_file_path(file.file_path):
            logger.error("File path validation failed for %s: %s", file.file_id, file.file_path)
            return False
        image = Image.open(file.file_path).convert("RGB")
        vector = embed_image(image)

        embedding_id = f"clip_{file.file_id}_{uuid.uuid4().hex[:8]}"
        _store_clip_embedding(
            session=session,
            embedding_id=embedding_id,
            file_id=file.file_id,
            vector=vector,
            content_preview=f"Image: {file.filename}",
        )

        file.clip_indexed = True
        return True

    except Exception as e:
        logger.error("CLIP image indexing failed for %s: %s", file.file_id, e)
        return False


def _index_video_clip(session: object, file: IndexedFile) -> bool:
    """Index CLIP embeddings for video key frames.

    Args:
        session: Database session.
        file: The indexed file record.

    Returns:
        True if successful.
    """
    try:
        if not validate_file_path(file.file_path):
            logger.error("File path validation failed for %s: %s", file.file_id, file.file_path)
            return False
        frames = _extract_frames_from_video(file.file_path, file.duration)

        if not frames:
            logger.warning("No frames extracted for %s", file.file_id)
            file.clip_indexed = True
            return True

        for timestamp, frame_path in frames:
            try:
                image = Image.open(str(frame_path)).convert("RGB")
                vector = embed_image(image)

                embedding_id = f"clip_{file.file_id}_{uuid.uuid4().hex[:8]}"
                _store_clip_embedding(
                    session=session,
                    embedding_id=embedding_id,
                    file_id=file.file_id,
                    vector=vector,
                    content_preview=f"Frame at {timestamp:.1f}s",
                    timestamp_start=timestamp,
                    timestamp_end=timestamp + 30,
                )
            except Exception as e:
                logger.warning(
                    "Failed to process frame at %ss for %s: %s",
                    timestamp, file.file_id, e,
                )

        file.clip_indexed = True
        return True

    except Exception as e:
        logger.error("CLIP video indexing failed for %s: %s", file.file_id, e)
        return False


def _store_clip_embedding(
    session: object,
    embedding_id: str,
    file_id: str,
    vector: np.ndarray,
    content_preview: str = "",
    timestamp_start: float | None = None,
    timestamp_end: float | None = None,
) -> None:
    """Store a CLIP embedding in the database.

    Args:
        session: Database session.
        embedding_id: Unique embedding ID.
        file_id: File this embedding belongs to.
        vector: CLIP embedding vector.
        content_preview: Human-readable description.
        timestamp_start: Optional start timestamp.
        timestamp_end: Optional end timestamp.
    """
    embedding_record = Embedding(
        id=embedding_id,
        file_id=file_id,
        embedding_type="clip",
        vector_table="vec_clip",
        content_preview=content_preview[:500],
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
    )
    session.add(embedding_record)
    session.flush()

    engine = get_search_engine()
    vec_bytes = vector.tobytes()
    with engine.connect() as conn:
        conn.execute(
            sql_text("INSERT INTO vec_clip(embedding_id, vector) VALUES(:id, :vec)"),
            {"id": embedding_id, "vec": vec_bytes},
        )
        conn.commit()


def _remove_clip_embeddings(session: object, file_id: str) -> None:
    """Remove existing CLIP embeddings for a file.

    Args:
        session: Database session.
        file_id: The file ID.
    """
    existing = (
        session.query(Embedding)
        .filter_by(file_id=file_id, embedding_type="clip")
        .all()
    )

    if not existing:
        return

    engine = get_search_engine()
    with engine.connect() as conn:
        for emb in existing:
            conn.execute(
                sql_text("DELETE FROM vec_clip WHERE embedding_id = :id"),
                {"id": emb.id},
            )
        conn.commit()

    for emb in existing:
        session.delete(emb)
    session.flush()
