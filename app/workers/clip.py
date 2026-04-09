"""CLIP analysis worker for image and video frame embeddings.

Extracts visual features using CLIP (configurable via search-config.yml).
Supports built-in open_clip models and HuggingFace Hub models (hf-hub:).
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
from app.database import get_search_db
from app.models import Embedding, IndexedFile
from app.workers import blip
from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)

# Model state (lazy-loaded, thread-safe)
_lock = threading.Lock()
_model: object | None = None
_preprocess: object | None = None
_tokenizer: object | None = None
_loaded = False

# Model name → embedding dimension mapping
_CLIP_DIMS: dict[str, int] = {
    "openai/clip-vit-b-32": 512,
    "openai/clip-vit-b-16": 512,
    "llm-jp/llm-jp-clip-vit-base-patch16": 512,
    "llm-jp/llm-jp-clip-vit-large-patch14": 1024,
}

# Default dimension (overwritten after model loads with actual value)
CLIP_DIM = _CLIP_DIMS.get(settings.models.clip, 512)


# Image file types that CLIP can process directly
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}

# Video types that need frame extraction
VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime", "video/x-matroska"}



def _ensure_loaded() -> tuple[object, object, object]:
    """Lazy-load the CLIP model on first use.

    Supports two loading styles:
    - Built-in open_clip models: "openai/clip-vit-b-32" etc.
    - HuggingFace Hub models: loaded via "hf-hub:" prefix.

    Returns:
        Tuple of (model, preprocess, tokenizer).
    """
    global _model, _preprocess, _tokenizer, _loaded, CLIP_DIM

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

            # Built-in open_clip models (openai/*)
            builtin_map = {
                "openai/clip-vit-b-32": ("ViT-B-32", "openai"),
                "openai/clip-vit-b-16": ("ViT-B-16", "openai"),
            }

            if model_name in builtin_map:
                oc_model, oc_pretrained = builtin_map[model_name]
                _model, _, _preprocess = open_clip.create_model_and_transforms(
                    oc_model,
                    pretrained=oc_pretrained,
                    cache_dir=cache_dir,
                    device="cpu",
                )
                _tokenizer = open_clip.get_tokenizer(oc_model)
            else:
                # HuggingFace Hub models (llm-jp/*, etc.)
                hf_name = f"hf-hub:{model_name}"
                _model, _preprocess = open_clip.create_model_from_pretrained(
                    hf_name,
                    cache_dir=cache_dir,
                )
                _tokenizer = open_clip.get_tokenizer(hf_name)

            _model.eval()

            # Detect actual embedding dimension from the model
            CLIP_DIM = _detect_clip_dim(_model)

            _loaded = True
            logger.info(
                "CLIP model loaded successfully (dim=%d)", CLIP_DIM
            )
            return _model, _preprocess, _tokenizer

        except Exception as e:
            logger.error("Failed to load CLIP model: %s", e)
            raise RuntimeError(f"CLIP model load failed: {e}") from e


def _detect_clip_dim(model: object) -> int:
    """Detect the embedding dimension from a loaded CLIP model.

    Runs a dummy forward pass through encode_image to get the actual
    output dimension after projection. This is more reliable than
    inspecting visual.output_dim, which may return the pre-projection
    backbone dimension (e.g. 512 for ViT-B) rather than the final
    projected dimension (e.g. 768 for llm-jp models).

    Args:
        model: A loaded open_clip model.

    Returns:
        The embedding dimension (e.g. 512, 768, 1024).
    """
    import torch

    # Preferred: dummy forward pass through image encoder for actual output dim
    try:
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = model.encode_image(dummy)
        return out.shape[-1]
    except Exception:
        pass

    # Fallback: dummy forward pass through text encoder
    try:
        dummy = torch.zeros(1, 77, dtype=torch.long)
        with torch.no_grad():
            out = model.encode_text(dummy)
        return out.shape[-1]
    except Exception:
        pass

    # Last resort: use the static mapping
    return _CLIP_DIMS.get(settings.models.clip, 512)


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


def _extract_frame_paths(
    video_path: str, duration: float | None, output_dir: Path
) -> list[tuple[float, Path]]:
    """Extract key frames from a video to disk using hybrid scene detection.

    Uses ffmpeg scene detection with minimum interval guarantee.
    Frames are saved as JPEG files in output_dir. The caller is
    responsible for managing the lifetime of output_dir.

    Args:
        video_path: Path to the video file.
        duration: Video duration in seconds (None if unknown).
        output_dir: Directory to save extracted frame files.

    Returns:
        List of (timestamp_seconds, frame_path) tuples.
    """
    frame_config = settings.indexing.frame_extraction

    # Step 1: Scene detection frames
    scene_frames = _extract_scene_frames(
        video_path, output_dir, frame_config.scene_threshold
    )

    # Step 2: Fill gaps with interval-based frames
    if duration is not None and duration > 0:
        all_frames = _fill_interval_gaps(
            video_path,
            output_dir,
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
        # Use showinfo filter to log timestamps, then parse them from stderr.
        # select + showinfo outputs one line per selected frame with pts_time.
        result = subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vf", (
                    f"select='gt(scene,{threshold})',"
                    "showinfo,"
                    "scale=224:224,"
                    "format=yuvj420p"
                ),
                "-fps_mode", "passthrough",
                "-q:v", "5",
                str(output_dir / "scene_%04d.jpg"),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        frame_files = sorted(output_dir.glob("scene_*.jpg"))
        if not frame_files:
            if result.returncode != 0:
                logger.warning("Scene detection failed: %s", result.stderr[:200])
            return []

        # Parse pts_time from showinfo lines in stderr
        import re
        pts_times: list[float] = []
        for line in result.stderr.splitlines():
            m = re.search(r"pts_time:\s*([\d.]+)", line)
            if m:
                pts_times = [*pts_times, float(m.group(1))]

        frames: list[tuple[float, Path]] = []
        for idx, frame_file in enumerate(frame_files):
            timestamp = pts_times[idx] if idx < len(pts_times) else float(idx * 30)
            frames = [*frames, (timestamp, frame_file)]

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
                    "-vf", "scale=224:224,format=yuvj420p",
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
    return await asyncio.to_thread(_index_clip_sync, file_id)


def _index_clip_sync(file_id: str) -> bool:
    """Synchronous CLIP indexing implementation.

    For images: single embedding, same as before.
    For videos: streams frames in batches to limit peak memory.
    Each batch is loaded from disk, embedded, and written to DB
    before the next batch is loaded.

    Args:
        file_id: The file ID to index.

    Returns:
        True if indexing succeeded.
    """
    # --- Phase 1: Read file info (short DB access) ---
    with get_search_db() as session:
        file = session.query(IndexedFile).filter_by(
            file_id=file_id, active=True
        ).first()

        if file is None:
            return False

        file_path = file.file_path
        mime_type = file.mime_type
        filename = file.filename
        duration = file.duration

    if mime_type not in IMAGE_TYPES and mime_type not in VIDEO_TYPES:
        with get_search_db() as session:
            file = session.query(IndexedFile).filter_by(file_id=file_id).first()
            if file is not None:
                file.clip_indexed = True
        return True

    if not validate_file_path(file_path):
        logger.error("File path validation failed for %s: %s", file_id, file_path)
        with get_search_db() as session:
            f = session.query(IndexedFile).filter_by(file_id=file_id).first()
            if f is not None:
                f.clip_indexed = True
        return False

    # --- Image: single embedding (no streaming needed) ---
    if mime_type in IMAGE_TYPES:
        return _index_clip_image(file_id, file_path, filename)

    # --- Video: streaming batch processing ---
    return _index_clip_video(file_id, file_path, duration)


def _index_clip_image(file_id: str, file_path: str, filename: str) -> bool:
    """Index a single image file with CLIP, and optionally generate BLIP caption.

    Args:
        file_id: The file ID.
        file_path: Path to the image file.
        filename: Original filename for preview text.

    Returns:
        True if indexing succeeded.
    """
    try:
        image = Image.open(file_path).convert("RGB")
        vector = embed_image(image)
        embedding_id = f"clip_{file_id}_{uuid.uuid4().hex[:8]}"
    except Exception as e:
        logger.error("CLIP compute failed for %s: %s", file_id, e)
        return False

    with get_search_db() as session:
        _remove_clip_embeddings(session, file_id)
        _store_clip_embedding(
            session=session,
            embedding_id=embedding_id,
            file_id=file_id,
            vector=vector,
            content_preview=f"Image: {filename}",
        )
        file = session.query(IndexedFile).filter_by(file_id=file_id).first()
        if file is not None:
            file.clip_indexed = True

    # Generate BLIP caption for the image (reuses the already-loaded PIL image)
    _generate_blip_caption_if_needed(file_id, image)

    return True


def _index_clip_video(file_id: str, file_path: str, duration: float | None) -> bool:
    """Index video frames with CLIP using streaming batch processing.

    Extracts frames to a temp directory, then processes them in
    batches of clip_frame_batch_size to limit peak memory usage.

    Args:
        file_id: The file ID.
        file_path: Path to the video file.
        duration: Video duration in seconds.

    Returns:
        True if indexing succeeded.
    """
    batch_size = settings.workers.clip_frame_batch_size

    try:
        with tempfile.TemporaryDirectory(prefix="clip_frames_") as tmpdir:
            tmp_path = Path(tmpdir)
            frame_paths = _extract_frame_paths(file_path, duration, tmp_path)

            if not frame_paths:
                logger.warning("No frames extracted for %s", file_id)
                with get_search_db() as session:
                    f = session.query(IndexedFile).filter_by(file_id=file_id).first()
                    if f is not None:
                        f.clip_indexed = True
                return True

            # Remove old embeddings once before streaming new ones
            with get_search_db() as session:
                _remove_clip_embeddings(session, file_id)

            # Process frames in batches
            total_stored = 0
            for batch_start in range(0, len(frame_paths), batch_size):
                batch = frame_paths[batch_start:batch_start + batch_size]
                stored = _process_frame_batch(file_id, batch)
                total_stored += stored

            logger.info(
                "CLIP indexed %d/%d frames for %s",
                total_stored, len(frame_paths), file_id,
            )

            # Generate BLIP caption from the first extracted frame
            if frame_paths:
                _generate_blip_caption_for_video_frame(file_id, frame_paths[0])

    except Exception as e:
        logger.error("CLIP video indexing failed for %s: %s", file_id, e)
        return False

    with get_search_db() as session:
        file = session.query(IndexedFile).filter_by(file_id=file_id).first()
        if file is not None:
            file.clip_indexed = True

    return True


def _process_frame_batch(
    file_id: str,
    frame_paths: list[tuple[float, Path]],
) -> int:
    """Load, embed, and store a batch of video frames.

    Each frame is loaded from disk, embedded with CLIP, and the
    result is written to the database. PIL images are released
    after embedding to keep memory bounded.

    Args:
        file_id: The file ID.
        frame_paths: List of (timestamp, path) tuples for this batch.

    Returns:
        Number of embeddings successfully stored.
    """
    embeddings: list[tuple[str, np.ndarray, str, float, float]] = []

    for timestamp, frame_path in frame_paths:
        try:
            img = Image.open(str(frame_path)).convert("RGB")
            img.load()
            vector = embed_image(img)
            del img
            embedding_id = f"clip_{file_id}_{uuid.uuid4().hex[:8]}"
            embeddings = [
                *embeddings,
                (embedding_id, vector, f"Frame at {timestamp:.1f}s", timestamp, timestamp + 30),
            ]
        except Exception as e:
            logger.warning(
                "Failed to process frame at %.1fs for %s: %s",
                timestamp, file_id, e,
            )

    if not embeddings:
        return 0

    with get_search_db() as session:
        for emb_id, vector, preview, ts_start, ts_end in embeddings:
            _store_clip_embedding(
                session=session,
                embedding_id=emb_id,
                file_id=file_id,
                vector=vector,
                content_preview=preview,
                timestamp_start=ts_start,
                timestamp_end=ts_end,
            )

    return len(embeddings)


def _has_blip_caption(file_id: str) -> bool:
    """Check if a BLIP caption already exists for a file.

    Args:
        file_id: The file ID to check.

    Returns:
        True if a blip_caption embedding exists.
    """
    with get_search_db() as session:
        count = (
            session.query(Embedding)
            .filter_by(file_id=file_id, embedding_type="blip_caption")
            .count()
        )
        return count > 0


def _store_blip_caption(file_id: str, caption: str) -> None:
    """Store a BLIP caption as an embedding record (no vector).

    Args:
        file_id: The file ID this caption belongs to.
        caption: The generated caption text.
    """
    embedding_id = f"blip_{file_id}_{uuid.uuid4().hex[:8]}"

    with get_search_db() as session:
        # Remove any existing BLIP captions for this file
        existing = (
            session.query(Embedding)
            .filter_by(file_id=file_id, embedding_type="blip_caption")
            .all()
        )
        for emb in existing:
            session.delete(emb)
        if existing:
            session.flush()

        record = Embedding(
            id=embedding_id,
            file_id=file_id,
            embedding_type="blip_caption",
            vector_table="",
            content_preview=caption[:500],
        )
        session.add(record)

    logger.info("BLIP caption stored for %s: %s", file_id, caption[:80])


def _generate_blip_caption_if_needed(
    file_id: str, image: Image.Image
) -> None:
    """Generate and store a BLIP caption for an image if BLIP is enabled.

    Skips if BLIP is disabled or a caption already exists.
    Logs and continues on errors without crashing.

    Args:
        file_id: The file ID.
        image: PIL Image (already loaded for CLIP).
    """
    if not blip.is_enabled():
        return

    if _has_blip_caption(file_id):
        return

    try:
        caption = blip.generate_caption(image)
        if caption:
            _store_blip_caption(file_id, caption)
    except Exception as e:
        logger.error("BLIP captioning failed for %s: %s", file_id, e)


def _generate_blip_caption_for_video_frame(
    file_id: str, frame_info: tuple[float, Path]
) -> None:
    """Generate and store a BLIP caption from a video frame.

    Uses the first extracted frame as a representative image
    to give auto_tags visual context for videos.

    Args:
        file_id: The file ID.
        frame_info: Tuple of (timestamp, frame_path).
    """
    if not blip.is_enabled():
        return

    if _has_blip_caption(file_id):
        return

    _, frame_path = frame_info
    try:
        image = Image.open(str(frame_path)).convert("RGB")
        caption = blip.generate_caption(image)
        del image
        if caption:
            _store_blip_caption(file_id, caption)
    except Exception as e:
        logger.error("BLIP video captioning failed for %s: %s", file_id, e)


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

    vec_bytes = vector.tobytes()
    session.execute(
        sql_text("INSERT INTO vec_clip(embedding_id, vector) VALUES(:id, :vec)"),
        {"id": embedding_id, "vec": vec_bytes},
    )


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

    for emb in existing:
        session.execute(
            sql_text("DELETE FROM vec_clip WHERE embedding_id = :id"),
            {"id": emb.id},
        )

    for emb in existing:
        session.delete(emb)
    session.flush()
