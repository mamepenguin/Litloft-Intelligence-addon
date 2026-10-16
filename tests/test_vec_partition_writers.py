"""SPEC-ADDON-022 I3: whatever writes or rewrites an embedding leaves its vec row
keyed by the embedding's type.

Each writer runs against a search DB built by ``init_search_db``, through the
app's own sessions, so a writer that omits the partition column stores NULL there
(sqlite-vec accepts that) and is caught here rather than by a KNN that silently
never returns the row.
"""

from __future__ import annotations

import shutil
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.config as config
import app.database as database
from app.models import TranscriptChunk, VideoVisualRun, VideoVisualScene
from app.workers import clip as clip_worker
from app.workers import metadata as metadata_worker
from app.workers import whisper as whisper_worker
from app.workers.transcription import ProviderCapabilities
from app.workers.transcription.base import TranscriptionSegment, WordToken
from tests.vec_partition_db import (  # noqa: F401
    VEC_TABLES,
    random_vector,
    search_index,
    start,
    unit,
)


@pytest.fixture()
def embedders(monkeypatch, search_index):
    """Deterministic embedders at the dimensions the tables were created with."""
    text_dim = search_index.dim("vec_text")
    clip_dim = search_index.dim("vec_clip")

    def _text(arg, *args, **kwargs):
        if isinstance(arg, str):
            return unit(text_dim, 1)
        return [unit(text_dim, i + 1) for i, _ in enumerate(arg)]

    import app.workers.embedder as embedder

    for name in ("embed_passages", "embed_query", "embed_texts"):
        monkeypatch.setattr(embedder, name, _text)
    monkeypatch.setattr(metadata_worker, "embed_passages", _text)
    monkeypatch.setattr(whisper_worker, "embed_passages", _text)
    monkeypatch.setattr(clip_worker, "embed_image", lambda image: unit(clip_dim, 2))
    monkeypatch.setattr(clip_worker, "embed_text_clip", lambda *a, **k: unit(clip_dim, 3))
    monkeypatch.setattr(config, "validate_file_path", lambda _path: True)
    return search_index


# ---------------------------------------------------------------------------
# metadata._store_embedding
# ---------------------------------------------------------------------------


def test_spec_addon_022_a_document_chunk_is_keyed_text_content(embedders, tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("fermenting koji at home, step by step.\n" * 20, encoding="utf-8")
    embedders.add_file("doc1", mime_type="text/plain", file_path=str(doc))

    assert metadata_worker.index_text_content("doc1") is True

    assert "text_content" in embedders.assert_partition_matches_type("doc1")


def test_spec_addon_022_a_metadata_embedding_is_keyed_metadata(embedders):
    embedders.add_file("vid1", mime_type="video/mp4")

    metadata_worker.index_metadata_batch(["vid1"])

    assert "metadata" in embedders.assert_partition_matches_type("vid1")


# ---------------------------------------------------------------------------
# whisper: _index_tfidf_keywords, _persist_transcript, _index_loft_vtt
# ---------------------------------------------------------------------------


def test_spec_addon_022_a_keyword_embedding_is_keyed_tfidf_keywords(embedders, monkeypatch):
    import app.tfidf as tfidf_mod

    monkeypatch.setattr(
        tfidf_mod, "extract_top_keywords",
        lambda text_arg, filename, *, drive, k=30: ["カレー", "玉ねぎ"],
    )
    embedders.add_file("vid1", mime_type="video/mp4")

    whisper_worker._index_tfidf_keywords("vid1", ["今日はカレーを作ります。" * 3])

    assert "tfidf_keywords" in embedders.assert_partition_matches_type("vid1")


def _provider(segments):
    provider = MagicMock()
    provider.name = "whisper_local"
    provider.capabilities = ProviderCapabilities(
        sends_audio_offhost=False,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
    )
    provider.transcribe = AsyncMock(return_value=segments)
    return provider


@pytest.mark.asyncio
async def test_spec_addon_022_a_transcript_chunk_is_keyed_whisper(embedders):
    embedders.add_file("vid1", mime_type="video/mp4", file_path="/drives/home/vid1.mp4")
    words = "we deployed it on the cluster and it worked fine".split()
    segments = [
        TranscriptionSegment(
            text=" ".join(words), start=0.0, end=float(len(words)), language="en",
            words=[
                WordToken(text=w, start=float(i), end=float(i) + 0.9)
                for i, w in enumerate(words)
            ],
        )
    ]

    with (
        patch.object(whisper_worker, "get_provider", return_value=_provider(segments)),
        patch.object(whisper_worker, "validate_file_path", return_value=True),
        patch("app.policy_client.is_feature_enabled", new=AsyncMock(return_value=True)),
    ):
        await whisper_worker.index_whisper("vid1")

    assert "whisper" in embedders.assert_partition_matches_type("vid1")


@pytest.mark.asyncio
async def test_spec_addon_022_a_loft_sidecar_transcript_is_keyed_whisper(embedders, tmp_path):
    folder = tmp_path / "dir"
    folder.mkdir()
    loft = folder / "Talk.loft"
    loft.write_text("{}", encoding="utf-8")
    (folder / "Talk.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nthe transcript of the talk\n",
        encoding="utf-8",
    )
    embedders.add_file("loft1", mime_type=whisper_worker.LOFT_MIME, file_path=str(loft))

    with (
        patch.object(whisper_worker, "validate_file_path", return_value=True),
        patch("app.policy_client.is_feature_enabled", new=AsyncMock(return_value=True)),
    ):
        await whisper_worker.index_whisper("loft1")

    assert "whisper" in embedders.assert_partition_matches_type("loft1")


# ---------------------------------------------------------------------------
# refine.recompute_chunk_embeddings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spec_addon_022_a_refined_chunk_embedding_is_keyed_whisper(embedders):
    from app.workers import refine

    embedders.add_file("vid1", mime_type="video/mp4")
    with database.get_search_db() as s:
        chunks = [
            TranscriptChunk(
                file_id="vid1", chunk_index=i, text=body, language="en",
                timestamp_start=4.0 * i, timestamp_end=4.0 * (i + 1),
            )
            for i, body in enumerate(["We deployed it on Kubernetes.", "It worked fine."])
        ]
        s.add_all(chunks)
        s.flush()
        chunk_ids = [c.id for c in chunks]

    await refine.recompute_chunk_embeddings(chunk_ids)

    assert "whisper" in embedders.assert_partition_matches_type("vid1")


# ---------------------------------------------------------------------------
# vision._embed_and_store, video_visual._embed_scene
# ---------------------------------------------------------------------------


def test_spec_addon_022_a_vision_description_embedding_is_keyed_by_its_type(embedders):
    from app.workers import vision

    embedders.add_file("img1", mime_type="image/jpeg")

    vision._embed_and_store("img1", "A black cat sleeping on a sofa")

    embedders.assert_partition_matches_type("img1")


def test_spec_addon_022_a_scene_label_embedding_is_keyed_by_its_type(embedders):
    from app.workers import video_visual

    embedders.add_file("vid1", mime_type="video/mp4", duration=600.0)
    with database.get_search_db() as s:
        s.add(VideoVisualRun(
            id="vvr_1", file_id="vid1", status="running", is_active=False,
            requested_by="manual", priority=100, vision_model="llava:13b",
            pipeline_version=2, candidate_fingerprint="fp", selected_count=1,
        ))
        scene = VideoVisualScene(
            run_id="vvr_1", ordering=0, clip_embedding_id="c0", start_time=5.0,
            status="pending",
        )
        s.add(scene)
        s.flush()
        scene_id = scene.id

    video_visual._embed_scene("vid1", scene_id, "Chicken marinade added", "", 5.0, None)

    embedders.assert_partition_matches_type("vid1")


# ---------------------------------------------------------------------------
# clip._store_clip_embedding: clip_thumbnail and scene clip
# ---------------------------------------------------------------------------


@pytest.fixture()
def clip_ready(embedders, monkeypatch):
    fake_image = MagicMock()
    fake_image.convert.return_value = fake_image
    monkeypatch.setattr(clip_worker, "Image", MagicMock(open=lambda p: fake_image))
    monkeypatch.setattr(clip_worker, "validate_file_path", lambda p: True)
    monkeypatch.setattr(
        clip_worker, "_generate_blip_caption_if_needed", lambda *a, **k: None,
    )
    return embedders


def test_spec_addon_022_an_image_embedding_is_keyed_clip_thumbnail(clip_ready, tmp_path):
    image = tmp_path / "cat.jpg"
    image.write_bytes(b"\xff\xd8\xff" + b"\x00" * 64)
    clip_ready.add_file("img1", mime_type="image/jpeg", file_path=str(image))

    assert clip_worker._index_clip_sync("img1") is True

    assert clip_ready.assert_partition_matches_type("img1") == ["clip_thumbnail"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_spec_addon_022_a_scene_frame_embedding_is_keyed_clip(clip_ready, tmp_path):
    video = tmp_path / "cut.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=red:s=64x64:d=1:r=10",
            "-f", "lavfi", "-i", "color=blue:s=64x64:d=1:r=10",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]",
            "-map", "[v]", "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )
    clip_ready.add_file(
        "vid1", mime_type="video/mp4", file_path=str(video), duration=2.0,
    )

    assert clip_worker._index_clip_sync("vid1") is True

    assert "clip" in clip_ready.assert_partition_matches_type("vid1")


# ---------------------------------------------------------------------------
# _migrate_image_clip_to_clip_thumbnail
# ---------------------------------------------------------------------------


def test_spec_addon_022_both_migrations_in_one_start_key_the_rewritten_type(search_index):
    """The first start after the upgrade: the image's clip rows are rewritten to
    clip_thumbnail and the partition copy keys them by the rewritten type."""
    for table in VEC_TABLES:
        search_index.downgrade(table)
    dim = search_index.dim("vec_clip")
    search_index.add_file("img1", mime_type="image/jpeg")
    search_index.add_file("vid1", mime_type="video/mp4")
    img_vec, vid_vec = random_vector(dim, 1), random_vector(dim, 2)
    search_index.add_embedding("e_img1", "img1", "clip", "vec_clip")
    search_index.put_vector("vec_clip", "e_img1", img_vec)
    search_index.add_embedding("e_vid1", "vid1", "clip", "vec_clip")
    search_index.put_vector("vec_clip", "e_vid1", vid_vec)

    start()

    assert search_index.embeddings("img1") == [("e_img1", "clip_thumbnail", "vec_clip")]
    assert search_index.keyed("vec_clip") == {
        "e_img1": ("clip_thumbnail", img_vec.tobytes()),
        "e_vid1": ("clip", vid_vec.tobytes()),
    }


def test_spec_addon_022_a_rewrite_on_a_partitioned_table_rekeys_each_row(
    search_index, monkeypatch,
):
    monkeypatch.setattr(database, "_IMAGE_MIGRATION_CHUNK", 3)
    dim = search_index.dim("vec_clip")
    expected = {}
    for n in range(7):
        fid = f"img{n}"
        vec = random_vector(dim, 10 + n)
        search_index.add_file(fid, mime_type="image/jpeg")
        search_index.add_embedding(f"e_{fid}", fid, "clip", "vec_clip")
        search_index.put_vector("vec_clip", f"e_{fid}", vec, partition="clip")
        expected[f"e_{fid}"] = ("clip_thumbnail", vec.tobytes())
    search_index.add_file("img_novec", mime_type="image/jpeg")
    search_index.add_embedding("e_img_novec", "img_novec", "clip", "vec_clip")
    search_index.add_file("vid1", mime_type="video/mp4")
    for n in range(2):
        vec = random_vector(dim, 100 + n)
        search_index.add_embedding(f"e_vid1_{n}", "vid1", "clip", "vec_clip")
        search_index.put_vector("vec_clip", f"e_vid1_{n}", vec, partition="clip")
        expected[f"e_vid1_{n}"] = ("clip", vec.tobytes())

    database._migrate_image_clip_to_clip_thumbnail()

    assert search_index.embeddings("img_novec") == [
        ("e_img_novec", "clip_thumbnail", "vec_clip")
    ]
    assert search_index.keyed("vec_clip") == expected
    found = search_index.knn("vec_clip", random_vector(dim, 999), 50, "clip_thumbnail")
    assert sorted(found) == sorted(i for i, (t, _) in expected.items() if t == "clip_thumbnail")
