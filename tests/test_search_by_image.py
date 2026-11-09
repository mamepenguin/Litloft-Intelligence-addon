"""SPEC-ADDON-049: ``POST /search/by-image`` and ``GET /search/by-image/available``,
over HTTP, against a real search DB with sqlite-vec loaded and real Pillow.

The image embedding, the OCR engine and the text embedding are stand-ins; the
query vectors are ``unit(dim, 0)``, so a row placed at score ``s`` is at L2
distance ``sqrt(2 (1 - s))`` from the query and scores ``s`` back.
"""

from __future__ import annotations

import asyncio
import builtins
import importlib
import io
import json
import logging
import tempfile
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from sqlalchemy import text

import app.database as database
import app.image_query as image_query_module
from app.config import FeaturesConfig, SearchConfig
from app.models import Embedding, IndexedFile
from tests import ocr_fixtures
from tests.ocr_fixtures import pillow_real  # noqa: F401
from tests.search_endpoint_db import (  # noqa: F401
    DRIVE,
    OTHER,
    patch_everywhere,
    search_index,
    statements,
)
from tests.vec_partition_db import at_distance, unit

PATH = "/search/by-image"
AVAILABLE = "/search/by-image/available"
MANIFEST = Path(__file__).resolve().parents[1] / "manifest.json"
LIMIT_BYTES = 20 * 1024 * 1024
EPUB = "application/epub+zip"
SEVEN_TYPES = (
    "image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp",
    "image/heic", "image/heif",
)
SECRET_LINE = "ZQXJ-SECRET-READ-TEXT-0042"
SPEC_OCR_TIMEOUT_SECONDS = 10.0
OCR_TIMEOUT_SECONDS = 2.0
OCR_TIMEOUT_SCALE = OCR_TIMEOUT_SECONDS / SPEC_OCR_TIMEOUT_SECONDS


@pytest.fixture()
def short_ocr_timeout(monkeypatch):
    assert image_query_module.OCR_TIMEOUT_SECONDS == SPEC_OCR_TIMEOUT_SECONDS
    monkeypatch.setattr(
        image_query_module, "OCR_TIMEOUT_SECONDS", OCR_TIMEOUT_SECONDS,
    )


IMAGE_HIT_KEYS = {"file_id", "drive", "filename", "file_type", "score", "match_types", "file"}
SCENE_HIT_KEYS = {"file_id", "drive", "filename", "start", "end", "peak", "score", "frame_url", "file"}
DOCUMENT_HIT_KEYS = {"file_id", "drive", "filename", "file_type", "score", "passages", "file"}


def _distance(score: float) -> float:
    return float(np.sqrt(2.0 * (1.0 - score)))


def _ok(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------


class World:
    """Files, picture vectors, scene frames and document rows, written together."""

    def __init__(self, index):
        self.index = index
        self.clip_dim = index.dim("vec_clip")
        self.text_dim = index.dim("vec_text")
        self._files: dict[str, dict] = {}
        self._embeddings: list[Embedding] = []
        self._vectors: list[tuple[str, str, str, float]] = []
        self._n = 0

    def file(self, file_id, *, drive=DRIVE, active=True, mime="image/jpeg", duration=None):
        if file_id in self._files:
            return
        file_type = mime.split("/")[0]
        if file_type not in ("video", "image", "audio"):
            file_type = "document"
        ext = {"image": "jpg", "video": "mp4"}.get(file_type, "pdf")
        self._files[file_id] = dict(
            drive=drive, active=active, mime=mime, file_type=file_type,
            filename=f"{file_id}.{ext}", duration=duration,
        )

    def _vector(self, table, etype, score, **embedding_kw) -> str:
        self._n += 1
        emb_id = f"e{self._n}"
        self._embeddings.append(Embedding(
            id=emb_id, embedding_type=etype, vector_table=table, **embedding_kw,
        ))
        self._vectors.append((table, emb_id, etype, score))
        return emb_id

    def picture(self, file_id, score, **file_kw):
        self.file(file_id, **file_kw)
        self._vector("vec_clip", "clip_thumbnail", score, file_id=file_id, content_preview="x")

    def frame(self, file_id, t, score, **file_kw):
        file_kw.setdefault("mime", "video/mp4")
        self.file(file_id, **file_kw)
        self._vector(
            "vec_clip", "clip", score, file_id=file_id, timestamp_start=t, content_preview="x",
        )

    def chunk(self, file_id, score, *, preview="stored preview", page=None, chunk_index=0, **file_kw):
        file_kw.setdefault("mime", "application/pdf")
        self.file(file_id, **file_kw)
        self._vector(
            "vec_text", "text_content", score, file_id=file_id, content_preview=preview,
            page=page, chunk_index=chunk_index,
        )

    def section(self, file_id, page, title):
        self.commit()
        self.index.sql(
            "INSERT INTO document_sections(file_id, page, title) VALUES (:f, :p, :t)",
            f=file_id, p=page, t=title,
        )

    def commit(self):
        with database.get_search_db() as s:
            s.add_all([
                IndexedFile(
                    file_id=f, drive=v["drive"], filename=v["filename"],
                    file_path=f"/drives/{v['drive']}/{v['filename']}",
                    file_type=v["file_type"], mime_type=v["mime"], file_size=1,
                    duration=v["duration"], active=v["active"],
                )
                for f, v in self._files.items()
            ])
            s.add_all(self._embeddings)
        for table in ("vec_clip", "vec_text"):
            dim = self.index.dim(table)
            rows = [
                {
                    "i": e, "t": t,
                    "v": at_distance(dim, _distance(score), n + 1).astype(np.float32).tobytes(),
                }
                for n, (tb, e, t, score) in enumerate(self._vectors) if tb == table
            ]
            if rows:
                with self.index.engine.begin() as conn:
                    conn.execute(
                        text(
                            f"INSERT INTO {table}(embedding_id, embedding_type, vector) "
                            "VALUES (:i, :t, :v)"
                        ),
                        rows,
                    )
        self._files.clear()
        self._embeddings.clear()
        self._vectors.clear()


# ---------------------------------------------------------------------------
# The models and the app
# ---------------------------------------------------------------------------


class Models:
    """Stand-ins for the image embedding, the OCR engine and the text embedding."""

    def __init__(self, clip_dim: int, text_dim: int, ocr):
        self.clip_dim = clip_dim
        self.text_dim = text_dim
        self.ocr = ocr
        self.images: list = []
        self.image_error: BaseException | None = None
        self.on_image = None
        self.texts: list[list[str]] = []
        self.text_error: BaseException | None = None

    def embed_image(self, image, *_a, **_k):
        self.images.append(image)
        if self.on_image is not None:
            self.on_image(image)
        if self.image_error is not None:
            raise self.image_error
        return unit(self.clip_dim, 0)

    def embed_texts(self, texts, *_a, **_k):
        batch = [texts] if isinstance(texts, str) else list(texts)
        self.texts.append(batch)
        if self.text_error is not None:
            raise self.text_error
        return [unit(self.text_dim, 0) for _ in batch]

    def set_lines(self, *lines: str) -> None:
        self.ocr.engine.lines = tuple(lines)


@pytest.fixture()
def world(search_index):  # noqa: F811
    return World(search_index)


@pytest.fixture()
def configure(monkeypatch, make_settings):
    def _set(*, image_ocr="manual", floor=0.70, min_score_text=0.50, margin=0.20):
        settings = make_settings(
            search=SearchConfig(  # type: ignore[call-arg]
                min_score_image_query=floor,
                min_score_text=min_score_text,
                score_cutoff_margin=margin,
            ),
            features=FeaturesConfig(image_ocr=image_ocr),  # type: ignore[call-arg]
        )
        patch_everywhere(monkeypatch, "settings", settings)
        return settings

    return _set


@pytest.fixture()
def hydration(monkeypatch):
    calls: list[list[str]] = []

    async def _hydrate(ids):
        calls.append(list(ids))
        return {fid: {"id": fid} for fid in ids}

    patch_everywhere(monkeypatch, "hydrate_files", _hydrate)
    return calls


@pytest.fixture()
def models(world, monkeypatch, pillow_real, configure, hydration):  # noqa: F811
    import app.main  # noqa: F401

    harness = ocr_fixtures.install(monkeypatch, lines=())
    image_query = importlib.import_module("app.image_query")
    importlib.reload(image_query)
    m = Models(world.clip_dim, world.text_dim, harness)
    patch_everywhere(monkeypatch, "embed_image", m.embed_image)
    patch_everywhere(monkeypatch, "embed_texts", m.embed_texts)

    def _cached(*_a, **_k):
        raise AssertionError("an image query must not go through an LRU-cached embedding")

    patch_everywhere(monkeypatch, "embed_query", _cached)
    patch_everywhere(monkeypatch, "embed_text_clip", _cached)

    def _no_llm(*_a, **_k):
        raise AssertionError("an image query must not resolve an LLM")

    patch_everywhere(monkeypatch, "resolve", _no_llm)
    configure()
    return m


@pytest.fixture()
def client(models):
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def png_bytes(size=(40, 20), mode="RGB", color=(250, 250, 250)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, "PNG")
    return buffer.getvalue()


def post(client, body: bytes, content_type: str | None = "image/png", drive: str | None = DRIVE):
    headers = {}
    if content_type is not None:
        headers["Content-Type"] = content_type
    if drive is not None:
        headers["X-Lit-Drive"] = drive
    return client.post(PATH, content=body, headers=headers)


def ask(client, world, **kw):
    world.commit()
    return post(client, png_bytes(), **kw)


# ---------------------------------------------------------------------------
# Step 9 / 9a / 10: the route's limits and the decode
# ---------------------------------------------------------------------------


class TestLimits:
    @pytest.mark.parametrize("content_type", ["image/avif", "image/tiff", "text/plain", "application/octet-stream"])
    def test_spec_addon_049_i1_a_type_outside_the_seven_is_415(self, client, models, hydration, content_type):
        response = post(client, png_bytes(), content_type=content_type)

        assert response.status_code == 415
        assert models.images == []
        assert models.ocr.engine.seen == []
        assert hydration == []

    def test_spec_addon_049_i1_a_missing_content_type_is_415(self, client, models):
        response = client.post(PATH, content=png_bytes(), headers={"X-Lit-Drive": DRIVE})

        assert response.status_code == 415
        assert models.images == []

    @pytest.mark.parametrize("content_type", SEVEN_TYPES)
    def test_spec_addon_049_i1_each_of_the_seven_types_is_read_and_garbage_is_422(
        self, client, models, hydration, content_type,
    ):
        response = post(client, b"not an image at all", content_type=content_type)

        assert response.status_code == 422
        assert models.images == []
        assert models.ocr.engine.seen == []
        assert hydration == []

    def test_spec_addon_049_content_type_parameters_are_ignored(self, client, world, models):
        world.commit()

        response = post(client, png_bytes(), content_type="image/png; charset=binary")

        assert response.status_code == 200, response.text

    def test_spec_addon_049_i1_a_body_one_byte_over_the_limit_is_413(self, client, models, hydration):
        response = post(client, b"\0" * (LIMIT_BYTES + 1))

        assert response.status_code == 413
        assert models.images == []
        assert hydration == []

    def test_spec_addon_049_a_body_of_exactly_the_limit_is_read_not_refused(self, client, models):
        response = post(client, b"\0" * LIMIT_BYTES)

        assert response.status_code == 422

    def test_spec_addon_049_i1_an_image_over_50_megapixels_is_413(self, client, models, tmp_path, hydration):
        body = ocr_fixtures.huge_pixel_png(tmp_path / "huge.png", 10000, 5001).read_bytes()

        response = post(client, body)

        assert response.status_code == 413
        assert models.images == []
        assert hydration == []

    def test_spec_addon_049_an_image_of_exactly_50_megapixels_passes_the_pixel_check(
        self, client, models, tmp_path,
    ):
        # A header with no pixel data behind it: past the pixel check, the load fails.
        body = ocr_fixtures.huge_pixel_png(tmp_path / "edge.png", 10000, 5000).read_bytes()

        response = post(client, body)

        assert response.status_code == 422

    def test_spec_addon_049_a_missing_drive_header_is_400(self, client, models):
        response = post(client, png_bytes(), drive=None)

        assert response.status_code == 400
        assert models.images == []

    def test_spec_addon_049_9a_available_answers_200_with_an_empty_object(self, client, models):
        response = client.get(AVAILABLE, headers={"X-Lit-Drive": DRIVE})

        assert response.status_code == 200
        assert response.json() == {}
        assert models.images == []


class TestDecode:
    def test_spec_addon_049_i1_heic_decodes(self, client, world, models, tmp_path):
        world.commit()
        body = ocr_fixtures.heic_picture(tmp_path / "shot.heic").read_bytes()

        response = post(client, body, content_type="image/heic")

        assert response.status_code == 200, response.text
        (image,) = models.images
        assert image.size == (32, 16)
        assert image.mode == "RGB"

    def test_spec_addon_049_the_exif_orientation_is_applied(self, client, world, models, tmp_path):
        world.commit()
        body = ocr_fixtures.jpeg_with_exif_orientation_6(tmp_path / "turned.jpg").read_bytes()

        _ok(post(client, body, content_type="image/jpeg"))

        (image,) = models.images
        assert image.size == (20, 40)

    def test_spec_addon_049_an_animated_image_gives_its_first_frame(self, client, world, models, tmp_path):
        world.commit()
        body = ocr_fixtures.animated_gif(tmp_path / "anim.gif").read_bytes()

        _ok(post(client, body, content_type="image/gif"))

        (image,) = models.images
        assert image.mode == "RGB"
        red, green, blue = image.getpixel((0, 0))
        assert red > 150 and blue < 80

    def test_spec_addon_049_transparency_is_flattened_onto_white(self, client, world, models, tmp_path):
        world.commit()
        body = ocr_fixtures.rgba_picture(tmp_path / "alpha.png").read_bytes()

        _ok(post(client, body))

        (image,) = models.images
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (255, 255, 255)

    def test_spec_addon_049_ocr_reads_the_same_decoded_picture(self, client, world, models, tmp_path):
        world.commit()
        body = ocr_fixtures.jpeg_with_exif_orientation_6(tmp_path / "turned.jpg").read_bytes()

        _ok(post(client, body, content_type="image/jpeg"))

        (seen,) = models.ocr.engine.seen
        assert seen.size == (20, 40)
        assert seen.mode == "RGB"


# ---------------------------------------------------------------------------
# Step 12 / 13: the three groups
# ---------------------------------------------------------------------------


class TestImages:
    def test_spec_addon_049_i3_only_active_pictures_of_the_request_drive(self, client, world, models):
        world.picture("own", 0.90)
        world.picture("foreign", 0.95, drive=OTHER)
        world.picture("gone", 0.93, active=False)

        body = _ok(ask(client, world))

        assert [h["file_id"] for h in body["images"]] == ["own"]
        assert body["images"][0]["drive"] == DRIVE

    def test_spec_addon_049_i3_the_images_group_holds_only_pictures(self, client, world, models):
        world.picture("pic", 0.80)
        world.picture("vid", 0.95, mime="video/mp4")

        body = _ok(ask(client, world))

        assert [h["file_id"] for h in body["images"]] == ["pic"]

    def test_spec_addon_049_i4_the_floor_is_min_score_image_query(self, client, world, models, configure):
        configure(floor=0.70)
        world.picture("above", 0.7005)
        world.picture("below", 0.6995)

        body = _ok(ask(client, world))

        assert [h["file_id"] for h in body["images"]] == ["above"]

    def test_spec_addon_049_the_floor_follows_the_setting(self, client, world, models, configure):
        configure(floor=0.50)
        world.picture("a", 0.60)
        world.picture("b", 0.40)

        body = _ok(ask(client, world))

        assert [h["file_id"] for h in body["images"]] == ["a"]

    def test_spec_addon_049_i4_at_most_24_images_best_first(self, client, world, models):
        for i in range(30):
            world.picture(f"p{i:02d}", 0.99 - 0.001 * i)

        body = _ok(ask(client, world))

        assert [h["file_id"] for h in body["images"]] == [f"p{i:02d}" for i in range(24)]

    def test_spec_addon_049_i4_an_image_hit_is_the_raw_cosine_rounded_to_4_places(self, client, world, models):
        world.picture("a", 0.876543)

        (hit,) = _ok(ask(client, world))["images"]

        assert set(hit) == IMAGE_HIT_KEYS
        assert hit["score"] == pytest.approx(0.8765, abs=1e-4)
        assert hit["score"] == round(hit["score"], 4)
        assert hit["match_types"] == ["clip_thumbnail"]
        assert hit["filename"] == "a.jpg"
        assert hit["file_type"] == "image"
        assert hit["file"] == {"id": "a"}

    def test_spec_addon_049_scene_frames_are_not_pictures(self, client, world, models):
        world.frame("vid", 10.0, 0.95)

        body = _ok(ask(client, world))

        assert body["images"] == []


class TestScenes:
    def test_spec_addon_049_i3_scenes_come_only_from_the_request_drive(self, client, world, models):
        world.frame("mine", 10.0, 0.90)
        world.frame("theirs", 20.0, 0.95, drive=OTHER)

        body = _ok(ask(client, world))

        assert [h["file_id"] for h in body["scenes"]] == ["mine"]
        assert body["scenes"][0]["drive"] == DRIVE

    def test_spec_addon_049_a_scene_hit_has_its_frame_url_and_rounded_score(self, client, world, models):
        world.frame("vid", 12.7, 0.912345)

        body = _ok(ask(client, world))

        (hit,) = body["scenes"]
        assert set(hit) == SCENE_HIT_KEYS
        assert hit["peak"] == 12.7
        assert hit["start"] == 12.7
        assert hit["frame_url"] == "/api/addons/intelligence/files/vid/frame?t=12"
        assert hit["score"] == pytest.approx(0.9123, abs=1e-4)
        assert hit["score"] == round(hit["score"], 4)
        assert hit["file"] == {"id": "vid"}

    def test_spec_addon_049_i4_frames_below_the_floor_never_seed_a_scene(self, client, world, models):
        world.frame("kept", 10.0, 0.7005)
        world.frame("below", 10.0, 0.6995)

        body = _ok(ask(client, world))

        assert [h["file_id"] for h in body["scenes"]] == ["kept"]

    def test_spec_addon_049_i4_the_floor_is_applied_before_the_relative_threshold(self, client, world, models):
        for i in range(10):
            world.frame(f"hi{i}", 10.0, 0.90 - 0.001 * i)
        for i in range(300):
            world.frame(f"lo{i}", 10.0, 0.30)

        body = _ok(ask(client, world))

        assert [h["file_id"] for h in body["scenes"]] == [f"hi{i}" for i in range(10)]

    def test_spec_addon_049_i4_at_most_20_scene_segments(self, client, world, models):
        for i in range(40):
            world.frame(f"v{i:02d}", 10.0, 0.95)

        body = _ok(ask(client, world))

        assert len(body["scenes"]) == 20

    def test_spec_addon_049_a_drive_without_scene_frames_has_no_scenes_and_is_not_partial(
        self, client, world, models,
    ):
        for i in range(4096):
            world.frame(f"crowd{i}", 10.0, 0.99, drive=OTHER)

        body = _ok(ask(client, world))

        assert body["scenes"] == []
        assert body["scenes_partial"] is False

    def test_spec_addon_049_i4_partial_when_a_full_knn_ends_at_or_above_the_floor(self, client, world, models):
        for i in range(4096):
            world.frame(f"crowd{i}", 10.0, 0.99, drive=OTHER)
        world.frame("own", 10.0, 0.80)

        body = _ok(ask(client, world))

        assert body["scenes_partial"] is True

    def test_spec_addon_049_i4_not_partial_when_a_full_knn_ends_below_the_floor(self, client, world, models):
        world.frame("own", 10.0, 0.90)
        for i in range(4100):
            world.frame(f"crowd{i}", 10.0, 0.40, drive=OTHER)

        body = _ok(ask(client, world))

        assert body["scenes_partial"] is False
        assert [h["file_id"] for h in body["scenes"]] == ["own"]

    def test_spec_addon_049_i4_not_partial_when_the_knn_is_not_full(self, client, world, models):
        for i in range(100):
            world.frame(f"crowd{i}", 10.0, 0.99, drive=OTHER)
        world.frame("own", 10.0, 0.80)

        body = _ok(ask(client, world))

        assert body["scenes_partial"] is False


class TestTextPart:
    def test_spec_addon_049_i5_image_ocr_false_is_off_and_never_calls_the_engine(
        self, client, world, models, configure,
    ):
        configure(image_ocr="false")
        models.set_lines("a long enough line of text to pass the gate")
        world.chunk("doc", 0.95)

        body = _ok(ask(client, world))

        assert body["text"] == {"status": "off", "excerpt": None}
        assert body["documents"] == []
        assert models.ocr.engine.seen == []
        assert models.ocr.factory.calls == []
        assert models.texts == []

    @pytest.mark.parametrize("mode", ["manual", "on_index"])
    def test_spec_addon_049_manual_and_on_index_both_read(self, client, world, models, configure, mode):
        configure(image_ocr=mode)
        models.set_lines("x" * 25)

        body = _ok(ask(client, world))

        assert body["text"]["status"] == "read"

    @pytest.mark.parametrize(
        "lines, status",
        [
            pytest.param(("abcdefghij", "klmnopqrs"), "too_short", id="19"),
            pytest.param(("abcdefghij", "klmnopqrst"), "read", id="20"),
            pytest.param(("a b c d e f g h i j k l m n o p q r s",), "too_short", id="19-spaced"),
        ],
    )
    def test_spec_addon_049_i5_the_read_length_gate_counts_non_whitespace(
        self, client, world, models, lines, status,
    ):
        models.set_lines(*lines)

        body = _ok(ask(client, world))

        assert body["text"]["status"] == status
        if status == "too_short":
            assert models.texts == []
            assert body["documents"] == []

    def test_spec_addon_049_nothing_read_is_too_short_with_no_excerpt(self, client, world, models):
        models.set_lines()

        body = _ok(ask(client, world))

        assert body["text"] == {"status": "too_short", "excerpt": None}

    def test_spec_addon_049_the_excerpt_collapses_whitespace(self, client, world, models):
        models.set_lines("first  line\tof", "the 　 text")

        body = _ok(ask(client, world))

        assert body["text"]["excerpt"] == "first line of the text"

    @pytest.mark.parametrize(
        "length, expected_len, ellipsis",
        [(200, 200, False), (201, 201, True), (250, 201, True)],
    )
    def test_spec_addon_049_the_excerpt_is_cut_to_200_with_an_ellipsis(
        self, client, world, models, length, expected_len, ellipsis,
    ):
        models.set_lines("x" * length)

        excerpt = _ok(ask(client, world))["text"]["excerpt"]

        assert len(excerpt) == expected_len
        assert excerpt.endswith("…") is ellipsis
        assert excerpt.startswith("x" * 200)

    def test_spec_addon_049_the_query_text_is_the_first_1000_characters_joined_by_newlines(
        self, client, world, models,
    ):
        models.set_lines("a" * 600, "b" * 600)

        _ok(ask(client, world))

        (batch,) = models.texts
        (query,) = batch
        expected = ("a" * 600 + "\n" + "b" * 600)[:1000]
        assert query.endswith(expected)
        assert "b" * 400 not in query

    def test_spec_addon_049_no_engine_is_failed_and_the_visual_groups_still_answer(
        self, client, world, models, monkeypatch,
    ):
        patch_everywhere(monkeypatch, "_get_engine", lambda *_a, **_k: None)
        world.picture("pic", 0.90)
        world.frame("vid", 10.0, 0.90)

        body = _ok(ask(client, world))

        assert body["text"]["status"] == "failed"
        assert body["documents"] == []
        assert [h["file_id"] for h in body["images"]] == ["pic"]
        assert [h["file_id"] for h in body["scenes"]] == ["vid"]

    def test_spec_addon_049_i1_an_engine_that_raises_is_failed_and_the_visual_groups_still_answer(
        self, client, world, models,
    ):
        models.ocr.engine.error = RuntimeError("engine broke")
        world.picture("pic", 0.90)

        body = _ok(ask(client, world))

        assert body["text"]["status"] == "failed"
        assert body["documents"] == []
        assert [h["file_id"] for h in body["images"]] == ["pic"]

    def test_spec_addon_049_i5_a_text_embedding_failure_is_failed_and_changes_no_visual_group(
        self, client, world, models,
    ):
        models.set_lines("x" * 40)
        world.picture("pic", 0.90)
        world.frame("vid", 10.0, 0.88)
        world.chunk("doc", 0.95)
        world.commit()
        plain = _ok(post(client, png_bytes()))

        models.text_error = RuntimeError("text model failed")
        broken = _ok(post(client, png_bytes()))

        assert broken["text"]["status"] == "failed"
        assert broken["documents"] == []
        assert broken["images"] == plain["images"]
        assert broken["scenes"] == plain["scenes"]
        assert broken["scenes_partial"] == plain["scenes_partial"]

    def test_spec_addon_049_i1_a_documents_ranking_failure_is_failed_and_the_visual_groups_answer(
        self, client, world, models, search_index,  # noqa: F811
    ):
        models.set_lines("x" * 40)
        world.picture("pic", 0.90)
        world.commit()
        search_index.sql("DROP TABLE vec_text")

        body = _ok(post(client, png_bytes()))

        assert body["text"]["status"] == "failed"
        assert body["documents"] == []
        assert [h["file_id"] for h in body["images"]] == ["pic"]


class TestDocuments:
    @pytest.fixture(autouse=True)
    def _readable(self, models):
        models.set_lines("a screenshot of a page with enough text")

    def test_spec_addon_049_i3_documents_are_active_files_of_the_drive_and_never_pictures(
        self, client, world, models,
    ):
        world.chunk("own", 0.90)
        world.chunk("foreign", 0.95, drive=OTHER)
        world.chunk("gone", 0.94, active=False)
        world.chunk("pic", 0.96, mime="image/png")

        body = _ok(ask(client, world))

        assert [h["file_id"] for h in body["documents"]] == ["own"]
        assert body["documents"][0]["drive"] == DRIVE

    def test_spec_addon_049_the_candidate_fetch_widens_past_nearer_rows_of_other_drives(
        self, client, world, models,
    ):
        for i in range(300):
            world.chunk(f"crowd{i}", 0.99, drive=OTHER)
        world.chunk("own", 0.80)

        body = _ok(ask(client, world))

        assert [h["file_id"] for h in body["documents"]] == ["own"]

    def test_spec_addon_049_i4_rows_under_min_score_text_are_dropped(self, client, world, models, configure):
        configure(min_score_text=0.50, margin=1.0)
        world.chunk("kept", 0.55)
        world.chunk("below", 0.45)

        body = _ok(ask(client, world))

        assert [h["file_id"] for h in body["documents"]] == ["kept"]

    def test_spec_addon_049_i4_rows_more_than_the_margin_below_the_best_are_dropped(
        self, client, world, models, configure,
    ):
        configure(min_score_text=0.0, margin=0.10)
        world.chunk("best", 0.90)
        world.chunk("near", 0.81)
        world.chunk("far", 0.79)

        body = _ok(ask(client, world))

        assert [h["file_id"] for h in body["documents"]] == ["best", "near"]

    def test_spec_addon_049_i4_at_most_10_files_ordered_by_best_row_ties_by_file_id(
        self, client, world, models, configure,
    ):
        configure(min_score_text=0.0, margin=1.0)
        world.chunk("b", 0.95)
        world.chunk("a", 0.95)
        for i in range(12):
            world.chunk(f"f{i:02d}", 0.90 - 0.01 * i)

        body = _ok(ask(client, world))

        assert [h["file_id"] for h in body["documents"]] == [
            "a", "b", "f00", "f01", "f02", "f03", "f04", "f05", "f06", "f07",
        ]

    def test_spec_addon_049_i4_each_file_keeps_its_3_best_passages_ties_by_chunk_index(
        self, client, world, models, configure,
    ):
        configure(min_score_text=0.0, margin=1.0)
        world.chunk("doc", 0.80, chunk_index=4, preview="four")
        world.chunk("doc", 0.90, chunk_index=7, preview="seven")
        world.chunk("doc", 0.80, chunk_index=2, preview="two")
        world.chunk("doc", 0.70, chunk_index=1, preview="one")

        (hit,) = _ok(ask(client, world))["documents"]

        assert set(hit) == DOCUMENT_HIT_KEYS
        assert [p["text"] for p in hit["passages"]] == ["seven", "two", "four"]

    def test_spec_addon_049_a_page_row_is_a_page_passage_built_from_the_stored_row(
        self, client, world, models,
    ):
        world.chunk("pdf", 0.912345, page=7, chunk_index=3, preview="the stored excerpt")

        (hit,) = _ok(ask(client, world))["documents"]

        (passage,) = hit["passages"]
        assert passage["kind"] == "page"
        assert passage["page"] == 7
        assert passage["text"] == "the stored excerpt"
        assert passage["keyword"] is False
        assert passage["highlight_terms"] == []
        assert passage["frame_url"] is None
        assert passage["score"] == pytest.approx(0.9123, abs=1e-4)
        assert passage["score"] == round(passage["score"], 4)
        assert hit["filename"] == "pdf.pdf"
        assert hit["file_type"] == "document"
        assert hit["file"] == {"id": "pdf"}

    def test_spec_addon_049_an_epub_row_is_a_section_passage(self, client, world, models):
        world.chunk("book", 0.90, page=3, mime=EPUB, preview="chapter text")
        world.section("book", 3, "The Warehouse")

        (hit,) = _ok(ask(client, world))["documents"]

        (passage,) = hit["passages"]
        assert passage["kind"] == "section"
        assert passage["section"] == 3
        assert passage["section_title"] == "The Warehouse"

    def test_spec_addon_049_a_row_without_a_page_is_a_text_passage(self, client, world, models):
        world.chunk("note", 0.90, page=None, mime="text/markdown")

        (hit,) = _ok(ask(client, world))["documents"]

        (passage,) = hit["passages"]
        assert passage["kind"] == "text"
        assert passage["page"] is None


class TestAnswer:
    def test_spec_addon_049_the_answer_has_exactly_the_five_keys(self, client, world, models):
        body = _ok(ask(client, world))

        assert set(body) == {"images", "scenes", "scenes_partial", "documents", "text"}
        assert set(body["text"]) == {"status", "excerpt"}

    def test_spec_addon_049_one_hydration_call_covers_every_id_of_the_three_groups(
        self, client, world, models, hydration,
    ):
        models.set_lines("a screenshot of a page with enough text")
        world.picture("pic", 0.90)
        world.frame("vid", 10.0, 0.90)
        world.chunk("doc", 0.90)

        _ok(ask(client, world))

        (call,) = hydration
        assert sorted(set(call)) == ["doc", "pic", "vid"]

    def test_spec_addon_049_a_failed_hydration_leaves_file_null(self, client, world, models, monkeypatch):
        async def _broken(_ids):
            raise httpx.ConnectError("core down")

        patch_everywhere(monkeypatch, "hydrate_files", _broken)
        world.picture("pic", 0.90)

        body = _ok(ask(client, world))

        (hit,) = body["images"]
        assert hit["file"] is None
        assert hit["filename"] == "pic.jpg"


# ---------------------------------------------------------------------------
# Step 12.1: the image embedding failing
# ---------------------------------------------------------------------------


def test_spec_addon_049_i1_an_embedding_failure_is_503_and_runs_nothing_else(client, world, models, hydration):
    models.set_lines("x" * 40)
    models.image_error = RuntimeError("model failed to load")
    world.picture("pic", 0.90)

    response = ask(client, world)

    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding unavailable"
    assert models.ocr.engine.seen == []
    assert models.texts == []
    assert hydration == []


def test_spec_addon_049_i1_a_visual_ranking_failure_is_500_search_failed(
    client, world, models, search_index,  # noqa: F811
):
    world.picture("pic", 0.90)
    world.commit()
    search_index.sql("DROP TABLE vec_clip")

    response = post(client, png_bytes())

    assert response.status_code == 500
    assert response.json()["detail"] == "Search failed"


# ---------------------------------------------------------------------------
# I2: nothing kept
# ---------------------------------------------------------------------------


class TestNothingKept:
    def test_spec_addon_049_i2_a_request_writes_nothing_to_the_search_db(
        self, client, world, models, statements,  # noqa: F811
    ):
        models.set_lines("a screenshot of a page with enough text")
        world.picture("pic", 0.90)
        world.frame("vid", 10.0, 0.90)
        world.chunk("doc", 0.90)
        world.commit()
        statements.reset()

        _ok(post(client, png_bytes()))

        assert statements.statements
        assert statements.writes() == []

    def test_spec_addon_049_i2_a_request_creates_no_file(self, client, world, models, monkeypatch):
        world.commit()
        created: list[str] = []
        real_open = builtins.open

        def _open(file, mode="r", *a, **k):
            if any(flag in mode for flag in "wax+"):
                created.append(str(file))
            return real_open(file, mode, *a, **k)

        def _temp(name):
            def _refuse(*_a, **_k):
                created.append(name)
                raise AssertionError(f"{name} called")
            return _refuse

        monkeypatch.setattr(builtins, "open", _open)
        for name in ("TemporaryFile", "NamedTemporaryFile", "SpooledTemporaryFile", "mkstemp", "mkdtemp"):
            monkeypatch.setattr(tempfile, name, _temp(name))

        _ok(post(client, png_bytes()))

        assert created == []

    def test_spec_addon_049_i2_the_read_text_is_never_logged(self, client, world, models, caplog):
        models.set_lines(SECRET_LINE, "and enough other text to read")
        world.chunk("doc", 0.90)
        world.commit()
        caplog.set_level(logging.DEBUG)

        _ok(post(client, png_bytes()))

        for record in caplog.records:
            assert SECRET_LINE not in record.getMessage()

    def test_spec_addon_049_i2_the_read_text_never_reaches_an_lru_cached_embedding(
        self, client, world, models,
    ):
        models.set_lines(SECRET_LINE, "and enough other text to read")
        world.chunk("doc", 0.90)

        body = _ok(ask(client, world))

        assert body["text"]["status"] == "read"
        assert models.texts


# ---------------------------------------------------------------------------
# Step 11 / I12: one at a time, and disconnects
# ---------------------------------------------------------------------------


async def test_spec_addon_049_i12_a_second_query_waits_for_the_first_and_a_bad_type_does_not(
    world, models,
):
    from app.main import app

    world.picture("pic", 0.90)
    world.commit()
    release = threading.Event()
    state = {"running": 0, "max": 0}

    def _hold(_image):
        state["running"] += 1
        state["max"] = max(state["max"], state["running"])
        try:
            if len(models.images) == 1:
                release.wait(10)
        finally:
            state["running"] -= 1

    models.on_image = _hold
    headers = {"X-Lit-Drive": DRIVE, "Content-Type": "image/png"}
    body = png_bytes()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
        first = asyncio.create_task(http.post(PATH, content=body, headers=headers))
        for _ in range(200):
            if models.images:
                break
            await asyncio.sleep(0.01)
        assert len(models.images) == 1

        second = asyncio.create_task(http.post(PATH, content=body, headers=headers))
        refused = await asyncio.wait_for(
            http.post(PATH, content=body, headers={**headers, "Content-Type": "text/plain"}),
            timeout=5,
        )
        await asyncio.sleep(0.3)
        assert refused.status_code == 415
        assert len(models.images) == 1
        assert not second.done()

        release.set()
        responses = await asyncio.gather(first, second)

    assert [r.status_code for r in responses] == [200, 200]
    assert len(models.images) == 2
    assert state["max"] == 1


async def test_spec_addon_049_i12_a_request_whose_client_is_gone_runs_nothing(models, caplog):
    from app.main import app

    caplog.set_level(logging.INFO)
    sent: list[dict] = []

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": PATH, "raw_path": PATH.encode(),
        "root_path": "", "query_string": b"", "server": ("t", 80), "client": ("c", 1),
        "headers": [
            (b"host", b"t"), (b"content-type", b"image/png"),
            (b"x-lit-drive", DRIVE.encode()), (b"content-length", b"100"),
        ],
    }

    await asyncio.wait_for(app(scope, receive, send), timeout=10)

    assert models.images == []
    assert models.ocr.engine.seen == []
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert sum("image query skipped: client gone" in m for m in messages) == 1


# ---------------------------------------------------------------------------
# Step 12 text part: the OCR lock
# ---------------------------------------------------------------------------


def test_spec_addon_049_run_engine_lets_one_caller_at_a_time_use_the_engine(models):
    from PIL import Image

    ocr = models.ocr.module
    engine = models.ocr.engine
    engine.on_call = lambda _image: time.sleep(0.05)
    image = Image.new("RGB", (8, 8), (255, 255, 255))
    threads = [threading.Thread(target=ocr.run_engine, args=(engine, image, None)) for _ in range(4)]

    for t in threads:
        t.start()
    for t in threads:
        t.join(5)

    assert len(engine.seen) == 4
    assert engine.max_concurrent == 1


def test_the_ocr_timeout_is_the_spec_value():
    assert image_query_module.OCR_TIMEOUT_SECONDS == SPEC_OCR_TIMEOUT_SECONDS


def test_spec_addon_049_i12_the_route_gives_up_on_a_busy_engine_after_10s(
    client, world, models, short_ocr_timeout,
):
    from PIL import Image

    ocr = models.ocr.module
    engine = models.ocr.engine
    models.set_lines("x" * 40)
    world.picture("pic", 0.90)
    world.commit()
    release = threading.Event()
    holding = threading.Event()

    def _indexing(_image):
        if not holding.is_set():
            holding.set()
            release.wait(30 * OCR_TIMEOUT_SCALE)

    engine.on_call = _indexing
    indexer = threading.Thread(
        target=ocr.run_engine, args=(engine, Image.new("RGB", (8, 8)), None),
    )
    indexer.start()
    assert holding.wait(5 * OCR_TIMEOUT_SCALE)
    try:
        started = time.monotonic()
        body = _ok(post(client, png_bytes()))
        elapsed = time.monotonic() - started
    finally:
        release.set()
        indexer.join(5 * OCR_TIMEOUT_SCALE)

    assert body["text"]["status"] == "failed"
    assert [h["file_id"] for h in body["images"]] == ["pic"]
    assert 9.5 * OCR_TIMEOUT_SCALE <= elapsed < 20 * OCR_TIMEOUT_SCALE


# ---------------------------------------------------------------------------
# Settings defaults and the manifest
# ---------------------------------------------------------------------------


def test_spec_addon_049_the_new_settings_default_to_070_and_600():
    from app.config import MemoryConfig

    assert SearchConfig().min_score_image_query == pytest.approx(0.70)  # type: ignore[attr-defined]
    assert MemoryConfig().clip_image_idle_unload == 600  # type: ignore[attr-defined]


def _routes(path: str) -> list[dict]:
    routes = json.loads(MANIFEST.read_text(encoding="utf-8"))["proxy"]["routes"]
    return [r for r in routes if r["path"] == path]


def test_spec_addon_049_i7_the_manifest_gives_both_routes_the_same_search_gate():
    (available,) = _routes(AVAILABLE)
    (search,) = _routes(PATH)

    assert search["methods"] == ["POST"]
    assert available["methods"] == ["GET"]
    assert search["pre_check"] == {"type": "addon_feature", "feature": "search"}
    assert available["pre_check"] == search["pre_check"]
    assert search["response_filter"] == {
        "type": "current_drive_only_nested",
        "paths": {"images": "drive", "scenes": "drive", "documents": "drive"},
    }
    assert search["timeout"] == 28
    assert available.get("response_filter") is None
