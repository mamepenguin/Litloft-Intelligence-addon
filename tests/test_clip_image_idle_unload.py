"""SPEC-ADDON-049 I6: the SigLIP full model's idle release and its in-flight guard."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

import app.workers.clip as clip


def _settings(timeout: int = 600) -> SimpleNamespace:
    return SimpleNamespace(
        models=SimpleNamespace(clip="google/siglip2-base-patch16-256"),
        memory=SimpleNamespace(clip_image_idle_unload=timeout),
    )


@pytest.fixture()
def loaded(monkeypatch):
    monkeypatch.setattr(clip, "_is_siglip_model", lambda _name: True)
    monkeypatch.setattr(clip, "_siglip_full_model", object())
    monkeypatch.setattr(clip, "_siglip_full_loaded", True)
    monkeypatch.setattr(clip, "_siglip_image_calls", 0)
    monkeypatch.setattr(clip, "_siglip_full_last_used", time.monotonic())
    monkeypatch.setattr(clip, "settings", _settings())
    return monkeypatch


def test_spec_addon_049_i6_releases_the_vision_tower_after_the_idle_timeout(loaded):
    loaded.setattr(clip, "_siglip_full_last_used", time.monotonic() - 601)

    clip.check_idle_unload()

    assert clip._siglip_full_model is None
    assert clip._siglip_full_loaded is False


def test_spec_addon_049_i6_keeps_the_model_inside_its_idle_window(loaded):
    clip.check_idle_unload()

    assert clip._siglip_full_model is not None
    assert clip._siglip_full_loaded is True


def test_spec_addon_049_i6_zero_means_never_release(loaded):
    loaded.setattr(clip, "settings", _settings(timeout=0))
    loaded.setattr(clip, "_siglip_full_last_used", time.monotonic() - 100_000)

    clip.check_idle_unload()

    assert clip._siglip_full_model is not None


def test_spec_addon_049_i6_never_releases_while_an_embedding_runs(loaded):
    loaded.setattr(clip, "_siglip_full_last_used", time.monotonic() - 601)
    seen: list[bool] = []

    def running(_image):
        clip.check_idle_unload()
        seen.append(clip._siglip_full_model is not None)
        return np.ones(4, dtype=np.float32)

    loaded.setattr(clip, "_embed_image_siglip", running)

    clip.embed_image(object())

    assert seen == [True]
    assert clip._siglip_full_model is not None
    assert clip._siglip_image_calls == 0


def test_spec_addon_049_i6_the_release_leaves_the_text_encoder_resident(loaded):
    text_encoder = object()
    loaded.setattr(clip, "_siglip_text_model", text_encoder)
    loaded.setattr(clip, "_siglip_text_loaded", True)
    loaded.setattr(clip, "_siglip_full_last_used", time.monotonic() - 601)

    clip.check_idle_unload()

    assert clip._siglip_full_model is None
    assert clip._siglip_text_model is text_encoder
    assert clip._siglip_text_loaded is True


def test_spec_addon_049_i6_an_embedding_after_a_release_answers_the_same_vector(loaded):
    vector = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    loaded.setattr(clip, "_embed_image_siglip", lambda _image: vector)
    first = clip.embed_image(object())

    loaded.setattr(clip, "_siglip_full_last_used", time.monotonic() - 601)
    clip.check_idle_unload()
    second = clip.embed_image(object())

    assert np.array_equal(first, second)
