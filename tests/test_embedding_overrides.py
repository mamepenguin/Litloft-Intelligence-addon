"""Unit tests for the embedding overrides reader/writer + config merge.

RED-phase tests for Phase 1 of spec
``2026-05-20-gui-text-embedding-model.md``.

``app/embedding_overrides.py`` is a new module mirroring
``app/llm_overrides.py`` exactly in shape (same ``_overrides_io``
mechanics: atomic write, idempotent delete, schema_version gate).
The single GUI-editable field is ``text_embedding``, which is
validated against the ``_MODEL_DIMS`` allowlist in
``app/workers/embedder.py`` — anything not in that dict is dropped
with a WARN so a typo silently falling back to a 384-dim baseline
(the real Ask-breaking incident, hako ``JxHJMk2V5bu603gr1HkAZ``)
is structurally impossible.

Phase 1 also wires ``embedding_overrides.merge_into_dict`` into the
``models`` parse path in ``app/config.py`` so a written override is
reflected in a freshly re-parsed ``settings.models.text_embedding``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.embedding_overrides import (  # noqa: E402 — Phase 1 module (RED)
    SCHEMA_VERSION,
    EmbeddingOverrides,
    delete_overrides,
    merge_into_dict,
    overrides_path,
    read_overrides,
    write_overrides,
)

# A model id that exists in app.workers.embedder._MODEL_DIMS.
_VALID_MODEL = "cl-nagoya/ruri-v3-30m"
# A second valid id (different dimension) used for merge assertions.
_VALID_MODEL_ALT = "ibm-granite/granite-embedding-311m-multilingual-r2"
# Baseline model id (the shipped default in search-config.yml.example).
_BASELINE_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
# A model id that is NOT in the allowlist (typo / unsupported).
_INVALID_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# 1. write -> read round-trip; delete is idempotent
# ---------------------------------------------------------------------------


def test_read_returns_none_when_file_absent(tmp_path: Path) -> None:
    assert read_overrides(tmp_path) is None


def test_write_then_read_roundtrips_text_embedding(tmp_path: Path) -> None:
    write_overrides(
        EmbeddingOverrides(text_embedding=_VALID_MODEL),
        data_dir=tmp_path,
        updated_at="2026-05-20T00:00:00Z",
    )
    out = read_overrides(tmp_path)
    assert out is not None
    assert out.text_embedding == _VALID_MODEL


def test_write_emits_atomic_rename_target(tmp_path: Path) -> None:
    """No leftover ``*.tmp`` and the target is the canonical name."""
    write_overrides(
        EmbeddingOverrides(text_embedding=_VALID_MODEL),
        data_dir=tmp_path,
    )
    target = overrides_path(tmp_path)
    assert target.is_file()
    assert target.name == "embedding-overrides.json"
    assert not (tmp_path / "embedding-overrides.json.tmp").exists()


def test_write_omits_absent_field_so_baseline_is_preserved(
    tmp_path: Path,
) -> None:
    write_overrides(EmbeddingOverrides(), data_dir=tmp_path)
    raw = json.loads(overrides_path(tmp_path).read_text(encoding="utf-8"))
    assert "text_embedding" not in raw
    assert raw["schema_version"] == SCHEMA_VERSION


def test_delete_returns_false_when_absent(tmp_path: Path) -> None:
    assert delete_overrides(tmp_path) is False


def test_delete_is_idempotent_can_call_twice(tmp_path: Path) -> None:
    write_overrides(
        EmbeddingOverrides(text_embedding=_VALID_MODEL), data_dir=tmp_path
    )
    assert delete_overrides(tmp_path) is True
    # Second call must not raise and must report nothing removed.
    assert delete_overrides(tmp_path) is False
    assert not overrides_path(tmp_path).exists()


# ---------------------------------------------------------------------------
# 2/3. allowlist validation (invariant §2.1-4)
# ---------------------------------------------------------------------------


def test_allowlist_drops_model_not_in_model_dims(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A model id absent from ``_MODEL_DIMS`` is dropped on read with a
    WARN; the typed field falls back to ``None`` (= use baseline)."""
    overrides_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    overrides_path(tmp_path).write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "text_embedding": _INVALID_MODEL,
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        out = read_overrides(tmp_path)
    assert out is not None
    assert out.text_embedding is None
    assert any(
        _INVALID_MODEL in rec.getMessage() for rec in caplog.records
    ), "expected a WARN naming the dropped model id"


def test_allowlist_preserves_valid_model_dims_key(tmp_path: Path) -> None:
    overrides_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    overrides_path(tmp_path).write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "text_embedding": _VALID_MODEL,
            }
        ),
        encoding="utf-8",
    )
    out = read_overrides(tmp_path)
    assert out is not None
    assert out.text_embedding == _VALID_MODEL


def test_allowlist_matches_embedder_model_dims_exactly() -> None:
    """The allowlist must BE the embedder's ``_MODEL_DIMS`` keys, not a
    hand-maintained copy that can drift."""
    from app.workers.embedder import _MODEL_DIMS

    for model_id in _MODEL_DIMS:
        ov = EmbeddingOverrides(text_embedding=model_id)
        assert ov.text_embedding == model_id


def test_read_drops_non_string_text_embedding(tmp_path: Path) -> None:
    """Defence in depth: a corrupt writer cannot smuggle a list."""
    overrides_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    overrides_path(tmp_path).write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "text_embedding": ["not", "a", "string"],
            }
        ),
        encoding="utf-8",
    )
    out = read_overrides(tmp_path)
    assert out is not None
    assert out.text_embedding is None


# ---------------------------------------------------------------------------
# 4. schema_version mismatch -> ignored
# ---------------------------------------------------------------------------


def test_read_ignores_unknown_schema_version(tmp_path: Path) -> None:
    overrides_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    overrides_path(tmp_path).write_text(
        json.dumps(
            {"schema_version": 99, "text_embedding": _VALID_MODEL}
        ),
        encoding="utf-8",
    )
    assert read_overrides(tmp_path) is None


def test_read_returns_none_for_malformed_json(tmp_path: Path) -> None:
    overrides_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    overrides_path(tmp_path).write_text("not json", encoding="utf-8")
    assert read_overrides(tmp_path) is None


# ---------------------------------------------------------------------------
# 5. merge_into_dict semantics
# ---------------------------------------------------------------------------


def test_merge_into_dict_preserves_baseline_when_overrides_absent() -> None:
    base = {"text_embedding": _BASELINE_MODEL, "clip": "x"}
    out = merge_into_dict(base, None)
    assert out == base
    assert out is not base  # new dict, no mutation


def test_merge_into_dict_override_replaces_baseline_value() -> None:
    base = {"text_embedding": _BASELINE_MODEL, "clip": "x"}
    out = merge_into_dict(
        base, EmbeddingOverrides(text_embedding=_VALID_MODEL_ALT)
    )
    assert out["text_embedding"] == _VALID_MODEL_ALT
    # Sibling keys in the models section are untouched.
    assert out["clip"] == "x"
    # Baseline dict not mutated.
    assert base["text_embedding"] == _BASELINE_MODEL


def test_merge_into_dict_none_field_leaves_baseline() -> None:
    base = {"text_embedding": _BASELINE_MODEL}
    out = merge_into_dict(base, EmbeddingOverrides(text_embedding=None))
    assert out["text_embedding"] == _BASELINE_MODEL


# ---------------------------------------------------------------------------
# 6. config integration: written override flows into settings.models
# ---------------------------------------------------------------------------


def test_config_load_settings_reflects_written_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After ``write_overrides`` the next ``load_settings()`` re-parse
    must surface the override at ``settings.models.text_embedding``,
    overriding whatever ``search-config.yml`` declared."""
    import app.config as cfg

    data_dir = tmp_path / "intelligence-data"
    yml = tmp_path / "search-config.yml"
    yml.write_text(
        "models:\n"
        f"  text_embedding: {_BASELINE_MODEL}\n"
        "  clip: keep-this-clip\n"
    )
    monkeypatch.setenv("SEARCH_CONFIG_PATH", str(yml))
    monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))

    write_overrides(
        EmbeddingOverrides(text_embedding=_VALID_MODEL_ALT),
        data_dir=data_dir,
    )

    settings = cfg.load_settings()
    assert settings.models.text_embedding == _VALID_MODEL_ALT
    # Non-overridden models key still comes from the YAML baseline.
    assert settings.models.clip == "keep-this-clip"


def test_config_load_settings_keeps_yaml_when_override_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An allowlist-failing override is dropped, so the YAML baseline
    (not a silent 384-dim fallback) remains in effect."""
    import app.config as cfg

    data_dir = tmp_path / "intelligence-data"
    yml = tmp_path / "search-config.yml"
    yml.write_text(
        "models:\n"
        f"  text_embedding: {_VALID_MODEL_ALT}\n"
    )
    monkeypatch.setenv("SEARCH_CONFIG_PATH", str(yml))
    monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))

    data_dir.mkdir(parents=True, exist_ok=True)
    overrides_path(data_dir).write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "text_embedding": _INVALID_MODEL,
            }
        ),
        encoding="utf-8",
    )

    settings = cfg.load_settings()
    assert settings.models.text_embedding == _VALID_MODEL_ALT
