"""SPEC-ADDON-127: the aligner reads a three-letter language code as the two-letter code
of the same language (invariant I1)."""

from __future__ import annotations

import wave

import pytest

from app.workers import aligner
from tests.refine_align_rig import install_whisperx

JA_TEXT = "今日は晴れ"
JA_GLYPHS = list(JA_TEXT)


@pytest.fixture(autouse=True)
def _clean_aligner():
    aligner.release_all()
    yield
    aligner.release_all()


def _segment(language, text=JA_TEXT):
    try:
        return aligner.align_segment(
            waveform=object(), chunk_start=0.0, chunk_end=5.0, text=text, language=language
        )
    except Exception as exc:  # noqa: BLE001 - the failure is what is being asserted on
        pytest.fail(f"align_segment({language!r}) raised {exc!r}")


@pytest.mark.parametrize("code", ["ja", "jpn", "JPN", "Jpn", "ja-JP", "ja_JP", "JA-jp"])
def test_spec_addon_127_I1_japanese_code_loads_ja_model_and_aligns_per_character(
    monkeypatch, code
):
    fake = install_whisperx(monkeypatch)
    units = _segment(code)
    assert fake.loaded == ["ja"], f"{code!r} loaded {fake.loaded}"
    assert units is not None
    assert [u["text"] for u in units] == JA_GLYPHS
    assert all(u["timestamp_start"] < u["timestamp_end"] for u in units)


@pytest.mark.parametrize(
    ("code", "model"),
    [
        ("eng", "en"), ("EN", "en"), ("en-US", "en"), ("kor", "ko"),
        ("zho", "zh"), ("chi", "zh"), ("pt_BR", "pt"),
    ],
)
def test_spec_addon_127_I1_other_codes_load_the_two_letter_model(monkeypatch, code, model):
    fake = install_whisperx(monkeypatch)
    _segment(code, text="hello there world")
    assert fake.loaded == [model], f"{code!r} loaded {fake.loaded}"


def test_spec_addon_127_code_with_no_two_letter_equivalent_is_passed_through(monkeypatch):
    fake = install_whisperx(monkeypatch, models=("haw",))
    _segment("haw", text="aloha mai")
    assert fake.loaded == ["haw"]


@pytest.mark.parametrize("code", ["ja", "jpn", "JPN", "ja-JP"])
def test_spec_addon_127_I1_align_span_reads_japanese_codes_the_same_way(
    monkeypatch, tmp_path, code
):
    fake = install_whisperx(monkeypatch)
    audio = tmp_path / "talk.wav"
    with wave.open(str(audio), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * 16000 * 8)
    try:
        result = aligner.align_span(str(audio), 0.0, 5.0, JA_TEXT, code)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"align_span({code!r}) raised {exc!r}")
    assert fake.loaded == ["ja"], f"{code!r} loaded {fake.loaded}"
    status, units = result
    assert units, f"align_span({code!r}) returned {result!r}"
    assert [u["text"] for u in units] == JA_GLYPHS
