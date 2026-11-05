"""SPEC-ADDON-062 I14: a transcription worker opens a file whose on-disk name is
NFD through the path reconcile stored, where the unresolved NFC path fails with
"Cannot stat audio file"."""

from __future__ import annotations

import unicodedata

import pytest

from app.workers.transcription.base import (
    ProviderCapabilities,
    TranscriptionSegment,
)
from app.workers.transcription.errors import FatalError
from app.workers.transcription.splitting_transcriber import SplittingTranscriber
from tests.idf_term_db import idx  # noqa: F401
from tests.nfd_fs import make_file, require_distinguishing_filesystem
from tests.test_idf_term_marks import WORDS, _manager, litloft  # noqa: F401

DRIVE = "a"
REL = "がぎ/ぱぴ.mp4"
NFD_REL = "nfd:がぎ/nfd:ぱぴ.mp4"


class OpeningProvider:
    """Reads the file it is handed, as every real provider does."""

    name = "fake"
    capabilities = ProviderCapabilities(
        sends_audio_offhost=False,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=False,
        max_input_bytes=None,
        accepts_initial_prompt=False,
        handles_own_retry=False,
    )

    async def transcribe(self, file_path, **kwargs):
        with open(file_path, "rb") as f:
            f.read()
        return [TranscriptionSegment(text="ok", start=0.0, end=1.0, language="ja", words=[])]


@pytest.mark.asyncio
async def test_spec_addon_062_a_worker_opens_the_file_through_the_reconciled_path(
    idx, litloft, tmp_path, monkeypatch,  # noqa: F811
):
    root = tmp_path / "mount"
    root.mkdir()
    require_distinguishing_filesystem(root)
    from app import config

    monkeypatch.setitem(config.settings.drive_mounts, DRIVE, str(root))
    monkeypatch.setattr("app.indexer.resolve_file_path", config.resolve_file_path)
    make_file(root, NFD_REL)
    nfc_path = str(root / unicodedata.normalize("NFC", REL))
    idx.add_file("k1", drive=DRIVE, filename="ぱぴ.mp4", transcript=WORDS, indexed=True, gen=3)
    idx.sql(f"UPDATE indexed_files SET file_path = '{nfc_path}' WHERE file_id = 'k1'")
    idx.sql(
        "UPDATE indexed_files SET whisper_indexed = 1, clip_indexed = 1, "
        "metadata_indexed = 1, text_indexed = 1 WHERE file_id = 'k1'"
    )
    litloft("k1", drive=DRIVE, file_path=REL, filename="ぱぴ.mp4")
    transcriber = SplittingTranscriber(OpeningProvider(), cap_bytes=10_000)

    with pytest.raises(FatalError, match="Cannot stat audio file"):
        await transcriber.transcribe(nfc_path)

    await _manager().reconcile()
    stored = idx.rows("SELECT file_path FROM indexed_files WHERE file_id = 'k1'")[0][0]

    assert idx.rows(
        "SELECT whisper_indexed, clip_indexed, metadata_indexed, text_indexed "
        "FROM indexed_files WHERE file_id = 'k1'"
    ) == [(1, 1, 1, 1)]

    segments = await transcriber.transcribe(stored)
    assert [s.text for s in segments] == ["ok"]


def test_spec_addon_062_a_repeat_resolve_reads_the_disk_again(tmp_path):
    from app import diskpath

    require_distinguishing_filesystem(tmp_path)
    first = make_file(tmp_path, "nfd:ぱぴ.mp4")
    rel = unicodedata.normalize("NFC", "ぱぴ.mp4")
    assert diskpath.resolve_on_disk(tmp_path, rel) == first

    first.unlink()
    second = make_file(tmp_path, "nfc:ぱぴ.mp4")

    assert diskpath.resolve_on_disk(tmp_path, rel) == second


def test_spec_addon_062_a_listing_error_resolves_to_the_stored_path_without_raising(
    tmp_path, monkeypatch,
):
    import os

    from app import diskpath

    make_file(tmp_path, "nfd:ぱぴ.mp4")

    def denied(*args, **kwargs):
        raise PermissionError("unreadable")

    monkeypatch.setattr(os, "scandir", denied)
    rel = unicodedata.normalize("NFC", "ぱぴ.mp4")

    assert diskpath.resolve_on_disk(tmp_path, rel) == tmp_path / rel
