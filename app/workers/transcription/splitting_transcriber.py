"""Decorator that splits cap-exceeding inputs and forwards chunks.

Phase 2B: when a cloud provider's ``max_input_bytes`` is exceeded,
:class:`SplittingTranscriber` runs ``ffmpeg``-based normalisation +
silence detection (see :mod:`app.workers.transcription.splitter`) to
slice the file into chunks small enough for the inner provider, then
calls the inner provider once per chunk and stitches the results back
together with timestamp offsets.

Spec: ``2026-05-08-transcription-providers-phase-2b.md``.

Lifecycle / retry semantics:

* The wrapper advertises ``handles_own_retry=True`` so the dispatch
  layer (``whisper.py``) skips its outer ``transcribe_with_retry``
  wrap. Per-chunk retry is performed inside this class via a fresh
  ``transcribe_with_retry`` call against the inner provider, which
  preserves successfully transcribed chunks 0..N-1 when chunk N hits
  a transient error (R1 spec H-R1-2).
* That capability is unconditional, so it also covers the calls this
  wrapper does **not** split. Every path out of ``transcribe`` must
  therefore go through ``transcribe_with_retry`` itself — delegating
  raw would silently drop retry and circuit-breaker gating for
  normal-sized inputs.
* The inner provider's ``name`` is exposed verbatim on the wrapper so
  ``JobRecord.provider`` and WS event payloads do not see a
  transparent intermediary.
* Cleanup of the temporary working directory (normalised FLAC + each
  chunk file) is centralized in the ``finally`` block; the splitter
  returns the tmpdir alongside the chunk list for that purpose.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from collections.abc import Callable
from dataclasses import replace

from app.workers.transcription.base import (
    ProviderCapabilities,
    TranscriptionProvider,
    TranscriptionSegment,
)
from app.workers.transcription.errors import FatalError
from app.workers.transcription.splitter import (
    offset_segment,
    segments_tail_text,
    split_audio,
)

# Tail length passed as ``initial_prompt`` to the next chunk. Stays
# well inside Whisper's 224-token decoder context for CJK inputs
# (≈ 1 token/char) while preserving enough context for vocabulary
# continuity. See ``segments_tail_text`` docstring.
PRIOR_TAIL_MAX_CHARS = 150


class SplittingTranscriber:
    """Wrap a TranscriptionProvider to split files exceeding its cap."""

    def __init__(
        self,
        inner: TranscriptionProvider,
        *,
        cap_bytes: int | None = None,
    ) -> None:
        """Wrap ``inner``, splitting anything larger than ``cap_bytes``.

        ``cap_bytes`` defaults to the inner provider's own
        ``max_input_bytes``. The factory passes an explicit value so
        the split threshold can also account for how much of the file
        we are willing to hold in memory, which is a stricter limit
        than what the remote API accepts (see
        :data:`app.workers.transcription.MAX_INPUT_MEMORY_BYTES`).
        """
        self._inner = inner
        self._cap_bytes = (
            cap_bytes
            if cap_bytes is not None
            else inner.capabilities.max_input_bytes
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        # Override ``handles_own_retry`` so the dispatch layer in
        # ``whisper.py`` skips its outer ``transcribe_with_retry``
        # wrap (R1 spec H-R1-2). Without this flip, per-chunk
        # failures double-count on the inner provider's circuit
        # breaker and a transient error on chunk N would re-run the
        # entire splitter from scratch.
        return replace(self._inner.capabilities, handles_own_retry=True)

    @property
    def name(self) -> str:
        # ``JobRecord.provider`` and WS event payloads see the inner
        # provider's name. The wrapper is structurally invisible to
        # downstream observability.
        return self._inner.name

    async def transcribe(
        self,
        file_path: str,
        *,
        language_hint: str | None = None,
        hotwords: list[str] | None = None,
        initial_prompt: str | None = None,
        progress: Callable[[float], None] | None = None,
    ) -> list[TranscriptionSegment]:
        cap = self._cap_bytes
        if cap is None:
            # No cap → always pass through. The wrapper would not
            # have been built in this case (the factory always
            # resolves a cap), but be defensive anyway — and still
            # run retry, for the reason given on the ``size <= cap``
            # branch below.
            return await self._delegate_with_retry(
                file_path,
                language_hint=language_hint,
                hotwords=hotwords,
                initial_prompt=initial_prompt,
                progress=progress,
            )

        try:
            with open(file_path, "rb") as f:
                size = os.fstat(f.fileno()).st_size
        except OSError as exc:
            raise FatalError(
                f"Cannot stat audio file {file_path}: {exc}"
            ) from exc

        if size <= cap:
            # File already fits — no splitting needed, but retry is
            # still ours to run: we advertise
            # ``handles_own_retry=True`` unconditionally, so the
            # dispatch layer skips its outer ``transcribe_with_retry``
            # for every call that reaches us, split or not. Delegating
            # raw here would leave normal-sized inputs with no retry
            # and no circuit breaker at all.
            return await self._delegate_with_retry(
                file_path,
                language_hint=language_hint,
                hotwords=hotwords,
                initial_prompt=initial_prompt,
                progress=progress,
            )

        # Lazy import: ``retry`` already imports from
        # ``transcription.base`` which imports our package. Module-
        # level import would create a circular dep.
        from app.workers.transcription.retry import transcribe_with_retry

        chunks, tmpdir = await split_audio(file_path, cap)
        try:
            segments_all: list[TranscriptionSegment] = []
            prior_tail = initial_prompt or ""
            total = len(chunks)
            for i, chunk in enumerate(chunks):
                ip = (
                    prior_tail
                    if self._inner.capabilities.accepts_initial_prompt
                    and prior_tail
                    else None
                )

                def chunk_progress(p: float, _i: int = i) -> None:
                    if progress is not None:
                        progress(
                            (_i + max(0.0, min(1.0, p))) / total
                        )

                # Per-chunk retry: a transient failure on chunk N
                # retries chunk N only, preserving chunks 0..N-1
                # already-emitted segments. The dispatch layer's
                # outer retry is skipped because our
                # ``handles_own_retry`` capability is True.
                chunk_segments = await transcribe_with_retry(
                    self._inner,
                    chunk.path,
                    language_hint=language_hint,
                    hotwords=hotwords,
                    initial_prompt=ip,
                    progress=chunk_progress,
                )
                for seg in chunk_segments:
                    segments_all.append(
                        offset_segment(seg, chunk.start_offset_s)
                    )
                if self._inner.capabilities.accepts_initial_prompt:
                    prior_tail = segments_tail_text(
                        chunk_segments, max_chars=PRIOR_TAIL_MAX_CHARS
                    )
            return segments_all
        finally:
            with contextlib.suppress(OSError):
                shutil.rmtree(tmpdir, ignore_errors=True)

    async def _delegate_with_retry(
        self,
        file_path: str,
        *,
        language_hint: str | None,
        hotwords: list[str] | None,
        initial_prompt: str | None,
        progress: Callable[[float], None] | None,
    ) -> list[TranscriptionSegment]:
        """Forward one whole file to the inner provider, with retry.

        Used by both non-splitting paths. There is no double-wrap
        risk: the dispatch layer keys off ``handles_own_retry``, which
        this wrapper always reports as True, so it never applies a
        retry of its own to anything we handle.
        """
        # Lazy import for the same circular-dependency reason as the
        # splitting path.
        from app.workers.transcription.retry import transcribe_with_retry

        return await transcribe_with_retry(
            self._inner,
            file_path,
            language_hint=language_hint,
            hotwords=hotwords,
            initial_prompt=initial_prompt,
            progress=progress,
        )
