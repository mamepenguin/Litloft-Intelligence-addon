"""Transcription provider comparison eval harness (Phase 2C).

Dev-time tool that runs every available transcription provider over
the same curated audio dataset and emits a markdown + JSON report
comparing WER / CER / latency / sa-WER. Mirrors the structure of
``app/evals/`` (Ask) and ``app/evals_citations/`` (citation) — pure
dev-time, ground-truth-based, no LLM-as-judge, no CI gate.

Spec: ``2026-05-08-transcription-providers-phase-2c.md``.
"""
