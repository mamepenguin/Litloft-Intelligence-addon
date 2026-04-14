# test-drive / raw/

Raw source files for the Ask eval harness live here. **This directory is
not tracked by git** (see `.gitignore` — everything except this README
and `.gitignore` is ignored). Only the derived index
(`../snapshot/search.db` + `manifest.json`) is committed, because that
is what the runner actually consults during evaluation.

## Why raw files are not committed

- Binary media (videos, audio, images) would bloat the repo.
- Some files may not be freely redistributable.
- The authoritative ground truth for retrieval / recall metrics is the
  **snapshot `search.db`**, not the raw files. Whisper / CLIP / BLIP
  model changes would otherwise cause recall numbers to drift.

So the contract is: **re-indexing from `raw/` is only required when you
want to regenerate the snapshot** (e.g. after upgrading an embedding
model). Day-to-day eval runs read exclusively from `../snapshot/`.

## Security notice

Files placed here will be sent to the configured LLM provider when the
eval harness runs (transcripts, captions, and document text are included
in Ask prompts). Do not drop anything confidential into `raw/`. For
privacy-sensitive evaluation, configure a local LLM (ollama, vLLM, LM
Studio) in `search-config.yml`.

## How to populate

See `docs/superpowers/specs/2026-04-14-intelligence-ask-eval-harness.md`
("テストドライブの運用") for the intended workflow:

1. Drop sample files under `raw/` matching the paths referenced by
   `../cases/*.yml`.
2. Register `eval-drive` in `drives.json` (temporarily `readonly: false`).
3. Let a full index run complete (Whisper + CLIP + optional BLIP).
4. Run `./addons/intelligence/scripts/snapshot.sh <out_dir>` to freeze
   the resulting `search.db` into `../snapshot/`.
5. Commit `../snapshot/` (raw/ stays ignored).
