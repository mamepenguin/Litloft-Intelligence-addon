# Transcription eval cases

Place audio fixtures and ground-truth YAML files in this directory to
run the comparison harness:

```
docker compose exec intelligence \
  python -m app.evals_transcription
```

## Layout

```
cases/
├── audio/                       # audio fixtures (gitignored)
│   ├── short_ja_news.wav
│   └── medium_en_lecture.mp3
├── short_ja_news.yml
├── medium_en_lecture.yml
└── sample.yml.example           # template (committed)
```

`*.yml` files are loaded; `*.yml.example` is skipped. Audio fixtures
must live under `audio/` — `audio_path` in the YAML is interpreted
relative to this `cases/` directory.

## Privacy & licensing

- **PII**: Anything you put in `audio/` and the matching reference
  transcript ends up in the eval report. Do not commit recordings of
  identifiable individuals or copyrighted content. The directory is
  in `.gitignore` for that reason.
- **Licensing**: Use audio you own or that has an explicit
  redistributable license (CC-BY, public domain, your own
  recordings). Public datasets like FLEURS / Common Voice are
  acceptable but typically not aligned with Litloft's long-form use
  case (see Phase 2C spec).
- **Whisper-bias caveat**: If your reference transcript was authored
  by Whisper itself (e.g., copied from Litloft's existing `.vtt`),
  every Whisper-family provider in the report — `whisper_local`,
  `openai_compatible` — will look unfairly good. Hand-curate the GT
  whenever you can; the report annotates Whisper-family rows with a
  `†` so future-you remembers.

## Tiers

Cases are tagged with `tier: short | medium | long` so the report can
break out per-tier averages:

- **short** ≤ 30 s — useful for latency-sensitive comparisons
- **medium** 30 s – 5 min — covers most Litloft podcast / talk content
- **long** > 5 min — exercises Phase 2B chunked transcription

A handful of cases per tier (5–10 each) gets the report out of the
"single segment can flip the column" noise floor — see the
`Brtx1wHg42EHWklTwBPm1` hako entry for the empirical N≥25 rule
established by the citation eval.

## Diarization

Set `speakers:` to evaluate provider diarization quality. Speaker IDs
are arbitrary strings; the harness solves the assignment between
provider-supplied IDs (`"0"` / `"speaker_0"` / `"A"`) and your GT
labels by majority vote, so you don't need to pre-align them.

## Split-test

For a long case, set `split_test:` to compare a single-shot run
against a forced split. The harness runs the case **twice** for each
listed provider (once at full size, once with the lower cap forcing
`SplittingTranscriber` to fire) so it can attribute any quality
delta to the chunk boundaries.

Use this on at most one long case per run — each `split_test`
provider doubles the API spend on that case.
