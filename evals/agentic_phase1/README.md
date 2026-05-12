# Phase 1.D: agentic Ask A/B evaluation

Outputs of the `force_legacy_rag=true` vs default (agentic) eval runs live here.
The folder is the operator's working space — every run is one tuple of
`<label>_<timestamp>_{legacy,agentic}.md` + `.json` sidecars, with the
agentic md also carrying a `## Pair comparison vs ...` section that compares
itself against the legacy run.

## How to run

```sh
cd addons/intelligence
scripts/run_agentic_ab.sh <label> [--filter NNN]
```

Prerequisites:

- A snapshot DB at `evals/test-drive/snapshot/search.db` (or pass `--snapshot`).
- `search-config.yml` pointing at the LLM you want to evaluate; that model
  must appear in `llm.agentic_models` so the agentic branch activates.
- For ollama-native local LLMs the agentic branch is intentionally **not**
  available (the native client does not implement `chat_with_tools`). Use the
  OpenAI-compatible `/v1` endpoint instead (e.g. `http://localhost:11434/v1`).

Repeat the run twice with different `search-config.yml` profiles:

1. **Local LLM** (e.g. `qwen2.5:14b`) via `/v1` against ollama.
2. **Cloud LLM** (e.g. `gpt-4o`) via OpenAI / compatible vendor.

## Go/No-Go criteria

All three must hold **on both LLMs** for Phase 1 to ship `agentic_mode: auto`:

| # | Criterion | Where to read |
|---|---|---|
| 1 | `must_mention_coverage` (agentic) ≥ legacy (within epsilon) | per-case summary + Pair comparison `Stage 3: must_mention` row |
| 2 | `citation_in_ground_truth` (agentic) ≥ legacy | same table, `citation_in_ground_truth` row |
| 3 | `tool_call_count` median ≤ 5 **and** 95p ≤ 10 | Aggregate section, `Agentic: tool_call_count` rows |

Secondary watch-axes (not gating but record in hako):

- `route_correctness` — when ≥ 0.7, the loop is choosing tools sanely.
- `max_context_tokens_used` 95p < `context_window × 0.7` — budget headroom intact.
- `forced_answer_rate` < 0.1 — most answers come from clean LLM termination.

## On failure

If any criterion fails on either LLM:

1. Inspect the failing case's stage3.runs[].agentic in the JSON sidecar.
   - High `tool_call_count` with low `route_correctness` → prompt issue.
   - High `forced_answer_rate` → tool returns are too verbose; tune the
     per-call cap in `app/rag/tools/get_file_chunks.py`.
   - Low `citation_in_ground_truth` with low `route_correctness` → the LLM
     is not following the expected tool order; either widen
     `expected_tool_sequence` or adjust the system prompt.
2. Iterate. Re-run.
3. After three honest iterations without convergence, freeze:
   set `llm.agentic_mode: off` in `search-config.yml.example`, ship the
   code anyway, and record the observed deltas in hako so the next
   improvement cycle has a baseline.
