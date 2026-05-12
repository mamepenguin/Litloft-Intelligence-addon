#!/usr/bin/env bash
# Phase 1.D A/B run: same eval cases, agentic vs legacy.
#
# Produces two md+json reports under evals/agentic_phase1/ and a third
# "comparison" md that diffs them. The comparison report is what the
# operator reads to make the Phase 1.D Go/No-Go call.
#
# Prerequisites:
#   - snapshot DB at ../evals/test-drive/snapshot/search.db (or pass --snapshot)
#   - LLM provider configured in search-config.yml (provider, base_url,
#     model, API key). Cloud model (e.g. gpt-4o) or local (e.g. qwen2.5:14b).
#   - The active model must be on llm.agentic_models so the agentic
#     branch activates.
#
# Usage:
#   scripts/run_agentic_ab.sh <label>  [extra args forwarded to app.evals]
# Example:
#   scripts/run_agentic_ab.sh qwen2_5_14b --filter 010
set -euo pipefail

LABEL="${1:-unlabeled}"
shift || true

OUT_DIR="evals/agentic_phase1"
mkdir -p "$OUT_DIR"

TS=$(date -u +%Y%m%d_%H%M%S)
LEGACY="$OUT_DIR/${LABEL}_${TS}_legacy.md"
AGENTIC="$OUT_DIR/${LABEL}_${TS}_agentic.md"

echo "==> Running legacy (force_legacy_rag=True)"
python -m app.evals \
  --label "${LABEL}_legacy" \
  --output "$LEGACY" \
  --force-legacy-rag \
  "$@"

echo "==> Running agentic (default branching: agentic if model is allow-listed)"
python -m app.evals \
  --label "${LABEL}_agentic" \
  --output "$AGENTIC" \
  --baseline "${LEGACY%.md}.json" \
  "$@"

echo
echo "==> Reports:"
echo "    legacy : $LEGACY (+ .json)"
echo "    agentic: $AGENTIC (+ .json, with pair-comparison appended)"
echo
echo "Go/No-Go criteria (see evals/agentic_phase1/README.md):"
echo "  - answer quality (agentic) >= legacy"
echo "  - citation correctness (agentic) >= legacy"
echo "  - tool_call_count median <= 5, 95p <= 10"
echo "Both LLMs (local + cloud) must clear all 3 to mark Phase 1 done."
