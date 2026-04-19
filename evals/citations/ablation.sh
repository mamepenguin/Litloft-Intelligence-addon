#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Citation pipeline ablation study.
#
# Runs the citation eval harness against the 5 curated cases under 4 config
# permutations, each disabling one mechanism at a time. The goal is to
# quantify how much each pipeline layer contributes to the baseline's 100%
# top-1 accuracy.
#
# Each configuration restarts the intelligence container so
# ``SummariesConfig`` is re-read from ``search-config.yml``. The original
# config is restored at the end via ``trap``.
#
# Usage:
#   ./evals/citations/ablation.sh
#
# Output:
#   evals/citations/reports/ablation_<timestamp>/
#     ├── baseline.md/json          (full pipeline — sanity)
#     ├── no_dp.md/json             (citation_section_alignment_enabled: false)
#     ├── no_anchor.md/json         (citation_section_anchor_enabled: false)
#     ├── no_hybrid.md/json         (citation_hybrid_enabled: false)
#     └── summary.md                (delta table across configs)
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CONFIG="${REPO_ROOT}/addons/intelligence/search-config.yml"
CONFIG_BAK="${CONFIG}.ablation-bak"
REPORTS_HOST="${REPO_ROOT}/addons/intelligence/evals/citations/reports"
TS="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="ablation_${TS}"
OUT_DIR_HOST="${REPORTS_HOST}/${OUT_DIR}"
OUT_DIR_CONTAINER="/eval-data/citations/reports/${OUT_DIR}"

mkdir -p "${OUT_DIR_HOST}"

# Snapshot original config so the script is always reversible.
cp "${CONFIG}" "${CONFIG_BAK}"
restore_config() {
  mv "${CONFIG_BAK}" "${CONFIG}"
  docker compose restart intelligence >/dev/null
  echo "  [restored original search-config.yml + restarted intelligence]"
}
trap restore_config EXIT

run_config() {
  local label="$1"
  local overrides="$2"

  # Rewrite config = original + optional summaries block
  cp "${CONFIG_BAK}" "${CONFIG}"
  if [[ -n "${overrides}" ]]; then
    printf '\n# --- ablation override (temporary) ---\nsummaries:\n%s\n' "${overrides}" >> "${CONFIG}"
  fi
  docker compose restart intelligence >/dev/null

  # Wait for intelligence readiness — DB init can take ~2-3s on restart.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if docker compose exec -T intelligence python -c 'from app.database import init_search_db; init_search_db()' 2>/dev/null; then
      break
    fi
    sleep 1
  done

  docker compose exec -T intelligence python -m app.evals_citations \
    --cases /eval-data/citations/cases/ \
    --snapshot /eval-data/citations/snapshot/search.db \
    --drive 動画 \
    --label "${label}" \
    --output "${OUT_DIR_CONTAINER}/${label}.md" >/dev/null

  # Extract headline metrics for summary.
  python3 -c "
import json, sys
with open('${OUT_DIR_HOST}/${label}.json') as f:
    j = json.load(f)
a = j['aggregate']
print(f\"  ${label:15s}  top1={a['top1_accuracy']:.1%}  r@3={a['recall_at_3']:.1%}  has_cit={a['has_citation_precision']:.1%}  miss_req={a['missing_required_citations']}\")"
}

echo "ablation run starting — output dir: ${OUT_DIR_HOST}"

run_config "baseline"    ""
run_config "no_dp"       "  citation_section_alignment_enabled: false"
run_config "no_anchor"   "  citation_section_anchor_enabled: false"
run_config "no_hybrid"   "  citation_hybrid_enabled: false"

# Build summary table.
python3 - <<'PY'
import json, os, glob
OUT = os.environ.get("OUT_DIR_HOST")
rows = []
for p in sorted(glob.glob(f"{OUT}/*.json")):
    name = os.path.basename(p)[:-5]
    with open(p) as f:
        j = json.load(f)
    a = j["aggregate"]
    rows.append((name, a["top1_accuracy"], a["recall_at_3"], a["has_citation_precision"], a["missing_required_citations"]))

lines = ["# Citation Pipeline Ablation", "", f"{len(rows)} configurations over the same 5 curated cases.", "", "| config | top-1 | recall@3 | has_cit prec | miss req |", "|---|---:|---:|---:|---:|"]
for n, t, r, h, m in rows:
    lines.append(f"| {n} | {t:.1%} | {r:.1%} | {h:.1%} | {m} |")
lines.append("")
lines.append("Deltas (vs baseline):")
lines.append("")
base = rows[0]
lines.append("| config | Δ top-1 | Δ recall@3 |")
lines.append("|---|---:|---:|")
for n, t, r, h, m in rows[1:]:
    lines.append(f"| {n} | {(t-base[1])*100:+.1f} pp | {(r-base[2])*100:+.1f} pp |")

with open(f"{OUT}/summary.md", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nsummary: {OUT}/summary.md")
print("\n" + "\n".join(lines))
PY

export OUT_DIR_HOST
