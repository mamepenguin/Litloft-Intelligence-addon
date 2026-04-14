#!/usr/bin/env bash
# Create an eval snapshot of the current intelligence search.db.
#
# Intended to run inside the `intelligence` container:
#   docker compose exec intelligence bash ./scripts/snapshot.sh <out_dir>
#
# <out_dir> is the directory that will receive search.db + manifest.json
# (e.g. /app/backend/../evals/test-drive/snapshot or an absolute path).
# The directory is created if missing.
#
# Produces:
#   <out_dir>/search.db       sqlite backup of the live search.db
#   <out_dir>/manifest.json   schema_version 1 snapshot metadata
#
# Spec: docs/superpowers/specs/2026-04-14-intelligence-ask-eval-harness.md
# Phase: A

set -euo pipefail

if [[ "${1:-}" == "" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<USAGE
Usage: snapshot.sh <out_dir>

Dumps the live search.db to <out_dir>/search.db and writes
<out_dir>/manifest.json describing the snapshot.
USAGE
  exit 1
fi

OUT_DIR="$1"
mkdir -p "$OUT_DIR"

# Resolve live DB and config paths with the same defaults as app/config.py.
SRC_DB="${INTELLIGENCE_SEARCH_DB_PATH_SRC:-${INTELLIGENCE_DATA_DIR:-/intelligence-data}/search.db}"
CONFIG_PATH="${SEARCH_CONFIG_PATH:-/app/search-config.yml}"

# Resolve the drive root. If DRIVE_MOUNTS is set (name=path,...) use the
# first entry; else fall back to the single-drive default /drives/default.
DRIVE_ROOT_DEFAULT="/drives/default"
if [[ -n "${DRIVE_MOUNTS:-}" ]]; then
  DRIVE_ROOT_DEFAULT="$(echo "$DRIVE_MOUNTS" | cut -d',' -f1 | cut -d'=' -f2- | tr -d '[:space:]')"
fi
DRIVE_ROOT="${EVAL_DRIVE_ROOT:-$DRIVE_ROOT_DEFAULT}"

if [[ ! -f "$SRC_DB" ]]; then
  echo "snapshot.sh: source search.db not found at $SRC_DB" >&2
  exit 2
fi

OUT_DB="$OUT_DIR/search.db"
OUT_MANIFEST="$OUT_DIR/manifest.json"

echo "snapshot.sh: dumping $SRC_DB -> $OUT_DB"
# Online backup via Python's sqlite3 module. Uses the SQLite backup API so
# it is safe even while the live intelligence process is writing. We use
# Python instead of the sqlite3 CLI because the intelligence container
# image does not ship the CLI binary (only the Python bindings).
python3 - "$SRC_DB" "$OUT_DB" <<'BACKUP_EOF'
import sqlite3
import sys

src_path, dst_path = sys.argv[1], sys.argv[2]
src = sqlite3.connect(src_path)
try:
    dst = sqlite3.connect(dst_path)
    try:
        with dst:
            src.backup(dst)
    finally:
        dst.close()
finally:
    src.close()
BACKUP_EOF

echo "snapshot.sh: writing $OUT_MANIFEST"
python3 - "$OUT_DB" "$OUT_MANIFEST" "$CONFIG_PATH" "$DRIVE_ROOT" <<'PYEOF'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

out_db = Path(sys.argv[1])
out_manifest = Path(sys.argv[2])
config_path = Path(sys.argv[3])
drive_root = Path(sys.argv[4])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model_names(config_path: Path) -> dict[str, str]:
    """Parse whisper/clip/blip from search-config.yml.

    Uses a tiny line-based parser to avoid adding PyYAML as a dependency
    just for this script — the container already has it, but shelling
    out to python with no deps keeps the script portable.
    """
    names = {"whisper": "", "clip": "", "blip": ""}
    if not config_path.is_file():
        return names
    in_models = False
    for raw in config_path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            in_models = line.strip().startswith("models:")
            continue
        if not in_models:
            continue
        stripped = line.strip()
        for key in names:
            prefix = f"{key}:"
            if stripped.startswith(prefix):
                value = stripped[len(prefix):].strip()
                # Strip inline comment + surrounding quotes.
                if "#" in value:
                    value = value.split("#", 1)[0].strip()
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                names[key] = value
    return names


def scan_drive(root: Path) -> dict:
    layout = {
        "root": str(root),
        "file_count": 0,
        "total_bytes": 0,
    }
    if not root.is_dir():
        return layout
    file_count = 0
    total_bytes = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                total_bytes += fp.stat().st_size
                file_count += 1
            except OSError:
                continue
    layout["file_count"] = file_count
    layout["total_bytes"] = total_bytes
    return layout


manifest = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "search_db_sha256": sha256_file(out_db),
    "indexed_with": load_model_names(config_path),
    "drive_layout": scan_drive(drive_root),
}
out_manifest.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(f"snapshot.sh: manifest={json.dumps(manifest, ensure_ascii=False)}")
PYEOF

echo "snapshot.sh: done"
