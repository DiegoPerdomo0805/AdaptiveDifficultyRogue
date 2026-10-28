#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"; [[ ! -f "$ROOT/experiment_config.json" ]] || REPO="$ROOT"
cd "$REPO"
EXP="${EXP:-experiments/20261021_main}"
[[ -d "$EXP" ]] || { echo "Experiment not found: $EXP" >&2; exit 1; }
DEST="${DEST:-$HOME/thesis_backups}"
mkdir -p "$DEST"
NAME="$(basename "$EXP")_$(date +%Y%m%d_%H%M%S)"
tar -czf "$DEST/$NAME.tar.gz" "$EXP"
sha256sum "$DEST/$NAME.tar.gz" | tee "$DEST/$NAME.tar.gz.sha256"
echo "Saved: $DEST/$NAME.tar.gz"
