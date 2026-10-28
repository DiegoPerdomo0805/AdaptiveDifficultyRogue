#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"; [[ ! -f "$ROOT/experiment_config.json" ]] || REPO="$ROOT"
cd "$REPO"
EXP="${EXP:-experiments/20261021_main}"
echo "Experiment: $EXP   $(date -Is)"
for stage in 01_smoke 02_collect 03_train_q 04_train_cgan 05_evaluate 06_analyze; do
 if [[ -f "$EXP/$stage/DONE" ]]; then printf '[DONE] %s\n' "$stage"
 elif [[ -d "$EXP/$stage" ]]; then printf '[INCOMPLETE/FAILED] %s\n' "$stage"
 else printf '[PENDING] %s\n' "$stage"; fi
done
if [[ -f "$EXP/orchestration/pipeline.log" ]]; then echo; echo 'Last pipeline messages:'; tail -n 18 "$EXP/orchestration/pipeline.log"; fi
if [[ -d "$EXP/02_collect/blocks" ]]; then
 echo; echo 'Collection block files:'; find "$EXP/02_collect/blocks" -maxdepth 1 -type f -name 'block_*.csv' -printf '%f %s bytes\n' | sort
fi
