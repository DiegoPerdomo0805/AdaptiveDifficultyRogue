#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
if [[ -f "$ROOT/experiment_config.json" ]]; then REPO="$ROOT"; fi
cd "$REPO"
EXP="${EXP:-experiments/20261021_main}"
[[ "$EXP" != /* && "$EXP" != *'..'* && "$EXP" == experiments/* ]] || { echo '[STOP] EXP must be a safe relative experiments/... path' >&2; exit 1; }
RUNS="${RUNS:-9000}"; WORKERS="${WORKERS:-6}"; Q_SEEDS="${Q_SEEDS:-0 1 2}"; SEEDS="${SEEDS:-0 1 2}"; BATCHES="${BATCHES:-64 128}"; EVAL_RUNS="${EVAL_RUNS:-400}"; DEVICE="${DEVICE:-cuda}"
mkdir -p "$EXP" "$EXP/orchestration"
exec > >(tee -a "$EXP/orchestration/pipeline.log") 2>&1
trap 'rc=$?; echo "[STOP] $(date -Is) exit=$rc; inspect $EXP/orchestration/pipeline.log and stage log; partial stage is NEVER auto-overwritten" >&2' ERR
printf '[START] %s\n' "$(date -Is)"
printf 'EXP=%s RUNS=%s WORKERS=%s Q_SEEDS=%s SEEDS=%s BATCHES=%s EVAL_RUNS=%s DEVICE=%s\n' "$EXP" "$RUNS" "$WORKERS" "$Q_SEEDS" "$SEEDS" "$BATCHES" "$EVAL_RUNS" "$DEVICE"
"$ROOT/00_preflight.sh"
# Stage versions/commit provenance. Avoid modifying tracked configuration while running.
if [[ ! -e "$EXP/orchestration/run_identity.txt" ]]; then
{
  echo "date=$(date -Is)"; echo "commit=$(git rev-parse HEAD)"; echo "branch=$(git branch --show-current)"
  echo "python=$(python --version 2>&1)"; echo "runs=$RUNS"; echo "workers=$WORKERS"
  echo "q_seeds=$Q_SEEDS"; echo "cgan_seeds=$SEEDS"; echo "batches=$BATCHES"; echo "eval_runs=$EVAL_RUNS"
} > "$EXP/orchestration/run_identity.txt"
else
  current_commit="$(git rev-parse HEAD)"
  previous_commit="$(sed -n 's/^commit=//p' "$EXP/orchestration/run_identity.txt" | head -n1)"
  [[ "$current_commit" == "$previous_commit" ]] || { echo "[STOP] Commit differs from recorded experiment identity" >&2; exit 4; }
  for setting in "runs:$RUNS" "workers:$WORKERS" "q_seeds:$Q_SEEDS" "cgan_seeds:$SEEDS" "batches:$BATCHES" "eval_runs:$EVAL_RUNS"; do
    key="${setting%%:*}"; value="${setting#*:}"
    previous="$(sed -n "s/^${key}=//p" "$EXP/orchestration/run_identity.txt" | head -n1)"
    [[ "$value" == "$previous" ]] || { echo "[STOP] Resume mismatch $key: $previous != $value" >&2; exit 4; }
  done
  echo '[INFO] Resume identity matches existing experiment.'
fi
# Stage design owned by repository itself: DONE markers indicate validated completion.
# An incomplete directory intentionally causes the repo wrappers to abort, preserving evidence.
run_stage(){
  local name="$1"; shift
  if [[ -f "$EXP/$name/DONE" ]]; then echo "[SKIP] $name: DONE exists"; return; fi
  if [[ -e "$EXP/$name" ]]; then echo "[STOP] $EXP/$name exists without DONE. Inspect logs before any recovery." >&2; exit 2; fi
  echo "[RUN] $name $(date -Is)"
  "$@" "${EXP}"
  [[ -f "$EXP/$name/DONE" ]] || { echo "[STOP] $name exited without DONE" >&2; exit 3; }
  echo "[PASS] $name $(date -Is)"
}
# If a prior smoke stage exists and is complete, reuse it; otherwise run frozen pilot.
run_stage 01_smoke env PYTHON="${PYTHON:-python}" WORKERS="$WORKERS" scripts/01_smoke.sh
run_stage 02_collect env PYTHON="${PYTHON:-python}" RUNS="$RUNS" WORKERS="$WORKERS" scripts/02_collect.sh
run_stage 03_train_q env PYTHON="${PYTHON:-python}" Q_SEEDS="$Q_SEEDS" DEVICE="$DEVICE" scripts/03_train_q.sh
run_stage 04_train_cgan env PYTHON="${PYTHON:-python}" SEEDS="$SEEDS" BATCHES="$BATCHES" DEVICE="$DEVICE" scripts/04_train_cgan.sh
run_stage 05_evaluate env PYTHON="${PYTHON:-python}" EVAL_RUNS="$EVAL_RUNS" WORKERS="$WORKERS" scripts/05_evaluate.sh
run_stage 06_analyze env PYTHON="${PYTHON:-python}" scripts/06_analyze.sh
printf '\n[SUCCESS] All six repository stages validated: %s\n' "$EXP"
printf 'Results: %s/06_analyze/results\n' "$EXP"
