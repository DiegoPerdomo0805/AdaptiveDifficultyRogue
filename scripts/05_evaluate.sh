#!/usr/bin/env bash
# Stage E -- frozen-checkpoint equivalent-seed evaluation (fixed, heuristic, cgan).
#   scripts/05_evaluate.sh <exp_dir>
# Env: EVAL_RUNS (default 300), EVAL_SEED (default: start of seed_blocks.evaluation), WORKERS (6).
source "$(dirname "$0")/common.sh"
usage_exp "$@"
require_done "$1" 04_train_cgan
stage_begin "$1" 05_evaluate "$@"
cd "$REPO"
EVAL_SEED="${EVAL_SEED:-$(json_get experiment_config.json 'd["seed_blocks"]["evaluation"][0]')}"
"$PY" evaluate_controllers.py --controllers fixed,heuristic,cgan \
    --model-dir "$EXP/04_train_cgan/selected" --runs "${EVAL_RUNS:-300}" --seed "$EVAL_SEED" \
    --workers "${WORKERS:-6}" --out-dir "$SDIR/eval" --train-csv "$EXP/02_collect/train_data.csv" --plots
[ "$(json_get "$SDIR/eval/summary.json" 'd["controllers"]["cgan"]["fallback_rows"]')" = "0" ] \
    || die "cGAN evaluation used heuristic fallbacks; see $SDIR/eval/summary.md"
stage_done
