#!/usr/bin/env bash
# Stage F -- assemble the thesis results bundle.
#   scripts/06_analyze.sh <exp_dir>
source "$(dirname "$0")/common.sh"
usage_exp "$@"
require_done "$1" 05_evaluate
stage_begin "$1" 06_analyze "$@"
cd "$REPO"
"$PY" analyze_results.py --eval-dir "$EXP/05_evaluate/eval" --out-dir "$SDIR/results" --plots --bootstrap 2000
cp "$EXP/03_train_q/q_selected/q_report.json" "$SDIR/results/q_report.json"
cp "$EXP/04_train_cgan/selection.json" "$SDIR/results/cgan_selection.json"
cp "$EXP/04_train_cgan/selected/cgan_report.json" "$SDIR/results/cgan_report.json"
cp "$EXP/02_collect/train_stats.json" "$SDIR/results/train_stats.json"
cp "$REPO/experiment_config.json" "$SDIR/results/experiment_config.json"
for s in 01_smoke 02_collect 03_train_q 04_train_cgan 05_evaluate; do
    cp "$EXP/$s/meta.json" "$SDIR/results/meta_$s.json"
done
stage_done
