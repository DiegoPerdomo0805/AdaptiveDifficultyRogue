#!/usr/bin/env bash
# Stage A -- smoke checks + calibration pilot.
#   scripts/01_smoke.sh <exp_dir>
# Env: PILOT_RUNS (per controller, default 50), PILOT_SEED (default: start of
# seed_blocks.pilot), WORKERS (default 4), PROJECT_ROOMS (default 30000).
source "$(dirname "$0")/common.sh"
usage_exp "$@"
stage_begin "$1" 01_smoke "$@"
cd "$REPO"
PILOT_RUNS="${PILOT_RUNS:-50}"
PILOT_SEED="${PILOT_SEED:-$(json_get experiment_config.json 'd["seed_blocks"]["pilot"][0]')}"
WORKERS="${WORKERS:-4}"

"$PY" -m compileall -q -x '(\.venv|DATA_AUGMENTATION)' .
"$PY" -m pytest -q
for s in bot_runner train_q train_cgan train_q_loot train_loot_policy loot_dataset analyze_loot \
         evaluate_controllers analyze_results dataset_tools; do
    "$PY" "$s.py" --help > /dev/null
done

S=$PILOT_SEED
for c in fixed heuristic exploration; do
    "$PY" bot_runner.py --runs "$PILOT_RUNS" --seed "$S" --enemy-controller "$c" \
        --reward-policy "${PILOT_REWARD:-exploration}" --workers "$WORKERS" \
        --out "$SDIR/pilot_$c.csv" --validate
    S=$((S + PILOT_RUNS))
done
P=("$SDIR/pilot_fixed.csv" "$SDIR/pilot_heuristic.csv" "$SDIR/pilot_exploration.csv")
"$PY" dataset_tools.py merge --csv "${P[0]}" --csv "${P[1]}" --csv "${P[2]}" --out "$SDIR/pilot_all.csv"
"$PY" dataset_tools.py stats --csv "$SDIR/pilot_all.csv" --out "$SDIR/pilot_stats.json" \
    --project-rooms "${PROJECT_ROOMS:-30000}"
"$PY" loot_dataset.py --csv "$SDIR/pilot_all.csv" --out "$SDIR/pilot_loot.csv"
"$PY" balance.py calibrate --csv "${P[0]}" --csv "${P[1]}" --csv "${P[2]}" \
    --loot-csv "$SDIR/pilot_loot.csv" --out "$SDIR/calibration_report.json"
"$PY" analyze_loot.py --loot-csv "$SDIR/pilot_loot.csv" --out-dir "$SDIR/loot_descriptive" --bootstrap 300 > /dev/null
echo
echo "NEXT: review $SDIR/calibration_report.json (proposed_targets follow the pre-specified rule),"
echo "      set enemy_balance_targets / loot_horizon_balance_targets in experiment_config.json,"
echo "      change status to FROZEN, bump experiment_config_version and commit BEFORE 02_collect.sh."
stage_done
