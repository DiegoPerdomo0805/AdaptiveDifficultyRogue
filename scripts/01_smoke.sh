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
         evaluate_controllers analyze_results dataset_tools freeze_experiment; do
    "$PY" "$s.py" --help > /dev/null
done

S=$PILOT_SEED
P=()
for c in fixed heuristic exploration; do
    for r in fixed_uniform heuristic exploration; do       # every collection reward policy
        "$PY" bot_runner.py --runs "$PILOT_RUNS" --seed "$S" --enemy-controller "$c" \
            --reward-policy "$r" --workers "$WORKERS" --out "$SDIR/pilot_${c}_${r}.csv" --validate
        P+=(--csv "$SDIR/pilot_${c}_${r}.csv")
        S=$((S + PILOT_RUNS))
    done
done
"$PY" dataset_tools.py merge "${P[@]}" --out "$SDIR/pilot_all.csv"
"$PY" dataset_tools.py stats --csv "$SDIR/pilot_all.csv" --out "$SDIR/pilot_stats.json" \
    --project-rooms "${PROJECT_ROOMS:-30000}"
"$PY" loot_dataset.py --csv "$SDIR/pilot_all.csv" --out "$SDIR/pilot_loot.csv"
"$PY" balance.py calibrate --csv "$SDIR/pilot_all.csv" \
    --loot-csv "$SDIR/pilot_loot.csv" --out "$SDIR/calibration_report.json"
"$PY" balance.py audit --csv "$SDIR/pilot_all.csv" --loot-csv "$SDIR/pilot_loot.csv" \
    --out "$SDIR/target_audit.json"
"$PY" analyze_loot.py --loot-csv "$SDIR/pilot_loot.csv" --out-dir "$SDIR/loot_descriptive" --bootstrap 300 > /dev/null
echo
echo "NEXT: review $SDIR/target_audit.json; targets are design-intent values (not pilot quantiles)."
echo "      Freeze: status FROZEN + commit, then 'python freeze_experiment.py record' + commit, BEFORE 02_collect.sh."
stage_done
