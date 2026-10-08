#!/usr/bin/env bash
# Stage B -- principal collection in isolated, validated seed blocks.
#   scripts/02_collect.sh <exp_dir>
# Env: RUNS (total, default 9000 ~ 30k combat rooms), BLOCK_RUNS (default 1500),
#      MIX / REWARD_MIX (default: experiment_config.principal_collection; learned_loot refused),
#      SEED (default: start of seed_blocks.train_collection), WORKERS (default 6),
#      ALLOW_PILOT_CONFIG=1 to collect with a non-FROZEN experiment config (NOT for the thesis).
source "$(dirname "$0")/common.sh"
usage_exp "$@"
stage_begin "$1" 02_collect "$@"
cd "$REPO"
STATUS="$(json_get experiment_config.json 'd["status"].split(":")[0]')"
if [ "${ALLOW_PILOT_CONFIG:-0}" != "1" ]; then
    [ "$STATUS" = "FROZEN" ] || die "experiment_config.json status is '$STATUS', not FROZEN."
    "$PY" freeze_experiment.py verify || die "freeze verification failed"
fi
[ -z "$(git status --porcelain --untracked-files=no)" ] || [ "${ALLOW_DIRTY:-0}" = "1" ] \
    || die "working tree has uncommitted changes; commit before the dataset freeze (ALLOW_DIRTY=1 overrides)."
RUNS="${RUNS:-9000}"; BLOCK_RUNS="${BLOCK_RUNS:-1500}"; WORKERS="${WORKERS:-6}"
MIX="${MIX:-$(json_get experiment_config.json 'd["principal_collection"]["enemy_controller_mix"]')}"
REWARD_MIX="${REWARD_MIX:-$(json_get experiment_config.json 'd["principal_collection"]["reward_policy_mix"]')}"
case "$REWARD_MIX" in *learned_loot*) die "learned_loot is never used during collection" ;; esac
SEED="${SEED:-$(json_get experiment_config.json 'd["seed_blocks"]["train_collection"][0]')}"
MAXSEED="$(json_get experiment_config.json 'd["seed_blocks"]["train_collection"][1]')"
[ $((SEED + RUNS - 1)) -le "$MAXSEED" ] || die "seed range exceeds seed_blocks.train_collection"

mkdir -p "$SDIR/blocks"
ARGS=()
done_runs=0; b=0
while [ "$done_runs" -lt "$RUNS" ]; do
    n=$(( RUNS - done_runs < BLOCK_RUNS ? RUNS - done_runs : BLOCK_RUNS ))
    out="$SDIR/blocks/block_$(printf %03d $b).csv"
    "$PY" bot_runner.py --runs "$n" --seed $((SEED + done_runs)) --enemy-controller "$MIX" \
        --reward-policy "$REWARD_MIX" \
        --workers "$WORKERS" --out "$out" --validate
    "$PY" dataset_tools.py stats --csv "$out" --out "${out%.csv}.stats.json" \
        || die "stop condition raised for $out (see ${out%.csv}.stats.json)"
    ARGS+=(--csv "$out")
    done_runs=$((done_runs + n)); b=$((b + 1))
done
"$PY" dataset_tools.py merge "${ARGS[@]}" --out "$SDIR/train_data.csv"
"$PY" dataset_tools.py stats --csv "$SDIR/train_data.csv" --out "$SDIR/train_stats.json" \
    || die "stop condition raised for the merged dataset"
"$PY" loot_dataset.py --csv "$SDIR/train_data.csv" --out "$SDIR/train_loot.csv"
"$PY" balance.py calibrate --csv "$SDIR/train_data.csv" --loot-csv "$SDIR/train_loot.csv" \
    --out "$SDIR/outcome_report.json"
stage_done
