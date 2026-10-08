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
# Enemy axis: vary the enemy controller, reward policy fixed to the heuristic.
"$PY" evaluate_controllers.py --axis enemy --controllers fixed,heuristic,cgan --reward-policy heuristic \
    --model-dir "$EXP/04_train_cgan/selected" --runs "${EVAL_RUNS:-300}" --seed "$EVAL_SEED" \
    --workers "${WORKERS:-6}" --out-dir "$SDIR/eval_enemy" --train-csv "$EXP/02_collect/train_data.csv" --plots
"$PY" - "$SDIR/eval_enemy/summary.json" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))["controllers"]["cgan"]
bad = {k: v for k, v in c["fallback_reasons"].items() if not k.startswith("ood_action:")}
if bad or c["fallback_rows"] > 0.05 * max(c["n_combat_rooms"], 1):
    sys.exit(f"cGAN fallbacks are not rare recorded OOD guards: {c['fallback_reasons']}")
PYEOF
# Reward axis: vary the reward policy, enemy controller fixed to the heuristic.
POLICIES="fixed_uniform,heuristic"; LOOTARGS=()
if [ -f "$EXP/04_train_cgan/loot_model/loot_policy_checkpoint.pt" ]; then
    POLICIES="$POLICIES,learned_loot"; LOOTARGS=(--loot-model-dir "$EXP/04_train_cgan/loot_model")
fi
"$PY" evaluate_controllers.py --axis reward --reward-policies "$POLICIES" --enemy-controller heuristic \
    "${LOOTARGS[@]}" --q-loot-dir "$EXP/03_train_q/q_loot" --runs "${EVAL_RUNS:-300}" --seed "$EVAL_SEED" \
    --workers "${WORKERS:-6}" --out-dir "$SDIR/eval_reward" --train-csv "$EXP/02_collect/train_data.csv" --plots
stage_done
