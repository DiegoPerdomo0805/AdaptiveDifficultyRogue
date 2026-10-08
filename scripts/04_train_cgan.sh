#!/usr/bin/env bash
# Stage D -- bounded cGAN sweep, selection by composite validation score.
#   scripts/04_train_cgan.sh <exp_dir>
# Env: SEEDS ("0 1 2"), BATCHES ("64 128"), LAMBDA_BALANCE ("10.0"),
#      LAMBDA_DIVERSITY ("1.0"), EPOCHS (200), DEVICE (auto).
# Also trains learned_loot ONLY if the Q_loot gate passed in 03_train_q.
source "$(dirname "$0")/common.sh"
usage_exp "$@"
require_done "$1" 03_train_q
stage_begin "$1" 04_train_cgan "$@"
cd "$REPO"
DATA="$EXP/02_collect/train_data.csv"
QDIR="$EXP/03_train_q/q_selected"
for s in ${SEEDS:-0 1 2}; do for b in ${BATCHES:-64 128}; do
for lb in ${LAMBDA_BALANCE:-10.0}; do for ld in ${LAMBDA_DIVERSITY:-1.0}; do
    tag="s${s}_b${b}_lb${lb}_ld${ld}"
    set +e
    "$PY" train_cgan.py --csv "$DATA" --q-dir "$QDIR" --out-dir "$SDIR/runs/$tag" --seed "$s" \
        --batch-size "$b" --lambda-balance "$lb" --lambda-diversity "$ld" \
        --epochs "${EPOCHS:-200}" --device "${DEVICE:-auto}"
    rc=$?
    set -e
    case $rc in
        0) ;;
        3) echo "sweep run $tag: closed-loop model-selection failure (no eligible checkpoint)" ;;
        *) die "train_cgan.py crashed for $tag (rc=$rc)" ;;
    esac
done; done; done; done
"$PY" - "$SDIR" <<'PYEOF'
import glob, json, math, os, sys
sdir = sys.argv[1]
cands = []
for p in glob.glob(os.path.join(sdir, "runs", "*", "cgan_report.json")):
    r = json.load(open(p))
    ok = r.get("selection") == "OK" and math.isfinite(r.get("best_score", float("nan")))
    cands.append({"run": os.path.dirname(p), "selection": r.get("selection"),
                  "score": r.get("best_score"), "checkpoint_id": r.get("checkpoint_id"),
                  "selected_epoch": r.get("best_epoch"), "reason": r.get("reason", "")})
# Every sweep run was scored on the SAME frozen validation seeds with the same
# observed closed-loop rule, so the lowest score across runs is selected.
valid = sorted((c for c in cands if c["selection"] == "OK"), key=lambda c: c["score"])
if not valid:
    sys.exit("MODEL-SELECTION FAILURE: no sweep run produced an eligible checkpoint")
json.dump({"selected": valid[0], "candidates": cands}, open(os.path.join(sdir, "selection.json"), "w"), indent=2)
os.symlink(os.path.relpath(valid[0]["run"], sdir), os.path.join(sdir, "selected"))
print("selected", valid[0]["run"], "score", valid[0]["score"], "id", valid[0]["checkpoint_id"])
PYEOF
if [ "$(json_get "$EXP/03_train_q/q_loot/q_loot_report.json" 'd["gate"]["passed"]')" = "True" ]; then
    "$PY" train_loot_policy.py --loot-csv "$EXP/02_collect/train_loot.csv" \
        --q-loot-dir "$EXP/03_train_q/q_loot" --out-dir "$SDIR/loot_model"
else
    echo "learned-loot gate failed in 03_train_q: learned_loot NOT trained (by design)."
fi
stage_done
