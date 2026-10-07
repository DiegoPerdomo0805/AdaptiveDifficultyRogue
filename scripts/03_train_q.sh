#!/usr/bin/env bash
# Stage C -- Q gate: train Q(x,a) and Q(x) for several seeds on the frozen dataset.
#   scripts/03_train_q.sh <exp_dir>
# Env: Q_SEEDS (default "0 1 2"), DEVICE (default auto). The first seed's Q guides the cGAN.
source "$(dirname "$0")/common.sh"
usage_exp "$@"
require_done "$1" 02_collect
stage_begin "$1" 03_train_q "$@"
cd "$REPO"
DATA="$EXP/02_collect/train_data.csv"
fail=0
for s in ${Q_SEEDS:-0 1 2}; do
    set +e
    "$PY" train_q.py --csv "$DATA" --out-dir "$SDIR/q_seed$s" --seed "$s" --device "${DEVICE:-auto}"
    rc=$?
    set -e
    [ $rc -eq 0 ] || { echo "Q seed $s FAILED its gate (rc=$rc)"; fail=1; }
done
[ $fail -eq 0 ] || die "Q gate failed for at least one seed: do not train the cGAN; improve action coverage."
first=$(echo ${Q_SEEDS:-0 1 2} | awk '{print $1}')
ln -s "q_seed$first" "$SDIR/q_selected"
stage_done
