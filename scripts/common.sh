#!/usr/bin/env bash
# Shared helpers for the stage wrappers. Source this file; do not run it.
#
# Every stage writes into  $EXP/<stage>/  and:
#   * refuses to start if that directory exists without a DONE marker
#     (a crashed/partial stage must be inspected and moved aside by hand);
#   * exits 0 immediately if DONE already exists (safe restart at a stage
#     boundary: re-running the pipeline skips completed stages);
#   * records git HEAD, dirty state, versions and config hashes in meta.json;
#   * tees stdout/stderr into log.txt;
#   * writes DONE only after its validation succeeded.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

die() { echo "ERROR: $*" >&2; exit 1; }

usage_exp() {
    [ $# -ge 1 ] || die "usage: $0 <experiment_dir>   (e.g. experiments/2026-10-07_main)"
}

stage_begin() {   # stage_begin <exp_dir> <stage_name>
    EXP="$1"; STAGE="$2"; SDIR="$EXP/$STAGE"
    if [ -f "$SDIR/DONE" ]; then
        echo "[$STAGE] already complete ($SDIR/DONE) -- skipping."
        exit 0
    fi
    [ -e "$SDIR" ] && die "$SDIR exists but has no DONE marker. Inspect it and move it aside; refusing to overwrite."
    mkdir -p "$SDIR"
    exec > >(tee -a "$SDIR/log.txt") 2>&1
    echo "[$STAGE] start $(date -u +%Y-%m-%dT%H:%M:%SZ) in $SDIR"
    (cd "$REPO" && "$PY" - "$SDIR/meta.json" "$STAGE" "$0 ${*:3}" <<'PYEOF'
import json, subprocess, sys, time
import balance, bot_runner, dda_core, telemetry
def git(*a):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"
cfg = balance.load_experiment_config()
extra = bot_runner.policy_extra()
extra["experiment_config_hash"] = balance.experiment_config_hash(cfg)
meta = {"stage": sys.argv[2], "command": sys.argv[3],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "schema_version": telemetry.SCHEMA_VERSION, "dda_version": telemetry.DDA_VERSION,
        "bot_policy_version": telemetry.BOT_POLICY_VERSION,
        "experiment_config_version": cfg["experiment_config_version"],
        "experiment_config_status": cfg.get("status", ""),
        "experiment_config_hash": extra["experiment_config_hash"],
        "bot_config_hash": dda_core.config_hash(extra=extra)}
json.dump(meta, open(sys.argv[1], "w"), indent=2)
print(json.dumps(meta, indent=2))
PYEOF
    )
}

stage_done() {
    date -u +%Y-%m-%dT%H:%M:%SZ > "$SDIR/DONE"
    echo "[$STAGE] DONE"
}

require_done() {  # require_done <exp_dir> <stage_name>
    [ -f "$1/$2/DONE" ] || die "stage '$2' has not completed in $1 (missing $1/$2/DONE)"
}

json_get() {      # json_get <file> <python-expression over d>
    "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print($2)" "$1"
}
