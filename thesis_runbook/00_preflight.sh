#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This file can be placed directly in the repository root or in thesis_runbook/.
if [[ -f "$ROOT/experiment_config.json" ]]; then REPO="$ROOT"; else REPO="$(cd "$ROOT/.." && pwd)"; fi
cd "$REPO"
PY="${PYTHON:-python}"
fail(){ echo "[STOP] $*" >&2; exit 1; }
[[ -f experiment_config.json && -f freeze_experiment.py && -f scripts/02_collect.sh ]] || fail "Not in AdaptiveDifficultyRogue: place thesis_runbook inside repository."
for c in git "$PY" tar df; do command -v "$c" >/dev/null || fail "Missing executable: $c"; done
[[ "$(git branch --show-current)" == "cGAN_vs_heuristic" ]] || fail "Wrong branch: $(git branch --show-current); expected cGAN_vs_heuristic"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || fail "Tracked changes detected; commit/review before experiment"
"$PY" - <<'PY'
import json, sys
c=json.load(open('experiment_config.json'))
if c.get('status','').split(':')[0] != 'FROZEN': sys.exit('[STOP] experiment_config.json is not FROZEN')
for block in ('train_collection','final_test'):
    seeds=c['seed_blocks'][block]
    print(f'{block}: {seeds}')
print('config version:',c.get('experiment_config_version'))
PY
"$PY" freeze_experiment.py verify || fail "Freeze validation failed"
"$PY" - <<'PY'
import sys
try: import torch
except Exception as ex: sys.exit('[STOP] PyTorch import failed: '+str(ex))
print('torch:',torch.__version__,'CUDA:',torch.cuda.is_available())
if not torch.cuda.is_available() and __import__('os').environ.get('DEVICE','cuda')=='cuda': sys.exit('[STOP] GPU CUDA required; check WSL torch installation or set DEVICE=cpu explicitly')
if torch.cuda.is_available():print('GPU:',torch.cuda.get_device_name(0))
PY
if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader || true; fi
df -h .
if [[ "${SKIP_TESTS:-0}" != 1 ]]; then "$PY" -m pytest -q || fail "Tests failed"; fi
printf '\n[PASS] Preflight completed. HEAD=%s\n' "$(git rev-parse HEAD)"
