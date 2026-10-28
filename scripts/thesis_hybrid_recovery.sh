#!/usr/bin/env bash
# Exploratory HYBRID recovery, NOT a rerun or amendment of frozen exp_v3.2.
# Run from the root of AdaptiveDifficultyRogue after activating its venv.
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "validate" && "$MODE" != "final" ]]; then
  echo "Usage: bash thesis_hybrid_recovery.sh validate|final" >&2
  exit 2
fi
for required in experiment_config.json selection.py controllers.py bot_runner.py; do
  [[ -f "$required" ]] || { echo "Run from the repository root; missing $required" >&2; exit 2; }
done
python -c 'import torch; import selection; import controllers' >/dev/null
python freeze_experiment.py verify

# Independent follow-up version: never write under 20261021_main.
OUTPUT="experiments/20261022_hybrid_v1"
if [[ "$MODE" == "validate" && -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing $OUTPUT. Inspect/retain it." >&2
  exit 3
fi
if [[ "$MODE" == "final" && ! -f "$OUTPUT/selection.lock.json" ]]; then
  echo "No sealed selection; final-test evaluation prohibited." >&2
  exit 3
fi
if [[ "$MODE" == "final" && -e "$OUTPUT/final" ]]; then
  echo "Final results already exist; refusing to overwrite." >&2
  exit 3
fi

python - "$MODE" "$OUTPUT" <<'PY'
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import torch
import analyze_results
import balance
import bot_runner
import controllers as ctl
import dda_core as core
import ml_data
import rewards
import selection

mode, out_string = sys.argv[1:]
out = Path(out_string)
cfg = balance.load_experiment_config()
assert cfg['experiment_config_version'] == 'exp_v3.2'
assert cfg['cgan_selection']['closed_loop']['max_ood_excess'] == 0.05

# This is a new preregistered FOLLOW-UP, not a modification of exp_v3.2.
# Avoid the 8000000..8000199 validation seeds already inspected in the failed sweep.
SEEDS = list(range(8100000, 8100200))
FINAL_SEEDS = list(range(9500000, 9500400))
ALPHAS = (0.25, 0.50, 0.75)
SOURCE = 'experiments/20261021_main/04_train_cgan/runs'
CHECKPOINTS = {
    'best_balance': f'{SOURCE}/s2_b64_lb10.0_ld1.0/candidates/epoch_0045',
    'closest_ood': f'{SOURCE}/s2_b128_lb10.0_ld1.0/candidates/epoch_0025',
}
EXPERIMENT_ID = 'hybrid_followup_v1'

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def snapshot():
    return {
        'experiment_id': EXPERIMENT_ID,
        'design': 'posthoc exploratory hybrid of two prior validation-selected candidates',
        'config_version': cfg['experiment_config_version'],
        'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'checkpoint_sha256': {k: digest(Path(v) / 'cgan_checkpoint.pt') for k,v in CHECKPOINTS.items()},
        'alphas': list(ALPHAS), 'validation_seeds': [SEEDS[0], SEEDS[-1]],
        'final_seeds': [FINAL_SEEDS[0], FINAL_SEEDS[-1]],
        'reward_policy': cfg['cgan_selection']['closed_loop']['reward_policy'],
        'unchanged_gates': {k: v for k,v in cfg['cgan_selection']['closed_loop'].items()
                            if k.startswith('max_')},
        'limitations': 'Hybrid differs from original cGAN; new validation subset, original final_test untouched until selection lock.'
    }

class HybridController(ctl.CGANController):
    """Runtime-only subclass: original source and checkpoints remain unchanged."""
    def __init__(self, model_dir, alpha, lm=None):
        super().__init__(model_dir, cfg=cfg, models=lm)
        self.alpha = float(alpha)
        self.version = 'hybrid_followup_v1'

    def decide(self, ctx):
        if self.models is None:
            return self._fallback(ctx, self.load_error or 'load_error:unknown')
        if self.disabled:
            return self._fallback(ctx, 'disabled_after_invalid_outputs')
        noise = [self.noise_rng.gauss(0.0,1.0) for _ in range(self.models.noise_dim)]
        try:
            lm = self.models
            x = torch.tensor(lm.cond_scaler.transform([list(map(float, ctx.cond_vec))]), dtype=torch.float32)
            z = torch.tensor([noise], dtype=torch.float32)
            with torch.no_grad():
                gn = lm.generator(x, z)[0].tolist()
            generated = ml_data.denormalize_action(gn)
            heur = self.fallback.decide(ctx).action
            raw = {key: self.alpha * generated[i] + (1-self.alpha) * float(heur[key])
                   for i,key in enumerate(core.ACTION_KEYS)}
            if not all(math.isfinite(v) for v in raw.values()):
                raise ValueError('nonfinite hybrid proposal')
            applied = core.finalize_action(raw)
            actions = [[float(applied[k]) for k in core.ACTION_KEYS]]
            ratio = selection.support_ratios(lm, x, actions)[0]
            if not math.isfinite(ratio) or ratio > lm.ood_factor:
                self.ood_count += 1
                return self._fallback(ctx, f'ood_action:support_ratio={ratio:.6f} > {lm.ood_factor}')
            return ctl.Decision(applied, 'cgan', self.version,
                                model_checkpoint_id=self.checkpoint_id,
                                action_source='model_hybrid', support_distance=f'{ratio:.6f}')
        except Exception as exc:
            self.invalid_count += 1
            if self.invalid_count >= self.max_invalid:
                self.disabled = True
            return self._fallback(ctx, f'invalid_output:{type(exc).__name__}:{str(exc)[:100]}')

def rollout(controller, seeds):
    policy = rewards.make_reward_policy(cfg['cgan_selection']['closed_loop']['reward_policy'], cfg=cfg)
    runs = []
    for seed in seeds:
        _, samples, reason, boss, _ = bot_runner.simulate_run(
            seed, controller, exp_version=cfg['experiment_config_version'], reward_policy=policy)
        runs.append({'seed': seed, 'rows': [dict(s.__dict__) for s in samples],
                     'termination': reason, 'boss_reached': boss})
    return runs

def room_diagnostics(lm, alpha, runs):
    """Independent hybrid diversity/sensitivity checks on reached states."""
    rows = [r for run in runs for r in run['rows'] if r['room_kind'] == 'enemy'][:100]
    if not rows:
        return {'validity': float('nan'), 'diversity': 0, 'sensitivity': 0}
    x = selection.state_tensor(lm, rows)
    h = torch.tensor([ml_data.normalize_action(ml_data.heuristic_action_for_row(r))
                      for r in rows], dtype=torch.float32)
    rng = torch.Generator().manual_seed(314159)
    with torch.no_grad():
        samples = []
        for _ in range(8):
            z = torch.randn(len(rows), lm.noise_dim, generator=rng)
            samples.append(alpha*lm.generator(x,z)+(1-alpha)*h)
        diversity = sum((samples[i]-samples[j]).abs().mean().item()
                        for i in range(8) for j in range(i+1,8)) / 28
        z = torch.randn(len(rows), lm.noise_dim, generator=rng)
        base = alpha*lm.generator(x,z)+(1-alpha)*h
        sensitivity = sum((alpha*lm.generator(x+torch.nn.functional.one_hot(
            torch.tensor(j),x.shape[1]).float(),z)+(1-alpha)*h-base).abs().mean().item()
                          for j in range(x.shape[1])) / x.shape[1]
    return {'validity': 1.0, 'diversity': diversity, 'sensitivity': sensitivity,
            'note': 'Generated continuous hybrid proposals; heuristic reference held fixed for state perturbations.'}

def report_for(lm, controller, seeds):
    runs = rollout(controller, seeds)
    metrics = selection.rollout_metrics(runs,cfg,lm)
    return runs,metrics

def eligible(m,diag,epoch=100):
    # Keep the ORIGINAL selection gates unchanged; independent hybrid room diagnostics.
    return selection.eligibility(m,diag,epoch,cfg)

def compact(m):
    keys = ('n_runs','n_decisions','mean_B_r','cvar_B_r','selection_score',
            'ordinary_room_death_rate','victory_rate','validity','cgan_ood_rate',
            'heuristic_same_state_ood_rate','ood_excess','runtime_guard_rate',
            'fallback_rate','fallback_reasons','q_diagnostics')
    return {k:m.get(k) for k in keys}

def clean_json(obj):
    # Strict JSON: explicit null instead of nonstandard NaN/Infinity tokens.
    if isinstance(obj,dict): return {k:clean_json(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [clean_json(v) for v in obj]
    if isinstance(obj,float) and not math.isfinite(obj): return None
    return obj

if mode == 'validate':
    for seed in SEEDS:
        selection.assert_not_test_seeds([seed],cfg)
    plan = snapshot()
    # Fail closed if artifacts absent; no partial experiment folder is created.
    for path in CHECKPOINTS.values():
        if not (Path(path)/'cgan_checkpoint.pt').is_file():
            raise SystemExit(f'Missing checkpoint: {path}')
    out.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    rows=[]
    for name,path in CHECKPOINTS.items():
        lm = ctl.load_model_runtime().LoadedModels.load(path,load_q=True)
        for alpha in ALPHAS:
            print(f'VALIDATING {name}, alpha={alpha:.2f} on {len(SEEDS)} new selection seeds',flush=True)
            ctrl=HybridController(path,alpha,lm)
            runs,m = report_for(lm,ctrl,SEEDS)
            diag=room_diagnostics(lm,alpha,runs)
            reasons=eligible(m,diag)
            entry={'candidate':name,'checkpoint_id':lm.checkpoint_id,'alpha':alpha,
                   'metrics':compact(m),'room_level':diag,'ineligible_reasons':reasons}
            rows.append(entry)
            print(f'  score={m["selection_score"]:.6f} excess={m["ood_excess"]:+.6f}'
                  f' diversity={diag["diversity"]:.5f} sensitivity={diag["sensitivity"]:.5f}'
                  f' -> {"ELIGIBLE" if not reasons else "REJECTED: "+"; ".join(reasons)}',flush=True)
            (out/'validation_progress.json').write_text(json.dumps(clean_json(rows),indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps(clean_json(rows),indent=2)+'\n')
    winners=[r for r in rows if not r['ineligible_reasons']]
    if not winners:
        (out/'NO_ELIGIBLE').write_text('Original gates retained. No candidate eligible.\n')
        print('NO ELIGIBLE HYBRID. Final test prohibited. Original exp_v3.2 unchanged.')
        raise SystemExit(4)
    win=min(winners,key=lambda r:(r['metrics']['selection_score'],r['candidate'],r['alpha']))
    lock={'plan_sha256':digest(out/'plan.json'),'candidate':win['candidate'],
          'checkpoint_id':win['checkpoint_id'],'alpha':win['alpha'],
          'validation_score':win['metrics']['selection_score'],
          'validation_sha256':digest(out/'validation.json')}
    (out/'selection.lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    print('SELECTION LOCKED:',lock,'\nFinal-test phase requires a separate explicit invocation.')

else:
    plan=json.loads((out/'plan.json').read_text())
    lock=json.loads((out/'selection.lock.json').read_text())
    if digest(out/'plan.json') != lock['plan_sha256'] or digest(out/'validation.json') != lock['validation_sha256']:
        raise SystemExit('Plan or validation artifact changed after selection lock.')
    current=snapshot()
    if plan != current:
        raise SystemExit('Code/config/checkpoint provenance changed; final evaluation prohibited.')
    if FINAL_SEEDS[0] < cfg['seed_blocks']['final_test'][0] or FINAL_SEEDS[-1] > cfg['seed_blocks']['final_test'][1]:
        raise SystemExit('Final seeds outside reserved block.')
    output = out/'final'
    output.mkdir(parents=True,exist_ok=False)
    chosen=CHECKPOINTS[lock['candidate']]
    lm=ctl.load_model_runtime().LoadedModels.load(chosen,load_q=True)
    if lm.checkpoint_id != lock['checkpoint_id']:
        raise SystemExit('Checkpoint identity mismatch.')
    result={}
    by_seed={}
    for name,controller in (
        ('fixed',ctl.FixedController(cfg)),
        ('heuristic',ctl.HeuristicController(cfg)),
        ('hybrid',HybridController(chosen,lock['alpha'],lm))):
        print(f'FINAL TEST {name}: {len(FINAL_SEEDS)} fresh held-out seeds',flush=True)
        runs=rollout(controller,FINAL_SEEDS)
        m=selection.rollout_metrics(runs,cfg,lm if name=='hybrid' else None)
        result[name]=dict(metrics=compact(m),B_r=m['B_r'])
        by_seed[name]={run['seed']: balance.balance_distance(
            analyze_results.run_level_vector(ordinary,cfg),cfg)
            for run in runs
            if (ordinary := [r for r in run['rows'] if r['room_kind']=='enemy'])}
        (output/'results_progress.json').write_text(json.dumps(clean_json(result),indent=2)+'\n')
    # Paired bootstrap of run-level B_r. Never used for selection.
    rng=random.Random(20261022)
    common=sorted(set.intersection(*(set(v) for v in by_seed.values())))
    n=len(common)
    paired={}
    if n:
        for ref in ('fixed','heuristic'):
            diffs=[by_seed['hybrid'][s]-by_seed[ref][s] for s in common]
            draws=[sum(diffs[rng.randrange(n)] for _ in range(n))/n for _ in range(2000)]
            draws.sort()
            paired[f'hybrid_minus_{ref}']={'mean_diff':sum(diffs)/n,
                'ci95_percentile':[draws[49],draws[1949]],'n_matched_seeds':n}
    else:
        paired['warning']='No common runs with ordinary combat rooms.'
    final={'lock':lock,'seed_block':'final_test','seed_range':[FINAL_SEEDS[0],FINAL_SEEDS[-1]],
           'results':result,'paired_bootstrap':paired,
           'caveat':'Single held-out comparison of exploratory hybrid; human perception not evaluated.'}
    (output/'final_report.json').write_text(json.dumps(clean_json(final),indent=2)+'\n')
    print(json.dumps(clean_json({k:v['metrics'] for k,v in result.items()}),indent=2))
    print('FINAL REPORT:',output/'final_report.json')
PY
