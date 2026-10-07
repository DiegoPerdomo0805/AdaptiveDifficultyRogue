# Adaptive Difficulty Rogue: outcome-guided cGAN vs. heuristic DDA

A pygame top-down roguelike used as a testbed for **dynamic difficulty adjustment (DDA)**.
Before each room, a controller picks the enemy intervention

```text
a = [enemy_hp_multiplier, enemy_damage_multiplier, enemy_speed_multiplier, spawn_count]
```

from the 9-dimensional pre-room condition vector

```text
x = [melee_ratio, magic_ratio, hits_per_kill_melee, hits_per_kill_magic,
     damage_taken_to_dealt_ratio, pre_room_hp_ratio, cumulative_kills_normalized,
     node_depth_normalized, run_progress_normalized]
```

The experiment compares three controllers under equivalent seeds and maps: **fixed**
difficulty, the rule-based **heuristic**, and an **outcome-guided cGAN**.

## How the cGAN learns from outcomes (not from the heuristic)

```text
Q(x, a) -> o      outcome evaluator, trained ONLY on observed (x, a, o) tuples
G(x, z) -> a      generator, bounded in the design action box by construction
D(x, a) -> logit  discriminator: observed vs. generated (x, a) pairs

L_G = L_adv + lambda_balance * L_balance + lambda_diversity * L_diversity
  L_adv       = BCE(D(x, G(x,z1)), 1)                       stay in the observed safe action support
  L_balance   = balance_loss(Q_frozen(x, G(x,z1)))          move predicted outcomes into target intervals
  L_diversity = mean(relu(tau - |G(x,z1)-G(x,z2)|_1 / |z1-z2|_1))   penalize noise collapse
```

* Training data mixes **exploration** (seeded, bounded perturbations around the heuristic,
  uniform samples, reference actions), **heuristic** and **fixed** actions, so action effects on
  outcomes are identifiable.
* The heuristic is a baseline, one data-collection policy, an exploration anchor and the runtime
  fallback. **It is never a label.** `train_cgan.py` has no reconstruction loss and no warm-up.
* `train_q.py` also trains the state-only ablation **Q(x)**. The *Q gate* allows cGAN training
  only if Q(x,a) beats the constant-mean / prevalence baselines on held-out runs **and** beats Q(x)
  by `q_gate.min_action_gain_rel`. If it doesn't, the data lacks evidence that interventions matter.
* The generator checkpoint is selected by a composite validation score (validity, Q-predicted
  balance distance, diversity, conditional sensitivity, temporal jitter), never by GAN loss.
* Bot archetype names (`player_profile`) are stratification metadata only. They are never a
  model input.
* **Reward/loot adaptation is not learned.** Every controller uses the heuristic loot bias.

### Observed outcome vector (`balance.py`)

| outcome | definition |
|---|---|
| `death` | `died_in_room` (Q predicts a logit; balance uses the probability) |
| `post_hp_ratio` | player HP / max HP after the room (0 on death) |
| `damage_share` | `damage_taken / (damage_taken + damage_dealt)` in [0, 1] (0 when both are 0); a bounded monotone transform r/(1+r) of the taken/dealt ratio |
| `combat_efficiency` | `kills / max(hits, 1)`, 0 when no hit landed |
| `duration_norm` | `time_in_room / 30 s`, clamped to [0, 2] |

Balance loss: `sum_i w_i * (d_i / s_i)^2 / sum_i w_i` with
`d_i = max(low_i - y_i, 0) + max(y_i - high_i, 0)`. It is zero inside every interval and smooth
outside. Intervals, scales and weights live in `experiment_config.json`.
**The current values are PILOT placeholders.** Calibrate them from Stage A, then set
`status` to `FROZEN` and bump `experiment_config_version` before the principal collection.

## Active pipeline

| file | role |
|---|---|
| `dda_core.py` | gameplay constants, condition vector, `rule_based_dda`, action bounds + `finalize_action`, boss modifiers, `config_hash` |
| `telemetry.py` | `RoomSample` schema (`room_samples_v3.0`), condition/sample builders, manifests, validation |
| `balance.py` + `experiment_config.json` | versioned controllers / outcomes / balance targets / gates / seed blocks; calibration report |
| `controllers.py` | `fixed`, `heuristic`, `exploration`, `cgan` (with recorded heuristic fallback) |
| `bot_runner.py` | **authoritative** headless data generator (DFS backtracking, abstract combat) |
| `ml_data.py` | validated loading, grouped train/val/test split by `run_id`, scalers |
| `models.py` | Q, G, D, losses, generator diagnostics, selection score |
| `train_q.py` | Q(x,a) + Q(x) ablation, per-output metrics, Q gate |
| `train_cgan.py` | **authoritative** outcome-guided cGAN trainer |
| `model_runtime.py` | checkpoint compatibility checks + bounded inference |
| `evaluate_controllers.py` / `analyze_results.py` | equivalent-seed comparison, CIs, tables, plots |
| `dataset_tools.py` | coverage / duplicate / range statistics with stop flags; safe merge |
| `rogue.py` | the game (`ROGUE_CONTROLLER=fixed|heuristic|cgan`); human telemetry uses the same contract |
| `scripts/0*.sh` | resumable stage wrappers (below) |

**Legacy, do not use:** `cGAN.py`, `dummy.py`, `DATA_AUGMENTATION/` (these fail fast;
`ROGUE_ALLOW_LEGACY=1` overrides), `cgan_training_phased.ipynb`, `runs.csv`, `logs.txt`.
The `human_sessions/*.csv` files were recorded under schema `room_samples_v2.1` and are rejected
by the current validator. Record new sessions for analysis.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt   # pygame, torch, numpy, matplotlib, pytest
# CPU-only torch: python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pytest -q
```

## Running

### Stage wrappers (recommended)

Each stage writes `EXP/<stage>/` with `meta.json` (git HEAD, versions, config hashes),
`log.txt` and a `DONE` marker that is written only after validation. A completed stage is
skipped on re-run. A partial stage directory is refused until you move it aside.

```bash
EXP=experiments/$(date +%Y%m%d)_main
scripts/01_smoke.sh $EXP          # compile, pytest, --help, pilot per controller, stats, calibration report
#   -> edit experiment_config.json targets, status FROZEN, bump version, commit
scripts/02_collect.sh $EXP        # seed-block collection (default 9000 runs ~ 30k combat rooms), validate, merge
scripts/03_train_q.sh $EXP        # Q(x,a) + Q(x) for Q_SEEDS; stops if any gate fails
scripts/04_train_cgan.sh $EXP     # bounded sweep (SEEDS x BATCHES x LAMBDAs), composite-score selection
scripts/05_evaluate.sh $EXP       # fixed / heuristic / cgan on reserved evaluation seeds
scripts/06_analyze.sh $EXP        # thesis bundle: summary.md/json, CSV tables, plots, reports, metas
```

Environment knobs are documented at the top of each script (`RUNS`, `MIX`, `WORKERS`,
`Q_SEEDS`, `SEEDS`, `BATCHES`, `LAMBDA_BALANCE`, `LAMBDA_DIVERSITY`, `EPOCHS`, `DEVICE`,
`EVAL_RUNS`, ...). For long remote jobs, run them inside `tmux` and set `PYTHON` if needed.

### Individual commands

```bash
python bot_runner.py --controller exploration --runs 50 --seed 100000 --out pilot.csv --validate
python bot_runner.py --controller "exploration=0.65,heuristic=0.20,fixed=0.15" --runs 9000 \
    --seed 1000000 --workers 6 --out train.csv --validate
python dataset_tools.py stats --csv train.csv --project-rooms 30000
python balance.py calibrate --csv train.csv
python train_q.py --csv train.csv --out-dir q_out
python train_cgan.py --csv train.csv --q-dir q_out --out-dir model_out
python evaluate_controllers.py --controllers fixed,heuristic,cgan --model-dir model_out \
    --runs 300 --seed 9000000 --workers 6 --out-dir experiments/eval --train-csv train.csv --plots
python analyze_results.py --eval-dir experiments/eval --plots

ROGUE_CONTROLLER=cgan ROGUE_MODEL_DIR=model_out python rogue.py   # play against the model
```

Run `i` of a collection uses `run_seed = --seed + i`. The map, bot archetype, combat RNG and
controller RNG are independent streams derived from `run_seed`, so results do not depend on
`--workers`, and the same seed range under two controllers gives paired runs. Seed blocks for
pilot, training and evaluation are reserved in `experiment_config.json`. `evaluate_controllers.py`
refuses evaluation seeds that overlap a training manifest.

## Versioning contract

* Changing any tunable in `dda_core.py`, the bot policy constants or `experiment_config.json`
  changes the manifest hashes. `validate_dataset()` rejects datasets that don't match the
  current code. Regenerate data instead of weakening validation.
* Changing the condition vector, actions, outcomes or columns requires bumping
  `telemetry.SCHEMA_VERSION` (and `DDA_VERSION` if heuristic semantics change) and retraining.
* `model_runtime.py` refuses checkpoints whose schema/DDA/experiment-config versions, dimensions,
  ordered columns, action bounds, recomputed `config_hash` or scalers differ from the current code.
  The cGAN controller then falls back to the heuristic and logs `fallback_used` / `fallback_reason`
  on every affected row. After `max_invalid_outputs` invalid outputs, the model is disabled.

## Known limitations

* The bot resolves combat abstractly (no pygame) and does not equip looted items. Results
  describe the bot environment. Human sessions are analyzed separately (`--source human`).
* Boss rooms are excluded from Q/G training by default. At runtime every controller proposes a
  raw action for boss rooms too, and the boss modifiers are applied on top.
* `node_depth_norm` saturates at depth 10 and `total_kills_norm` at 30 kills.
* Q-predicted balance is a model estimate. Thesis claims must rest on the observed outcomes
  from `evaluate_controllers.py`.
