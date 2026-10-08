# Adaptive Difficulty Rogue: outcome-guided DDA vs. heuristic

A pygame top-down roguelike used as a testbed for **dynamic difficulty adjustment (DDA)**.

## Frozen project scope

**Included**
- Outcome-guided adaptation of enemy HP, damage, speed and spawn count.
- Outcome-guided adaptation of the **loot category** (weapon, spell, armor, boots).
- Fixed item definitions and fixed item statistics (already-declared item effects made functional,
  e.g. boots' dash distance).
- Evaluating a selected reward over the **next three combat rooms** (or until the run ends).
- A **shared deterministic boss policy** used by every experimental controller.

**Excluded**
- Literal enemy-size or hitbox adaptation.
- Learned modification or generation of individual item statistics.
- New item categories or additional gameplay systems.

## Decisions and their inputs

Before each room the game builds the 9-dimensional pre-room condition

```text
x = [melee_ratio, magic_ratio, hits_per_kill_melee, hits_per_kill_magic,
     damage_taken_to_dealt_ratio, pre_room_hp_ratio, cumulative_kills_normalized,
     node_depth_normalized, run_progress_normalized]
```

Then one of two things happens:

* **Ordinary combat room:** the run's enemy controller chooses `a = [hp_mult, dmg_mult, speed_mult, spawn_count]`.
* **Loot room:** the run's reward policy proposes category probabilities `[P_weapon, P_spell, P_armor, P_boots]`.
  Three offers are drawn from them, each an item drawn uniformly from the category's fixed pool. The bot or the
  human then selects one offer, and it is equipped.

Boss rooms always use `experiment_config.shared_boss`, and telemetry marks them `controller_type = shared_boss`.
Enemy-free rooms log a placeholder action. No enemy model is called in boss rooms or enemy-free rooms.

| controller (`--enemy-controller`) | role |
|---|---|
| `fixed` | **fixed_default**: hp/dmg/speed multipliers 1.0, spawn_count 2 (`DDA_SPAWN_BASE`); no adaptation |
| `heuristic` | `rule_based_dda` (baseline, exploration anchor, runtime fallback; **never a label**) |
| `exploration` | seeded bounded variation for data collection: perturb / uniform / reference |
| `cgan` | outcome-guided generator with a logged heuristic fallback |

| reward policy (`--reward-policy`) | role |
|---|---|
| `fixed_uniform` | equal category probabilities |
| `heuristic` | documented category preference (`dda_core.loot_bias_for_metrics`) |
| `exploration` | seeded category coverage for data collection: focus / uniform / heuristic |
| `learned_loot` | `G_loot`. It is trained **only if the Q_loot gate passes**; otherwise it falls back (logged) |

## Outcome-guided learning (not imitation)

```text
Enemy:  Q(x, a) -> o        trained only on observed (x, a, o); bounded outputs
        G(x, z) -> a        inside the observed action-support box
        D(x, a) -> logit    observed vs. generated
        L_G = L_adv + λ_bal·L_balance(Q_frozen(x,G)) + λ_div·L_div + λ_sup·L_support

Loot:   Q_loot(x, c, item, prior_equipment) -> next-3-combat-room outcome profile
        G_loot(x, prior_equipment, z) -> category probabilities (floor + support mask)
        L = E[Σ_c p_c · U_c] − λ_ent·H(p) + λ_div·L_div        (U_c = Q_loot horizon balance loss)
```

* Q outputs are **bounded by architecture**. The binary outcome is a logit; continuous outcomes are
  `lo + (hi − lo)·sigmoid`, so predictions can never leave their physical range.
* **Action-support / OOD protection for G.** Outputs are confined to the observed action-quantile box.
  A kNN support penalty applies beyond the 95th-percentile training distance. The runtime guard rejects
  actions beyond 1.5× that threshold and falls back to the heuristic, logged as `ood_action`.
* **Gates.**
  * `train_q.py`: Q(x,a) must beat the naive baselines and the state-only Q(x).
  * `train_q_loot.py`: a reward-aware Q_loot must beat its **information-matched** state-only twin
    (`x+c` vs `x`, `x+c+item+prior` vs `x+prior`). That means a mean gain ≥ `min_reward_gain_rel`
    with a positive gain on **every** grouped split. If the loot gate fails, `learned_loot` is not
    trained, and only the descriptive loot analysis is reported.
* Model selection uses validation diagnostics: validity, Q-predicted balance, OOD rate, diversity,
  sensitivity and jitter, with eligibility rules. It never uses GAN loss.
* Archetype names (`player_profile`) are stratification metadata only. They are never a model input.

### Outcomes (`balance.py`)

Enemy outcomes, per ordinary combat room:

| outcome | definition |
|---|---|
| `death` | died in the room |
| `post_hp_ratio` | HP after the room (0 on death) |
| `damage_share` | taken / (taken + dealt) |
| `combat_efficiency` | kills / max(hits, 1) |
| `duration_norm` | min(time / 30 s, 1): a **bounded** normalized duration in [0, 1] |

Loot horizon (`loot_dataset.py`): the **next 3 combat rooms** after a loot event in the same run.
Non-combat rooms never count. The horizon stops early at death, at victory, or when the run's log ends.

| loot outcome | definition |
|---|---|
| `death_within` | died within the horizon |
| `end_hp_ratio` | HP after the last observed room |
| `damage_share` | as above, over the horizon |
| `combat_efficiency` | as above, over the horizon |
| `duration_norm_per_room` | mean per-room duration |

The derived CSV also records progress gained, boss reached, incomplete horizon and termination reason.
Balance loss = `Σ wᵢ(dᵢ/sᵢ)² / Σ wᵢ`, which is zero inside each target interval. Targets live in
`experiment_config.json` (`enemy_balance_targets`, `loot_horizon_balance_targets`).
Every Q_enemy / Q_loot output is contractually bounded to [0, 1] (probabilities, shares, ratios and
normalized durations), and the networks enforce that bound by construction.

The ordinary-room targets are **design-intent** values: death [0, 0.08], post_hp [0.35, 0.85],
damage_share [0.09, 0.31], combat_efficiency [0.15, 0.50], duration_norm [0.10, 0.50]. They are
**not** pilot quantiles. `balance.py audit` checks them component by component: zero loss inside,
smooth increasing loss outside, the normalization identity, and empirical contribution shares.
Boss outcomes are not part of the ordinary-room utility.

## Equipment audit (human game vs. bot)

| category | items: fixed stats | human equips by | affects combat in the game | bot (v3) |
|---|---|---|---|---|
| weapon | Sword dmg 14, cd 1.00, heal 1.00; Hatchet 20, 1.12, 1.05; Rapier 10, 0.80, 0.95 | loot UI (E at pedestal, arrows, Enter) | dmg per melee hit; melee cooldown `0.45·cd`; heal on kill `int(8·heal_mult)` | same dmg; exchange time `0.35 s·cd`; same heal (shared function) |
| spell | Magic Bolt dmg 10, cd 1.00, ammo 12; Magic Spike 8, 0.78, 16; Hex Bomb 30, 1.25, 9 | loot UI (ammo refilled on pick) | projectile dmg; magic cooldown `0.60·cd`; ammo per room (refilled on entry) | same dmg; exchange time `0.35 s·cd`; ammo per room, melee once empty |
| armor | Cloth Wrap absorb 0.5, heal +0; Chain Shirt 2.0, +0; Blood Harness 1.0, +6 | loot UI | flat absorb per hit before the defend multiplier; heal bonus on kill | identical (`dda_core.incoming_damage`, `heal_on_kill_amount`) |
| boots | Leather 1.00; Sprint Greaves 1.25; Voidstep Treads 1.45 (`dash_dist_mult`) | loot UI | dash distance `600·0.12·mult` = 72 / 90 / 104 px (speed scaled, `DASH_TIME` and `DASH_CD` unchanged). Before game 0.6.0 the stat was displayed but never applied (fixed defect) | the bot dashes with archetype `p_dash` under `DASH_CD`; a dash spends the exchange moving and evades the incoming ram with p = min(1, dash distance / enemy ram distance) |

Both the game and the bot call the same `dda_core` item functions. The starting loadout is one
uniform item per slot, derived from `run_seed` in both.

The bot's abstract resolver uses these documented approximations:
- One enemy engages at a time.
- Melee hits a single target.
- A cooldown scales the exchange duration and the enemy's per-exchange hit chance.
- Hit and damage variation is ±15%.
- A dash's avoidance value is its distance relative to the enemy ram's reach
  (`ENEMY_SPEED·speed_mult·ENEMY_DASH_SPEED_MULT·ENEMY_DASH_TIME`).

`tests/test_equipment.py` checks the following:
- Representative items change the corresponding bot variables.
- Each boot produces its declared dash distance in the game logic.
- The bot consumes the same `Boots` definition, with no bot-only statistic.
- Better boots raise evasions and lower damage taken in the bot.

## Active pipeline

| file | role |
|---|---|
| `dda_core.py` | constants, items/equipment and item effects, condition vector, `rule_based_dda`, action bounds, boss modifiers, `config_hash` |
| `telemetry.py` | `RoomSample` schema `room_samples_v4.0` (incl. loot events), manifests, validation |
| `balance.py` + `experiment_config.json` | versioned controllers, reward policies, shared boss, targets, support limits, gates, calibration |
| `controllers.py` / `rewards.py` | enemy controllers + shared boss dispatch / reward policies |
| `bot_runner.py` | **authoritative** headless generator (`--enemy-controller`, `--reward-policy`, mixtures) |
| `loot_dataset.py` | deterministic next-3-combat-room loot dataset + manifest (room telemetry stays authoritative) |
| `ml_data.py`, `models.py` | data plumbing; Q / G / D / support / Q_loot / G_loot |
| `train_q.py`, `train_cgan.py` | enemy Q gate and outcome-guided cGAN |
| `train_q_loot.py`, `train_loot_policy.py` | Q_loot ablations + gate, gated learned-loot policy |
| `model_runtime.py` | compatibility checks, bounded inference, OOD guard, loot loaders |
| `evaluate_controllers.py` | equivalent-seed evaluation, `--axis enemy` or `--axis reward` |
| `analyze_results.py`, `analyze_loot.py`, `dataset_tools.py` | comparisons (ordinary / boss-only / end-to-end), descriptive loot analysis, coverage stats |
| `rogue.py` | the game (`ROGUE_CONTROLLER`, `ROGUE_REWARD_POLICY`); human telemetry uses the same contract |

**Legacy (do not use):** `cGAN.py`, `dummy.py`, `DATA_AUGMENTATION/` (these fail fast),
`cgan_training_phased.ipynb`, `runs.csv`, `logs.txt`. The `human_sessions/*.csv` files use
schema v2.1 and are rejected; record new sessions.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
```

## Running

```bash
EXP=experiments/$(date +%Y%m%d)_main
scripts/01_smoke.sh $EXP          # compile, pytest, pilot (3 controllers × 3 collection reward policies), loot dataset, audit
#   -> freeze: status FROZEN + commit; python freeze_experiment.py record; commit experiment_freeze.json
scripts/02_collect.sh $EXP        # seed blocks: MIX (enemy) × REWARD_MIX (reward) -> train_data.csv + train_loot.csv
scripts/03_train_q.sh $EXP        # Q_enemy seeds (must pass) + Q_loot ablations/gate (result recorded)
scripts/04_train_cgan.sh $EXP     # cGAN sweep; learned_loot only if the loot gate passed
scripts/05_evaluate.sh $EXP       # enemy axis (fixed/heuristic/cgan) and reward axis (fixed_uniform/heuristic[/learned_loot])
scripts/06_analyze.sh $EXP        # results bundle
```

Individual tools:

```bash
python bot_runner.py --enemy-controller "exploration=0.65,heuristic=0.20,fixed=0.15" \
    --reward-policy "exploration=0.60,heuristic=0.20,fixed_uniform=0.20" --runs 9000 --seed 1000000 \
    --workers 6 --out train.csv --validate
python loot_dataset.py --csv train.csv --out train_loot.csv
python balance.py calibrate --csv train.csv --loot-csv train_loot.csv
python train_q.py --csv train.csv --out-dir q_out
python train_cgan.py --csv train.csv --q-dir q_out --out-dir model_out
python train_q_loot.py --loot-csv train_loot.csv --out-dir q_loot_out
python train_loot_policy.py --loot-csv train_loot.csv --q-loot-dir q_loot_out --out-dir loot_model_out   # refuses unless gate passed
python analyze_loot.py --loot-csv train_loot.csv --q-loot-dir q_loot_out --out-dir loot_results
python evaluate_controllers.py --axis enemy --controllers fixed,heuristic,cgan --model-dir model_out \
    --runs 300 --seed 9000000 --out-dir experiments/eval_enemy --train-csv train.csv --plots
python evaluate_controllers.py --axis reward --reward-policies fixed_uniform,heuristic --q-loot-dir q_loot_out \
    --runs 300 --seed 9000000 --out-dir experiments/eval_reward --train-csv train.csv --plots

ROGUE_CONTROLLER=cgan ROGUE_MODEL_DIR=model_out ROGUE_REWARD_POLICY=heuristic python rogue.py
```

Run `i` uses `run_seed = --seed + i`. The map, archetype, starting loadout, combat, loot and policy RNGs
are independent streams derived from it. Results therefore don't depend on `--workers`, and equal seeds
are paired across conditions.

## Freeze

`freeze_experiment.py record` writes `experiment_freeze.json`. It holds:
- the commit at which the config was frozen;
- the schema, DDA, bot-policy and game versions;
- the config hashes;
- the targets and weights, `fixed_default`, the shared boss policy and the collection mixtures;
- the enemy and loot gates (with split seeds), the action-support limits and the 3-room loot horizon.

`scripts/02_collect.sh` runs `freeze_experiment.py verify`. It refuses to collect if any of these
values, or any contract file, changed after the freeze. `learned_loot` is never used during
collection; it is trained afterwards only if the predefined Q_loot gate passes on the principal data.

## Versioning contract

* Changing any `dda_core` tunable (including item stats), the bot policy constants or
  `experiment_config.json` changes the manifest hashes. Old datasets and checkpoints are rejected;
  regenerate them instead of weakening validation.
* Changing the condition vector, actions, outcomes or columns requires bumping `telemetry.SCHEMA_VERSION`.
* `model_runtime.py` rejects any checkpoint whose schema, versions, dimensions, ordered columns, action
  bounds, recomputed `config_hash`, experiment-config hash, scalers or support state differ from the
  current code. Every fallback is logged per row (`fallback_reason`, `reward_fallback_reason`).

## Known limitations

* Bot combat is an abstract resolver. Results describe the bot environment; human sessions are
  analyzed separately.
* The bot's dash is an avoidance approximation (it has no positions); the human game applies the
  exact dash distance.
* `node_depth_norm` saturates at depth 10, and `total_kills_norm` at 30 kills.
* Q-predicted balance and utility are model estimates. Claims must rest on observed outcomes
  from equivalent-seed evaluation.
* Loot results from logged data are **descriptive associations**. Only the reward-axis evaluation
  compares reward policies under controlled conditions.
