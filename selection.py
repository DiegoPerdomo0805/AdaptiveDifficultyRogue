"""
selection.py
============
Closed-loop cGAN checkpoint selection (experiment_config.cgan_selection.closed_loop).

Final selection among candidate checkpoints uses OBSERVED bot rollouts on the
frozen validation seeds (seed_blocks.selection_validation) -- never Q
predictions. For each candidate:

  B_r              run-level ordinary-combat balance per validation run
                   (analyze_results.run_level_vector + the frozen balance loss)
  selection_score  mean(B_r) + cvar_coef * CVaR_alpha(B_r)        (lower is better)
  CVaR_alpha       mean of the worst ceil((1 - alpha) * n) B_r values

Eligibility (all required): room-level validity / diversity / sensitivity
gates, every validation action valid, observed ordinary-room death rate <=
max_ordinary_death_rate, every fallback recorded and of OOD-guard type, the
checkpoint reloads through model_runtime, and the OOD decomposition limits:

  ood_excess          = cgan_ood_rate - heuristic_same_state_ood_rate <= max_ood_excess
  runtime_guard_rate  (generated actions beyond the 1.5x runtime guard)  <= max_runtime_guard_rate
  fallback_rate                                                          <= max_fallback_rate

The decomposition is computed POST HOC from the logged rollout rows
(same_state_support): for every ordinary-room state the candidate reached, the
support ratio of the candidate's generated action and of the heuristic
(rule_based_dda) counterfactual action for exactly that state, both evaluated
on the same standardized condition tensor. The counterfactual never touches
the rollout or its RNG streams. The absolute cGAN OOD rate is reported as a
diagnostic only. No eligible candidate -> SelectionFailure.

Q-predicted run death rate and B_r are reported per candidate as calibration
DIAGNOSTICS only.
"""

from typing import Dict, List, Optional
import math

import analyze_results
import balance
import bot_runner
import controllers as ctl
import dda_core as core
import rewards


class SelectionFailure(RuntimeError):
    pass


def cvar_upper(values: List[float], alpha: float) -> float:
    """CVaR_alpha of a loss: mean of the worst ceil((1 - alpha) * n) values."""
    if not values:
        raise ValueError("CVaR of an empty sample")
    k = max(1, math.ceil((1.0 - alpha) * len(values) - 1e-9))
    worst = sorted(values, reverse=True)[:k]
    return sum(worst) / k


def selection_score(mean_b: float, cvar_b: float, coef: float) -> float:
    return mean_b + coef * cvar_b


def validation_seeds(cfg: Optional[dict] = None) -> List[int]:
    cfg = cfg or balance.load_experiment_config()
    cl = cfg["cgan_selection"]["closed_loop"]
    lo, hi = cfg["seed_blocks"][cl["validation_seed_block"]]
    seeds = list(range(lo, lo + int(cl["n_runs"])))
    if seeds[-1] > hi:
        raise ValueError("validation n_runs exceeds the selection_validation seed block")
    assert_not_test_seeds(seeds, cfg)
    return seeds


def assert_not_test_seeds(seeds: List[int], cfg: dict):
    """Selection must never touch final-test (or smoke-test / training) seeds."""
    for name in ("final_test", "smoke_test", "train_collection", "pilot"):
        if name not in cfg["seed_blocks"]:
            continue
        a, b = cfg["seed_blocks"][name]
        bad = [s for s in seeds if a <= s <= b]
        if bad:
            raise SelectionFailure(f"selection would use seeds from seed_blocks.{name} (e.g. {bad[0]})")


def rollout(model_dir: str, seeds: List[int], cfg: Optional[dict] = None):
    """Closed-loop rollouts of one candidate: enemy controller = the candidate
    (loaded through model_runtime, i.e. with full compatibility validation),
    reward policy = closed_loop.reward_policy, shared boss. Deterministic in
    (candidate, seeds)."""
    cfg = cfg or balance.load_experiment_config()
    cl = cfg["cgan_selection"]["closed_loop"]
    assert_not_test_seeds(seeds, cfg)
    mr = ctl.load_model_runtime()
    lm = mr.LoadedModels.load(model_dir, load_q=True)          # raises on any incompatibility
    controller = ctl.CGANController(model_dir, cfg=cfg, models=lm)
    policy = rewards.make_reward_policy(cl["reward_policy"], cfg=cfg)
    runs = []
    for s in seeds:
        _, samples, reason, boss, _ = bot_runner.simulate_run(
            s, controller, exp_version=cfg["experiment_config_version"], reward_policy=policy)
        runs.append({"seed": s, "rows": [dict(x.__dict__) for x in samples], "termination": reason,
                     "boss_reached": boss})
    return lm, runs


def _ratio_from_reason(reason: str) -> float:
    """Support ratio of a generated action rejected by the runtime OOD guard,
    as logged in its fallback reason ('ood_action:support_ratio=1.575 > 1.5')."""
    try:
        return float(reason.split("support_ratio=")[1].split()[0])
    except (IndexError, ValueError):
        return float("nan")


def state_tensor(lm, rows: List[dict]):
    """Standardized pre-room condition tensor of the logged rows (THE state)."""
    import torch
    import ml_data
    return torch.tensor(lm.cond_scaler.transform([[float(r[c]) for c in ml_data.COND_COLUMNS] for r in rows]),
                        dtype=torch.float32)


def support_ratios(lm, x, actions: List[List[float]]) -> List[float]:
    """Support ratio (kNN distance / training threshold) of raw actions on states x."""
    import torch
    import ml_data
    a = torch.tensor([ml_data.normalize_action(v) for v in actions], dtype=torch.float32)
    with torch.no_grad():
        return (lm.support.distance(x, a) / lm.support.threshold).tolist()


def same_state_support(lm, rows: List[dict]):
    """(cgan_ratios, heuristic_ratios) for ordinary-room rows decided by the
    candidate, on IDENTICAL states.

    cGAN side: the ratio the runtime OOD guard computed for the generated
    action on that state (logged support_distance; for actions the guard
    rejected, the ratio logged in the fallback reason).
    Heuristic side: rule_based_dda's action for exactly the same pre-room
    state (ml_data.heuristic_action_for_row), evaluated on the state tensor
    built from the same logged condition.
    Pure function of the logged rows: it cannot alter the rollout or RNG."""
    import ml_data
    if not rows:
        return [], []
    x = state_tensor(lm, rows)
    heur_ratio = support_ratios(lm, x, [ml_data.heuristic_action_for_row(r) for r in rows])
    cgan_ratio = []
    for r in rows:
        if int(r["fallback_used"]):
            cgan_ratio.append(_ratio_from_reason(r["fallback_reason"]))
        else:
            cgan_ratio.append(float(r["support_distance"]))
    return cgan_ratio, heur_ratio


def _corr(a: List[float], b: List[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / den if den > 0 else float("nan")


def _quant(v: List[float], q: float) -> float:
    v = sorted(v)
    pos = q * (len(v) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def rollout_metrics(runs: List[dict], cfg: Optional[dict] = None, lm=None) -> dict:
    """Observed validation metrics (+ Q calibration diagnostics if `lm` has Q)."""
    cfg = cfg or balance.load_experiment_config()
    cl = cfg["cgan_selection"]["closed_loop"]
    B, per_run_obs, per_run_rows = [], [], []
    rooms = deaths = d1_rooms = d1_deaths = 0
    early = victories = boss_reached = 0
    boss_hp = []
    model_dec = fallbacks = 0
    decided_rows = []
    fallback_reasons: Dict[str, int] = {}
    invalid = 0
    for run in runs:
        rows = run["rows"]
        ordinary = [r for r in rows if r["room_kind"] == "enemy"]
        rooms += len(ordinary)
        deaths += sum(int(r["died_in_room"]) for r in ordinary)
        d1 = [r for r in ordinary if int(r["node_depth"]) == 1]
        d1_rooms += len(d1)
        d1_deaths += sum(int(r["died_in_room"]) for r in d1)
        if any(int(r["died_in_room"]) for r in ordinary):
            early += 1                                   # run ended by death before the boss
        boss = [r for r in rows if r["room_kind"] == "boss"]
        if boss:
            boss_hp.append(float(boss[0]["pre_hp_ratio"]))
            victories += int(boss[0]["room_result"] == "cleared")
            boss_reached += 1
        for r in ordinary:
            for k, col in zip(core.ACTION_KEYS, ("applied_hp_mult", "applied_dmg_mult",
                                                  "applied_speed_mult", "applied_spawn_count")):
                v = float(r[col])
                lo, hi = core.ACTION_BOUNDS[k]
                if not (math.isfinite(v) and lo - 1e-9 <= v <= hi + 1e-9):
                    invalid += 1
                    break
            model_dec += 1
            decided_rows.append(r)
            if int(r["fallback_used"]):
                fallbacks += 1
                reason = r["fallback_reason"]
                fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
        if ordinary:
            v = analyze_results.run_level_vector(ordinary, cfg)
            B.append(balance.balance_distance(v, cfg))
            per_run_obs.append(v)
            per_run_rows.append(ordinary)
    n_runs = len(runs)
    out = {
        "n_runs": n_runs, "n_runs_with_ordinary_rooms": len(B),
        "mean_B_r": sum(B) / len(B) if B else float("nan"),
        "median_B_r": _quant(B, 0.5) if B else float("nan"),
        "sd_B_r": (math.sqrt(sum((x - sum(B) / len(B)) ** 2 for x in B) / max(len(B) - 1, 1)) if B
                   else float("nan")),
        "iqr_B_r": (_quant(B, 0.75) - _quant(B, 0.25)) if B else float("nan"),
        "cvar_B_r": cvar_upper(B, cl["cvar_alpha"]) if B else float("nan"),
        "ordinary_room_death_rate": deaths / rooms if rooms else float("nan"),
        "depth1_death_rate": d1_deaths / d1_rooms if d1_rooms else float("nan"),
        "early_termination_rate": early / n_runs if n_runs else float("nan"),
        "victory_rate": victories / n_runs if n_runs else float("nan"),
        "hp_entering_boss": sum(boss_hp) / len(boss_hp) if boss_hp else float("nan"),
        "boss_survival": victories / boss_reached if boss_reached else float("nan"),
        "fallback_rate": fallbacks / model_dec if model_dec else float("nan"),
        "fallback_reasons": fallback_reasons,
        "all_fallbacks_recorded_ood": all(k.startswith("ood_action:") for k in fallback_reasons)
                                      and all(k for k in fallback_reasons),
        "validity": 1.0 - invalid / model_dec if model_dec else float("nan"),
        "B_r": B,
    }
    out["selection_score"] = selection_score(out["mean_B_r"], out["cvar_B_r"], cl["cvar_coef"])
    if lm is not None:
        out.update(ood_decomposition(lm, decided_rows, cfg))
    if lm is not None and lm.q is not None and per_run_rows:
        out["q_diagnostics"] = q_run_level_diagnostics(lm, per_run_rows, per_run_obs, cfg)
    return out


def ood_decomposition(lm, decided_rows: List[dict], cfg: dict) -> dict:
    """Absolute cGAN OOD (diagnostic), heuristic same-state OOD, the unclipped
    excess (gate) and the runtime-guard rate (gate)."""
    guard = float(cfg["enemy_action_support"]["runtime_ood_factor"])
    cg, he = same_state_support(lm, decided_rows)
    n = len(cg)
    if n == 0:
        nan = float("nan")
        return {"cgan_ood_rate": nan, "heuristic_same_state_ood_rate": nan, "ood_excess": nan,
                "runtime_guard_rate": nan, "n_decisions": 0}
    cgan_rate = sum(1 for v in cg if v > 1.0) / n
    heur_rate = sum(1 for v in he if v > 1.0) / n
    return {"n_decisions": n, "cgan_ood_rate": cgan_rate, "heuristic_same_state_ood_rate": heur_rate,
            "ood_excess": cgan_rate - heur_rate,                       # NOT clipped: may be negative
            "runtime_guard_rate": sum(1 for v in cg if v > guard) / n}


def q_run_level_diagnostics(lm, per_run_rows, per_run_obs, cfg) -> dict:
    """Q room predictions aggregated per run like the observed run-level metric."""
    import torch
    import ml_data
    import models
    pd_, od_, pb, ob = [], [], [], []
    for rows, obs in zip(per_run_rows, per_run_obs):
        x = torch.tensor(lm.cond_scaler.transform([[float(r[c]) for c in ml_data.COND_COLUMNS] for r in rows]),
                         dtype=torch.float32)
        a = torch.tensor([ml_data.normalize_action([float(r[c]) for c in ml_data.ACTION_COLUMNS]) for r in rows],
                         dtype=torch.float32)
        with torch.no_grad():
            p = models.outcomes_physical(lm.q(x, a)).tolist()
        v = [sum(row[j] for row in p) / len(p) for j in range(5)]
        pd_.append(v[0])
        od_.append(obs[0])
        pb.append(balance.balance_distance(v, cfg))
        ob.append(balance.balance_distance(obs, cfg))
    n = len(pb)

    def block(pred, obs):
        return {"predicted_mean": sum(pred) / n, "observed_mean": sum(obs) / n,
                "bias": sum(a - b for a, b in zip(pred, obs)) / n,
                "mae": sum(abs(a - b) for a, b in zip(pred, obs)) / n, "pearson_r": _corr(pred, obs)}
    return {"n_runs": n, "run_death_rate": block(pd_, od_), "B_r": block(pb, ob)}


def eligibility(metrics: dict, room_diag: dict, epoch: int, cfg: Optional[dict] = None) -> List[str]:
    """Reasons a candidate is ineligible (empty list = eligible)."""
    import models
    cfg = cfg or balance.load_experiment_config()
    sel = cfg["cgan_selection"]
    cl = sel["closed_loop"]
    reasons = []
    if not models.selection_eligible(room_diag, epoch, sel):
        reasons.append("room-level validity/diversity/sensitivity/min_epoch gate failed "
                       f"(validity {room_diag['validity']:.3f}, diversity {room_diag['diversity']:.4f}, "
                       f"sensitivity {room_diag['sensitivity']:.4f}, epoch {epoch})")
    if metrics.get("load_error"):
        reasons.append(f"compatibility validation failed: {metrics['load_error']}")
        return reasons
    if not (metrics["validity"] == 1.0):
        reasons.append(f"invalid validation actions (validity {metrics['validity']:.4f})")
    if not (metrics["ordinary_room_death_rate"] <= cl["max_ordinary_death_rate"]):
        reasons.append(f"ordinary-room death rate {metrics['ordinary_room_death_rate']:.4f} > "
                       f"{cl['max_ordinary_death_rate']}")
    if not metrics["all_fallbacks_recorded_ood"]:
        reasons.append(f"non-OOD or unrecorded fallbacks {metrics['fallback_reasons']}")
    # Absolute cGAN OOD (metrics["cgan_ood_rate"]) is a diagnostic and is NOT gated.
    if not (metrics["ood_excess"] <= cl["max_ood_excess"]):
        reasons.append(f"OOD excess over heuristic on the same states {metrics['ood_excess']:+.4f} > "
                       f"{cl['max_ood_excess']}")
    if not (metrics["runtime_guard_rate"] <= cl["max_runtime_guard_rate"]):
        reasons.append(f"runtime-guard rate {metrics['runtime_guard_rate']:.4f} > {cl['max_runtime_guard_rate']}")
    if not (metrics["fallback_rate"] <= cl["max_fallback_rate"]):
        reasons.append(f"fallback rate {metrics['fallback_rate']:.4f} > {cl['max_fallback_rate']}")
    if not math.isfinite(metrics["selection_score"]):
        reasons.append("non-finite selection score")
    return reasons


def choose(candidates: List[dict]) -> dict:
    """Lowest selection_score among eligible candidates; SelectionFailure if none."""
    eligible = [c for c in candidates if not c["ineligible_reasons"]]
    if not eligible:
        raise SelectionFailure("no eligible checkpoint: " + "; ".join(
            f"epoch {c['epoch']}: {', '.join(c['ineligible_reasons'])}" for c in candidates))
    return min(eligible, key=lambda c: (c["metrics"]["selection_score"], c["epoch"]))
