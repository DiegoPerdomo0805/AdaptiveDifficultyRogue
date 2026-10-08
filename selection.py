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
checkpoint reloads through model_runtime, OOD action rate <= max_ood_rate.
No eligible candidate -> SelectionFailure (no checkpoint is produced).

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
    early = victories = 0
    boss_hp = []
    model_dec = ood = fallbacks = 0
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
        for r in ordinary:
            for k, col in zip(core.ACTION_KEYS, ("applied_hp_mult", "applied_dmg_mult",
                                                  "applied_speed_mult", "applied_spawn_count")):
                v = float(r[col])
                lo, hi = core.ACTION_BOUNDS[k]
                if not (math.isfinite(v) and lo - 1e-9 <= v <= hi + 1e-9):
                    invalid += 1
                    break
            model_dec += 1
            if int(r["fallback_used"]):
                fallbacks += 1
                reason = r["fallback_reason"]
                fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
                if reason.startswith("ood_action:"):
                    ood += 1
            elif r["support_distance"] != "" and float(r["support_distance"]) > 1.0:
                ood += 1
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
        "ood_action_rate": ood / model_dec if model_dec else float("nan"),
        "fallback_rate": fallbacks / model_dec if model_dec else float("nan"),
        "fallback_reasons": fallback_reasons,
        "all_fallbacks_recorded_ood": all(k.startswith("ood_action:") for k in fallback_reasons)
                                      and all(k for k in fallback_reasons),
        "validity": 1.0 - invalid / model_dec if model_dec else float("nan"),
        "B_r": B,
    }
    out["selection_score"] = selection_score(out["mean_B_r"], out["cvar_B_r"], cl["cvar_coef"])
    if lm is not None and lm.q is not None and per_run_rows:
        out["q_diagnostics"] = q_run_level_diagnostics(lm, per_run_rows, per_run_obs, cfg)
    return out


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
    if not (metrics["ood_action_rate"] <= cl["max_ood_rate"]):
        reasons.append(f"OOD action rate {metrics['ood_action_rate']:.4f} > {cl['max_ood_rate']}")
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
