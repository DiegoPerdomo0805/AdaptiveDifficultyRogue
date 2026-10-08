"""
balance.py
==========
Single definition of (a) the versioned experiment configuration loader,
(b) the OBSERVED outcome vectors the outcome models predict, with their
physical ranges, and (c) the balance loss that scores an outcome vector
against configurable target intervals.

Two outcome families share one implementation, selected by `family`:

  "enemy"  per ordinary combat room (Q_enemy).    OUTCOME_COLUMNS
           targets: experiment_config.enemy_balance_targets
  "loot"   over the next `loot_horizon.combat_rooms` combat rooms after a
           loot event (Q_loot).                    LOOT_OUTCOME_COLUMNS
           targets: experiment_config.loot_horizon_balance_targets

Enemy outcome vector (order is part of the checkpoint contract):

  death              died_in_room in {0, 1}; Q predicts a logit, the balance
                     loss sees the probability.
  post_hp_ratio      player hp / max hp right after the room (0 on death).
  damage_share       damage_taken / (damage_taken + damage_dealt) in [0, 1],
                     0 when both are 0 (bounded monotone transform r/(1+r) of
                     the taken/dealt ratio).
  combat_efficiency  kills / max(hits, 1) in [0, 1]; 0 when no hit landed.
  duration_norm      min(time_in_room / duration_norm_seconds, 1) in [0, 1]
                     (bounded normalized duration; rooms longer than 30 s saturate).

Loot-horizon outcome vector: death_within, end_hp_ratio, damage_share,
combat_efficiency, duration_norm_per_room (see loot_dataset.py).

Balance loss (experiment_config.balance_loss.shape = "pseudo_huber_capped_scale"):
  below the interval: d = low_i - y,  s = min(scale_i, low_i - physical_low_i)
  above the interval: d = y - high_i, s = min(scale_i, physical_high_i - high_i)
  term_i = w_i * 2 * (sqrt(1 + (d / s)^2) - 1)          (0 inside the interval)
  L = sum_i term_i / sum_i w_i
Pseudo-Huber is quadratic near the interval and grows linearly far away, so
a component that can sit many scales outside (e.g. death up to 1.0) does not
dominate through quadratic growth; capping each side's scale at the distance
to the physical limit lets a side whose bound is close to the limit (e.g.
duration below 0.10) still reach a full scale of deviation. A side whose
bound IS the physical limit (death below 0) cannot be violated. The loss is
exactly 0 inside, C1 at the bounds and strictly increasing outside.

Stdlib-only at import time (torch is imported lazily).
"""

from typing import Dict, List, Optional, Sequence
import argparse
import csv
import hashlib
import json
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(_THIS_DIR, "experiment_config.json")

OUTCOME_COLUMNS = ["death", "post_hp_ratio", "damage_share", "combat_efficiency", "duration_norm"]
OUTCOME_DIM = len(OUTCOME_COLUMNS)
LOOT_OUTCOME_COLUMNS = ["death_within", "end_hp_ratio", "damage_share", "combat_efficiency",
                        "duration_norm_per_room"]
LOOT_OUTCOME_DIM = len(LOOT_OUTCOME_COLUMNS)

FAMILIES = {
    "enemy": {"columns": OUTCOME_COLUMNS, "targets": "enemy_balance_targets"},
    "loot": {"columns": LOOT_OUTCOME_COLUMNS, "targets": "loot_horizon_balance_targets"},
}


class ExperimentConfigError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

_CONFIG_CACHE: Dict[str, dict] = {}


def load_experiment_config(path: Optional[str] = None) -> dict:
    path = os.path.abspath(path or os.environ.get("ROGUE_EXPERIMENT_CONFIG") or DEFAULT_CONFIG_PATH)
    if path not in _CONFIG_CACHE:
        with open(path) as f:
            cfg = json.load(f)
        _validate_config(cfg)
        _CONFIG_CACHE[path] = cfg
    return json.loads(json.dumps(_CONFIG_CACHE[path]))   # defensive deep copy


def experiment_config_hash(cfg: Optional[dict] = None) -> str:
    cfg = cfg if cfg is not None else load_experiment_config()
    blob = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _validate_config(cfg: dict):
    for key in ("experiment_config_version", "enemy_controller", "shared_boss", "reward_policy",
                "outcomes", "enemy_balance_targets", "loot_horizon", "loot_horizon_balance_targets",
                "enemy_action_support", "loot_action_support"):
        if key not in cfg:
            raise ExperimentConfigError(f"experiment config is missing {key!r}")
    if cfg["outcomes"].get("columns") != OUTCOME_COLUMNS:
        raise ExperimentConfigError(f"outcomes.columns must be {OUTCOME_COLUMNS}")
    if cfg["loot_horizon"].get("columns") != LOOT_OUTCOME_COLUMNS:
        raise ExperimentConfigError(f"loot_horizon.columns must be {LOOT_OUTCOME_COLUMNS}")
    blocks = {k: v for k, v in cfg.get("seed_blocks", {}).items() if isinstance(v, list)}
    names = sorted(blocks)
    for i, a_ in enumerate(names):
        lo, hi = blocks[a_]
        if lo > hi:
            raise ExperimentConfigError(f"seed block {a_} is empty")
        for b_ in names[i + 1:]:
            if not (hi < blocks[b_][0] or blocks[b_][1] < lo):
                raise ExperimentConfigError(f"seed blocks {a_} and {b_} overlap")
    for req in ("train_collection", "selection_validation", "final_test"):
        if req not in blocks:
            raise ExperimentConfigError(f"seed_blocks.{req} is required")
    cl = cfg.get("cgan_selection", {}).get("closed_loop")
    if not cl or cl.get("validation_seed_block") != "selection_validation":
        raise ExperimentConfigError("cgan_selection.closed_loop must validate on seed_blocks.selection_validation")
    if cfg.get("balance_loss", {}).get("shape") != "pseudo_huber_capped_scale":
        raise ExperimentConfigError("balance_loss.shape must be 'pseudo_huber_capped_scale'")
    if float(cfg["outcomes"].get("duration_norm_cap", 0)) != 1.0:
        raise ExperimentConfigError("duration_norm is a bounded normalized outcome: duration_norm_cap must be 1.0")
    if int(cfg["loot_horizon"].get("combat_rooms", 0)) != 3:
        raise ExperimentConfigError("loot_horizon.combat_rooms is fixed at 3 by the project scope")
    for fam in FAMILIES.values():
        for col in fam["columns"]:
            t = cfg[fam["targets"]].get(col)
            if t is None:
                raise ExperimentConfigError(f"{fam['targets']} is missing {col!r}")
            if not (t["low"] <= t["high"]) or t["scale"] <= 0 or t["weight"] < 0:
                raise ExperimentConfigError(f"invalid balance target for {col!r}: {t}")
    for sect, key in (("enemy_controller", "exploration"), ("reward_policy", "exploration")):
        mix = cfg[sect][key]["mixture"]
        if abs(sum(mix.values()) - 1.0) > 1e-6 or any(v < 0 for v in mix.values()):
            raise ExperimentConfigError(f"{sect}.{key}.mixture must be non-negative and sum to 1: {mix}")


def output_ranges(family: str = "enemy", cfg: Optional[dict] = None) -> List[tuple]:
    """Physical (low, high) range of each outcome, aligned with the family's
    columns. Every model output is contractually bounded to [0, 1]: binary
    outcomes are probabilities, the others are ratios / shares / normalized
    durations (duration_norm_cap is validated to be 1.0)."""
    return [(0.0, 1.0) for _ in FAMILIES[family]["columns"]]


def target_arrays(cfg: Optional[dict] = None, family: str = "enemy"):
    """(low, high, scale, weight) lists aligned with the family's columns."""
    cfg = cfg or load_experiment_config()
    fam = FAMILIES[family]
    t = cfg[fam["targets"]]
    cols = fam["columns"]
    return ([t[c]["low"] for c in cols], [t[c]["high"] for c in cols],
            [t[c]["scale"] for c in cols], [t[c]["weight"] for c in cols])


# ---------------------------------------------------------------------------
# Observed per-room enemy outcomes
# ---------------------------------------------------------------------------

def _f(row, key) -> float:
    return float(row[key]) if isinstance(row, dict) else float(getattr(row, key))


def damage_share(taken: float, dealt: float) -> float:
    taken, dealt = max(taken, 0.0), max(dealt, 0.0)
    return taken / (taken + dealt) if (taken + dealt) > 0 else 0.0


def combat_efficiency(kills: float, hits: float) -> float:
    return min(kills / hits, 1.0) if hits > 0 else 0.0


def duration_norm(seconds: float, cfg: Optional[dict] = None) -> float:
    oc = (cfg or load_experiment_config())["outcomes"]
    return min(max(seconds, 0.0) / oc["duration_norm_seconds"], oc["duration_norm_cap"])


def outcome_vector(row, cfg: Optional[dict] = None) -> List[float]:
    """Observed outcome vector for one COMBAT room (dict row or RoomSample)."""
    cfg = cfg or load_experiment_config()
    died = 1.0 if _f(row, "died_in_room") >= 0.5 else 0.0
    post_hp = 0.0 if died else min(max(_f(row, "post_hp_ratio"), 0.0), 1.0)
    share = damage_share(_f(row, "damage_taken_in_room"), _f(row, "damage_dealt_in_room"))
    kills = _f(row, "melee_kills_in_room") + _f(row, "magic_kills_in_room")
    hits = _f(row, "melee_hits_in_room") + _f(row, "magic_hits_in_room")
    return [died, post_hp, share, combat_efficiency(kills, hits),
            duration_norm(_f(row, "time_in_room"), cfg)]


# ---------------------------------------------------------------------------
# Balance loss
# ---------------------------------------------------------------------------

def side_scales(cfg: Optional[dict] = None, family: str = "enemy"):
    """(s_below, s_above) per component: the configured scale capped at the
    distance from the interval bound to the physical limit (0 = side cannot
    be violated)."""
    low, high, scale, _ = target_arrays(cfg, family)
    rng = output_ranges(family, cfg)
    s_lo = [min(scale[i], low[i] - rng[i][0]) for i in range(len(low))]
    s_hi = [min(scale[i], rng[i][1] - high[i]) for i in range(len(low))]
    return s_lo, s_hi


def _phuber(u: float) -> float:
    return 2.0 * (math.sqrt(1.0 + u * u) - 1.0)


def balance_terms(y: Sequence[float], cfg: Optional[dict] = None, family: str = "enemy") -> List[float]:
    low, high, _, weight = target_arrays(cfg, family)
    s_lo, s_hi = side_scales(cfg, family)
    out = []
    for i in range(len(low)):
        if y[i] < low[i] and s_lo[i] > 0:
            out.append(weight[i] * _phuber((low[i] - y[i]) / s_lo[i]))
        elif y[i] > high[i] and s_hi[i] > 0:
            out.append(weight[i] * _phuber((y[i] - high[i]) / s_hi[i]))
        else:
            out.append(0.0)
    return out


def balance_distance(y: Sequence[float], cfg: Optional[dict] = None, family: str = "enemy") -> float:
    """Pure-python balance loss of one outcome vector (binary outcomes as probabilities)."""
    _, _, _, weight = target_arrays(cfg, family)
    return sum(balance_terms(y, cfg, family)) / max(sum(weight), 1e-12)


def expected_balance_distance(pred: Sequence[float], cfg: Optional[dict] = None, family: str = "enemy") -> float:
    """Expected balance loss when the binary outcome (index 0) is Bernoulli
    with the predicted probability: p * L(y|1) + (1 - p) * L(y|0). This is
    the model's prediction of the OBSERVED per-event balance distance (which
    sees a 0/1 outcome); plugging p itself into the loss is not, because the
    loss is non-linear in that component."""
    p = min(max(float(pred[0]), 0.0), 1.0)
    one, zero = [1.0] + list(pred[1:]), [0.0] + list(pred[1:])
    return p * balance_distance(one, cfg, family) + (1 - p) * balance_distance(zero, cfg, family)


def in_target(y: Sequence[float], cfg: Optional[dict] = None, family: str = "enemy") -> List[bool]:
    low, high, _, _ = target_arrays(cfg, family)
    return [low[i] <= y[i] <= high[i] for i in range(len(low))]


def balance_loss_torch(y, cfg: Optional[dict] = None, reduce: bool = True, family: str = "enemy"):
    """Differentiable balance loss over (N, D) outcome tensors whose binary
    column is already a probability."""
    import torch
    low, high, _, weight = (torch.tensor(a, dtype=y.dtype, device=y.device)
                            for a in target_arrays(cfg, family))
    s_lo, s_hi = (torch.tensor(a, dtype=y.dtype, device=y.device) for a in side_scales(cfg, family))
    u_lo = torch.where(s_lo > 0, torch.relu(low - y) / s_lo.clamp_min(1e-12), torch.zeros_like(y))
    u_hi = torch.where(s_hi > 0, torch.relu(y - high) / s_hi.clamp_min(1e-12), torch.zeros_like(y))
    u = u_lo + u_hi                       # at most one side is non-zero
    per_row = (2.0 * (torch.sqrt(1.0 + u * u) - 1.0) * weight).sum(dim=1) / weight.sum().clamp_min(1e-12)
    return per_row.mean() if reduce else per_row


# ---------------------------------------------------------------------------
# Calibration report (Stage A)
# ---------------------------------------------------------------------------

def _quantile(v: List[float], q: float) -> float:
    pos = q * (len(v) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def _quantiles(vals: List[float], qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> Dict[str, float]:
    if not vals:
        return {}
    v = sorted(vals)
    out = {f"q{int(q * 100):02d}": _quantile(v, q) for q in qs}
    out["mean"] = sum(v) / len(v)
    return out


def _family_report(groups: Dict[str, List[List[float]]], cfg: dict, family: str) -> dict:
    cols = FAMILIES[family]["columns"]
    out = {}
    for name, ys in sorted(groups.items()):
        entry = {"n": len(ys), "outcomes": {}}
        for i, col in enumerate(cols):
            vals = [y[i] for y in ys]
            entry["outcomes"][col] = {"quantiles": _quantiles(vals),
                                      "frac_in_target": sum(1 for y in ys if in_target(y, cfg, family)[i]) / len(ys)}
        entry["mean_balance_distance"] = sum(balance_distance(y, cfg, family) for y in ys) / len(ys)
        out[name] = entry
    return out


def propose_targets(pooled: List[List[float]], cfg: dict, family: str) -> dict:
    """Pre-specified, controller-agnostic rule (experiment_config.calibration_rule):
    continuous targets = pooled [q_lo, q_hi]; binary death targets keep the
    configured design ceiling. Scale = max(interval width, 0.05)."""
    qlo, qhi = cfg["calibration_rule"]["continuous_quantiles"]
    cols = FAMILIES[family]["columns"]
    cur = cfg[FAMILIES[family]["targets"]]
    out = {}
    for i, col in enumerate(cols):
        if col.startswith("death"):
            out[col] = dict(cur[col])
            continue
        v = sorted(y[i] for y in pooled)
        lo, hi = round(_quantile(v, qlo), 3), round(_quantile(v, qhi), 3)
        out[col] = {"low": lo, "high": hi, "scale": round(max(hi - lo, 0.05), 3),
                    "weight": cur[col]["weight"]}
    return out


def calibration_report(csv_paths: List[str], cfg: Optional[dict] = None,
                       loot_csv_paths: Optional[List[str]] = None) -> dict:
    """Observed outcome distributions per collection policy vs. the current
    targets, plus PROPOSED targets from the pooled data (not applied)."""
    cfg = cfg or load_experiment_config()
    groups: Dict[str, List[List[float]]] = {}
    for p in csv_paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r["is_combat_room"] != "1" or r["room_kind"] == "boss":
                    continue
                y = outcome_vector(r, cfg)
                groups.setdefault("enemy_controller=" + r["enemy_controller"], []).append(y)
                groups.setdefault("ALL", []).append(y)
    report = {
        "experiment_config_version": cfg["experiment_config_version"],
        "experiment_config_hash": experiment_config_hash(cfg),
        "status": cfg.get("status", ""),
        "calibration_rule": cfg["calibration_rule"],
        "enemy": {"current_targets": cfg["enemy_balance_targets"],
                  "by_policy": _family_report(groups, cfg, "enemy"),
                  "proposed_targets": propose_targets(groups.get("ALL", []), cfg, "enemy")
                  if groups.get("ALL") else {}},
    }
    if loot_csv_paths:
        lg: Dict[str, List[List[float]]] = {}
        for p in loot_csv_paths:
            with open(p, newline="") as f:
                for r in csv.DictReader(f):
                    if r["selected_category"] == "none" or int(r["horizon_combat_rooms"]) == 0:
                        continue
                    y = [float(r["h_" + c]) for c in LOOT_OUTCOME_COLUMNS]
                    lg.setdefault("reward_policy=" + r["reward_policy"], []).append(y)
                    lg.setdefault("ALL", []).append(y)
        report["loot"] = {"current_targets": cfg["loot_horizon_balance_targets"],
                          "by_policy": _family_report(lg, cfg, "loot"),
                          "proposed_targets": propose_targets(lg.get("ALL", []), cfg, "loot")
                          if lg.get("ALL") else {}}
    return report


# ---------------------------------------------------------------------------
# Target component-contribution audit (pre-freeze)
# ---------------------------------------------------------------------------

def target_audit(cfg: Optional[dict] = None, family: str = "enemy",
                 vectors: Optional[Dict[str, List[List[float]]]] = None) -> dict:
    """
    Analytic + empirical audit of the balance loss for one family.

    Analytic (per component i, others held at their interval midpoints):
      zero_inside        max loss over a 101-point grid of [low, high] is 0
      positive_below/above, increasing_below/above
                         loss > 0 and strictly increasing as the value moves
                         away from the interval (sampled to the physical bound)
      c1_at_bounds       one-sided derivative at each bound is 0 (smooth join)
      one_scale_loss_*   loss exactly one SIDE scale outside = 2(sqrt2-1) w_i / sum(w)
                         (normalization identity); can_contribute: some side can be violated
      max_loss_below/above  loss at the physical bound (exposes range effects)

    Empirical (optional `vectors`: {label: [outcome vectors]}, binary column as
    a probability or rate): mean contribution share of each component, share
    of vectors where the component is outside its interval, and flags:
      dominant   share > 0.80
      inactive   share < 0.02 and outside-fraction < 0.02
    """
    cfg = cfg or load_experiment_config()
    cols = FAMILIES[family]["columns"]
    low, high, scale, weight = target_arrays(cfg, family)
    ranges = output_ranges(family, cfg)
    wsum = sum(weight)
    mid = [(lo + hi) / 2 for lo, hi in zip(low, high)]
    s_lo, s_hi = side_scales(cfg, family)

    def loss_with(i, v):
        y = list(mid)
        y[i] = v
        return balance_distance(y, cfg, family)

    comps = {}
    for i, col in enumerate(cols):
        lo, hi, s = low[i], high[i], scale[i]
        plo, phi = ranges[i]
        inside = [lo + (hi - lo) * k / 100 for k in range(101)]
        below = [lo - (lo - plo) * k / 50 for k in range(1, 51)] if lo > plo else []
        above = [hi + (phi - hi) * k / 50 for k in range(1, 51)] if hi < phi else []
        lb, la = [loss_with(i, v) for v in below], [loss_with(i, v) for v in above]
        eps = 1e-6
        comps[col] = {
            "interval": [lo, hi], "scale": s, "weight": weight[i], "normalized_weight": weight[i] / wsum,
            "physical_range": list(ranges[i]),
            "zero_inside": max(loss_with(i, v) for v in inside) == 0.0,
            "positive_below": all(x > 0 for x in lb) if lb else None,
            "increasing_below": all(b > a for a, b in zip(lb, lb[1:])) if lb else None,
            "positive_above": all(x > 0 for x in la) if la else None,
            "increasing_above": all(b > a for a, b in zip(la, la[1:])) if la else None,
            "c1_at_bounds": abs(loss_with(i, lo - eps) - loss_with(i, lo)) / eps < 1e-4
                            and abs(loss_with(i, hi + eps) - loss_with(i, hi)) / eps < 1e-4,
            "side_scale_below": s_lo[i], "side_scale_above": s_hi[i],
            "one_scale_loss_below": loss_with(i, lo - s_lo[i]) if s_lo[i] > 0 else None,
            "one_scale_loss_above": loss_with(i, hi + s_hi[i]) if s_hi[i] > 0 else None,
            "one_scale_identity_ok": all(abs(loss_with(i, v) - _phuber(1.0) * weight[i] / wsum) < 1e-9
                                         for v in ([lo - s_lo[i]] if s_lo[i] > 0 else [])
                                         + ([hi + s_hi[i]] if s_hi[i] > 0 else [])),
            "can_contribute": (lo > plo and loss_with(i, plo) > 0) or (hi < phi and loss_with(i, phi) > 0),
            "max_loss_below": loss_with(i, plo) if lo > plo else 0.0,
            "max_loss_above": loss_with(i, phi) if hi < phi else 0.0,
            "lower_bound_active": lo > plo,
            "upper_bound_active": hi < phi,
        }
    out = {"family": family, "targets_section": FAMILIES[family]["targets"], "weight_sum": wsum,
           "components": comps, "empirical": {}}
    for label, ys in (vectors or {}).items():
        if not ys:
            continue
        terms = [balance_terms(y, cfg, family) for y in ys]
        tot = sum(sum(t) for t in terms)
        e = {"n": len(ys), "mean_balance_distance": tot / wsum / len(ys), "components": {}}
        for i, col in enumerate(cols):
            share = (sum(t[i] for t in terms) / tot) if tot > 0 else 0.0
            outside = sum(1 for y in ys if not (low[i] <= y[i] <= high[i])) / len(ys)
            e["components"][col] = {"share": share, "outside_fraction": outside,
                                    "dominant": share > 0.80, "inactive": share < 0.02 and outside < 0.02}
        out["empirical"][label] = e
    # Audit criterion (approved): every component must be analytically able
    # to contribute, correctly scaled (side-scale identity) and monotonically
    # penalized outside its interval. Empirical shares are REPORTED, not gated:
    # components whose violations are rare act as guard rails.
    out["analytic_ok"] = all(c["zero_inside"] and c["c1_at_bounds"] and c["one_scale_identity_ok"]
                             and c["can_contribute"]
                             and c["positive_below"] is not False and c["increasing_below"] is not False
                             and c["positive_above"] is not False and c["increasing_above"] is not False
                             for c in comps.values())
    return out


def _print_family(name: str, rep: dict):
    print(f"== {name} ==")
    for grp, e in rep["by_policy"].items():
        print(f"[{grp}] n={e['n']} mean_balance_distance={e['mean_balance_distance']:.4f}")
        for col, o in e["outcomes"].items():
            q = o["quantiles"]
            print(f"   {col:22s} mean={q['mean']:.3f} q25={q['q25']:.3f} q50={q['q50']:.3f} "
                  f"q75={q['q75']:.3f} in_target={100 * o['frac_in_target']:.1f}%")
    print("proposed targets (NOT applied):")
    for col, t in rep["proposed_targets"].items():
        cur = rep["current_targets"][col]
        print(f"   {col:22s} current [{cur['low']}, {cur['high']}] -> proposed [{t['low']}, {t['high']}] "
              f"scale {t['scale']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Outcome / balance-target utilities.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="Observed outcomes vs. current targets + proposed targets.")
    c.add_argument("--csv", action="append", required=True, help="RoomSample CSV (repeatable).")
    c.add_argument("--loot-csv", action="append", default=[], help="loot_dataset.py output (repeatable).")
    c.add_argument("--out", default=None, help="Write the JSON report here.")
    sub.add_parser("show", help="Print the current experiment config version/hash and targets.")
    au = sub.add_parser("audit", help="Target component-contribution audit (analytic + empirical).")
    au.add_argument("--csv", action="append", default=[], help="room telemetry for empirical shares")
    au.add_argument("--loot-csv", action="append", default=[], help="loot dataset for empirical shares")
    au.add_argument("--q-dir", default=None, help="Q_enemy dir: also audit Q-predicted (probability) vectors")
    au.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_experiment_config()
    if args.cmd == "show":
        print(json.dumps({"version": cfg["experiment_config_version"],
                          "hash": experiment_config_hash(cfg), "status": cfg.get("status"),
                          "enemy_balance_targets": cfg["enemy_balance_targets"],
                          "loot_horizon_balance_targets": cfg["loot_horizon_balance_targets"]}, indent=2))
        return 0
    if args.cmd == "audit":
        return _run_audit(args, cfg)
    report = calibration_report(args.csv, cfg, args.loot_csv)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print(f"experiment_config {report['experiment_config_version']} ({report['experiment_config_hash']})")
    _print_family("enemy (ordinary combat rooms)", report["enemy"])
    if "loot" in report:
        _print_family("loot horizon (next 3 combat rooms)", report["loot"])
    return 0


def _rate_vectors(rows: List[List[float]], groups: List[str]) -> List[List[float]]:
    """Group-level mean outcome vectors (binary column becomes a rate)."""
    by: Dict[str, List[List[float]]] = {}
    for y, g in zip(rows, groups):
        by.setdefault(g, []).append(y)
    return [[sum(v[j] for v in ys) / len(ys) for j in range(len(ys[0]))] for ys in by.values()]


def _run_audit(args, cfg) -> int:
    enemy_vecs, loot_vecs = {}, {}
    rows = []
    for p in args.csv:
        with open(p, newline="") as f:
            rows.extend(r for r in csv.DictReader(f) if r["room_kind"] == "enemy")
    if rows:
        ys = [outcome_vector(r, cfg) for r in rows]
        enemy_vecs["observed per room (0/1 death)"] = ys
        enemy_vecs["observed per run (death rate)"] = _rate_vectors(ys, [r["run_id"] for r in rows])
        enemy_vecs["observed per controller x depth (death rate)"] = _rate_vectors(
            ys, [r["enemy_controller"] + ":" + r["node_depth"] for r in rows])
        if args.q_dir:
            import torch
            import ml_data
            import models
            ck = torch.load(os.path.join(args.q_dir, "q_checkpoint.pt"), map_location="cpu", weights_only=True)
            q = models.freeze(models.OutcomeModel(**ck["q_xa_hparams"]))
            q.load_state_dict(ck["q_xa_state"])
            sc = ml_data.Standardizer.from_json(ck["cond_scaler"])
            x = torch.tensor(sc.transform(ml_data.cond_matrix(rows)), dtype=torch.float32)
            a = torch.tensor([ml_data.normalize_action(v) for v in ml_data.action_matrix(rows)], dtype=torch.float32)
            with torch.no_grad():
                enemy_vecs["Q-predicted per room (P(death))"] = models.outcomes_physical(q(x, a)).tolist()
    for p in args.loot_csv:
        with open(p, newline="") as f:
            ev = [r for r in csv.DictReader(f) if r["selected_category"] != "none" and int(r["horizon_combat_rooms"]) >= 1]
        lys = [[float(r["h_" + c]) for c in LOOT_OUTCOME_COLUMNS] for r in ev]
        loot_vecs["observed per event (0/1 death)"] = lys
        loot_vecs["observed per reward policy x category (death rate)"] = _rate_vectors(
            lys, [r["reward_policy"] + ":" + r["selected_category"] for r in ev])
    rep = {"experiment_config_version": cfg["experiment_config_version"],
           "experiment_config_hash": experiment_config_hash(cfg),
           "enemy": target_audit(cfg, "enemy", enemy_vecs), "loot": target_audit(cfg, "loot", loot_vecs)}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
    for fam in ("enemy", "loot"):
        a = rep[fam]
        print(f"== {fam} ({a['targets_section']}), weight sum {a['weight_sum']}, analytic checks "
              f"{'PASS' if a['analytic_ok'] else 'FAIL'}")
        print(f"   {'component':22s} {'interval':12s} {'s_below':>7s} {'s_above':>7s} {'w/sum':>6s} "
              f"{'maxL<':>7s} {'maxL>':>7s} zero_in C1    identity contrib incr<  incr>")
        for col, c in a["components"].items():
            print(f"   {col:22s} [{c['interval'][0]:.2f},{c['interval'][1]:.2f}]  {c['side_scale_below']:7.3f} "
                  f"{c['side_scale_above']:7.3f} {c['normalized_weight']:6.3f} {c['max_loss_below']:7.3f} "
                  f"{c['max_loss_above']:7.3f} {str(c['zero_inside']):7s} {str(c['c1_at_bounds']):5s} "
                  f"{str(c['one_scale_identity_ok']):8s} {str(c['can_contribute']):7s} "
                  f"{str(c['increasing_below']):6s} {str(c['increasing_above'])}")
        for label, e in a["empirical"].items():
            cells = "  ".join(f"{col}:{v['share']:.2f}/{v['outside_fraction']:.2f}"
                              + ("[DOMINANT]" if v["dominant"] else "") + ("[inactive]" if v["inactive"] else "")
                              for col, v in e["components"].items())
            print(f"   [{label}] n={e['n']} mean L={e['mean_balance_distance']:.4f}\n      share/outside: {cells}")
    return 0 if rep["enemy"]["analytic_ok"] and rep["loot"]["analytic_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
