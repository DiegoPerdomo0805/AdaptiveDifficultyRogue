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
  duration_norm      time_in_room / duration_norm_seconds, in [0, cap].

Loot-horizon outcome vector: death_within, end_hp_ratio, damage_share,
combat_efficiency, duration_norm_per_room (see loot_dataset.py).

Balance loss: d_i = max(low_i - y_i, 0) + max(y_i - high_i, 0) and
  L = sum_i w_i * (d_i / s_i)^2 / sum_i w_i
which is exactly 0 inside every interval and grows smoothly outside.

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
    columns. Binary outcomes are probabilities in [0, 1]."""
    cfg = cfg or load_experiment_config()
    cap = float(cfg["outcomes"]["duration_norm_cap"])
    return [(0.0, cap) if c.startswith("duration") else (0.0, 1.0) for c in FAMILIES[family]["columns"]]


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

def balance_terms(y: Sequence[float], cfg: Optional[dict] = None, family: str = "enemy") -> List[float]:
    low, high, scale, weight = target_arrays(cfg, family)
    out = []
    for i in range(len(low)):
        d = max(low[i] - y[i], 0.0) + max(y[i] - high[i], 0.0)
        out.append(weight[i] * (d / scale[i]) ** 2)
    return out


def balance_distance(y: Sequence[float], cfg: Optional[dict] = None, family: str = "enemy") -> float:
    """Pure-python balance loss of one outcome vector (binary outcomes as probabilities)."""
    _, _, _, weight = target_arrays(cfg, family)
    return sum(balance_terms(y, cfg, family)) / max(sum(weight), 1e-12)


def in_target(y: Sequence[float], cfg: Optional[dict] = None, family: str = "enemy") -> List[bool]:
    low, high, _, _ = target_arrays(cfg, family)
    return [low[i] <= y[i] <= high[i] for i in range(len(low))]


def balance_loss_torch(y, cfg: Optional[dict] = None, reduce: bool = True, family: str = "enemy"):
    """Differentiable balance loss over (N, D) outcome tensors whose binary
    column is already a probability."""
    import torch
    low, high, scale, weight = (torch.tensor(a, dtype=y.dtype, device=y.device)
                                for a in target_arrays(cfg, family))
    d = torch.relu(low - y) + torch.relu(y - high)
    per_row = ((d / scale) ** 2 * weight).sum(dim=1) / weight.sum().clamp_min(1e-12)
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
    args = ap.parse_args(argv)

    cfg = load_experiment_config()
    if args.cmd == "show":
        print(json.dumps({"version": cfg["experiment_config_version"],
                          "hash": experiment_config_hash(cfg), "status": cfg.get("status"),
                          "enemy_balance_targets": cfg["enemy_balance_targets"],
                          "loot_horizon_balance_targets": cfg["loot_horizon_balance_targets"]}, indent=2))
        return 0
    report = calibration_report(args.csv, cfg, args.loot_csv)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print(f"experiment_config {report['experiment_config_version']} ({report['experiment_config_hash']})")
    _print_family("enemy (ordinary combat rooms)", report["enemy"])
    if "loot" in report:
        _print_family("loot horizon (next 3 combat rooms)", report["loot"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
