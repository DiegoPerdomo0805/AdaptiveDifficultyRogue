"""
balance.py
==========
Single definition of (a) the versioned experiment configuration loader,
(b) the OBSERVED per-room outcome vector the outcome model Q predicts, and
(c) the balance loss that scores an outcome vector against configurable
target intervals.

Stdlib-only at import time (torch is imported lazily by
`balance_loss_torch`), so the data-generation path stays dependency-free.

Outcome vector (OUTCOME_COLUMNS, order is part of the checkpoint contract):

  death              died_in_room in {0, 1}. Q predicts a logit; the balance
                     loss sees the probability sigmoid(logit).
  post_hp_ratio      player hp / max hp right after the room (0 on death).
  damage_share       damage_taken / (damage_taken + damage_dealt) in [0, 1]
                     (0 when both are 0). A bounded, monotone transform of the
                     taken/dealt ratio r: share = r / (1 + r). exp_v1.0 used
                     taken / max(dealt, 1) capped at 5, whose heavy tail (death
                     before dealing damage) dominated Q's MSE and test R^2.
  combat_efficiency  kills_in_room / max(hits_in_room, 1) in [0, 1];
                     0 when no hit landed.
  duration_norm      time_in_room / duration_norm_seconds, clamped to
                     [0, duration_norm_cap].

Balance loss: for each output i with interval [low_i, high_i], scale s_i and
weight w_i,

  d_i = max(low_i - y_i, 0) + max(y_i - high_i, 0)
  L_balance = sum_i w_i * (d_i / s_i)^2 / sum_i w_i

which is exactly 0 inside every interval and grows smoothly (C1) outside.
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
BINARY_OUTCOMES = ["death"]


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
    for key in ("experiment_config_version", "controllers", "outcomes", "balance_targets"):
        if key not in cfg:
            raise ExperimentConfigError(f"experiment config is missing {key!r}")
    if cfg["outcomes"].get("columns") != OUTCOME_COLUMNS:
        raise ExperimentConfigError(
            f"experiment config outcomes.columns {cfg['outcomes'].get('columns')} "
            f"!= balance.OUTCOME_COLUMNS {OUTCOME_COLUMNS}")
    for col in OUTCOME_COLUMNS:
        t = cfg["balance_targets"].get(col)
        if t is None:
            raise ExperimentConfigError(f"balance_targets is missing {col!r}")
        if not (t["low"] <= t["high"]) or t["scale"] <= 0 or t["weight"] < 0:
            raise ExperimentConfigError(f"invalid balance target for {col!r}: {t}")
    mix = cfg["controllers"]["exploration"]["mixture"]
    if abs(sum(mix.values()) - 1.0) > 1e-6 or any(v < 0 for v in mix.values()):
        raise ExperimentConfigError(f"exploration mixture must be non-negative and sum to 1: {mix}")


def target_arrays(cfg: Optional[dict] = None):
    """(low, high, scale, weight) lists aligned with OUTCOME_COLUMNS."""
    cfg = cfg or load_experiment_config()
    t = cfg["balance_targets"]
    return ([t[c]["low"] for c in OUTCOME_COLUMNS], [t[c]["high"] for c in OUTCOME_COLUMNS],
            [t[c]["scale"] for c in OUTCOME_COLUMNS], [t[c]["weight"] for c in OUTCOME_COLUMNS])


# ---------------------------------------------------------------------------
# Observed outcomes
# ---------------------------------------------------------------------------

def _f(row, key) -> float:
    return float(row[key]) if isinstance(row, dict) else float(getattr(row, key))


def outcome_vector(row, cfg: Optional[dict] = None) -> List[float]:
    """Observed outcome vector for one COMBAT room (dict row or RoomSample)."""
    oc = (cfg or load_experiment_config())["outcomes"]
    died = 1.0 if _f(row, "died_in_room") >= 0.5 else 0.0
    post_hp = 0.0 if died else min(max(_f(row, "post_hp_ratio"), 0.0), 1.0)
    taken = max(_f(row, "damage_taken_in_room"), 0.0)
    dealt = max(_f(row, "damage_dealt_in_room"), 0.0)
    damage_share = taken / (taken + dealt) if (taken + dealt) > 0 else 0.0
    kills = _f(row, "melee_kills_in_room") + _f(row, "magic_kills_in_room")
    hits = _f(row, "melee_hits_in_room") + _f(row, "magic_hits_in_room")
    efficiency = min(kills / hits, 1.0) if hits > 0 else 0.0
    duration = min(max(_f(row, "time_in_room"), 0.0) / oc["duration_norm_seconds"],
                   oc["duration_norm_cap"])
    return [died, post_hp, damage_share, efficiency, duration]


# ---------------------------------------------------------------------------
# Balance loss
# ---------------------------------------------------------------------------

def balance_terms(y: Sequence[float], cfg: Optional[dict] = None) -> List[float]:
    low, high, scale, weight = target_arrays(cfg)
    out = []
    for i in range(OUTCOME_DIM):
        d = max(low[i] - y[i], 0.0) + max(y[i] - high[i], 0.0)
        out.append(weight[i] * (d / scale[i]) ** 2)
    return out


def balance_distance(y: Sequence[float], cfg: Optional[dict] = None) -> float:
    """Pure-python balance loss of one outcome vector (death as probability)."""
    _, _, _, weight = target_arrays(cfg)
    return sum(balance_terms(y, cfg)) / max(sum(weight), 1e-12)


def in_target(y: Sequence[float], cfg: Optional[dict] = None) -> List[bool]:
    low, high, _, _ = target_arrays(cfg)
    return [low[i] <= y[i] <= high[i] for i in range(OUTCOME_DIM)]


def balance_loss_torch(y, cfg: Optional[dict] = None, reduce: bool = True):
    """Torch version of balance_distance over a (N, OUTCOME_DIM) tensor whose
    death column is already a probability. Differentiable."""
    import torch
    low, high, scale, weight = (torch.tensor(a, dtype=y.dtype, device=y.device)
                                for a in target_arrays(cfg))
    d = torch.relu(low - y) + torch.relu(y - high)
    per_row = ((d / scale) ** 2 * weight).sum(dim=1) / weight.sum().clamp_min(1e-12)
    return per_row.mean() if reduce else per_row


# ---------------------------------------------------------------------------
# Calibration report (Stage A): observed outcome distribution per controller
# ---------------------------------------------------------------------------

def _quantiles(vals: List[float], qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> Dict[str, float]:
    if not vals:
        return {}
    v = sorted(vals)
    out = {}
    for q in qs:
        pos = q * (len(v) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        out[f"q{int(q * 100):02d}"] = v[lo] + (v[hi] - v[lo]) * (pos - lo)
    out["mean"] = sum(v) / len(v)
    return out


def calibration_report(csv_paths: List[str], cfg: Optional[dict] = None,
                       include_boss: bool = False) -> dict:
    cfg = cfg or load_experiment_config()
    groups: Dict[str, List[List[float]]] = {}
    for p in csv_paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r["is_combat_room"] != "1":
                    continue
                if r["room_kind"] == "boss" and not include_boss:
                    continue
                y = outcome_vector(r, cfg)
                groups.setdefault(r["controller_type"], []).append(y)
                groups.setdefault("ALL", []).append(y)
    report = {
        "experiment_config_version": cfg["experiment_config_version"],
        "experiment_config_hash": experiment_config_hash(cfg),
        "status": cfg.get("status", ""),
        "targets": cfg["balance_targets"],
        "by_controller": {},
    }
    for ctrl, ys in sorted(groups.items()):
        entry = {"n_rooms": len(ys), "outcomes": {}}
        for i, col in enumerate(OUTCOME_COLUMNS):
            vals = [y[i] for y in ys]
            frac_in = sum(1 for y in ys if in_target(y, cfg)[i]) / len(ys)
            entry["outcomes"][col] = {"quantiles": _quantiles(vals), "frac_in_target": frac_in}
        entry["mean_balance_distance"] = sum(balance_distance(y, cfg) for y in ys) / len(ys)
        report["by_controller"][ctrl] = entry
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Outcome / balance-target utilities.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="Observed outcome distribution vs. current targets.")
    c.add_argument("--csv", action="append", required=True, help="RoomSample CSV (repeatable).")
    c.add_argument("--out", default=None, help="Write the JSON report here.")
    c.add_argument("--include-boss", action="store_true")
    sub.add_parser("show", help="Print the current experiment config version/hash and targets.")
    args = ap.parse_args(argv)

    cfg = load_experiment_config()
    if args.cmd == "show":
        print(json.dumps({"version": cfg["experiment_config_version"],
                          "hash": experiment_config_hash(cfg), "status": cfg.get("status"),
                          "balance_targets": cfg["balance_targets"]}, indent=2))
        return 0
    report = calibration_report(args.csv, cfg, include_boss=args.include_boss)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print(f"experiment_config {report['experiment_config_version']} "
          f"({report['experiment_config_hash']})")
    for ctrl, e in report["by_controller"].items():
        print(f"[{ctrl}] rooms={e['n_rooms']} mean_balance_distance={e['mean_balance_distance']:.4f}")
        for col, o in e["outcomes"].items():
            q = o["quantiles"]
            print(f"   {col:18s} mean={q['mean']:.3f} q05={q['q05']:.3f} q50={q['q50']:.3f} "
                  f"q95={q['q95']:.3f} in_target={100 * o['frac_in_target']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
