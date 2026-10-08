"""
analyze_loot.py
===============
DESCRIPTIVE post-hoc analysis of loot events and their next-3-combat-room
outcomes (loot_dataset.py output). Differences between categories are
DESCRIPTIVE ASSOCIATIONS, not causal effects: the selected category is
chosen by a reward policy from the pre-reward state, and the horizon is
shaped by the enemy controller and the run. Only a controlled evaluation
(evaluate_controllers.py --axis reward) compares reward policies causally.

  python analyze_loot.py --loot-csv loot.csv --out-dir loot_results [--q-loot-dir q_loot_out]

Outputs: loot_summary.json, loot_by_category.csv, loot_stratified.csv,
loot_summary.md. 95% CIs are run-clustered bootstrap intervals.

Stratifiers (all pre-reward): pre-reward HP (low < 0.4 <= mid < 0.8 <= high),
progress (early < 0.33 <= mid < 0.66 <= late), recent combat tendency
(cond_melee_ratio >= 0.6 melee / <= 0.4 magic / mixed), prior equipment in the
selected slot (same item re-picked vs. replaced), and reward policy.
"Adjusted" means are direct standardization of each category's horizon mean
to the pooled HP x progress stratum distribution (descriptive adjustment for
two observed confounders only).
"""

from typing import Callable, Dict, List
import argparse
import csv
import json
import math
import os
import random
import sys

import balance
import dda_core as core
import loot_dataset

METRICS = ["survived_horizon", "died_within", "end_hp_ratio", "damage_share", "combat_efficiency",
           "duration_norm_per_room", "cum_duration_norm", "progress_gained", "terminated_early",
           "horizon_balance_distance"]
CATS = list(core.SLOTS) + ["none"]


def hp_bin(r):
    h = float(r["pre_hp_ratio"])
    return "low" if h < 0.4 else ("mid" if h < 0.8 else "high")


def prog_bin(r):
    p = float(r["progress_norm"])
    return "early" if p < 0.33 else ("mid" if p < 0.66 else "late")


def tendency(r):
    m = float(r["cond_melee_ratio"])
    return "melee" if m >= 0.6 else ("magic" if m <= 0.4 else "mixed")


def prior_bin(r):
    if r["selected_category"] == "none":
        return "no_selection"
    return "replaced" if r["replaced_existing"] == "1" else "same_item"


STRATIFIERS = {"pre_hp": hp_bin, "progress": prog_bin, "recent_tendency": tendency,
               "prior_equipment": prior_bin, "reward_policy": lambda r: r["reward_policy"]}


def cluster_ci(rows: List[dict], fn: Callable[[List[dict]], float], b: int, seed: int):
    est = fn(rows)
    runs: Dict[str, List[dict]] = {}
    for r in rows:
        runs.setdefault(r["run_id"], []).append(r)
    keys = sorted(runs)
    if len(keys) < 2 or b <= 0:
        return est, float("nan"), float("nan")
    rng = random.Random(seed)
    vals = sorted(fn([x for _ in keys for x in runs[keys[rng.randrange(len(keys))]]]) for _ in range(b))
    return est, vals[int(0.025 * (b - 1))], vals[int(0.975 * (b - 1))]


def mean_of(col):
    return lambda rs: sum(float(r[col]) for r in rs) / len(rs) if rs else float("nan")


def standardized_mean(rows_cat: List[dict], pooled: List[dict], col: str) -> float:
    """Direct standardization over HP x progress strata (pooled weights);
    strata absent in the category are dropped and weights renormalized."""
    w: Dict[tuple, int] = {}
    for r in pooled:
        k = (hp_bin(r), prog_bin(r))
        w[k] = w.get(k, 0) + 1
    by: Dict[tuple, List[float]] = {}
    for r in rows_cat:
        by.setdefault((hp_bin(r), prog_bin(r)), []).append(float(r[col]))
    num = den = 0.0
    for k, n in w.items():
        if k in by:
            num += n * sum(by[k]) / len(by[k])
            den += n
    return num / den if den else float("nan")


def analyze(paths: List[str], out_dir: str, b: int = 1000, seed: int = 0, q_loot_dir: str = None) -> dict:
    rows = []
    for p in paths:
        loot_dataset.validate_loot_dataset(p)
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    if any(r["player_profile"] == "human" for r in rows) and any(r["player_profile"] != "human" for r in rows):
        raise SystemExit("human and bot loot events must be analyzed separately")
    os.makedirs(out_dir, exist_ok=True)
    observed = [r for r in rows if int(r["horizon_combat_rooms"]) >= 1]
    summary = {"label": "DESCRIPTIVE ASSOCIATIONS (not causal effects)",
               "n_events": len(rows), "n_events_with_horizon": len(observed),
               "bootstrap": {"B": b, "seed": seed, "unit": "run"}, "by_category": {}, "stratified": {},
               "category_distribution_by_policy": {}}

    for pol in sorted({r["reward_policy"] for r in rows}):
        sub = [r for r in rows if r["reward_policy"] == pol]
        counts = {c: sum(1 for r in sub if r["selected_category"] == c) for c in CATS}
        probs = [sum(float(r["reward_prob_" + s]) for r in sub) / len(sub) for s in core.SLOTS]
        ent = -sum(p * math.log(p) for p in probs if p > 0)
        summary["category_distribution_by_policy"][pol] = {"selected_counts": counts,
                                                           "mean_proposed_probs": dict(zip(core.SLOTS, probs)),
                                                           "entropy_of_mean_proposal_nats": ent}

    pred = None
    if q_loot_dir:
        import model_runtime
        qp = model_runtime.QLootPredictor.load(q_loot_dir)
        sel = [r for r in observed if r["selected_category"] != "none"]
        preds = qp.predict_rows(sel) if sel else []
        cfg = balance.load_experiment_config()
        pred = {r["loot_event_id"]: (balance.balance_distance(p, cfg, "loot"), p) for r, p in zip(sel, preds)}
        summary["q_loot_variant"] = qp.variant

    for cat in CATS:
        rs = [r for r in observed if r["selected_category"] == cat]
        if not rs:
            continue
        entry = {"n_events": len(rs), "n_runs": len({r["run_id"] for r in rs}),
                 "items": {}, "metrics": {}, "adjusted": {}}
        for r in rs:
            entry["items"][r["selected_item_id"] or "-"] = entry["items"].get(r["selected_item_id"] or "-", 0) + 1
        for m in METRICS:
            est, lo, hi = cluster_ci(rs, mean_of(m), b, seed)
            entry["metrics"][m] = {"estimate": est, "ci_low": lo, "ci_high": hi}
            entry["adjusted"][m] = standardized_mean(rs, observed, m)
        entry["incomplete_horizon_rate"] = entry["metrics"]["terminated_early"]["estimate"]
        entry["balance_utility"] = -entry["metrics"]["horizon_balance_distance"]["estimate"]
        if pred is not None and cat != "none":
            pv = [pred[r["loot_event_id"]][0] for r in rs if r["loot_event_id"] in pred]
            if pv:
                entry["predicted_utility"] = -sum(pv) / len(pv)
                entry["observed_utility"] = entry["balance_utility"]
        summary["by_category"][cat] = entry

    for name, fn in STRATIFIERS.items():
        tab = {}
        for r in observed:
            key = (fn(r), r["selected_category"])
            tab.setdefault(key, []).append(r)
        summary["stratified"][name] = {f"{k[0]}|{k[1]}": {
            "n": len(v), **{m: mean_of(m)(v) for m in ("died_within", "end_hp_ratio", "damage_share",
                                                      "horizon_balance_distance")}}
            for k, v in sorted(tab.items())}

    with open(os.path.join(out_dir, "loot_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(out_dir, "loot_by_category.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "metric", "estimate", "ci_low", "ci_high", "stratum_adjusted", "n_events"])
        for cat, e in summary["by_category"].items():
            for m, v in e["metrics"].items():
                w.writerow([cat, m, v["estimate"], v["ci_low"], v["ci_high"], e["adjusted"][m], e["n_events"]])
    with open(os.path.join(out_dir, "loot_stratified.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stratifier", "stratum", "category", "n", "died_within", "end_hp_ratio", "damage_share",
                    "horizon_balance_distance"])
        for name, t in summary["stratified"].items():
            for key, v in t.items():
                s, c = key.split("|")
                w.writerow([name, s, c, v["n"], v["died_within"], v["end_hp_ratio"], v["damage_share"],
                            v["horizon_balance_distance"]])
    _write_md(summary, out_dir)
    return summary


def _fmt(m):
    if not math.isfinite(m["ci_low"]):
        return f"{m['estimate']:.3f}"
    return f"{m['estimate']:.3f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}]"


def _write_md(s, out_dir):
    L = ["# Loot events: next-3-combat-room outcomes", "",
         "**Descriptive associations, not causal effects.** Categories are chosen from the pre-reward "
         "state by the reward policy; compare reward policies causally only with the controlled "
         "equivalent-seed evaluation.", "",
         f"Loot events: {s['n_events']} ({s['n_events_with_horizon']} followed by >= 1 combat room). "
         f"95% CIs: run-clustered bootstrap, B={s['bootstrap']['B']}.", "",
         "## Category distribution by reward policy", "",
         "| reward policy | " + " | ".join(CATS) + " | mean proposed probs (w/s/a/b) | entropy (nats) |",
         "|---" * (len(CATS) + 3) + "|"]
    for pol, d in s["category_distribution_by_policy"].items():
        L.append(f"| {pol} | " + " | ".join(str(d["selected_counts"][c]) for c in CATS) + " | "
                 + "/".join(f"{d['mean_proposed_probs'][x]:.2f}" for x in core.SLOTS)
                 + f" | {d['entropy_of_mean_proposal_nats']:.3f} |")
    cols = ["survived_horizon", "died_within", "end_hp_ratio", "damage_share", "combat_efficiency",
            "duration_norm_per_room", "progress_gained", "terminated_early", "horizon_balance_distance"]
    L += ["", "## Horizon outcomes by selected category", "",
          "| category | events | runs | " + " | ".join(cols) + " |", "|---" * (len(cols) + 3) + "|"]
    for cat, e in s["by_category"].items():
        L.append(f"| {cat} | {e['n_events']} | {e['n_runs']} | "
                 + " | ".join(_fmt(e["metrics"][c]) for c in cols) + " |")
    L += ["", "## HP x progress stratum-standardized means (descriptive adjustment)", "",
          "| category | died_within | end_hp_ratio | damage_share | horizon_balance_distance |",
          "|---|---|---|---|---|"]
    for cat, e in s["by_category"].items():
        a = e["adjusted"]
        L.append(f"| {cat} | {a['died_within']:.3f} | {a['end_hp_ratio']:.3f} | {a['damage_share']:.3f} | "
                 f"{a['horizon_balance_distance']:.3f} |")
    L += ["", "## Items represented", "", "| category | items (events) |", "|---|---|"]
    for cat, e in s["by_category"].items():
        L.append(f"| {cat} | " + ", ".join(f"{k}: {v}" for k, v in sorted(e["items"].items())) + " |")
    if any("predicted_utility" in e for e in s["by_category"].values()):
        L += ["", f"## Predicted vs. observed horizon utility (Q_loot variant {s.get('q_loot_variant')})", "",
              "| category | predicted utility | observed utility |", "|---|---|---|"]
        for cat, e in s["by_category"].items():
            if "predicted_utility" in e:
                L.append(f"| {cat} | {e['predicted_utility']:.3f} | {e['observed_utility']:.3f} |")
    for name, t in s["stratified"].items():
        L += ["", f"## Stratified by {name}", "", "| stratum | category | n | died_within | end_hp | damage_share | balance distance |",
              "|---|---|---|---|---|---|---|"]
        for key, v in t.items():
            st, c = key.split("|")
            L.append(f"| {st} | {c} | {v['n']} | {v['died_within']:.3f} | {v['end_hp_ratio']:.3f} | "
                     f"{v['damage_share']:.3f} | {v['horizon_balance_distance']:.3f} |")
    with open(os.path.join(out_dir, "loot_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Descriptive next-3-combat-room loot analysis.")
    ap.add_argument("--loot-csv", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--q-loot-dir", default=None, help="add predicted-vs-observed utility")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    analyze(args.loot_csv, args.out_dir, args.bootstrap, args.seed, args.q_loot_dir)
    with open(os.path.join(args.out_dir, "loot_summary.md")) as f:
        print(f.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
