"""
analyze_results.py
==================
Controller comparison from RoomSample CSVs (stdlib; matplotlib optional for
plots). Bot and human data are never pooled: --source bot (default) refuses
rows whose player_profile is "human", --source human requires them.

  python analyze_results.py --eval-dir experiments/eval_X        # eval_*.csv
  python analyze_results.py --csv a.csv --csv b.csv --out-dir results

Outputs (in --out-dir, default = --eval-dir):
  summary.json              everything below, machine-readable
  controller_summary.csv    one row per controller, metric, estimate, 95% CI
  death_by_depth.csv        rooms / deaths / rate per controller x depth
  paired_differences.csv    controller - baseline (heuristic, fixed) on equivalent run_seeds
  summary.md                thesis-ready Markdown tables
  plots/*.png               (with --plots and matplotlib installed)

Confidence intervals: 95% percentile bootstrap that resamples whole RUNS
(rooms within a run are not independent), B=--bootstrap, fixed seed.
Room-level metrics cover non-boss combat rooms; balance distance uses the
OBSERVED outcome vector (death as 0/1) against experiment_config targets.
"""

from typing import Callable, Dict, List
import argparse
import csv
import glob
import json
import math
import os
import random
import sys

import balance

CONTROLLER_ORDER = ["fixed", "heuristic", "exploration", "cgan"]
# Validated categorical slots (light surface), fixed per controller entity.
CONTROLLER_COLORS = {"fixed": "#2a78d6", "heuristic": "#eb6834", "cgan": "#1baf7a",
                     "exploration": "#eda100"}


def load_rows(paths: List[str]) -> List[dict]:
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def group_runs(rows) -> Dict[str, Dict[str, List[dict]]]:
    out: Dict[str, Dict[str, List[dict]]] = {}
    for r in rows:
        out.setdefault(r["controller_type"], {}).setdefault(r["run_id"], []).append(r)
    for runs in out.values():
        for rr in runs.values():
            rr.sort(key=lambda r: int(r["room_seq"]))
    return out


def run_summary(rr: List[dict], cfg) -> dict:
    combat = [r for r in rr if r["is_combat_room"] == "1" and r["room_kind"] != "boss"]
    ys = [balance.outcome_vector(r, cfg) for r in combat]
    died = any(r["died_in_room"] == "1" for r in rr)
    boss = any(r["room_kind"] == "boss" for r in rr)
    win = any(r["room_kind"] == "boss" and r["room_result"] == "cleared" for r in rr)
    s = {"run_seed": rr[0]["run_seed"], "died": float(died), "boss_reached": float(boss),
         "victory": float(win), "combat_rooms": float(len(combat)),
         "final_progress": max(float(r["progress_norm"]) for r in rr),
         "n": len(ys), "fallback_rows": sum(int(r["fallback_used"]) for r in rr)}
    for i, col in enumerate(balance.OUTCOME_COLUMNS):
        s["sum_" + col] = sum(y[i] for y in ys)
        s["intarget_" + col] = sum(1 for y in ys if balance.in_target(y, cfg)[i])
    s["sum_balance"] = sum(balance.balance_distance(y, cfg) for y in ys)
    return s


RUN_METRICS = ["victory", "died", "boss_reached", "combat_rooms", "final_progress"]
PAIRED_METRICS = ["run_victory", "run_died", "run_combat_rooms", "room_mean_balance_distance",
                  "room_mean_post_hp_ratio", "room_mean_death", "room_mean_dmg_ratio",
                  "room_mean_duration_norm"]


def _metric_fns() -> Dict[str, Callable[[List[dict]], float]]:
    fns: Dict[str, Callable[[List[dict]], float]] = {}
    for m in RUN_METRICS:
        fns["run_" + m] = (lambda key: lambda S: sum(s[key] for s in S) / len(S))(m)

    def ratio(num):
        return lambda S: (sum(s[num] for s in S) / max(sum(s["n"] for s in S), 1))
    for col in balance.OUTCOME_COLUMNS:
        fns["room_mean_" + col] = ratio("sum_" + col)
        fns["room_in_target_" + col] = ratio("intarget_" + col)
    fns["room_mean_balance_distance"] = ratio("sum_balance")
    return fns


def bootstrap_ci(S: List[dict], fn, b: int, seed: int):
    est = fn(S)
    if len(S) < 2 or b <= 0:
        return est, float("nan"), float("nan")
    rng = random.Random(seed)
    vals = sorted(fn([S[rng.randrange(len(S))] for _ in range(len(S))]) for _ in range(b))
    return est, vals[int(0.025 * (b - 1))], vals[int(0.975 * (b - 1))]


def analyze(paths: List[str], out_dir: str, source: str = "bot", b: int = 1000, seed: int = 0,
            plots: bool = False, model_diagnostics: dict = None) -> dict:
    cfg = balance.load_experiment_config()
    rows = load_rows(paths)
    human = [r for r in rows if r["player_profile"] == "human"]
    if source == "bot" and human:
        raise SystemExit("human rows found in a bot analysis; analyze human sessions separately "
                         "(--source human)")
    if source == "human" and len(human) != len(rows):
        raise SystemExit("--source human requires only human rows")
    os.makedirs(out_dir, exist_ok=True)
    by_ctrl = group_runs(rows)
    ctrls = [c for c in CONTROLLER_ORDER if c in by_ctrl] + sorted(set(by_ctrl) - set(CONTROLLER_ORDER))
    fns = _metric_fns()

    summary = {"inputs": [os.path.abspath(p) for p in paths], "source": source,
               "experiment_config_version": cfg["experiment_config_version"],
               "experiment_config_hash": balance.experiment_config_hash(cfg),
               "bootstrap": {"B": b, "seed": seed, "unit": "run"},
               "controllers": {}, "death_by_depth": {}, "paired": {}}
    run_sums = {}
    for c in ctrls:
        S = [run_summary(rr, cfg) for rr in by_ctrl[c].values()]
        run_sums[c] = S
        ctrl_rows = [r for rr in by_ctrl[c].values() for r in rr]
        combat = [r for r in ctrl_rows if r["is_combat_room"] == "1"]
        entry = {"n_runs": len(S), "n_rows": len(ctrl_rows), "n_combat_rooms": len(combat),
                 "metrics": {}, "action_source": {}, "fallback_reasons": {},
                 "fallback_rows": sum(int(r["fallback_used"]) for r in ctrl_rows),
                 "checkpoint_ids": sorted(set(r["model_checkpoint_id"] for r in ctrl_rows if r["model_checkpoint_id"])),
                 "actions_at_bound_share": 0.0, "mean_action": {}}
        for name, fn in fns.items():
            est, lo, hi = bootstrap_ci(S, fn, b, seed)
            entry["metrics"][name] = {"estimate": est, "ci_low": lo, "ci_high": hi}
        for r in combat:
            entry["action_source"][r["action_source"]] = entry["action_source"].get(r["action_source"], 0) + 1
        for r in ctrl_rows:
            if r["fallback_used"] == "1":
                entry["fallback_reasons"][r["fallback_reason"]] = entry["fallback_reasons"].get(r["fallback_reason"], 0) + 1
        nb = [r for r in combat if r["room_kind"] != "boss"]
        if nb:
            import dda_core as core
            at_bound = 0
            for r in nb:
                for k, col in zip(core.ACTION_KEYS, ("applied_raw_hp_mult", "applied_raw_dmg_mult",
                                                      "applied_raw_speed_mult", "applied_raw_spawn_count")):
                    lo_, hi_ = core.ACTION_BOUNDS[k]
                    v = float(r[col])
                    at_bound += (abs(v - lo_) < 1e-6) or (abs(v - hi_) < 1e-6)
            entry["actions_at_bound_share"] = at_bound / (4 * len(nb))
            for col in ("applied_raw_hp_mult", "applied_raw_dmg_mult", "applied_raw_speed_mult",
                        "applied_raw_spawn_count"):
                entry["mean_action"][col] = sum(float(r[col]) for r in nb) / len(nb)
        summary["controllers"][c] = entry

        depth: Dict[int, List[int]] = {}
        for r in combat:
            d = int(r["node_depth"])
            depth.setdefault(d, [0, 0])
            depth[d][0] += 1
            depth[d][1] += int(r["died_in_room"])
        summary["death_by_depth"][c] = {d: {"rooms": v[0], "deaths": v[1], "rate": v[1] / v[0]}
                                        for d, v in sorted(depth.items())}

    for base_name in ("heuristic", "fixed"):
        if base_name not in run_sums:
            continue
        base = {s["run_seed"]: s for s in run_sums[base_name]}
        for c in ctrls:
            if c == base_name:
                continue
            pairs = [(s, base[s["run_seed"]]) for s in run_sums[c] if s["run_seed"] in base]
            if len(pairs) < 2:
                continue
            res = {"n_pairs": len(pairs)}
            for name in PAIRED_METRICS:
                fn = fns[name]

                def diff(P, fn=fn):
                    return fn([p[0] for p in P]) - fn([p[1] for p in P])
                est, lo, hi = bootstrap_ci(pairs, diff, b, seed)
                res[name] = {"diff": est, "ci_low": lo, "ci_high": hi}
            summary["paired"].setdefault(f"vs_{base_name}", {})[c] = res

    if model_diagnostics:
        summary["model_diagnostics"] = model_diagnostics

    _write_tables(summary, ctrls, out_dir)
    if plots:
        summary["plots"] = _plots(summary, ctrls, rows, cfg, out_dir)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


def _fmt(m):
    if not math.isfinite(m["ci_low"]):
        return f"{m['estimate']:.3f}"
    return f"{m['estimate']:.3f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}]"


def _write_tables(summary, ctrls, out_dir):
    with open(os.path.join(out_dir, "controller_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller", "metric", "estimate", "ci_low", "ci_high"])
        for c in ctrls:
            for name, m in summary["controllers"][c]["metrics"].items():
                w.writerow([c, name, m["estimate"], m["ci_low"], m["ci_high"]])
    with open(os.path.join(out_dir, "death_by_depth.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller", "node_depth", "rooms", "deaths", "rate"])
        for c in ctrls:
            for d, v in summary["death_by_depth"][c].items():
                w.writerow([c, d, v["rooms"], v["deaths"], v["rate"]])
    with open(os.path.join(out_dir, "paired_differences.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "controller", "metric", "diff", "ci_low", "ci_high", "n_pairs"])
        for comp, per in summary["paired"].items():
            for c, res in per.items():
                for name, m in res.items():
                    if name != "n_pairs":
                        w.writerow([comp, c, name, m["diff"], m["ci_low"], m["ci_high"], res["n_pairs"]])

    L = ["# Controller comparison", "",
         f"Experiment config `{summary['experiment_config_version']}` "
         f"(`{summary['experiment_config_hash']}`), source: {summary['source']}. "
         f"95% CIs: run-level bootstrap, B={summary['bootstrap']['B']}.", "",
         "## Counts, validity and fallbacks", "",
         "| controller | runs | rows | combat rooms | fallback rows | actions at bound | action sources |",
         "|---|---|---|---|---|---|---|"]
    for c in ctrls:
        e = summary["controllers"][c]
        L.append(f"| {c} | {e['n_runs']} | {e['n_rows']} | {e['n_combat_rooms']} | {e['fallback_rows']} | "
                 f"{100 * e['actions_at_bound_share']:.1f}% | "
                 + ", ".join(f"{k}: {v}" for k, v in sorted(e["action_source"].items())) + " |")
    L += ["", "## Run outcomes", "", "| controller | victory | death | boss reached | combat rooms | final progress |",
          "|---|---|---|---|---|---|"]
    for c in ctrls:
        m = summary["controllers"][c]["metrics"]
        L.append(f"| {c} | " + " | ".join(_fmt(m["run_" + k]) for k in RUN_METRICS) + " |")
    L += ["", "## Per-room observed outcomes (non-boss combat rooms)", "",
          "| controller | " + " | ".join(balance.OUTCOME_COLUMNS) + " | balance distance |",
          "|---" * (len(balance.OUTCOME_COLUMNS) + 2) + "|"]
    for c in ctrls:
        m = summary["controllers"][c]["metrics"]
        L.append(f"| {c} | " + " | ".join(_fmt(m["room_mean_" + k]) for k in balance.OUTCOME_COLUMNS)
                 + f" | {_fmt(m['room_mean_balance_distance'])} |")
    L += ["", "## Share of rooms inside each target interval", "",
          "| controller | " + " | ".join(balance.OUTCOME_COLUMNS) + " |",
          "|---" * (len(balance.OUTCOME_COLUMNS) + 1) + "|"]
    for c in ctrls:
        m = summary["controllers"][c]["metrics"]
        L.append(f"| {c} | " + " | ".join(_fmt(m["room_in_target_" + k]) for k in balance.OUTCOME_COLUMNS) + " |")
    for comp, per in summary["paired"].items():
        L += ["", f"## Paired differences {comp.replace('_', ' ')} (controller minus baseline, "
                  f"equivalent run seeds)", "",
              "| controller | pairs | metric | diff [95% CI] |", "|---|---|---|---|"]
        for c, res in per.items():
            for name, m in res.items():
                if name == "n_pairs":
                    continue
                ci = f"[{m['ci_low']:.3f}, {m['ci_high']:.3f}]" if math.isfinite(m["ci_low"]) else ""
                L.append(f"| {c} | {res['n_pairs']} | {name} | {m['diff']:.3f} {ci} |")
    L += ["", "## Death rate by depth (combat rooms)", "", "| controller | depth | rooms | deaths | rate |",
          "|---|---|---|---|---|"]
    for c in ctrls:
        for d, v in summary["death_by_depth"][c].items():
            L.append(f"| {c} | {d} | {v['rooms']} | {v['deaths']} | {v['rate']:.3f} |")
    md = summary.get("model_diagnostics")
    if md:
        L += ["", "## Generator diagnostics (fixed evaluation conditions)", "",
              "| metric | value |", "|---|---|"]
        for k in ("checkpoint_id", "n_conditions", "validity", "saturation", "diversity", "sensitivity",
                  "jitter", "balance_pred"):
            if k in md:
                v = md[k]
                L.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
        for k, v in md.get("balance_ref", {}).items():
            L.append(f"| balance_pred_{k} | {v:.4f} |")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def _plots(summary, ctrls, rows, cfg, out_dir) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots", file=sys.stderr)
        return []
    pdir = os.path.join(out_dir, "plots")
    os.makedirs(pdir, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "axes.axisbelow": True, "font.size": 10})
    made = []

    def bar_ci(metric, title, ylabel, fname):
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for i, c in enumerate(ctrls):
            m = summary["controllers"][c]["metrics"][metric]
            err = None
            if math.isfinite(m["ci_low"]):
                err = [[m["estimate"] - m["ci_low"]], [m["ci_high"] - m["estimate"]]]
            ax.bar(i, m["estimate"], width=0.6, color=CONTROLLER_COLORS.get(c, "#888"),
                   yerr=err, capsize=4, ecolor="#52514e")
        ax.set_xticks(range(len(ctrls)), ctrls)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        p = os.path.join(pdir, fname)
        fig.savefig(p, dpi=160)
        plt.close(fig)
        made.append(p)

    bar_ci("run_victory", "Victory rate (95% CI)", "share of runs", "victory_rate.png")
    bar_ci("room_mean_balance_distance", "Observed balance distance per room (lower is better)",
           "balance loss", "balance_distance.png")
    bar_ci("room_mean_post_hp_ratio", "Post-room HP ratio", "HP / max HP", "post_hp.png")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for c in ctrls:
        dd = summary["death_by_depth"][c]
        xs = [d for d, v in dd.items() if v["rooms"] >= 5]
        ax.plot(xs, [dd[d]["rate"] for d in xs], marker="o", markersize=5, linewidth=2,
                color=CONTROLLER_COLORS.get(c, "#888"), label=c)
    ax.set_title("Death rate per combat room by depth", loc="left")
    ax.set_xlabel("node depth")
    ax.set_ylabel("deaths / rooms")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = os.path.join(pdir, "death_by_depth.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    made.append(p)

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))
    cols = ["applied_raw_hp_mult", "applied_raw_dmg_mult", "applied_raw_speed_mult", "applied_raw_spawn_count"]
    for ax, col in zip(axes, cols):
        for c in ctrls:
            v = [float(r[col]) for r in rows if r["controller_type"] == c and r["is_combat_room"] == "1"
                 and r["room_kind"] != "boss"]
            ax.hist(v, bins=20, histtype="step", linewidth=2, color=CONTROLLER_COLORS.get(c, "#888"), label=c)
        ax.set_title(col.replace("applied_raw_", ""), loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = os.path.join(pdir, "action_distributions.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    made.append(p)
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare controllers from RoomSample CSVs.")
    ap.add_argument("--eval-dir", default=None, help="directory with eval_*.csv (evaluate_controllers.py)")
    ap.add_argument("--csv", action="append", default=[])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--source", choices=["bot", "human"], default="bot")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args(argv)
    paths = list(args.csv)
    diag = None
    if args.eval_dir:
        paths += sorted(glob.glob(os.path.join(args.eval_dir, "eval_*.csv")))
        dpath = os.path.join(args.eval_dir, "model_diagnostics.json")
        if os.path.exists(dpath):
            with open(dpath) as f:
                diag = json.load(f)
    if not paths:
        ap.error("no input CSVs (use --eval-dir or --csv)")
    out_dir = args.out_dir or args.eval_dir or "results"
    s = analyze(paths, out_dir, args.source, args.bootstrap, args.seed, args.plots, diag)
    with open(os.path.join(out_dir, "summary.md")) as f:
        print(f.read())
    print("plots:", s.get("plots", "not requested"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
