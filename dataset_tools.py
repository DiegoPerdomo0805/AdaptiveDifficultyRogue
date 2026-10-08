"""
dataset_tools.py
================
Dataset inspection and safe merging (stdlib only).

  python dataset_tools.py stats --csv pilot.csv [--csv ...] [--out stats.json]
         [--project-rooms 30000]
  python dataset_tools.py merge --csv a.csv --csv b.csv --out merged.csv

`stats` reports runs/rooms, controller and action-source distribution,
deaths/victories/boss reach, condition-feature ranges and zero-variance
columns, action coverage (range, spread, histogram occupancy, distance and
correlation to the heuristic reference action), duplicate keys, invalid rows,
throughput and projected time/disk for a target number of combat rooms, and
a list of STOP-CONDITION flags. It exits non-zero when any flag is raised.

`merge` concatenates datasets only if their manifests agree on every version
string, the experiment config and config_hash, and no (run_id, room_seq)
key repeats. The merged manifest lists its sources.
"""

from typing import Dict, List
import argparse
import csv
import json
import math
import os
import sys

import balance
import dda_core as core
import ml_data
import telemetry


def _mean(v):
    return sum(v) / len(v) if v else float("nan")


def _std(v):
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def _corr(a, b):
    sa, sb = _std(a), _std(b)
    if sa == 0 or sb == 0:
        return float("nan")
    ma, mb = _mean(a), _mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (len(a) * sa * sb)


def dataset_stats(csv_paths: List[str], project_rooms: int = 0) -> dict:
    rows = []
    elapsed = 0.0
    n_runs_manifest = 0
    bytes_total = 0
    for p in csv_paths:
        mpath = ml_data.manifest_path_for(p)
        if os.path.exists(mpath):
            man = telemetry.load_manifest(mpath)
            elapsed += float(man.get("elapsed_sec", 0.0) or 0.0)
            n_runs_manifest += int((man.get("run_stats") or {}).get("runs", 0))
        bytes_total += os.path.getsize(p)
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))

    flags: List[str] = []
    keys = [(r["run_id"], r["room_seq"]) for r in rows]
    dup = len(keys) - len(set(keys))
    if dup:
        flags.append(f"{dup} duplicate (run_id, room_seq) keys")

    runs: Dict[str, List[dict]] = {}
    for r in rows:
        runs.setdefault(r["run_id"], []).append(r)
    run_ctrl: Dict[str, int] = {}
    run_rew: Dict[str, int] = {}
    victories = deaths = boss_reach = 0
    for rid, rr in runs.items():
        rr.sort(key=lambda r: int(r["room_seq"]))
        run_ctrl[rr[0]["enemy_controller"]] = run_ctrl.get(rr[0]["enemy_controller"], 0) + 1
        run_rew[rr[0]["reward_policy"]] = run_rew.get(rr[0]["reward_policy"], 0) + 1
        if any(r["died_in_room"] == "1" for r in rr):
            deaths += 1
        if any(r["room_kind"] == "boss" for r in rr):
            boss_reach += 1
            if any(r["room_kind"] == "boss" and r["room_result"] == "cleared" for r in rr):
                victories += 1

    combat = [r for r in rows if r["room_kind"] == "enemy"]     # ordinary combat rooms
    out = {
        "files": csv_paths, "n_rows": len(rows), "n_runs": len(runs),
        "n_combat_rooms_nonboss": len(combat),
        "n_boss_rooms": sum(1 for r in rows if r["room_kind"] == "boss"),
        "runs_by_controller": run_ctrl, "runs_by_reward_policy": run_rew,
        "rows_by_controller": {}, "combat_rooms_by_action_source": {},
        "run_outcomes": {"deaths": deaths, "victories": victories, "boss_reached": boss_reach,
                         "death_rate": deaths / max(len(runs), 1),
                         "victory_rate": victories / max(len(runs), 1)},
        "room_death_rate": _mean([float(r["died_in_room"]) for r in combat]) if combat else float("nan"),
        "duplicate_keys": dup,
    }
    for r in rows:
        out["rows_by_controller"][r["enemy_controller"]] = out["rows_by_controller"].get(r["enemy_controller"], 0) + 1
    for r in combat:
        out["combat_rooms_by_action_source"][r["action_source"]] = \
            out["combat_rooms_by_action_source"].get(r["action_source"], 0) + 1

    feats = {}
    for c in ml_data.COND_COLUMNS:
        v = [float(r[c]) for r in combat]
        feats[c] = {"min": min(v), "max": max(v), "mean": _mean(v), "std": _std(v)} if v else {}
        if v and _std(v) < 1e-9:
            flags.append(f"zero-variance condition column {c}")
    out["condition_features"] = feats

    invalid = 0
    for r in rows:
        for k, c in zip(core.ACTION_KEYS, ml_data.ACTION_RAW_COLUMNS):
            v = float(r[c])
            lo, hi = core.ACTION_BOUNDS[k]
            if not math.isfinite(v) or v < lo - 1e-9 or v > hi + 1e-9:
                invalid += 1
                break
    out["invalid_action_rows"] = invalid
    if invalid:
        flags.append(f"{invalid} rows with non-finite/out-of-bounds raw actions")

    cov = {}
    for ctrl in sorted(set(r["controller_type"] for r in combat)):
        sub = [r for r in combat if r["controller_type"] == ctrl]   # ordinary rooms: == enemy_controller
        acts = ml_data.action_matrix(sub)
        heur = [ml_data.heuristic_action_for_row(r) for r in sub]
        per = {}
        for i, col in enumerate(ml_data.ACTION_COLUMNS):
            v = [a[i] for a in acts]
            h = [a[i] for a in heur]
            vn = [ml_data.normalize_action(a)[i] for a in acts]
            bins = set(min(int(x * 10), 9) for x in vn)
            per[col] = {"min": min(v), "max": max(v), "mean": _mean(v), "std_normalized": _std(vn),
                        "hist_bins_occupied_of_10": len(bins),
                        "mean_abs_diff_from_heuristic": _mean([abs(x - y) for x, y in zip(v, h)]),
                        "corr_with_heuristic": _corr(v, h)}
        cov[ctrl] = per
        if ctrl == "exploration":
            for col, s in per.items():
                if s["std_normalized"] < 0.05:
                    flags.append(f"exploration action {col} barely varies (std_normalized {s['std_normalized']:.3f})")
    out["action_coverage"] = cov

    loot = [r for r in rows if r["loot_event_id"]]
    out["loot"] = {"events": len(loot),
                   "selected_categories": {c: sum(1 for r in loot if r["selected_category"] == c)
                                           for c in list(core.SLOTS) + ["none"]},
                   "selected_items": {i: sum(1 for r in loot if r["selected_item_id"] == i)
                                      for i in core.ALL_ITEM_IDS},
                   "reward_fallback_events": sum(1 for r in loot if r["reward_fallback_used"] == "1")}
    out["boss_rows_shared_policy"] = all(r["controller_type"] == "shared_boss"
                                         for r in rows if r["room_kind"] == "boss")
    if not out["boss_rows_shared_policy"]:
        flags.append("boss rows not decided by the shared boss policy")
    if loot and any(v == 0 for c, v in out["loot"]["selected_categories"].items() if c != "none"):
        flags.append("a loot category was never selected")
    if runs and (out["run_outcomes"]["death_rate"] > 0.95 or out["run_outcomes"]["victory_rate"] > 0.95):
        flags.append("run outcomes collapsed (almost every run dies or wins)")

    n_runs = n_runs_manifest or len(runs)
    out["throughput"] = {"elapsed_sec": elapsed, "sec_per_run": elapsed / max(n_runs, 1),
                         "combat_rooms_per_run": len(combat) / max(len(runs), 1),
                         "bytes_per_row": bytes_total / max(len(rows), 1)}
    if project_rooms:
        cpr = max(out["throughput"]["combat_rooms_per_run"], 1e-9)
        runs_needed = math.ceil(project_rooms / cpr)
        out["projection"] = {"target_combat_rooms": project_rooms, "runs_needed": runs_needed,
                             "single_worker_sec": runs_needed * out["throughput"]["sec_per_run"],
                             "approx_disk_mb": runs_needed * len(rows) / max(len(runs), 1)
                             * out["throughput"]["bytes_per_row"] / 1e6}
    out["stop_flags"] = flags
    return out


def merge_datasets(csv_paths: List[str], out_csv: str) -> dict:
    mans = [telemetry.load_manifest(ml_data.manifest_path_for(p)) for p in csv_paths]
    for p in csv_paths:
        telemetry.validate_dataset(p, ml_data.manifest_path_for(p))
    keys = ("schema_version", "game_version", "generator_version", "bot_policy_version",
            "dda_version", "experiment_config_version", "experiment_config_hash", "config_hash")
    for m, p in zip(mans[1:], csv_paths[1:]):
        for k in keys:
            if m.get(k) != mans[0].get(k):
                raise telemetry.DatasetVersionError(
                    f"cannot merge {p}: {k}={m.get(k)!r} != {mans[0].get(k)!r}")
    n = telemetry.merge_shards(csv_paths, out_csv)
    with open(out_csv, newline="") as f:
        reader = csv.DictReader(f)
        seen = set()
        for r in reader:
            k = (r["run_id"], r["room_seq"])
            if k in seen:
                os.remove(out_csv)
                raise telemetry.DatasetVersionError(f"duplicate key {k} across merged datasets")
            seen.add(k)
    extra = {k: v for k, v in (mans[0].get("extra") or {}).items() if k != "experiment_config_hash"}
    counts: Dict[str, int] = {}
    for m in mans:
        for c, v in (m.get("controller_counts") or {}).items():
            counts[c] = counts.get(c, 0) + v
    mpath = ml_data.manifest_path_for(out_csv)
    man = telemetry.write_manifest(mpath, n_rows=n,
                                   seed_range=[m.get("seed_range") for m in mans],
                                   extra=extra, controller_counts=counts)
    man["merged_from"] = [{"csv": os.path.abspath(p), "sha256": ml_data.file_sha256(p),
                           "seed_range": m.get("seed_range")} for p, m in zip(csv_paths, mans)]
    man["elapsed_sec"] = sum(float(m.get("elapsed_sec", 0) or 0) for m in mans)
    man["run_stats"] = {"runs": sum(int((m.get("run_stats") or {}).get("runs", 0)) for m in mans)}
    with open(mpath, "w") as f:
        json.dump(man, f, indent=2)
    telemetry.validate_dataset(out_csv, mpath)
    return {"n_rows": n, "manifest": mpath}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dataset statistics and safe merging.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--csv", action="append", required=True)
    s.add_argument("--out", default=None)
    s.add_argument("--project-rooms", type=int, default=0)
    m = sub.add_parser("merge")
    m.add_argument("--csv", action="append", required=True)
    m.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "merge":
        if os.path.exists(args.out):
            ap.error(f"{args.out} exists")
        print(merge_datasets(args.csv, args.out))
        return 0
    st = dataset_stats(args.csv, args.project_rooms)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(st, f, indent=2, default=float)
    print(f"rows={st['n_rows']} runs={st['n_runs']} combat(non-boss)={st['n_combat_rooms_nonboss']} "
          f"boss_rooms={st['n_boss_rooms']}")
    print("runs by enemy controller:", st["runs_by_controller"], "| by reward policy:",
          st["runs_by_reward_policy"])
    print("loot:", st["loot"]["events"], "events, categories", st["loot"]["selected_categories"],
          "| boss rows shared policy:", st["boss_rows_shared_policy"])
    print("combat rooms by action source:", st["combat_rooms_by_action_source"])
    print("run outcomes:", {k: round(v, 3) if isinstance(v, float) else v
                            for k, v in st["run_outcomes"].items()},
          f"room death rate {st['room_death_rate']:.3f}")
    for c, f in st["condition_features"].items():
        print(f"  {c:24s} min {f['min']:.3f} max {f['max']:.3f} std {f['std']:.3f}")
    for ctrl, per in st["action_coverage"].items():
        print(f"  action coverage [{ctrl}]")
        for col, a in per.items():
            print(f"    {col:20s} [{a['min']:.2f}, {a['max']:.2f}] std_n {a['std_normalized']:.3f} "
                  f"bins {a['hist_bins_occupied_of_10']}/10 |a-heur| {a['mean_abs_diff_from_heuristic']:.3f} "
                  f"corr {a['corr_with_heuristic']:.3f}")
    print("throughput:", {k: round(v, 4) for k, v in st["throughput"].items()})
    if "projection" in st:
        print("projection:", {k: round(v, 2) if isinstance(v, float) else v for k, v in st["projection"].items()})
    print("STOP FLAGS:", st["stop_flags"] or "none")
    return 1 if st["stop_flags"] else 0


if __name__ == "__main__":
    sys.exit(main())
