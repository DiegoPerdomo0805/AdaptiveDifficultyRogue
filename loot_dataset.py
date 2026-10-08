"""
loot_dataset.py
===============
Deterministic derivation of the delayed-outcome LOOT dataset from validated
room telemetry. The room telemetry stays authoritative; this file is a
reproducible view of it (one row per loot event) with its own manifest.

  python loot_dataset.py --csv rooms.csv [--csv more.csv] --out loot.csv

For each loot event (a loot room; selected or not) the horizon is the next
H = experiment_config.loot_horizon.combat_rooms (= 3) COMBAT rooms (enemy or
boss) of the SAME run, in room_seq order. Navigation / empty / loot rooms are
never counted. The horizon stops early at death, at a cleared boss (victory),
or when the run's log ends (exhausted / step cap).

Pre-reward features come ONLY from the loot room's own row (its cond_*
vector, pre-room HP, depth/progress and the equipment held before the
selection). Outcome columns come ONLY from later combat rows of that run.

Horizon outcome columns (h_* feed Q_loot; see balance.LOOT_OUTCOME_COLUMNS):
  horizon_combat_rooms   combat rooms observed (0..3)
  horizon_complete       1 if 3 combat rooms were observed
  survived_horizon       1 if the player did not die within the observed rooms
  died_within            1 if the player died within the horizon
  end_hp_ratio           HP ratio after the last observed room (0 on death; the
                         loot-room HP if no combat room followed)
  cum_damage_taken / cum_damage_dealt
  damage_share           cum taken / (cum taken + cum dealt), 0 if both are 0
  combat_efficiency      cum kills / max(cum hits, 1), 0 with no hits
  cum_duration_norm      sum of per-room duration_norm
  duration_norm_per_room cum_duration_norm / observed rooms (0 if none)
  progress_gained        progress_norm(last observed room) - progress_norm(loot
                         room); both are "rooms finished before the room"
  boss_reached           1 if a boss room is inside the horizon
  terminated_early       1 if the horizon ended before 3 combat rooms
  termination_reason     complete | death | victory | run_end
  horizon_balance_distance  balance.balance_distance(h_*, family="loot")
                            (utility = -distance)
"""

from typing import Dict, List, Optional, Sequence
import argparse
import csv
import json
import os
import sys
import time

import balance
import dda_core as core
import ml_data
import telemetry

LOOT_DATASET_VERSION = "loot_horizon_v1"

STATE_COLUMNS = list(telemetry.COND_COLUMNS) + ["pre_hp_ratio", "node_depth", "progress_norm"]
EQUIP_BEFORE_COLUMNS = ["equip_before_" + s for s in core.SLOTS]
H_COLUMNS = ["h_" + c for c in balance.LOOT_OUTCOME_COLUMNS]

LOOT_DATASET_COLUMNS = [
    "loot_event_id", "run_id", "run_seed", "room_seq", "player_profile",
    "enemy_controller", "reward_policy",
    *STATE_COLUMNS, *EQUIP_BEFORE_COLUMNS,
    "reward_policy_type", "reward_proposal_source", "reward_checkpoint_id", "reward_fallback_used",
    *telemetry.REWARD_PROB_COLUMNS,
    "selection_mechanism", "selected_offer_index", "selected_category", "selected_item_id",
    "replaced_existing", *telemetry.EQUIP_AFTER_COLUMNS,
    "horizon_combat_rooms", "horizon_complete", "survived_horizon", "died_within", "end_hp_ratio",
    "cum_damage_taken", "cum_damage_dealt", "damage_share", "combat_efficiency",
    "cum_duration_norm", "duration_norm_per_room", "progress_gained", "boss_reached",
    "terminated_early", "termination_reason",
    *H_COLUMNS, "horizon_balance_distance",
    "schema_version", "experiment_config_version",
]


def horizon_outcome(loot_row: dict, later_rows: List[dict], horizon: int, cfg: dict) -> dict:
    """Aggregate the next `horizon` combat rooms after a loot event.
    `later_rows` must be the SAME run's rows with room_seq > the loot room's,
    sorted by room_seq."""
    combat = [r for r in later_rows if r["is_combat_room"] == "1"]
    obs: List[dict] = []
    reason = "run_end"
    for r in combat:
        obs.append(r)
        if r["died_in_room"] == "1":
            reason = "death"
            break
        if r["room_kind"] == "boss" and r["room_result"] == "cleared":
            reason = "victory"
            break
        if len(obs) == horizon:
            reason = "complete"
            break
    n = len(obs)
    died = any(r["died_in_room"] == "1" for r in obs)
    taken = sum(float(r["damage_taken_in_room"]) for r in obs)
    dealt = sum(float(r["damage_dealt_in_room"]) for r in obs)
    kills = sum(float(r["melee_kills_in_room"]) + float(r["magic_kills_in_room"]) for r in obs)
    hits = sum(float(r["melee_hits_in_room"]) + float(r["magic_hits_in_room"]) for r in obs)
    durs = [balance.duration_norm(float(r["time_in_room"]), cfg) for r in obs]
    if n:
        end_hp = 0.0 if died else float(obs[-1]["post_hp_ratio"])
        progress_gained = float(obs[-1]["progress_norm"]) - float(loot_row["progress_norm"])
    else:
        end_hp = float(loot_row["post_hp_ratio"])
        progress_gained = 0.0
    out = {
        "horizon_combat_rooms": n,
        "horizon_complete": int(n == horizon),
        "survived_horizon": int(not died),
        "died_within": int(died),
        "end_hp_ratio": round(end_hp, 6),
        "cum_damage_taken": round(taken, 6), "cum_damage_dealt": round(dealt, 6),
        "damage_share": round(balance.damage_share(taken, dealt), 6),
        "combat_efficiency": round(balance.combat_efficiency(kills, hits), 6),
        "cum_duration_norm": round(sum(durs), 6),
        "duration_norm_per_room": round(sum(durs) / n, 6) if n else 0.0,
        "progress_gained": round(progress_gained, 6),
        "boss_reached": int(any(r["room_kind"] == "boss" for r in obs)),
        "terminated_early": int(n < horizon),
        # death/victory take precedence even in the 3rd room; run_end = log ended
        "termination_reason": reason,
    }
    h = [float(out["died_within"]), out["end_hp_ratio"], out["damage_share"],
         out["combat_efficiency"], out["duration_norm_per_room"]]
    for c, v in zip(H_COLUMNS, h):
        out[c] = v
    out["horizon_balance_distance"] = round(balance.balance_distance(h, cfg, "loot"), 6)
    return out


def derive_from_rows(rows: List[dict], cfg: Optional[dict] = None) -> List[dict]:
    """All loot events of `rows` (any number of runs), deterministically ordered."""
    cfg = cfg or balance.load_experiment_config()
    horizon = int(cfg["loot_horizon"]["combat_rooms"])
    runs: Dict[str, List[dict]] = {}
    for r in rows:
        runs.setdefault(r["run_id"], []).append(r)
    out = []
    for run_id in sorted(runs):
        rr = sorted(runs[run_id], key=lambda r: int(r["room_seq"]))
        for i, r in enumerate(rr):
            if not r["loot_event_id"]:
                continue
            ev = {c: r[c] for c in ("loot_event_id", "run_id", "run_seed", "room_seq", "player_profile",
                                    "enemy_controller", "reward_policy", *STATE_COLUMNS,
                                    "reward_policy_type", "reward_proposal_source", "reward_checkpoint_id",
                                    "reward_fallback_used", *telemetry.REWARD_PROB_COLUMNS,
                                    "selection_mechanism", "selected_offer_index", "selected_category",
                                    "selected_item_id", "replaced_existing", *telemetry.EQUIP_AFTER_COLUMNS,
                                    "schema_version", "experiment_config_version")}
            for s in core.SLOTS:
                ev["equip_before_" + s] = r["equip_" + s]
            ev.update(horizon_outcome(r, rr[i + 1:], horizon, cfg))
            out.append(ev)
    return out


def derive(csv_paths: Sequence[str], out_csv: str, overwrite: bool = False) -> dict:
    if os.path.exists(out_csv) and not overwrite:
        raise FileExistsError(f"{out_csv} exists")
    cfg = balance.load_experiment_config()
    sources, rows, ref_hash = [], [], None
    for p in csv_paths:
        mpath = ml_data.manifest_path_for(p)
        rep = telemetry.validate_dataset(p, mpath)
        man = telemetry.load_manifest(mpath)
        if ref_hash is None:
            ref_hash = man["config_hash"]
        elif man["config_hash"] != ref_hash:
            raise telemetry.DatasetVersionError(f"{p}: config_hash differs from the first dataset")
        sources.append({"csv": os.path.abspath(p), "sha256": ml_data.file_sha256(p),
                        "config_hash": man["config_hash"], "n_rows": rep["n_rows"],
                        "seed_range": man.get("seed_range")})
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    keys = [(r["run_id"], r["room_seq"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise telemetry.DatasetVersionError("duplicate (run_id, room_seq) across source datasets")
    events = derive_from_rows(rows, cfg)
    tmp = out_csv + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOOT_DATASET_COLUMNS)
        w.writeheader()
        for ev in events:
            w.writerow(ev)
    os.replace(tmp, out_csv)
    manifest = {
        "loot_dataset_version": LOOT_DATASET_VERSION,
        "schema_version": telemetry.SCHEMA_VERSION,
        "experiment_config_version": cfg["experiment_config_version"],
        "experiment_config_hash": balance.experiment_config_hash(cfg),
        "horizon_combat_rooms": int(cfg["loot_horizon"]["combat_rooms"]),
        "config_hash": ref_hash,
        "derived_from": sources,
        "columns": LOOT_DATASET_COLUMNS,
        "n_rows": len(events),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": telemetry.git_commit_short(),
    }
    with open(ml_data.manifest_path_for(out_csv), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def validate_loot_dataset(csv_path: str, check_sources: bool = True) -> dict:
    """Checks the derived dataset against its manifest, the current code and
    (optionally) that every source CSV is unchanged since derivation."""
    man = telemetry.load_manifest(ml_data.manifest_path_for(csv_path))
    cfg = balance.load_experiment_config()
    expect = {"loot_dataset_version": LOOT_DATASET_VERSION, "schema_version": telemetry.SCHEMA_VERSION,
              "experiment_config_version": cfg["experiment_config_version"],
              "experiment_config_hash": balance.experiment_config_hash(cfg),
              "columns": LOOT_DATASET_COLUMNS}
    for k, v in expect.items():
        if man.get(k) != v:
            raise telemetry.DatasetVersionError(f"loot manifest {k}={man.get(k)!r} != current {v!r}")
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        if next(reader, None) != LOOT_DATASET_COLUMNS:
            raise telemetry.DatasetVersionError("loot dataset header mismatch")
        n = sum(1 for _ in reader)
    if n != man["n_rows"]:
        raise telemetry.DatasetVersionError(f"loot manifest declares {man['n_rows']} rows, CSV has {n}")
    if check_sources:
        for s in man["derived_from"]:
            if not os.path.exists(s["csv"]) or ml_data.file_sha256(s["csv"]) != s["sha256"]:
                raise telemetry.DatasetVersionError(f"source {s['csv']} changed or missing since derivation")
    return {"ok": True, "n_rows": n, "manifest": man}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Derive the next-3-combat-room loot outcome dataset.")
    ap.add_argument("--csv", action="append", required=True, help="room telemetry CSV (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    man = derive(args.csv, args.out, args.overwrite)
    rep = validate_loot_dataset(args.out)
    with open(args.out, newline="") as f:
        rows = list(csv.DictReader(f))
    by = {}
    for r in rows:
        by.setdefault(r["termination_reason"], 0)
        by[r["termination_reason"]] += 1
    cats = {}
    for r in rows:
        cats[r["selected_category"]] = cats.get(r["selected_category"], 0) + 1
    print(f"wrote {man['n_rows']} loot events -> {args.out} (validated: {rep['ok']})")
    print("selected categories:", dict(sorted(cats.items())))
    print("horizon termination:", dict(sorted(by.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
