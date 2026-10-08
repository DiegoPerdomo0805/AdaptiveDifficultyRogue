"""
train_q_loot.py
===============
Loot outcome evaluator Q_loot, trained ONLY on observed delayed-outcome
tuples from loot_dataset.py (selected events with >= 1 observed combat room),
with grouped train/val/test splits by complete run_id.

  python train_q_loot.py --loot-csv loot.csv --out-dir q_loot_out

Variants (ml_data.LOOT_VARIANTS), all with the same bounded architecture:
  x     Q_loot(x)                      state only (no reward information)
  xe    Q_loot(x, prior_equipment)     state + prior equipment (no reward information)
  xc    Q_loot(x, c)                   + selected category
  xcie  Q_loot(x, c, item, prior)      + selected item and prior equipment
plus the naive constant/prevalence predictor (skill 0 by definition).

LEARNED-LOOT GATE (experiment_config.q_loot_gate), evaluated on TEST runs of
several grouped splits (q_loot_gate.split_seeds): reward information must
improve prediction beyond state alone. Each reward-aware variant is compared
with its information-matched twin (xc vs x, xcie vs xe) so prior equipment
cannot masquerade as reward information. A variant passes only if its MEAN
gain is >= min_reward_gain_rel, the gain is positive on EVERY split, and it
beats the naive predictor on every split. (A single-split rule flipped
between pass and fail on two pilot datasets, so the decision is required to
be robust to the split.) The full variant is "supported" only if every item
id has at least loot_action_support.min_category_events / 3 training events.

If the gate fails, learned_loot must not be trained (train_loot_policy.py
refuses); the descriptive analysis (analyze_loot.py) remains valid.
"""

from typing import Dict, List
import argparse
import copy
import json
import os
import random
import sys
import time

import balance
import dda_core as core
import loot_dataset
import ml_data
import telemetry
from train_q import aggregate_skill, evaluate_predictions

Q_LOOT_CHECKPOINT_SCHEMA = "q_loot_v1"
COLS = balance.LOOT_OUTCOME_COLUMNS


def load_events(paths: List[str]):
    rows, mans = [], []
    for p in paths:
        rep = loot_dataset.validate_loot_dataset(p)
        mans.append(rep["manifest"])
        import csv
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    hashes = {m["config_hash"] for m in mans}
    if len(hashes) != 1:
        raise telemetry.DatasetVersionError(f"loot datasets disagree on config_hash: {hashes}")
    usable = [r for r in rows if r["selected_category"] != "none" and int(r["horizon_combat_rooms"]) >= 1]
    fp = {"config_hash": hashes.pop(), "loot_dataset_version": mans[0]["loot_dataset_version"],
          "sources": [{"csv": os.path.abspath(p), "sha256": ml_data.file_sha256(p)} for p in paths],
          "n_events_total": len(rows), "n_events_usable": len(usable)}
    import hashlib
    fp["dataset_id"] = hashlib.sha256(json.dumps(sorted(s["sha256"] for s in fp["sources"])).encode()).hexdigest()[:16]
    return usable, fp


def build(rows, variant, cond_scaler, torch, device):
    cond = cond_scaler.transform([[float(r[c]) for c in ml_data.COND_COLUMNS] for r in rows])
    feats = [ml_data.loot_features(cs, ml_data.loot_row_equip_before(r), r["selected_category"],
                                   r["selected_item_id"], variant) for cs, r in zip(cond, rows)]
    y = [[float(r["h_" + c]) for c in COLS] for r in rows]
    return (torch.tensor(feats, dtype=torch.float32, device=device),
            torch.tensor([v[0] for v in y], dtype=torch.float32, device=device),
            torch.tensor([v[1:] for v in y], dtype=torch.float32, device=device), y)


def fit(net, tr, va, cont_var, args, torch, models):
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x, yb, yc, _ = tr
    xv, ybv, ycv, _ = va
    best = (float("inf"), None, -1)
    bad, hist = 0, []
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(x.shape[0], device=x.device)
        for s in range(0, x.shape[0], args.batch_size):
            idx = perm[s:s + args.batch_size]
            loss, _, _ = models.outcome_loss(net(x[idx]), yb[idx], yc[idx], cont_var)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(models.outcome_loss(net(xv), ybv, ycv, cont_var)[0])
        hist.append({"epoch": epoch, "val_loss": vl})
        if vl < best[0] - 1e-6:
            best, bad = (vl, copy.deepcopy(net.state_dict()), epoch), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(best[1])
    return {"best_val_loss": best[0], "best_epoch": best[2], "history": hist}


def prediction_ranges(net, variant, rows, cond_scaler, torch, models, n_states=60, n_loadouts=16) -> dict:
    """Min/max of every output over held-out states x every (category, item)
    choice x random prior loadouts; must lie inside the physical ranges."""
    import random as _r
    rng = _r.Random(0)
    cond = cond_scaler.transform([[float(r[c]) for c in ml_data.COND_COLUMNS] for r in rows[:n_states]])
    feats = []
    for cs in cond:
        for _ in range(n_loadouts):
            eq = {s: rng.choice(core.ITEMS_BY_SLOT[s]).item_id for s in core.SLOTS}
            for cat in core.SLOTS:
                for it in core.ITEMS_BY_SLOT[cat]:
                    feats.append(ml_data.loot_features(cs, eq, cat, it.item_id, variant))
    net.eval()
    with torch.no_grad():
        y = models.outcomes_physical(net(torch.tensor(feats, dtype=torch.float32)))
    rngs = balance.output_ranges("loot")
    return {c: {"min": float(y[:, j].min()), "max": float(y[:, j].max()), "physical_range": list(rngs[j]),
                "inside_physical_range": bool(y[:, j].min() >= rngs[j][0] - 1e-6 and y[:, j].max() <= rngs[j][1] + 1e-6)}
            for j, c in enumerate(COLS)}


def fit_split(rows, split_seed, args, cfg, torch, models, device) -> dict:
    """Fit every variant on one grouped split; return metrics + states."""
    split = ml_data.grouped_split(rows, tuple(float(v) for v in args.split.split(",")), seed=split_seed)
    tr_rows, va_rows, te_rows = (ml_data.select(rows, split[k]) for k in ("train", "val", "test"))
    if min(len(tr_rows), len(va_rows), len(te_rows)) < 20:
        raise SystemExit(f"too few loot events per split: {len(tr_rows)}/{len(va_rows)}/{len(te_rows)}")
    cond_scaler = ml_data.Standardizer.fit([[float(r[c]) for c in ml_data.COND_COLUMNS] for r in tr_rows],
                                           ml_data.COND_COLUMNS)
    y_tr = [[float(r["h_" + c]) for c in COLS] for r in tr_rows]
    prev = sum(v[0] for v in y_tr) / len(y_tr)
    mean = [sum(v[j] for v in y_tr) / len(y_tr) for j in range(1, len(COLS))]
    var = [max(sum((v[j] - mean[j - 1]) ** 2 for v in y_tr) / len(y_tr), 1e-6) for j in range(1, len(COLS))]
    cont_var = torch.tensor(var, dtype=torch.float32, device=device)
    ranges = [list(r) for r in balance.output_ranges("loot")[1:]]

    support = {"events_per_category": {}, "events_per_item": {}}
    for r in tr_rows:
        for key, val in (("events_per_category", r["selected_category"]), ("events_per_item", r["selected_item_id"])):
            support[key][val] = support[key].get(val, 0) + 1
    min_ev = cfg["loot_action_support"]["min_category_events"]
    item_supported = all(support["events_per_item"].get(i, 0) >= min_ev / 3 for i in core.ALL_ITEM_IDS)
    cat_allowed = [support["events_per_category"].get(c, 0) >= min_ev for c in core.SLOTS]

    out = {"split_seed": split_seed, "split": split, "cond_scaler": cond_scaler, "var": var,
           "n_events": {"train": len(tr_rows), "val": len(va_rows), "test": len(te_rows)},
           "n_runs": {k: len(v) for k, v in split.items()}, "support": support,
           "item_supported": item_supported, "cat_allowed": cat_allowed, "prev": prev,
           "variants": {}, "states": {}, "hparams": {}}
    for variant in ml_data.LOOT_VARIANTS:
        torch.manual_seed(args.seed)
        net = models.BoundedOutcomeNet(ml_data.loot_feature_dim(variant), ranges, args.hidden,
                                       args.layers, args.dropout).to(device)
        tr, va, te = (build(r, variant, cond_scaler, torch, device) for r in (tr_rows, va_rows, te_rows))
        res = fit(net, tr, va, cont_var, args, torch, models)
        entry = {"best_epoch": res["best_epoch"], "best_val_loss": res["best_val_loss"]}
        for name, part in (("val", va), ("test", te)):
            net.eval()
            with torch.no_grad():
                pred = models.outcomes_physical(net(part[0])).cpu().tolist()
            m = evaluate_predictions(pred, part[3], prev, mean, COLS)
            entry[name] = {"metrics": m, "skill": aggregate_skill(m, COLS)}
        entry["prediction_ranges"] = prediction_ranges(net, variant, te_rows, cond_scaler, torch, models)
        out["variants"][variant] = entry
        out["states"][variant] = net.state_dict()
        out["hparams"][variant] = net.hparams
    return out


def decide_gate(fits: list, cfg: dict) -> dict:
    """Robust learned-loot gate over several grouped splits. A reward-aware
    variant passes iff (a) its MEAN test-skill gain over its information-
    matched twin is >= min_reward_gain_rel, (b) the gain is positive on EVERY
    split, (c) it beats the naive predictor on every split (if required) and
    (d) for xcie, the item representation is supported on every split."""
    g = cfg["q_loot_gate"]
    comps = {}
    for ra, twin in ml_data.REWARD_AWARE.items():
        if ra == "xcie" and not all(f["item_supported"] for f in fits):
            comps[ra] = {"twin": twin, "eligible": False, "reason": "item representation insufficiently supported"}
            continue
        gains = [f["variants"][ra]["test"]["skill"] - f["variants"][twin]["test"]["skill"] for f in fits]
        skills = [f["variants"][ra]["test"]["skill"] for f in fits]
        mean_gain = sum(gains) / len(gains)
        beats = all(sk > 0.0 for sk in skills)
        comps[ra] = {"twin": twin, "eligible": True, "gains_per_split": gains, "mean_gain": mean_gain,
                     "test_skills": skills, "beats_naive_all_splits": beats,
                     "passes": (mean_gain >= g["min_reward_gain_rel"] and all(x > 0 for x in gains)
                                and (beats or not g["require_beats_naive"]))}
    passing = [k for k, c in comps.items() if c.get("passes")]
    chosen = max(passing, key=lambda k: comps[k]["mean_gain"]) if passing else None
    return {"passed": bool(passing), "chosen_variant": chosen, "comparisons": comps,
            "split_seeds": [f["split_seed"] for f in fits], "rule": g["description"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train Q_loot ablations and evaluate the learned-loot gate.")
    ap.add_argument("--loot-csv", action="append", required=True)
    ap.add_argument("--out-dir", default="q_loot_out")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--split", default="0.70,0.15,0.15")
    ap.add_argument("--split-seeds", default=None,
                    help="comma-separated grouped-split seeds for the gate (default: q_loot_gate.split_seeds); "
                         "the first one provides the saved models")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    import torch
    import models

    ck_path = os.path.join(args.out_dir, "q_loot_checkpoint.pt")
    if os.path.exists(ck_path) and not args.overwrite:
        ap.error(f"{ck_path} exists (pass --overwrite)")
    os.makedirs(args.out_dir, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    random.seed(args.seed)
    t0 = time.time()
    cfg = balance.load_experiment_config()
    split_seeds = ([int(v) for v in args.split_seeds.split(",")] if args.split_seeds
                   else list(cfg["q_loot_gate"]["split_seeds"]))

    rows, fp = load_events(args.loot_csv)
    fits = []
    for sp in split_seeds:
        f = fit_split(rows, sp, args, cfg, torch, models, device)
        fits.append(f)
        print(f"[split {sp}] " + "  ".join(f"{v}: test {f['variants'][v]['test']['skill']:.4f}"
                                          for v in ml_data.LOOT_VARIANTS))
    gate = decide_gate(fits, cfg)
    primary = fits[0]
    report = {"n_events": primary["n_events"], "n_runs": primary["n_runs"], "support": primary["support"],
              "item_representation_supported": all(f["item_supported"] for f in fits),
              "category_allowed": primary["cat_allowed"],
              "train_prevalence_death_within": primary["prev"],
              "variants": primary["variants"],
              "per_split_test_skill": {f["split_seed"]: {v: f["variants"][v]["test"]["skill"]
                                                         for v in ml_data.LOOT_VARIANTS} for f in fits},
              "gate": gate}

    ck = {"checkpoint_schema": Q_LOOT_CHECKPOINT_SCHEMA, "schema_version": telemetry.SCHEMA_VERSION,
          "loot_dataset_version": loot_dataset.LOOT_DATASET_VERSION, "dataset_fingerprint": fp,
          "config_hash": fp["config_hash"], "experiment_config_version": cfg["experiment_config_version"],
          "experiment_config_hash": balance.experiment_config_hash(cfg),
          "outcome_columns": COLS, "cond_columns": ml_data.COND_COLUMNS, "item_ids": core.ALL_ITEM_IDS,
          "slots": core.SLOTS, "cond_scaler": primary["cond_scaler"].to_json(), "cont_var": primary["var"],
          "split": primary["split"], "split_seed": primary["split_seed"], "seed": args.seed,
          "hparams": vars(args), "variant_states": primary["states"], "variant_hparams": primary["hparams"],
          "gate": gate, "category_allowed": primary["cat_allowed"], "support": primary["support"]}
    torch.save(ck, ck_path)
    report["elapsed_sec"] = round(time.time() - t0, 2)
    report["dataset_fingerprint"] = fp
    with open(os.path.join(args.out_dir, "q_loot_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)

    print(f"events train/val/test (split {primary['split_seed']}) = "
          f"{primary['n_events']['train']}/{primary['n_events']['val']}/{primary['n_events']['test']}; "
          f"item representation supported: {report['item_representation_supported']}; "
          f"categories allowed: {primary['cat_allowed']}")
    for col in COLS:
        cells = []
        for v in ml_data.LOOT_VARIANTS:
            m = primary["variants"][v]["test"]["metrics"][col]
            cells.append(f"{v}:{(m['brier'] if col == COLS[0] else m['r2_vs_mean']):.4f}")
        base = primary["variants"]["x"]["test"]["metrics"][COLS[0]]["brier_baseline"] if col == COLS[0] else 0.0
        label = "Brier (naive %.4f)" % base if col == COLS[0] else "R2 (naive 0)"
        print(f"   {col:24s} {label:20s} " + "  ".join(cells))
    for k, c in gate["comparisons"].items():
        if c["eligible"]:
            print(f"   gate {k} vs {c['twin']}: gains per split "
                  + ", ".join(f"{x:+.4f}" for x in c["gains_per_split"])
                  + f" (mean {c['mean_gain']:+.4f}) beats_naive {c['beats_naive_all_splits']} -> "
                  + ("PASS" if c["passes"] else "fail"))
        else:
            print(f"   gate {k} vs {c['twin']}: {c['reason']}")
    for v in ml_data.LOOT_VARIANTS:
        r = primary["variants"][v]["prediction_ranges"]
        print(f"   ranges {v:4s} " + "  ".join(f"{c}:[{x['min']:.3f},{x['max']:.3f}]" for c, x in r.items())
              + ("" if all(x["inside_physical_range"] for x in r.values()) else "  OUT OF RANGE"))
    print(f"LEARNED-LOOT GATE: {'PASSED (' + gate['chosen_variant'] + ')' if gate['passed'] else 'FAILED'}")
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
