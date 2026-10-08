"""
train_q.py
==========
Trains the outcome evaluator Q(x, a) -> o on OBSERVED (x, a, o) tuples, plus
the state-only ablation Q(x) with the identical architecture/budget, and
decides whether Q is good enough to guide the generator (the "Q gate").

  python train_q.py --csv pilot.csv [--csv more.csv] --out-dir q_out

Outputs (in --out-dir):
  q_checkpoint.pt   both models' best states, optimizer states, scalers,
                    split run_ids, dataset fingerprint, versions, history
  q_report.json     per-output val/test metrics vs. naive baselines,
                    Q(x,a) vs Q(x), per-action-source errors, extreme-action
                    probe and the gate decision (machine-readable)

Metrics: death -> BCE, Brier, AUC, ECE (10 bins) vs. the train prevalence;
continuous outcomes -> MSE, MAE, R^2 vs. the train mean, and a decile
calibration error |mean(pred) - mean(obs)| averaged over prediction deciles.
Preprocessing (condition/outcome standardizers) is fit on TRAIN runs only.
"""

from typing import Dict, List
import argparse
import copy
import json
import math
import os
import random
import sys
import time

import balance
import dda_core as core
import ml_data
import telemetry

Q_CHECKPOINT_SCHEMA = "q_outcome_v2_bounded"


def _auc(scores: List[float], labels: List[int]) -> float:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_pos = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (rank_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def _ece(probs: List[float], labels: List[int], bins: int = 10) -> float:
    n = len(probs)
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(probs[i] for i in idx) / len(idx)
            acc = sum(labels[i] for i in idx) / len(idx)
            tot += len(idx) / n * abs(conf - acc)
    return tot


def _decile_calibration(pred: List[float], obs: List[float]) -> float:
    order = sorted(range(len(pred)), key=lambda i: pred[i])
    k = max(1, len(order) // 10)
    errs = []
    for s in range(0, len(order), k):
        idx = order[s:s + k]
        errs.append(abs(sum(pred[i] for i in idx) / len(idx) - sum(obs[i] for i in idx) / len(idx)))
    return sum(errs) / len(errs)


def evaluate_predictions(pred_phys: List[List[float]], y: List[List[float]],
                         train_prevalence: float, train_mean: List[float],
                         columns: List[str] = None) -> Dict[str, dict]:
    """Per-output metrics for physical predictions vs. observed outcomes and
    naive baselines (train prevalence / train mean). Column 0 is binary."""
    columns = columns or balance.OUTCOME_COLUMNS
    out: Dict[str, dict] = {}
    eps = 1e-6
    p = [min(max(r[0], eps), 1 - eps) for r in pred_phys]
    lab = [int(r[0] >= 0.5) for r in y]
    prev = min(max(train_prevalence, eps), 1 - eps)
    n = len(y)
    bce = -sum(l * math.log(pi) + (1 - l) * math.log(1 - pi) for pi, l in zip(p, lab)) / n
    bce0 = -sum(l * math.log(prev) + (1 - l) * math.log(1 - prev) for l in lab) / n
    brier = sum((pi - l) ** 2 for pi, l in zip(p, lab)) / n
    brier0 = sum((prev - l) ** 2 for l in lab) / n
    out[columns[0]] = {"n": n, "observed_rate": sum(lab) / n, "mean_pred": sum(p) / n,
                    "bce": bce, "bce_baseline": bce0, "brier": brier, "brier_baseline": brier0,
                    "auc": _auc(p, lab), "ece": _ece(p, lab)}
    for j, col in enumerate(columns[1:], start=1):
        pr = [r[j] for r in pred_phys]
        ob = [r[j] for r in y]
        mse = sum((a - b) ** 2 for a, b in zip(pr, ob)) / n
        mse0 = sum((train_mean[j - 1] - b) ** 2 for b in ob) / n
        mae = sum(abs(a - b) for a, b in zip(pr, ob)) / n
        out[col] = {"n": n, "mse": mse, "mse_baseline": mse0, "mae": mae,
                    "r2_vs_mean": 1.0 - mse / mse0 if mse0 > 0 else float("nan"),
                    "decile_calibration_error": _decile_calibration(pr, ob)}
    return out


def aggregate_skill(metrics: Dict[str, dict], columns: List[str] = None) -> float:
    """Mean relative improvement over the naive baseline across outputs
    (binary via BCE, continuous via MSE). Higher is better; 0 = naive."""
    columns = columns or balance.OUTCOME_COLUMNS
    b = metrics[columns[0]]
    vals = [1.0 - b["bce"] / max(b["bce_baseline"], 1e-12)]
    for col in columns[1:]:
        m = metrics[col]
        vals.append(1.0 - m["mse"] / max(m["mse_baseline"], 1e-12))
    return sum(vals) / len(vals)


def q_gate(test_xa: dict, test_x: dict, cfg: dict) -> dict:
    g = cfg["q_gate"]
    reasons = []
    for col in g["key_continuous_outputs"]:
        r2 = test_xa[col]["r2_vs_mean"]
        if not (r2 > g["min_r2_vs_mean"]):
            reasons.append(f"{col}: test R^2 {r2:.4f} does not beat the constant-mean baseline")
    if g.get("require_death_brier_below_prevalence", True):
        d = test_xa["death"]
        if not (d["brier"] < d["brier_baseline"]):
            reasons.append(f"death: test Brier {d['brier']:.4f} >= prevalence baseline "
                           f"{d['brier_baseline']:.4f}")
    for col, m in test_xa.items():
        for k, v in m.items():
            # AUC is undefined (nan) when the split holds a single class.
            if isinstance(v, float) and k != "auc" and not math.isfinite(v):
                reasons.append(f"{col}: non-finite {k}")
    s_xa, s_x = aggregate_skill(test_xa), aggregate_skill(test_x)
    gain = s_xa - s_x
    if not (gain >= g["min_action_gain_rel"]):
        reasons.append(f"Q(x,a) skill {s_xa:.4f} vs Q(x) skill {s_x:.4f}: action gain {gain:.4f} "
                       f"< required {g['min_action_gain_rel']} -- data lacks evidence that the "
                       f"intervention affects outcomes")
    return {"passed": not reasons, "reasons": reasons, "skill_q_xa": s_xa, "skill_q_x": s_x,
            "action_gain": gain}


def _tensors(rows, cond_scaler, cfg, torch, device):
    x = torch.tensor(cond_scaler.transform(ml_data.cond_matrix(rows)), dtype=torch.float32, device=device)
    a = torch.tensor([ml_data.normalize_action(r) for r in ml_data.action_matrix(rows)],
                     dtype=torch.float32, device=device)
    y = ml_data.outcome_matrix(rows, cfg)
    yd = torch.tensor([r[0] for r in y], dtype=torch.float32, device=device)
    yc = torch.tensor([r[1:] for r in y], dtype=torch.float32, device=device)   # physical units
    return x, a, yd, yc, y


def _predict_phys(model, x, a, models, torch):
    model.eval()
    with torch.no_grad():
        return models.outcomes_physical(model(x, a)).cpu().tolist()


def train_one(model, tr, va, args, torch, models, cont_var, log_prefix=""):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x, a, yd, yc, _ = tr
    xv, av, ydv, ycv, _ = va
    n = x.shape[0]
    best = (float("inf"), None, None, -1)
    history = []
    bad = 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=x.device)
        tot = 0.0
        nb = 0
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            loss, _, _ = models.q_loss(model(x[idx], a[idx]), yd[idx], yc[idx], cont_var, args.death_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        model.eval()
        with torch.no_grad():
            vl, vb, vm = models.q_loss(model(xv, av), ydv, ycv, cont_var, args.death_weight)
        history.append({"epoch": epoch, "train_loss": tot / max(nb, 1), "val_loss": float(vl),
                        "val_bce": float(vb), "val_mse": float(vm)})
        if float(vl) < best[0] - 1e-6:
            best = (float(vl), copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()), epoch)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best[1])
    print(f"{log_prefix} best epoch {best[3]} val_loss {best[0]:.4f} (ran {len(history)} epochs)")
    return {"best_val_loss": best[0], "best_epoch": best[3], "optimizer_state": best[2],
            "history": history}


def extreme_action_probe(q, cond_scaler, rows, torch, models, device, n_states: int = 50):
    """Q predictions at the box corners / centre for the median test state,
    plus the physical range of predictions over ALL 16 corners of the action
    box for up to n_states test states. Sanity: harder actions should not
    predict LOWER death than easier ones; ranges must stay physical."""
    xs = ml_data.cond_matrix(rows)
    med = [sorted(c[i] for c in xs)[len(xs) // 2] for i in range(core.COND_DIM)]
    probes = {"easiest": [0.0, 0.0, 0.0, 0.0], "centre": [0.5, 0.5, 0.5, 0.5],
              "hardest": [1.0, 1.0, 1.0, 1.0], "max_spawn_only": [0.0, 0.0, 0.0, 1.0],
              "max_dmg_only": [0.0, 1.0, 0.0, 0.0]}
    x = torch.tensor(cond_scaler.transform([med] * len(probes)), dtype=torch.float32, device=device)
    a = torch.tensor(list(probes.values()), dtype=torch.float32, device=device)
    q.eval()
    with torch.no_grad():
        y = models.outcomes_physical(q(x, a)).cpu().tolist()
        import itertools
        corners = torch.tensor(list(itertools.product([0.0, 1.0], repeat=4)), dtype=torch.float32, device=device)
        xs_t = torch.tensor(cond_scaler.transform(xs[:n_states]), dtype=torch.float32, device=device)
        xx = xs_t.repeat_interleave(len(corners), dim=0)
        aa = corners.repeat(xs_t.shape[0], 1)
        yy = models.outcomes_physical(q(xx, aa))
    out = {k: {"action": ml_data.denormalize_action(v), "pred": dict(zip(balance.OUTCOME_COLUMNS, y[i]))}
           for i, (k, v) in enumerate(probes.items())}
    ranges = balance.output_ranges("enemy")
    out["corner_prediction_ranges"] = {
        col: {"min": float(yy[:, j].min()), "max": float(yy[:, j].max()),
              "physical_range": list(ranges[j]),
              "inside_physical_range": bool(yy[:, j].min() >= ranges[j][0] - 1e-6
                                            and yy[:, j].max() <= ranges[j][1] + 1e-6)}
        for j, col in enumerate(balance.OUTCOME_COLUMNS)}
    out["median_condition"] = dict(zip(ml_data.COND_COLUMNS, med))
    out["death_monotone_easy_lt_hard"] = y[0][0] <= y[2][0]
    out["post_hp_monotone_easy_gt_hard"] = y[0][1] >= y[2][1]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the outcome evaluator Q(x,a) and the Q(x) ablation.")
    ap.add_argument("--csv", action="append", required=True, help="dataset CSV (repeatable)")
    ap.add_argument("--out-dir", default="q_out")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--death-weight", type=float, default=1.0)
    ap.add_argument("--split", default="0.70,0.15,0.15", help="train,val,test fractions of run_ids")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    import torch
    import models

    if os.path.exists(os.path.join(args.out_dir, "q_checkpoint.pt")) and not args.overwrite:
        ap.error(f"{args.out_dir} already holds a Q checkpoint (pass --overwrite)")
    os.makedirs(args.out_dir, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()

    import bot_runner
    cfg = balance.load_experiment_config()
    rows, fp = ml_data.load_training_rows(args.csv, expected_extra_subset=bot_runner.policy_extra())
    fracs = tuple(float(v) for v in args.split.split(","))
    split = ml_data.grouped_split(rows, fracs, seed=args.split_seed)
    tr_rows, va_rows, te_rows = (ml_data.select(rows, split[k]) for k in ("train", "val", "test"))
    if min(len(tr_rows), len(va_rows), len(te_rows)) < 10:
        raise SystemExit(f"too few combat rows per split: train={len(tr_rows)} val={len(va_rows)} "
                         f"test={len(te_rows)}")

    cond_scaler = ml_data.Standardizer.fit(ml_data.cond_matrix(tr_rows), ml_data.COND_COLUMNS)
    y_tr = ml_data.outcome_matrix(tr_rows, cfg)
    out_scaler = ml_data.Standardizer.fit([r[1:] for r in y_tr], balance.OUTCOME_COLUMNS[1:])
    cont_var = torch.tensor([s ** 2 for s in out_scaler.std], dtype=torch.float32, device=device)
    train_prev = sum(r[0] for r in y_tr) / len(y_tr)
    train_mean = [sum(r[j] for r in y_tr) / len(y_tr) for j in range(1, balance.OUTCOME_DIM)]

    tr, va, te = (_tensors(r, cond_scaler, cfg, torch, device) for r in (tr_rows, va_rows, te_rows))
    hp = dict(hidden=args.hidden, layers=args.layers, dropout=args.dropout)

    torch.manual_seed(args.seed)
    q_xa = models.OutcomeModel(use_action=True, **hp).to(device)
    res_xa = train_one(q_xa, tr, va, args, torch, models, cont_var, "Q(x,a)")
    torch.manual_seed(args.seed)
    q_x = models.OutcomeModel(use_action=False, **hp).to(device)
    res_x = train_one(q_x, tr, va, args, torch, models, cont_var, "Q(x)  ")

    report = {"n_rows": {"train": len(tr_rows), "val": len(va_rows), "test": len(te_rows)},
              "n_runs": {k: len(v) for k, v in split.items()},
              "train_prevalence_death": train_prev, "train_mean_continuous":
                  dict(zip(balance.OUTCOME_COLUMNS[1:], train_mean))}
    for name, part in (("val", va), ("test", te)):
        report[name] = {
            "q_xa": evaluate_predictions(_predict_phys(q_xa, part[0], part[1], models, torch),
                                         part[4], train_prev, train_mean),
            "q_x": evaluate_predictions(_predict_phys(q_x, part[0], part[1], models, torch),
                                        part[4], train_prev, train_mean),
        }
        report[name]["skill_q_xa"] = aggregate_skill(report[name]["q_xa"])
        report[name]["skill_q_x"] = aggregate_skill(report[name]["q_x"])

    # Held-out actions: test error broken down by how the action was produced.
    by_src: Dict[str, dict] = {}
    for src in sorted(set(r["action_source"] for r in te_rows)):
        sub = [r for r in te_rows if r["action_source"] == src]
        if len(sub) < 5:
            continue
        part = _tensors(sub, cond_scaler, cfg, torch, device)
        by_src[src] = evaluate_predictions(_predict_phys(q_xa, part[0], part[1], models, torch),
                                           part[4], train_prev, train_mean)
    report["test_by_action_source"] = by_src
    report["extreme_action_probe"] = extreme_action_probe(q_xa, cond_scaler, te_rows, torch, models, device)
    report["gate"] = q_gate(report["test"]["q_xa"], report["test"]["q_x"], cfg)

    ckpt = {
        "checkpoint_schema": Q_CHECKPOINT_SCHEMA,
        "schema_version": telemetry.SCHEMA_VERSION, "dda_version": telemetry.DDA_VERSION,
        "config_hash": fp["config_hash"], "dataset_fingerprint": fp,
        "experiment_config_version": cfg["experiment_config_version"],
        "experiment_config_hash": balance.experiment_config_hash(cfg),
        "cond_columns": ml_data.COND_COLUMNS, "action_columns": ml_data.ACTION_COLUMNS,
        "outcome_columns": balance.OUTCOME_COLUMNS,
        "action_lo": ml_data.ACTION_LO, "action_hi": ml_data.ACTION_HI,
        "cond_scaler": cond_scaler.to_json(), "outcome_scaler": out_scaler.to_json(),
        "split": split, "split_fracs": list(fracs), "split_seed": args.split_seed,
        "seed": args.seed, "hparams": vars(args),
        "q_xa_hparams": q_xa.hparams, "q_x_hparams": q_x.hparams,
        "q_xa_state": q_xa.state_dict(), "q_x_state": q_x.state_dict(),
        "q_xa_optimizer": res_xa["optimizer_state"], "q_x_optimizer": res_x["optimizer_state"],
        "best_epoch": {"q_xa": res_xa["best_epoch"], "q_x": res_x["best_epoch"]},
        "best_val_loss": {"q_xa": res_xa["best_val_loss"], "q_x": res_x["best_val_loss"]},
        "history": {"q_xa": res_xa["history"], "q_x": res_x["history"]},
        "gate": report["gate"],
    }
    torch.save(ckpt, os.path.join(args.out_dir, "q_checkpoint.pt"))
    report["elapsed_sec"] = round(time.time() - t0, 2)
    report["dataset_fingerprint"] = fp
    with open(os.path.join(args.out_dir, "q_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)

    print(f"rows train/val/test = {len(tr_rows)}/{len(va_rows)}/{len(te_rows)} "
          f"runs = {report['n_runs']}")
    for name in ("val", "test"):
        r = report[name]
        print(f"[{name}] skill Q(x,a)={r['skill_q_xa']:.4f}  Q(x)={r['skill_q_x']:.4f}")
        d = r["q_xa"]["death"]
        print(f"   death  brier {d['brier']:.4f} (prev {d['brier_baseline']:.4f}) "
              f"auc {d['auc']:.3f} ece {d['ece']:.3f}")
        for col in balance.OUTCOME_COLUMNS[1:]:
            m, mx = r["q_xa"][col], r["q_x"][col]
            print(f"   {col:18s} R2 Q(x,a)={m['r2_vs_mean']:.4f}  Q(x)={mx['r2_vs_mean']:.4f}  "
                  f"mse {m['mse']:.4f} (mean {m['mse_baseline']:.4f})")
    print("physical prediction ranges over all 16 action-box corners:")
    for col, r in report["extreme_action_probe"]["corner_prediction_ranges"].items():
        print(f"   {col:18s} [{r['min']:.3f}, {r['max']:.3f}] within {r['physical_range']}: "
              f"{r['inside_physical_range']}")
    g = report["gate"]
    print(f"Q GATE: {'PASSED' if g['passed'] else 'FAILED'} (action gain {g['action_gain']:.4f})")
    for reason in g["reasons"]:
        print("   -", reason)
    print(f"checkpoint -> {os.path.join(args.out_dir, 'q_checkpoint.pt')}")
    return 0 if g["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
