"""
train_loot_policy.py
====================
Learned loot-category policy G_loot(x, e, z) -> [P_weapon, P_spell, P_armor,
P_boots], guided by a FROZEN Q_loot. Runs ONLY if train_q_loot.py's gate
passed (held-out evidence that reward information improves delayed-outcome
prediction beyond state alone); otherwise it refuses.

  python train_loot_policy.py --loot-csv loot.csv --q-loot-dir q_loot_out --out-dir loot_model_out

Objective (no supervised category label; the heuristic is not a target):
  L = E_x[ sum_c p_c(x,e,z) * U_c(x,e) ]                 expected horizon balance loss
      - lambda_entropy * H(p)                              anti-collapse
      + lambda_diversity * relu(tau - |p(z1)-p(z2)|/|z1-z2|)  noise diversity
where U_c is balance.balance_distance(Q_loot(x, c[, item, e]), family="loot"),
averaged uniformly over the FIXED item pool of c when the chosen Q_loot
variant uses item identity (items are drawn uniformly within a category).

Support / OOD protection (experiment_config.loot_action_support): categories
with fewer than min_category_events training events get probability 0, every
allowed category keeps at least min_category_prob, inputs are the observed
pre-reward states only, and selection rejects collapsed policies.
"""

import argparse
import copy
import hashlib
import io
import json
import os
import random
import sys
import time

import balance
import dda_core as core
import ml_data
import telemetry
from train_q_loot import load_events

LOOT_POLICY_SCHEMA_VERSION = "loot_policy_v1"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the learned loot-category policy (gated).")
    ap.add_argument("--loot-csv", action="append", required=True)
    ap.add_argument("--q-loot-dir", required=True)
    ap.add_argument("--out-dir", default="loot_model_out")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise-dim", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    import torch
    import models
    import model_runtime

    path = os.path.join(args.out_dir, model_runtime.LOOT_POLICY_FILE)
    if os.path.exists(path) and not args.overwrite:
        ap.error(f"{path} exists (pass --overwrite)")
    cfg = balance.load_experiment_config()
    lcfg = cfg["loot_action_support"]
    qck = torch.load(os.path.join(args.q_loot_dir, model_runtime.Q_LOOT_FILE), map_location="cpu",
                     weights_only=True)
    if not qck["gate"]["passed"]:
        raise SystemExit("Q_loot gate FAILED: current evidence does not support a learned reward policy. "
                         "Keep the descriptive analysis; do not train or enable learned_loot.")
    variant = qck["gate"]["chosen_variant"]
    qp = model_runtime.QLootPredictor(qck, variant)
    rows, fp = load_events(args.loot_csv)
    if fp["dataset_id"] != qck["dataset_fingerprint"]["dataset_id"]:
        raise SystemExit("loot dataset differs from Q_loot's training dataset")
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()

    split = qck["split"]
    cs = qp.cond_scaler

    def state_tensors(rs):
        cond = cs.transform([[float(r[c]) for c in ml_data.COND_COLUMNS] for r in rs])
        eq = [ml_data.loot_row_equip_before(r) for r in rs]
        g_in = torch.tensor([list(c) + ml_data.equip_onehot(e) for c, e in zip(cond, eq)], dtype=torch.float32)
        # U[n, c]: Q_loot-predicted horizon balance loss of choosing category c
        util = torch.zeros(len(rs), len(core.SLOTS))
        with torch.no_grad():
            for j, cat in enumerate(core.SLOTS):
                items = [it.item_id for it in core.ITEMS_BY_SLOT[cat]] if "item" in ml_data.LOOT_VARIANTS[variant] else [None]
                acc = torch.zeros(len(rs))
                for iid in items:
                    f = torch.tensor([ml_data.loot_features(c, e, cat, iid, variant) for c, e in zip(cond, eq)],
                                     dtype=torch.float32)
                    acc += balance.balance_loss_torch(models.outcomes_physical(qp.net(f)), cfg, reduce=False,
                                                      family="loot")
                util[:, j] = acc / len(items)
        return g_in, util

    tr_rows, va_rows = ml_data.select(rows, split["train"]), ml_data.select(rows, split["val"])
    x_tr, u_tr = state_tensors(tr_rows)
    x_va, u_va = state_tensors(va_rows)
    gen = models.LootGenerator(x_tr.shape[1], args.noise_dim, args.hidden, 2, lcfg["min_category_prob"],
                               qck["category_allowed"])
    opt = torch.optim.Adam(gen.parameters(), lr=args.lr)
    uniform = torch.full((1, len(core.SLOTS)), 0.25)

    def evaluate(xx, uu):
        gen.eval()
        with torch.no_grad():
            g = torch.Generator().manual_seed(args.seed)
            ps = torch.stack([gen(xx, torch.randn(xx.shape[0], gen.noise_dim, generator=g)) for _ in range(8)])
            p = ps.mean(dim=0)
            exp_bal = float((ps * uu.unsqueeze(0)).sum(dim=2).mean())
            div = float(sum((ps[i] - ps[j]).abs().mean() for i in range(8) for j in range(i + 1, 8)) / 28)
            return {"expected_balance": exp_bal,
                    "expected_balance_uniform": float((uniform * uu).sum(dim=1).mean()),
                    "expected_balance_best_fixed_category": float(uu.mean(dim=0).min()),
                    "entropy": float(models.entropy(p).mean()), "diversity": div,
                    "mean_probs": dict(zip(core.SLOTS, p.mean(dim=0).tolist())),
                    "max_mean_prob": float(p.mean(dim=0).max())}

    history, best = [], {"score": float("inf"), "epoch": -1}
    n = x_tr.shape[0]
    for epoch in range(args.epochs):
        gen.train()
        perm = torch.randperm(n)
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            z1 = torch.randn(len(idx), gen.noise_dim)
            z2 = torch.randn(len(idx), gen.noise_dim)
            p1, p2 = gen(x_tr[idx], z1), gen(x_tr[idx], z2)
            l_bal = (p1 * u_tr[idx]).sum(dim=1).mean()
            l_ent = -models.entropy(p1).mean()
            l_div = models.diversity_loss(p1, p2, z1, z2, lcfg["diversity_tau"])
            loss = l_bal + lcfg["lambda_entropy"] * l_ent + lcfg["lambda_diversity"] * l_div
            opt.zero_grad()
            loss.backward()
            opt.step()
        d = evaluate(x_va, u_va)
        eligible = epoch >= 5 and d["max_mean_prob"] < 0.9 and d["diversity"] > 0.0
        history.append({"epoch": epoch, **{k: v for k, v in d.items() if not isinstance(v, dict)},
                        "eligible": eligible})
        if eligible and d["expected_balance"] < best["score"]:
            best = {"score": d["expected_balance"], "epoch": epoch, "diag": d,
                    "state": copy.deepcopy(gen.state_dict()), "opt": copy.deepcopy(opt.state_dict())}
    if best["epoch"] < 0:
        raise SystemExit("no eligible learned-loot checkpoint (collapsed or non-diverse at every epoch)")
    buf = io.BytesIO()
    torch.save(best["state"], buf)
    ckpt_id = hashlib.sha256(buf.getvalue()).hexdigest()[:16]
    ck = {"checkpoint_schema_version": LOOT_POLICY_SCHEMA_VERSION, "checkpoint_id": ckpt_id,
          "schema_version": telemetry.SCHEMA_VERSION,
          "experiment_config_version": cfg["experiment_config_version"],
          "experiment_config_hash": balance.experiment_config_hash(cfg),
          "config_hash": qck["config_hash"], "dataset_extra": _bot_extra(),
          "outcome_columns": balance.LOOT_OUTCOME_COLUMNS, "cond_columns": ml_data.COND_COLUMNS,
          "item_ids": core.ALL_ITEM_IDS, "slots": core.SLOTS, "cond_scaler": qck["cond_scaler"],
          "generator_hparams": gen.hparams, "generator_state_dict": best["state"],
          "optimizer_state": best["opt"], "epoch": best["epoch"], "best_score": best["score"],
          "best_diagnostics": best["diag"], "q_loot_variant": variant,
          "q_loot_state": qck["variant_states"][variant], "q_loot_hparams": qck["variant_hparams"][variant],
          "q_loot_gate": qck["gate"], "seed": args.seed, "hparams": vars(args), "history": history,
          "objective": "E[sum_c p_c U_c] - lambda_entropy H(p) + lambda_diversity L_div"}
    torch.save(ck, path)
    with open(os.path.join(args.out_dir, "loot_policy_report.json"), "w") as f:
        json.dump({"checkpoint_id": ckpt_id, "best_epoch": best["epoch"], "diagnostics": best["diag"],
                   "variant": variant, "elapsed_sec": round(time.time() - t0, 2)}, f, indent=2)
    print(f"best epoch {best['epoch']} checkpoint {ckpt_id}: {best['diag']}")
    return 0


def _bot_extra() -> dict:
    """The config_hash recorded by bot datasets is computed over the bot policy
    extra plus the experiment-config hash (telemetry.write_manifest)."""
    import bot_runner
    extra = bot_runner.policy_extra()
    extra["experiment_config_hash"] = balance.experiment_config_hash()
    return extra


if __name__ == "__main__":
    sys.exit(main())
