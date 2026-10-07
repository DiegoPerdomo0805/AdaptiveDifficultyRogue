"""
train_cgan.py
=============
Outcome-guided conditional GAN (replaces the former heuristic-imitation
trainer, which minimized BCE_adv + 0.5 * MSE(G(x), heuristic applied_*)).

  python train_cgan.py --csv pilot.csv --q-dir q_out --out-dir model_out

  G(x, z) -> candidate action a (bounded in the design box by construction)
  D(x, a) -> plausibility of observed vs. generated (x, a) pairs
  Q(x, a) -> expected outcome; trained by train_q.py on observed outcomes
             and FROZEN here

  L_D = BCE(D(x, a_obs), 1) + BCE(D(x, G(x, z)), 0)
  L_G = L_adv + lambda_balance * L_balance + lambda_diversity * L_diversity
    L_adv       = BCE(D(x, G(x, z1)), 1)                (stay in the observed
                                                          safe action support)
    L_balance   = balance.balance_loss_torch(Q(x, G(x, z1)))   (targets from
                                                          experiment_config.json)
    L_diversity = mean(relu(tau - |G(x,z1)-G(x,z2)|_1 / |z1-z2|_1))

The observed actions a_obs (from fixed, heuristic AND exploration rows)
only enter the discriminator as samples of the plausible action support.
There is no reconstruction loss against heuristic actions and no warm-up.

The best checkpoint is selected by models.selection_score() on validation
runs (validity, Q-predicted balance, diversity, conditional sensitivity,
temporal jitter) -- never by generator loss. The train/val/test split and
the condition scaler are taken from the Q checkpoint so G is validated on
exactly the runs Q never trained on.
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

CHECKPOINT_SCHEMA_VERSION = "cgan_outcome_v3"
COND_COLUMNS = ml_data.COND_COLUMNS
ACTION_COLUMNS = ml_data.ACTION_COLUMNS


def _ordered_runs(rows):
    """Rows sorted by (run_id, room_seq) and the parallel run index list."""
    rows = sorted(rows, key=lambda r: (r["run_id"], int(r["room_seq"])))
    return rows, [r["run_id"] for r in rows]


def _actions_n(matrix, torch, device):
    return torch.tensor([ml_data.normalize_action(a) for a in matrix], dtype=torch.float32, device=device)


def checkpoint_id_for(state_dict, torch) -> str:
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in state_dict.items()}, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the outcome-guided cGAN generator.")
    ap.add_argument("--csv", action="append", required=True,
                    help="dataset CSV(s) -- must be exactly the ones Q was trained on")
    ap.add_argument("--q-dir", required=True, help="directory holding q_checkpoint.pt")
    ap.add_argument("--out-dir", default="model_out")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr-g", type=float, default=5e-4)
    ap.add_argument("--lr-d", type=float, default=5e-4)
    ap.add_argument("--noise-dim", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lambda-balance", type=float, default=1.0)
    ap.add_argument("--lambda-diversity", type=float, default=1.0)
    ap.add_argument("--diversity-tau", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=40, help="epochs without a better selection score")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    ap.add_argument("--allow-failed-q-gate", action="store_true",
                    help="SMOKE TESTS ONLY: train even though Q failed its gate (recorded)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    import torch
    import models
    import bot_runner

    ckpt_path = os.path.join(args.out_dir, "cgan_checkpoint.pt")
    if os.path.exists(ckpt_path) and not args.overwrite:
        ap.error(f"{ckpt_path} exists (pass --overwrite)")
    os.makedirs(args.out_dir, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    cfg = balance.load_experiment_config()
    sel_cfg = cfg["cgan_selection"]

    # --- Q (frozen) and its data contract --------------------------------
    qck = torch.load(os.path.join(args.q_dir, "q_checkpoint.pt"), map_location=device, weights_only=True)
    gate = qck["gate"]
    if not gate["passed"]:
        msg = "Q failed its gate: " + "; ".join(gate["reasons"])
        if not args.allow_failed_q_gate:
            raise SystemExit(msg + "\nRefusing to train a generator guided by an inadequate Q "
                                   "(collect more varied data). --allow-failed-q-gate is for smoke tests.")
        print("WARNING:", msg, file=sys.stderr)
    for key, cur in (("experiment_config_hash", balance.experiment_config_hash(cfg)),
                     ("schema_version", telemetry.SCHEMA_VERSION),
                     ("dda_version", telemetry.DDA_VERSION)):
        if qck[key] != cur:
            raise SystemExit(f"Q checkpoint {key}={qck[key]!r} != current {cur!r}; retrain Q.")

    rows, fp = ml_data.load_training_rows(args.csv, include_boss=qck["dataset_fingerprint"]["include_boss"],
                                          expected_extra_subset=bot_runner.policy_extra())
    if fp["dataset_id"] != qck["dataset_fingerprint"]["dataset_id"]:
        raise SystemExit(f"dataset {fp['dataset_id']} differs from Q's training dataset "
                         f"{qck['dataset_fingerprint']['dataset_id']}; pass the same --csv files.")
    split = qck["split"]
    tr_rows = ml_data.select(rows, split["train"])
    va_rows, va_runs = _ordered_runs(ml_data.select(rows, split["val"]))

    cond_scaler = ml_data.Standardizer.from_json(qck["cond_scaler"])
    out_scaler_json = qck["outcome_scaler"]
    out_scaler_t = models.OutcomeScaler(out_scaler_json["mean"], out_scaler_json["std"]).to(device)
    q = models.OutcomeModel(**qck["q_xa_hparams"]).to(device)
    q.load_state_dict(qck["q_xa_state"])
    models.freeze(q)

    x_tr = torch.tensor(cond_scaler.transform(ml_data.cond_matrix(tr_rows)), dtype=torch.float32, device=device)
    a_tr = _actions_n(ml_data.action_matrix(tr_rows), torch, device)
    x_va = torch.tensor(cond_scaler.transform(ml_data.cond_matrix(va_rows)), dtype=torch.float32, device=device)
    refs = {
        "observed": _actions_n(ml_data.action_matrix(va_rows), torch, device),
        "heuristic": _actions_n([ml_data.heuristic_action_for_row(r) for r in va_rows], torch, device),
        "fixed": _actions_n([[cfg["controllers"]["fixed"]["action"][k] for k in core.ACTION_KEYS]]
                            * len(va_rows), torch, device),
    }
    # Dequantize the integer spawn dimension of real actions so D cannot
    # separate real/fake merely by integrality.
    spawn_half_bin = 0.5 / (ml_data.ACTION_HI[-1] - ml_data.ACTION_LO[-1])

    gen = models.Generator(noise_dim=args.noise_dim, hidden=args.hidden).to(device)
    disc = models.Discriminator(hidden=args.hidden).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    n = x_tr.shape[0]
    history = []
    best = {"score": float("inf"), "epoch": -1}
    bad = 0
    for epoch in range(args.epochs):
        gen.train()
        disc.train()
        perm = torch.randperm(n, device=device)
        sums = {"d": 0.0, "g": 0.0, "adv": 0.0, "balance": 0.0, "diversity": 0.0}
        nb = 0
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            if idx.numel() < 2:
                continue
            x_b = x_tr[idx]
            a_real = a_tr[idx].clone()
            a_real[:, -1] = (a_real[:, -1] + (torch.rand_like(a_real[:, -1]) * 2 - 1) * spawn_half_bin).clamp(0, 1)

            a_fake = gen(x_b)
            d_loss = models.discriminator_loss(disc, x_b, a_real, a_fake)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            terms = models.generator_loss(gen, disc, q, out_scaler_t, x_b, args.lambda_balance,
                                          args.lambda_diversity, args.diversity_tau, cfg)
            opt_g.zero_grad()
            terms["total"].backward()
            opt_g.step()

            sums["d"] += d_loss.item()
            sums["g"] += terms["total"].item()
            for k in models.GENERATOR_LOSS_TERMS:
                sums[k] += terms[k].item()
            nb += 1

        diag = models.generator_diagnostics(gen, q, out_scaler_t, x_va, run_index=va_runs,
                                            reference_actions_n=refs, k=sel_cfg["diversity_samples"],
                                            seed=args.seed, exp_cfg=cfg)
        score = models.selection_score(diag, sel_cfg)
        rec = {"epoch": epoch, **{f"loss_{k}": v / max(nb, 1) for k, v in sums.items()},
               "val_score": score, **{f"val_{k}": v for k, v in diag.items()
                                      if not isinstance(v, dict)},
               "val_balance_ref": diag["balance_ref"]}
        history.append(rec)
        if score < best["score"] - 1e-6:
            best = {"score": score, "epoch": epoch, "diag": diag,
                    "gen": copy.deepcopy(gen.state_dict()), "disc": copy.deepcopy(disc.state_dict()),
                    "opt_g": copy.deepcopy(opt_g.state_dict()), "opt_d": copy.deepcopy(opt_d.state_dict())}
            bad = 0
        else:
            bad += 1
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d} D {rec['loss_d']:.3f} adv {rec['loss_adv']:.3f} "
                  f"bal {rec['loss_balance']:.4f} div {rec['loss_diversity']:.4f} | val score {score:.4f} "
                  f"bal_pred {diag['balance_pred']:.4f} (heur {diag['balance_ref']['heuristic']:.4f}) "
                  f"div {diag['diversity']:.3f} sens {diag['sensitivity']:.3f}")
        if bad >= args.patience:
            print(f"early stop at epoch {epoch} (best {best['epoch']})")
            break

    gen.load_state_dict(best["gen"])
    ckpt_id = checkpoint_id_for(best["gen"], torch)
    ckpt = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": ckpt_id,
        "dataset_schema_version": telemetry.SCHEMA_VERSION,
        "dda_version": telemetry.DDA_VERSION,
        "config_hash": fp["config_hash"],
        "dataset_extra": fp["manifest_extra"],
        "dataset_fingerprint": fp,
        "experiment_config_version": cfg["experiment_config_version"],
        "experiment_config_hash": balance.experiment_config_hash(cfg),
        "cond_dim": core.COND_DIM, "target_dim": core.ACTION_DIM, "outcome_dim": balance.OUTCOME_DIM,
        "cond_columns": COND_COLUMNS, "action_columns": ACTION_COLUMNS, "action_keys": core.ACTION_KEYS,
        "outcome_columns": balance.OUTCOME_COLUMNS,
        "action_lo": ml_data.ACTION_LO, "action_hi": ml_data.ACTION_HI,
        "cond_scaler": cond_scaler.to_json(), "outcome_scaler": out_scaler_json,
        "generator_hparams": gen.hparams, "discriminator_hparams": disc.hparams,
        "q_hparams": qck["q_xa_hparams"],
        "generator_state_dict": best["gen"], "discriminator_state_dict": best["disc"],
        "q_state_dict": qck["q_xa_state"],
        "optimizer_g_state": best["opt_g"], "optimizer_d_state": best["opt_d"],
        "epoch": best["epoch"], "best_score": best["score"], "best_diagnostics": best["diag"],
        "seed": args.seed, "hparams": vars(args), "split": split,
        "q_gate": gate, "q_gate_overridden": (not gate["passed"]) and args.allow_failed_q_gate,
        "loss_equation": "L_G = L_adv + lambda_balance*L_balance + lambda_diversity*L_diversity",
        "history": history,
    }
    torch.save(ckpt, ckpt_path)
    with open(os.path.join(args.out_dir, "scaler.json"), "w") as f:
        json.dump({"cond_scaler": cond_scaler.to_json(), "outcome_scaler": out_scaler_json,
                   "action_lo": ml_data.ACTION_LO, "action_hi": ml_data.ACTION_HI}, f, indent=2)
    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    report = {"checkpoint_id": ckpt_id, "best_epoch": best["epoch"], "best_score": best["score"],
              "best_diagnostics": best["diag"], "q_gate": gate, "hparams": vars(args),
              "elapsed_sec": round(time.time() - t0, 2), "n_train_rows": len(tr_rows),
              "n_val_rows": len(va_rows), "dataset_id": fp["dataset_id"]}
    with open(os.path.join(args.out_dir, "cgan_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    d = best["diag"]
    print(f"best epoch {best['epoch']} score {best['score']:.4f} checkpoint_id {ckpt_id}")
    print(f"  validity {d['validity']:.3f} saturation {d['saturation']:.3f} diversity {d['diversity']:.4f} "
          f"sensitivity {d['sensitivity']:.4f} jitter {d['jitter']:.4f}")
    print(f"  Q-predicted balance: G {d['balance_pred']:.4f} | "
          + " ".join(f"{k} {v:.4f}" for k, v in d["balance_ref"].items()))
    print(f"checkpoint -> {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
