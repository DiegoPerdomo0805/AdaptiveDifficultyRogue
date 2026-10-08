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
        + lambda_support * L_support
    L_adv       = BCE(D(x, G(x, z1)), 1)                (stay in the observed
                                                          safe action support)
    L_balance   = balance.balance_loss_torch(Q(x, G(x, z1)))   (targets from
                                                          experiment_config.json)
    L_diversity = mean(relu(tau - |G(x,z1)-G(x,z2)|_1 / |z1-z2|_1))
    L_support   = mean(relu(d_kNN(x, G(x,z1)) / threshold - 1)^2)   (OOD guard)

Action-support / OOD protection (experiment_config.enemy_action_support):
  * G's outputs are confined by construction to the per-dimension
    [q_low, q_high] quantile box of OBSERVED training actions;
  * L_support penalizes (x, a) pairs farther from the observed training
    pairs than the threshold quantile of their own kNN distances, so G
    cannot exploit Q's extrapolation in unobserved regions;
  * the support reference set and threshold are saved in the checkpoint and
    model_runtime.py rejects runtime actions beyond runtime_ood_factor x
    threshold (logged heuristic fallback, reason ood_action).

The observed actions a_obs (from fixed, heuristic AND exploration rows)
only enter the discriminator as samples of the plausible action support.
There is no reconstruction loss against heuristic actions and no warm-up.

Checkpoint selection is CLOSED-LOOP (selection.py, experiment_config
cgan_selection.closed_loop): every candidate_every_epochs epochs an epoch that
passes the room-level gates (validity, diversity, sensitivity, min_epoch)
becomes a candidate; after training the top_k_candidates by room-level score
(models.selection_score, a filter only) are saved, reloaded through
model_runtime and rolled out with the bot on the frozen validation seeds.
The checkpoint with the lowest observed mean(B_r) + 0.5 * CVaR_0.90(B_r) among
ELIGIBLE candidates is selected; if none is eligible the run fails (exit 3)
and no checkpoint is written. Never selected by generator loss or by Q. The train/val/test split and
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

CHECKPOINT_SCHEMA_VERSION = "cgan_outcome_v4"
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
    ap.add_argument("--lambda-balance", type=float, default=None,
                    help="default: experiment_config.cgan_training.lambda_balance")
    ap.add_argument("--lambda-diversity", type=float, default=None,
                    help="default: experiment_config.cgan_training.lambda_diversity")
    ap.add_argument("--diversity-tau", type=float, default=0.05)
    ap.add_argument("--lambda-support", type=float, default=None,
                    help="default: experiment_config.enemy_action_support.lambda_support")
    ap.add_argument("--patience", type=int, default=40, help="epochs without a better selection score")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    ap.add_argument("--validation-runs", type=int, default=None,
                    help="SMOKE TESTS ONLY: use the first N of the frozen validation seeds (<= config n_runs; recorded)")
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
    if args.lambda_balance is None:
        args.lambda_balance = cfg["cgan_training"]["lambda_balance"]
    if args.lambda_diversity is None:
        args.lambda_diversity = cfg["cgan_training"]["lambda_diversity"]

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

    rows, fp = ml_data.load_training_rows(args.csv, expected_extra_subset=bot_runner.policy_extra())
    if fp["dataset_id"] != qck["dataset_fingerprint"]["dataset_id"]:
        raise SystemExit(f"dataset {fp['dataset_id']} differs from Q's training dataset "
                         f"{qck['dataset_fingerprint']['dataset_id']}; pass the same --csv files.")
    split = qck["split"]
    tr_rows = ml_data.select(rows, split["train"])
    va_rows, va_runs = _ordered_runs(ml_data.select(rows, split["val"]))

    cond_scaler = ml_data.Standardizer.from_json(qck["cond_scaler"])
    out_scaler_json = qck["outcome_scaler"]
    q = models.OutcomeModel(**qck["q_xa_hparams"]).to(device)
    q.load_state_dict(qck["q_xa_state"])
    models.freeze(q)

    x_tr = torch.tensor(cond_scaler.transform(ml_data.cond_matrix(tr_rows)), dtype=torch.float32, device=device)
    a_tr = _actions_n(ml_data.action_matrix(tr_rows), torch, device)

    sup_cfg = cfg["enemy_action_support"]
    lambda_support = sup_cfg["lambda_support"] if args.lambda_support is None else args.lambda_support
    support = models.SupportModel.fit(x_tr, a_tr, sup_cfg["knn_k"], sup_cfg["knn_threshold_quantile"],
                                      sup_cfg["reference_max_rows"], seed=args.seed)
    box_lo, box_hi = models.support_box(a_tr, *sup_cfg["box_quantiles"])
    print(f"action-support box (normalized) lo {[round(v, 3) for v in box_lo.tolist()]} "
          f"hi {[round(v, 3) for v in box_hi.tolist()]}; kNN threshold {support.threshold:.4f}")
    x_va = torch.tensor(cond_scaler.transform(ml_data.cond_matrix(va_rows)), dtype=torch.float32, device=device)
    refs = {
        "observed": _actions_n(ml_data.action_matrix(va_rows), torch, device),
        "heuristic": _actions_n([ml_data.heuristic_action_for_row(r) for r in va_rows], torch, device),
        "fixed": _actions_n([[cfg["enemy_controller"]["fixed"]["action"][k] for k in core.ACTION_KEYS]]
                            * len(va_rows), torch, device),
    }
    # Dequantize the integer spawn dimension of real actions so D cannot
    # separate real/fake merely by integrality.
    spawn_half_bin = 0.5 / (ml_data.ACTION_HI[-1] - ml_data.ACTION_LO[-1])

    gen = models.Generator(noise_dim=args.noise_dim, hidden=args.hidden,
                           box_lo=box_lo.tolist(), box_hi=box_hi.tolist()).to(device)
    disc = models.Discriminator(hidden=args.hidden).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    n = x_tr.shape[0]
    history = []
    best = {"score": float("inf"), "epoch": -1}
    candidates = []                     # room-level-eligible epochs at the candidate cadence
    cl_cfg = sel_cfg["closed_loop"]
    bad = 0
    for epoch in range(args.epochs):
        gen.train()
        disc.train()
        perm = torch.randperm(n, device=device)
        sums = {"d": 0.0, "g": 0.0, **{k: 0.0 for k in models.GENERATOR_LOSS_TERMS}}
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

            terms = models.generator_loss(gen, disc, q, support, x_b, args.lambda_balance,
                                          args.lambda_diversity, args.diversity_tau, lambda_support, cfg)
            opt_g.zero_grad()
            terms["total"].backward()
            opt_g.step()

            sums["d"] += d_loss.item()
            sums["g"] += terms["total"].item()
            for k in models.GENERATOR_LOSS_TERMS:
                sums[k] += terms[k].item()
            nb += 1

        diag = models.generator_diagnostics(gen, q, support, x_va, run_index=va_runs,
                                            reference_actions_n=refs, k=sel_cfg["diversity_samples"],
                                            seed=args.seed, exp_cfg=cfg)
        score = models.selection_score(diag, sel_cfg)
        rec = {"epoch": epoch, **{f"loss_{k}": v / max(nb, 1) for k, v in sums.items()},
               "val_score": score, **{f"val_{k}": v for k, v in diag.items()
                                      if not isinstance(v, dict)},
               "val_balance_ref": diag["balance_ref"], "val_ood_rate_ref": diag["ood_rate_ref"]}
        history.append(rec)
        eligible = models.selection_eligible(diag, epoch, sel_cfg)
        rec["eligible"] = eligible
        if eligible and epoch % cl_cfg["candidate_every_epochs"] == 0:
            candidates.append({"epoch": epoch, "room_score": score, "diag": diag,
                               "gen": copy.deepcopy(gen.state_dict()), "disc": copy.deepcopy(disc.state_dict()),
                               "opt_g": copy.deepcopy(opt_g.state_dict()),
                               "opt_d": copy.deepcopy(opt_d.state_dict())})
        if eligible and score < best["score"] - 1e-6:
            best = {"score": score, "epoch": epoch, "diag": diag,
                    "gen": copy.deepcopy(gen.state_dict()), "disc": copy.deepcopy(disc.state_dict()),
                    "opt_g": copy.deepcopy(opt_g.state_dict()), "opt_d": copy.deepcopy(opt_d.state_dict())}
            bad = 0
        elif epoch >= sel_cfg.get("min_epoch", 0):
            bad += 1
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d} D {rec['loss_d']:.3f} adv {rec['loss_adv']:.3f} "
                  f"bal {rec['loss_balance']:.4f} div {rec['loss_diversity']:.4f} | val score {score:.4f} "
                  f"bal_pred {diag['balance_pred']:.4f} (heur {diag['balance_ref']['heuristic']:.4f}) "
                  f"div {diag['diversity']:.3f} sens {diag['sensitivity']:.3f} ood {diag['ood_rate']:.3f}")
        if bad >= args.patience:
            print(f"early stop at epoch {epoch} (best {best['epoch']})")
            break

    import selection

    def build_ckpt(c, extra=None):
        ck = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id_for(c["gen"], torch),
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
            "support_state": support.to_state(), "support_config": sup_cfg,
            "action_support_box": {"lo": box_lo.tolist(), "hi": box_hi.tolist()},
            "lambda_support": lambda_support,
            "generator_state_dict": c["gen"], "discriminator_state_dict": c["disc"],
            "q_state_dict": qck["q_xa_state"],
            "optimizer_g_state": c["opt_g"], "optimizer_d_state": c["opt_d"],
            "epoch": c["epoch"], "room_level_score": c["room_score"], "room_level_diagnostics": c["diag"],
            "seed": args.seed, "hparams": vars(args), "split": split,
            "q_gate": gate, "q_gate_overridden": (not gate["passed"]) and args.allow_failed_q_gate,
            "loss_equation": ("L_G = L_adv + lambda_balance*L_balance + lambda_diversity*L_diversity"
                              " + lambda_support*L_support"),
            "history": history,
        }
        ck.update(extra or {})
        return ck

    # ---- closed-loop candidate evaluation on the frozen validation seeds ----
    seeds = selection.validation_seeds(cfg)
    if args.validation_runs is not None:
        if not 0 < args.validation_runs <= len(seeds):
            raise SystemExit("--validation-runs must be in [1, closed_loop.n_runs]")
        seeds = seeds[:args.validation_runs]
    top = sorted(candidates, key=lambda c: c["room_score"])[:cl_cfg["top_k_candidates"]]
    table = []
    for c in sorted(top, key=lambda c: c["epoch"]):
        cdir = os.path.join(args.out_dir, "candidates", f"epoch_{c['epoch']:04d}")
        os.makedirs(cdir, exist_ok=True)
        torch.save(build_ckpt(c), os.path.join(cdir, "cgan_checkpoint.pt"))
        try:
            lm, runs = selection.rollout(cdir, seeds, cfg)
            metrics = selection.rollout_metrics(runs, cfg, lm)
        except Exception as e:                                   # incompatible / broken candidate
            metrics = {"load_error": f"{type(e).__name__}: {e}", "selection_score": float("nan")}
        reasons = selection.eligibility(metrics, c["diag"], c["epoch"], cfg)
        row = {"epoch": c["epoch"], "checkpoint_id": checkpoint_id_for(c["gen"], torch), "dir": cdir,
               "room_level_score": c["room_score"],
               "room_level": {k: c["diag"][k] for k in ("validity", "diversity", "sensitivity", "ood_rate",
                                                        "balance_pred")},
               "metrics": {k: v for k, v in metrics.items() if k != "B_r"},
               "ineligible_reasons": reasons}
        table.append(row)
        m = metrics
        if "load_error" in m:
            print(f"candidate epoch {c['epoch']:4d}: LOAD ERROR {m['load_error']}")
        else:
            print(f"candidate epoch {c['epoch']:4d}: score {m['selection_score']:.4f} mean {m['mean_B_r']:.4f} "
                  f"median {m['median_B_r']:.4f} CVaR {m['cvar_B_r']:.4f} death {m['ordinary_room_death_rate']:.4f} "
                  f"d1 {m['depth1_death_rate']:.4f} ood {m['ood_action_rate']:.4f} -> "
                  + ("ELIGIBLE" if not reasons else "ineligible: " + "; ".join(reasons)))

    report = {"q_gate": gate, "hparams": vars(args), "n_train_rows": len(tr_rows), "n_val_rows": len(va_rows),
              "dataset_id": fp["dataset_id"], "candidate_filter": {
                  "candidate_every_epochs": cl_cfg["candidate_every_epochs"],
                  "top_k_candidates": cl_cfg["top_k_candidates"], "n_room_level_candidates": len(candidates)},
              "validation_seeds": [seeds[0], seeds[-1]], "n_validation_runs": len(seeds),
              "candidates": table}
    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    try:
        if not table:
            raise selection.SelectionFailure("no room-level-eligible candidate epoch")
        chosen = selection.choose(table)
    except selection.SelectionFailure as e:
        report.update({"selection": "FAILED", "reason": str(e), "elapsed_sec": round(time.time() - t0, 2)})
        with open(os.path.join(args.out_dir, "cgan_report.json"), "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"MODEL-SELECTION FAILURE: {e}")
        return 3

    c = next(x for x in candidates if x["epoch"] == chosen["epoch"])
    ck = build_ckpt(c, {"closed_loop_selection": {
        "selected_epoch": chosen["epoch"], "selection_score": chosen["metrics"]["selection_score"],
        "rule": "mean(B_r) + cvar_coef * CVaR_alpha(B_r) on observed validation rollouts",
        "cvar_alpha": cl_cfg["cvar_alpha"], "cvar_coef": cl_cfg["cvar_coef"],
        "validation_seed_range": [seeds[0], seeds[-1]], "n_validation_runs": len(seeds),
        "candidates": [{k: v for k, v in r.items() if k != "dir"} for r in table]}})
    torch.save(ck, ckpt_path)
    with open(os.path.join(args.out_dir, "scaler.json"), "w") as f:
        json.dump({"cond_scaler": cond_scaler.to_json(), "outcome_scaler": out_scaler_json,
                   "action_lo": ml_data.ACTION_LO, "action_hi": ml_data.ACTION_HI}, f, indent=2)
    report.update({"selection": "OK", "checkpoint_id": ck["checkpoint_id"], "best_epoch": chosen["epoch"],
                   "best_score": chosen["metrics"]["selection_score"], "selected": chosen,
                   "best_diagnostics": c["diag"], "elapsed_sec": round(time.time() - t0, 2)})
    with open(os.path.join(args.out_dir, "cgan_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    m = chosen["metrics"]
    print(f"SELECTED epoch {chosen['epoch']} checkpoint {ck['checkpoint_id']}: closed-loop score "
          f"{m['selection_score']:.4f} (mean B_r {m['mean_B_r']:.4f}, CVaR {m['cvar_B_r']:.4f})")
    print(f"checkpoint -> {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
