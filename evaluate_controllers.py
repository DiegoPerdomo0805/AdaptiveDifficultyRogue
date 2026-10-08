"""
evaluate_controllers.py
=======================
Equivalent-seed comparison on held-out evaluation seeds, along ONE axis at a
time so enemy-controller and reward-policy conclusions stay separate:

  --axis enemy   vary the enemy controller (fixed, heuristic, cgan), hold the
                 reward policy fixed (--reward-policy, default heuristic)
  --axis reward  vary the reward policy (fixed_uniform, heuristic[, learned_loot]),
                 hold the enemy controller fixed (--enemy-controller, default heuristic)

  python evaluate_controllers.py --axis enemy --controllers fixed,heuristic,cgan \
      --model-dir model_out --runs 200 --seed 9000000 --out-dir experiments/eval_enemy \
      --train-csv pilot.csv --plots
  python evaluate_controllers.py --axis reward --reward-policies fixed_uniform,heuristic \
      --runs 200 --seed 9000000 --out-dir experiments/eval_reward --q-loot-dir q_loot_out

Every condition plays the SAME run_seeds (run i -> seed + i): identical maps,
bot archetypes, starting loadouts and RNG streams. Boss rooms use the shared
boss policy in every condition.

Outputs: eval_<condition>.csv/.manifest.json, model_diagnostics.json,
eval_meta.json (git commit, command, checkpoints, hashes, predicted-vs-observed)
and the analyze_results outputs.

Safety: refuses an existing non-empty --out-dir; refuses evaluation seeds
outside experiment_config seed_blocks.evaluation (unless
--allow-seeds-outside-eval-block) or overlapping any --train-csv manifest;
learned_loot is only evaluated with a loadable checkpoint produced under a
passed Q_loot gate.
"""

import argparse
import csv
import json
import os
import sys
import time

import analyze_results
import balance
import bot_runner
import ml_data
import telemetry


def _ranges(seed_range):
    """Normalize a manifest seed_range ([a, b] or a list of them)."""
    if not seed_range:
        return []
    if isinstance(seed_range[0], list):
        return [r for sr in seed_range for r in _ranges(sr)]
    return [(int(seed_range[0]), int(seed_range[1]))]


def check_seeds(seed: int, runs: int, train_csvs, cfg, allow_outside: bool):
    lo, hi = seed, seed + runs - 1
    eb = cfg.get("seed_blocks", {}).get("evaluation")
    if eb and not allow_outside and not (eb[0] <= lo and hi <= eb[1]):
        raise SystemExit(f"evaluation seeds [{lo}, {hi}] are outside seed_blocks.evaluation {eb}")
    for p in train_csvs or []:
        man = telemetry.load_manifest(ml_data.manifest_path_for(p))
        for a, b in _ranges(man.get("seed_range")):
            if not (hi < a or b < lo):
                raise SystemExit(f"evaluation seeds [{lo}, {hi}] overlap training dataset {p} "
                                 f"seed_range [{a}, {b}]")


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def enemy_model_diagnostics(model_dir: str, cond_rows, k: int, seed: int) -> dict:
    import torch
    import controllers
    import dda_core as core
    import models
    mr = controllers.load_model_runtime()
    lm = mr.LoadedModels.load(model_dir, load_q=True)
    cfg = balance.load_experiment_config()
    rows = sorted(cond_rows, key=lambda r: (r["run_id"], int(r["room_seq"])))
    x = torch.tensor(lm.cond_scaler.transform(ml_data.cond_matrix(rows)), dtype=torch.float32)

    def an(m):
        return torch.tensor([ml_data.normalize_action(a) for a in m], dtype=torch.float32)
    refs = {"heuristic": an([ml_data.heuristic_action_for_row(r) for r in rows]),
            "fixed": an([[cfg["enemy_controller"]["fixed"]["action"][kk] for kk in core.ACTION_KEYS]] * len(rows))}
    d = models.generator_diagnostics(lm.generator, lm.q, lm.support, x,
                                     run_index=[r["run_id"] for r in rows], reference_actions_n=refs,
                                     k=k, seed=seed, exp_cfg=cfg)
    d["checkpoint_id"] = lm.checkpoint_id
    d["conditions_from"] = "heuristic evaluation runs (ordinary combat rooms)"
    d["q_gate"] = lm.checkpoint.get("q_gate")
    d["q_gate_overridden"] = lm.checkpoint.get("q_gate_overridden")
    return d


def enemy_predicted_vs_observed(q_source: str, per_condition_rows: dict) -> dict:
    """Q(x, a_applied) vs. observed outcomes on every condition's ordinary rooms."""
    import torch
    import models
    import controllers
    cfg = balance.load_experiment_config()
    mr = controllers.load_model_runtime()
    if os.path.exists(os.path.join(q_source, mr.CHECKPOINT_FILE)):
        lm = mr.LoadedModels.load(q_source, load_q=True)
        q, scaler, label = lm.q, lm.cond_scaler, f"Q_enemy from cGAN checkpoint {lm.checkpoint_id}"
    else:
        ck = torch.load(os.path.join(q_source, "q_checkpoint.pt"), map_location="cpu", weights_only=True)
        q = models.freeze(models.OutcomeModel(**ck["q_xa_hparams"]))
        q.load_state_dict(ck["q_xa_state"])
        scaler, label = ml_data.Standardizer.from_json(ck["cond_scaler"]), f"Q_enemy from {q_source}"
    out = {"model": label, "by_condition": {}}
    for cond, rows in per_condition_rows.items():
        rs = [r for r in rows if r["room_kind"] == "enemy"]
        if not rs:
            continue
        x = torch.tensor(scaler.transform(ml_data.cond_matrix(rs)), dtype=torch.float32)
        a = torch.tensor([ml_data.normalize_action(v) for v in ml_data.action_matrix(rs)], dtype=torch.float32)
        with torch.no_grad():
            pred = models.outcomes_physical(q(x, a)).tolist()
        obs = ml_data.outcome_matrix(rs, cfg)
        d = {}
        for j, col in enumerate(balance.OUTCOME_COLUMNS):
            d["mean_" + col] = {"predicted": sum(p[j] for p in pred) / len(pred),
                                "observed": sum(o[j] for o in obs) / len(obs)}
        d["mean_balance_distance"] = {
            "predicted": sum(balance.expected_balance_distance(p, cfg) for p in pred) / len(pred),
            "observed": sum(balance.balance_distance(o, cfg) for o in obs) / len(obs)}
        # PRIMARY: run-level aggregation, identical definitions for predicted
        # and observed (analyze_results.run_level_vector); Q predicts per-room
        # ratios, so the predicted damage share / efficiency of a run is the
        # mean of its rooms' predictions (Q does not predict raw damage).
        by_run = {}
        for r, p in zip(rs, pred):
            by_run.setdefault(r["run_id"], ([], []))
            by_run[r["run_id"]][0].append(r)
            by_run[r["run_id"]][1].append(p)
        pv, ov = [], []
        for rows_r, preds_r in by_run.values():
            pv.append([sum(p[j] for p in preds_r) / len(preds_r) for j in range(5)])
            ov.append(analyze_results.run_level_vector(rows_r, cfg))
        pb = [balance.balance_distance(v, cfg) for v in pv]
        ob = [balance.balance_distance(v, cfg) for v in ov]
        n = len(pb)
        mpb, mob = sum(pb) / n, sum(ob) / n
        cov = sum((a - mpb) * (b - mob) for a, b in zip(pb, ob))
        den = (sum((a - mpb) ** 2 for a in pb) * sum((b - mob) ** 2 for b in ob)) ** 0.5
        run_level = {"n_runs": n,
                     "B_r": {"predicted_mean": mpb, "observed_mean": mob,
                             "mae": sum(abs(a - b) for a, b in zip(pb, ob)) / n,
                             "bias": mpb - mob, "pearson_r": cov / den if den > 0 else float("nan")}}
        for j, c in enumerate(analyze_results.RUN_LEVEL_COLUMNS):
            run_level[c] = {"predicted_mean": sum(v[j] for v in pv) / n, "observed_mean": sum(v[j] for v in ov) / n,
                            "mae": sum(abs(a[j] - b[j]) for a, b in zip(pv, ov)) / n,
                            "bias": sum(a[j] - b[j] for a, b in zip(pv, ov)) / n}
        d["run_level_B_r_mean"] = {"predicted": mpb, "observed": mob}
        out.setdefault("run_level", {})[cond] = run_level
        out["by_condition"][cond] = d
    out["note"] = ("PRIMARY = run_level (Q room predictions aggregated per run like the observed run-level "
                   "metric; calibration = bias/MAE/Pearson r across runs). Room-level rows are diagnostic: "
                   "mean_balance_distance predicted = expected loss with death ~ Bernoulli(P(death)).")
    return out


def loot_predicted_vs_observed(q_loot_dir: str, per_condition_rows: dict) -> dict:
    import loot_dataset
    import controllers
    mr = controllers.load_model_runtime()
    qp = mr.QLootPredictor.load(q_loot_dir)
    cfg = balance.load_experiment_config()
    out = {"model": f"Q_loot variant {qp.variant} from {q_loot_dir}", "by_condition": {}}
    for cond, rows in per_condition_rows.items():
        ev = [e for e in loot_dataset.derive_from_rows(rows, cfg)
              if e["selected_category"] != "none" and e["horizon_combat_rooms"] >= 1]
        if not ev:
            continue
        evs = [{k: str(v) for k, v in e.items()} for e in ev]
        pred = qp.predict_rows(evs)
        obs = [[float(e["h_" + c]) for c in balance.LOOT_OUTCOME_COLUMNS] for e in ev]
        d = {}
        for j, col in enumerate(balance.LOOT_OUTCOME_COLUMNS):
            d["mean_" + col] = {"predicted": sum(p[j] for p in pred) / len(pred),
                                "observed": sum(o[j] for o in obs) / len(obs)}
        d["delayed_utility"] = {
            "predicted": -sum(balance.expected_balance_distance(p, cfg, "loot") for p in pred) / len(pred),
            "observed": -sum(balance.balance_distance(o, cfg, "loot") for o in obs) / len(obs)}
        mp = [sum(p[j] for p in pred) / len(pred) for j in range(len(pred[0]))]
        mo = [sum(o[j] for o in obs) / len(obs) for j in range(len(obs[0]))]
        d["rate_level_utility"] = {"predicted": -balance.balance_distance(mp, cfg, "loot"),
                                   "observed": -balance.balance_distance(mo, cfg, "loot")}
        out["by_condition"][cond] = d
    out["note"] = ("utility = -horizon balance distance; predicted = expected loss with death_within ~ "
                   "Bernoulli(P), comparable to the observed mean over 0/1 outcomes")
    return out


def loot_model_diagnostics(loot_model_dir: str, loot_rows, k: int = 8, seed: int = 0) -> dict:
    import torch
    import controllers
    import models
    mr = controllers.load_model_runtime()
    lm = mr.LoadedLootModel.load(loot_model_dir)
    cs = lm.cond_scaler.transform([[float(r[c]) for c in ml_data.COND_COLUMNS] for r in loot_rows])
    feats = torch.tensor([list(c) + ml_data.equip_onehot({s: r["equip_" + s] for s in ("weapon", "spell", "armor", "boots")})
                          for c, r in zip(cs, loot_rows)], dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        ps = torch.stack([lm.generator(feats, torch.randn(len(loot_rows), lm.noise_dim, generator=g))
                          for _ in range(k)])
    p = ps.mean(dim=0)
    div = sum(float((ps[i] - ps[j]).abs().mean()) for i in range(k) for j in range(i + 1, k)) / (k * (k - 1) / 2)
    return {"checkpoint_id": lm.checkpoint_id, "n_conditions": len(loot_rows),
            "entropy": float(models.entropy(p).mean()), "repeated_noise_diversity": div,
            "max_mean_prob": float(p.mean(dim=0).max()),
            "valid": bool(torch.isfinite(ps).all() and (ps >= 0).all()
                          and ((ps.sum(dim=2) - 1).abs() < 1e-4).all())}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Equivalent-seed evaluation along one axis.")
    ap.add_argument("--axis", choices=["enemy", "reward"], default="enemy")
    ap.add_argument("--controllers", default="fixed,heuristic,cgan", help="enemy axis conditions")
    ap.add_argument("--reward-policy", default="heuristic", help="reward policy held fixed on the enemy axis")
    ap.add_argument("--reward-policies", default="fixed_uniform,heuristic", help="reward axis conditions")
    ap.add_argument("--enemy-controller", default="heuristic", help="enemy controller held fixed on the reward axis")
    ap.add_argument("--model-dir", default=None, help="enemy cGAN directory")
    ap.add_argument("--q-dir", default=None, help="Q_enemy directory for predicted-vs-observed (if no cGAN)")
    ap.add_argument("--loot-model-dir", default=None, help="learned-loot directory (gate-passed only)")
    ap.add_argument("--q-loot-dir", default=None, help="Q_loot directory for predicted-vs-observed utility")
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=9000000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-csv", action="append", default=[],
                    help="training dataset(s); evaluation seeds must not overlap their seed ranges")
    ap.add_argument("--allow-seeds-outside-eval-block", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args(argv)

    if args.axis == "enemy":
        conds = [c.strip() for c in args.controllers.split(",") if c.strip()]
        allowed, group_by = telemetry.CONTROLLER_TYPES, "enemy_controller"
        if args.reward_policy not in telemetry.REWARD_POLICIES:
            ap.error(f"unknown reward policy {args.reward_policy!r}")
    else:
        conds = [c.strip() for c in args.reward_policies.split(",") if c.strip()]
        allowed, group_by = telemetry.REWARD_POLICIES, "reward_policy"
        if args.enemy_controller not in telemetry.CONTROLLER_TYPES:
            ap.error(f"unknown enemy controller {args.enemy_controller!r}")
    for c in conds:
        if c not in allowed:
            ap.error(f"unknown condition {c!r} for the {args.axis} axis")
    uses_cgan = "cgan" in conds if args.axis == "enemy" else args.enemy_controller == "cgan"
    if uses_cgan and not args.model_dir:
        ap.error("--model-dir is required for the cgan controller")
    uses_loot_model = "learned_loot" in conds if args.axis == "reward" else args.reward_policy == "learned_loot"
    if uses_loot_model:
        import controllers
        if not args.loot_model_dir:
            ap.error("learned_loot requires --loot-model-dir (only exists if the Q_loot gate passed)")
        try:
            controllers.load_model_runtime().LoadedLootModel.load(args.loot_model_dir)
        except Exception as e:
            ap.error(f"learned_loot checkpoint is not usable ({type(e).__name__}: {e}); "
                     f"it is evaluated only when its gate passed")
    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir):
        ap.error(f"{args.out_dir} exists and is not empty; refusing to overwrite an experiment")
    cfg = balance.load_experiment_config()
    check_seeds(args.seed, args.runs, args.train_csv, cfg, args.allow_seeds_outside_eval_block)
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    meta = {"git_commit": telemetry.git_commit_short(), "command": " ".join(sys.argv),
            "axis": args.axis, "group_by": group_by, "conditions": conds,
            "held_fixed": ({"reward_policy": args.reward_policy} if args.axis == "enemy"
                           else {"enemy_controller": args.enemy_controller}),
            "seed_range": [args.seed, args.seed + args.runs - 1], "runs_per_condition": args.runs,
            "model_dir": args.model_dir, "loot_model_dir": args.loot_model_dir,
            "experiment_config_version": cfg["experiment_config_version"],
            "experiment_config_hash": balance.experiment_config_hash(cfg),
            "schema_version": telemetry.SCHEMA_VERSION, "dda_version": telemetry.DDA_VERSION,
            "train_csvs": [os.path.abspath(p) for p in args.train_csv], "per_condition": {}}
    per_rows = {}
    for c in conds:
        out = os.path.join(args.out_dir, f"eval_{c}.csv")
        ctrl = c if args.axis == "enemy" else args.enemy_controller
        rp = args.reward_policy if args.axis == "enemy" else c
        bargs = ["--runs", str(args.runs), "--seed", str(args.seed), "--enemy-controller", ctrl,
                 "--reward-policy", rp, "--workers", str(args.workers), "--out", out, "--validate"]
        if ctrl == "cgan":
            bargs += ["--model-dir", args.model_dir]
        if rp == "learned_loot":
            bargs += ["--loot-model-dir", args.loot_model_dir]
        print(f"=== {c} (enemy={ctrl}, reward={rp}) ===")
        bot_runner.main(bargs)
        man = telemetry.load_manifest(ml_data.manifest_path_for(out))
        meta["per_condition"][c] = {"csv": out, "config_hash": man["config_hash"],
                                    "controllers": man.get("controllers"),
                                    "reward_policies": man.get("reward_policies"),
                                    "run_stats": man.get("run_stats")}
        per_rows[c] = _read(out)
    hashes = {v["config_hash"] for v in meta["per_condition"].values()}
    if len(hashes) != 1:
        raise SystemExit(f"condition datasets disagree on config_hash: {hashes}")

    diag = None
    if args.axis == "enemy" and "cgan" in conds:
        src = "heuristic" if "heuristic" in conds else conds[0]
        cond_rows = [r for r in per_rows[src] if r["room_kind"] == "enemy"]
        try:
            diag = enemy_model_diagnostics(args.model_dir, cond_rows, cfg["cgan_selection"]["diversity_samples"], 0)
            meta["checkpoint_id"] = diag["checkpoint_id"]
        except Exception as e:     # incompatible model: evaluation still reports the fallbacks
            diag = {"error": f"{type(e).__name__}: {e}"}
    if args.axis == "reward" and "learned_loot" in conds:
        loot_rows = [r for r in per_rows["heuristic" if "heuristic" in conds else conds[0]] if r["loot_event_id"]]
        diag = loot_model_diagnostics(args.loot_model_dir, loot_rows)
    if diag is not None:
        with open(os.path.join(args.out_dir, "model_diagnostics.json"), "w") as f:
            json.dump(diag, f, indent=2, default=float)

    extra = {}
    if args.axis == "enemy" and (args.model_dir or args.q_dir):
        try:
            extra["predicted_vs_observed"] = enemy_predicted_vs_observed(
                args.model_dir if uses_cgan else args.q_dir, per_rows)
        except Exception as e:
            extra["predicted_vs_observed"] = {"error": f"{type(e).__name__}: {e}", "by_condition": {}}
    if args.axis == "reward" and args.q_loot_dir:
        extra["predicted_vs_observed"] = loot_predicted_vs_observed(args.q_loot_dir, per_rows)
    meta.update(extra)

    analyze_results.analyze(sorted(os.path.join(args.out_dir, f"eval_{c}.csv") for c in conds),
                            args.out_dir, "bot", args.bootstrap, 0, args.plots,
                            diag if diag and "error" not in diag else None, group_by, extra)
    meta["elapsed_sec"] = round(time.time() - t0, 2)
    with open(os.path.join(args.out_dir, "eval_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=float)
    with open(os.path.join(args.out_dir, "summary.md")) as f:
        print(f.read())
    print(f"evaluation complete -> {args.out_dir} ({meta['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
