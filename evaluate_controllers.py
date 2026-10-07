"""
evaluate_controllers.py
=======================
Equivalent-seed comparison of DDA controllers on held-out evaluation seeds.

  python evaluate_controllers.py --controllers fixed,heuristic,cgan \
      --model-dir model_out --runs 300 --seed 9000000 --workers 6 \
      --out-dir experiments/eval_<stamp> --train-csv pilot.csv --plots

Every controller plays the SAME run_seeds (run i -> seed + i), so maps, bot
archetypes and combat RNG streams are paired across controllers. Then:

  * eval_<controller>.csv/.manifest.json  (validated telemetry)
  * model_diagnostics.json  (cgan: validity, diversity, sensitivity, jitter,
    Q-predicted balance vs. heuristic/fixed on the evaluation conditions)
  * eval_meta.json          (git commit, command, checkpoint id, config hashes)
  * analyze_results outputs (summary.json/.md, CSV tables, plots)

Safety: refuses an existing non-empty --out-dir; refuses evaluation seeds
outside experiment_config seed_blocks.evaluation (unless
--allow-seeds-outside-eval-block) and seeds that overlap any --train-csv
manifest seed_range, so final evaluation runs are never trained on.
"""

import argparse
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


def model_diagnostics(model_dir: str, cond_rows, k: int, seed: int) -> dict:
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
            "fixed": an([[cfg["controllers"]["fixed"]["action"][kk] for kk in core.ACTION_KEYS]] * len(rows))}
    d = models.generator_diagnostics(lm.generator, lm.q, lm.out_scaler, x,
                                     run_index=[r["run_id"] for r in rows], reference_actions_n=refs,
                                     k=k, seed=seed, exp_cfg=cfg)
    d["checkpoint_id"] = lm.checkpoint_id
    d["conditions_from"] = "heuristic evaluation runs (non-boss combat rooms)"
    d["q_gate"] = lm.checkpoint.get("q_gate")
    d["q_gate_overridden"] = lm.checkpoint.get("q_gate_overridden")
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description="Equivalent-seed controller evaluation.")
    ap.add_argument("--controllers", default="fixed,heuristic,cgan")
    ap.add_argument("--model-dir", default=None)
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

    ctrls = [c.strip() for c in args.controllers.split(",") if c.strip()]
    for c in ctrls:
        if c not in telemetry.CONTROLLER_TYPES:
            ap.error(f"unknown controller {c!r}")
    if "cgan" in ctrls and not args.model_dir:
        ap.error("--model-dir is required for the cgan controller")
    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir):
        ap.error(f"{args.out_dir} exists and is not empty; refusing to overwrite an experiment")
    cfg = balance.load_experiment_config()
    check_seeds(args.seed, args.runs, args.train_csv, cfg, args.allow_seeds_outside_eval_block)
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    meta = {"git_commit": telemetry.git_commit_short(), "command": " ".join(sys.argv),
            "controllers": ctrls, "seed_range": [args.seed, args.seed + args.runs - 1],
            "runs_per_controller": args.runs, "model_dir": args.model_dir,
            "experiment_config_version": cfg["experiment_config_version"],
            "experiment_config_hash": balance.experiment_config_hash(cfg),
            "schema_version": telemetry.SCHEMA_VERSION, "dda_version": telemetry.DDA_VERSION,
            "train_csvs": [os.path.abspath(p) for p in args.train_csv], "per_controller": {}}
    for c in ctrls:
        out = os.path.join(args.out_dir, f"eval_{c}.csv")
        bargs = ["--runs", str(args.runs), "--seed", str(args.seed), "--controller", c,
                 "--workers", str(args.workers), "--out", out, "--validate"]
        if c == "cgan":
            bargs += ["--model-dir", args.model_dir]
        print(f"=== {c} ===")
        bot_runner.main(bargs)
        man = telemetry.load_manifest(ml_data.manifest_path_for(out))
        meta["per_controller"][c] = {"csv": out, "config_hash": man["config_hash"],
                                     "controllers": man.get("controllers"), "run_stats": man.get("run_stats")}
    hashes = {v["config_hash"] for v in meta["per_controller"].values()}
    if len(hashes) != 1:
        raise SystemExit(f"controller datasets disagree on config_hash: {hashes}")

    diag = None
    if "cgan" in ctrls:
        import csv
        src = "heuristic" if "heuristic" in ctrls else ctrls[0]
        with open(os.path.join(args.out_dir, f"eval_{src}.csv"), newline="") as f:
            cond_rows = [r for r in csv.DictReader(f) if r["is_combat_room"] == "1" and r["room_kind"] != "boss"]
        try:
            diag = model_diagnostics(args.model_dir, cond_rows, cfg["cgan_selection"]["diversity_samples"], 0)
            meta["checkpoint_id"] = diag["checkpoint_id"]
        except Exception as e:     # incompatible model: evaluation still reports the fallbacks
            diag = {"error": f"{type(e).__name__}: {e}"}
        with open(os.path.join(args.out_dir, "model_diagnostics.json"), "w") as f:
            json.dump(diag, f, indent=2, default=float)

    analyze_results.analyze(sorted(os.path.join(args.out_dir, f"eval_{c}.csv") for c in ctrls),
                            args.out_dir, "bot", args.bootstrap, 0, args.plots,
                            diag if diag and "error" not in diag else None)
    meta["elapsed_sec"] = round(time.time() - t0, 2)
    with open(os.path.join(args.out_dir, "eval_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.out_dir, "summary.md")) as f:
        print(f.read())
    print(f"evaluation complete -> {args.out_dir} ({meta['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
