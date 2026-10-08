"""
freeze_experiment.py
====================
Records and verifies the frozen experimental contract (experiment_freeze.json).

  python freeze_experiment.py record    # after status is FROZEN and committed; clean tree required
  python freeze_experiment.py verify    # used by scripts/02_collect.sh; non-zero exit on any mismatch

The record pins the exact commit at which experiment_config.json was frozen,
the schema/DDA/bot-policy versions, the experiment-config and bot config
hashes, and a readable copy of everything the principal collection depends
on (targets and weights, fixed_default, shared boss, collection mixtures,
gates and split seeds, action-support limits, loot horizon). `verify` fails if
the config or any hashed constant changed after the freeze, so these values
cannot drift once collection starts.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import balance
import bot_runner
import dda_core as core
import telemetry

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FREEZE_FILE = os.path.join(_THIS_DIR, "experiment_freeze.json")
CONTRACT_FILES = ["experiment_config.json", "dda_core.py", "telemetry.py", "balance.py", "bot_runner.py",
                  "controllers.py", "rewards.py", "node_map_gen.py", "loot_dataset.py"]


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=_THIS_DIR, capture_output=True, text=True, timeout=20).stdout.strip()


def current_contract() -> dict:
    cfg = balance.load_experiment_config()
    extra = bot_runner.policy_extra()
    extra["experiment_config_hash"] = balance.experiment_config_hash(cfg)
    return {
        "experiment_config_version": cfg["experiment_config_version"],
        "experiment_config_status": cfg["status"],
        "experiment_config_hash": balance.experiment_config_hash(cfg),
        "bot_config_hash": core.config_hash(extra=extra),
        "schema_version": telemetry.SCHEMA_VERSION, "dda_version": telemetry.DDA_VERSION,
        "bot_policy_version": telemetry.BOT_POLICY_VERSION, "game_version": telemetry.GAME_VERSION,
        "enemy_balance_targets": cfg["enemy_balance_targets"],
        "loot_horizon_balance_targets": cfg["loot_horizon_balance_targets"],
        "fixed_default": cfg["enemy_controller"]["fixed"],
        "shared_boss": cfg["shared_boss"],
        "principal_collection": cfg["principal_collection"],
        "enemy_gate": cfg["q_gate"], "loot_gate": cfg["q_loot_gate"],
        "loot_gate_split_seeds": cfg["q_loot_gate"]["split_seeds"],
        "enemy_action_support": cfg["enemy_action_support"],
        "loot_action_support": cfg["loot_action_support"],
        "loot_horizon": cfg["loot_horizon"],
        "seed_blocks": cfg["seed_blocks"],
    }


def record() -> dict:
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("working tree has uncommitted changes; commit before recording the freeze")
    c = current_contract()
    if not c["experiment_config_status"].startswith("FROZEN"):
        raise SystemExit("experiment_config.json status is not FROZEN")
    rec = {"frozen_commit": _git("rev-parse", "HEAD"),
           "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "contract_files": CONTRACT_FILES, **c}
    with open(FREEZE_FILE, "w") as f:
        json.dump(rec, f, indent=2)
    return rec


def verify() -> dict:
    if not os.path.exists(FREEZE_FILE):
        raise SystemExit(f"no freeze record ({FREEZE_FILE}); run `python freeze_experiment.py record`")
    with open(FREEZE_FILE) as f:
        rec = json.load(f)
    cur = current_contract()
    problems = [k for k in cur if json.dumps(cur[k], sort_keys=True) != json.dumps(rec.get(k), sort_keys=True)]
    head = _git("rev-parse", "HEAD")
    if subprocess.run(["git", "merge-base", "--is-ancestor", rec["frozen_commit"], head],
                      cwd=_THIS_DIR).returncode != 0:
        problems.append("frozen_commit is not an ancestor of HEAD")
    changed = _git("diff", "--name-only", rec["frozen_commit"], "HEAD", "--", *CONTRACT_FILES)
    if changed:
        problems.append(f"contract files changed since the freeze: {changed.split()}")
    if problems:
        raise SystemExit("FREEZE VIOLATION: " + "; ".join(problems))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="Record / verify the frozen experimental contract.")
    ap.add_argument("cmd", choices=["record", "verify", "show"])
    args = ap.parse_args(argv)
    if args.cmd == "record":
        rec = record()
        print(f"freeze recorded at {rec['frozen_commit']} -> {FREEZE_FILE}")
    elif args.cmd == "verify":
        rec = verify()
        print(f"freeze OK: {rec['experiment_config_version']} at {rec['frozen_commit'][:12]} "
              f"(config {rec['experiment_config_hash']}, bot {rec['bot_config_hash']})")
    else:
        print(json.dumps(current_contract(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
