"""
ml_data.py
==========
Torch-free dataset plumbing shared by train_q.py, train_cgan.py and the
analysis tools:

  * load_training_rows(): validates every CSV against its manifest AND the
    current code/experiment config, refuses to mix incompatible datasets,
    keeps combat rooms (boss rooms excluded by default) and returns a
    dataset fingerprint that later stages use to prove they trained on the
    same data.
  * grouped_split(): deterministic train/val/test split by complete run_id.
  * array builders for x (condition), a (action) and o (observed outcome).
  * Standardizer (fit on TRAIN rows only) and the fixed action normalization
    a_n = (a - lo) / (hi - lo) over dda_core.ACTION_BOUNDS.

Model inputs are ONLY the 9 cond_* columns and the 4 effective applied_*
action columns. player_profile (bot archetype) is never a model input.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import csv
import hashlib
import json
import math
import os
import random

import balance
import dda_core as core
import telemetry

COND_COLUMNS = list(telemetry.COND_COLUMNS)
ACTION_COLUMNS = list(telemetry.ACTION_COLUMNS)
ACTION_RAW_COLUMNS = list(telemetry.ACTION_RAW_COLUMNS)
OUTCOME_COLUMNS = list(balance.OUTCOME_COLUMNS)

ACTION_LO = [float(core.ACTION_BOUNDS[k][0]) for k in core.ACTION_KEYS]
ACTION_HI = [float(core.ACTION_BOUNDS[k][1]) for k in core.ACTION_KEYS]


class DatasetMixError(telemetry.DatasetVersionError):
    pass


def manifest_path_for(csv_path: str) -> str:
    return os.path.splitext(csv_path)[0] + ".manifest.json"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_training_rows(csv_paths: Sequence[str], include_boss: bool = False,
                       expected_extra_subset: Optional[dict] = None) -> Tuple[List[dict], dict]:
    """Returns (combat_rows, fingerprint). Raises DatasetVersionError on any
    validation failure, incompatible mix or duplicate (run_id, room_seq)."""
    if not csv_paths:
        raise DatasetMixError("no dataset CSV given")
    datasets = []
    ref_hash = None
    rows: List[dict] = []
    seen = set()
    for p in csv_paths:
        mpath = manifest_path_for(p)
        rep = telemetry.validate_dataset(p, mpath, expected_extra_subset=expected_extra_subset)
        man = telemetry.load_manifest(mpath)
        if ref_hash is None:
            ref_hash = man["config_hash"]
        elif man["config_hash"] != ref_hash:
            raise DatasetMixError(
                f"{p} has config_hash {man['config_hash']} but the first dataset has "
                f"{ref_hash}; refusing to mix incompatible configurations.")
        datasets.append({"csv": os.path.abspath(p), "sha256": file_sha256(p),
                         "n_rows": rep["n_rows"], "seed_range": man.get("seed_range"),
                         "controller_counts": rep["controller_counts"]})
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["run_id"], r["room_seq"])
                if key in seen:
                    raise DatasetMixError(f"duplicate (run_id, room_seq) {key} across datasets")
                seen.add(key)
                if r["is_combat_room"] != "1":
                    continue
                if r["room_kind"] == "boss" and not include_boss:
                    continue
                if r["room_kind"] != "boss":
                    # raw == effective for non-boss rooms (no room-kind modifier)
                    for rc, ec in zip(ACTION_RAW_COLUMNS, ACTION_COLUMNS):
                        if abs(float(r[rc]) - float(r[ec])) > 1e-6:
                            raise DatasetMixError(f"raw/effective action mismatch in {key}")
                rows.append(r)
    man0 = telemetry.load_manifest(manifest_path_for(csv_paths[0]))
    fingerprint = {
        "config_hash": ref_hash,
        "schema_version": man0["schema_version"],
        "dda_version": man0["dda_version"],
        "experiment_config_version": man0.get("experiment_config_version"),
        "experiment_config_hash": man0.get("experiment_config_hash"),
        "manifest_extra": man0.get("extra") or {},
        "include_boss": include_boss,
        "datasets": datasets,
        "dataset_id": hashlib.sha256(json.dumps(
            sorted(d["sha256"] for d in datasets)).encode()).hexdigest()[:16],
    }
    return rows, fingerprint


def grouped_split(rows: List[dict], fracs=(0.70, 0.15, 0.15), seed: int = 0) -> Dict[str, List[str]]:
    """Deterministic split of complete run_ids into train/val/test."""
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1: {fracs}")
    run_ids = sorted(set(r["run_id"] for r in rows))
    rng = random.Random(seed)
    rng.shuffle(run_ids)
    n = len(run_ids)
    n_val = max(1, int(round(n * fracs[1]))) if fracs[1] > 0 else 0
    n_test = max(1, int(round(n * fracs[2]))) if fracs[2] > 0 else 0
    if n - n_val - n_test < 1:
        raise ValueError(f"only {n} runs: too few for a train/val/test split")
    return {"val": sorted(run_ids[:n_val]), "test": sorted(run_ids[n_val:n_val + n_test]),
            "train": sorted(run_ids[n_val + n_test:])}


def select(rows: List[dict], run_ids: Sequence[str]) -> List[dict]:
    keep = set(run_ids)
    return [r for r in rows if r["run_id"] in keep]


def cond_matrix(rows) -> List[List[float]]:
    return [[float(r[c]) for c in COND_COLUMNS] for r in rows]


def action_matrix(rows) -> List[List[float]]:
    return [[float(r[c]) for c in ACTION_COLUMNS] for r in rows]


def outcome_matrix(rows, cfg=None) -> List[List[float]]:
    cfg = cfg or balance.load_experiment_config()
    return [balance.outcome_vector(r, cfg) for r in rows]


def normalize_action(a: Sequence[float]) -> List[float]:
    return [(a[i] - ACTION_LO[i]) / (ACTION_HI[i] - ACTION_LO[i]) for i in range(len(a))]


def denormalize_action(a_n: Sequence[float]) -> List[float]:
    return [ACTION_LO[i] + a_n[i] * (ACTION_HI[i] - ACTION_LO[i]) for i in range(len(a_n))]


@dataclass
class Standardizer:
    mean: List[float]
    std: List[float]
    columns: List[str]

    @staticmethod
    def fit(x: List[List[float]], columns: List[str]) -> "Standardizer":
        n, d = len(x), len(x[0])
        mean = [sum(r[i] for r in x) / n for i in range(d)]
        var = [sum((r[i] - mean[i]) ** 2 for r in x) / n for i in range(d)]
        std = [math.sqrt(v) if v > 1e-12 else 1.0 for v in var]   # zero-variance -> 1
        return Standardizer(mean, std, list(columns))

    def transform(self, x: List[List[float]]) -> List[List[float]]:
        return [[(r[i] - self.mean[i]) / self.std[i] for i in range(len(r))] for r in x]

    def inverse_row(self, r: Sequence[float]) -> List[float]:
        return [r[i] * self.std[i] + self.mean[i] for i in range(len(r))]

    def to_json(self) -> dict:
        return {"mean": self.mean, "std": self.std, "columns": self.columns}

    @staticmethod
    def from_json(d: dict) -> "Standardizer":
        return Standardizer(list(d["mean"]), list(d["std"]), list(d["columns"]))

    def check(self, columns: List[str]):
        if self.columns != list(columns):
            raise ValueError(f"scaler columns {self.columns} != expected {columns}")
        if len(self.mean) != len(columns) or len(self.std) != len(columns):
            raise ValueError("scaler length mismatch")
        if not all(math.isfinite(v) for v in self.mean + self.std) or min(self.std) <= 0:
            raise ValueError("scaler contains non-finite or non-positive values")


def heuristic_action_for_row(r: dict) -> List[float]:
    """What rule_based_dda would have chosen for this row's pre-room state
    (reference for coverage analysis; never a training label)."""
    m = core.CombatMetrics(melee_kills=int(float(r["pre_melee_kills"])),
                           magic_kills=int(float(r["pre_magic_kills"])))
    h = core.rule_based_dda(m, node_depth=int(float(r["node_depth"])),
                            hp_ratio=float(r["pre_hp_ratio"]))
    return [float(h[k]) for k in core.ACTION_KEYS]
