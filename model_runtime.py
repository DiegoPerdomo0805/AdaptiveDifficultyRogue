"""
model_runtime.py
=================
Loads a trained outcome-guided cGAN checkpoint (train_cgan.py) for use by
controllers.CGANController (bot_runner.py, evaluate_controllers.py and
rogue.py), independent of the current working directory.

LoadedModels.load() is the consumer-side guardian. Before a model may be
used it verifies, against the CURRENT code and experiment config:

  * checkpoint schema version
  * telemetry schema version and DDA version
  * condition / action / outcome dimensions and their EXACT ordered columns
  * action bounds
  * config_hash (recomputed from dda_core with the dataset's recorded extra)
  * experiment config version and hash (balance targets / controllers)
  * scaler fields (names, lengths, finite, positive std)

Any mismatch raises ModelVersionError; the controller then falls back to the
heuristic and records the reason. predict() raises InvalidModelOutput for
non-finite or out-of-bounds outputs (also a recorded fallback).

The checkpoint is loaded with torch.load(weights_only=True): it contains only
tensors and plain Python containers.
"""

from typing import List, Optional
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import balance
import dda_core as core
import ml_data
import telemetry

CHECKPOINT_SCHEMA_VERSION = "cgan_outcome_v4"
CHECKPOINT_FILE = "cgan_checkpoint.pt"


class ModelVersionError(Exception):
    pass


class InvalidModelOutput(ValueError):
    pass


class OODAction(InvalidModelOutput):
    """Generated action lies outside the observed (x, a) training support."""


def verify_checkpoint(ck: dict, exp_cfg: Optional[dict] = None):
    """Raises ModelVersionError naming the first incompatibility."""
    exp_cfg = exp_cfg or balance.load_experiment_config()

    def need(key, expected):
        got = ck.get(key)
        if got != expected:
            raise ModelVersionError(
                f"checkpoint {key}={got!r} does not match the current code's {expected!r}; "
                f"refusing to load an incompatible model (retrain against the current schema).")

    need("checkpoint_schema_version", CHECKPOINT_SCHEMA_VERSION)
    need("dataset_schema_version", telemetry.SCHEMA_VERSION)
    need("dda_version", telemetry.DDA_VERSION)
    need("cond_dim", core.COND_DIM)
    need("target_dim", core.ACTION_DIM)
    need("outcome_dim", balance.OUTCOME_DIM)
    need("cond_columns", ml_data.COND_COLUMNS)
    need("action_columns", ml_data.ACTION_COLUMNS)
    need("action_keys", core.ACTION_KEYS)
    need("outcome_columns", balance.OUTCOME_COLUMNS)
    need("action_lo", ml_data.ACTION_LO)
    need("action_hi", ml_data.ACTION_HI)
    need("experiment_config_version", exp_cfg["experiment_config_version"])
    need("experiment_config_hash", balance.experiment_config_hash(exp_cfg))
    recomputed = core.config_hash(extra=ck.get("dataset_extra") or {})
    if ck.get("config_hash") != recomputed:
        raise ModelVersionError(
            f"checkpoint config_hash={ck.get('config_hash')!r} but the current dda_core constants "
            f"give {recomputed!r} for the same dataset extra; gameplay/DDA constants changed since "
            f"training -- refusing to load.")
    try:
        ml_data.Standardizer.from_json(ck["cond_scaler"]).check(ml_data.COND_COLUMNS)
        ml_data.Standardizer.from_json(ck["outcome_scaler"]).check(balance.OUTCOME_COLUMNS[1:])
    except (KeyError, TypeError, ValueError) as e:
        raise ModelVersionError(f"invalid scaler in checkpoint: {e}")
    for key in ("generator_state_dict", "generator_hparams", "checkpoint_id", "support_state",
                "action_support_box"):
        if key not in ck:
            raise ModelVersionError(f"checkpoint is missing {key!r}")


class LoadedModels:
    def __init__(self, generator, cond_scaler, checkpoint: dict, q=None, support=None,
                 ood_factor: float = 1.5):
        self.generator = generator
        self.cond_scaler = cond_scaler
        self.checkpoint = checkpoint
        self.q = q
        self.support = support
        self.ood_factor = float(ood_factor)
        self.noise_dim = generator.noise_dim
        self.checkpoint_id = checkpoint["checkpoint_id"]

    @staticmethod
    def load(model_dir: str, load_q: bool = False) -> "LoadedModels":
        import torch
        import models

        ckpt_path = os.path.join(model_dir, CHECKPOINT_FILE)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        verify_checkpoint(ck)

        gen = models.Generator(**ck["generator_hparams"])
        gen.load_state_dict(ck["generator_state_dict"])
        gen.eval()
        q = None
        if load_q:
            q = models.OutcomeModel(**ck["q_hparams"])
            q.load_state_dict(ck["q_state_dict"])
            models.freeze(q)
        support = models.SupportModel.from_state(ck["support_state"])
        factor = balance.load_experiment_config()["enemy_action_support"]["runtime_ood_factor"]
        return LoadedModels(gen, ml_data.Standardizer.from_json(ck["cond_scaler"]), ck, q, support, factor)

    def predict(self, cond_vec: List[float], noise: Optional[List[float]] = None) -> dict:
        """One bounded, finite action for one condition vector. Raises
        ModelVersionError for a wrong-shaped/non-finite input and
        InvalidModelOutput for a non-finite/out-of-bounds output."""
        import torch
        if len(cond_vec) != core.COND_DIM:
            raise ModelVersionError(
                f"predict() called with a {len(cond_vec)}-dim condition vector, expected {core.COND_DIM}.")
        if not all(math.isfinite(float(v)) for v in cond_vec):
            raise ModelVersionError("predict() called with a non-finite condition vector")
        x = torch.tensor(self.cond_scaler.transform([list(map(float, cond_vec))]), dtype=torch.float32)
        z = None
        if noise is not None:
            if len(noise) != self.noise_dim:
                raise ModelVersionError(f"noise has {len(noise)} dims, expected {self.noise_dim}")
            z = torch.tensor([noise], dtype=torch.float32)
        with torch.no_grad():
            a_t = self.generator(x, z)
            ratio = float(self.support.distance(x, a_t)[0]) / self.support.threshold
        a_n = a_t[0].tolist()
        raw = ml_data.denormalize_action(a_n)
        if not all(math.isfinite(v) for v in raw):
            raise InvalidModelOutput(f"non-finite generator output {raw}")
        for i, k in enumerate(core.ACTION_KEYS):
            lo, hi = core.ACTION_BOUNDS[k]
            if not (lo - 1e-6 <= raw[i] <= hi + 1e-6):
                raise InvalidModelOutput(f"{k}={raw[i]} outside design bounds [{lo}, {hi}]")
        if not math.isfinite(ratio) or ratio > self.ood_factor:
            raise OODAction(f"support_ratio={ratio:.3f} > {self.ood_factor}")
        out = core.finalize_action(dict(zip(core.ACTION_KEYS, raw)))
        out["_support_distance"] = ratio      # distance / training threshold (logged)
        return out


# ---------------------------------------------------------------------------
# Loot models
# ---------------------------------------------------------------------------

LOOT_POLICY_SCHEMA_VERSION = "loot_policy_v1"
LOOT_POLICY_FILE = "loot_policy_checkpoint.pt"
Q_LOOT_FILE = "q_loot_checkpoint.pt"


def _verify_loot_common(ck: dict, exp_cfg: dict, label: str):
    def need(key, expected):
        if ck.get(key) != expected:
            raise ModelVersionError(f"{label} {key}={ck.get(key)!r} does not match the current {expected!r}")
    need("schema_version", telemetry.SCHEMA_VERSION)
    need("experiment_config_version", exp_cfg["experiment_config_version"])
    need("experiment_config_hash", balance.experiment_config_hash(exp_cfg))
    need("outcome_columns", balance.LOOT_OUTCOME_COLUMNS)
    need("cond_columns", ml_data.COND_COLUMNS)
    need("item_ids", core.ALL_ITEM_IDS)
    need("slots", core.SLOTS)
    try:
        ml_data.Standardizer.from_json(ck["cond_scaler"]).check(ml_data.COND_COLUMNS)
    except (KeyError, TypeError, ValueError) as e:
        raise ModelVersionError(f"invalid {label} scaler: {e}")


class QLootPredictor:
    """Frozen Q_loot variant for predicted-vs-observed delayed utility."""

    def __init__(self, ck: dict, variant: str):
        import models
        self.ck = ck
        self.variant = variant
        self.net = models.BoundedOutcomeNet(**ck["variant_hparams"][variant])
        self.net.load_state_dict(ck["variant_states"][variant])
        models.freeze(self.net)
        self.cond_scaler = ml_data.Standardizer.from_json(ck["cond_scaler"])

    @staticmethod
    def load(q_loot_dir: str, variant: Optional[str] = None) -> "QLootPredictor":
        import torch
        ck = torch.load(os.path.join(q_loot_dir, Q_LOOT_FILE), map_location="cpu", weights_only=True)
        if ck.get("checkpoint_schema") != "q_loot_v1":
            raise ModelVersionError(f"unexpected Q_loot checkpoint schema {ck.get('checkpoint_schema')!r}")
        _verify_loot_common(ck, balance.load_experiment_config(), "Q_loot checkpoint")
        variant = variant or ck["gate"].get("chosen_variant") or "xcie"
        return QLootPredictor(ck, variant)

    def predict_rows(self, rows: List[dict]) -> List[List[float]]:
        """Physical predictions [P(death_within), end_hp, ...] for loot-dataset rows."""
        import torch
        import models
        cond = self.cond_scaler.transform([[float(r[c]) for c in ml_data.COND_COLUMNS] for r in rows])
        feats = [ml_data.loot_features(cs, ml_data.loot_row_equip_before(r), r["selected_category"],
                                       r["selected_item_id"], self.variant) for cs, r in zip(cond, rows)]
        with torch.no_grad():
            return models.outcomes_physical(self.net(torch.tensor(feats, dtype=torch.float32))).tolist()


class LoadedLootModel:
    """G_loot for the learned_loot reward policy (only exists if the gate passed)."""

    def __init__(self, gen, cond_scaler, checkpoint: dict):
        self.generator = gen
        self.cond_scaler = cond_scaler
        self.checkpoint = checkpoint
        self.noise_dim = gen.noise_dim
        self.checkpoint_id = checkpoint["checkpoint_id"]

    @staticmethod
    def load(model_dir: str) -> "LoadedLootModel":
        import torch
        import models
        path = os.path.join(model_dir, LOOT_POLICY_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No learned-loot checkpoint at {path}")
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck.get("checkpoint_schema_version") != LOOT_POLICY_SCHEMA_VERSION:
            raise ModelVersionError(f"loot checkpoint schema {ck.get('checkpoint_schema_version')!r} != "
                                    f"{LOOT_POLICY_SCHEMA_VERSION!r}")
        exp_cfg = balance.load_experiment_config()
        _verify_loot_common(ck, exp_cfg, "loot checkpoint")
        if not ck.get("q_loot_gate", {}).get("passed"):
            raise ModelVersionError("loot checkpoint was not produced under a passed Q_loot gate")
        recomputed = core.config_hash(extra=ck.get("dataset_extra") or {})
        if ck.get("config_hash") != recomputed:
            raise ModelVersionError("loot checkpoint config_hash does not match the current constants")
        gen = models.LootGenerator(**ck["generator_hparams"])
        gen.load_state_dict(ck["generator_state_dict"])
        gen.eval()
        return LoadedLootModel(gen, ml_data.Standardizer.from_json(ck["cond_scaler"]), ck)

    def predict(self, cond_vec: List[float], equipment: dict, noise: Optional[List[float]] = None) -> List[float]:
        import torch
        if len(cond_vec) != core.COND_DIM or not all(math.isfinite(float(v)) for v in cond_vec):
            raise ModelVersionError("invalid condition vector for the loot model")
        cs = self.cond_scaler.transform([list(map(float, cond_vec))])[0]
        feats = torch.tensor([list(cs) + ml_data.equip_onehot(equipment)], dtype=torch.float32)
        z = None
        if noise is not None:
            if len(noise) != self.noise_dim:
                raise ModelVersionError(f"noise has {len(noise)} dims, expected {self.noise_dim}")
            z = torch.tensor([noise], dtype=torch.float32)
        with torch.no_grad():
            p = self.generator(feats, z)[0].tolist()
        if not all(math.isfinite(v) and v >= -1e-9 for v in p) or abs(sum(p) - 1.0) > 1e-4:
            raise InvalidModelOutput(f"invalid category probabilities {p}")
        return [max(v, 0.0) for v in p]
