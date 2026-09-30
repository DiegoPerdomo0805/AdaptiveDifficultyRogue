"""
model_runtime.py
=================
Loads a trained cGAN checkpoint for use inside rogue.py, independent of
the current working directory (loaded via importlib.util.spec_from_file_location
by the caller, so this module never relies on sys.path/cwd either).

Finding #9 fix: LoadedModels.load() is the consumer-side guardian that was
previously missing ("el versionado esta en el productor, pero no en el
consumidor"). It verifies the checkpoint's recorded cond_dim/target_dim/
dda_version against the CURRENT code's dda_core.COND_DIM /
telemetry.TARGET_DIM / telemetry.DDA_VERSION before allowing the model to
be used at all. A checkpoint trained under an old/incompatible schema is
rejected with a clear message rather than silently fed a wrong-shaped
condition vector (which would previously either crash deep inside a
tensor op or -- worse -- silently produce garbage from misaligned dims).
"""

from typing import List, Optional
import json
import os
import sys

# This module is loaded via importlib.util.spec_from_file_location by
# rogue.py, specifically so it works regardless of the caller's cwd or
# sys.path. But it in turn depends on dda_core.py/telemetry.py, which are
# plain `import x` statements -- those only resolve if this file's own
# directory is on sys.path. A caller in a different cwd (confirmed by an
# end-to-end test: loading this module while cwd'd elsewhere raised
# ModuleNotFoundError: dda_core) would otherwise crash here. Fix: insert
# this file's own directory at the front of sys.path before importing our
# siblings, so the same spec_from_file_location trick that isolates THIS
# module's own loading also isolates what it depends on.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import dda_core as core
import telemetry


class ModelVersionError(Exception):
    pass


class LoadedModels:
    def __init__(self, generator, scaler: dict, checkpoint: dict):
        self.generator = generator
        self.scaler = scaler
        self.checkpoint = checkpoint

    @staticmethod
    def load(model_dir: str) -> "LoadedModels":
        import torch
        import torch.nn as nn

        ckpt_path = os.path.join(model_dir, "cgan_checkpoint.pt")
        scaler_path = os.path.join(model_dir, "scaler.json")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"No scaler at {scaler_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        with open(scaler_path) as f:
            scaler = json.load(f)

        # --- The guardian the review said was missing ----------------------
        ckpt_cond_dim = checkpoint.get("cond_dim")
        ckpt_target_dim = checkpoint.get("target_dim")
        ckpt_dda_version = checkpoint.get("dda_version")

        if ckpt_cond_dim != core.COND_DIM:
            raise ModelVersionError(
                f"Checkpoint was trained with cond_dim={ckpt_cond_dim}, but "
                f"the current code's dda_core.COND_DIM={core.COND_DIM}. "
                f"Refusing to load an incompatible model -- retrain against "
                f"the current schema."
            )
        if ckpt_target_dim != telemetry.TARGET_DIM:
            raise ModelVersionError(
                f"Checkpoint target_dim={ckpt_target_dim} != current "
                f"telemetry.TARGET_DIM={telemetry.TARGET_DIM}."
            )
        if ckpt_dda_version != telemetry.DDA_VERSION:
            raise ModelVersionError(
                f"Checkpoint was trained against dda_version="
                f"{ckpt_dda_version!r}, but the running code is "
                f"{telemetry.DDA_VERSION!r}. The heuristic/config semantics "
                f"may have changed since this model was trained -- refusing "
                f"to load."
            )

        class Generator(nn.Module):
            def __init__(self, cond_dim, noise_dim=8, target_dim=4, hidden=64):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(cond_dim + noise_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                    nn.Linear(hidden, target_dim),
                )
                self.noise_dim = noise_dim

            def forward(self, cond, noise=None):
                if noise is None:
                    noise = torch.randn(cond.shape[0], self.noise_dim, device=cond.device)
                x = torch.cat([cond, noise], dim=1)
                return self.net(x)

        gen = Generator(cond_dim=ckpt_cond_dim, target_dim=ckpt_target_dim)
        gen.load_state_dict(checkpoint["generator_state_dict"])
        gen.eval()

        return LoadedModels(generator=gen, scaler=scaler, checkpoint=checkpoint)

    def predict(self, cond_vec: List[float]) -> dict:
        import torch
        if len(cond_vec) != core.COND_DIM:
            raise ModelVersionError(
                f"predict() called with a {len(cond_vec)}-dim condition "
                f"vector, expected {core.COND_DIM}."
            )
        with torch.no_grad():
            cond_t = torch.tensor([cond_vec], dtype=torch.float32)
            scaled_out = self.generator(cond_t)[0].tolist()

        mean = self.scaler["target_mean"]
        std = self.scaler["target_std"]
        raw = [scaled_out[i] * std[i] + mean[i] for i in range(len(scaled_out))]
        cols = self.scaler["target_columns"]
        result = dict(zip([c.replace("applied_", "") for c in cols], raw))

        # Clamp into the same bounds rule_based_dda() itself respects, so a
        # model prediction can never produce an out-of-design-space value.
        result["hp_mult"] = core.clamp(result["hp_mult"], *core.DDA_HP_MULT_BOUNDS)
        result["dmg_mult"] = core.clamp(result["dmg_mult"], *core.DDA_DMG_MULT_BOUNDS)
        result["speed_mult"] = core.clamp(result["speed_mult"], *core.DDA_SPEED_MULT_BOUNDS)
        result["spawn_count"] = int(round(core.clamp(
            result["spawn_count"], *core.DDA_SPAWN_BOUNDS)))
        return {
            "enemy_hp_mult": round(result["hp_mult"], 3),
            "enemy_dmg_mult": round(result["dmg_mult"], 3),
            "enemy_speed_mult": round(result["speed_mult"], 3),
            "spawn_count": result["spawn_count"],
        }
