"""
controllers.py
==============
DDA controllers that choose the 4-dim enemy intervention for a room:

  fixed        one constant reference action (no adaptation) -- baseline.
  heuristic    dda_core.rule_based_dda() -- baseline, exploration anchor and
               runtime fallback. NOT a label / ground truth.
  exploration  seeded, bounded variation around the valid action space used
               for DATA COLLECTION (perturbations around the heuristic,
               uniform samples and fixed reference actions; mixture and
               scales come from experiment_config.json).
  cgan         outcome-guided generator G(x, z) loaded via model_runtime.py,
               with a recorded heuristic fallback on any failure.

Every controller returns a Decision whose `action` has already passed
dda_core.finalize_action() (finite, clamped, int spawn_count). Room-kind
modifiers (boss) are applied afterwards by dda_core.effective_applied_config()
exactly as for gameplay. Loot bias is NOT learned: every controller uses the
heuristic loot bias so the comparison isolates the enemy intervention.

Stdlib-only except CGANController, which imports torch via model_runtime.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import hashlib
import importlib.util
import math
import os
import random
import sys

import balance
import dda_core as core

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def derive_seed(run_seed: int, tag: str) -> int:
    """Independent, reproducible RNG stream per (run, purpose)."""
    h = hashlib.sha256(f"{int(run_seed)}:{tag}".encode()).hexdigest()
    return int(h[:12], 16)


@dataclass
class DecisionContext:
    cond_vec: List[float]
    metrics: core.CombatMetrics
    node_depth: int
    hp_ratio: float
    room_kind: str


@dataclass
class Decision:
    action: dict                      # finalized raw action (+ loot_bias)
    controller_type: str
    controller_version: str
    model_checkpoint_id: str = ""
    action_source: str = ""
    fallback_used: int = 0
    fallback_reason: str = ""

    def meta(self) -> dict:
        return {
            "controller_type": self.controller_type,
            "controller_version": self.controller_version,
            "model_checkpoint_id": self.model_checkpoint_id,
            "action_source": self.action_source,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
        }


class Controller:
    controller_type = "base"

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or balance.load_experiment_config()
        self.version = self.cfg["controllers"][self.controller_type]["version"]

    def begin_run(self, run_seed: int):
        """Reseed per-run state. Called once at the start of every run."""

    def decide(self, ctx: DecisionContext) -> Decision:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"controller_type": self.controller_type, "version": self.version}

    @staticmethod
    def _with_loot(action: dict, ctx: DecisionContext) -> dict:
        out = core.finalize_action(action)
        out["loot_bias"] = core.loot_bias_for_metrics(ctx.metrics)
        return out


class FixedController(Controller):
    controller_type = "fixed"

    def decide(self, ctx):
        action = dict(self.cfg["controllers"]["fixed"]["action"])
        return Decision(self._with_loot(action, ctx), "fixed", self.version, action_source="fixed")


class HeuristicController(Controller):
    controller_type = "heuristic"

    def decide(self, ctx):
        raw = core.rule_based_dda(ctx.metrics, node_depth=ctx.node_depth, hp_ratio=ctx.hp_ratio)
        return Decision(self._with_loot(raw, ctx), "heuristic", self.version,
                        action_source="heuristic")


class ExplorationController(Controller):
    controller_type = "exploration"
    BRANCHES = ("perturb", "uniform", "reference")

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.ecfg = self.cfg["controllers"]["exploration"]
        self.rng = random.Random(0)

    def begin_run(self, run_seed: int):
        self.rng = random.Random(derive_seed(run_seed, "exploration"))

    def _branch(self) -> str:
        mix = self.ecfg["mixture"]
        r = self.rng.random()
        acc = 0.0
        for b in self.BRANCHES:
            acc += mix.get(b, 0.0)
            if r < acc:
                return b
        return self.BRANCHES[0]

    def decide(self, ctx):
        branch = self._branch()
        heur = core.rule_based_dda(ctx.metrics, node_depth=ctx.node_depth, hp_ratio=ctx.hp_ratio)
        if branch == "perturb":
            sig = self.ecfg["perturb_log_sigma"]
            action = {k: heur[k] * math.exp(self.rng.gauss(0.0, sig[k]))
                      for k in ("enemy_hp_mult", "enemy_dmg_mult", "enemy_speed_mult")}
            action["spawn_count"] = heur["spawn_count"] + self.rng.choice(self.ecfg["perturb_spawn_delta"])
        elif branch == "uniform":
            action = {k: self.rng.uniform(*core.ACTION_BOUNDS[k])
                      for k in ("enemy_hp_mult", "enemy_dmg_mult", "enemy_speed_mult")}
            action["spawn_count"] = self.rng.randint(*self.ecfg["uniform_spawn_range"])
        else:
            action = dict(self.rng.choice(self.ecfg["reference_actions"]))
        return Decision(self._with_loot(action, ctx), "exploration", self.version,
                        action_source=branch)

    def describe(self):
        d = super().describe()
        d["exploration"] = self.ecfg
        return d


def load_model_runtime():
    """Import model_runtime.py by path (independent of cwd / sys.path)."""
    if "model_runtime" in sys.modules:
        return sys.modules["model_runtime"]
    spec = importlib.util.spec_from_file_location(
        "model_runtime", os.path.join(_THIS_DIR, "model_runtime.py"))
    if spec is None or spec.loader is None:
        raise ImportError("Could not load model_runtime.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_runtime"] = mod
    spec.loader.exec_module(mod)
    return mod


class CGANController(Controller):
    """
    Outcome-guided generator with a recorded heuristic fallback. Fallback
    reasons: load_error:<...> (checkpoint missing/incompatible -- every room
    falls back), invalid_output:<...> (NaN/inf/out-of-bounds/exception for
    one room) and disabled_after_<n>_invalid_outputs (model switched off for
    the rest of the process after repeated invalid outputs).
    """
    controller_type = "cgan"

    def __init__(self, model_dir: str, cfg=None, models=None):
        super().__init__(cfg)
        self.model_dir = model_dir
        self.max_invalid = int(self.cfg["controllers"]["cgan"].get("max_invalid_outputs", 3))
        self.fallback = HeuristicController(self.cfg)
        self.models = models
        self.load_error = ""
        self.invalid_count = 0
        self.disabled = False
        self.noise_rng = random.Random(0)
        if self.models is None:
            try:
                self.models = load_model_runtime().LoadedModels.load(model_dir)
            except Exception as e:      # incompatible/missing -> recorded fallback
                self.load_error = f"load_error:{type(e).__name__}:{str(e)[:160]}"
        self.checkpoint_id = self.models.checkpoint_id if self.models is not None else ""

    def begin_run(self, run_seed: int):
        self.noise_rng = random.Random(derive_seed(run_seed, "cgan_noise"))

    def _fallback(self, ctx, reason: str) -> Decision:
        d = self.fallback.decide(ctx)
        return Decision(d.action, "cgan", self.version, model_checkpoint_id=self.checkpoint_id,
                        action_source="heuristic_fallback", fallback_used=1,
                        fallback_reason=reason.replace(",", ";").replace("\n", " "))

    def decide(self, ctx):
        if self.models is None:
            return self._fallback(ctx, self.load_error or "load_error:unknown")
        if self.disabled:
            return self._fallback(ctx, f"disabled_after_{self.max_invalid}_invalid_outputs")
        noise = [self.noise_rng.gauss(0.0, 1.0) for _ in range(self.models.noise_dim)]
        try:
            action = self.models.predict(ctx.cond_vec, noise=noise)
        except Exception as e:
            self.invalid_count += 1
            if self.invalid_count >= self.max_invalid:
                self.disabled = True
            return self._fallback(ctx, f"invalid_output:{type(e).__name__}:{str(e)[:120]}")
        return Decision(self._with_loot(action, ctx), "cgan", self.version,
                        model_checkpoint_id=self.checkpoint_id, action_source="model")

    def describe(self):
        d = super().describe()
        d.update({"model_checkpoint_id": self.checkpoint_id, "load_error": self.load_error,
                  "max_invalid_outputs": self.max_invalid})
        return d


def make_controller(name: str, model_dir: Optional[str] = None, cfg: Optional[dict] = None) -> Controller:
    if name == "fixed":
        return FixedController(cfg)
    if name == "heuristic":
        return HeuristicController(cfg)
    if name == "exploration":
        return ExplorationController(cfg)
    if name == "cgan":
        if not model_dir:
            raise ValueError("the cgan controller requires a model directory")
        return CGANController(model_dir, cfg)
    raise ValueError(f"unknown controller {name!r}")
