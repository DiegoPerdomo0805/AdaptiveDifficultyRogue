"""
rewards.py
==========
Reward (loot-category) policies, configured independently of the enemy
controller. A policy PROPOSES category probabilities over
dda_core.SLOTS = [weapon, spell, armor, boots]; offers are then drawn from
those probabilities (category) and uniformly from the FIXED item pool of each
category (dda_core.generate_loot_options). Selecting one of the offers is a
separate mechanism (the bot's pick rule or the human's choice) and is logged
separately (telemetry.selection_mechanism).

  fixed_uniform  equal probabilities [0.25, 0.25, 0.25, 0.25]
  heuristic      dda_core.loot_bias_for_metrics (documented archetype-like
                 preference from cumulative melee share / damage ratio)
  exploration    seeded coverage policy for data collection: with prob
                 `focus` put `focus_mass` on a uniformly drawn category (rest
                 shared), with prob `uniform` equal probabilities, otherwise
                 the heuristic. Uses only the pre-reward state.
  learned_loot   G_loot(x, z) via model_runtime.LoadedLootModel; any load
                 failure / incompatibility / invalid output falls back to the
                 configured fallback policy with a logged reason.

Stdlib-only except LearnedLootPolicy (torch via model_runtime).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math
import random

import balance
import dda_core as core
from controllers import derive_seed, load_model_runtime


@dataclass
class LootContext:
    cond_vec: List[float]
    metrics: core.CombatMetrics
    equipment: Dict[str, str]          # slot -> item_id (pre-reward)
    node_depth: int
    hp_ratio: float


@dataclass
class RewardDecision:
    probs: List[float]
    policy_type: str
    version: str
    proposal_source: str
    checkpoint_id: str = ""
    fallback_used: int = 0
    fallback_reason: str = ""

    def meta(self) -> dict:
        return {"reward_policy_type": self.policy_type, "reward_policy_version": self.version,
                "reward_checkpoint_id": self.checkpoint_id,
                "reward_proposal_source": self.proposal_source,
                "reward_fallback_used": self.fallback_used,
                "reward_fallback_reason": self.fallback_reason}


def normalize_probs(p: List[float]) -> List[float]:
    if len(p) != len(core.SLOTS) or not all(math.isfinite(v) and v >= 0 for v in p) or sum(p) <= 0:
        raise ValueError(f"invalid category probabilities {p}")
    s = sum(p)
    return [v / s for v in p]


class RewardPolicy:
    policy_type = "base"

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or balance.load_experiment_config()
        self.version = self.cfg["reward_policy"][self.policy_type]["version"]

    def begin_run(self, run_seed: int):
        pass

    def propose(self, ctx: LootContext) -> RewardDecision:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"reward_policy": self.policy_type, "version": self.version}


class FixedUniformReward(RewardPolicy):
    policy_type = "fixed_uniform"

    def propose(self, ctx):
        return RewardDecision([0.25] * 4, self.policy_type, self.version, "fixed_uniform")


def heuristic_probs(metrics: core.CombatMetrics) -> List[float]:
    return normalize_probs(core.loot_bias_for_metrics(metrics))


class HeuristicReward(RewardPolicy):
    policy_type = "heuristic"

    def propose(self, ctx):
        return RewardDecision(heuristic_probs(ctx.metrics), self.policy_type, self.version, "heuristic")


class ExplorationReward(RewardPolicy):
    policy_type = "exploration"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.ecfg = self.cfg["reward_policy"]["exploration"]
        self.rng = random.Random(0)

    def begin_run(self, run_seed: int):
        self.rng = random.Random(derive_seed(run_seed, "reward_exploration"))

    def propose(self, ctx):
        mix = self.ecfg["mixture"]
        r = self.rng.random()
        if r < mix["focus"]:
            k = self.rng.randrange(len(core.SLOTS))
            rest = (1.0 - self.ecfg["focus_mass"]) / (len(core.SLOTS) - 1)
            probs = [self.ecfg["focus_mass"] if i == k else rest for i in range(len(core.SLOTS))]
            src = "focus_" + core.SLOTS[k]
        elif r < mix["focus"] + mix["uniform"]:
            probs, src = [0.25] * 4, "uniform"
        else:
            probs, src = heuristic_probs(ctx.metrics), "heuristic"
        return RewardDecision(normalize_probs(probs), self.policy_type, self.version, src)

    def describe(self):
        d = super().describe()
        d["exploration"] = self.ecfg
        return d


class LearnedLootPolicy(RewardPolicy):
    policy_type = "learned_loot"

    def __init__(self, model_dir: Optional[str], cfg=None, model=None):
        super().__init__(cfg)
        lcfg = self.cfg["reward_policy"]["learned_loot"]
        self.max_invalid = int(lcfg.get("max_invalid_outputs", 3))
        fb = lcfg.get("fallback", "heuristic")
        self.fallback = HeuristicReward(self.cfg) if fb == "heuristic" else FixedUniformReward(self.cfg)
        self.model = model
        self.load_error = ""
        self.invalid_count = 0
        self.disabled = False
        self.noise_rng = random.Random(0)
        if self.model is None:
            try:
                if not model_dir:
                    raise FileNotFoundError("no learned-loot model directory configured")
                self.model = load_model_runtime().LoadedLootModel.load(model_dir)
            except Exception as e:
                self.load_error = f"load_error:{type(e).__name__}:{str(e)[:160]}"
        self.checkpoint_id = self.model.checkpoint_id if self.model is not None else ""

    def begin_run(self, run_seed: int):
        self.noise_rng = random.Random(derive_seed(run_seed, "loot_noise"))

    def _fallback(self, ctx, reason: str) -> RewardDecision:
        d = self.fallback.propose(ctx)
        return RewardDecision(d.probs, self.policy_type, self.version,
                              f"{self.fallback.policy_type}_fallback", self.checkpoint_id, 1,
                              reason.replace(",", ";").replace("\n", " "))

    def propose(self, ctx):
        if self.model is None:
            return self._fallback(ctx, self.load_error or "load_error:unknown")
        if self.disabled:
            return self._fallback(ctx, f"disabled_after_{self.max_invalid}_invalid_outputs")
        noise = [self.noise_rng.gauss(0.0, 1.0) for _ in range(self.model.noise_dim)]
        try:
            probs = normalize_probs(self.model.predict(ctx.cond_vec, ctx.equipment, noise=noise))
        except Exception as e:
            self.invalid_count += 1
            if self.invalid_count >= self.max_invalid:
                self.disabled = True
            return self._fallback(ctx, f"invalid_output:{type(e).__name__}:{str(e)[:120]}")
        return RewardDecision(probs, self.policy_type, self.version, "model", self.checkpoint_id)

    def describe(self):
        d = super().describe()
        d.update({"checkpoint_id": self.checkpoint_id, "load_error": self.load_error})
        return d


def make_reward_policy(name: str, loot_model_dir: Optional[str] = None,
                       cfg: Optional[dict] = None) -> RewardPolicy:
    if name == "fixed_uniform":
        return FixedUniformReward(cfg)
    if name == "heuristic":
        return HeuristicReward(cfg)
    if name == "exploration":
        return ExplorationReward(cfg)
    if name == "learned_loot":
        return LearnedLootPolicy(loot_model_dir, cfg)
    raise ValueError(f"unknown reward policy {name!r}")


def draw_offers(probs: List[float], rng: random.Random, count: int = 3) -> List[tuple]:
    """[(category, item)] -- category ~ probs, item uniform within the fixed pool."""
    return core.generate_loot_options(probs, count=count, rng=rng)
