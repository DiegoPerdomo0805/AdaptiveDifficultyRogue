"""
dda_core.py
===========
Single source of truth for everything that used to be copy-pasted across
rogue.py, bot_runner.py, cGAN.py and train_cgan.py:

  * shared gameplay constants (arena size, speeds, cooldowns, base enemy stats)
  * the CombatMetrics dataclass and its condition-vector definition
  * the rule_based_dda() heuristic (difficulty + loot bias)
  * knockback tuning + helper
  * enemy "ram" (offensive dash) tuning
  * config_hash() — a semantic fingerprint of every DDA-relevant constant

CONDITION VECTOR CONTRACT — v1.1 (COND_DIM = 9)
------------------------------------------------
This revises the original 6-dim vector after an external review found two
real defects:

  1. rule_based_dda() computed its targets from `total_kills` (derived from
     metrics) and `node_depth` (an external argument) — NEITHER of which
     was present in the recorded 6-dim condition vector. Two rooms could
     therefore share an identical condition vector while the heuristic
     produced different targets for them, because the heuristic was
     secretly using information the model never saw. A model trained on
     that data faces an ambiguous regression problem by construction.

  2. The 6th dimension (`death_rate`, computed pre-room) is a structural
     constant: in the current one-life-per-run design, `deaths` is always
     0 going into every room of a run that hasn't ended yet (a run simply
     terminates the instant a death occurs, so no room is ever entered
     with deaths > 0). Every sample's 6th feature was therefore exactly
     0.0 — zero variance, unlearnable, and it also meant rule_based_dda's
     "ease off a struggling player" branch (which keyed off deaths_per_min)
     could structurally never fire; only the "ramp up a coasting player"
     branch was ever reachable. That is a real DDA bug, not just a data
     artifact: the game currently has no way to detect and ease off a
     player who is taking heavy damage but hasn't died yet.

Fixes applied here (both are the SAME change viewed from two angles: make
the condition vector a sufficient statistic for what the heuristic uses):

  * `death_rate` is replaced by `hp_ratio` (current_hp / max_hp), a
    continuous, always-meaningful, mid-run signal of player distress that
    varies room to room instead of being pinned at 0. rule_based_dda's
    easing/ramping logic now keys off hp_ratio instead of a deaths-per-
    minute figure that could never vary within a run.
  * Three new dimensions are appended, exactly the three the review
    recommended adding: `total_kills_norm` (cumulative kill magnitude,
    normalized), `node_depth_norm` (dungeon depth, normalized), and
    `progress_norm` (rooms cleared / total rooms in the map). These are
    precisely the extra inputs rule_based_dda secretly depended on, now
    made explicit and visible to any model trained on this data.

COND_DIM goes from 6 to 9. This is a breaking schema change — see
telemetry.SCHEMA_VERSION, which is bumped accordingly, and
config_hash(), which folds in every constant below so an incompatible
retuning is detectable even without a manual version bump.
"""

from dataclasses import dataclass
from typing import List, Optional

# ---------------------------------------------------------------------------
# Condition-vector normalization caps (own dimensions, so kept alongside the
# vector definition rather than buried in the DDA formula below).
# ---------------------------------------------------------------------------

TOTAL_KILLS_NORM_CAP = 30.0   # total_kills clamp ceiling for normalization
NODE_DEPTH_NORM_CAP  = 10.0   # node_depth clamp ceiling for normalization

COND_DIM = 9   # see module docstring — was 6 under schema v1


def clamp(v, a, b):
    return max(a, min(b, v))


# ---------------------------------------------------------------------------
# Arena / shared gameplay constants
# ---------------------------------------------------------------------------

W, H = 960, 540
ARENA_MARGIN = 40

PLAYER_SPEED = 220.0
DASH_SPEED = 600.0
DASH_TIME = 0.12
DASH_CD = 1.25

MELEE_RANGE = 45
MELEE_ARC_DEG = 90
MELEE_CD_BASE = 0.45
PROJECTILE_SPEED = 420.0
MAGIC_CD_BASE = 0.60
MAGIC_AMMO_MAX = 12

DEFEND_SLOW = 0.55
DEFEND_DMG_MULT = 0.55

HEAL_ON_KILL_BASE = 8

ENEMY_BASE_HP = 30
ENEMY_BASE_DMG = 6
ENEMY_SPEED = 110.0
ENEMY_AGGRO_R = 260.0
ENEMY_SPAWN_MIN_DIST = 130

# ---------------------------------------------------------------------------
# Enemy "ram" (offensive dash) tuning
# ---------------------------------------------------------------------------

ENEMY_DASH_TRIGGER_R  = 120.0
ENEMY_WINDUP_TIME     = 0.50
ENEMY_DASH_SPEED_MULT = 3.4
ENEMY_DASH_TIME       = 0.22
ENEMY_DASH_RECOVER    = 1.50

# ---------------------------------------------------------------------------
# Knockback tuning
# ---------------------------------------------------------------------------

KNOCKBACK_PER_DAMAGE = 7.0
KNOCKBACK_MAX_SPEED  = 480.0
KNOCKBACK_FRICTION   = 9.0


def knockback_impulse(dmg: float) -> float:
    return clamp(dmg * KNOCKBACK_PER_DAMAGE, 0.0, KNOCKBACK_MAX_SPEED)


def apply_knockback_decay(kvx: float, kvy: float, dt: float) -> tuple:
    import math
    decay = math.exp(-KNOCKBACK_FRICTION * dt)
    kvx *= decay
    kvy *= decay
    if kvx * kvx + kvy * kvy < 1.0:
        return 0.0, 0.0
    return kvx, kvy


# ---------------------------------------------------------------------------
# Loadout dataclasses + fixed pools
# ---------------------------------------------------------------------------

@dataclass
class Weapon:
    name: str
    dmg: float
    cd_mult: float
    heal_mult: float


@dataclass
class Spell:
    name: str
    dmg: float
    cd_mult: float
    ammo_max: int


@dataclass
class Boots:
    name: str
    dash_dist_mult: float


@dataclass
class Armor:
    name: str
    dmg_absorb: float
    heal_on_kill_bonus: int


WEAPONS = [
    Weapon("Sword", dmg=14, cd_mult=1.00, heal_mult=1.00),
    Weapon("Hatchet", dmg=20, cd_mult=1.12, heal_mult=1.05),
    Weapon("Rapier", dmg=10, cd_mult=0.80, heal_mult=0.95),
]
SPELLS = [
    Spell("Magic Bolt", dmg=10, cd_mult=1.00, ammo_max=12),
    Spell("Magic Spike", dmg=8, cd_mult=0.78, ammo_max=16),
    Spell("Hex Bomb", dmg=30, cd_mult=1.25, ammo_max=9),
]
BOOTS = [
    Boots("Leather Boots", dash_dist_mult=1.00),
    Boots("Sprint Greaves", dash_dist_mult=1.25),
    Boots("Voidstep Treads", dash_dist_mult=1.45),
]
ARMORS = [
    Armor("Cloth Wrap", dmg_absorb=0.5, heal_on_kill_bonus=0),
    Armor("Chain Shirt", dmg_absorb=2.0, heal_on_kill_bonus=0),
    Armor("Blood Harness", dmg_absorb=1.0, heal_on_kill_bonus=6),
]

# ---------------------------------------------------------------------------
# Combat metrics + condition vector (THE single definition)
# ---------------------------------------------------------------------------


@dataclass
class CombatMetrics:
    melee_kills: int = 0
    magic_kills: int = 0
    melee_hits: int = 0
    magic_hits: int = 0
    damage_taken: float = 0.0
    damage_dealt: float = 0.0
    deaths: int = 0
    time_alive: float = 0.0

    def to_feature_vector(self, node_depth: int = 0, hp_ratio: float = 1.0,
                           progress_norm: float = 0.0) -> List[float]:
        return feature_vector_from_raw(
            self.melee_kills, self.magic_kills,
            self.melee_hits, self.magic_hits,
            self.damage_taken, self.damage_dealt,
            hp_ratio, node_depth, progress_norm,
        )


def feature_vector_from_raw(
    melee_k, magic_k, melee_h, magic_h,
    dmg_taken, dmg_dealt,
    hp_ratio, node_depth, progress_norm,
) -> List[float]:
    """
    THE single definition of the 9-dim condition vector (schema v1.1):

      [melee_ratio, magic_ratio, hits_per_kill_melee, hits_per_kill_magic,
       dmg_taken/dmg_dealt, hp_ratio, total_kills_norm, node_depth_norm,
       progress_norm]

    `hp_ratio` (current_hp / max_hp, in [0,1]) replaces the old `death_rate`
    dimension — see the module docstring for why death_rate was structurally
    constant and therefore unusable. `total_kills_norm`/`node_depth_norm`/
    `progress_norm` are the three additions the review requested, exposing
    exactly what rule_based_dda() actually conditions on.

    Every caller (CombatMetrics.to_feature_vector, bot_runner.py,
    train_cgan_v2.py) goes through this one function.
    """
    melee_k, magic_k = float(melee_k), float(magic_k)
    melee_h, magic_h = float(melee_h), float(magic_h)
    dmg_taken, dmg_dealt = float(dmg_taken), float(dmg_dealt)

    total_kills = melee_k + magic_k
    melee_ratio = (melee_k / total_kills) if total_kills > 0 else 0.5
    magic_ratio = (magic_k / total_kills) if total_kills > 0 else 0.5

    hpk_melee = (melee_h / melee_k) if melee_k > 0 else float(melee_h + 1)
    hpk_magic = (magic_h / magic_k) if magic_k > 0 else float(magic_h + 1)

    dmg_ratio = (dmg_taken / dmg_dealt) if dmg_dealt > 1e-6 else 1.0

    total_kills_norm = clamp(total_kills / TOTAL_KILLS_NORM_CAP, 0.0, 1.0)
    node_depth_norm = clamp(float(node_depth) / NODE_DEPTH_NORM_CAP, 0.0, 1.0)

    return [
        clamp(melee_ratio, 0.0, 1.0),
        clamp(magic_ratio, 0.0, 1.0),
        clamp(hpk_melee, 0.0, 10.0),
        clamp(hpk_magic, 0.0, 10.0),
        clamp(dmg_ratio, 0.0, 5.0),
        clamp(hp_ratio, 0.0, 1.0),
        total_kills_norm,
        node_depth_norm,
        clamp(progress_norm, 0.0, 1.0),
    ]


# Loot archetypes, in feature-vector-derived-weight form.
LOOT_BIAS_KNIGHT    = [0.55, 0.05, 0.25, 0.15]
LOOT_BIAS_BERSERKER = [0.30, 0.05, 0.45, 0.20]
LOOT_BIAS_SNIPER    = [0.05, 0.55, 0.20, 0.20]


def loot_bias_for_metrics(metrics: CombatMetrics) -> list:
    total_k = metrics.melee_kills + metrics.magic_kills
    melee_ratio = (metrics.melee_kills / total_k) if total_k > 0 else 0.5
    dmg_ratio = metrics.damage_taken / max(metrics.damage_dealt, 1e-6)
    if melee_ratio >= 0.55 and dmg_ratio > 1.2:
        return list(LOOT_BIAS_BERSERKER)
    elif melee_ratio >= 0.55:
        return list(LOOT_BIAS_KNIGHT)
    else:
        return list(LOOT_BIAS_SNIPER)


# ---------------------------------------------------------------------------
# rule_based_dda tunable constants — named here (not inlined below) so
# config_hash() can fold in every one of them individually. A change to any
# of these invalidates the semantic meaning of previously-collected data.
# ---------------------------------------------------------------------------

DDA_KILL_SCALE_COEF   = 0.045   # base_mult growth per accumulated kill
DDA_DEPTH_SCALE_COEF  = 0.06    # depth_mult growth per node_depth
DDA_DMG_MULT_FACTOR   = 0.85    # dmg_mult = hp_mult_raw * this
DDA_SPEED_MULT_FACTOR = 0.5     # how much of (base_mult-1) carries into speed
DDA_SPAWN_BASE        = 2       # spawn_count formula: base + kills*coef + depth//3
DDA_SPAWN_KILL_COEF   = 0.18
DDA_SPAWN_DEPTH_DIV   = 3

DDA_HP_STRUGGLE_HP_RATIO   = 0.35   # below this hp_ratio: ease off
DDA_HP_STRUGGLE_MULT       = 0.78
DDA_HP_COASTING_HP_RATIO   = 0.85   # above this hp_ratio (+min kills): ramp up
DDA_HP_COASTING_MIN_KILLS  = 5
DDA_HP_COASTING_MULT       = 1.22

DDA_HP_MULT_BOUNDS    = (0.55, 3.0)
DDA_DMG_MULT_BOUNDS   = (0.45, 2.4)
DDA_SPEED_MULT_BOUNDS = (0.80, 1.70)
DDA_SPAWN_BOUNDS      = (2, 10)

BOSS_HP_MULT    = 4.0
BOSS_DMG_MULT   = 1.7
BOSS_SPEED_MULT = 0.9
BOSS_SPAWN_N    = 1


def rule_based_dda(metrics: CombatMetrics, node_depth: int = 0, hp_ratio: float = 1.0) -> dict:
    """
    Heuristic DDA used until enough runs exist to train the cGAN (and always
    used as the safe fallback if the model is unavailable/fails to load).

    FIX (post-review): the struggling-player easing branch used to key off
    deaths-per-minute, which is structurally always 0 before a run-ending
    death (see module docstring) — that branch could never fire in
    practice. It now keys off `hp_ratio` (current_hp / max_hp), which is
    exactly the same continuous distress signal now exposed in the
    condition vector's 6th dimension, so the model and the heuristic look
    at the same thing.
    """
    total_kills = metrics.melee_kills + metrics.magic_kills

    base_mult = 1.0 + (total_kills * DDA_KILL_SCALE_COEF)
    if hp_ratio < DDA_HP_STRUGGLE_HP_RATIO:
        base_mult *= DDA_HP_STRUGGLE_MULT
    elif hp_ratio > DDA_HP_COASTING_HP_RATIO and total_kills > DDA_HP_COASTING_MIN_KILLS:
        base_mult *= DDA_HP_COASTING_MULT

    depth_mult = 1.0 + node_depth * DDA_DEPTH_SCALE_COEF

    hp_mult = clamp(base_mult * depth_mult, *DDA_HP_MULT_BOUNDS)
    dmg_mult = clamp(base_mult * depth_mult * DDA_DMG_MULT_FACTOR, *DDA_DMG_MULT_BOUNDS)
    speed_mult = clamp(1.0 + (base_mult - 1.0) * DDA_SPEED_MULT_FACTOR, *DDA_SPEED_MULT_BOUNDS)
    spawn_n = clamp(int(DDA_SPAWN_BASE + total_kills * DDA_SPAWN_KILL_COEF) + node_depth // DDA_SPAWN_DEPTH_DIV,
                     *DDA_SPAWN_BOUNDS)

    return {
        "enemy_hp_mult": round(hp_mult, 3),
        "enemy_dmg_mult": round(dmg_mult, 3),
        "enemy_speed_mult": round(speed_mult, 3),
        "spawn_count": spawn_n,
        "loot_bias": loot_bias_for_metrics(metrics),
    }


def effective_applied_config(applied: dict, room_kind: str) -> dict:
    """
    The single source for turning a raw rule_based_dda()/model output into
    the EFFECTIVE configuration actually used to spawn a room — i.e. with
    the boss stat multipliers layered on top when room_kind == "boss".

    FIX (post-review, finding #5): spawn_room()/simulate_combat_room() used
    to apply the boss multipliers (hp*=4, dmg*=1.7, speed*=0.9, n=1) AFTER
    computing `applied`, but telemetry logged the PRE-multiplier `applied`
    dict — so a boss room's logged applied_hp_mult never matched what was
    actually simulated. Every caller that spawns a room AND every caller
    that logs telemetry for that room must now go through this function,
    so "what we logged" and "what we simulated" can never diverge again.
    """
    out = dict(applied)
    if room_kind == "boss":
        out["enemy_hp_mult"] = round(applied["enemy_hp_mult"] * BOSS_HP_MULT, 3)
        out["enemy_dmg_mult"] = round(applied["enemy_dmg_mult"] * BOSS_DMG_MULT, 3)
        out["enemy_speed_mult"] = round(applied["enemy_speed_mult"] * BOSS_SPEED_MULT, 3)
        out["spawn_count"] = BOSS_SPAWN_N
    return out


# ---------------------------------------------------------------------------
# Loot option generation — shared by rogue.py and bot_runner.py.
# ---------------------------------------------------------------------------

import random as _random

LOOT_KINDS = ["weapon", "spell", "armor", "boots"]
LOOT_POOLS = {"weapon": WEAPONS, "spell": SPELLS, "armor": ARMORS, "boots": BOOTS}


def generate_loot_options(
    loot_bias: Optional[List[float]] = None,
    count: int = 3,
    rng: Optional[_random.Random] = None,
) -> List[tuple]:
    r = rng or _random
    if loot_bias is not None and len(loot_bias) == 4 and sum(loot_bias) > 0:
        total = sum(loot_bias)
        weights = [w / total for w in loot_bias]
    else:
        weights = [0.25, 0.25, 0.25, 0.25]

    options = []
    for _ in range(count):
        kind = r.choices(LOOT_KINDS, weights=weights, k=1)[0]
        item = r.choice(LOOT_POOLS[kind])
        options.append((kind, item))
    return options


# ---------------------------------------------------------------------------
# config_hash — a semantic fingerprint of every DDA/gameplay constant that
# determines what a dataset's applied_* / cond_* columns actually MEAN.
# ---------------------------------------------------------------------------

import hashlib
import json


def config_hash(extra: Optional[dict] = None) -> str:
    """
    Stable short hash over every tunable this module owns. A change to any
    of these invalidates the semantic meaning of previously-collected data,
    even if nobody remembers to bump a version string by hand.

    FIX: The original hash only covered a handful
    of combat constants (enemy base stats, ram timing, knockback). It did
    NOT cover the rule_based_dda() formula's own coefficients, its clamp
    bounds, the boss stat multipliers, or the loot-bias tables — meaning a
    rebalance of any of those could silently produce an incompatible
    dataset without changing the hash. Every constant rule_based_dda() and
    effective_applied_config() actually use is enumerated below.

    `extra` lets callers outside this module (e.g. bot_runner.py's
    archetype definitions and navigation policy, telemetry.py's
    RECENT_WINDOW_ROOMS/DEFAULT_NODE_COUNT) fold their own semantically-
    relevant constants into the SAME hash — see telemetry.write_manifest().
    """
    payload = {
        "COND_DIM": COND_DIM,
        "TOTAL_KILLS_NORM_CAP": TOTAL_KILLS_NORM_CAP,
        "NODE_DEPTH_NORM_CAP": NODE_DEPTH_NORM_CAP,

        "ENEMY_BASE_HP": ENEMY_BASE_HP, "ENEMY_BASE_DMG": ENEMY_BASE_DMG,
        "ENEMY_SPEED": ENEMY_SPEED, "ENEMY_AGGRO_R": ENEMY_AGGRO_R,
        "ENEMY_SPAWN_MIN_DIST": ENEMY_SPAWN_MIN_DIST,
        "ENEMY_DASH_TRIGGER_R": ENEMY_DASH_TRIGGER_R,
        "ENEMY_WINDUP_TIME": ENEMY_WINDUP_TIME,
        "ENEMY_DASH_SPEED_MULT": ENEMY_DASH_SPEED_MULT,
        "ENEMY_DASH_TIME": ENEMY_DASH_TIME,
        "ENEMY_DASH_RECOVER": ENEMY_DASH_RECOVER,
        "KNOCKBACK_PER_DAMAGE": KNOCKBACK_PER_DAMAGE,
        "KNOCKBACK_MAX_SPEED": KNOCKBACK_MAX_SPEED,
        "KNOCKBACK_FRICTION": KNOCKBACK_FRICTION,
        "MELEE_CD_BASE": MELEE_CD_BASE, "MAGIC_CD_BASE": MAGIC_CD_BASE,
        "MELEE_RANGE": MELEE_RANGE, "MELEE_ARC_DEG": MELEE_ARC_DEG,

        "DDA_KILL_SCALE_COEF": DDA_KILL_SCALE_COEF,
        "DDA_DEPTH_SCALE_COEF": DDA_DEPTH_SCALE_COEF,
        "DDA_DMG_MULT_FACTOR": DDA_DMG_MULT_FACTOR,
        "DDA_SPEED_MULT_FACTOR": DDA_SPEED_MULT_FACTOR,
        "DDA_SPAWN_BASE": DDA_SPAWN_BASE,
        "DDA_SPAWN_KILL_COEF": DDA_SPAWN_KILL_COEF,
        "DDA_SPAWN_DEPTH_DIV": DDA_SPAWN_DEPTH_DIV,
        "DDA_HP_STRUGGLE_HP_RATIO": DDA_HP_STRUGGLE_HP_RATIO,
        "DDA_HP_STRUGGLE_MULT": DDA_HP_STRUGGLE_MULT,
        "DDA_HP_COASTING_HP_RATIO": DDA_HP_COASTING_HP_RATIO,
        "DDA_HP_COASTING_MIN_KILLS": DDA_HP_COASTING_MIN_KILLS,
        "DDA_HP_COASTING_MULT": DDA_HP_COASTING_MULT,
        "DDA_HP_MULT_BOUNDS": DDA_HP_MULT_BOUNDS,
        "DDA_DMG_MULT_BOUNDS": DDA_DMG_MULT_BOUNDS,
        "DDA_SPEED_MULT_BOUNDS": DDA_SPEED_MULT_BOUNDS,
        "DDA_SPAWN_BOUNDS": DDA_SPAWN_BOUNDS,

        "BOSS_HP_MULT": BOSS_HP_MULT, "BOSS_DMG_MULT": BOSS_DMG_MULT,
        "BOSS_SPEED_MULT": BOSS_SPEED_MULT, "BOSS_SPAWN_N": BOSS_SPAWN_N,

        "LOOT_BIAS_KNIGHT": LOOT_BIAS_KNIGHT,
        "LOOT_BIAS_BERSERKER": LOOT_BIAS_BERSERKER,
        "LOOT_BIAS_SNIPER": LOOT_BIAS_SNIPER,
    }
    if extra:
        payload.update(extra)
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
