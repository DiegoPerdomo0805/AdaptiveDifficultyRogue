"""
bot_runner.py
=============
Headless bot that simulates multiple playstyle archetypes and logs runs to
runs.csv in the same format as the human-playable game.

Run:
    python bot_runner.py [--runs 200] [--seed 42]

    # Automatically launch cGAN training when done:
    python bot_runner.py --runs 400 --train --train-epochs 200

New in this version
-------------------
  * Reads existing runs.csv before starting so it can measure which archetypes
    are already well-represented. Under-represented archetypes get proportionally
    more new runs to fill gaps rather than further crowding what is already
    abundant.

  * Parallel simulation via ProcessPoolExecutor: each archetype batch runs in
    its own worker process. The GIL doesn't apply here (pure Python + math),
    so wall-clock time scales close to linearly with --workers.

  * Loot picking is now weighted by archetype category biases (consistent with
    rogue.py and cGAN.py) instead of the old melee-ratio if/else branches.

  * --train flag: after bot runs finish, spawns train_cgan.py as a subprocess
    so the full pipeline (simulate → train) runs unattended in one command.

Playstyle archetypes
--------------------
  melee_aggressive  – rushes enemies, spams melee, never defends
  magic_sniper      – kites at range, uses magic exclusively
  hybrid            – mixes melee and magic evenly
  tank              – defends heavily, mostly melee
  reckless          – random actions, high damage taken (noise run)
"""

import argparse
import csv
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from node_map_gen import generate_node_map_graph, NodeMap, MapNode

# ---------------------------------------------------------------------------
# Constants (mirrors game.py — no pygame import needed)
# ---------------------------------------------------------------------------

W, H             = 960, 540
ARENA_MARGIN     = 40
PLAYER_SPEED     = 220.0
DASH_SPEED       = 600.0
DASH_TIME        = 0.12
DASH_CD          = 1.25
MELEE_RANGE      = 42
MELEE_ARC_DEG    = 90
MELEE_CD_BASE    = 0.45
PROJECTILE_SPEED = 420.0
MAGIC_CD_BASE    = 0.60
DEFEND_SLOW      = 0.55
DEFEND_DMG_MULT  = 0.55
HEAL_ON_KILL_BASE= 8
ENEMY_BASE_HP    = 42
ENEMY_BASE_DMG   = 9
ENEMY_SPEED      = 110.0
ENEMY_AGGRO_R    = 260.0
ENEMY_SPAWN_MIN_DIST = 130
SIM_DT           = 1 / 60.0

LOG_FILE = "runs.csv"

# ---------------------------------------------------------------------------
# Pure-Python helpers
# ---------------------------------------------------------------------------

def clamp(v, a, b):
    return max(a, min(b, v))

def vec_len(x, y):
    return math.hypot(x, y)

def norm(x, y):
    l = vec_len(x, y)
    if l <= 1e-9:
        return 0.0, 0.0
    return x / l, y / l

def angle_deg(x, y):
    return math.degrees(math.atan2(y, x))

def angle_diff_deg(a, b):
    return (a - b + 180) % 360 - 180

# ---------------------------------------------------------------------------
# Item definitions (same as game.py)
# ---------------------------------------------------------------------------

@dataclass
class Weapon:
    name: str; dmg: float; cd_mult: float; heal_mult: float

@dataclass
class Spell:
    name: str; dmg: float; cd_mult: float; ammo_max: int

@dataclass
class Boots:
    name: str; dash_dist_mult: float

@dataclass
class Armor:
    name: str; dmg_absorb: float; heal_on_kill_bonus: int

WEAPONS = [
    Weapon("Rusty Blade", dmg=11, cd_mult=1.00, heal_mult=1.00),
    Weapon("Hatchet",     dmg=14, cd_mult=1.12, heal_mult=1.05),
    Weapon("Rapier",      dmg=9,  cd_mult=0.80, heal_mult=0.95),
]
SPELLS = [
    Spell("Ember Bolt", dmg=10, cd_mult=1.00, ammo_max=12),
    Spell("Ice Needle", dmg=8,  cd_mult=0.78, ammo_max=16),
    Spell("Hex Spike",  dmg=14, cd_mult=1.25, ammo_max=9),
]
BOOTS = [
    Boots("Leather Boots",   dash_dist_mult=1.00),
    Boots("Sprint Greaves",  dash_dist_mult=1.25),
    Boots("Voidstep Treads", dash_dist_mult=1.45),
]
ARMORS = [
    Armor("Cloth Wrap",    dmg_absorb=0.5, heal_on_kill_bonus=0),
    Armor("Chain Shirt",   dmg_absorb=2.0, heal_on_kill_bonus=0),
    Armor("Blood Harness", dmg_absorb=1.0, heal_on_kill_bonus=6),
]

# Loot weight vectors per archetype loot_bias key.
# Order: [weapon_w, spell_w, armor_w, boots_w]  — mirrors cGAN.py LOOT_TARGETS_BY_ARCHETYPE.
_LOOT_WEIGHTS: Dict[str, List[float]] = {
    "melee":    [0.55, 0.05, 0.25, 0.15],  # Knight
    "tank":     [0.30, 0.05, 0.45, 0.20],  # Berserker
    "magic":    [0.05, 0.55, 0.20, 0.20],  # Sniper
    "balanced": [0.25, 0.25, 0.25, 0.25],  # Hybrid / reckless
}
_LOOT_KINDS = ["weapon", "spell", "armor", "boots"]
_LOOT_POOLS = {"weapon": WEAPONS, "spell": SPELLS, "armor": ARMORS, "boots": BOOTS}

# ---------------------------------------------------------------------------
# Combat metrics
# ---------------------------------------------------------------------------

@dataclass
class CombatMetrics:
    melee_kills:  int   = 0
    magic_kills:  int   = 0
    melee_hits:   int   = 0
    magic_hits:   int   = 0
    damage_taken: float = 0.0
    damage_dealt: float = 0.0
    deaths:       int   = 0
    time_alive:   float = 0.0

# ---------------------------------------------------------------------------
# Simulation objects
# ---------------------------------------------------------------------------

class BotProjectile:
    def __init__(self, x, y, vx, vy, dmg):
        self.x, self.y   = x, y
        self.vx, self.vy = vx, vy
        self.dmg   = dmg
        self.r     = max(6, min(18, int(4 + dmg * 0.3)))
        self.alive = True

    def update(self, dt, arena):
        self.x += self.vx * dt
        self.y += self.vy * dt
        ax, ay, aw, ah = arena
        if not (ax <= self.x <= ax + aw and ay <= self.y <= ay + ah):
            self.alive = False


class BotEnemy:
    def __init__(self, x, y, hp, dmg, speed, aggro_r):
        self.x, self.y = x, y
        self.hp        = hp
        self.max_hp    = hp
        self.dmg       = dmg
        self.speed     = speed
        self.aggro_r   = aggro_r
        self.r         = 16
        self.alive     = True
        self.atk_cd    = 0.0
        self.last_hit_source = "melee"

    def take_damage(self, amount, source="melee"):
        self.last_hit_source = source
        self.hp -= amount
        if self.hp <= 0:
            self.alive = False

    def update(self, dt, player, arena):
        if not self.alive:
            return
        self.atk_cd = max(0.0, self.atk_cd - dt)
        dx, dy = player.x - self.x, player.y - self.y
        d = vec_len(dx, dy)
        if d <= self.aggro_r:
            nx, ny  = norm(dx, dy)
            self.x += nx * self.speed * dt
            self.y += ny * self.speed * dt
        ax, ay, aw, ah = arena
        self.x = clamp(self.x, ax + self.r, ax + aw - self.r)
        self.y = clamp(self.y, ay + self.r, ay + ah - self.r)
        if d <= (self.r + player.r + 6) and self.atk_cd <= 0.0:
            self.atk_cd = 0.75
            player.receive_damage(self.dmg)


class BotPlayer:
    def __init__(self):
        ax = ARENA_MARGIN
        ay = ARENA_MARGIN
        self.x = ax + (W - 2 * ARENA_MARGIN) / 2
        self.y = ay + (H - 2 * ARENA_MARGIN) / 2
        self.r = 18

        self.max_hp = 100
        self.hp     = 100

        self.weapon = random.choice(WEAPONS)
        self.spell  = random.choice(SPELLS)
        self.boots  = random.choice(BOOTS)
        self.armor  = random.choice(ARMORS)

        self.facing_deg = 0.0
        self.melee_cd   = 0.0
        self.magic_cd   = 0.0
        self.magic_ammo = self.spell.ammo_max
        self.dash_cd    = 0.0
        self.dashing    = False
        self.dash_t     = 0.0
        self.dash_dir   = (0.0, 0.0)
        self.defending  = False
        self.alive      = True
        self.metrics    = CombatMetrics()

    def receive_damage(self, raw):
        if not self.alive:
            return
        amount = max(0.0, raw - self.armor.dmg_absorb)
        if self.defending:
            amount *= DEFEND_DMG_MULT
        self.hp -= amount
        self.metrics.damage_taken += amount
        if self.hp <= 0:
            self.alive = False
            self.metrics.deaths += 1

    def heal_on_kill(self):
        heal    = int(HEAL_ON_KILL_BASE * self.weapon.heal_mult) + self.armor.heal_on_kill_bonus
        self.hp = min(self.max_hp, self.hp + heal)

    def apply_movement(self, dx, dy, dt, arena):
        nx, ny = norm(dx, dy)
        speed  = PLAYER_SPEED * (DEFEND_SLOW if self.defending else 1.0)
        if self.dashing:
            self.dash_t += dt
            self.x += self.dash_dir[0] * DASH_SPEED * dt
            self.y += self.dash_dir[1] * DASH_SPEED * dt
            if self.dash_t >= DASH_TIME:
                self.dashing = False
                self.dash_t  = 0.0
        else:
            self.x += nx * speed * dt
            self.y += ny * speed * dt
        ax, ay, aw, ah = arena
        self.x = clamp(self.x, ax + self.r, ax + aw - self.r)
        self.y = clamp(self.y, ay + self.r, ay + ah - self.r)
        self.melee_cd = max(0.0, self.melee_cd - dt)
        self.magic_cd = max(0.0, self.magic_cd - dt)
        self.dash_cd  = max(0.0, self.dash_cd  - dt)
        self.metrics.time_alive += dt

    def try_melee(self, enemies):
        if not self.alive or self.melee_cd > 0.0:
            return
        self.melee_cd = MELEE_CD_BASE * self.weapon.cd_mult
        for e in enemies:
            if not e.alive:
                continue
            dx, dy = e.x - self.x, e.y - self.y
            d = vec_len(dx, dy)
            if d > MELEE_RANGE + e.r:
                continue
            ang = angle_deg(dx, dy)
            if abs(angle_diff_deg(ang, self.facing_deg)) <= (MELEE_ARC_DEG / 2):
                e.take_damage(self.weapon.dmg, source="melee")
                self.metrics.melee_hits   += 1
                self.metrics.damage_dealt += self.weapon.dmg

    def try_magic(self, projectiles):
        if not self.alive or self.magic_cd > 0.0 or self.magic_ammo <= 0:
            return
        self.magic_cd   = MAGIC_CD_BASE * self.spell.cd_mult
        self.magic_ammo -= 1
        rad = math.radians(self.facing_deg)
        vx  = math.cos(rad) * PROJECTILE_SPEED
        vy  = math.sin(rad) * PROJECTILE_SPEED
        px  = self.x + math.cos(rad) * (self.r + 8)
        py  = self.y + math.sin(rad) * (self.r + 8)
        projectiles.append(BotProjectile(px, py, vx, vy, self.spell.dmg))

    def try_dash(self):
        if self.dash_cd > 0.0 or self.dashing:
            return
        rad           = math.radians(self.facing_deg)
        self.dashing  = True
        self.dash_dir = (math.cos(rad), math.sin(rad))
        self.dash_cd  = DASH_CD

# ---------------------------------------------------------------------------
# Playstyle archetypes
# ---------------------------------------------------------------------------

@dataclass
class Archetype:
    name: str
    preferred_range: float
    p_melee:   float
    p_magic:   float
    p_defend:  float
    p_dash:    float
    loot_bias: str    # key into _LOOT_WEIGHTS


ARCHETYPES = [
    Archetype("melee_aggressive", preferred_range=30,  p_melee=0.92, p_magic=0.08, p_defend=0.05, p_dash=0.3,  loot_bias="melee"),
    Archetype("magic_sniper",     preferred_range=200, p_melee=0.05, p_magic=0.95, p_defend=0.10, p_dash=0.5,  loot_bias="magic"),
    Archetype("hybrid",           preferred_range=90,  p_melee=0.50, p_magic=0.50, p_defend=0.15, p_dash=0.4,  loot_bias="balanced"),
    Archetype("tank",             preferred_range=25,  p_melee=0.80, p_magic=0.10, p_defend=0.55, p_dash=0.1,  loot_bias="tank"),
    Archetype("reckless",         preferred_range=0,   p_melee=0.60, p_magic=0.60, p_defend=0.00, p_dash=0.8,  loot_bias="balanced"),
]

# Module-level lookup so worker processes can find archetypes by name.
ARCHETYPE_BY_NAME = {a.name: a for a in ARCHETYPES}

# ---------------------------------------------------------------------------
# Existing-run analysis: read CSV, count runs per archetype
# ---------------------------------------------------------------------------

def _infer_archetype_from_row(row: dict) -> str:
    """
    Map a CSV row back to the closest bot archetype name.
    Bot rows embed the archetype name in the run_id prefix; human rows are
    bucketed by melee/magic ratio so they contribute to coverage counting.
    """
    run_id = row.get("run_id", "")
    for name in ARCHETYPE_BY_NAME:
        if run_id.startswith(f"bot_{name}"):
            return name
    # Human run: bucket by melee/magic split
    try:
        mk    = float(row.get("melee_kills", 0))
        mgk   = float(row.get("magic_kills", 0))
        total = mk + mgk
        if total == 0:
            return "hybrid"
        ratio = mk / total
        if ratio > 0.75:
            return "melee_aggressive"
        if ratio < 0.25:
            return "magic_sniper"
        return "hybrid"
    except (ValueError, TypeError):
        return "hybrid"


def read_existing_counts(log_path: str) -> Counter:
    """Return per-archetype run counts already present in log_path."""
    counts: Counter = Counter()
    if not os.path.exists(log_path):
        return counts
    try:
        with open(log_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                counts[_infer_archetype_from_row(row)] += 1
    except Exception:
        pass
    return counts


def compute_target_runs(
    total_new: int,
    existing: Counter,
    archetypes: List[Archetype],
) -> Dict[str, int]:
    """
    Distribute `total_new` runs so the final combined dataset is as balanced
    as possible.  Each archetype receives at least 1 new run.

    Algorithm:
      target_per_arch = (existing_total + total_new) / n
      raw_allocation  = max(1, target_per_arch - existing[arch])
      scale to sum to total_new; fix rounding on the most under-represented arch.
    """
    n          = len(archetypes)
    names      = [a.name for a in archetypes]
    ex_total   = sum(existing[name] for name in names)
    grand      = ex_total + total_new
    target     = grand / n

    raw     = {name: max(1.0, target - existing[name]) for name in names}
    raw_sum = sum(raw.values())

    scaled  = {name: max(1, int(round(v / raw_sum * total_new))) for name, v in raw.items()}

    # Correct integer rounding drift
    diff = total_new - sum(scaled.values())
    if diff != 0:
        most_under = min(names, key=lambda n: existing[n] + scaled[n])
        scaled[most_under] += diff

    return scaled

# ---------------------------------------------------------------------------
# Rule-based DDA (same logic as game.py / cGAN.py)
# ---------------------------------------------------------------------------

def rule_based_dda(metrics: CombatMetrics, node_depth: int) -> dict:
    total_kills    = metrics.melee_kills + metrics.magic_kills
    deaths_per_min = (metrics.deaths / max(metrics.time_alive, 1e-6)) * 60.0
    base_mult      = 1.0 + total_kills * 0.045
    if deaths_per_min > 1.5:
        base_mult *= 0.78
    elif deaths_per_min < 0.2 and total_kills > 5:
        base_mult *= 1.22
    depth_mult = 1.0 + node_depth * 0.06
    hp_mult    = clamp(base_mult * depth_mult,         0.55, 3.0)
    dmg_mult   = clamp(base_mult * depth_mult * 0.85,  0.45, 2.4)
    speed_mult = clamp(1.0 + (base_mult - 1.0) * 0.5, 0.80, 1.70)
    spawn_n    = clamp(int(2 + total_kills * 0.18) + node_depth // 3, 2, 10)
    return {
        "enemy_hp_mult":    round(hp_mult,    3),
        "enemy_dmg_mult":   round(dmg_mult,   3),
        "enemy_speed_mult": round(speed_mult, 3),
        "spawn_count":      spawn_n,
    }

# ---------------------------------------------------------------------------
# Node-depth helper
# ---------------------------------------------------------------------------

def compute_node_depth(node_map: NodeMap, start: int):
    from collections import deque
    depth = [-1] * len(node_map.nodes)
    depth[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in node_map.neighbors(u):
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                q.append(v)
    return depth

# ---------------------------------------------------------------------------
# Loot picking — consistent with rogue.py weighted sampler + cGAN loot targets
# ---------------------------------------------------------------------------

def pick_loot_weighted(arch: Archetype, rng: random.Random) -> Tuple[str, object]:
    """Pick a (kind, item) tuple using the archetype's loot weight vector."""
    weights = _LOOT_WEIGHTS.get(arch.loot_bias, _LOOT_WEIGHTS["balanced"])
    kind    = rng.choices(_LOOT_KINDS, weights=weights, k=1)[0]
    item    = rng.choice(_LOOT_POOLS[kind])
    return kind, item

# ---------------------------------------------------------------------------
# Room simulation — mirrors rogue.py spawn_room node-kind logic
# ---------------------------------------------------------------------------

def simulate_room(
    player: BotPlayer,
    arch: Archetype,
    kind: str,
    applied: dict,
    rng: random.Random,
) -> bool:
    """Simulate one room. Returns True if player survived, False on death."""
    # Non-combat nodes need no fight — mirrors rogue.py spawn_room
    if kind in ("start", "loot", "shop"):
        return True

    arena = (ARENA_MARGIN, ARENA_MARGIN,
             W - 2 * ARENA_MARGIN, H - 2 * ARENA_MARGIN)
    ax, ay, aw, ah = arena

    n     = applied["spawn_count"]
    hp_b  = ENEMY_BASE_HP  * applied["enemy_hp_mult"]
    dmg_b = ENEMY_BASE_DMG * applied["enemy_dmg_mult"]
    spd_b = ENEMY_SPEED    * applied["enemy_speed_mult"]

    if kind == "boss":
        n = 1
        hp_b  *= 4.0
        dmg_b *= 1.7
        spd_b *= 0.9
    elif kind == "elite":
        hp_b  *= 1.8
        dmg_b *= 1.4
        spd_b *= 1.15
        n = max(1, n - 1)

    enemies: List[BotEnemy] = []
    for _ in range(n):
        for _a in range(50):
            ex = rng.randint(ax + 80, ax + aw - 80)
            ey = rng.randint(ay + 80, ay + ah - 80)
            if vec_len(ex - player.x, ey - player.y) >= ENEMY_SPAWN_MIN_DIST:
                break
        enemies.append(BotEnemy(ex, ey, hp=hp_b, dmg=dmg_b, speed=spd_b, aggro_r=ENEMY_AGGRO_R))

    player.magic_ammo = player.spell.ammo_max
    projectiles: List[BotProjectile] = []

    max_ticks = int(120 / SIM_DT)   # 2 simulated minutes
    tick = 0

    while tick < max_ticks:
        tick += 1
        dt = SIM_DT

        alive_enemies = [e for e in enemies if e.alive]
        if not alive_enemies or not player.alive:
            break

        nearest    = min(alive_enemies, key=lambda e: vec_len(e.x - player.x, e.y - player.y))
        nx_e, ny_e = nearest.x, nearest.y
        d_near     = vec_len(nx_e - player.x, ny_e - player.y)

        player.facing_deg = angle_deg(nx_e - player.x, ny_e - player.y)

        if d_near < arch.preferred_range - 10:
            move_x = -(nx_e - player.x)
            move_y = -(ny_e - player.y)
        else:
            move_x = (nx_e - player.x)
            move_y = (ny_e - player.y)

        if arch.name == "reckless":
            move_x += rng.uniform(-0.5, 0.5) * abs(move_x + 1)
            move_y += rng.uniform(-0.5, 0.5) * abs(move_y + 1)

        player.defending = rng.random() < arch.p_defend * dt * 10

        if d_near <= MELEE_RANGE + nearest.r + 5:
            if rng.random() < arch.p_melee:
                player.try_melee(enemies)

        if rng.random() < arch.p_magic * dt * 8:
            player.try_magic(projectiles)

        if rng.random() < arch.p_dash * dt:
            player.try_dash()

        player.apply_movement(move_x, move_y, dt, arena)

        for e in enemies:
            e.update(dt, player, arena)

        for p in projectiles:
            p.update(dt, arena)
        projectiles = [p for p in projectiles if p.alive]

        for p in projectiles:
            for e in enemies:
                if not e.alive:
                    continue
                if vec_len(e.x - p.x, e.y - p.y) <= (e.r + p.r):
                    e.take_damage(p.dmg, source="magic")
                    player.metrics.magic_hits   += 1
                    player.metrics.damage_dealt += p.dmg
                    p.alive = False
                    break

        for e in enemies:
            if e.alive or getattr(e, "_counted", False):
                continue
            setattr(e, "_counted", True)
            if e.last_hit_source == "melee":
                player.metrics.melee_kills += 1
            else:
                player.metrics.magic_kills += 1
            player.heal_on_kill()

    return player.alive

# ---------------------------------------------------------------------------
# Full run simulation
# ---------------------------------------------------------------------------

def simulate_run(arch: Archetype, rng: random.Random) -> Tuple[BotPlayer, dict]:
    """Simulate one full run. Returns (player, last_applied_params)."""
    node_map    = generate_node_map_graph(18, seed=rng.randint(0, 999999))
    node_depths = compute_node_depth(node_map, node_map.start)

    player  = BotPlayer()
    applied = rule_based_dda(player.metrics, node_depth=0)

    current_idx = node_map.start

    def greedy_next(cur_idx: int) -> Optional[int]:
        neighbors = [
            i for i in node_map.neighbors(cur_idx)
            if not node_map.nodes[i].cleared
        ]
        if not neighbors:
            return None
        return max(neighbors, key=lambda i: node_depths[i])

    MAX_ROOMS = 25
    rooms_visited = 0

    while rooms_visited < MAX_ROOMS:
        node  = node_map.nodes[current_idx]
        depth = node_depths[current_idx]
        applied = rule_based_dda(player.metrics, node_depth=depth)

        survived = simulate_room(player, arch, node.kind, applied, rng)
        rooms_visited += 1

        if not survived:
            break

        node.cleared = True

        if node.kind == "boss":
            break

        # Loot pick — archetype-weighted, consistent with rogue.py + cGAN loot targets
        kind, item = pick_loot_weighted(arch, rng)
        if kind == "weapon":
            player.weapon = item
        elif kind == "spell":
            player.spell      = item
            player.magic_ammo = item.ammo_max
        elif kind == "boots":
            player.boots = item
        elif kind == "armor":
            player.armor = item

        nxt = greedy_next(current_idx)
        if nxt is None:
            break
        current_idx = nxt

    return player, applied

# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

def ensure_log_header(log_path: str):
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "run_id",
                "melee_kills", "magic_kills",
                "melee_hits",  "magic_hits",
                "damage_taken", "damage_dealt",
                "deaths", "time_alive",
                "weapon", "spell", "boots", "armor",
                "enemy_hp_mult", "enemy_dmg_mult", "enemy_speed_mult", "spawn_count",
                "heal_on_kill_base",
            ])


def append_run(log_path: str, run_id: str, player: BotPlayer, applied: dict):
    m = player.metrics
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            run_id,
            m.melee_kills, m.magic_kills,
            m.melee_hits,  m.magic_hits,
            round(m.damage_taken, 3), round(m.damage_dealt, 3),
            m.deaths, round(m.time_alive, 3),
            player.weapon.name, player.spell.name,
            player.boots.name,  player.armor.name,
            applied["enemy_hp_mult"], applied["enemy_dmg_mult"],
            applied["enemy_speed_mult"], applied["spawn_count"],
            HEAL_ON_KILL_BASE,
        ])

# ---------------------------------------------------------------------------
# Worker function — executed inside a subprocess via ProcessPoolExecutor.
# Must be a top-level function (not a closure) to be picklable.
# ---------------------------------------------------------------------------

def _run_archetype_batch(
    arch_name: str,
    count: int,
    base_seed: int,
    log_path: str,
    start_index: int,
) -> Tuple[str, int, float]:
    """
    Simulate `count` runs for arch_name, append results to log_path.
    Returns (arch_name, runs_completed, wall_seconds).
    """
    arch = ARCHETYPE_BY_NAME[arch_name]
    rng  = random.Random(base_seed)
    t0   = time.perf_counter()

    for i in range(count):
        global_idx      = start_index + i
        player, applied = simulate_run(arch, rng)
        run_id          = f"bot_{arch_name}_{global_idx:05d}"
        append_run(log_path, run_id, player, applied)

    return arch_name, count, time.perf_counter() - t0

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Headless bot runner for DDA data generation")
    parser.add_argument("--runs",         type=int, default=200,
                        help="Total NEW bot runs to simulate")
    parser.add_argument("--seed",         type=int, default=None,
                        help="Global RNG seed for reproducibility")
    parser.add_argument("--log",          type=str, default=LOG_FILE,
                        help="Output CSV path")
    parser.add_argument("--workers",      type=int, default=os.cpu_count() or 4,
                        help="Parallel worker processes (default: all CPU cores)")
    parser.add_argument("--train",        action="store_true",
                        help="Launch train_cgan.py automatically when bot runs finish")
    parser.add_argument("--train-script", type=str, default="train_cgan.py",
                        help="Path to train_cgan.py (used with --train)")
    parser.add_argument("--train-out",    type=str, default="models",
                        help="Model output dir passed to train_cgan.py")
    parser.add_argument("--train-epochs", type=int, default=150,
                        help="Training epochs passed to train_cgan.py")
    parser.add_argument("--train-batch",  type=int, default=64,
                        help="Batch size passed to train_cgan.py")
    args = parser.parse_args()

    log_path = args.log
    ensure_log_header(log_path)

    # ------------------------------------------------------------------
    # 1. Analyse existing data and plan archetype distribution
    # ------------------------------------------------------------------
    existing_counts = read_existing_counts(log_path)
    total_existing  = sum(existing_counts.values())

    print(f"\nExisting runs in {log_path}: {total_existing}")
    if total_existing:
        for a in ARCHETYPES:
            print(f"  {a.name:<22} {existing_counts[a.name]:>5}")

    target_runs = compute_target_runs(args.runs, existing_counts, ARCHETYPES)

    print(f"\nNew runs planned (total {args.runs}, re-balanced to fill gaps):")
    for a in ARCHETYPES:
        n   = target_runs[a.name]
        pct = n / args.runs * 100
        print(f"  {a.name:<22} {n:>4} new  ({pct:.1f}%)")
    print()

    # ------------------------------------------------------------------
    # 2. Per-archetype seeds derived from master seed
    # ------------------------------------------------------------------
    master_rng = random.Random(args.seed)
    arch_seeds = {a.name: master_rng.randint(0, 2**31 - 1) for a in ARCHETYPES}
    # Start indices so new run IDs never collide with existing ones
    existing_per_arch = {a.name: existing_counts[a.name] for a in ARCHETYPES}

    # ------------------------------------------------------------------
    # 3. Parallel simulation — one worker per archetype
    # ------------------------------------------------------------------
    workers = max(1, min(args.workers, len(ARCHETYPES)))
    print(f"Simulating with {workers} worker process(es)…\n")

    t_start   = time.perf_counter()
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run_archetype_batch,
                arch.name,
                target_runs[arch.name],
                arch_seeds[arch.name],
                log_path,
                existing_per_arch[arch.name],
            ): arch.name
            for arch in ARCHETYPES
            if target_runs[arch.name] > 0
        }

        for future in as_completed(futures):
            arch_name, count, elapsed = future.result()
            completed += count
            rate = count / elapsed if elapsed > 0 else 0.0
            print(f"  ✓ {arch_name:<22} {count:>4} runs  "
                  f"({elapsed:.1f}s, {rate:.1f} runs/s)  "
                  f"[{completed}/{args.runs} done]")

    wall = time.perf_counter() - t_start
    total_after = total_existing + completed
    print(f"\nDone. {completed} new runs appended → {log_path}  "
          f"(wall: {wall:.1f}s,  total rows: {total_after})")

    # ------------------------------------------------------------------
    # 4. Optionally chain into cGAN training
    # ------------------------------------------------------------------
    if args.train:
        train_script = args.train_script
        if not os.path.exists(train_script):
            print(f"\n[ERROR] --train set but '{train_script}' not found. "
                  f"Use --train-script to specify its path.")
            sys.exit(1)

        cmd = [
            sys.executable, train_script,
            "--csv",    log_path,
            "--out",    args.train_out,
            "--epochs", str(args.train_epochs),
            "--batch",  str(args.train_batch),
            "--device", "auto",          # train_cgan.py resolves auto → CUDA/MPS/CPU
        ]
        print(f"\nLaunching cGAN training:\n  {' '.join(cmd)}\n")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[ERROR] train_cgan.py exited with code {result.returncode}")
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
