import math
import os
import csv
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import pygame

from node_map_gen import NodeMap, MapNode, generate_node_map_graph




W, H = 960, 540
FPS = 60
ARENA_MARGIN = 40

PLAYER_SPEED = 220.0
DASH_SPEED = 600.0
DASH_TIME = 0.12
DASH_CD = 1.25  

MELEE_RANGE = 42
MELEE_ARC_DEG = 90
MELEE_CD_BASE = 0.45  
PROJECTILE_SPEED = 420.0
MAGIC_CD_BASE = 0.60  
MAGIC_AMMO_MAX = 12

DEFEND_SLOW = 0.55
DEFEND_DMG_MULT = 0.55

HEAL_ON_KILL_BASE = 8 

ENEMY_BASE_HP = 42
ENEMY_BASE_DMG = 9
ENEMY_SPEED = 110.0
ENEMY_AGGRO_R = 260.0

FONT_NAME = None

ENABLE_MODEL = False
MODEL_DIR = "models"

# Minimum distance from player center at which enemies can spawn
ENEMY_SPAWN_MIN_DIST = 130




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
    d = (a - b + 180) % 360 - 180
    return d






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
    Weapon("Rusty Blade", dmg=11, cd_mult=1.00, heal_mult=1.00),
    Weapon("Hatchet", dmg=14, cd_mult=1.12, heal_mult=1.05),
    Weapon("Rapier", dmg=9, cd_mult=0.80, heal_mult=0.95),
]
SPELLS = [
    Spell("Ember Bolt", dmg=10, cd_mult=1.00, ammo_max=12),
    Spell("Ice Needle", dmg=8, cd_mult=0.78, ammo_max=16),
    Spell("Hex Spike", dmg=14, cd_mult=1.25, ammo_max=9),
]
BOOTS = [
    Boots("Leather Boots", dash_dist_mult=1.00),
    Boots("Sprint Greaves", dash_dist_mult=1.25),
    Boots("Voidstep Treads", dash_dist_mult=1.45),
]
ARMORS = [
    Armor("Cloth Wrap", dmg_absorb=0.5, heal_on_kill_bonus=0),
    Armor("Chain Shirt", dmg_absorb=2.0, heal_on_kill_bonus=0),
    Armor("Blood Harness", dmg_absorb=1.0, heal_on_kill_bonus=6),  # berserker
]



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

    def to_feature_vector(self) -> List[float]:
        # melee_ratio, magic_ratio, hits_per_kill_melee, hits_per_kill_magic, dmg_taken/dmg_dealt, death_rate
        total_kills = self.melee_kills + self.magic_kills
        melee_ratio = (self.melee_kills / total_kills) if total_kills > 0 else 0.5
        magic_ratio = (self.magic_kills / total_kills) if total_kills > 0 else 0.5
        hpk_melee = (self.melee_hits / self.melee_kills) if self.melee_kills > 0 else float(self.melee_hits + 1)
        hpk_magic = (self.magic_hits / self.magic_kills) if self.magic_kills > 0 else float(self.magic_hits + 1)
        dmg_ratio = (self.damage_taken / self.damage_dealt) if self.damage_dealt > 1e-6 else 1.0
        death_rate = (self.deaths / max(self.time_alive, 1e-6)) * 60.0  # deaths per minute
        return [
            clamp(melee_ratio, 0.0, 1.0),
            clamp(magic_ratio, 0.0, 1.0),
            clamp(hpk_melee, 0.0, 10.0),
            clamp(hpk_magic, 0.0, 10.0),
            clamp(dmg_ratio, 0.0, 5.0),
            clamp(death_rate, 0.0, 5.0),
        ]


def rule_based_dda(player: "Player", node_depth: int = 0) -> dict:
    """
    Heuristic DDA used until enough runs exist to train the cGAN.
    """
    m = player.metrics

    total_kills = m.melee_kills + m.magic_kills
    deaths_per_min = (m.deaths / max(m.time_alive, 1e-6)) * 60.0

    # Combat style ratios
    if total_kills > 0:
        melee_ratio = m.melee_kills / total_kills
        magic_ratio = m.magic_kills / total_kills
    else:
        melee_ratio = 0.5
        magic_ratio = 0.5

    dmg_ratio = m.damage_taken / max(m.damage_dealt, 1e-6)

    # Base: grows with kills, dampened by deaths
    base_mult = 1.0 + (total_kills * 0.045)

    if deaths_per_min > 1.5:
        base_mult *= 0.78
    elif deaths_per_min < 0.2 and total_kills > 5:
        base_mult *= 1.22

    depth_mult = 1.0 + node_depth * 0.06

    hp_mult = clamp(base_mult * depth_mult, 0.55, 3.0)
    dmg_mult = clamp(base_mult * depth_mult * 0.85, 0.45, 2.4)
    speed_mult = clamp(1.0 + (base_mult - 1.0) * 0.5, 0.80, 1.70)
    spawn_n = clamp(int(2 + total_kills * 0.18) + node_depth // 3, 2, 10)

    # Rule-based loot bias
    # weights order: [weapon, spell, armor, boots]
    if melee_ratio >= 0.55 and dmg_ratio > 1.2:
        loot_bias = [0.30, 0.05, 0.45, 0.20]  # Berserker
    elif melee_ratio >= 0.55:
        loot_bias = [0.55, 0.05, 0.25, 0.15]  # Knight
    else:
        loot_bias = [0.05, 0.55, 0.20, 0.20]  # Spell Sniper

    return {
        "enemy_hp_mult": round(hp_mult, 3),
        "enemy_dmg_mult": round(dmg_mult, 3),
        "enemy_speed_mult": round(speed_mult, 3),
        "spawn_count": spawn_n,
        "loot_bias": loot_bias,
    }


def apply_model_tuning_if_available(player: "Player", node_depth: int = 0) -> Optional[dict]:
    """
    Tries the cGAN model first; falls back to rule_based_dda transparently.

    The returned dict always contains both difficulty keys AND a 'loot_bias'
    key ([weapon_w, spell_w, armor_w, boots_w]) so generate_loot_options can
    do archetype-aware weighted sampling without needing a separate call.
    """
    if ENABLE_MODEL:
        try:
            import torch
            from model_runtime import LoadedModels
            global _LOADED
            if "_LOADED" not in globals():
                _LOADED = LoadedModels.load(MODEL_DIR)

            features = torch.tensor([player.metrics.to_feature_vector()], dtype=torch.float32)

            # Difficulty tuning (existing path)
            gen = _LOADED.generate(features)
            if gen:
                # Loot bias from LootGenerator, conditioned on the same
                # archetype soft-vector the difficulty GAN uses internally.
                try:
                    loot_bias = _LOADED.generate_loot_bias(features)
                    if loot_bias:
                        gen["loot_bias"] = loot_bias
                    else:
                        gen.setdefault("loot_bias", _fallback_loot_bias(player))
                except Exception:
                    gen.setdefault("loot_bias", _fallback_loot_bias(player))
                return gen
        except Exception as e:
            print("[MODEL] Failed to load/apply model:", e)

    return rule_based_dda(player, node_depth)


def _fallback_loot_bias(player: "Player") -> list:
    """Rule-based loot bias when the model is unavailable."""
    m = player.metrics
    total_k = m.melee_kills + m.magic_kills
    melee_ratio = (m.melee_kills / total_k) if total_k > 0 else 0.5
    dmg_ratio = m.damage_taken / max(m.damage_dealt, 1e-6)
    if melee_ratio >= 0.55 and dmg_ratio > 1.2:
        return [0.30, 0.05, 0.45, 0.20]  # Berserker
    elif melee_ratio >= 0.55:
        return [0.55, 0.05, 0.25, 0.15]  # Knight
    else:
        return [0.05, 0.55, 0.20, 0.20]  # Sniper



### ============= Entity Classes ======== ###
class Projectile:
    def __init__(self, x, y, vx, vy, dmg):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.dmg = dmg
        self.r = max(6, min(18, int(4 + dmg * 0.3)))
        self.alive = True

    def update(self, dt, walls_rect):
        self.x += self.vx * dt
        self.y += self.vy * dt
        if not walls_rect.collidepoint(self.x, self.y):
            self.alive = False

    def draw(self, surf):
        pygame.draw.circle(surf, (160, 220, 255), (int(self.x), int(self.y)), self.r)

class Enemy:
    def __init__(self, x, y, hp, dmg, speed, aggro_r):
        self.x, self.y = x, y
        self.hp = hp
        self.max_hp = hp
        self.dmg = dmg
        self.speed = speed
        self.aggro_r = aggro_r
        self.r = 16
        self.alive = True
        self.atk_cd = 0.0
        self.last_hit_source: str = "melee"

    def take_damage(self, amount, source: str = "melee"):
        self.last_hit_source = source  
        self.hp -= amount
        if self.hp <= 0:
            self.alive = False

    def update(self, dt, player, arena_rect):
        if not self.alive:
            return
        self.atk_cd = max(0.0, self.atk_cd - dt)
        dx, dy = (player.x - self.x), (player.y - self.y)
        dist   = vec_len(dx, dy)

        if dist <= self.aggro_r:
            nx, ny  = norm(dx, dy)
            self.x += nx * self.speed * dt
            self.y += ny * self.speed * dt

        self.x = clamp(self.x, arena_rect.left  + self.r, arena_rect.right  - self.r)
        self.y = clamp(self.y, arena_rect.top   + self.r, arena_rect.bottom - self.r)

        if dist <= (self.r + player.r + 6) and self.atk_cd <= 0.0:
            self.atk_cd = 0.75
            player.receive_damage(self.dmg)

    def draw(self, surf):
        if not self.alive:
            return
        pygame.draw.circle(surf, (240, 120, 120), (int(self.x), int(self.y)), self.r)
        bar_w    = 34
        hp_ratio = max(0.0, self.hp / self.max_hp)
        pygame.draw.rect(surf, (40, 40, 40),   (int(self.x - bar_w/2), int(self.y - 28), bar_w, 6))
        pygame.draw.rect(surf, (80, 220, 80),  (int(self.x - bar_w/2), int(self.y - 28), int(bar_w * hp_ratio), 6))


class Player:
    def __init__(self, x, y):
        self.x, self.y  = x, y
        self.r          = 18
        self.max_hp     = 100
        self.hp         = 100

        self.weapon = random.choice(WEAPONS)
        self.spell  = random.choice(SPELLS)
        self.boots  = random.choice(BOOTS)
        self.armor  = random.choice(ARMORS)

        self.facing_deg = 0.0
        self.melee_cd   = 0.0
        self.magic_cd   = 0.0
        self.magic_ammo = self.spell.ammo_max

        self.defending = False
        self.dashing   = False
        self.dash_t    = 0.0
        self.dash_cd   = 0.0
        self.dash_dir  = (0.0, 0.0)

        self.alive   = True
        self.metrics = CombatMetrics()

    def set_loadout(self, weapon=None, spell=None, boots=None, armor=None):
        if weapon: self.weapon = weapon
        if spell:
            self.spell      = spell
            self.magic_ammo = min(self.magic_ammo, self.spell.ammo_max)
        if boots: self.boots = boots
        if armor: self.armor = armor

    def heal_on_kill(self):
        heal   = int(HEAL_ON_KILL_BASE * self.weapon.heal_mult) + self.armor.heal_on_kill_bonus
        self.hp = min(self.max_hp, self.hp + heal)

    def receive_damage(self, raw_amount):
        if not self.alive:
            return
        amount = max(0.0, raw_amount - self.armor.dmg_absorb)
        if self.defending:
            amount *= DEFEND_DMG_MULT
        self.hp -= amount
        self.metrics.damage_taken += amount
        if self.hp <= 0:
            self.alive  = False
            self.metrics.deaths += 1

    def update(self, dt, keys, arena_rect):
        if not self.alive:
            return

        self.metrics.time_alive += dt
        self.melee_cd = max(0.0, self.melee_cd - dt)
        self.magic_cd = max(0.0, self.magic_cd - dt)
        self.dash_cd  = max(0.0, self.dash_cd  - dt)

        self.defending = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        mx = (1 if keys[pygame.K_d] else 0) - (1 if keys[pygame.K_a] else 0)
        my = (1 if keys[pygame.K_s] else 0) - (1 if keys[pygame.K_w] else 0)
        nx, ny = norm(mx, my)

        if self.dashing:
            self.dash_t += dt
            self.x += self.dash_dir[0] * DASH_SPEED * dt
            self.y += self.dash_dir[1] * DASH_SPEED * dt
            if self.dash_t >= DASH_TIME:
                self.dashing = False
                self.dash_t  = 0.0
        else:
            speed = PLAYER_SPEED
            if self.defending:
                speed *= DEFEND_SLOW
            self.x += nx * speed * dt
            self.y += ny * speed * dt

        self.x = clamp(self.x, arena_rect.left  + self.r, arena_rect.right  - self.r)
        self.y = clamp(self.y, arena_rect.top   + self.r, arena_rect.bottom - self.r)

    def try_dash(self):
        if not self.alive:
            return
        if self.dash_cd > 0.0 or self.dashing:
            return
        rad = math.radians(self.facing_deg)
        dx, dy       = math.cos(rad), math.sin(rad)
        self.dashing  = True
        self.dash_dir = (dx, dy)
        self.dash_cd  = DASH_CD

    def try_melee(self, enemies: List[Enemy]):
        if not self.alive or self.melee_cd > 0.0:
            return False
        self.melee_cd = MELEE_CD_BASE * self.weapon.cd_mult
        hit_any = False
        for e in enemies:
            if not e.alive:
                continue
            dx, dy = (e.x - self.x), (e.y - self.y)
            d_dist  = vec_len(dx, dy)
            if d_dist > MELEE_RANGE + e.r:
                continue
            ang = angle_deg(dx, dy)
            if abs(angle_diff_deg(ang, self.facing_deg)) <= (MELEE_ARC_DEG / 2):
                # FIX: pass source tag so kill attribution is accurate
                e.take_damage(self.weapon.dmg, source="melee")
                self.metrics.melee_hits  += 1
                self.metrics.damage_dealt += self.weapon.dmg
                hit_any = True
        return hit_any

    def try_magic(self, projectiles: List[Projectile]):
        if not self.alive or self.magic_cd > 0.0 or self.magic_ammo <= 0:
            return
        self.magic_cd   = MAGIC_CD_BASE * self.spell.cd_mult
        self.magic_ammo -= 1
        rad = math.radians(self.facing_deg)
        vx  = math.cos(rad) * PROJECTILE_SPEED
        vy  = math.sin(rad) * PROJECTILE_SPEED
        px  = self.x + math.cos(rad) * (self.r + 8)
        py  = self.y + math.sin(rad) * (self.r + 8)
        projectiles.append(Projectile(px, py, vx, vy, self.spell.dmg))

    def draw(self, surf):
        col = (130, 210, 140) if self.alive else (80, 80, 80)
        pygame.draw.circle(surf, col, (int(self.x), int(self.y)), self.r)
        rad = math.radians(self.facing_deg)
        fx  = self.x + math.cos(rad) * (self.r + 12)
        fy  = self.y + math.sin(rad) * (self.r + 12)
        pygame.draw.line(surf, (20, 20, 20), (int(self.x), int(self.y)), (int(fx), int(fy)), 3)


# ---------------------------------------------------------------------------
# Loot helpers
# ---------------------------------------------------------------------------

def generate_loot_options(
    player: Player,
    count: int = 3,
    loot_bias: Optional[List[float]] = None,
) -> List[Tuple[str, object]]:
    """
    Generate `count` loot choices weighted by `loot_bias`.

    loot_bias is a 4-element weight vector [weapon_w, spell_w, armor_w, boots_w]
    produced either by the LootGenerator model or the rule-based fallback.
    When None (e.g. very first room before any applied dict exists) we fall
    back to the old melee-ratio heuristic so existing callers keep working.
    """
    KINDS = ["weapon", "spell", "armor", "boots"]
    POOLS = {
        "weapon": WEAPONS,
        "spell":  SPELLS,
        "armor":  ARMORS,
        "boots":  BOOTS,
    }

    if loot_bias is not None and len(loot_bias) == 4:
        total = sum(loot_bias)
        weights = [w / total for w in loot_bias]  # normalise
    else:
        # Legacy fallback: simple melee/magic split
        total_k     = player.metrics.melee_kills + player.metrics.magic_kills
        melee_ratio = player.metrics.melee_kills / total_k if total_k > 0 else 0.5
        if melee_ratio >= 0.55:
            weights = [0.55, 0.05, 0.25, 0.15]
        else:
            weights = [0.05, 0.55, 0.20, 0.20]

    options: List[Tuple[str, object]] = []
    for _ in range(count):
        # Weighted random pick of item category
        kind = random.choices(KINDS, weights=weights, k=1)[0]
        item = random.choice(POOLS[kind])
        options.append((kind, item))
    return options


def describe_loot_option(kind: str, item: object) -> List[str]:
    if kind == "weapon":
        return [f"Weapon: {item.name}", f"Damage: {item.dmg}",
                f"Cooldown: {item.cd_mult:.2f}", f"Heal mult: {item.heal_mult:.2f}"]
    if kind == "spell":
        return [f"Spell: {item.name}", f"Damage: {item.dmg}",
                f"Cooldown: {item.cd_mult:.2f}", f"Ammo: {item.ammo_max}"]
    if kind == "boots":
        return [f"Boots: {item.name}", f"Dash mult: {item.dash_dist_mult:.2f}"]
    if kind == "armor":
        return [f"Armor: {item.name}", f"Absorb: {item.dmg_absorb:.2f}",
                f"Heal bonus: {item.heal_on_kill_bonus}"]
    return [f"Item: {item.name}"]


def apply_loot_choice(player: Player, choice: Tuple[str, object]):
    kind, item = choice
    if kind == "weapon":
        player.set_loadout(weapon=item)
    elif kind == "spell":
        player.set_loadout(spell=item)
        player.magic_ammo = player.spell.ammo_max
    elif kind == "boots":
        player.set_loadout(boots=item)
    elif kind == "armor":
        player.set_loadout(armor=item)


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

LOG_FILE = "runs.csv"

def ensure_log_header():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "run_id",
                "melee_kills","magic_kills",
                "melee_hits","magic_hits",
                "damage_taken","damage_dealt",
                "deaths","time_alive",
                "weapon","spell","boots","armor",
                "enemy_hp_mult","enemy_dmg_mult","enemy_speed_mult","spawn_count",
                "heal_on_kill_base",
            ])

def append_run(run_id: str, player: Player, applied: Dict[str, float]):
    ensure_log_header()
    m = player.metrics
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
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
# Node-depth helper
# ---------------------------------------------------------------------------

def compute_node_depth(node_map: NodeMap, start: int) -> List[int]:
    """BFS from start; returns depth[i] for every node i."""
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
# Main
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Roguelike DDA Prototype")
    clock = pygame.time.Clock()
    font  = pygame.font.Font(FONT_NAME, 18)
    big   = pygame.font.Font(FONT_NAME, 28)

    arena = pygame.Rect(ARENA_MARGIN, ARENA_MARGIN, W - 2*ARENA_MARGIN, H - 2*ARENA_MARGIN)

    node_map        = generate_node_map_graph(18)
    current_node_idx = node_map.start
    node_depths     = compute_node_depth(node_map, node_map.start)

    applied = rule_based_dda(Player(W/2, H/2), node_depth=0)

    player = Player(W/2, H/2)

    projectiles: List[Projectile]        = []
    enemies:     List[Enemy]             = []
    loot_options: List[Tuple[str, object]] = []
    loot_choice_idx = 0

    scale_x = arena.width  / 820.0
    scale_y = arena.height / 420.0

    def current_node():
        return node_map.nodes[current_node_idx]

    def all_enemies_dead():
        return all((not e.alive) for e in enemies)

    def spawn_room(kind: str):
        nonlocal enemies, projectiles, applied, state, loot_options, loot_choice_idx
        projectiles = []
        enemies     = []

        depth    = node_depths[current_node_idx]
        applied  = apply_model_tuning_if_available(player, node_depth=depth)

        # FIX: refill ammo on every room entry so magic is never permanently exhausted
        player.magic_ammo = player.spell.ammo_max

        # Non-combat nodes: skip enemy spawning entirely and jump straight to the
        # appropriate follow-up state so the player is never stuck fighting ghosts.
        if kind == "start":
            # Starting node — just show the arena with no enemies; player can open the map.
            state = "map"
            return

        if kind == "loot":
            # Pure loot node — immediately offer rewards with no fight.
            current_node().cleared = True
            loot_options    = generate_loot_options(player, 3, loot_bias=applied.get("loot_bias"))
            loot_choice_idx = 0
            state = "loot"
            return

        if kind == "shop":
            # Shop node — treat like loot for now (same reward screen).
            current_node().cleared = True
            loot_options    = generate_loot_options(player, 3, loot_bias=applied.get("loot_bias"))
            loot_choice_idx = 0
            state = "loot"
            return

        # --- Combat nodes: "combat", "elite", "boss" ---
        n = applied["spawn_count"]
        if kind == "boss":
            n = 1

        hp_base  = ENEMY_BASE_HP  * applied["enemy_hp_mult"]
        dmg_base = ENEMY_BASE_DMG * applied["enemy_dmg_mult"]
        spd_base = ENEMY_SPEED    * applied["enemy_speed_mult"]

        # Elite modifier: tougher enemies, slightly fewer of them
        if kind == "elite":
            hp_base  *= 1.8
            dmg_base *= 1.4
            spd_base *= 1.15
            n = max(1, n - 1)

        for _ in range(n):
            # FIX: ensure enemies never spawn on top of the player
            for _attempt in range(50):
                ex = random.randint(arena.left + 80, arena.right  - 80)
                ey = random.randint(arena.top  + 80, arena.bottom - 80)
                if vec_len(ex - player.x, ey - player.y) >= ENEMY_SPAWN_MIN_DIST:
                    break

            hp  = hp_base
            dmg = dmg_base
            spd = spd_base
            if kind == "boss":
                hp  *= 4.0
                dmg *= 1.7
                spd *= 0.9
            enemies.append(Enemy(ex, ey, hp=hp, dmg=dmg, speed=spd, aggro_r=ENEMY_AGGRO_R))

    state = "arena"   # default; spawn_room may override for non-combat start nodes
    spawn_room(node_map.nodes[current_node_idx].kind)

    run_id = f"run_{random.randint(10000, 99999)}"
    ensure_log_header()

    def draw_ui():
        hp_txt   = f"HP {int(player.hp)}/{player.max_hp}"
        ammo_txt = f"Ammo {player.magic_ammo}/{player.spell.ammo_max}"
        gear_txt = (f"W:{player.weapon.name} | S:{player.spell.name} "
                    f"| B:{player.boots.name} | A:{player.armor.name}")
        depth_txt = f"Depth {node_depths[current_node_idx]}"
        node_txt  = f"Node [{current_node().kind}]  {depth_txt}"
        m         = player.metrics
        met_txt   = (f"K(M:{m.melee_kills} / Mg:{m.magic_kills})  "
                     f"Hits(M:{m.melee_hits} / Mg:{m.magic_hits})  "
                     f"Dmg(T:{m.damage_taken:.0f} / D:{m.damage_dealt:.0f})  "
                     f"Deaths:{m.deaths}")

        screen.blit(font.render(node_txt,  True, (230, 230, 230)), (12, 10))
        screen.blit(font.render(hp_txt + "  " + ammo_txt, True, (230, 230, 230)), (12, 32))
        screen.blit(font.render(gear_txt,  True, (230, 230, 230)), (12, 54))
        screen.blit(font.render(met_txt,   True, (200, 200, 200)), (12, 76))

        dda_txt = (f"DDA  hp×{applied['enemy_hp_mult']:.2f}  "
                   f"dmg×{applied['enemy_dmg_mult']:.2f}  "
                   f"spd×{applied['enemy_speed_mult']:.2f}  "
                   f"n={applied['spawn_count']}")
        screen.blit(font.render(dda_txt, True, (150, 180, 200)), (12, 98))

        c = "WASD move | LMB melee | RMB magic | SPACE dash | SHIFT defend | ENTER continue"
        screen.blit(font.render(c, True, (170, 170, 170)), (12, H - 26))

    def draw_center(text):
        surf = big.render(text, True, (240, 240, 240))
        screen.blit(surf, (W/2 - surf.get_width()/2, H/2 - surf.get_height()/2))

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False

                if ev.key == pygame.K_SPACE and state == "arena":
                    player.try_dash()

                if state == "loot":
                    if ev.key in (pygame.K_RIGHT, pygame.K_d):
                        loot_choice_idx = (loot_choice_idx + 1) % len(loot_options)
                    if ev.key in (pygame.K_LEFT, pygame.K_a):
                        loot_choice_idx = (loot_choice_idx - 1) % len(loot_options)

                if ev.key == pygame.K_RETURN:
                    if state == "loot":
                        if loot_options:
                            apply_loot_choice(player, loot_options[loot_choice_idx])
                        state = "map"

                    elif state in ("dead", "win"):
                        append_run(run_id, player, applied)
                        run_id           = f"run_{random.randint(10000, 99999)}"
                        player           = Player(W/2, H/2)
                        node_map         = generate_node_map_graph(18)
                        node_depths      = compute_node_depth(node_map, node_map.start)
                        current_node_idx = node_map.start
                        state            = "arena"  # default; spawn_room may override
                        spawn_room(node_map.nodes[current_node_idx].kind)

            # FIX: map navigation is click-only (no conflicting WASD cursor)
            if ev.type == pygame.MOUSEBUTTONDOWN:
                if state == "arena":
                    if ev.button == 1:
                        player.try_melee(enemies)
                    elif ev.button == 3:
                        player.try_magic(projectiles)

                elif state == "map" and ev.button == 1:
                    mx, my = ev.pos
                    for i, node in enumerate(node_map.nodes):
                        nx_ = arena.left + node.x * scale_x
                        ny_ = arena.top  + node.y * scale_y
                        if (vec_len(mx - nx_, my - ny_) <= 22
                                and i in node_map.edges.get(current_node_idx, set())
                                and not node.cleared):
                            current_node_idx = i
                            spawn_room(node.kind)
                            state = "arena"
                            break

        # ---------------------------------------------------------------
        screen.fill((18, 18, 24))
        pygame.draw.rect(screen, (40, 40, 52),   arena, border_radius=8)
        pygame.draw.rect(screen, (80, 80, 100),  arena, 2, border_radius=8)

        keys = pygame.key.get_pressed()
        mx, my = pygame.mouse.get_pos()
        player.facing_deg = angle_deg(mx - player.x, my - player.y)

        # ---------------------------------------------------------------
        if state == "arena":
            player.update(dt, keys, arena)

            for e in enemies:
                e.update(dt, player, arena)

            for p in projectiles:
                p.update(dt, arena)
            projectiles = [p for p in projectiles if p.alive]

            # Projectile-enemy collision
            for p in projectiles:
                for e in enemies:
                    if not e.alive:
                        continue
                    if vec_len(e.x - p.x, e.y - p.y) <= (e.r + p.r):
                        # FIX: pass source tag for accurate kill attribution
                        e.take_damage(p.dmg, source="magic")
                        player.metrics.magic_hits   += 1
                        player.metrics.damage_dealt += p.dmg
                        p.alive = False
                        break

            # Kill accounting — use last_hit_source, not proximity heuristic
            for e in enemies:
                if e.alive or getattr(e, "_counted", False):
                    continue
                setattr(e, "_counted", True)
                # FIX: attribute kill to the source that dealt the killing blow
                if e.last_hit_source == "melee":
                    player.metrics.melee_kills += 1
                else:
                    player.metrics.magic_kills += 1
                player.heal_on_kill()

            if not player.alive:
                state = "dead"
                append_run(run_id, player, applied)
            elif all_enemies_dead():
                current_node().cleared = True
                if current_node().kind == "boss":
                    state = "win"
                    append_run(run_id, player, applied)
                else:
                    state            = "loot"
                    loot_options     = generate_loot_options(player, 3, loot_bias=applied.get("loot_bias"))
                    loot_choice_idx  = 0

        # ---------------------------------------------------------------
        for e in enemies:
            e.draw(screen)
        for p in projectiles:
            p.draw(screen)
        player.draw(screen)

        # ---------------------------------------------------------------
        if state == "map":
            # Draw edges
            for a, nbrs in node_map.edges.items():
                for b in nbrs:
                    if a < b:
                        ax_ = arena.left + node_map.nodes[a].x * scale_x
                        ay_ = arena.top  + node_map.nodes[a].y * scale_y
                        bx_ = arena.left + node_map.nodes[b].x * scale_x
                        by_ = arena.top  + node_map.nodes[b].y * scale_y
                        pygame.draw.line(screen, (100, 100, 100), (ax_, ay_), (bx_, by_), 2)

            # Draw nodes
            for i, node in enumerate(node_map.nodes):
                nx_ = arena.left + node.x * scale_x
                ny_ = arena.top  + node.y * scale_y
                col = (200, 200, 200)
                if node.kind == "start":  col = (0, 255, 0)
                elif node.kind == "boss": col = (255, 0, 0)
                elif node.kind == "elite":col = (255, 165, 0)
                elif node.kind == "shop": col = (255, 255, 0)
                elif node.kind == "loot": col = (0, 255, 255)
                elif node.kind == "combat":col = (100, 100, 220)

                if node.cleared:
                    col = tuple(c // 3 for c in col)

                # Highlight connected, uncleared neighbors on hover
                mouse_pos = pygame.mouse.get_pos()
                is_neighbor = (i in node_map.edges.get(current_node_idx, set())
                               and not node.cleared)
                is_hovered  = vec_len(mouse_pos[0] - nx_, mouse_pos[1] - ny_) <= 22

                radius = 15
                if i == current_node_idx:
                    pygame.draw.circle(screen, (255, 255, 255), (int(nx_), int(ny_)), 20, 2)
                if is_neighbor and is_hovered:
                    pygame.draw.circle(screen, (255, 255, 255), (int(nx_), int(ny_)), 22, 2)
                    radius = 17

                pygame.draw.circle(screen, col, (int(nx_), int(ny_)), radius)

                # Draw depth number on each node
                d_surf = font.render(str(node_depths[i]), True, (30, 30, 30))
                screen.blit(d_surf, (int(nx_) - d_surf.get_width()//2,
                                     int(ny_) - d_surf.get_height()//2))

            draw_center("Click an adjacent node to move there")

        # ---------------------------------------------------------------
        draw_ui()

        if state == "loot":
            draw_center("Choose a reward  —  LEFT/RIGHT to preview, ENTER to pick")
            if loot_options:
                box_w = 240
                box_h = 110
                gap   = 18
                total_width = len(loot_options) * box_w + (len(loot_options) - 1) * gap
                start_x = W/2 - total_width / 2
                y = H/2 - box_h / 2 + 20
                for idx, (kind, item) in enumerate(loot_options):
                    x    = start_x + idx * (box_w + gap)
                    rect = pygame.Rect(x, y, box_w, box_h)
                    col  = (80, 80, 110) if idx != loot_choice_idx else (120, 120, 180)
                    pygame.draw.rect(screen, col,          rect, border_radius=8)
                    pygame.draw.rect(screen, (200,200,200),rect, 2, border_radius=8)
                    lines = describe_loot_option(kind, item)
                    for line_i, line in enumerate(lines):
                        surf = font.render(line, True, (240, 240, 240))
                        screen.blit(surf, (x + 10, y + 10 + line_i * 22))

        elif state == "dead":
            draw_center("YOU DIED — press ENTER")
        elif state == "win":
            draw_center("RUN COMPLETE — press ENTER")

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
