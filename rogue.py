import math
import os
import csv
import random
from typing import List, Tuple, Optional, Dict

import pygame

from node_map_gen import (
    NodeMap, MapNode, generate_node_map_graph, compute_node_depth, OPPOSITE,
)
import dda_core as core
from dda_core import (
    W, H, ARENA_MARGIN,
    PLAYER_SPEED, DASH_SPEED, DASH_TIME, DASH_CD,
    MELEE_RANGE, MELEE_ARC_DEG, MELEE_CD_BASE,
    PROJECTILE_SPEED, MAGIC_CD_BASE,
    DEFEND_SLOW, DEFEND_DMG_MULT, HEAL_ON_KILL_BASE,
    ENEMY_BASE_HP, ENEMY_BASE_DMG, ENEMY_SPEED, ENEMY_AGGRO_R,
    ENEMY_SPAWN_MIN_DIST,
    ENEMY_DASH_TRIGGER_R, ENEMY_WINDUP_TIME, ENEMY_DASH_SPEED_MULT,
    ENEMY_DASH_TIME, ENEMY_DASH_RECOVER,
    knockback_impulse, apply_knockback_decay,
    Weapon, Spell, Boots, Armor, WEAPONS, SPELLS, BOOTS, ARMORS,
    CombatMetrics, rule_based_dda,
)

FPS = 60
FONT_NAME = None

ENABLE_MODEL = False
MODEL_DIR = "models"

# Doorway geometry: how wide the walkable opening in a wall is, and how far
# past the wall the player has to walk before the room transition fires.
DOOR_GAP_HALF   = 46
DOOR_EXIT_MARGIN = 46
ENTRY_INSET     = 34

# Loot pedestal interaction radius (loot-kind rooms only).
LOOT_INTERACT_R = 46


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

            gen = _LOADED.generate(features)
            if gen:
                try:
                    loot_bias = _LOADED.generate_loot_bias(features)
                    if loot_bias:
                        gen["loot_bias"] = loot_bias
                    else:
                        gen.setdefault("loot_bias", core.loot_bias_for_metrics(player.metrics))
                except Exception:
                    gen.setdefault("loot_bias", core.loot_bias_for_metrics(player.metrics))
                return gen
        except Exception as e:
            print("[MODEL] Failed to load/apply model:", e)

    return rule_based_dda(player.metrics, node_depth)


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
    """
    Enemies chase the player normally, but no longer deal damage just by
    standing in contact. Instead, once close enough they telegraph briefly
    ("windup") and then commit to a fast ram/dash toward where the player
    was standing at that moment. Contact during the dash deals damage once
    and knocks the player back; after the dash there's a short recovery
    before the enemy can chase/ram again. This turns enemies into threats
    the player reacts to, instead of passive walking damage.
    """

    def __init__(self, x, y, hp, dmg, speed, aggro_r):
        self.x, self.y = x, y
        self.hp = hp
        self.max_hp = hp
        self.dmg = dmg
        self.speed = speed
        self.aggro_r = aggro_r
        self.r = 16
        self.alive = True
        self.last_hit_source: str = "melee"

        self.state = "chase"     # chase -> windup -> dashing -> recover -> chase
        self.state_t = 0.0
        self.dash_dir = (0.0, 0.0)
        self.dash_hit_done = False
        self.atk_cd = 0.0        # cooldown before another ram may be triggered

        # Knockback impulse velocity, decays with friction each frame.
        self.kvx = 0.0
        self.kvy = 0.0

    def take_damage(self, amount, source: str = "melee", knock_dir: Optional[Tuple[float, float]] = None):
        self.last_hit_source = source
        self.hp -= amount
        if self.hp <= 0:
            self.alive = False
        if knock_dir is not None and amount > 0:
            speed = knockback_impulse(amount)
            self.kvx += knock_dir[0] * speed
            self.kvy += knock_dir[1] * speed

    def update(self, dt, player, arena_rect):
        if not self.alive:
            return
        self.atk_cd = max(0.0, self.atk_cd - dt)

        dx, dy = (player.x - self.x), (player.y - self.y)
        dist = vec_len(dx, dy)

        if self.state == "chase":
            if dist <= self.aggro_r:
                if dist <= ENEMY_DASH_TRIGGER_R and self.atk_cd <= 0.0:
                    self.state = "windup"
                    self.state_t = 0.0
                    self.dash_dir = norm(dx, dy)
                else:
                    nx, ny = norm(dx, dy)
                    self.x += nx * self.speed * dt
                    self.y += ny * self.speed * dt

        elif self.state == "windup":
            # Stand still and telegraph — the player gets a beat to react.
            self.state_t += dt
            if self.state_t >= ENEMY_WINDUP_TIME:
                self.state = "dashing"
                self.state_t = 0.0
                self.dash_hit_done = False

        elif self.state == "dashing":
            self.state_t += dt
            spd = self.speed * ENEMY_DASH_SPEED_MULT
            self.x += self.dash_dir[0] * spd * dt
            self.y += self.dash_dir[1] * spd * dt

            if not self.dash_hit_done:
                nd = vec_len(player.x - self.x, player.y - self.y)
                if nd <= (self.r + player.r + 4):
                    self.dash_hit_done = True
                    kdir = norm(player.x - self.x, player.y - self.y)
                    player.receive_damage(self.dmg, knock_dir=kdir)

            if self.state_t >= ENEMY_DASH_TIME:
                self.state = "recover"
                self.state_t = 0.0
                self.atk_cd = ENEMY_DASH_RECOVER

        elif self.state == "recover":
            self.state_t += dt
            if self.state_t >= ENEMY_DASH_RECOVER:
                self.state = "chase"

        # Knockback displacement applies on top of whatever state we're in.
        self.x += self.kvx * dt
        self.y += self.kvy * dt
        self.kvx, self.kvy = apply_knockback_decay(self.kvx, self.kvy, dt)

        self.x = clamp(self.x, arena_rect.left + self.r, arena_rect.right - self.r)
        self.y = clamp(self.y, arena_rect.top + self.r, arena_rect.bottom - self.r)

    def draw(self, surf):
        if not self.alive:
            return
        if self.state == "windup":
            col = (255, 225, 90)      # telegraphing — about to ram
        elif self.state == "dashing":
            col = (255, 120, 40)      # mid-ram
        else:
            col = (240, 120, 120)
        pygame.draw.circle(surf, col, (int(self.x), int(self.y)), self.r)
        bar_w = 34
        hp_ratio = max(0.0, self.hp / self.max_hp)
        pygame.draw.rect(surf, (40, 40, 40), (int(self.x - bar_w / 2), int(self.y - 28), bar_w, 6))
        pygame.draw.rect(surf, (80, 220, 80), (int(self.x - bar_w / 2), int(self.y - 28), int(bar_w * hp_ratio), 6))


class Player:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.r = 18
        self.max_hp = 100
        self.hp = 100

        self.weapon = random.choice(WEAPONS)
        self.spell = random.choice(SPELLS)
        self.boots = random.choice(BOOTS)
        self.armor = random.choice(ARMORS)

        self.facing_deg = 0.0
        self.melee_cd = 0.0
        self.magic_cd = 0.0
        self.magic_ammo = self.spell.ammo_max

        self.defending = False
        self.dashing = False
        self.dash_t = 0.0
        self.dash_cd = 0.0
        self.dash_dir = (0.0, 0.0)

        self.alive = True
        self.metrics = CombatMetrics()

        # Knockback impulse velocity, decays with friction each frame.
        self.kvx = 0.0
        self.kvy = 0.0

        # Set by update() when the player has walked through an open,
        # unlocked doorway far enough to trigger a room transition.
        self.exit_dir: Optional[str] = None

    def set_loadout(self, weapon=None, spell=None, boots=None, armor=None):
        if weapon: self.weapon = weapon
        if spell:
            self.spell = spell
            self.magic_ammo = min(self.magic_ammo, self.spell.ammo_max)
        if boots: self.boots = boots
        if armor: self.armor = armor

    def heal_on_kill(self):
        heal = int(HEAL_ON_KILL_BASE * self.weapon.heal_mult) + self.armor.heal_on_kill_bonus
        self.hp = min(self.max_hp, self.hp + heal)

    def receive_damage(self, raw_amount, knock_dir: Optional[Tuple[float, float]] = None):
        if not self.alive:
            return
        amount = max(0.0, raw_amount - self.armor.dmg_absorb)
        if self.defending:
            amount *= DEFEND_DMG_MULT
        self.hp -= amount
        self.metrics.damage_taken += amount
        if knock_dir is not None and amount > 0:
            speed = knockback_impulse(amount)
            self.kvx += knock_dir[0] * speed
            self.kvy += knock_dir[1] * speed
        if self.hp <= 0:
            self.alive = False
            self.metrics.deaths += 1

    def update(self, dt, keys, arena_rect, doors_unlocked=frozenset()):
        self.exit_dir = None
        if not self.alive:
            return

        self.metrics.time_alive += dt
        self.melee_cd = max(0.0, self.melee_cd - dt)
        self.magic_cd = max(0.0, self.magic_cd - dt)
        self.dash_cd = max(0.0, self.dash_cd - dt)

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
                self.dash_t = 0.0
        else:
            speed = PLAYER_SPEED
            if self.defending:
                speed *= DEFEND_SLOW
            self.x += nx * speed * dt
            self.y += ny * speed * dt

        # Knockback displacement (from being hit) layers on top of input movement.
        self.x += self.kvx * dt
        self.y += self.kvy * dt
        self.kvx, self.kvy = apply_knockback_decay(self.kvx, self.kvy, dt)

        # ---- Wall collision, with door-aware pass-through -----------------
        cx, cy = arena_rect.centerx, arena_rect.centery

        if self.x < arena_rect.left + self.r:
            if "W" in doors_unlocked and abs(self.y - cy) <= DOOR_GAP_HALF:
                if self.exit_dir is None and self.x < arena_rect.left - DOOR_EXIT_MARGIN:
                    self.exit_dir = "W"
            else:
                self.x = arena_rect.left + self.r
        elif self.x > arena_rect.right - self.r:
            if "E" in doors_unlocked and abs(self.y - cy) <= DOOR_GAP_HALF:
                if self.exit_dir is None and self.x > arena_rect.right + DOOR_EXIT_MARGIN:
                    self.exit_dir = "E"
            else:
                self.x = arena_rect.right - self.r

        if self.y < arena_rect.top + self.r:
            if "N" in doors_unlocked and abs(self.x - cx) <= DOOR_GAP_HALF:
                if self.exit_dir is None and self.y < arena_rect.top - DOOR_EXIT_MARGIN:
                    self.exit_dir = "N"
            else:
                self.y = arena_rect.top + self.r
        elif self.y > arena_rect.bottom - self.r:
            if "S" in doors_unlocked and abs(self.x - cx) <= DOOR_GAP_HALF:
                if self.exit_dir is None and self.y > arena_rect.bottom + DOOR_EXIT_MARGIN:
                    self.exit_dir = "S"
            else:
                self.y = arena_rect.bottom - self.r

    def try_dash(self):
        if not self.alive:
            return
        if self.dash_cd > 0.0 or self.dashing:
            return
        rad = math.radians(self.facing_deg)
        dx, dy = math.cos(rad), math.sin(rad)
        self.dashing = True
        self.dash_dir = (dx, dy)
        self.dash_cd = DASH_CD

    def try_melee(self, enemies: List[Enemy]):
        if not self.alive or self.melee_cd > 0.0:
            return False
        self.melee_cd = MELEE_CD_BASE * self.weapon.cd_mult
        hit_any = False
        for e in enemies:
            if not e.alive:
                continue
            dx, dy = (e.x - self.x), (e.y - self.y)
            d_dist = vec_len(dx, dy)
            if d_dist > MELEE_RANGE + e.r:
                continue
            ang = angle_deg(dx, dy)
            if abs(angle_diff_deg(ang, self.facing_deg)) <= (MELEE_ARC_DEG / 2):
                knock_dir = norm(dx, dy)  # push the enemy away from the player
                e.take_damage(self.weapon.dmg, source="melee", knock_dir=knock_dir)
                self.metrics.melee_hits += 1
                self.metrics.damage_dealt += self.weapon.dmg
                hit_any = True
        return hit_any

    def try_magic(self, projectiles: List[Projectile]):
        if not self.alive or self.magic_cd > 0.0 or self.magic_ammo <= 0:
            return
        self.magic_cd = MAGIC_CD_BASE * self.spell.cd_mult
        self.magic_ammo -= 1
        rad = math.radians(self.facing_deg)
        vx = math.cos(rad) * PROJECTILE_SPEED
        vy = math.sin(rad) * PROJECTILE_SPEED
        px = self.x + math.cos(rad) * (self.r + 8)
        py = self.y + math.sin(rad) * (self.r + 8)
        projectiles.append(Projectile(px, py, vx, vy, self.spell.dmg))

    def draw(self, surf):
        col = (130, 210, 140) if self.alive else (80, 80, 80)
        pygame.draw.circle(surf, col, (int(self.x), int(self.y)), self.r)
        rad = math.radians(self.facing_deg)
        fx = self.x + math.cos(rad) * (self.r + 12)
        fy = self.y + math.sin(rad) * (self.r + 12)
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
    """
    KINDS = ["weapon", "spell", "armor", "boots"]
    POOLS = {
        "weapon": WEAPONS,
        "spell": SPELLS,
        "armor": ARMORS,
        "boots": BOOTS,
    }

    if loot_bias is not None and len(loot_bias) == 4 and sum(loot_bias) > 0:
        total = sum(loot_bias)
        weights = [w / total for w in loot_bias]
    else:
        weights = [0.25, 0.25, 0.25, 0.25]

    options: List[Tuple[str, object]] = []
    for _ in range(count):
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
                "melee_kills", "magic_kills",
                "melee_hits", "magic_hits",
                "damage_taken", "damage_dealt",
                "deaths", "time_alive",
                "weapon", "spell", "boots", "armor",
                "enemy_hp_mult", "enemy_dmg_mult", "enemy_speed_mult", "spawn_count",
                "heal_on_kill_base",
            ])

def append_run(run_id: str, player: Player, applied: Dict[str, float]):
    """
    Logs exactly one row per finished run.

    FIX: this used to be called twice per run — once when `state` flipped to
    "dead"/"win", and again when ENTER was pressed to acknowledge that
    screen — silently duplicating every human-played run in runs.csv. Now
    there is exactly one call site (the state transition itself, in main());
    the ENTER handler only resets for the next run.
    """
    ensure_log_header()
    m = player.metrics
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            run_id,
            m.melee_kills, m.magic_kills,
            m.melee_hits, m.magic_hits,
            round(m.damage_taken, 3), round(m.damage_dealt, 3),
            m.deaths, round(m.time_alive, 3),
            player.weapon.name, player.spell.name,
            player.boots.name, player.armor.name,
            applied["enemy_hp_mult"], applied["enemy_dmg_mult"],
            applied["enemy_speed_mult"], applied["spawn_count"],
            HEAL_ON_KILL_BASE,
        ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def entry_position(arena: pygame.Rect, came_from_dir: str) -> Tuple[float, float]:
    """
    Where the player appears in a newly-entered room, given the direction
    they exited the *previous* room through. If they left heading East, they
    arrive on the West side of the new room (and so on).
    """
    cx, cy = arena.centerx, arena.centery
    if came_from_dir == "E":
        return arena.left + ENTRY_INSET, cy
    if came_from_dir == "W":
        return arena.right - ENTRY_INSET, cy
    if came_from_dir == "S":
        return cx, arena.top + ENTRY_INSET
    if came_from_dir == "N":
        return cx, arena.bottom - ENTRY_INSET
    return cx, cy


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Roguelike DDA Prototype")
    clock = pygame.time.Clock()
    font = pygame.font.Font(FONT_NAME, 18)
    big = pygame.font.Font(FONT_NAME, 28)

    arena = pygame.Rect(ARENA_MARGIN, ARENA_MARGIN, W - 2 * ARENA_MARGIN, H - 2 * ARENA_MARGIN)

    node_map = generate_node_map_graph(16, seed=None)
    current_node_idx = node_map.start
    node_depths = compute_node_depth(node_map, node_map.start)

    player = Player(*arena.center)
    applied = rule_based_dda(player.metrics, node_depth=0)

    projectiles: List[Projectile] = []
    enemies: List[Enemy] = []

    loot_options: List[Tuple[str, object]] = []
    loot_choice_idx = 0

    def current_node() -> MapNode:
        return node_map.nodes[current_node_idx]

    def spawn_room(idx: int):
        """
        Sets up `idx` as the active room. Unlike the old design, this never
        forces a state change to a menu — the player keeps full movement
        control. Combat rooms simply start with their doors locked (see
        doors_unlocked in the main loop) until every enemy is dead.
        """
        nonlocal enemies, projectiles, applied
        projectiles = []
        enemies = []

        node = node_map.nodes[idx]
        depth = node_depths[idx]
        applied = apply_model_tuning_if_available(player, node_depth=depth)

        # Refill ammo on every room entry so magic is never permanently exhausted.
        player.magic_ammo = player.spell.ammo_max

        if node.kind in ("nothing", "loot"):
            node.cleared = True
            return

        # "enemy" / "boss" — if we're backtracking into an already-cleared
        # combat room, don't respawn a fresh wave.
        if node.cleared:
            return

        n = applied["spawn_count"]
        if node.kind == "boss":
            n = 1

        hp_base = ENEMY_BASE_HP * applied["enemy_hp_mult"]
        dmg_base = ENEMY_BASE_DMG * applied["enemy_dmg_mult"]
        spd_base = ENEMY_SPEED * applied["enemy_speed_mult"]

        for _ in range(n):
            for _attempt in range(50):
                ex = random.randint(arena.left + 80, arena.right - 80)
                ey = random.randint(arena.top + 80, arena.bottom - 80)
                if vec_len(ex - player.x, ey - player.y) >= ENEMY_SPAWN_MIN_DIST:
                    break

            hp, dmg, spd = hp_base, dmg_base, spd_base
            if node.kind == "boss":
                hp *= 4.0
                dmg *= 1.7
                spd *= 0.9
            enemies.append(Enemy(ex, ey, hp=hp, dmg=dmg, speed=spd, aggro_r=ENEMY_AGGRO_R))

    def start_new_run():
        nonlocal node_map, current_node_idx, node_depths, player, state
        node_map = generate_node_map_graph(16, seed=None)
        current_node_idx = node_map.start
        node_depths = compute_node_depth(node_map, node_map.start)
        player = Player(*arena.center)
        state = "arena"
        spawn_room(current_node_idx)

    run_id = f"run_{random.randint(10000, 99999)}"
    ensure_log_header()

    state = "arena"     # "arena" | "loot_ui" | "dead" | "win"
    spawn_room(current_node_idx)

    def draw_ui():
        node = current_node()
        hp_txt = f"HP {int(player.hp)}/{player.max_hp}"
        ammo_txt = f"Ammo {player.magic_ammo}/{player.spell.ammo_max}"
        gear_txt = (f"W:{player.weapon.name} | S:{player.spell.name} "
                    f"| B:{player.boots.name} | A:{player.armor.name}")
        depth_txt = f"Depth {node_depths[current_node_idx]}"
        node_txt = f"Room [{node.kind}]  {depth_txt}"
        m = player.metrics
        met_txt = (f"K(M:{m.melee_kills} / Mg:{m.magic_kills})  "
                   f"Hits(M:{m.melee_hits} / Mg:{m.magic_hits})  "
                   f"Dmg(T:{m.damage_taken:.0f} / D:{m.damage_dealt:.0f})  "
                   f"Deaths:{m.deaths}")

        screen.blit(font.render(node_txt, True, (230, 230, 230)), (12, 10))
        screen.blit(font.render(hp_txt + "  " + ammo_txt, True, (230, 230, 230)), (12, 32))
        screen.blit(font.render(gear_txt, True, (230, 230, 230)), (12, 54))
        screen.blit(font.render(met_txt, True, (200, 200, 200)), (12, 76))

        dda_txt = (f"DDA  hp×{applied['enemy_hp_mult']:.2f}  "
                   f"dmg×{applied['enemy_dmg_mult']:.2f}  "
                   f"spd×{applied['enemy_speed_mult']:.2f}  "
                   f"n={applied['spawn_count']}")
        screen.blit(font.render(dda_txt, True, (150, 180, 200)), (12, 98))

        c = "WASD move | LMB melee | RMB magic | SPACE dash | SHIFT defend | E interact | walk through open doors"
        screen.blit(font.render(c, True, (170, 170, 170)), (12, H - 26))

    def draw_center(text):
        surf = big.render(text, True, (240, 240, 240))
        screen.blit(surf, (W / 2 - surf.get_width() / 2, H / 2 - surf.get_height() / 2))

    def door_state():
        """Returns (doors: {dir: neighbor_idx}, locked: bool) for the current room."""
        node = current_node()
        doors = node_map.doors(current_node_idx)
        enemies_alive = any(e.alive for e in enemies)
        locked = node.kind in ("enemy", "boss") and enemies_alive and not node.cleared
        return doors, locked

    def draw_doors():
        doors, locked = door_state()
        bg = (18, 18, 24)
        color = (170, 60, 60) if locked else (90, 200, 110)
        gh = DOOR_GAP_HALF
        for d in doors:
            if d == "N":
                gx = arena.centerx
                pygame.draw.rect(screen, bg, (gx - gh, arena.top - 3, gh * 2, 8))
                pygame.draw.rect(screen, color, (gx - gh, arena.top - 7, gh * 2, 6), border_radius=3)
            elif d == "S":
                gx = arena.centerx
                pygame.draw.rect(screen, bg, (gx - gh, arena.bottom - 4, gh * 2, 8))
                pygame.draw.rect(screen, color, (gx - gh, arena.bottom + 1, gh * 2, 6), border_radius=3)
            elif d == "W":
                gy = arena.centery
                pygame.draw.rect(screen, bg, (arena.left - 3, gy - gh, 8, gh * 2))
                pygame.draw.rect(screen, color, (arena.left - 7, gy - gh, 6, gh * 2), border_radius=3)
            elif d == "E":
                gy = arena.centery
                pygame.draw.rect(screen, bg, (arena.right - 4, gy - gh, 8, gh * 2))
                pygame.draw.rect(screen, color, (arena.right + 1, gy - gh, 6, gh * 2), border_radius=3)

    def draw_minimap():
        """Small, view-only overview in the corner — no clicking, purely for
        orientation. Travel happens by walking through doors, never here."""
        box = pygame.Rect(W - 168, 8, 160, 130)
        pygame.draw.rect(screen, (26, 26, 34), box, border_radius=6)
        pygame.draw.rect(screen, (70, 70, 86), box, 1, border_radius=6)

        cols = [nd.col for nd in node_map.nodes]
        rows = [nd.row for nd in node_map.nodes]
        cmin, cmax = min(cols), max(cols)
        rmin, rmax = min(rows), max(rows)
        cspan = max(1, cmax - cmin)
        rspan = max(1, rmax - rmin)
        pad = 14

        def pt(nd):
            px = box.left + pad + (nd.col - cmin) / cspan * (box.width - 2 * pad)
            py = box.top + pad + (nd.row - rmin) / rspan * (box.height - 2 * pad)
            return px, py

        for a, nbrs in node_map.edges.items():
            for b in nbrs:
                if a < b:
                    pygame.draw.line(screen, (90, 90, 100), pt(node_map.nodes[a]), pt(node_map.nodes[b]), 1)

        kind_color = {"nothing": (170, 170, 170), "enemy": (170, 90, 90),
                      "loot": (90, 190, 190), "boss": (220, 60, 60)}
        for i, nd in enumerate(node_map.nodes):
            px, py = pt(nd)
            col = kind_color.get(nd.kind, (170, 170, 170))
            if nd.cleared:
                col = tuple(c // 2 for c in col)
            r = 5 if i == current_node_idx else 3
            pygame.draw.circle(screen, col, (int(px), int(py)), r)
            if i == current_node_idx:
                pygame.draw.circle(screen, (255, 255, 255), (int(px), int(py)), r + 2, 1)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    if state == "loot_ui":
                        state = "arena"   # cancel — take nothing, keep the pedestal available
                    else:
                        running = False

                if ev.key == pygame.K_SPACE and state == "arena":
                    player.try_dash()

                if state == "arena" and current_node().kind == "loot" and not current_node().looted:
                    if ev.key == pygame.K_e:
                        dist_to_center = vec_len(player.x - arena.centerx, player.y - arena.centery)
                        if dist_to_center <= LOOT_INTERACT_R:
                            loot_options = generate_loot_options(player, 3, loot_bias=applied.get("loot_bias"))
                            loot_choice_idx = 0
                            state = "loot_ui"

                if state == "loot_ui":
                    if ev.key in (pygame.K_RIGHT, pygame.K_d) and loot_options:
                        loot_choice_idx = (loot_choice_idx + 1) % len(loot_options)
                    if ev.key in (pygame.K_LEFT, pygame.K_a) and loot_options:
                        loot_choice_idx = (loot_choice_idx - 1) % len(loot_options)
                    if ev.key == pygame.K_RETURN and loot_options:
                        apply_loot_choice(player, loot_options[loot_choice_idx])
                        current_node().looted = True
                        state = "arena"

                elif ev.key == pygame.K_RETURN and state in ("dead", "win"):
                    # NOTE: append_run() already fired once, at the moment the
                    # state transitioned to "dead"/"win" below. Do not log again here.
                    run_id = f"run_{random.randint(10000, 99999)}"
                    start_new_run()

            if ev.type == pygame.MOUSEBUTTONDOWN and state == "arena":
                if ev.button == 1:
                    player.try_melee(enemies)
                elif ev.button == 3:
                    player.try_magic(projectiles)

        # ---------------------------------------------------------------
        screen.fill((18, 18, 24))
        pygame.draw.rect(screen, (40, 40, 52), arena, border_radius=8)
        pygame.draw.rect(screen, (80, 80, 100), arena, 2, border_radius=8)

        keys = pygame.key.get_pressed()
        mx, my = pygame.mouse.get_pos()
        player.facing_deg = angle_deg(mx - player.x, my - player.y)

        # ---------------------------------------------------------------
        if state == "arena":
            doors, locked = door_state()
            doors_unlocked = set() if locked else set(doors.keys())

            player.update(dt, keys, arena, doors_unlocked=doors_unlocked)

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
                        knock_dir = norm(p.vx, p.vy)
                        e.take_damage(p.dmg, source="magic", knock_dir=knock_dir)
                        player.metrics.magic_hits += 1
                        player.metrics.damage_dealt += p.dmg
                        p.alive = False
                        break

            # Kill accounting — use last_hit_source, not proximity heuristic
            for e in enemies:
                if e.alive or getattr(e, "_counted", False):
                    continue
                setattr(e, "_counted", True)
                if e.last_hit_source == "melee":
                    player.metrics.melee_kills += 1
                else:
                    player.metrics.magic_kills += 1
                player.heal_on_kill()

            node = current_node()
            enemies_alive = any(e.alive for e in enemies)

            if not player.alive:
                state = "dead"
                append_run(run_id, player, applied)   # single log site (see append_run docstring)

            elif node.kind in ("enemy", "boss") and not enemies_alive and not node.cleared:
                node.cleared = True
                if node.kind == "boss":
                    state = "win"
                    append_run(run_id, player, applied)  # single log site

            elif player.exit_dir is not None:
                direction = player.exit_dir
                next_idx = doors.get(direction)
                if next_idx is not None:
                    node.cleared = True
                    current_node_idx = next_idx
                    spawn_room(current_node_idx)
                    player.x, player.y = entry_position(arena, direction)
                player.exit_dir = None

        # ---------------------------------------------------------------
        for e in enemies:
            e.draw(screen)
        for p in projectiles:
            p.draw(screen)
        player.draw(screen)

        draw_doors()

        # Loot pedestal marker
        node = current_node()
        if node.kind == "loot":
            pcol = (90, 90, 90) if node.looted else (240, 220, 90)
            pygame.draw.circle(screen, pcol, (arena.centerx, arena.centery), 12)
            pygame.draw.circle(screen, (255, 255, 255), (arena.centerx, arena.centery), 12, 2)
            if not node.looted:
                hint = font.render("Walk up and press E — or just walk out to take nothing", True, (230, 220, 160))
                screen.blit(hint, (arena.centerx - hint.get_width() / 2, arena.centery + 24))

        # ---------------------------------------------------------------
        draw_ui()
        draw_minimap()

        if state == "loot_ui":
            draw_center("Choose a reward  —  LEFT/RIGHT to preview, ENTER to pick, ESC to leave empty-handed")
            if loot_options:
                box_w = 240
                box_h = 110
                gap = 18
                total_width = len(loot_options) * box_w + (len(loot_options) - 1) * gap
                start_x = W / 2 - total_width / 2
                y = H / 2 - box_h / 2 + 30
                for idx, (kind, item) in enumerate(loot_options):
                    x = start_x + idx * (box_w + gap)
                    rect = pygame.Rect(x, y, box_w, box_h)
                    col = (80, 80, 110) if idx != loot_choice_idx else (120, 120, 180)
                    pygame.draw.rect(screen, col, rect, border_radius=8)
                    pygame.draw.rect(screen, (200, 200, 200), rect, 2, border_radius=8)
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
