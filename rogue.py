import math
import os
import random
import sys
import uuid
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
    CombatMetrics, rule_based_dda, effective_applied_config,
)
import telemetry

FPS = 60
FONT_NAME = None

ENABLE_MODEL = os.environ.get("ROGUE_ENABLE_MODEL", "0") == "1"
MODEL_DIR = os.environ.get(
    "ROGUE_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_out"),
)

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


# ---------------------------------------------------------------------------
# Model loading -- isolated from cwd via spec_from_file_location, matching
# EXACTLY the API model_runtime.LoadedModels actually exposes
# (predict(cond_vec) -> dict), not an older "generate(features)" shape that
# never matched what model_runtime.py implements.
# ---------------------------------------------------------------------------

_model_runtime = None
_loaded_models = None
_model_load_failed = False


def apply_model_tuning_if_available(cond_vec: List[float], fallback: dict) -> dict:
    """
    Returns a rule_based_dda()-shaped dict (enemy_hp_mult/enemy_dmg_mult/
    enemy_speed_mult/spawn_count/loot_bias), either from the trained model
    (if ENABLE_MODEL and it loads/validates successfully) or `fallback`
    (the heuristic's own output). Any failure to load, a version mismatch,
    or a runtime error falls back SAFELY to the heuristic instead of
    crashing the game -- and is only attempted once per process (a broken
    model doesn't retry every single room).
    """
    global _model_runtime, _loaded_models, _model_load_failed
    if not ENABLE_MODEL or _model_load_failed:
        return fallback
    try:
        if _model_runtime is None:
            import importlib.util
            this_dir = os.path.dirname(os.path.abspath(__file__))
            spec = importlib.util.spec_from_file_location(
                "model_runtime", os.path.join(this_dir, "model_runtime.py"))
            if spec is None or spec.loader is None:
                raise ImportError("Could not load model_runtime.py")
            _model_runtime = importlib.util.module_from_spec(spec)
            sys.modules["model_runtime"] = _model_runtime
            spec.loader.exec_module(_model_runtime)
        if _loaded_models is None:
            _loaded_models = _model_runtime.LoadedModels.load(MODEL_DIR)

        pred = _loaded_models.predict(cond_vec)
        pred["loot_bias"] = fallback.get("loot_bias", [0.25, 0.25, 0.25, 0.25])
        return pred
    except Exception as e:
        print(f"[MODEL] Falling back to heuristic DDA: {e}", file=sys.stderr)
        _model_load_failed = True
        return fallback


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

    def hp_ratio(self) -> float:
        return core.clamp(self.hp / self.max_hp, 0.0, 1.0)

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
    """Thin wrapper over dda_core.generate_loot_options — kept here so the
    rest of rogue.py's call sites don't change, but the actual weighted
    sampling logic is the single copy shared with bot_runner.py."""
    return core.generate_loot_options(loot_bias=loot_bias, count=count)


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
# Per-room telemetry (finding #8): begin_room() / finalize_room() sharing
# the EXACT same telemetry.RoomSample/ROOM_SAMPLE_COLUMNS contract as
# bot_runner.py, so human and synthetic data merge cleanly. Human combat
# logic (above) is NOT shared with the bot's abstract resolver -- only
# this telemetry CONTRACT is shared, as intended.
#
# Room-visit semantics mirror bot_runner's DFS: a telemetry sample is only
# produced the FIRST time a room is entered (node.cleared == False at
# entry). Backtracking into an already-cleared room does not re-spawn
# enemies and does not produce a duplicate/meaningless sample.
# ---------------------------------------------------------------------------

class RoomTelemetrySession:
    def __init__(self, writer: "telemetry.ShardWriter", node_map: NodeMap, run_id: str):
        self.writer = writer
        self.node_map = node_map
        self.run_id = run_id
        self.depths = compute_node_depth(node_map, node_map.start)
        self.total_rooms = len(node_map.nodes)
        self.recent = telemetry.RecentWindow()
        self.cleared_count = 0
        self.room_seq = 0
        self._pending: Optional[dict] = None

    @property
    def pending(self) -> bool:
        return self._pending is not None

    def begin_room(self, room_idx: int, player: Player) -> dict:
        assert self._pending is None, (
            "begin_room() called while a previous room's sample was still "
            "pending -- every room must be finalize_room()'d before the "
            "next one begins."
        )
        node = self.node_map.nodes[room_idx]
        depth = self.depths.get(room_idx, 0)
        progress_norm = core.clamp(self.cleared_count / max(self.total_rooms, 1), 0.0, 1.0)
        is_combat = 1 if node.kind in ("enemy", "boss") else 0

        pre = telemetry.metrics_snapshot(player.metrics)
        pre_hp_ratio = player.hp_ratio()
        recent_totals = self.recent.totals()
        recent_n = self.recent.n_rooms()

        # Official condition contract (finding #3): behavioral components
        # from the recent window; hp_ratio instantaneous; kills/depth/
        # progress cumulative & contextual. Falls back to cumulative-so-far
        # if the window is still empty (first room of the run).
        if recent_n > 0:
            mk, gk = recent_totals["melee_kills"], recent_totals["magic_kills"]
            mh, gh = recent_totals["melee_hits"], recent_totals["magic_hits"]
            dt_, dd_ = recent_totals["damage_taken"], recent_totals["damage_dealt"]
        else:
            mk, gk = pre["melee_kills"], pre["magic_kills"]
            mh, gh = pre["melee_hits"], pre["magic_hits"]
            dt_, dd_ = pre["damage_taken"], pre["damage_dealt"]

        cond_vec = core.feature_vector_from_raw(
            mk, gk, mh, gh, dt_, dd_, pre_hp_ratio, depth, progress_norm,
        )

        applied_raw = rule_based_dda(player.metrics, node_depth=depth, hp_ratio=pre_hp_ratio)
        applied_raw = apply_model_tuning_if_available(cond_vec, applied_raw)
        # Single choke point (finding #5): boss multipliers folded in HERE,
        # so what's returned to spawn_room() to build enemies IS what gets
        # logged -- they can never diverge again.
        applied = effective_applied_config(applied_raw, node.kind)

        self.room_seq += 1
        self._pending = {
            "room_idx": room_idx, "room_kind": node.kind, "depth": depth,
            "progress_norm": progress_norm, "is_combat": is_combat,
            "pre": pre, "pre_hp_ratio": pre_hp_ratio,
            "recent_totals": recent_totals, "recent_n": recent_n,
            "cond_vec": cond_vec, "applied_raw": applied_raw, "applied": applied,
            "room_seq": self.room_seq,
        }
        return applied

    def finalize_room(self, player: Player, room_result: str, loot_taken_kind: str = ""):
        p = self._pending
        assert p is not None, "finalize_room() called without a matching begin_room()"

        post = telemetry.metrics_snapshot(player.metrics)
        delta = telemetry.snapshot_delta(p["pre"], post)
        died_in_room = 1 if room_result == "died" else 0
        self.recent.push(delta)
        if room_result != "died":
            self.cleared_count += 1

        loot_cols = telemetry.loot_bias_to_cols(p["applied_raw"].get("loot_bias"))
        (cond_melee_ratio, cond_magic_ratio, cond_hpk_melee, cond_hpk_magic,
         cond_dmg_ratio, cond_hp_ratio, cond_total_kills_norm,
         cond_node_depth_norm, cond_progress_norm) = p["cond_vec"]

        sample = telemetry.RoomSample(
            run_id=self.run_id, room_seq=p["room_seq"], room_idx=p["room_idx"],
            room_kind=p["room_kind"], node_depth=p["depth"],
            progress_norm=p["progress_norm"], is_combat_room=p["is_combat"],

            pre_melee_kills=p["pre"]["melee_kills"], pre_magic_kills=p["pre"]["magic_kills"],
            pre_melee_hits=p["pre"]["melee_hits"], pre_magic_hits=p["pre"]["magic_hits"],
            pre_damage_taken=p["pre"]["damage_taken"], pre_damage_dealt=p["pre"]["damage_dealt"],
            pre_deaths=p["pre"]["deaths"], pre_time_alive=p["pre"]["time_alive"],
            pre_hp_ratio=p["pre_hp_ratio"],

            recent_melee_kills=p["recent_totals"]["melee_kills"],
            recent_magic_kills=p["recent_totals"]["magic_kills"],
            recent_melee_hits=p["recent_totals"]["melee_hits"],
            recent_magic_hits=p["recent_totals"]["magic_hits"],
            recent_damage_taken=p["recent_totals"]["damage_taken"],
            recent_damage_dealt=p["recent_totals"]["damage_dealt"],
            recent_n_rooms=p["recent_n"],

            cond_melee_ratio=cond_melee_ratio, cond_magic_ratio=cond_magic_ratio,
            cond_hpk_melee=cond_hpk_melee, cond_hpk_magic=cond_hpk_magic,
            cond_dmg_ratio=cond_dmg_ratio, cond_hp_ratio=cond_hp_ratio,
            cond_total_kills_norm=cond_total_kills_norm,
            cond_node_depth_norm=cond_node_depth_norm,
            cond_progress_norm=cond_progress_norm,

            applied_raw_hp_mult=p["applied_raw"]["enemy_hp_mult"],
            applied_raw_dmg_mult=p["applied_raw"]["enemy_dmg_mult"],
            applied_raw_speed_mult=p["applied_raw"]["enemy_speed_mult"],
            applied_raw_spawn_count=p["applied_raw"]["spawn_count"],

            applied_hp_mult=p["applied"]["enemy_hp_mult"],
            applied_dmg_mult=p["applied"]["enemy_dmg_mult"],
            applied_speed_mult=p["applied"]["enemy_speed_mult"],
            applied_spawn_count=p["applied"]["spawn_count"],
            applied_loot_bias_weapon=loot_cols["weapon"],
            applied_loot_bias_spell=loot_cols["spell"],
            applied_loot_bias_armor=loot_cols["armor"],
            applied_loot_bias_boots=loot_cols["boots"],

            room_result=room_result, died_in_room=died_in_room,
            melee_kills_in_room=delta["melee_kills"], magic_kills_in_room=delta["magic_kills"],
            melee_hits_in_room=delta["melee_hits"], magic_hits_in_room=delta["magic_hits"],
            damage_taken_in_room=delta["damage_taken"], damage_dealt_in_room=delta["damage_dealt"],
            time_in_room=post["time_alive"] - p["pre"]["time_alive"],
            loot_taken_kind=loot_taken_kind,
        )
        self.writer.write(sample)
        self._pending = None
        return sample


HUMAN_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "human_sessions")


def new_human_telemetry_session(node_map: NodeMap) -> Tuple["RoomTelemetrySession", "telemetry.ShardWriter", str]:
    os.makedirs(HUMAN_LOG_DIR, exist_ok=True)
    run_id = uuid.uuid4().hex
    shard_path = os.path.join(HUMAN_LOG_DIR, f"human_{run_id}.csv")
    writer = telemetry.open_shard_writer(shard_path)
    session = RoomTelemetrySession(writer, node_map, run_id)
    return session, writer, shard_path


def close_human_telemetry_session(session: "RoomTelemetrySession", writer: "telemetry.ShardWriter",
                                   shard_path: str, player: Optional[Player] = None):
    """Flushes any still-pending sample (e.g. the process was quit mid-room)
    as an honest 'skipped' record rather than losing or duplicating it, then
    closes the shard and writes its manifest."""
    if session.pending and player is not None:
        session.finalize_room(player, "skipped")
    n_written = writer.close()
    manifest_path = shard_path[:-4] + ".manifest.json"
    telemetry.write_manifest(
        manifest_path, n_rows=n_written, seed_range=[0, 0],
        extra={"source": "human", "player_run_id": session.run_id},
    )
    return n_written


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


def main(max_frames: Optional[int] = None):
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
    applied = rule_based_dda(player.metrics, node_depth=0, hp_ratio=1.0)

    session, writer, shard_path = new_human_telemetry_session(node_map)
    last_loot_kind = ""

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

        Telemetry (finding #8): a RoomSample is only opened (begin_room())
        the FIRST time a room is entered -- node.cleared is False. Revisiting
        an already-cleared room while backtracking neither respawns enemies
        nor opens a new pending sample.
        """
        nonlocal enemies, projectiles, applied, last_loot_kind
        projectiles = []
        enemies = []
        last_loot_kind = ""

        node = node_map.nodes[idx]

        if node.cleared:
            # Backtracking into a cleared room: no new sample, no respawn.
            return

        applied = session.begin_room(idx, player)

        # Refill ammo on every (first) room entry so magic is never
        # permanently exhausted.
        player.magic_ammo = player.spell.ammo_max

        if node.kind in ("nothing", "loot"):
            # Finalized when the player actually leaves (see exit_dir
            # handling below) so a loot pick made in this room is captured.
            return

        # "enemy" / "boss" -- applied already has boss multipliers folded
        # in via effective_applied_config() inside session.begin_room(), so
        # no separate boss branch is needed here (finding #5 fix).
        n = applied["spawn_count"]
        hp_base = ENEMY_BASE_HP * applied["enemy_hp_mult"]
        dmg_base = ENEMY_BASE_DMG * applied["enemy_dmg_mult"]
        spd_base = ENEMY_SPEED * applied["enemy_speed_mult"]

        for _ in range(n):
            ex, ey = player.x, player.y
            for _attempt in range(50):
                ex = random.randint(arena.left + 80, arena.right - 80)
                ey = random.randint(arena.top + 80, arena.bottom - 80)
                if vec_len(ex - player.x, ey - player.y) >= ENEMY_SPAWN_MIN_DIST:
                    break
            enemies.append(Enemy(ex, ey, hp=hp_base, dmg=dmg_base, speed=spd_base, aggro_r=ENEMY_AGGRO_R))

    def start_new_run():
        nonlocal node_map, current_node_idx, node_depths, player, state
        nonlocal session, writer, shard_path
        close_human_telemetry_session(session, writer, shard_path, player)
        node_map = generate_node_map_graph(16, seed=None)
        current_node_idx = node_map.start
        node_depths = compute_node_depth(node_map, node_map.start)
        player = Player(*arena.center)
        session, writer, shard_path = new_human_telemetry_session(node_map)
        state = "arena"
        spawn_room(current_node_idx)

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
        """Returns (doors: [dir, ...], locked: bool) for the current room."""
        node = current_node()
        doors = node_map.doors(current_node_idx)  # list of direction strings
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

        cols = [nd.col for nd in node_map.nodes.values()]
        rows = [nd.row for nd in node_map.nodes.values()]
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
            for b in nbrs.values():
                if a < b:
                    pygame.draw.line(screen, (90, 90, 100), pt(node_map.nodes[a]), pt(node_map.nodes[b]), 1)

        kind_color = {"nothing": (170, 170, 170), "enemy": (170, 90, 90),
                      "loot": (90, 190, 190), "boss": (220, 60, 60)}
        for i, nd in node_map.nodes.items():
            px, py = pt(nd)
            col = kind_color.get(nd.kind, (170, 170, 170))
            if nd.cleared:
                col = tuple(c // 2 for c in col)
            r = 5 if i == current_node_idx else 3
            pygame.draw.circle(screen, col, (int(px), int(py)), r)
            if i == current_node_idx:
                pygame.draw.circle(screen, (255, 255, 255), (int(px), int(py)), r + 2, 1)

    running = True
    frame = 0
    while running and (max_frames is None or frame < max_frames):
        frame += 1
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
                        kind, item = loot_options[loot_choice_idx]
                        apply_loot_choice(player, (kind, item))
                        current_node().looted = True
                        last_loot_kind = kind
                        state = "arena"

                elif ev.key == pygame.K_RETURN and state in ("dead", "win"):
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
            doors_unlocked = set() if locked else set(doors)

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
                if session.pending:
                    session.finalize_room(player, "died")

            elif node.kind in ("enemy", "boss") and not enemies_alive and not node.cleared:
                node.cleared = True
                if session.pending:
                    session.finalize_room(player, "cleared")
                if node.kind == "boss":
                    state = "win"

            elif player.exit_dir is not None:
                direction = player.exit_dir
                next_idx = node_map.neighbors(current_node_idx).get(direction)
                if next_idx is not None:
                    if session.pending:
                        # Non-combat room (nothing/loot) being left for the
                        # first time -- finalize now, capturing whatever
                        # loot was (or wasn't) taken.
                        session.finalize_room(player, "cleared", loot_taken_kind=last_loot_kind)
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

    n_written = close_human_telemetry_session(session, writer, shard_path, player)
    pygame.quit()
    return shard_path, n_written


if __name__ == "__main__":
    main()
