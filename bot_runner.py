"""
bot_runner.py
=============
Headless bot that plays full runs through a generated NodeMap using real
DFS backtracking, simulating combat abstractly (no pygame), and emitting
one RoomSample per room via telemetry.py.

Post-review fixes applied here:

  * finding #3 (recent window computed but never consumed): cond_vec's
    behavioral components (melee/magic ratio, hits-per-kill, dmg_ratio)
    are now built from the RECENT WINDOW totals, not lifetime-cumulative
    `pre` metrics. hp_ratio stays instantaneous (current hp / max hp).
    total_kills_norm/node_depth_norm/progress_norm stay CUMULATIVE/
    contextual, per the review's own recommendation ("utilizar la
    ventana reciente para comportamiento y agregar por separado
    variables contextuales acumuladas como profundidad, progreso y
    encuentros completados"). This is the "official condition"
    definition finding #1's recommended fix order asked to nail down
    first.
  * finding #5 (boss config logged != simulated): every room now calls
    dda_core.effective_applied_config(applied_raw, room_kind) to get the
    EFFECTIVE config, which is BOTH what's used to simulate combat AND
    what's written to applied_*; applied_raw_* preserves the pre-boss-
    multiplier DDA output separately so both are inspectable.
  * finding #4 (combat vs. non-combat rows): every room still gets a
    RoomSample (so context/progression data isn't lost), but
    is_combat_room is now 1 only for room_kind in ("enemy", "boss"),
    letting a consumer cleanly filter.
  * finding #2 (dead death_rate dimension): condition vector no longer
    has a death_rate slot at all -- see dda_core.py's hp_ratio-based
    replacement, threaded through here via `player.hp / player.max_hp`.

DFS backtracking (validated correct pre-review, preserved unchanged):
visited-set + explicit stack, descend into unvisited neighbors, backtrack
to the parent when a branch is exhausted, hard step cap as an escape
hatch, explicit termination reasons.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import argparse
import csv
import glob
import os
import random
import sys
import time
import uuid

import dda_core as core
import node_map_gen as nmg
import telemetry

MAX_STEPS_PER_RUN = 300

BOT_RUNNER_VERSION_EXTRA = {
    "MAX_STEPS_PER_RUN": MAX_STEPS_PER_RUN,
    "RECENT_WINDOW_ROOMS": telemetry.RECENT_WINDOW_ROOMS,
    "node_count": 16, "cols": 6, "rows": 6, "extra_loops": 2,
    "enemy_weight": 0.55, "loot_weight": 0.22,
}


# ---------------------------------------------------------------------------
# Archetypes
# ---------------------------------------------------------------------------

@dataclass
class Archetype:
    name: str
    preferred_range: str      # "melee" | "magic" | "mixed"
    p_melee: float
    p_magic: float
    p_defend: float
    p_dash: float
    loot_bias: List[float]
    p_take_loot: float = 0.85


ARCHETYPES = [
    Archetype("melee_aggressive", "melee", 0.75, 0.10, 0.05, 0.10,
              core.LOOT_BIAS_BERSERKER, p_take_loot=0.90),
    Archetype("magic_sniper", "magic", 0.10, 0.75, 0.10, 0.05,
              core.LOOT_BIAS_SNIPER, p_take_loot=0.80),
    Archetype("hybrid", "mixed", 0.40, 0.40, 0.10, 0.10,
              core.LOOT_BIAS_KNIGHT, p_take_loot=0.75),
    Archetype("tank", "melee", 0.55, 0.10, 0.30, 0.05,
              core.LOOT_BIAS_KNIGHT, p_take_loot=0.70),
    Archetype("reckless", "mixed", 0.45, 0.35, 0.02, 0.18,
              core.LOOT_BIAS_BERSERKER, p_take_loot=0.95),
]

PLAYER_MAX_HP = 100.0


@dataclass
class BotPlayer:
    max_hp: float = PLAYER_MAX_HP
    hp: float = PLAYER_MAX_HP
    metrics: core.CombatMetrics = field(default_factory=core.CombatMetrics)

    def hp_ratio(self) -> float:
        return core.clamp(self.hp / self.max_hp, 0.0, 1.0)

    def receive_damage(self, dmg: float):
        self.hp -= dmg
        self.metrics.damage_taken += dmg
        if self.hp <= 0:
            self.hp = 0
            self.metrics.deaths += 1

    def heal_on_kill(self, amount: float):
        self.hp = min(self.max_hp, self.hp + amount)


def _simulate_combat_room(
    player: BotPlayer, arch: Archetype, cfg: dict, rng: random.Random,
) -> Tuple[bool, dict]:
    """
    Abstract combat resolution (no pygame): rolls a sequence of exchanges
    based on archetype action-selection probabilities and the room's
    EFFECTIVE difficulty config, updating player.metrics and player.hp in
    place. Returns (survived: bool, delta: dict of this room's raw deltas).
    """
    pre = telemetry.metrics_snapshot(player.metrics)
    t0 = player.metrics.time_alive

    enemies_left = cfg["spawn_count"]
    enemy_hp = core.ENEMY_BASE_HP * cfg["enemy_hp_mult"]
    enemy_dmg = core.ENEMY_BASE_DMG * cfg["enemy_dmg_mult"]
    enemy_speed_factor = cfg["enemy_speed_mult"]

    max_exchanges = 400
    exchanges = 0
    while enemies_left > 0 and player.hp > 0 and exchanges < max_exchanges:
        exchanges += 1
        player.metrics.time_alive += 0.35

        roll = rng.random()
        use_magic = roll < arch.p_magic
        use_defend = (not use_magic) and roll < arch.p_magic + arch.p_defend

        hit_chance = 0.78 if not use_defend else 0.65
        if rng.random() < hit_chance:
            if use_magic:
                dmg = core.SPELLS[0].dmg * rng.uniform(0.85, 1.15)
                player.metrics.magic_hits += 1
            else:
                dmg = core.WEAPONS[0].dmg * rng.uniform(0.85, 1.15)
                player.metrics.melee_hits += 1
            player.metrics.damage_dealt += dmg
            enemy_hp -= dmg
            if enemy_hp <= 0:
                enemies_left -= 1
                enemy_hp = core.ENEMY_BASE_HP * cfg["enemy_hp_mult"]
                if use_magic:
                    player.metrics.magic_kills += 1
                else:
                    player.metrics.melee_kills += 1
                player.heal_on_kill(core.HEAL_ON_KILL_BASE)

        # Only ONE enemy is ever actively engaging at a time (the rest are
        # queued), so incoming chance does not scale with enemies_left.
        incoming_chance = 0.20 * enemy_speed_factor
        if enemies_left > 0 and rng.random() < core.clamp(incoming_chance, 0.03, 0.85):
            dmg_mult = core.DEFEND_DMG_MULT if use_defend else 1.0
            dmg_in = enemy_dmg * dmg_mult * rng.uniform(0.8, 1.2)
            player.receive_damage(dmg_in)
            if player.hp <= 0:
                break

    post = telemetry.metrics_snapshot(player.metrics)
    delta = telemetry.snapshot_delta(pre, post)
    delta["time_in_room"] = player.metrics.time_alive - t0
    survived = player.hp > 0
    return survived, delta


def _pick_from_offered(arch: Archetype, options: List[tuple], rng: random.Random) -> Optional[int]:
    if rng.random() > arch.p_take_loot:
        return None
    return rng.randrange(len(options))


def simulate_run(
    arch: Archetype, master_rng: random.Random, invincible: bool = False,
) -> Tuple[BotPlayer, List[telemetry.RoomSample], str, bool]:
    """
    Plays one full run via real DFS backtracking. Returns
    (player, samples, termination_reason, boss_room_reached).

    termination_reason in {"victory", "death", "exhausted_no_boss",
    "max_steps_exceeded"}.
    """
    seed = master_rng.randrange(2**31)
    rng = random.Random(seed)
    node_map = nmg.generate_node_map_graph(seed=seed, **{
        k: v for k, v in BOT_RUNNER_VERSION_EXTRA.items()
        if k in ("node_count", "cols", "rows", "extra_loops", "enemy_weight", "loot_weight")
    })
    total_rooms = len(node_map.nodes)
    depths = nmg.compute_node_depth(node_map, node_map.start)

    player = BotPlayer()
    recent = telemetry.RecentWindow()
    run_id = uuid.uuid4().hex
    samples: List[telemetry.RoomSample] = []

    visited = {node_map.start}
    cleared = {node_map.start}
    stack = [node_map.start]
    room_seq = 0
    steps = 0
    boss_room_reached = False
    termination_reason = "exhausted_no_boss"

    while stack:
        steps += 1
        if steps > MAX_STEPS_PER_RUN:
            termination_reason = "max_steps_exceeded"
            break

        current = stack[-1]
        node = node_map.nodes[current]

        if current not in cleared:
            cleared.add(current)
            room_seq += 1
            depth = depths.get(current, 0)
            progress_norm = core.clamp(len(cleared) / max(total_rooms, 1), 0.0, 1.0)
            is_combat = 1 if node.kind in ("enemy", "boss") else 0

            pre = telemetry.metrics_snapshot(player.metrics)
            pre_hp_ratio = player.hp_ratio()
            recent_totals = recent.totals()
            recent_n = recent.n_rooms()

            # --- Official condition vector (finding #3 decision) ---------
            # Behavioral components from the RECENT WINDOW; hp_ratio is
            # instantaneous; total_kills/node_depth/progress are cumulative
            # contextual features. Falls back to cumulative-so-far if the
            # window is still empty (first room of the run).
            if recent_n > 0:
                mk, gk = recent_totals["melee_kills"], recent_totals["magic_kills"]
                mh, gh = recent_totals["melee_hits"], recent_totals["magic_hits"]
                dt, dd = recent_totals["damage_taken"], recent_totals["damage_dealt"]
            else:
                mk, gk = pre["melee_kills"], pre["magic_kills"]
                mh, gh = pre["melee_hits"], pre["magic_hits"]
                dt, dd = pre["damage_taken"], pre["damage_dealt"]

            cond_vec = core.feature_vector_from_raw(
                mk, gk, mh, gh, dt, dd,
                pre_hp_ratio, depth, progress_norm,
            )
            (cond_melee_ratio, cond_magic_ratio, cond_hpk_melee, cond_hpk_magic,
             cond_dmg_ratio, cond_hp_ratio, cond_total_kills_norm,
             cond_node_depth_norm, cond_progress_norm) = cond_vec

            applied_raw = core.rule_based_dda(
                player.metrics, node_depth=depth, hp_ratio=pre_hp_ratio,
            )
            applied = core.effective_applied_config(applied_raw, node.kind)
            loot_cols = telemetry.loot_bias_to_cols(applied_raw.get("loot_bias"))

            room_result = "skipped"
            died_in_room = 0
            delta = {"melee_kills": 0, "magic_kills": 0, "melee_hits": 0,
                     "magic_hits": 0, "damage_taken": 0.0, "damage_dealt": 0.0,
                     "time_in_room": 0.0}
            loot_taken_kind = ""

            if node.kind in ("enemy", "boss"):
                survived, delta = _simulate_combat_room(player, arch, applied, rng)
                if invincible and player.hp <= 0:
                    player.hp = player.max_hp
                    player.metrics.deaths = max(0, player.metrics.deaths - 1)
                    survived = True
                room_result = "cleared" if survived else "died"
                died_in_room = 0 if survived else 1
                if not survived:
                    termination_reason = "death"
            elif node.kind == "loot":
                options = core.generate_loot_options(
                    applied_raw.get("loot_bias"), count=3, rng=rng,
                )
                pick = _pick_from_offered(arch, options, rng)
                if pick is not None:
                    loot_taken_kind = options[pick][0]
                room_result = "cleared"
            else:
                room_result = "cleared"

            recent.push(delta)

            sample = telemetry.RoomSample(
                run_id=run_id, room_seq=room_seq, room_idx=current,
                room_kind=node.kind, node_depth=depth,
                progress_norm=progress_norm, is_combat_room=is_combat,

                pre_melee_kills=pre["melee_kills"], pre_magic_kills=pre["magic_kills"],
                pre_melee_hits=pre["melee_hits"], pre_magic_hits=pre["magic_hits"],
                pre_damage_taken=pre["damage_taken"], pre_damage_dealt=pre["damage_dealt"],
                pre_deaths=pre["deaths"], pre_time_alive=pre["time_alive"],
                pre_hp_ratio=pre_hp_ratio,

                recent_melee_kills=recent_totals["melee_kills"],
                recent_magic_kills=recent_totals["magic_kills"],
                recent_melee_hits=recent_totals["melee_hits"],
                recent_magic_hits=recent_totals["magic_hits"],
                recent_damage_taken=recent_totals["damage_taken"],
                recent_damage_dealt=recent_totals["damage_dealt"],
                recent_n_rooms=recent_n,

                cond_melee_ratio=cond_melee_ratio, cond_magic_ratio=cond_magic_ratio,
                cond_hpk_melee=cond_hpk_melee, cond_hpk_magic=cond_hpk_magic,
                cond_dmg_ratio=cond_dmg_ratio, cond_hp_ratio=cond_hp_ratio,
                cond_total_kills_norm=cond_total_kills_norm,
                cond_node_depth_norm=cond_node_depth_norm,
                cond_progress_norm=cond_progress_norm,

                applied_raw_hp_mult=applied_raw["enemy_hp_mult"],
                applied_raw_dmg_mult=applied_raw["enemy_dmg_mult"],
                applied_raw_speed_mult=applied_raw["enemy_speed_mult"],
                applied_raw_spawn_count=applied_raw["spawn_count"],

                applied_hp_mult=applied["enemy_hp_mult"],
                applied_dmg_mult=applied["enemy_dmg_mult"],
                applied_speed_mult=applied["enemy_speed_mult"],
                applied_spawn_count=applied["spawn_count"],
                applied_loot_bias_weapon=loot_cols["weapon"],
                applied_loot_bias_spell=loot_cols["spell"],
                applied_loot_bias_armor=loot_cols["armor"],
                applied_loot_bias_boots=loot_cols["boots"],

                room_result=room_result, died_in_room=died_in_room,
                melee_kills_in_room=delta["melee_kills"],
                magic_kills_in_room=delta["magic_kills"],
                melee_hits_in_room=delta["melee_hits"],
                magic_hits_in_room=delta["magic_hits"],
                damage_taken_in_room=delta["damage_taken"],
                damage_dealt_in_room=delta["damage_dealt"],
                time_in_room=delta["time_in_room"],
                loot_taken_kind=loot_taken_kind,
            )
            samples.append(sample)

            if current == node_map.boss and room_result == "cleared":
                boss_room_reached = True
                termination_reason = "victory"
                break
            if died_in_room:
                break

        # advance: push first unvisited neighbor, else backtrack
        advanced = False
        for d, nb in node_map.neighbors(current).items():
            if nb not in visited:
                visited.add(nb)
                stack.append(nb)
                advanced = True
                break
        if not advanced:
            stack.pop()

    return player, samples, termination_reason, boss_room_reached


# ---------------------------------------------------------------------------
# Batch worker + CLI
# ---------------------------------------------------------------------------

def _run_archetype_batch(args) -> Tuple[str, int, dict]:
    worker_id, n_runs, seed, invincible, shard_dir = args
    master_rng = random.Random(seed)
    shard_path = os.path.join(shard_dir, f"shard_{worker_id}.csv")
    writer = telemetry.open_shard_writer(shard_path)

    stats = {"runs": 0, "boss_reached": 0, "termination_reasons": {}}
    for _ in range(n_runs):
        arch = ARCHETYPES[master_rng.randrange(len(ARCHETYPES))]
        player, samples, reason, boss_reached = simulate_run(arch, master_rng, invincible=invincible)
        for s in samples:
            writer.write(s)
        stats["runs"] += 1
        if boss_reached:
            stats["boss_reached"] += 1
        stats["termination_reasons"][reason] = stats["termination_reasons"].get(reason, 0) + 1

    n_written = writer.close()
    return shard_path, n_written, stats


def _print_validation_report(all_stats: List[dict]):
    total_runs = sum(s["runs"] for s in all_stats)
    total_boss = sum(s["boss_reached"] for s in all_stats)
    reasons = {}
    for s in all_stats:
        for k, v in s["termination_reasons"].items():
            reasons[k] = reasons.get(k, 0) + v
    print(f"runs={total_runs} boss_reached={total_boss} "
          f"({100.0*total_boss/max(total_runs,1):.1f}%)")
    for k, v in sorted(reasons.items()):
        print(f"  {k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=str, default="room_samples_v2.csv")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--invincible", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--keep-shards", action="store_true")
    args = ap.parse_args()

    shard_dir = args.out + ".shards"
    os.makedirs(shard_dir, exist_ok=True)

    per_worker = max(1, args.runs // args.workers)
    tasks = []
    for w in range(args.workers):
        n = per_worker if w < args.workers - 1 else (args.runs - per_worker * (args.workers - 1))
        tasks.append((w, n, args.seed + w * 100003, args.invincible, shard_dir))

    results = []
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_run_archetype_batch, tasks))
    else:
        results = [_run_archetype_batch(t) for t in tasks]

    shard_paths = [r[0] for r in results]
    total_written = telemetry.merge_shards(shard_paths, args.out)

    manifest_path = os.path.splitext(args.out)[0] + ".manifest.json"
    extra = dict(BOT_RUNNER_VERSION_EXTRA)
    extra["archetypes"] = [a.name for a in ARCHETYPES]
    telemetry.write_manifest(
        manifest_path, n_rows=total_written,
        seed_range=[args.seed, args.seed + args.workers * 100003],
        extra=extra,
    )

    if not args.keep_shards:
        for sp in shard_paths:
            os.remove(sp)
        try:
            os.rmdir(shard_dir)
        except OSError:
            pass

    print(f"wrote {total_written} rows -> {args.out}")
    print(f"manifest -> {manifest_path}")

    _print_validation_report([r[2] for r in results])

    if args.validate:
        report = telemetry.validate_dataset(args.out, manifest_path, extra=extra)
        print("validate_dataset:", report)


if __name__ == "__main__":
    main()
