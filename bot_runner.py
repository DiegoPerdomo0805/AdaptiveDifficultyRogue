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

cGAN_vs_heuristic changes (schema room_samples_v3.0, dfs_backtrack_v2):

  * The DDA decision is delegated to a controller (controllers.py):
    --controller fixed|heuristic|exploration|cgan, or a seeded per-run
    mixture such as "exploration=0.65,heuristic=0.20,fixed=0.15".
    Controller identity/fallback is logged on every row.
  * Run i uses run_seed = --seed + i. Map, bot archetype, combat RNG and
    controller RNG are independent streams derived from run_seed, so the
    same seed range under two controllers yields equivalent maps/bots, and
    results do not depend on --workers.
  * run_id is deterministic: sha256(run_seed | controller | policy version).
  * progress_norm counts rooms finished BEFORE the current one (as rogue.py).
  * post_hp_ratio is logged so observed outcomes can be built.
  * The archetype name is logged as player_profile (metadata only; never a
    model input).

DFS backtracking (validated correct pre-review, preserved unchanged):
visited-set + explicit stack, descend into unvisited neighbors, backtrack
to the parent when a branch is exhausted, hard step cap as an escape
hatch, explicit termination reasons.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import argparse
import hashlib
import json
import os
import random
import sys
import time

import balance
import controllers as ctl
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


def make_run_id(run_seed: int, controller_type: str) -> str:
    key = f"{int(run_seed)}|{controller_type}|{telemetry.BOT_POLICY_VERSION}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def archetype_for_seed(run_seed: int) -> Archetype:
    rng = random.Random(ctl.derive_seed(run_seed, "archetype"))
    return ARCHETYPES[rng.randrange(len(ARCHETYPES))]


def simulate_run(
    run_seed: int, controller: "ctl.Controller", invincible: bool = False,
    exp_version: str = "",
) -> Tuple[BotPlayer, List[telemetry.RoomSample], str, bool, str]:
    """
    Plays one full run via real DFS backtracking under `controller`. Returns
    (player, samples, termination_reason, boss_room_reached, archetype_name).

    termination_reason in {"victory", "death", "exhausted_no_boss",
    "max_steps_exceeded"}.
    """
    arch = archetype_for_seed(run_seed)
    rng = random.Random(ctl.derive_seed(run_seed, "combat"))
    node_map = nmg.generate_node_map_graph(seed=ctl.derive_seed(run_seed, "map"), **{
        k: v for k, v in BOT_RUNNER_VERSION_EXTRA.items()
        if k in ("node_count", "cols", "rows", "extra_loops", "enemy_weight", "loot_weight")
    })
    total_rooms = len(node_map.nodes)
    depths = nmg.compute_node_depth(node_map, node_map.start)
    controller.begin_run(run_seed)

    player = BotPlayer()
    recent = telemetry.RecentWindow()
    run_id = make_run_id(run_seed, controller.controller_type)
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
            # progress = rooms finished before this one (incl. start) / total
            progress_norm = core.clamp(len(cleared) / max(total_rooms, 1), 0.0, 1.0)
            cleared.add(current)
            room_seq += 1
            depth = depths.get(current, 0)

            pre = telemetry.metrics_snapshot(player.metrics)
            pre_hp_ratio = player.hp_ratio()
            cond_vec, recent_totals, recent_n = telemetry.build_condition(
                pre, recent, pre_hp_ratio, depth, progress_norm)

            decision = controller.decide(ctl.DecisionContext(
                cond_vec=cond_vec, metrics=player.metrics, node_depth=depth,
                hp_ratio=pre_hp_ratio, room_kind=node.kind))
            applied_raw = decision.action
            applied = core.effective_applied_config(applied_raw, node.kind)

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

            samples.append(telemetry.build_room_sample(
                run_id=run_id, run_seed=run_seed, player_profile=arch.name,
                room_seq=room_seq, room_idx=current, room_kind=node.kind,
                node_depth=depth, progress_norm=progress_norm,
                pre=pre, pre_hp_ratio=pre_hp_ratio,
                recent_totals=recent_totals, recent_n=recent_n, cond_vec=cond_vec,
                controller_meta=decision.meta(), applied_raw=applied_raw, applied=applied,
                room_result=room_result, delta=delta, time_in_room=delta["time_in_room"],
                post_hp_ratio=player.hp_ratio(), loot_taken_kind=loot_taken_kind,
                experiment_config_version=exp_version,
            ))

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

    return player, samples, termination_reason, boss_room_reached, arch.name


# ---------------------------------------------------------------------------
# Controller specification: "exploration" or "exploration=0.65,heuristic=0.2"
# ---------------------------------------------------------------------------

def parse_controller_spec(spec: str) -> Dict[str, float]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) == 1 and "=" not in parts[0]:
        mix = {parts[0]: 1.0}
    else:
        mix = {}
        for p in parts:
            if "=" not in p:
                raise ValueError(f"bad controller mixture element {p!r} (want name=weight)")
            k, v = p.split("=", 1)
            mix[k.strip()] = float(v)
    for k, w in mix.items():
        if k not in telemetry.CONTROLLER_TYPES:
            raise ValueError(f"unknown controller {k!r}; choose from {telemetry.CONTROLLER_TYPES}")
        if w < 0:
            raise ValueError(f"negative weight for controller {k!r}")
    total = sum(mix.values())
    if total <= 0:
        raise ValueError("controller mixture weights must sum to > 0")
    return {k: v / total for k, v in mix.items()}


def controller_for_run(run_seed: int, mix: Dict[str, float]) -> str:
    if len(mix) == 1:
        return next(iter(mix))
    r = random.Random(ctl.derive_seed(run_seed, "controller_mix")).random()
    acc = 0.0
    for k in sorted(mix):
        acc += mix[k]
        if r < acc:
            return k
    return sorted(mix)[-1]


# ---------------------------------------------------------------------------
# Batch worker + CLI
# ---------------------------------------------------------------------------

def _run_batch(args) -> Tuple[str, int, dict]:
    worker_id, run_seeds, mix, model_dir, invincible, shard_dir = args
    exp_cfg = balance.load_experiment_config()
    controllers = {name: ctl.make_controller(name, model_dir=model_dir, cfg=exp_cfg) for name in mix}
    shard_path = os.path.join(shard_dir, f"shard_{worker_id:04d}.csv")
    writer = telemetry.open_shard_writer(shard_path)

    stats = {"runs": 0, "boss_reached": 0, "termination_reasons": {},
             "controller_runs": {}, "fallback_rows": 0, "fallback_reasons": {},
             "controllers": {k: c.describe() for k, c in controllers.items()}}
    for run_seed in run_seeds:
        name = controller_for_run(run_seed, mix)
        player, samples, reason, boss_reached, _ = simulate_run(
            run_seed, controllers[name], invincible=invincible,
            exp_version=exp_cfg["experiment_config_version"])
        for s in samples:
            writer.write(s)
            if s.fallback_used:
                stats["fallback_rows"] += 1
                stats["fallback_reasons"][s.fallback_reason] = \
                    stats["fallback_reasons"].get(s.fallback_reason, 0) + 1
        stats["runs"] += 1
        stats["controller_runs"][name] = stats["controller_runs"].get(name, 0) + 1
        if boss_reached:
            stats["boss_reached"] += 1
        stats["termination_reasons"][reason] = stats["termination_reasons"].get(reason, 0) + 1

    n_written = writer.close()
    return shard_path, n_written, stats


def _merge_stats(all_stats: List[dict]) -> dict:
    out = {"runs": 0, "boss_reached": 0, "termination_reasons": {}, "controller_runs": {},
           "fallback_rows": 0, "fallback_reasons": {}}
    for s in all_stats:
        out["runs"] += s["runs"]
        out["boss_reached"] += s["boss_reached"]
        out["fallback_rows"] += s["fallback_rows"]
        for key in ("termination_reasons", "controller_runs", "fallback_reasons"):
            for k, v in s[key].items():
                out[key][k] = out[key].get(k, 0) + v
    return out


def _print_validation_report(stats: dict):
    total_runs = stats["runs"]
    print(f"runs={total_runs} boss_reached={stats['boss_reached']} "
          f"({100.0 * stats['boss_reached'] / max(total_runs, 1):.1f}%)")
    for k, v in sorted(stats["termination_reasons"].items()):
        print(f"  {k}: {v}")
    print("controller runs:", dict(sorted(stats["controller_runs"].items())))
    print(f"fallback rows: {stats['fallback_rows']}", dict(stats["fallback_reasons"]) or "")


def policy_extra() -> dict:
    """Semantically relevant bot constants folded into the manifest/config_hash."""
    extra = dict(BOT_RUNNER_VERSION_EXTRA)
    extra["archetypes"] = [a.__dict__ for a in ARCHETYPES]
    extra["source"] = "bot"
    return extra


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Headless bot data generator (authoritative). Emits one RoomSample per room.")
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234,
                    help="first run_seed; run i uses run_seed = seed + i")
    ap.add_argument("--controller", type=str, default="heuristic",
                    help="fixed|heuristic|exploration|cgan, or a per-run mixture such as "
                         "'exploration=0.65,heuristic=0.20,fixed=0.15'")
    ap.add_argument("--model-dir", type=str, default=None,
                    help="cGAN model directory (required when the cgan controller is used)")
    ap.add_argument("--out", type=str, default="room_samples_v3.csv")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--invincible", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--keep-shards", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing --out")
    args = ap.parse_args(argv)

    mix = parse_controller_spec(args.controller)
    if "cgan" in mix and not args.model_dir:
        ap.error("--model-dir is required when the cgan controller is used")
    if os.path.exists(args.out) and not args.overwrite:
        ap.error(f"{args.out} already exists (pass --overwrite to replace it)")

    shard_dir = args.out + ".shards"
    os.makedirs(shard_dir, exist_ok=True)
    t0 = time.time()

    seeds = [args.seed + i for i in range(args.runs)]
    n_workers = max(1, min(args.workers, len(seeds)))
    chunk = (len(seeds) + n_workers - 1) // n_workers
    tasks = [(w, seeds[w * chunk:(w + 1) * chunk], mix, args.model_dir, args.invincible, shard_dir)
             for w in range(n_workers) if seeds[w * chunk:(w + 1) * chunk]]

    if len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
            results = list(ex.map(_run_batch, tasks))
    else:
        results = [_run_batch(t) for t in tasks]

    shard_paths = [r[0] for r in results]
    total_written = telemetry.merge_shards(shard_paths, args.out)
    stats = _merge_stats([r[2] for r in results])

    manifest_path = os.path.splitext(args.out)[0] + ".manifest.json"
    extra = policy_extra()
    manifest = telemetry.write_manifest(
        manifest_path, n_rows=total_written,
        seed_range=[args.seed, args.seed + args.runs - 1],
        extra=extra, controller_counts=stats["controller_runs"],
    )
    # Controller details are recorded (not hashed) so that datasets from
    # different controllers under the same code/config can be merged.
    manifest["controller_spec"] = mix
    manifest["controllers"] = results[0][2]["controllers"]
    manifest["run_stats"] = stats
    manifest["command"] = " ".join([os.path.basename(sys.argv[0])] + (argv if argv is not None else sys.argv[1:]))
    manifest["elapsed_sec"] = round(time.time() - t0, 3)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if not args.keep_shards:
        for sp in shard_paths:
            os.remove(sp)
        try:
            os.rmdir(shard_dir)
        except OSError:
            pass

    print(f"wrote {total_written} rows -> {args.out}  ({time.time() - t0:.1f}s)")
    print(f"manifest -> {manifest_path}")

    _print_validation_report(stats)

    if args.validate:
        report = telemetry.validate_dataset(args.out, manifest_path, expected_extra_subset=extra)
        print("validate_dataset:", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
