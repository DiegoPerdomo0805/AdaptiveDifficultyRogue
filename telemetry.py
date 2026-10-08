"""
telemetry.py
============
Per-room telemetry schema (v2), dataset versioning, manifest, sharded
concurrent-safe writing, and dataset validation.

Post-review fixes applied in this version:

  * COND_DIM reconciled to 9 (was 6) to match dda_core.py's expanded
    condition-vector contract (finding #1/#2).
  * ROOM_SAMPLE_COLUMNS gained `is_combat_room` (finding #4: lets a
    consumer cleanly filter combat rows from context/loot rows without
    guessing from room_kind string values) and split `applied_raw_*`
    (the raw DDA/model output) from `applied_*` (the EFFECTIVE,
    post-room-kind-modifier configuration that was actually simulated —
    finding #5). recent_* columns are unchanged in shape but are now
    actually consumed by bot_runner.py's cond_* columns (finding #3).
  * write_manifest() now actually threads its `extra` dict into
    dda_core.config_hash(extra=...) -- previously `extra` was written
    into the manifest JSON but never fed into the hash itself, so two
    datasets with different bot policies / room weights could share a
    hash. This module's own config_hash() wrapper is now a thin
    passthrough to dda_core.config_hash so there is exactly one hash
    implementation (previously there were two, which could drift).
  * validate_dataset() is substantially strengthened (finding #6): it
    now recomputes the current code's version strings/config_hash and
    requires them to match the manifest exactly (no more silently
    accepting a tampered/stale manifest), verifies the actual row count
    against the declared n_rows, and checks every row has exactly
    len(ROOM_SAMPLE_COLUMNS) fields (catches malformed/truncated rows).
"""

from dataclasses import dataclass, field, fields as dc_fields
from typing import Dict, List, Optional, Tuple
import csv
import json
import os
import subprocess
import time

import balance
import dda_core as core

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

# v4.0: run-level enemy_controller / reward_policy; shared_boss decisions;
# pre-room equipment; full loot-event block (policy, proposed category
# probabilities, offers with stable item ids, selection, equipment before/
# after); support_distance for model decisions; applied_loot_bias_* and
# loot_taken_kind removed (superseded by the loot block).
# v3.0: controller identity/fallback columns, run_seed, player_profile,
# post_hp_ratio, experiment_config_version; cumulative total_kills_norm.
SCHEMA_VERSION = "room_samples_v4.0"
GAME_VERSION = "0.6.0"          # boots' dash_dist_mult now applied (game + bot)
GENERATOR_VERSION = "node_map_gen_v1"
# v3: equipment-aware abstract combat, seeded starting loadout, offers
# generated once per loot room by the reward policy, shared boss policy.
# v4: bot dash (boots' dash_dist_mult) with DASH_CD and ram-evasion approximation.
BOT_POLICY_VERSION = "dfs_backtrack_v4"
DDA_VERSION = "rule_based_dda_v1.2"    # matches dda_core's v1.2 condition contract

COND_DIM = core.COND_DIM     # 9 -- single source of truth is dda_core
TARGET_DIM = core.ACTION_DIM  # 4: enemy_hp_mult, enemy_dmg_mult, enemy_speed_mult, spawn_count

CONTROLLER_TYPES = ("fixed", "heuristic", "exploration", "cgan")
SHARED_BOSS = "shared_boss"
DECISION_TYPES = CONTROLLER_TYPES + (SHARED_BOSS,)
REWARD_POLICIES = ("fixed_uniform", "heuristic", "exploration", "learned_loot")
N_OFFERS = 3

COND_COLUMNS = ["cond_" + n for n in core.COND_NAMES]
ACTION_RAW_COLUMNS = ["applied_raw_hp_mult", "applied_raw_dmg_mult",
                      "applied_raw_speed_mult", "applied_raw_spawn_count"]
ACTION_COLUMNS = ["applied_hp_mult", "applied_dmg_mult", "applied_speed_mult",
                  "applied_spawn_count"]
EQUIP_COLUMNS = ["equip_" + s for s in core.SLOTS]
EQUIP_AFTER_COLUMNS = ["equip_after_" + s for s in core.SLOTS]
REWARD_PROB_COLUMNS = ["reward_prob_" + s for s in core.SLOTS]
OFFER_COLUMNS = [f"offer_{i}_{f}" for i in range(1, N_OFFERS + 1) for f in ("category", "item_id")]

# Loot-event columns are blank ("") on rooms without a loot event.
LOOT_EVENT_COLUMNS = [
    "loot_event_id", "reward_policy_type", "reward_policy_version", "reward_checkpoint_id",
    "reward_proposal_source", "reward_fallback_used", "reward_fallback_reason",
    *REWARD_PROB_COLUMNS, *OFFER_COLUMNS,
    "selection_mechanism", "selected_offer_index", "selected_category", "selected_item_id",
    *EQUIP_AFTER_COLUMNS, "replaced_existing",
]


RECENT_WINDOW_ROOMS = 3

def git_commit_short() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            h = out.stdout.strip()
            if h:
                return h
    except Exception:
        pass
    return "unknown"


def config_hash(extra: Optional[dict] = None) -> str:
    """Thin passthrough to dda_core.config_hash -- the ONE hash implementation."""
    return core.config_hash(extra=extra)


# ---------------------------------------------------------------------------
# Room sample schema
# ---------------------------------------------------------------------------

ROOM_SAMPLE_COLUMNS = [
    # identity / position. run_seed reproduces the map/bot of a run (the
    # same run_seed under two conditions = an equivalent-seed pair).
    # player_profile is METADATA ONLY (bot archetype name or "human") --
    # used for stratification, never as a model input. enemy_controller and
    # reward_policy are the run's experimental conditions.
    "run_id", "run_seed", "player_profile", "enemy_controller", "reward_policy",
    "room_seq", "room_idx", "room_kind", "node_depth", "progress_norm", "is_combat_room",

    # pre-state (cumulative, captured BEFORE this room's config is decided)
    "pre_melee_kills", "pre_magic_kills", "pre_melee_hits", "pre_magic_hits",
    "pre_damage_taken", "pre_damage_dealt", "pre_deaths", "pre_time_alive",
    "pre_hp_ratio",

    # equipment held when the room starts (stable item ids)
    *EQUIP_COLUMNS,

    # recent window (last RECENT_WINDOW_ROOMS rooms, captured BEFORE this room)
    "recent_melee_kills", "recent_magic_kills", "recent_melee_hits",
    "recent_magic_hits", "recent_damage_taken", "recent_damage_dealt",
    "recent_n_rooms",

    # cond_* -- the 9-dim pre-room condition (dda_core contract v1.2)
    *COND_COLUMNS,

    # enemy decision for THIS room. controller_type is the decision maker:
    # the run's enemy_controller, or "shared_boss" in boss rooms (identical
    # deterministic policy for every condition). action_source: exploration
    # branch, controller type, "model", "heuristic_fallback", "shared_boss"
    # or "non_combat" (no enemies; the logged action is not applied).
    "controller_type", "controller_version", "model_checkpoint_id",
    "action_source", "fallback_used", "fallback_reason", "support_distance",

    # raw DDA/model output BEFORE room-kind modifiers (e.g. boss multipliers)
    *ACTION_RAW_COLUMNS,
    # EFFECTIVE config actually used to simulate the room
    *ACTION_COLUMNS,

    # loot event (loot rooms only)
    *LOOT_EVENT_COLUMNS,

    # outcome (post-room delta only -- strictly this room's contribution)
    "room_result", "died_in_room", "post_hp_ratio", "melee_kills_in_room",
    "magic_kills_in_room", "melee_hits_in_room", "magic_hits_in_room",
    "damage_taken_in_room", "damage_dealt_in_room", "time_in_room",

    # versioning (redundant per-row so a row is self-describing)
    "schema_version", "game_version", "generator_version",
    "bot_policy_version", "dda_version", "experiment_config_version",
]

_VERSION_DEFAULTS = {
    "schema_version": SCHEMA_VERSION, "game_version": GAME_VERSION,
    "generator_version": GENERATOR_VERSION, "bot_policy_version": BOT_POLICY_VERSION,
    "dda_version": DDA_VERSION,
}


class RoomSample:
    """One row of ROOM_SAMPLE_COLUMNS. Construction requires EXACTLY the
    schema's columns (version columns default to the current code's), so a
    missing or misspelled field fails loudly instead of writing a bad row."""

    def __init__(self, **kw):
        data = dict(_VERSION_DEFAULTS)
        data.update(kw)
        missing = [c for c in ROOM_SAMPLE_COLUMNS if c not in data]
        unknown = [k for k in data if k not in ROOM_SAMPLE_COLUMNS]
        if missing or unknown:
            raise ValueError(f"RoomSample missing={missing} unknown={unknown}")
        self.__dict__.update(data)

    def as_row(self) -> List:
        return [self.__dict__[c] for c in ROOM_SAMPLE_COLUMNS]


# ---------------------------------------------------------------------------
# Recent window
# ---------------------------------------------------------------------------

class RecentWindow:
    FIELDS = ["melee_kills", "magic_kills", "melee_hits", "magic_hits",
              "damage_taken", "damage_dealt"]

    def __init__(self, maxlen: int = RECENT_WINDOW_ROOMS):
        self.maxlen = maxlen
        self._rooms: List[dict] = []

    def push(self, delta: dict):
        self._rooms.append({k: delta.get(k, 0) for k in self.FIELDS})
        if len(self._rooms) > self.maxlen:
            self._rooms.pop(0)

    def totals(self) -> dict:
        out = {k: 0.0 for k in self.FIELDS}
        for room in self._rooms:
            for k in self.FIELDS:
                out[k] += room[k]
        return out

    def n_rooms(self) -> int:
        return len(self._rooms)


def metrics_snapshot(metrics: core.CombatMetrics) -> dict:
    return {
        "melee_kills": metrics.melee_kills,
        "magic_kills": metrics.magic_kills,
        "melee_hits": metrics.melee_hits,
        "magic_hits": metrics.magic_hits,
        "damage_taken": metrics.damage_taken,
        "damage_dealt": metrics.damage_dealt,
        "deaths": metrics.deaths,
        "time_alive": metrics.time_alive,
    }


def snapshot_delta(pre: dict, post: dict) -> dict:
    return {k: post[k] - pre[k] for k in pre.keys()}


def build_condition(pre: dict, recent: "RecentWindow", hp_ratio: float,
                    node_depth: int, progress_norm: float) -> Tuple[List[float], dict, int]:
    """
    THE official pre-room condition (shared by bot_runner.py and rogue.py):
    behavioral components from the RECENT WINDOW (falling back to the
    cumulative-so-far snapshot when the window is empty, i.e. the first room
    of a run); hp_ratio instantaneous; total_kills_norm CUMULATIVE (from
    `pre`); node_depth_norm/progress_norm contextual. Uses only information
    available before the room starts. Returns (cond_vec, recent_totals,
    recent_n).
    """
    recent_totals = recent.totals()
    recent_n = recent.n_rooms()
    src = recent_totals if recent_n > 0 else pre
    cond = core.feature_vector_from_raw(
        src["melee_kills"], src["magic_kills"], src["melee_hits"], src["magic_hits"],
        src["damage_taken"], src["damage_dealt"],
        hp_ratio, node_depth, progress_norm,
        total_kills=pre["melee_kills"] + pre["magic_kills"],
    )
    return cond, recent_totals, recent_n


def blank_loot_event() -> dict:
    return {c: "" for c in LOOT_EVENT_COLUMNS}


def make_loot_event(*, run_id: str, room_seq: int, reward_meta: dict, probs: List[float],
                    offers: List[tuple], selection_mechanism: str, selected_index: int,
                    equip_before: dict, equip_after: dict, replaced: bool) -> dict:
    """Loot-event columns for one loot room. offers: [(category, item_id)]."""
    if len(offers) != N_OFFERS:
        raise ValueError(f"expected {N_OFFERS} offers, got {len(offers)}")
    ev = {
        "loot_event_id": f"{run_id}:{room_seq}",
        "reward_policy_type": reward_meta["reward_policy_type"],
        "reward_policy_version": reward_meta["reward_policy_version"],
        "reward_checkpoint_id": reward_meta.get("reward_checkpoint_id", ""),
        "reward_proposal_source": reward_meta.get("reward_proposal_source", ""),
        "reward_fallback_used": int(reward_meta.get("reward_fallback_used", 0)),
        "reward_fallback_reason": reward_meta.get("reward_fallback_reason", ""),
        "selection_mechanism": selection_mechanism,
        "selected_offer_index": selected_index,
        "selected_category": offers[selected_index][0] if selected_index >= 0 else "none",
        "selected_item_id": offers[selected_index][1] if selected_index >= 0 else "",
        "replaced_existing": int(replaced),
    }
    for slot, p in zip(core.SLOTS, probs):
        ev["reward_prob_" + slot] = round(float(p), 6)
    for i, (cat, iid) in enumerate(offers, start=1):
        ev[f"offer_{i}_category"] = cat
        ev[f"offer_{i}_item_id"] = iid
    for slot in core.SLOTS:
        ev["equip_after_" + slot] = equip_after[slot]
    return ev


def build_room_sample(
    *, run_id: str, run_seed: int, player_profile: str, enemy_controller: str,
    reward_policy: str, room_seq: int, room_idx: int,
    room_kind: str, node_depth: int, progress_norm: float,
    pre: dict, pre_hp_ratio: float, equip_before: dict, recent_totals: dict, recent_n: int,
    cond_vec: List[float], controller_meta: dict, applied_raw: dict, applied: dict,
    room_result: str, delta: dict, time_in_room: float, post_hp_ratio: float,
    loot_event: Optional[dict], experiment_config_version: str,
) -> RoomSample:
    """Assembles one RoomSample. Shared by bot_runner.py and rogue.py so the
    human and synthetic contracts cannot drift."""
    kw = dict(
        run_id=run_id, run_seed=int(run_seed), player_profile=player_profile,
        enemy_controller=enemy_controller, reward_policy=reward_policy,
        room_seq=room_seq, room_idx=room_idx, room_kind=room_kind, node_depth=node_depth,
        progress_norm=progress_norm,
        is_combat_room=1 if room_kind in ("enemy", "boss") else 0,

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

        controller_type=controller_meta["controller_type"],
        controller_version=controller_meta["controller_version"],
        model_checkpoint_id=controller_meta.get("model_checkpoint_id", ""),
        action_source=controller_meta.get("action_source", ""),
        fallback_used=int(controller_meta.get("fallback_used", 0)),
        fallback_reason=controller_meta.get("fallback_reason", ""),
        support_distance=controller_meta.get("support_distance", ""),

        applied_raw_hp_mult=applied_raw["enemy_hp_mult"],
        applied_raw_dmg_mult=applied_raw["enemy_dmg_mult"],
        applied_raw_speed_mult=applied_raw["enemy_speed_mult"],
        applied_raw_spawn_count=applied_raw["spawn_count"],
        applied_hp_mult=applied["enemy_hp_mult"],
        applied_dmg_mult=applied["enemy_dmg_mult"],
        applied_speed_mult=applied["enemy_speed_mult"],
        applied_spawn_count=applied["spawn_count"],

        room_result=room_result, died_in_room=1 if room_result == "died" else 0,
        post_hp_ratio=round(float(post_hp_ratio), 6),
        melee_kills_in_room=delta["melee_kills"], magic_kills_in_room=delta["magic_kills"],
        melee_hits_in_room=delta["melee_hits"], magic_hits_in_room=delta["magic_hits"],
        damage_taken_in_room=delta["damage_taken"], damage_dealt_in_room=delta["damage_dealt"],
        time_in_room=time_in_room,
        experiment_config_version=experiment_config_version,
    )
    kw.update(dict(zip(COND_COLUMNS, cond_vec)))
    kw.update({"equip_" + s: equip_before[s] for s in core.SLOTS})
    kw.update(loot_event if loot_event is not None else blank_loot_event())
    return RoomSample(**kw)


# ---------------------------------------------------------------------------
# Sharded writer
# ---------------------------------------------------------------------------

class ShardWriter:
    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(ROOM_SAMPLE_COLUMNS)
        self._n = 0

    def write(self, sample: RoomSample):
        self._writer.writerow(sample.as_row())
        self._n += 1

    def close(self):
        self._fh.close()
        return self._n


def open_shard_writer(path: str) -> ShardWriter:
    return ShardWriter(path)


class DatasetVersionError(Exception):
    pass


def write_manifest(
    manifest_path: str,
    n_rows: int,
    seed_range: List[int],
    extra: Optional[dict] = None,
    controller_counts: Optional[Dict[str, int]] = None,
) -> dict:
    """
    FIX (finding #7 wiring): `extra` is now actually passed into
    config_hash(extra=...) so bot-policy/room-generation constants that
    live outside dda_core.py (RECENT_WINDOW_ROOMS, node/room counts,
    archetype definitions, etc.) are folded into the SAME hash that gets
    checked at load time -- not just recorded alongside it.
    """
    exp_cfg = balance.load_experiment_config()
    extra = dict(extra or {})
    extra["experiment_config_hash"] = balance.experiment_config_hash(exp_cfg)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "game_version": GAME_VERSION,
        "generator_version": GENERATOR_VERSION,
        "bot_policy_version": BOT_POLICY_VERSION,
        "dda_version": DDA_VERSION,
        "experiment_config_version": exp_cfg["experiment_config_version"],
        "experiment_config_hash": extra["experiment_config_hash"],
        "config_hash": config_hash(extra=extra),
        "git_commit": git_commit_short(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed_range": seed_range,
        "cond_dim": COND_DIM,
        "target_dim": TARGET_DIM,
        "n_rows": n_rows,
        "columns": ROOM_SAMPLE_COLUMNS,
        "extra": extra,
    }
    if controller_counts is not None:
        manifest["controller_counts"] = controller_counts
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_manifest(manifest_path: str) -> dict:
    with open(manifest_path) as f:
        return json.load(f)


def merge_shards(shard_paths: List[str], out_csv_path: str) -> int:
    """
    Merges shard CSVs into one file. Verifies every shard's header matches
    ROOM_SAMPLE_COLUMNS exactly before merging (rejects mismatched shards
    rather than silently concatenating incompatible data). Writes to a
    temp path and renames at the end (atomic on POSIX), so a failure
    partway through never leaves a partially-merged file at out_csv_path.
    """
    tmp_path = out_csv_path + ".tmp"
    total = 0
    with open(tmp_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(ROOM_SAMPLE_COLUMNS)
        for sp in shard_paths:
            with open(sp, newline="") as in_f:
                reader = csv.reader(in_f)
                header = next(reader, None)
                if header != ROOM_SAMPLE_COLUMNS:
                    os.remove(tmp_path)
                    raise DatasetVersionError(
                        f"Shard {sp} has mismatched header; refusing to merge."
                    )
                for row in reader:
                    if len(row) != len(ROOM_SAMPLE_COLUMNS):
                        os.remove(tmp_path)
                        raise DatasetVersionError(
                            f"Shard {sp} has a malformed row with "
                            f"{len(row)} columns (expected {len(ROOM_SAMPLE_COLUMNS)})."
                        )
                    writer.writerow(row)
                    total += 1
    os.replace(tmp_path, out_csv_path)
    return total


# ---------------------------------------------------------------------------
# Validation (finding #6: substantially strengthened)
# ---------------------------------------------------------------------------

def row_problem(row: List[str], col: Dict[str, int]) -> str:
    """Semantic checks for one CSV row; returns "" when the row is valid."""
    g = lambda c: row[col[c]]
    if g("schema_version") != SCHEMA_VERSION:
        return f"schema_version {g('schema_version')!r} != {SCHEMA_VERSION!r} (mixed-schema dataset)"
    if g("enemy_controller") not in CONTROLLER_TYPES:
        return f"enemy_controller {g('enemy_controller')!r} not in {CONTROLLER_TYPES}"
    if g("reward_policy") not in REWARD_POLICIES:
        return f"reward_policy {g('reward_policy')!r} not in {REWARD_POLICIES}"
    ctrl = g("controller_type")
    if ctrl not in DECISION_TYPES:
        return f"controller_type {ctrl!r} not in {DECISION_TYPES}"
    if g("room_kind") == "boss":
        if ctrl != SHARED_BOSS:
            return f"boss room decided by {ctrl!r}; boss rooms must use {SHARED_BOSS!r}"
    elif ctrl != g("enemy_controller"):
        return f"controller_type {ctrl!r} != run enemy_controller {g('enemy_controller')!r}"
    if g("fallback_used") not in ("0", "1"):
        return f"fallback_used {g('fallback_used')!r} (expected 0/1)"
    for c in EQUIP_COLUMNS:
        if g(c) not in core.SLOT_OF_ITEM or core.SLOT_OF_ITEM[g(c)] != c[len("equip_"):]:
            return f"{c}={g(c)!r} is not a valid item id for that slot"
    if g("room_kind") == "loot":
        if not g("loot_event_id"):
            return "loot room without a loot event"
        try:
            probs = [float(g(c)) for c in REWARD_PROB_COLUMNS]
        except ValueError:
            return "non-numeric reward probabilities"
        if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-4:
            return f"reward probabilities {probs} are not a distribution"
        if g("reward_policy_type") != g("reward_policy"):
            return "reward_policy_type differs from the run's reward_policy"
        for i in range(1, N_OFFERS + 1):
            iid, cat = g(f"offer_{i}_item_id"), g(f"offer_{i}_category")
            if core.SLOT_OF_ITEM.get(iid) != cat:
                return f"offer {i} item {iid!r} is not in category {cat!r}"
        k = int(g("selected_offer_index"))
        if k == -1:
            if g("selected_category") != "none" or g("selected_item_id"):
                return "no selection but a selected item is recorded"
        elif 0 <= k < N_OFFERS:
            if (g("selected_item_id") != g(f"offer_{k + 1}_item_id")
                    or g("selected_category") != g(f"offer_{k + 1}_category")):
                return "selected item does not match the selected offer"
            if g("equip_after_" + g("selected_category")) != g("selected_item_id"):
                return "selected item is not equipped after selection"
        else:
            return f"selected_offer_index {k} out of range"
        if g("reward_fallback_used") not in ("0", "1"):
            return "reward_fallback_used must be 0/1 on loot rows"
    elif g("loot_event_id"):
        return "loot event recorded on a non-loot room"
    return ""


def validate_dataset(csv_path: str, manifest_path: str, extra: Optional[dict] = None,
                     expected_extra_subset: Optional[dict] = None,
                     strict_experiment_config: bool = True) -> dict:
    """
    Validates a dataset against BOTH internal consistency (manifest vs. CSV
    contents) AND the CURRENT CODE's own versions (manifest vs. what this
    exact codebase would produce right now). Raises DatasetVersionError on
    any mismatch, with a message naming exactly what failed.

    Specifically fixes the three tampering scenarios the review demonstrated
    were previously accepted:
      1. A manifest with a wrong config_hash / game_version is now REJECTED
         (previously only schema_version/cond_dim/target_dim were checked).
      2. A manifest claiming n_rows=999999 when the CSV actually has far
         fewer rows is now REJECTED (row count is now counted, not trusted).
      3. A malformed row with the wrong column count is now REJECTED
         (previously rows were never counted or column-checked at all).
    """
    if not os.path.exists(csv_path):
        raise DatasetVersionError(f"Dataset CSV not found: {csv_path}")
    if not os.path.exists(manifest_path):
        raise DatasetVersionError(f"Manifest not found: {manifest_path}")

    manifest = load_manifest(manifest_path)

    # 1. Compare declared schema/version strings against what THIS code
    #    currently defines -- not just internal self-consistency.
    current_expected = {
        "schema_version": SCHEMA_VERSION,
        "game_version": GAME_VERSION,
        "generator_version": GENERATOR_VERSION,
        "bot_policy_version": BOT_POLICY_VERSION,
        "dda_version": DDA_VERSION,
        "cond_dim": COND_DIM,
        "target_dim": TARGET_DIM,
    }
    for key, expected in current_expected.items():
        got = manifest.get(key)
        if got != expected:
            raise DatasetVersionError(
                f"Manifest field '{key}' = {got!r} does not match current "
                f"code's {expected!r}. Dataset is incompatible with this "
                f"codebase (re-generate it, or check out the matching commit)."
            )

    # 1b. Experiment configuration (controllers, outcome definitions, balance
    #     targets) must be the one currently on disk -- mixing datasets
    #     collected under different exploration/targets is refused.
    if strict_experiment_config:
        exp_cfg = balance.load_experiment_config()
        for key, expected in (
            ("experiment_config_version", exp_cfg["experiment_config_version"]),
            ("experiment_config_hash", balance.experiment_config_hash(exp_cfg)),
        ):
            if manifest.get(key) != expected:
                raise DatasetVersionError(
                    f"Manifest field '{key}' = {manifest.get(key)!r} does not match the "
                    f"current experiment_config.json ({expected!r}).")

    # 1c. Policy constants the CALLER knows (e.g. bot_runner's map/policy
    #     parameters) must equal what the manifest recorded; the manifest's
    #     own extra is otherwise only self-consistent, not current.
    if expected_extra_subset:
        mextra = manifest.get("extra") or {}
        for k, v in expected_extra_subset.items():
            if json.dumps(mextra.get(k), sort_keys=True) != json.dumps(v, sort_keys=True):
                raise DatasetVersionError(
                    f"Manifest extra[{k!r}] = {mextra.get(k)!r} does not match the "
                    f"current code's {v!r}.")

    # 2. config_hash must match what the CURRENT code computes for the
    #    SAME extra payload the manifest itself recorded (recomputed, not
    #    trusted) -- this is the actual defect the review reproduced.
    manifest_extra = manifest.get("extra") or {}
    recomputed_hash = config_hash(extra=manifest_extra if manifest_extra else extra)
    if manifest.get("config_hash") != recomputed_hash:
        raise DatasetVersionError(
            f"config_hash mismatch: manifest says {manifest.get('config_hash')!r}, "
            f"current code computes {recomputed_hash!r} for the same extra "
            f"payload. The dataset's semantic contract does not match this "
            f"codebase's current constants."
        )

    if manifest.get("columns") != ROOM_SAMPLE_COLUMNS:
        raise DatasetVersionError(
            "Manifest's recorded column list does not match "
            "telemetry.ROOM_SAMPLE_COLUMNS in the current code."
        )

    # 3. Count actual rows and verify every row's column count -- catches
    #    both an inflated/wrong declared n_rows AND malformed rows.
    declared_n_rows = manifest.get("n_rows")
    actual_n_rows = 0
    seen_keys = set()
    duplicate_keys = 0
    controller_counts: Dict[str, int] = {}
    reward_counts: Dict[str, int] = {}
    n_fallback = n_loot_events = n_reward_fallback = 0
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ROOM_SAMPLE_COLUMNS:
            raise DatasetVersionError(
                "CSV header does not match telemetry.ROOM_SAMPLE_COLUMNS."
            )
        col = {c: i for i, c in enumerate(ROOM_SAMPLE_COLUMNS)}
        run_id_idx, room_seq_idx = col["run_id"], col["room_seq"]
        for i, row in enumerate(reader):
            if len(row) != len(ROOM_SAMPLE_COLUMNS):
                raise DatasetVersionError(
                    f"Row {i} has {len(row)} columns, expected "
                    f"{len(ROOM_SAMPLE_COLUMNS)} (malformed/truncated row)."
                )
            problem = row_problem(row, col)
            if problem:
                raise DatasetVersionError(f"Row {i}: {problem}")
            ctrl = row[col["enemy_controller"]]
            controller_counts[ctrl] = controller_counts.get(ctrl, 0) + 1
            rp = row[col["reward_policy"]]
            reward_counts[rp] = reward_counts.get(rp, 0) + 1
            if row[col["fallback_used"]] == "1":
                n_fallback += 1
            if row[col["loot_event_id"]]:
                n_loot_events += 1
                if row[col["reward_fallback_used"]] == "1":
                    n_reward_fallback += 1
            actual_n_rows += 1
            key = (row[run_id_idx], row[room_seq_idx])
            if key in seen_keys:
                duplicate_keys += 1
            seen_keys.add(key)

    if declared_n_rows != actual_n_rows:
        raise DatasetVersionError(
            f"Manifest declares n_rows={declared_n_rows}, but the CSV "
            f"actually contains {actual_n_rows} rows."
        )

    if duplicate_keys > 0:
        raise DatasetVersionError(
            f"Found {duplicate_keys} duplicate (run_id, room_seq) pairs in "
            f"the dataset -- this should never happen and indicates a "
            f"corrupted merge or a bot_runner bug."
        )

    return {
        "ok": True,
        "n_rows": actual_n_rows,
        "schema_version": manifest["schema_version"],
        "config_hash": manifest["config_hash"],
        "experiment_config_version": manifest.get("experiment_config_version"),
        "controller_counts": controller_counts,
        "reward_policy_counts": reward_counts,
        "n_fallback_rows": n_fallback,
        "n_loot_events": n_loot_events,
        "n_reward_fallback_events": n_reward_fallback,
    }
