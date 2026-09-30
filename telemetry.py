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
from typing import Dict, List, Optional
import csv
import json
import os
import subprocess
import time

import dda_core as core

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "room_samples_v2.1"   # bumped: COND_DIM 6->9, new columns
GAME_VERSION = "0.4.0"
GENERATOR_VERSION = "node_map_gen_v1"
BOT_POLICY_VERSION = "dfs_backtrack_v1"
DDA_VERSION = "rule_based_dda_v1.1"    # matches dda_core's v1.1 condition contract

COND_DIM = core.COND_DIM     # 9 -- single source of truth is dda_core
TARGET_DIM = 4               # enemy_hp_mult, enemy_dmg_mult, enemy_speed_mult, spawn_count

RECENT_WINDOW_ROOMS = 3

LOOT_KIND_COLUMNS = ["weapon", "spell", "armor", "boots"]


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
    # identity / position
    "run_id", "room_seq", "room_idx", "room_kind", "node_depth",
    "progress_norm", "is_combat_room",

    # pre-state (cumulative, captured BEFORE this room's config is decided)
    "pre_melee_kills", "pre_magic_kills", "pre_melee_hits", "pre_magic_hits",
    "pre_damage_taken", "pre_damage_dealt", "pre_deaths", "pre_time_alive",
    "pre_hp_ratio",

    # recent window (last RECENT_WINDOW_ROOMS rooms, captured BEFORE this
    # room -- these ARE the values cond_* below are derived from)
    "recent_melee_kills", "recent_magic_kills", "recent_melee_hits",
    "recent_magic_hits", "recent_damage_taken", "recent_damage_dealt",
    "recent_n_rooms",

    # cond_* -- the ACTUAL 9-dim vector fed to rule_based_dda()/the model
    # for this room. Behavioral components (ratios/hpk/dmg_ratio) are
    # derived from the recent window; hp_ratio is instantaneous;
    # total_kills_norm/node_depth_norm/progress_norm are cumulative
    # contextual features. See bot_runner.simulate_run / dda_core module
    # docstring for the rationale (finding #3).
    "cond_melee_ratio", "cond_magic_ratio", "cond_hpk_melee", "cond_hpk_magic",
    "cond_dmg_ratio", "cond_hp_ratio", "cond_total_kills_norm",
    "cond_node_depth_norm", "cond_progress_norm",

    # raw DDA/model output BEFORE room-kind modifiers (e.g. boss multipliers)
    "applied_raw_hp_mult", "applied_raw_dmg_mult", "applied_raw_speed_mult",
    "applied_raw_spawn_count",

    # EFFECTIVE config actually used to simulate the room (finding #5 fix)
    "applied_hp_mult", "applied_dmg_mult", "applied_speed_mult",
    "applied_spawn_count",
    "applied_loot_bias_weapon", "applied_loot_bias_spell",
    "applied_loot_bias_armor", "applied_loot_bias_boots",

    # outcome (post-room delta only -- strictly this room's contribution)
    "room_result", "died_in_room", "melee_kills_in_room", "magic_kills_in_room",
    "melee_hits_in_room", "magic_hits_in_room", "damage_taken_in_room",
    "damage_dealt_in_room", "time_in_room", "loot_taken_kind",

    # versioning (redundant per-row so a row is self-describing even split
    # out of its manifest/shard context)
    "schema_version", "game_version", "generator_version",
    "bot_policy_version", "dda_version",
]


@dataclass
class RoomSample:
    run_id: str
    room_seq: int
    room_idx: int
    room_kind: str
    node_depth: int
    progress_norm: float
    is_combat_room: int

    pre_melee_kills: int
    pre_magic_kills: int
    pre_melee_hits: int
    pre_magic_hits: int
    pre_damage_taken: float
    pre_damage_dealt: float
    pre_deaths: int
    pre_time_alive: float
    pre_hp_ratio: float

    recent_melee_kills: float
    recent_magic_kills: float
    recent_melee_hits: float
    recent_magic_hits: float
    recent_damage_taken: float
    recent_damage_dealt: float
    recent_n_rooms: int

    cond_melee_ratio: float
    cond_magic_ratio: float
    cond_hpk_melee: float
    cond_hpk_magic: float
    cond_dmg_ratio: float
    cond_hp_ratio: float
    cond_total_kills_norm: float
    cond_node_depth_norm: float
    cond_progress_norm: float

    applied_raw_hp_mult: float
    applied_raw_dmg_mult: float
    applied_raw_speed_mult: float
    applied_raw_spawn_count: int

    applied_hp_mult: float
    applied_dmg_mult: float
    applied_speed_mult: float
    applied_spawn_count: int
    applied_loot_bias_weapon: float
    applied_loot_bias_spell: float
    applied_loot_bias_armor: float
    applied_loot_bias_boots: float

    room_result: str
    died_in_room: int
    melee_kills_in_room: int
    magic_kills_in_room: int
    melee_hits_in_room: int
    magic_hits_in_room: int
    damage_taken_in_room: float
    damage_dealt_in_room: float
    time_in_room: float
    loot_taken_kind: str

    schema_version: str = SCHEMA_VERSION
    game_version: str = GAME_VERSION
    generator_version: str = GENERATOR_VERSION
    bot_policy_version: str = BOT_POLICY_VERSION
    dda_version: str = DDA_VERSION

    def as_row(self) -> List:
        d = self.__dict__
        return [d[c] for c in ROOM_SAMPLE_COLUMNS]


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


def loot_bias_to_cols(loot_bias: Optional[List[float]]) -> Dict[str, float]:
    if not loot_bias or len(loot_bias) != 4:
        loot_bias = [0.25, 0.25, 0.25, 0.25]
    return dict(zip(["weapon", "spell", "armor", "boots"], loot_bias))


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
) -> dict:
    """
    FIX (finding #7 wiring): `extra` is now actually passed into
    config_hash(extra=...) so bot-policy/room-generation constants that
    live outside dda_core.py (RECENT_WINDOW_ROOMS, node/room counts,
    archetype definitions, etc.) are folded into the SAME hash that gets
    checked at load time -- not just recorded alongside it.
    """
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "game_version": GAME_VERSION,
        "generator_version": GENERATOR_VERSION,
        "bot_policy_version": BOT_POLICY_VERSION,
        "dda_version": DDA_VERSION,
        "config_hash": config_hash(extra=extra),
        "git_commit": git_commit_short(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed_range": seed_range,
        "cond_dim": COND_DIM,
        "target_dim": TARGET_DIM,
        "n_rows": n_rows,
        "columns": ROOM_SAMPLE_COLUMNS,
        "extra": extra or {},
    }
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

def validate_dataset(csv_path: str, manifest_path: str, extra: Optional[dict] = None) -> dict:
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
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ROOM_SAMPLE_COLUMNS:
            raise DatasetVersionError(
                "CSV header does not match telemetry.ROOM_SAMPLE_COLUMNS."
            )
        run_id_idx = ROOM_SAMPLE_COLUMNS.index("run_id")
        room_seq_idx = ROOM_SAMPLE_COLUMNS.index("room_seq")
        for i, row in enumerate(reader):
            if len(row) != len(ROOM_SAMPLE_COLUMNS):
                raise DatasetVersionError(
                    f"Row {i} has {len(row)} columns, expected "
                    f"{len(ROOM_SAMPLE_COLUMNS)} (malformed/truncated row)."
                )
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
    }
