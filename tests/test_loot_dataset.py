"""Delayed next-3-combat-room loot outcomes: horizon rules, leakage and determinism."""
import csv
import random

import pytest

import loot_dataset
import telemetry
from conftest import make_dataset


def _row(run, seq, kind, **kw):
    r = {c: "" for c in telemetry.ROOM_SAMPLE_COLUMNS}
    r.update({"run_id": run, "run_seed": "1", "player_profile": "hybrid", "enemy_controller": "heuristic",
              "reward_policy": "exploration", "room_seq": str(seq), "room_idx": str(seq), "room_kind": kind,
              "node_depth": "1", "progress_norm": f"{seq / 16:.4f}",
              "is_combat_room": "1" if kind in ("enemy", "boss") else "0",
              "pre_hp_ratio": "0.8", "post_hp_ratio": "0.7", "room_result": "cleared", "died_in_room": "0",
              "damage_taken_in_room": "10", "damage_dealt_in_room": "40", "melee_kills_in_room": "2",
              "magic_kills_in_room": "0", "melee_hits_in_room": "5", "magic_hits_in_room": "3",
              "time_in_room": "6", "schema_version": telemetry.SCHEMA_VERSION,
              "experiment_config_version": "x"})
    for c in telemetry.COND_COLUMNS:
        r[c] = "0.5"
    for s in ("weapon", "spell", "armor", "boots"):
        r["equip_" + s] = {"weapon": "weapon.sword", "spell": "spell.magic_bolt", "armor": "armor.cloth_wrap",
                           "boots": "boots.leather"}[s]
    if kind == "loot":
        r.update({"loot_event_id": f"{run}:{seq}", "reward_policy_type": "exploration",
                  "reward_fallback_used": "0", "selected_offer_index": "0", "selected_category": "armor",
                  "selected_item_id": "armor.chain_shirt", "replaced_existing": "1",
                  "damage_taken_in_room": "0", "damage_dealt_in_room": "0", "time_in_room": "0",
                  "melee_kills_in_room": "0", "melee_hits_in_room": "0", "magic_hits_in_room": "0"})
        for s in ("weapon", "spell", "armor", "boots"):
            r["reward_prob_" + s] = "0.25"
            r["equip_after_" + s] = r["equip_" + s]
        r["equip_after_armor"] = "armor.chain_shirt"
    r.update({k: str(v) for k, v in kw.items()})
    return r


def _ev(rows):
    return {e["loot_event_id"]: e for e in loot_dataset.derive_from_rows(rows)}


def test_exactly_three_subsequent_combat_rooms_ignoring_noncombat():
    rows = [_row("A", 1, "loot"), _row("A", 2, "nothing"), _row("A", 3, "enemy"), _row("A", 4, "loot"),
            _row("A", 5, "enemy"), _row("A", 6, "nothing"), _row("A", 7, "enemy", post_hp_ratio=0.55),
            _row("A", 8, "enemy", damage_taken_in_room=999)]
    e = _ev(rows)["A:1"]
    assert e["horizon_combat_rooms"] == 3 and e["horizon_complete"] == 1 and e["terminated_early"] == 0
    assert e["termination_reason"] == "complete"
    assert e["end_hp_ratio"] == pytest.approx(0.55)            # room 7 is the 3rd combat room
    assert e["cum_damage_taken"] == pytest.approx(30.0)        # room 8 excluded
    assert e["damage_share"] == pytest.approx(30 / 150)
    assert e["combat_efficiency"] == pytest.approx(6 / 24)
    assert e["duration_norm_per_room"] == pytest.approx(6 / 30)
    assert e["progress_gained"] == pytest.approx((7 - 1) / 16, abs=1e-4)


@pytest.mark.parametrize("death_at,expected_n", [(1, 1), (2, 2)])
def test_death_after_room_one_or_two(death_at, expected_n):
    rows = [_row("A", 1, "loot")]
    for i in range(1, 4):
        died = i == death_at
        rows.append(_row("A", 1 + i, "enemy", died_in_room=int(died), room_result="died" if died else "cleared",
                         post_hp_ratio=0.0 if died else 0.6))
        if died:
            break
    e = _ev(rows)["A:1"]
    assert e["horizon_combat_rooms"] == expected_n
    assert e["died_within"] == 1 and e["survived_horizon"] == 0 and e["end_hp_ratio"] == 0.0
    assert e["termination_reason"] == "death" and e["terminated_early"] == 1
    assert e["h_death_within"] == 1.0


def test_run_ends_before_three_rooms_and_victory():
    rows = [_row("A", 1, "loot"), _row("A", 2, "enemy"), _row("A", 3, "nothing")]
    e = _ev(rows)["A:1"]
    assert e["horizon_combat_rooms"] == 1 and e["termination_reason"] == "run_end" and e["terminated_early"] == 1
    rows = [_row("B", 1, "loot"), _row("B", 2, "enemy"), _row("B", 3, "boss")]
    e = _ev(rows)["B:1"]
    assert e["termination_reason"] == "victory" and e["boss_reached"] == 1 and e["horizon_combat_rooms"] == 2
    e = _ev([_row("C", 1, "loot", post_hp_ratio=0.42)])["C:1"]
    assert e["horizon_combat_rooms"] == 0 and e["end_hp_ratio"] == pytest.approx(0.42)


def test_multiple_loot_events_in_one_run_and_no_run_crossing():
    rows = [_row("A", 1, "loot"), _row("A", 2, "enemy"), _row("A", 3, "loot"), _row("A", 4, "enemy"),
            _row("B", 5, "enemy", damage_taken_in_room=500), _row("B", 6, "enemy"),
            _row("A", 5, "enemy")]
    ev = _ev(rows)
    assert set(ev) == {"A:1", "A:3"}
    assert ev["A:1"]["horizon_combat_rooms"] == 3          # A:2, A:4, A:5
    assert ev["A:3"]["horizon_combat_rooms"] == 2          # A:4, A:5 (run A ends)
    assert ev["A:1"]["cum_damage_taken"] == pytest.approx(30.0)   # run B never enters


def test_no_temporal_leakage():
    base = [_row("A", 1, "loot"), _row("A", 2, "enemy"), _row("A", 3, "enemy"), _row("A", 4, "enemy")]
    e0 = _ev(base)["A:1"]
    # Pre-reward features come only from the loot row ...
    assert e0["pre_hp_ratio"] == base[0]["pre_hp_ratio"]
    assert all(e0[c] == base[0][c] for c in telemetry.COND_COLUMNS)
    # ... and changing an EARLIER room cannot change the horizon outcome.
    earlier = [_row("A", 0, "enemy", damage_taken_in_room=777, died_in_room=0)] + base
    e1 = _ev(earlier)["A:1"]
    assert {k: e1[k] for k in loot_dataset.H_COLUMNS} == {k: e0[k] for k in loot_dataset.H_COLUMNS}
    # Changing a horizon room changes the outcome but never the pre-reward features.
    changed = [dict(r) for r in base]
    changed[2]["damage_taken_in_room"] = "90"
    e2 = _ev(changed)["A:1"]
    assert e2["h_damage_share"] != e0["h_damage_share"]
    assert all(e2[c] == e0[c] for c in telemetry.COND_COLUMNS + ["pre_hp_ratio", "node_depth"])


def test_deterministic_reconstruction_on_bot_data(tmp_path):
    p = make_dataset(tmp_path / "d.csv", runs=60, controller="exploration", extra=("--reward-policy", "exploration"))
    rows = list(csv.DictReader(open(p)))
    a = loot_dataset.derive_from_rows(rows)
    shuffled = list(rows)
    random.Random(1).shuffle(shuffled)
    assert loot_dataset.derive_from_rows(shuffled) == a
    loot_dataset.derive([p], str(tmp_path / "l1.csv"))
    loot_dataset.derive([p], str(tmp_path / "l2.csv"))
    assert open(tmp_path / "l1.csv").read() == open(tmp_path / "l2.csv").read()
    assert loot_dataset.validate_loot_dataset(str(tmp_path / "l1.csv"))["n_rows"] == len(a) > 0
    # every event's equipment-after matches the selection; the bot really equipped it
    by_key = {(r["run_id"], int(r["room_seq"])): r for r in rows}
    for e in a:
        if e["selected_category"] != "none":
            assert e["equip_after_" + e["selected_category"]] == e["selected_item_id"]
            nxt = by_key.get((e["run_id"], int(e["room_seq"]) + 1))
            if nxt:
                assert nxt["equip_" + e["selected_category"]] == e["selected_item_id"]


def test_source_change_invalidates_derived_dataset(tmp_path):
    p = make_dataset(tmp_path / "d.csv", runs=10)
    loot_dataset.derive([p], str(tmp_path / "l.csv"))
    with open(p, "a") as f:
        f.write("")
    loot_dataset.validate_loot_dataset(str(tmp_path / "l.csv"))
    with open(p, "a") as f:
        f.write("\n")
    with pytest.raises(telemetry.DatasetVersionError):
        loot_dataset.validate_loot_dataset(str(tmp_path / "l.csv"))
