import csv

import pytest

import dda_core as core
import telemetry
from conftest import make_dataset


EXPECTED_COND = [
    "cond_melee_ratio", "cond_magic_ratio", "cond_hpk_melee", "cond_hpk_magic",
    "cond_dmg_ratio", "cond_hp_ratio", "cond_total_kills_norm",
    "cond_node_depth_norm", "cond_progress_norm",
]


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_condition_vector_order_and_dimension():
    assert core.COND_DIM == 9
    assert telemetry.COND_COLUMNS == EXPECTED_COND
    i = telemetry.ROOM_SAMPLE_COLUMNS.index(EXPECTED_COND[0])
    assert telemetry.ROOM_SAMPLE_COLUMNS[i:i + 9] == EXPECTED_COND
    v = core.feature_vector_from_raw(3, 1, 6, 4, 10.0, 40.0, 0.5, 4, 0.25, total_kills=12)
    assert len(v) == 9
    assert v[0] == pytest.approx(0.75) and v[1] == pytest.approx(0.25)
    assert v[2] == pytest.approx(2.0) and v[3] == pytest.approx(4.0)
    assert v[4] == pytest.approx(0.25) and v[5] == pytest.approx(0.5)
    assert v[6] == pytest.approx(12 / core.TOTAL_KILLS_NORM_CAP)
    assert v[7] == pytest.approx(0.4) and v[8] == pytest.approx(0.25)


def test_total_kills_norm_is_cumulative_not_windowed():
    pre = {"melee_kills": 20, "magic_kills": 4, "melee_hits": 40, "magic_hits": 8,
           "damage_taken": 50.0, "damage_dealt": 300.0, "deaths": 0, "time_alive": 30.0}
    w = telemetry.RecentWindow()
    w.push({"melee_kills": 2, "magic_kills": 0, "melee_hits": 4, "magic_hits": 0,
            "damage_taken": 5.0, "damage_dealt": 30.0})
    cond, totals, n = telemetry.build_condition(pre, w, 0.8, 3, 0.5)
    assert n == 1 and totals["melee_kills"] == 2
    assert cond[6] == pytest.approx(24 / core.TOTAL_KILLS_NORM_CAP)   # cumulative
    assert cond[0] == pytest.approx(1.0)                              # behavioral: window


def test_recent_window_values_and_no_temporal_leakage(tmp_path):
    rows = _rows(make_dataset(tmp_path / "d.csv", runs=30))
    runs = {}
    for r in rows:
        runs.setdefault(r["run_id"], []).append(r)
    checked = 0
    for rr in runs.values():
        rr.sort(key=lambda r: int(r["room_seq"]))
        for k, r in enumerate(rr):
            prev = rr[max(0, k - telemetry.RECENT_WINDOW_ROOMS):k]
            assert int(r["recent_n_rooms"]) == len(prev)
            for f in ("melee_kills", "magic_kills", "damage_taken"):
                assert float(r["recent_" + f]) == pytest.approx(sum(float(p[f + "_in_room"]) for p in prev))
            # pre-room state equals the previous room's post-room state
            if k > 0:
                assert float(r["pre_hp_ratio"]) == pytest.approx(float(rr[k - 1]["post_hp_ratio"]), abs=1e-5)
                assert int(r["pre_melee_kills"]) == int(rr[k - 1]["pre_melee_kills"]) + int(rr[k - 1]["melee_kills_in_room"])
            # the logged condition is reproducible from PRE-room columns only
            pre = {f: float(r["pre_" + f]) for f in ("melee_kills", "magic_kills", "melee_hits",
                                                     "magic_hits", "damage_taken", "damage_dealt")}
            src = {f: float(r["recent_" + f]) for f in pre} if int(r["recent_n_rooms"]) > 0 else pre
            cond = core.feature_vector_from_raw(
                src["melee_kills"], src["magic_kills"], src["melee_hits"], src["magic_hits"],
                src["damage_taken"], src["damage_dealt"], float(r["pre_hp_ratio"]),
                int(r["node_depth"]), float(r["progress_norm"]),
                total_kills=pre["melee_kills"] + pre["magic_kills"])
            assert [float(r[c]) for c in EXPECTED_COND] == pytest.approx(cond)
            checked += 1
    assert checked > 100


def test_no_duplicate_keys_and_controller_identity(tmp_path):
    rows = _rows(make_dataset(tmp_path / "d.csv", runs=30, controller="exploration=0.5,fixed=0.5"))
    keys = [(r["run_id"], r["room_seq"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert all(r["controller_type"] in ("exploration", "fixed") for r in rows)
    assert all(r["controller_version"] for r in rows)
    assert all(r["fallback_used"] == "0" and r["fallback_reason"] == "" for r in rows)
    assert all(r["player_profile"] and r["player_profile"] != "human" for r in rows)
    # one controller per run
    per_run = {}
    for r in rows:
        per_run.setdefault(r["run_id"], set()).add(r["controller_type"])
    assert all(len(v) == 1 for v in per_run.values())


def test_effective_config_and_boss_semantics(tmp_path):
    rows = _rows(make_dataset(tmp_path / "d.csv", runs=40, controller="exploration"))
    boss = [r for r in rows if r["room_kind"] == "boss"]
    assert boss
    for r in boss:
        assert float(r["applied_hp_mult"]) == pytest.approx(round(float(r["applied_raw_hp_mult"]) * core.BOSS_HP_MULT, 3))
        assert int(r["applied_spawn_count"]) == core.BOSS_SPAWN_N
    for r in rows:
        if r["room_kind"] != "boss":
            assert r["applied_hp_mult"] == r["applied_raw_hp_mult"]
