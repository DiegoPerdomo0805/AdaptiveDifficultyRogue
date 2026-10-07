import csv
import statistics

import dda_core as core
import controllers as ctl
from conftest import make_dataset


def test_exploration_is_deterministic_and_worker_independent(tmp_path):
    a = make_dataset(tmp_path / "a.csv", runs=40, controller="exploration")
    b = make_dataset(tmp_path / "b.csv", runs=40, controller="exploration")
    c = make_dataset(tmp_path / "c.csv", runs=40, controller="exploration", extra=("--workers", "3"))
    ta, tb, tc = (open(p).read() for p in (a, b, c))
    assert ta == tb == tc
    d = make_dataset(tmp_path / "d.csv", runs=40, seed=200000, controller="exploration")
    assert open(d).read() != ta


def test_exploration_bounds_variation_and_branches(tmp_path):
    with open(make_dataset(tmp_path / "e.csv", runs=150, controller="exploration"), newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["is_combat_room"] == "1"]
    cols = ["applied_raw_hp_mult", "applied_raw_dmg_mult", "applied_raw_speed_mult", "applied_raw_spawn_count"]
    for k, c in zip(core.ACTION_KEYS, cols):
        lo, hi = core.ACTION_BOUNDS[k]
        vals = [float(r[c]) for r in rows]
        assert all(lo <= v <= hi for v in vals)
        assert statistics.pstdev(vals) / (hi - lo) > 0.08, c
    assert all(float(r["applied_raw_spawn_count"]).is_integer() for r in rows)
    assert {r["action_source"] for r in rows} == {"perturb", "uniform", "reference"}


def test_finalize_action_rejects_nonfinite_and_clamps():
    import pytest
    out = core.finalize_action({"enemy_hp_mult": 99, "enemy_dmg_mult": -1, "enemy_speed_mult": 1.2345,
                                "spawn_count": 4.6})
    assert out["enemy_hp_mult"] == core.ACTION_BOUNDS["enemy_hp_mult"][1]
    assert out["enemy_dmg_mult"] == core.ACTION_BOUNDS["enemy_dmg_mult"][0]
    assert out["enemy_speed_mult"] == 1.234 or out["enemy_speed_mult"] == 1.235
    assert out["spawn_count"] == 5 and isinstance(out["spawn_count"], int)
    with pytest.raises(core.ActionValidationError):
        core.finalize_action({"enemy_hp_mult": float("nan"), "enemy_dmg_mult": 1, "enemy_speed_mult": 1,
                              "spawn_count": 3})


def test_controller_reseeds_per_run():
    c = ctl.make_controller("exploration")
    ctx = ctl.DecisionContext([0.5] * 9, core.CombatMetrics(), 2, 0.9, "enemy")
    c.begin_run(7)
    a = [c.decide(ctx).action for _ in range(5)]
    c.begin_run(7)
    b = [c.decide(ctx).action for _ in range(5)]
    assert a == b
