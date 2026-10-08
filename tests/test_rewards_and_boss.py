"""Reward policies, loot telemetry, the shared boss policy and room-kind dispatch."""
import csv
import math

import pytest

import balance
import bot_runner
import controllers as ctl
import dda_core as core
import rewards
import telemetry
from conftest import make_dataset

CTX = rewards.LootContext([0.5] * 9, core.CombatMetrics(melee_kills=5, magic_kills=1),
                          {"weapon": "weapon.sword", "spell": "spell.magic_bolt", "armor": "armor.cloth_wrap",
                           "boots": "boots.leather"}, 2, 0.8)


@pytest.mark.parametrize("name", ["fixed_uniform", "heuristic", "exploration"])
def test_reward_policies_propose_valid_distributions(name):
    pol = rewards.make_reward_policy(name)
    pol.begin_run(3)
    for _ in range(50):
        d = pol.propose(CTX)
        assert len(d.probs) == 4 and all(p >= 0 and math.isfinite(p) for p in d.probs)
        assert sum(d.probs) == pytest.approx(1.0)
        assert d.policy_type == name and d.fallback_used == 0


def test_fixed_uniform_and_heuristic_rules():
    assert rewards.make_reward_policy("fixed_uniform").propose(CTX).probs == [0.25] * 4
    h = rewards.make_reward_policy("heuristic").propose(CTX).probs
    assert h == pytest.approx(rewards.normalize_probs(core.loot_bias_for_metrics(CTX.metrics)))


def test_exploration_reward_covers_all_categories_and_is_seeded():
    pol = rewards.make_reward_policy("exploration")
    pol.begin_run(11)
    a = [pol.propose(CTX).probs for _ in range(400)]
    pol.begin_run(11)
    assert [pol.propose(CTX).probs for _ in range(400)] == a
    focus = {max(range(4), key=lambda i: p[i]) for p in a if max(p) > 0.5}
    assert focus == {0, 1, 2, 3}


def test_learned_loot_without_model_falls_back_and_logs_reason():
    pol = rewards.make_reward_policy("learned_loot", loot_model_dir="/nonexistent")
    d = pol.propose(CTX)
    assert d.fallback_used == 1 and d.fallback_reason.startswith("load_error")
    assert d.policy_type == "learned_loot" and d.proposal_source == "heuristic_fallback"
    assert d.probs == pytest.approx(rewards.make_reward_policy("heuristic").propose(CTX).probs)


def test_loot_events_logged_and_items_from_fixed_pool(tmp_path):
    p = make_dataset(tmp_path / "r.csv", runs=80, controller="heuristic", extra=("--reward-policy", "exploration"))
    rows = list(csv.DictReader(open(p)))
    loot = [r for r in rows if r["room_kind"] == "loot"]
    assert loot and all(r["loot_event_id"] == f"{r['run_id']}:{r['room_seq']}" for r in loot)
    for r in loot:
        for i in (1, 2, 3):
            assert r[f"offer_{i}_item_id"] in core.ITEM_BY_ID
            assert core.SLOT_OF_ITEM[r[f"offer_{i}_item_id"]] == r[f"offer_{i}_category"]
        assert r["selection_mechanism"] == "bot_uniform_offer_pick_v1"
        assert r["reward_proposal_source"]
        if r["selected_offer_index"] != "-1":
            assert r["equip_after_" + r["selected_category"]] == r["selected_item_id"]
            before = r["equip_" + r["selected_category"]]
            assert r["replaced_existing"] == str(int(before != r["selected_item_id"]))
    assert all(r["loot_event_id"] == "" for r in rows if r["room_kind"] != "loot")
    assert {r["selected_category"] for r in loot} >= {"weapon", "spell", "armor", "boots"}


def test_enemy_controller_and_reward_policy_are_independent(tmp_path):
    a = make_dataset(tmp_path / "a.csv", runs=20, controller="fixed", extra=("--reward-policy", "fixed_uniform"))
    b = make_dataset(tmp_path / "b.csv", runs=20, controller="fixed", extra=("--reward-policy", "heuristic"))
    ra, rb = list(csv.DictReader(open(a))), list(csv.DictReader(open(b)))
    assert {r["reward_policy"] for r in ra} == {"fixed_uniform"} and {r["reward_policy"] for r in rb} == {"heuristic"}
    assert {r["enemy_controller"] for r in ra + rb} == {"fixed"}
    # the first room of every run (before any loot) is identical: same seeds, maps, loadouts
    fa = {r["run_seed"]: r for r in ra if r["room_seq"] == "1"}
    fb = {r["run_seed"]: r for r in rb if r["room_seq"] == "1"}
    for s in fa:
        assert fa[s]["room_kind"] == fb[s]["room_kind"]
        assert all(fa[s]["equip_" + x] == fb[s]["equip_" + x] for x in core.SLOTS)


class _CountingModels:
    noise_dim = 8
    checkpoint_id = "counting"

    def __init__(self):
        self.calls = 0

    def predict(self, cond_vec, noise=None):
        self.calls += 1
        return {"enemy_hp_mult": 1.0, "enemy_dmg_mult": 1.0, "enemy_speed_mult": 1.0, "spawn_count": 2}


def test_shared_boss_policy_identical_and_no_model_call_outside_ordinary_rooms():
    sb = balance.load_experiment_config()["shared_boss"]
    for kind in ("boss", "loot", "nothing"):
        stub = _CountingModels()
        c = ctl.CGANController("unused", models=stub)
        d = ctl.decide_room(c, ctl.DecisionContext([0.5] * 9, core.CombatMetrics(), 3, 0.9, kind))
        assert stub.calls == 0
        if kind == "boss":
            assert d.controller_type == "shared_boss" and d.action_source == "shared_boss"
            assert d.controller_version == sb["version"]
            assert {k: d.action[k] for k in core.ACTION_KEYS} == core.finalize_action(sb["raw_action"])
        else:
            assert d.action_source == "non_combat"
    stub = _CountingModels()
    ctl.decide_room(ctl.CGANController("unused", models=stub),
                    ctl.DecisionContext([0.5] * 9, core.CombatMetrics(), 3, 0.9, "enemy"))
    assert stub.calls == 1


def test_boss_rows_equal_across_controllers_for_equal_states(tmp_path):
    rows = {}
    for c in ("fixed", "heuristic", "exploration"):
        p = make_dataset(tmp_path / f"{c}.csv", runs=40, controller=c)
        rows[c] = [r for r in csv.DictReader(open(p)) if r["room_kind"] == "boss"]
    for c, rs in rows.items():
        assert rs and all(r["controller_type"] == "shared_boss" for r in rs)
        assert len({(r["applied_hp_mult"], r["applied_dmg_mult"], r["applied_speed_mult"],
                     r["applied_spawn_count"]) for r in rs}) == 1
    assert len({rs[0]["applied_hp_mult"] for rs in rows.values()}) == 1


def test_boss_row_with_other_controller_rejected(tmp_path):
    import shutil
    src = make_dataset(tmp_path / "s.csv", runs=10)
    lines = open(src).read().splitlines()
    header = lines[0].split(",")
    for i, line in enumerate(lines[1:], start=1):
        cells = line.split(",")
        if cells[header.index("room_kind")] == "boss":
            cells[header.index("controller_type")] = "heuristic"
            lines[i] = ",".join(cells)
            break
    open(src, "w").write("\n".join(lines) + "\n")
    with pytest.raises(telemetry.DatasetVersionError, match="shared_boss"):
        telemetry.validate_dataset(src, src[:-4] + ".manifest.json")


def test_enemy_training_rows_exclude_boss(tmp_path):
    import ml_data
    p = make_dataset(tmp_path / "t.csv", runs=40, controller="exploration")
    rows, _ = ml_data.load_training_rows([p])
    assert rows and all(r["room_kind"] == "enemy" for r in rows)
