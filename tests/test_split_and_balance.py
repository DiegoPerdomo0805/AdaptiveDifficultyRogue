import math

import pytest

import balance
import ml_data


def test_grouped_split_disjoint_complete_deterministic():
    rows = [{"run_id": f"r{i}", "room_seq": str(j)} for i in range(50) for j in range(4)]
    s1 = ml_data.grouped_split(rows, (0.7, 0.15, 0.15), seed=3)
    s2 = ml_data.grouped_split(rows, (0.7, 0.15, 0.15), seed=3)
    assert s1 == s2
    tr, va, te = (set(s1[k]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == {f"r{i}" for i in range(50)}
    assert len(va) == 8 and len(te) == 8
    assert ml_data.grouped_split(rows, (0.7, 0.15, 0.15), seed=4) != s1


def _row(**kw):
    r = {"died_in_room": 0, "post_hp_ratio": 0.6, "damage_taken_in_room": 20.0,
         "damage_dealt_in_room": 100.0, "melee_kills_in_room": 2, "magic_kills_in_room": 1,
         "melee_hits_in_room": 6, "magic_hits_in_room": 4, "time_in_room": 6.0}
    r.update(kw)
    return r


def test_outcome_vector_definitions_and_edge_cases():
    y = balance.outcome_vector(_row())
    assert y == pytest.approx([0.0, 0.6, 20.0 / 120.0, 0.3, 0.2])
    # zero damage dealt and taken -> deterministic 0; zero dealt -> 1; bounded
    assert balance.outcome_vector(_row(damage_taken_in_room=0.0, damage_dealt_in_room=0.0))[2] == 0.0
    assert balance.outcome_vector(_row(damage_taken_in_room=3.0, damage_dealt_in_room=0.0))[2] == 1.0
    assert balance.outcome_vector(_row(damage_taken_in_room=0.0, damage_dealt_in_room=50.0))[2] == 0.0
    # zero hits -> efficiency 0; death forces post hp 0
    assert balance.outcome_vector(_row(melee_hits_in_room=0, magic_hits_in_room=0,
                                       melee_kills_in_room=0, magic_kills_in_room=0))[3] == 0.0
    d = balance.outcome_vector(_row(died_in_room=1, post_hp_ratio=0.4))
    assert d[0] == 1.0 and d[1] == 0.0
    assert all(math.isfinite(v) for v in d)


def test_balance_loss_zero_inside_and_grows_outside():
    low, high, _, _ = balance.target_arrays()
    inside = [(lo + hi) / 2 for lo, hi in zip(low, high)]
    assert balance.balance_distance(inside) == 0.0
    prev = 0.0
    for delta in (0.01, 0.05, 0.2, 0.5):
        y = list(inside)
        y[1] = high[1] + delta
        d = balance.balance_distance(y)
        assert d > prev
        prev = d
    y = list(inside)
    y[1] = low[1] - 0.1
    assert balance.balance_distance(y) > 0


def test_expected_balance_distance_is_bernoulli_mixture():
    y = [0.25, 0.6, 0.15, 0.3, 0.2]
    e = balance.expected_balance_distance(y)
    assert e == pytest.approx(0.25 * balance.balance_distance([1.0] + y[1:])
                              + 0.75 * balance.balance_distance([0.0] + y[1:]))
    assert e > balance.balance_distance(y)          # convex loss: E[L] >= L(E)


def test_balance_loss_torch_matches_python():
    torch = pytest.importorskip("torch")
    ys = [[0.3, 0.9, 0.05, 0.6, 0.7], [0.0, 0.5, 0.2, 0.3, 0.3], [1.0, 0.0, 2.0, 0.0, 1.5]]
    t = balance.balance_loss_torch(torch.tensor(ys), reduce=False)
    assert t.tolist() == pytest.approx([balance.balance_distance(y) for y in ys], rel=1e-5)


def test_experiment_config_targets_not_in_model_code():
    import inspect
    models = pytest.importorskip("models")
    src = inspect.getsource(models)
    cfg = balance.load_experiment_config()
    for col, t in {**cfg["enemy_balance_targets"], **cfg["loot_horizon_balance_targets"]}.items():
        assert f'"{col}": {{"low"' not in src


def test_duration_norm_is_bounded_normalized():
    r = _row(time_in_room=500.0)
    assert balance.outcome_vector(r)[4] == 1.0
    assert balance.duration_norm(15.0) == pytest.approx(0.5)
    assert all(rng == (0.0, 1.0) for rng in balance.output_ranges("enemy"))
    assert all(rng == (0.0, 1.0) for rng in balance.output_ranges("loot"))


def test_target_audit_analytic_properties():
    for fam in ("enemy", "loot"):
        a = balance.target_audit(family=fam)
        assert a["analytic_ok"]
        for c in a["components"].values():
            assert c["zero_inside"] and c["c1_at_bounds"] and c["one_scale_identity_ok"]
            assert c["positive_above"] in (True, None) and c["increasing_above"] in (True, None)
            assert c["positive_below"] in (True, None) and c["increasing_below"] in (True, None)
        assert sum(c["normalized_weight"] for c in a["components"].values()) == pytest.approx(1.0)


def test_design_intent_enemy_targets_are_the_frozen_values():
    t = balance.load_experiment_config()["enemy_balance_targets"]
    assert {k: (v["low"], v["high"]) for k, v in t.items()} == {
        "death": (0.0, 0.08), "post_hp_ratio": (0.35, 0.85), "damage_share": (0.09, 0.31),
        "combat_efficiency": (0.15, 0.50), "duration_norm": (0.10, 0.50)}


def test_fixed_default_is_the_documented_neutral_configuration():
    import controllers
    import dda_core as core
    fx = balance.load_experiment_config()["enemy_controller"]["fixed"]
    assert fx["name"] == "fixed_default"
    assert fx["action"] == {"enemy_hp_mult": 1.0, "enemy_dmg_mult": 1.0, "enemy_speed_mult": 1.0,
                            "spawn_count": core.DDA_SPAWN_BASE}
    d = controllers.make_controller("fixed").decide(
        controllers.DecisionContext([0.5] * 9, core.CombatMetrics(melee_kills=40), 7, 0.1, "enemy"))
    assert {k: d.action[k] for k in core.ACTION_KEYS} == fx["action"]     # state-independent


def test_principal_collection_mixture_excludes_learned_loot():
    import bot_runner
    pc = balance.load_experiment_config()["principal_collection"]
    rmix = bot_runner.parse_mix_spec(pc["reward_policy_mix"], ("fixed_uniform", "heuristic", "exploration",
                                                               "learned_loot"))
    assert set(rmix) == {"fixed_uniform", "heuristic", "exploration"} and pc["learned_loot_allowed"] is False


def test_pseudo_huber_capped_scale_shape():
    s_lo, s_hi = balance.side_scales(family="enemy")
    cfg = balance.load_experiment_config()
    t = cfg["enemy_balance_targets"]
    # capped side scales: death cannot go below 0; duration below 0.10 gets s = 0.10
    assert s_lo[0] == 0.0 and s_hi[0] == pytest.approx(0.10)
    assert s_lo[4] == pytest.approx(0.10) and s_hi[1] == pytest.approx(0.15)
    mid = [(t[c]["low"] + t[c]["high"]) / 2 for c in balance.OUTCOME_COLUMNS]
    # quadratic near the bound, ~linear far away: increments become constant
    def L(v):
        y = list(mid)
        y[0] = v
        return balance.balance_distance(y)
    inc = [L(0.08 + 0.1 * (k + 1)) - L(0.08 + 0.1 * k) for k in range(9)]
    assert all(b > a for a, b in zip(inc, inc[1:]))           # convex, increasing
    assert inc[-1] / inc[-2] < 1.02                          # asymptotically linear
    assert L(1.0) < 8.0                                      # no quadratic blow-up (was 36.3)


def test_run_level_vector_uses_summed_damage_and_efficiency():
    import analyze_results
    rows = [_row(died_in_room=0, post_hp_ratio=0.8, damage_taken_in_room=10, damage_dealt_in_room=90,
                 melee_kills_in_room=2, magic_kills_in_room=0, melee_hits_in_room=4, magic_hits_in_room=0,
                 time_in_room=6),
            _row(died_in_room=1, post_hp_ratio=0.0, damage_taken_in_room=50, damage_dealt_in_room=10,
                 melee_kills_in_room=0, magic_kills_in_room=0, melee_hits_in_room=6, magic_hits_in_room=0,
                 time_in_room=12)]
    v = analyze_results.run_level_vector(rows, balance.load_experiment_config())
    assert v[0] == pytest.approx(0.5)                         # death rate
    assert v[1] == pytest.approx(0.4)                         # mean post HP (fatal room = 0)
    assert v[2] == pytest.approx(60 / 160)                    # summed share, NOT mean of ratios
    assert v[3] == pytest.approx(2 / 10)                      # summed kills / summed hits
    assert v[4] == pytest.approx((6 / 30 + 12 / 30) / 2)


def test_primary_stats_and_paired_differences():
    import analyze_results
    S = [{"run_seed": str(i), "run_balance": float(i), "run_vector": [0.0] * 5} for i in range(10)]
    S.append({"run_seed": "x", "run_balance": None, "run_vector": None})
    p = analyze_results.primary_stats(S, 200, 0)
    assert p["n_runs_with_ordinary_rooms"] == 10 and p["n_runs_excluded"] == 1
    assert p["mean"] == pytest.approx(4.5) and p["median"] == pytest.approx(4.5)
    assert p["iqr"] == pytest.approx(p["q75"] - p["q25"]) and p["mean_ci"][0] <= 4.5 <= p["mean_ci"][1]
    base = [dict(s, run_balance=(s["run_balance"] + 1.0) if s["run_balance"] is not None else None) for s in S]
    d = analyze_results.primary_paired(S, base, 200, 0)
    assert d["n_pairs"] == 10 and d["mean_diff"] == pytest.approx(-1.0) and d["share_lower_balance_loss"] == 1.0
