"""Closed-loop checkpoint selection, seed partitions and the freeze contract."""
import copy
import json
import os

import pytest

import balance
import freeze_experiment
import selection

CFG = balance.load_experiment_config()
CL = CFG["cgan_selection"]["closed_loop"]
GOOD_DIAG = {"validity": 1.0, "diversity": 0.1, "sensitivity": 0.05}


def _metrics(**kw):
    m = {"validity": 1.0, "ordinary_room_death_rate": 0.02, "all_fallbacks_recorded_ood": True,
         "fallback_reasons": {}, "cgan_ood_rate": 0.10, "heuristic_same_state_ood_rate": 0.06,
         "ood_excess": 0.04, "runtime_guard_rate": 0.0, "fallback_rate": 0.0,
         "mean_B_r": 0.2, "cvar_B_r": 1.0}
    m.update(kw)
    m["selection_score"] = selection.selection_score(m["mean_B_r"], m["cvar_B_r"], CL["cvar_coef"])
    return m


def test_cvar_090_definition():
    assert selection.cvar_upper(list(range(1, 11)), 0.90) == 10                    # worst 1 of 10
    assert selection.cvar_upper(list(range(1, 21)), 0.90) == pytest.approx(19.5)   # worst 2 of 20
    assert selection.cvar_upper(list(range(1, 16)), 0.90) == pytest.approx(14.5)   # ceil(1.5) = 2
    assert selection.cvar_upper([0.0] * 9 + [7.0], 0.90) == 7.0
    with pytest.raises(ValueError):
        selection.cvar_upper([], 0.9)


def test_exact_selection_score():
    assert CL["cvar_coef"] == 0.5 and CL["cvar_alpha"] == 0.90
    assert selection.selection_score(0.3, 2.0, CL["cvar_coef"]) == pytest.approx(1.3)


def test_rejects_death_rate_above_cap():
    assert selection.eligibility(_metrics(), GOOD_DIAG, 20, CFG) == []
    r = selection.eligibility(_metrics(ordinary_room_death_rate=0.081), GOOD_DIAG, 20, CFG)
    assert any("death rate" in x for x in r)


@pytest.mark.parametrize("diag", [dict(GOOD_DIAG, diversity=0.0), dict(GOOD_DIAG, sensitivity=0.0),
                                  dict(GOOD_DIAG, validity=0.99)])
def test_rejects_room_level_gate_failures(diag):
    assert selection.eligibility(_metrics(), diag, 20, CFG)


def test_rejects_unrecorded_fallbacks_and_load_errors():
    assert selection.eligibility(_metrics(all_fallbacks_recorded_ood=False,
                                          fallback_reasons={"invalid_output:x": 1}), GOOD_DIAG, 20, CFG)
    assert selection.eligibility({"load_error": "ModelVersionError: x", "selection_score": float("nan")},
                                 GOOD_DIAG, 20, CFG)


def test_selects_lowest_scoring_eligible_and_fails_explicitly():
    cands = [{"epoch": 10, "metrics": _metrics(mean_B_r=0.1, cvar_B_r=0.2), "ineligible_reasons": ["x"]},
             {"epoch": 15, "metrics": _metrics(mean_B_r=0.3, cvar_B_r=0.4), "ineligible_reasons": []},
             {"epoch": 20, "metrics": _metrics(mean_B_r=0.2, cvar_B_r=0.4), "ineligible_reasons": []}]
    assert selection.choose(cands)["epoch"] == 20          # lowest score among ELIGIBLE (10 is ineligible)
    for c in cands:
        c["ineligible_reasons"] = ["death"]
    with pytest.raises(selection.SelectionFailure, match="no eligible checkpoint"):
        selection.choose(cands)


def test_seed_partitions_do_not_overlap():
    b = CFG["seed_blocks"]
    blocks = {k: v for k, v in b.items() if isinstance(v, list)}
    names = sorted(blocks)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            assert blocks[x][1] < blocks[y][0] or blocks[y][1] < blocks[x][0], (x, y)
    seeds = selection.validation_seeds(CFG)
    assert len(seeds) == CL["n_runs"]
    assert all(b["selection_validation"][0] <= s <= b["selection_validation"][1] for s in seeds)
    bad = copy.deepcopy(CFG)
    bad["seed_blocks"]["selection_validation"] = [9500000, 9500100]          # overlaps final_test
    with pytest.raises(balance.ExperimentConfigError, match="overlap"):
        balance._validate_config(bad)


def test_selection_never_uses_test_seeds():
    for block in ("final_test", "smoke_test", "train_collection"):
        lo = CFG["seed_blocks"][block][0]
        with pytest.raises(selection.SelectionFailure):
            selection.assert_not_test_seeds([lo], CFG)
        with pytest.raises(selection.SelectionFailure):
            selection.rollout("unused", [lo], CFG)
    import evaluate_controllers
    with pytest.raises(SystemExit):        # evaluation refuses validation seeds
        evaluate_controllers.check_seeds(CFG["seed_blocks"]["selection_validation"][0], 10, [], CFG, True,
                                         "final_test")


def test_freeze_hash_changes_when_cgan_selection_changes():
    h0 = balance.experiment_config_hash(CFG)
    alt = copy.deepcopy(CFG)
    alt["cgan_selection"]["closed_loop"]["cvar_coef"] = 0.6
    assert balance.experiment_config_hash(alt) != h0
    alt2 = copy.deepcopy(CFG)
    alt2["cgan_selection"]["w_jitter"] = 0.3
    assert balance.experiment_config_hash(alt2) != h0
    c = freeze_experiment.current_contract()
    assert c["cgan_selection"] == CFG["cgan_selection"] and c["experiment_config_hash"] == h0


def test_closed_loop_rollouts_are_deterministic(trained):
    seeds = selection.validation_seeds(CFG)[:8]
    lm1, r1 = selection.rollout(trained["model_dir"], seeds, CFG)
    lm2, r2 = selection.rollout(trained["model_dir"], seeds, CFG)
    m1, m2 = selection.rollout_metrics(r1, CFG, lm1), selection.rollout_metrics(r2, CFG, lm2)
    # NaN-safe equality (pearson r is NaN when a sample has no deaths)
    assert json.dumps(m1, sort_keys=True, default=str) == json.dumps(m2, sort_keys=True, default=str)
    assert r1 == r2
    assert set(m1["q_diagnostics"]) == {"n_runs", "run_death_rate", "B_r"}


def test_trained_selection_report_uses_validation_seeds_only(trained):
    rep = json.load(open(os.path.join(trained["model_dir"], "cgan_report.json")))
    lo, hi = rep["validation_seeds"]
    vb = CFG["seed_blocks"]["selection_validation"]
    assert vb[0] <= lo <= hi <= vb[1]
    assert rep["selection"] == "OK" and rep["candidates"]
    chosen = rep["selected"]
    eligible = [c for c in rep["candidates"] if not c["ineligible_reasons"]]
    assert chosen["metrics"]["selection_score"] == min(c["metrics"]["selection_score"] for c in eligible)
    for c in rep["candidates"]:
        if "mean_B_r" in c["metrics"]:
            assert c["metrics"]["selection_score"] == pytest.approx(
                c["metrics"]["mean_B_r"] + 0.5 * c["metrics"]["cvar_B_r"])


# ---------------------------------------------------------------------------
# Action-relative OOD, extreme-support guard and fallback gates
# ---------------------------------------------------------------------------

def test_gate_boundaries_are_inclusive():
    assert CL["max_ood_excess"] == 0.05 and CL["max_runtime_guard_rate"] == 0.01 and CL["max_fallback_rate"] == 0.01
    assert selection.eligibility(_metrics(ood_excess=0.05), GOOD_DIAG, 20, CFG) == []
    assert selection.eligibility(_metrics(ood_excess=0.0501), GOOD_DIAG, 20, CFG)
    assert selection.eligibility(_metrics(runtime_guard_rate=0.01), GOOD_DIAG, 20, CFG) == []
    assert selection.eligibility(_metrics(runtime_guard_rate=0.0101), GOOD_DIAG, 20, CFG)
    assert selection.eligibility(_metrics(fallback_rate=0.01), GOOD_DIAG, 20, CFG) == []
    assert selection.eligibility(_metrics(fallback_rate=0.0101), GOOD_DIAG, 20, CFG)


def test_absolute_ood_is_diagnostic_not_a_gate():
    assert "max_ood_rate" not in CL
    m = _metrics(cgan_ood_rate=0.60, heuristic_same_state_ood_rate=0.58, ood_excess=0.02)
    assert selection.eligibility(m, GOOD_DIAG, 20, CFG) == []


def test_negative_excess_allowed_and_not_clipped():
    assert selection.eligibility(_metrics(ood_excess=-0.03), GOOD_DIAG, 20, CFG) == []

    class _Sup:
        threshold = 1.0

        def distance(self, x, a):
            import torch
            return torch.full((x.shape[0],), 2.0)        # every heuristic counterfactual is OOD

    class _LM:
        support = _Sup()
        cond_scaler = __import__("ml_data").Standardizer([0.0] * 9, [1.0] * 9, __import__("ml_data").COND_COLUMNS)

    rows = []
    for i in range(4):
        r = {c: "0.5" for c in __import__("ml_data").COND_COLUMNS}
        r.update({"pre_melee_kills": "1", "pre_magic_kills": "0", "node_depth": "1", "pre_hp_ratio": "0.9",
                  "fallback_used": "0", "fallback_reason": "", "support_distance": "0.5000"})
        rows.append(r)
    d = selection.ood_decomposition(_LM(), rows, CFG)
    assert d["cgan_ood_rate"] == 0.0 and d["heuristic_same_state_ood_rate"] == 1.0
    assert d["ood_excess"] == -1.0 and d["runtime_guard_rate"] == 0.0


def test_runtime_guard_rate_uses_rejected_generated_actions():
    rows = [{"fallback_used": "1", "fallback_reason": "ood_action:support_ratio=1.575 > 1.5", "support_distance": ""},
            {"fallback_used": "0", "fallback_reason": "", "support_distance": "1.2000"}]
    assert selection._ratio_from_reason(rows[0]["fallback_reason"]) == pytest.approx(1.575)


def test_heuristic_counterfactual_equals_the_heuristic_decision_on_that_state(tmp_path):
    """The counterfactual action for a state equals what rule_based_dda
    actually chose in that exact state (checked on real heuristic rollouts)."""
    import csv
    import ml_data
    from conftest import make_dataset
    p = make_dataset(tmp_path / "h.csv", runs=30, controller="heuristic")
    rows = [r for r in csv.DictReader(open(p)) if r["room_kind"] == "enemy"]
    for r in rows:
        assert ml_data.heuristic_action_for_row(r) == pytest.approx([float(r[c]) for c in ml_data.ACTION_COLUMNS])


def test_counterfactual_uses_identical_states_and_never_alters_rollouts(trained):
    import torch
    import ml_data
    seeds = selection.validation_seeds(CFG)[:6]
    lm, runs = selection.rollout(trained["model_dir"], seeds, CFG)
    snapshot = copy.deepcopy(runs)
    rows = [r for x in runs for r in x["rows"] if r["room_kind"] == "enemy"]
    x = selection.state_tensor(lm, rows)
    manual = torch.tensor(lm.cond_scaler.transform([[float(r[c]) for c in ml_data.COND_COLUMNS] for r in rows]),
                          dtype=torch.float32)
    assert torch.equal(x, manual)                                   # the state IS the logged condition
    cg, he = selection.same_state_support(lm, rows)
    assert he == selection.support_ratios(lm, manual, [ml_data.heuristic_action_for_row(r) for r in rows])
    for r, v in zip(rows, cg):                                      # cGAN side = runtime-evaluated ratio
        if r["fallback_used"] == 0 or r["fallback_used"] == "0":
            assert v == float(r["support_distance"])
    m = selection.rollout_metrics(runs, CFG, lm)                    # computes the counterfactual
    assert runs == snapshot                                         # rows untouched
    lm2, runs2 = selection.rollout(trained["model_dir"], seeds, CFG)
    assert runs2 == snapshot                                        # RNG / rollout unaffected
    for k in ("cgan_ood_rate", "heuristic_same_state_ood_rate", "ood_excess", "runtime_guard_rate",
              "boss_survival", "fallback_rate"):
        assert k in m
