"""Bounded Q outputs, action-support/OOD protection, Q_loot gate and G_loot."""
import os

import pytest

torch = pytest.importorskip("torch")

import balance  # noqa: E402
import controllers as ctl  # noqa: E402
import dda_core as core  # noqa: E402
import models  # noqa: E402
import model_runtime  # noqa: E402


def test_q_outputs_bounded_by_architecture_even_for_extreme_inputs():
    torch.manual_seed(0)
    q = models.OutcomeModel()
    with torch.no_grad():
        for layer in q.net:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.mul_(50.0)
        y = models.outcomes_physical(q(torch.randn(256, core.COND_DIM) * 100, torch.rand(256, 4) * 10 - 5))
    ranges = balance.output_ranges("enemy")
    for j, (lo, hi) in enumerate(ranges):
        assert float(y[:, j].min()) >= lo and float(y[:, j].max()) <= hi
    ql = models.BoundedOutcomeNet(10, balance.output_ranges("loot")[1:])
    with torch.no_grad():
        yl = models.outcomes_physical(ql(torch.randn(64, 10) * 1e3))
    assert torch.isfinite(yl).all() and float(yl.max()) <= 2.0 and float(yl.min()) >= 0.0


def test_support_model_threshold_penalty_and_generator_box():
    torch.manual_seed(0)
    x = torch.randn(400, core.COND_DIM)
    a = torch.rand(400, 4) * 0.4 + 0.3                       # observed actions in [0.3, 0.7]
    sup = models.SupportModel.fit(x, a, k=5, quantile=0.95, max_rows=1000)
    d_in = sup.distance(x[:50], a[:50])
    assert float((d_in > sup.threshold).float().mean()) < 0.2
    far = sup.distance(x[:50], torch.ones(50, 4))
    assert float(far.min()) > sup.threshold
    assert float(sup.penalty(x[:50], torch.ones(50, 4))) > 0.0
    lo, hi = models.support_box(a, 0.01, 0.99)
    g = models.Generator(box_lo=lo.tolist(), box_hi=hi.tolist())
    out = g(torch.randn(200, core.COND_DIM) * 10)
    assert bool(((out >= lo - 1e-6) & (out <= hi + 1e-6)).all())
    restored = models.SupportModel.from_state(sup.to_state())
    assert torch.allclose(restored.distance(x[:5], a[:5]), sup.distance(x[:5], a[:5]))


class _OODModels:
    noise_dim = 8
    checkpoint_id = "ood"

    def predict(self, cond_vec, noise=None):
        raise model_runtime.OODAction("support_ratio=3.000 > 1.5")


def test_runtime_ood_action_falls_back_with_reason_without_disabling():
    c = ctl.CGANController("unused", models=_OODModels())
    ctx = ctl.DecisionContext([0.5] * 9, core.CombatMetrics(), 2, 0.9, "enemy")
    for _ in range(5):
        d = c.decide(ctx)
        assert d.fallback_used == 1 and d.fallback_reason.startswith("ood_action:")
    assert not c.disabled and c.ood_count == 5


def test_trained_runtime_reports_support_and_rejects_far_actions(trained, monkeypatch):
    lm = model_runtime.LoadedModels.load(trained["model_dir"])
    a = lm.predict([0.5, 0.5, 2.0, 2.0, 0.3, 0.8, 0.2, 0.3, 0.4])
    assert 0.0 <= a["_support_distance"] < lm.ood_factor
    lm.ood_factor = 0.0                                     # every action now counts as OOD
    with pytest.raises(model_runtime.OODAction):
        lm.predict([0.5, 0.5, 2.0, 2.0, 0.3, 0.8, 0.2, 0.3, 0.4])


def test_q_report_has_bounded_corner_ranges(trained):
    import json
    rep = json.load(open(os.path.join(trained["q_dir"], "q_report.json")))
    rng = rep["extreme_action_probe"]["corner_prediction_ranges"]
    assert set(rng) == set(balance.OUTCOME_COLUMNS)
    assert all(v["inside_physical_range"] for v in rng.values())


def test_loot_generator_outputs_valid_probabilities_and_respects_support():
    g = models.LootGenerator(21, min_prob=0.05, allowed=[True, True, False, True])
    torch.set_grad_enabled(False)
    p = g(torch.randn(100, 21) * 20)
    assert torch.isfinite(p).all() and bool((p >= 0).all())
    assert torch.allclose(p.sum(dim=1), torch.ones(100), atol=1e-5)
    assert float(p[:, 2].max()) == 0.0                       # unsupported category never proposed
    assert float(p[:, [0, 1, 3]].min()) >= 0.05 - 1e-6       # floor on supported categories
    p1, p2 = g(torch.zeros(4, 21)), g(torch.zeros(4, 21))
    assert float((p1 - p2).abs().sum()) > 0.0                # noise changes the output
    torch.set_grad_enabled(True)


@pytest.fixture(scope="module")
def loot_setup(tmp_path_factory, pilot_csv):
    import loot_dataset
    import train_q_loot
    d = tmp_path_factory.mktemp("loot")
    lcsv = str(d / "loot.csv")
    loot_dataset.derive([pilot_csv], lcsv)
    rc = train_q_loot.main(["--loot-csv", lcsv, "--out-dir", str(d / "ql"), "--epochs", "30", "--patience", "8"])
    return {"loot_csv": lcsv, "q_loot_dir": str(d / "ql"), "rc": rc, "dir": d}


def test_q_loot_reports_all_ablations_and_matched_gate(loot_setup):
    import json
    rep = json.load(open(os.path.join(loot_setup["q_loot_dir"], "q_loot_report.json")))
    assert set(rep["variants"]) == {"x", "xe", "xc", "xcie"}
    for v in rep["variants"].values():
        assert set(v["test"]["metrics"]) == set(balance.LOOT_OUTCOME_COLUMNS)
    comps = rep["gate"]["comparisons"]
    assert comps["xc"]["twin"] == "x" and comps["xcie"]["twin"] == "xe"
    assert loot_setup["rc"] == (0 if rep["gate"]["passed"] else 2)


def test_loot_policy_refuses_without_passed_gate(loot_setup):
    import train_loot_policy
    ck_path = os.path.join(loot_setup["q_loot_dir"], "q_loot_checkpoint.pt")
    ck = torch.load(ck_path, weights_only=True)
    ck["gate"] = dict(ck["gate"], passed=False, chosen_variant=None)
    failed_dir = loot_setup["dir"] / "ql_failed"
    failed_dir.mkdir()
    torch.save(ck, failed_dir / "q_loot_checkpoint.pt")
    with pytest.raises(SystemExit, match="gate FAILED"):
        train_loot_policy.main(["--loot-csv", loot_setup["loot_csv"], "--q-loot-dir", str(failed_dir),
                                "--out-dir", str(loot_setup["dir"] / "never")])
    assert not os.path.exists(loot_setup["dir"] / "never")


def test_loot_policy_mechanics_on_test_fixture_gate(loot_setup):
    """MECHANICS TEST ONLY: the gate below is forced to 'passed' on a test
    fixture to exercise the trainer and runtime; it says nothing about
    whether learned_loot is supported by the data."""
    import rewards
    import train_loot_policy
    ck = torch.load(os.path.join(loot_setup["q_loot_dir"], "q_loot_checkpoint.pt"), weights_only=True)
    ck["gate"] = dict(ck["gate"], passed=True, chosen_variant="xcie")
    forced = loot_setup["dir"] / "ql_forced_for_test"
    forced.mkdir()
    torch.save(ck, forced / "q_loot_checkpoint.pt")
    out = loot_setup["dir"] / "lm"
    train_loot_policy.main(["--loot-csv", loot_setup["loot_csv"], "--q-loot-dir", str(forced),
                            "--out-dir", str(out), "--epochs", "8"])
    lm = model_runtime.LoadedLootModel.load(str(out))
    eq = {"weapon": "weapon.sword", "spell": "spell.magic_bolt", "armor": "armor.cloth_wrap", "boots": "boots.leather"}
    p = lm.predict([0.5] * 9, eq, noise=[0.0] * lm.noise_dim)
    assert len(p) == 4 and abs(sum(p) - 1) < 1e-4 and min(p) >= 0
    pol = rewards.LearnedLootPolicy(str(out))
    pol.begin_run(1)
    d = pol.propose(rewards.LootContext([0.5] * 9, core.CombatMetrics(), eq, 1, 0.9))
    assert d.fallback_used == 0 and d.checkpoint_id == lm.checkpoint_id
    # incompatible checkpoint -> logged fallback
    bad = torch.load(out / "loot_policy_checkpoint.pt", weights_only=True)
    bad["item_ids"] = list(reversed(bad["item_ids"]))
    (loot_setup["dir"] / "lm_bad").mkdir()
    torch.save(bad, loot_setup["dir"] / "lm_bad" / "loot_policy_checkpoint.pt")
    d = rewards.LearnedLootPolicy(str(loot_setup["dir"] / "lm_bad")).propose(
        rewards.LootContext([0.5] * 9, core.CombatMetrics(), eq, 1, 0.9))
    assert d.fallback_used == 1 and "ModelVersionError" in d.fallback_reason
