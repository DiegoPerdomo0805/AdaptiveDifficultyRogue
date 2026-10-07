import inspect
import json
import math
import os

import pytest

torch = pytest.importorskip("torch")

import balance  # noqa: E402
import controllers as ctl  # noqa: E402
import dda_core as core  # noqa: E402
import ml_data  # noqa: E402
import models  # noqa: E402
import model_runtime  # noqa: E402
import train_cgan  # noqa: E402


def test_q_shapes_and_finite_loss():
    q = models.OutcomeModel()
    x = torch.randn(16, core.COND_DIM)
    a = torch.rand(16, core.ACTION_DIM)
    out = q(x, a)
    assert out.shape == (16, balance.OUTCOME_DIM)
    loss, bce, mse = models.q_loss(out, torch.randint(0, 2, (16,)).float(), torch.randn(16, 4))
    assert torch.isfinite(loss)
    loss.backward()
    qx = models.OutcomeModel(use_action=False)
    assert qx(x).shape == (16, balance.OUTCOME_DIM)


def test_generator_objective_has_no_reconstruction_term():
    params = list(inspect.signature(models.generator_loss).parameters)
    assert params == ["gen", "disc", "q", "out_scaler", "x_s", "lambda_balance",
                      "lambda_diversity", "diversity_tau", "exp_cfg"]
    q = models.freeze(models.OutcomeModel())
    terms = models.generator_loss(models.Generator(), models.Discriminator(), q,
                                  models.OutcomeScaler([0.5] * 4, [0.2] * 4),
                                  torch.randn(8, core.COND_DIM), 1.0, 1.0, 0.05)
    assert set(terms) == {"total", *models.GENERATOR_LOSS_TERMS}
    assert set(models.GENERATOR_LOSS_TERMS) == {"adv", "balance", "diversity"}
    src = inspect.getsource(train_cgan)
    assert "mse_loss" not in src and "l1_loss" not in src
    # frozen Q receives no gradient
    terms["total"].backward()
    assert all(p.grad is None for p in q.parameters())


def test_generator_bounds_noise_sensitivity_and_diversity():
    torch.manual_seed(0)
    g = models.Generator()
    x = torch.randn(64, core.COND_DIM)
    a1, a2 = g(x), g(x)
    assert ((a1 > 0) & (a1 < 1)).all()
    raw = models.decode_actions(a1)
    lo, hi = torch.tensor(ml_data.ACTION_LO), torch.tensor(ml_data.ACTION_HI)
    assert ((raw >= lo) & (raw <= hi)).all()
    assert (a1 - a2).abs().mean() > 0           # noise changes the action
    assert models.diversity_loss(a1, a1, torch.zeros(64, 8), torch.ones(64, 8), 0.05) > 0   # collapse penalized
    assert models.diversity_loss(a1, 1 - a1, torch.zeros(64, 8), torch.ones(64, 8), 1e-4) == 0


def test_trained_q_report_and_generator_diagnostics(trained):
    rep = json.load(open(os.path.join(trained["q_dir"], "q_report.json")))
    for part in ("val", "test"):
        for model in ("q_xa", "q_x"):
            for col in balance.OUTCOME_COLUMNS:
                assert col in rep[part][model]
    assert "passed" in rep["gate"] and "action_gain" in rep["gate"]
    d = json.load(open(os.path.join(trained["model_dir"], "cgan_report.json")))["best_diagnostics"]
    assert d["validity"] == 1.0
    assert 0.0 < d["diversity"] < 1.0
    assert d["sensitivity"] > 0.0
    assert math.isfinite(d["balance_pred"])


def test_checkpoint_contents(trained):
    ck = torch.load(os.path.join(trained["model_dir"], "cgan_checkpoint.pt"), weights_only=True)
    for key in ("generator_state_dict", "discriminator_state_dict", "q_state_dict", "optimizer_g_state",
                "optimizer_d_state", "epoch", "best_score", "seed", "hparams", "cond_columns",
                "action_columns", "outcome_columns", "cond_scaler", "outcome_scaler", "history",
                "config_hash", "experiment_config_hash", "dda_version", "dataset_schema_version"):
        assert key in ck, key


def _save_variant(trained, tmp_path, mutate):
    ck = torch.load(os.path.join(trained["model_dir"], "cgan_checkpoint.pt"), weights_only=True)
    mutate(ck)
    d = tmp_path / "variant"
    d.mkdir()
    torch.save(ck, d / "cgan_checkpoint.pt")
    return str(d)


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(checkpoint_schema_version="cgan_v2_checkpoint_1"),
    lambda c: c.update(dataset_schema_version="room_samples_v2.1"),
    lambda c: c.update(dda_version="rule_based_dda_v1.1"),
    lambda c: c.update(cond_dim=6),
    lambda c: c.update(cond_columns=list(reversed(c["cond_columns"]))),
    lambda c: c.update(action_columns=c["action_columns"][1:] + c["action_columns"][:1]),
    lambda c: c.update(experiment_config_hash="0000"),
    lambda c: c.update(config_hash="deadbeefdeadbeef"),
    lambda c: c["cond_scaler"].update(std=[0.0] * 9),
])
def test_incompatible_checkpoints_rejected(trained, tmp_path, mutate):
    d = _save_variant(trained, tmp_path, mutate)
    with pytest.raises(model_runtime.ModelVersionError):
        model_runtime.LoadedModels.load(d)


def test_runtime_predict_bounded_and_fallback_on_incompatible(trained, tmp_path):
    lm = model_runtime.LoadedModels.load(trained["model_dir"])
    a = lm.predict([0.5, 0.5, 2.0, 2.0, 0.3, 0.8, 0.2, 0.3, 0.4])
    for k in core.ACTION_KEYS:
        lo, hi = core.ACTION_BOUNDS[k]
        assert lo <= a[k] <= hi
    assert isinstance(a["spawn_count"], int)
    with pytest.raises(model_runtime.ModelVersionError):
        lm.predict([0.5] * 8)

    bad = _save_variant(trained, tmp_path, lambda c: c.update(dda_version="old"))
    c = ctl.CGANController(bad)
    ctx = ctl.DecisionContext([0.5] * 9, core.CombatMetrics(), 2, 0.9, "enemy")
    dec = c.decide(ctx)
    assert dec.fallback_used == 1 and dec.fallback_reason.startswith("load_error:ModelVersionError")
    assert dec.controller_type == "cgan" and dec.action_source == "heuristic_fallback"
    heur = ctl.make_controller("heuristic").decide(ctx).action
    assert {k: dec.action[k] for k in core.ACTION_KEYS} == {k: heur[k] for k in core.ACTION_KEYS}

    good = ctl.CGANController(trained["model_dir"])
    good.begin_run(1)
    dec = good.decide(ctx)
    assert dec.fallback_used == 0 and dec.model_checkpoint_id == lm.checkpoint_id


class _NaNModels:
    noise_dim = 8
    checkpoint_id = "nan-model"

    def predict(self, cond_vec, noise=None):
        raise model_runtime.InvalidModelOutput("non-finite generator output [nan]")


def test_repeated_invalid_outputs_disable_model():
    c = ctl.CGANController("unused", models=_NaNModels())
    ctx = ctl.DecisionContext([0.5] * 9, core.CombatMetrics(), 2, 0.9, "enemy")
    reasons = [c.decide(ctx).fallback_reason for _ in range(5)]
    assert all(r for r in reasons)
    assert reasons[0].startswith("invalid_output:InvalidModelOutput")
    assert reasons[-1] == f"disabled_after_{c.max_invalid}_invalid_outputs"


def test_bot_runner_records_cgan_fallback(tmp_path):
    import csv
    from conftest import make_dataset
    p = make_dataset(tmp_path / "f.csv", runs=5, controller="cgan",
                     extra=("--model-dir", str(tmp_path / "missing")))
    rows = list(csv.DictReader(open(p)))
    assert rows and all(r["fallback_used"] == "1" and r["fallback_reason"].startswith("load_error")
                        for r in rows)
