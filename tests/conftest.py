import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import bot_runner  # noqa: E402

PILOT_MIX = "exploration=0.70,heuristic=0.20,fixed=0.10"


def make_dataset(path, runs=40, seed=100000, controller="heuristic", extra=()):
    rc = bot_runner.main(["--runs", str(runs), "--seed", str(seed), "--controller", controller,
                          "--out", str(path), "--validate", *extra])
    assert rc == 0
    return str(path)


@pytest.fixture(scope="session")
def pilot_csv(tmp_path_factory):
    d = tmp_path_factory.mktemp("pilot")
    return make_dataset(d / "pilot.csv", runs=200, controller=PILOT_MIX)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, pilot_csv):
    """Small Q + cGAN training run shared by the model tests."""
    pytest.importorskip("torch")
    import train_cgan
    import train_q
    d = tmp_path_factory.mktemp("models")
    q_dir, m_dir = str(d / "q"), str(d / "m")
    train_q.main(["--csv", pilot_csv, "--out-dir", q_dir, "--epochs", "40", "--patience", "10"])
    train_cgan.main(["--csv", pilot_csv, "--q-dir", q_dir, "--out-dir", m_dir, "--epochs", "6",
                     "--allow-failed-q-gate"])
    return {"csv": pilot_csv, "q_dir": q_dir, "model_dir": m_dir}
