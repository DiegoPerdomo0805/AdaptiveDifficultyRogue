import csv
import json
import os

import pytest

import telemetry
import evaluate_controllers


def test_rogue_session_logs_controller_and_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pytest.importorskip("pygame")
    import rogue
    import controllers
    from node_map_gen import generate_node_map_graph

    nm = generate_node_map_graph(16, seed=5)
    ctrl = controllers.make_controller("cgan", model_dir=str(tmp_path / "missing"))
    session, writer, path = rogue.new_human_telemetry_session(nm, 5, ctrl, log_dir=str(tmp_path))
    player = rogue.Player(100, 100)
    enemy_rooms = [i for i, n in nm.nodes.items() if n.kind == "enemy"]
    session.begin_room(enemy_rooms[0], player)
    session.finalize_room(player, "cleared")
    rogue.close_human_telemetry_session(session, writer, path, player)
    report = telemetry.validate_dataset(path, path[:-4] + ".manifest.json")
    assert report["n_fallback_rows"] == 1
    row = next(csv.DictReader(open(path)))
    assert row["player_profile"] == "human" and row["controller_type"] == "cgan"
    assert row["fallback_reason"].startswith("load_error")


def test_eval_seed_guards(pilot_csv):
    import balance
    cfg = balance.load_experiment_config()
    with pytest.raises(SystemExit):     # outside evaluation block
        evaluate_controllers.check_seeds(100000, 10, [], cfg, False)
    with pytest.raises(SystemExit):     # overlaps training seeds
        evaluate_controllers.check_seeds(100050, 10, [pilot_csv], cfg, True)
    evaluate_controllers.check_seeds(9000000, 10, [pilot_csv], cfg, False)


def test_equivalent_seed_evaluation_end_to_end(trained, tmp_path):
    out = tmp_path / "eval"
    rc = evaluate_controllers.main(["--axis", "enemy", "--controllers", "fixed,heuristic,cgan", "--model-dir",
                                    trained["model_dir"], "--runs", "20", "--seed", "9000000",
                                    "--out-dir", str(out), "--train-csv", trained["csv"],
                                    "--bootstrap", "50"])
    assert rc == 0
    s = json.load(open(out / "summary.json"))
    assert set(s["controllers"]) == {"fixed", "heuristic", "cgan"}
    assert all(s["controllers"][c]["n_runs"] == 20 for c in s["controllers"])
    cg = s["controllers"]["cgan"]
    # only recorded OOD-guard fallbacks are acceptable, and they must be rare
    assert all(k.startswith("ood_action:") for k in cg["fallback_reasons"])
    assert cg["fallback_rows"] <= 0.1 * cg["n_combat_rooms"]
    assert s["paired"]["vs_heuristic"]["cgan"]["n_pairs"] == 20
    d = json.load(open(out / "model_diagnostics.json"))
    assert d["validity"] == 1.0 and 0.0 <= d["ood_rate"] <= 1.0
    for c in s["controllers"].values():                       # boss rows: shared policy everywhere
        assert c["boss_rows_shared_policy"] is True
    assert "boss_survival" in s["controllers"]["cgan"]["metrics"]
    pvo = s["predicted_vs_observed"]["by_condition"]
    assert set(pvo) == {"fixed", "heuristic", "cgan"}
    seeds = {c: sorted({r["run_seed"] for r in csv.DictReader(open(out / f"eval_{c}.csv"))})
             for c in ("fixed", "heuristic", "cgan")}
    assert seeds["fixed"] == seeds["heuristic"] == seeds["cgan"]
    with pytest.raises(SystemExit):     # refuses to overwrite an experiment
        evaluate_controllers.main(["--axis", "enemy", "--controllers", "fixed", "--runs", "2", "--seed", "9000000",
                                   "--out-dir", str(out)])
