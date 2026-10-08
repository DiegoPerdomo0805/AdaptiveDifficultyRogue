"""Freeze contract: what is recorded and that verification refuses drift."""
import json

import pytest

import freeze_experiment


def test_contract_records_every_frozen_item():
    c = freeze_experiment.current_contract()
    for k in ("experiment_config_hash", "bot_config_hash", "schema_version", "enemy_balance_targets",
              "loot_horizon_balance_targets", "fixed_default", "shared_boss", "principal_collection",
              "enemy_gate", "loot_gate", "loot_gate_split_seeds", "enemy_action_support",
              "loot_action_support", "loot_horizon"):
        assert k in c
    assert c["loot_horizon"]["combat_rooms"] == 3


def test_verify_rejects_a_tampered_record(tmp_path, monkeypatch):
    rec = {"frozen_commit": freeze_experiment._git("rev-parse", "HEAD"), **freeze_experiment.current_contract()}
    rec["enemy_balance_targets"] = {"death": {"low": 0, "high": 1, "scale": 1, "weight": 1}}
    path = tmp_path / "experiment_freeze.json"
    path.write_text(json.dumps(rec))
    monkeypatch.setattr(freeze_experiment, "FREEZE_FILE", str(path))
    with pytest.raises(SystemExit, match="FREEZE VIOLATION"):
        freeze_experiment.verify()
    monkeypatch.setattr(freeze_experiment, "FREEZE_FILE", str(tmp_path / "missing.json"))
    with pytest.raises(SystemExit, match="no freeze record"):
        freeze_experiment.verify()
