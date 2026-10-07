import json
import shutil

import pytest

import telemetry
import bot_runner
from conftest import make_dataset


def _copy(tmp_path, name="v"):
    src = make_dataset(tmp_path / "src.csv", runs=10)
    dst = tmp_path / f"{name}.csv"
    shutil.copy(src, dst)
    shutil.copy(src[:-4] + ".manifest.json", tmp_path / f"{name}.manifest.json")
    return str(dst), str(tmp_path / f"{name}.manifest.json")


def _edit_manifest(mpath, **kv):
    with open(mpath) as f:
        m = json.load(f)
    m.update(kv)
    with open(mpath, "w") as f:
        json.dump(m, f)


@pytest.mark.parametrize("field,value", [
    ("config_hash", "deadbeefdeadbeef"), ("schema_version", "room_samples_v2.1"),
    ("dda_version", "rule_based_dda_v1.1"), ("experiment_config_hash", "0000"),
    ("experiment_config_version", "old"), ("n_rows", 999999),
])
def test_manifest_tampering_rejected(tmp_path, field, value):
    csv_path, mpath = _copy(tmp_path)
    telemetry.validate_dataset(csv_path, mpath)
    _edit_manifest(mpath, **{field: value})
    with pytest.raises(telemetry.DatasetVersionError):
        telemetry.validate_dataset(csv_path, mpath)


def test_policy_extra_mismatch_rejected(tmp_path):
    csv_path, mpath = _copy(tmp_path)
    extra = bot_runner.policy_extra()
    telemetry.validate_dataset(csv_path, mpath, expected_extra_subset=extra)
    extra["node_count"] = 99
    with pytest.raises(telemetry.DatasetVersionError):
        telemetry.validate_dataset(csv_path, mpath, expected_extra_subset=extra)


@pytest.mark.parametrize("col,value", [("controller_type", "oracle"), ("schema_version", "x"),
                                       ("fallback_used", "2")])
def test_bad_rows_rejected(tmp_path, col, value):
    csv_path, mpath = _copy(tmp_path)
    lines = open(csv_path).read().splitlines()
    header = lines[0].split(",")
    cells = lines[1].split(",")
    cells[header.index(col)] = value
    lines[1] = ",".join(cells)
    open(csv_path, "w").write("\n".join(lines) + "\n")
    with pytest.raises(telemetry.DatasetVersionError):
        telemetry.validate_dataset(csv_path, mpath)


def test_duplicate_keys_rejected(tmp_path):
    csv_path, mpath = _copy(tmp_path)
    lines = open(csv_path).read().splitlines()
    lines.append(lines[1])
    open(csv_path, "w").write("\n".join(lines) + "\n")
    _edit_manifest(mpath, n_rows=len(lines) - 1)
    with pytest.raises(telemetry.DatasetVersionError, match="duplicate"):
        telemetry.validate_dataset(csv_path, mpath)


def test_out_overwrite_guard(tmp_path):
    p = make_dataset(tmp_path / "o.csv", runs=3)
    with pytest.raises(SystemExit):
        bot_runner.main(["--runs", "3", "--out", p])
