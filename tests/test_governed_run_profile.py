"""The governed wrapper declares this repository's inputs, outputs and metrics."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_GOV = Path(os.environ.get("DATA_GOV_CHECKOUT") or ROOT.parent / "data-gov")

pytestmark = pytest.mark.skipif(
    not (DATA_GOV / "tools" / "governed_exec.py").is_file(), reason="data-gov checkout not available"
)


def _load():
    spec = importlib.util.spec_from_file_location("preprocessor_governed_run", ROOT / "tools" / "governed_run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["preprocessor_governed_run"] = module
    spec.loader.exec_module(module)
    return module


def test_profile_builds_a_valid_spec(tmp_path, capsys):
    lake = tmp_path / "lake" / "phase_1"
    lake.mkdir(parents=True)
    (lake / "normalized_d4.csv").write_text("DATE_TIME,typical_price\n2013-01-01 00:00:00,1\n")
    config = tmp_path / "phase_x" / "config.json"
    config.parent.mkdir()
    config.write_text(json.dumps({
        "input_file": str(lake / "normalized_d4.csv"), "dataset_prefix": "examples/out/base_",
        "target_prefix": "examples/out/normalized_", "normalization_config_a": "examples/out/norm_a.json",
        "normalization_config_b": "examples/out/norm_b.json", "save_config": "examples/out/preprocessor_config.json",
        "headers": True, "plugin": "plugin_default",
    }))
    module = _load()
    code = module.main([
        "--load_config", str(config), "--experiment-key", "prep-001", "--lake", "predictor_examples",
        "--lake-root", str(tmp_path / "lake"), "--out-dir", str(tmp_path / "out"), "--print-spec",
    ])
    assert code == 0
    spec = json.loads(capsys.readouterr().out)
    assert spec["schema"] == "governed_exec_spec.v1" and spec["project"] == "preprocessor"
    assert spec["datasets"] == [{"lake": "predictor_examples", "resource": "phase_1/normalized_d4.csv",
                                 "role": "input_file", "from": None, "to": None}]
    assert spec["input_keys"] == {"input_file": "input_file"} and spec["classification"] == "GOVERNING"
    assert "dataset_prefix" in spec["output_keys"] and spec["command"][1:] == ["app/main.py", "--load_config", "{config}"]
    assert spec["metrics"]["kind"] == "row_counts" and len(spec["metrics"]["files"]) == 12
    assert spec["metrics"]["files"][0] == {"path": "base_d1.csv", "split": "base_d1", "header": True}
    assert spec["tags"]["phase"] == "phase_x"
    # an override of a governed input is refused before anything is registered
    assert module.main([
        "--load_config", str(config), "--experiment-key", "prep-002", "--lake", "predictor_examples",
        "--lake-root", str(tmp_path / "lake"), "--out-dir", str(tmp_path / "out2"), "--print-spec",
        "--", "--input_file", "other.csv",
    ]) == 1
    assert "would override a governed input" in capsys.readouterr().err
