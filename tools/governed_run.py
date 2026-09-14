#!/usr/bin/env python3
"""Governed preprocessor run (data-gov Flow v3).

Declares what this repository consumes and produces; the protocol itself lives
in data-gov (`tools/governed_exec.py`): campaign before data, governed download
with hash confirmation, fresh output namespace, command on CPU, terminal
COMPLETED | FAILED | INCONCLUSIVE | REFUSED through a durable outbox, then
reconciliation. A result that is not reconciled is not governing.

    python tools/governed_run.py --load_config examples/config_downsampled/phase_1b.json \
      --experiment-key prep-phase1b-001 --gov-url http://127.0.0.1:5055 \
      --api-key-file <service-key-file> --lake predictor_examples \
      --lake-root ../predictor/examples/data_downsampled --metrics-lake olap_cube \
      --out-dir <fresh-dir>

The data-gov checkout is found through DATA_GOV_CHECKOUT or the sibling
directory `../data-gov`. Metrics: row counts of every produced split file.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _governed_exec():
    checkout = Path(os.environ.get("DATA_GOV_CHECKOUT") or REPO_ROOT.parent / "data-gov").expanduser()
    path = checkout / "tools" / "governed_exec.py"
    if not path.is_file():
        raise SystemExit(f"governed_run: data-gov checkout not found at {checkout} (set DATA_GOV_CHECKOUT)")
    spec = importlib.util.spec_from_file_location("data_gov_governed_exec", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["data_gov_governed_exec"] = module
    spec.loader.exec_module(module)
    return module


def _metrics(config: dict) -> dict:
    files = []
    for key, split_prefix in (("dataset_prefix", "base_"), ("target_prefix", "normalized_")):
        prefix = Path(str(config.get(key) or split_prefix)).name
        files += [{"path": f"{prefix}d{i}.csv", "split": f"{prefix}d{i}", "header": bool(config.get("headers", True))}
                  for i in range(1, 7)]
    return {"kind": "row_counts", "files": files}


PROFILE = {
    "project": "preprocessor",
    "input_keys": ["input_file"],
    "output_keys": ["dataset_prefix", "target_prefix", "normalization_config_a", "normalization_config_b",
                    "save_config", "save_log", "output_file", "debug_file"],
    "command": [sys.executable, "app/main.py", "--load_config", "{config}"],
    "metrics": _metrics,
    "artifacts": {"base": "dataset_prefix", "normalized": "target_prefix",
                  "normalization_config_a": "normalization_config_a",
                  "normalization_config_b": "normalization_config_b",
                  "effective_config": "save_config", "debug_log": "save_log"},
    "tags": {"plugin": "plugin_default"},
}


def main(argv=None) -> int:
    return _governed_exec().consumer_main(PROFILE, argv, repo_root=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
