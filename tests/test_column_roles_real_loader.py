"""The contract, exercised through the loader this application actually runs.

R1 of `predictor/docs/handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md`:
the previous round tested a copied helper, so the rules proved what the copy did, not what a
run does. Everything here goes through `app.data_handler.load_csv` with a file on disk.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from app.column_roles import ColumnRoleError
from app.data_handler import load_csv

CONTRACT = {"time": "DATE_TIME", "features": ["OPEN", "HIGH", "LOW", "CLOSE"],
            "targets": ["CLOSE"], "metadata": ["available_time"],
            "allow_target_as_feature": True}


def csv_at(tmp_path, columns, rows=3):
    """A small file whose timestamp columns are written as timestamps, as a producer writes."""
    stamps = [f"2024-01-0{i + 1} 00:00:00" for i in range(rows)]
    body = {}
    for name in columns:
        body[name] = stamps if name in ("DATE_TIME", "available_time") else [
            float(i + 1) for i in range(rows)]
    path = tmp_path / "series.csv"
    pd.DataFrame(body).to_csv(path, index=False)
    return str(path)


def test_the_declared_features_survive_in_the_declared_order(tmp_path):
    path = csv_at(tmp_path, ["DATE_TIME", "CLOSE", "OPEN", "available_time", "HIGH", "LOW"])
    config = dict(column_roles=CONTRACT)
    frame = load_csv(path, config)
    assert list(frame.columns) == ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE"]
    record = config["column_roles_applied"]["input_file"]
    assert record["features"] == ["OPEN", "HIGH", "LOW", "CLOSE"]
    assert len(record["contract_sha256"]) == 64


def test_an_undeclared_column_stops_the_run(tmp_path):
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "available_time",
                             "SURPRISE"])
    with pytest.raises(ColumnRoleError, match="SURPRISE"):
        load_csv(path, dict(column_roles=CONTRACT))


def test_the_target_overlap_needs_the_declaration_at_the_real_loader(tmp_path):
    """Musashi's first counterexample, through the loader rather than through the helper."""
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "available_time"])
    without = {key: value for key, value in CONTRACT.items()
               if key != "allow_target_as_feature"}
    with pytest.raises(ColumnRoleError, match="CLOSE"):
        load_csv(path, dict(column_roles=without))


def test_a_contradictory_metadata_feature_stops_the_run(tmp_path):
    """Musashi's second counterexample: numeric metadata must not reach the model."""
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN", "available_time"])
    contract = {"time": "DATE_TIME", "features": ["OPEN", "available_time"],
                "metadata": ["available_time"]}
    with pytest.raises(ColumnRoleError, match="available_time"):
        load_csv(path, dict(column_roles=contract))


def test_a_timestamp_declared_as_a_feature_is_refused_not_coerced(tmp_path):
    """The incident itself, at the real loader.

    `load_csv` coerces every non-time column with `errors="coerce"` *before* the contract is
    resolved, so an ISO timestamp declared as a feature became a column of NaN instead of a
    refusal. A silent column of NaN is worse than the original defect: the run continues.
    """
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN", "available_time"])
    contract = {"time": "DATE_TIME", "features": ["OPEN", "available_time"], "metadata": []}
    with pytest.raises(ColumnRoleError, match="available_time"):
        load_csv(path, dict(column_roles=contract))


def test_a_run_without_a_contract_is_refused_at_the_loader(tmp_path):
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN"])
    with pytest.raises(ColumnRoleError, match="column_roles"):
        load_csv(path, {})


def test_the_declared_legacy_migration_still_loads_everything(tmp_path):
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN"])
    frame = load_csv(path, {"column_roles_migration": "LEGACY_ALL_COLUMNS_ARE_FEATURES"})
    assert list(frame.columns) == ["DATE_TIME", "OPEN"]
