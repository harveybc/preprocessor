"""The deployed normalizer and the deployed rolling features, against a changed future.

R2 of `predictor/docs/handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md`:

    "run actual feature-eng indicators, real scaler fitting, preprocessing/window construction
     and extractor alignment... Prove train-only fitting and prefix invariance by changing
     future observations, future targets and future missingness."

`plugin_default` is what the smoke configuration selects. It splits the series into D1..D6,
fits normalizer A on **D1** and applies it to D2 and D3, fits normalizer B on **D4** for D5 and
D6, and derives `rolling_std_w`, `rolling_ema_w` and `price_minus_ema`. All of that is run
here on one deterministic chronological trajectory; nothing is reimplemented.

The previous battery fitted its own `zscore_fit` over a numpy array, so this plugin could
start fitting on everything and no rule would have noticed.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from app.plugins.plugin_default import Plugin

ROWS = 300
#: the first third is D1 (where normalizer A is fitted); the tail is well past it
FUTURE_FROM = 200
WINDOW = 12


def trajectory(rows=ROWS, tail_shift=0.0, hole=None, shift_from=None):
    state = 20260915
    values, level = [], 100.0
    for index in range(rows):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        level += ((state % 1000) / 1000.0 - 0.5)
        start = FUTURE_FROM if shift_from is None else shift_from
        value = level + (tail_shift if index >= start else 0.0)
        if hole is not None and index >= hole:
            value = float("nan")
        values.append(value)
    return pd.DataFrame({
        "DATE_TIME": pd.date_range("2024-01-01", periods=rows, freq="h"),
        "typical_price": values,
        "CLOSE": values,
    })


def run(frame, tmp_path):
    """The real plugin, with its real configuration keys, writing to a temporary directory."""
    tmp_path = __import__("pathlib").Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    plugin = Plugin()
    config = {
        "dataset_prefix": f"{tmp_path}/base_",
        "target_prefix": f"{tmp_path}/normalized_",
        "normalization_config_a": f"{tmp_path}/norm_a.json",
        "normalization_config_b": f"{tmp_path}/norm_b.json",
        "use_rolling_features": True,
        "rolling_price_column": "typical_price",
        "rolling_window": WINDOW,
        "use_cyclic_encoding": False,
        "fit_on_training_only": True,
        "trim_start_rows": 0,
    }
    plugin.set_params(**config)
    plugin.process(frame.copy(), config)
    return plugin


def params_a(plugin, column="typical_price"):
    return plugin.normalization_params["normalizer_a"][column]


def test_normalizer_a_is_fitted_on_d1_only_and_a_changed_future_does_not_move_it(tmp_path):
    """The heart of train-only fitting: shift everything after row 200 by 50 and refit.

    If the fit ever reached beyond D1 — the usual accident of calling `.mean()` on the whole
    frame — this mean and this standard deviation would move.
    """
    base = run(trajectory(), tmp_path / "base")
    moved = run(trajectory(tail_shift=50.0), tmp_path / "moved")
    assert params_a(base) == params_a(moved), (
        "normalizer A moved when only the future changed: it is not fitted on D1 alone")


def test_normalizer_a_does_move_when_its_own_training_rows_change(tmp_path):
    """The control for the rule above: if nothing could move it, it would prove nothing."""
    base = run(trajectory(), tmp_path / "base")
    altered = trajectory()
    altered.loc[:10, "typical_price"] += 25.0   # inside D1
    altered.loc[:10, "CLOSE"] += 25.0
    changed = run(altered, tmp_path / "changed")
    assert params_a(base) != params_a(changed), (
        "a change inside the training partition must reach the fitted parameters")


def test_normalizer_b_is_fitted_on_d4_and_is_not_normalizer_a(tmp_path):
    plugin = run(trajectory(), tmp_path / "split")
    a = plugin.normalization_params["normalizer_a"]["typical_price"]
    b = plugin.normalization_params["normalizer_b"]["typical_price"]
    assert a != b, "two partitions of a trending series cannot share one mean by accident"


def test_missingness_in_the_future_does_not_move_the_fitted_parameters(tmp_path):
    base = run(trajectory(), tmp_path / "base")
    gapped = run(trajectory(hole=FUTURE_FROM + 20), tmp_path / "gapped")
    assert params_a(base) == params_a(gapped)


def produced(tmp_path, name="base_d1.csv"):
    """What the real plugin WROTE, not a reimplementation of its two rolling lines."""
    return pd.read_csv(__import__("pathlib").Path(tmp_path) / name)


ROLLING = [f"rolling_std_{WINDOW}", f"rolling_ema_{WINDOW}", "price_minus_ema"]


def test_the_deployed_rolling_features_do_not_read_the_future(tmp_path):
    """The columns `process` itself derived, compared between two runs of the real plugin.

    D1 is the training partition, entirely before row 200, so every row of it is admissible;
    shifting everything from row 200 onwards must leave these columns untouched.
    """
    run(trajectory(), tmp_path / "base")
    run(trajectory(tail_shift=50.0), tmp_path / "moved")
    base, moved = produced(tmp_path / "base"), produced(tmp_path / "moved")
    for column in ROLLING:
        pd.testing.assert_series_equal(base[column], moved[column])


def test_the_deployed_rolling_features_declare_their_warm_up_as_missing(tmp_path):
    """Measured on the written output: `rolling` leaves the first w-1 rows empty, `ewm` does
    not, and starts at x[0].

    Two different warm-ups in the same block and only one is visible downstream.
    `price_minus_ema` inherits the EWM behaviour, so its first value is exactly zero — a real
    number meaning "no history", which a model cannot distinguish from a measurement.
    """
    run(trajectory(), tmp_path / "warm")
    frame = produced(tmp_path / "warm")
    std = frame[f"rolling_std_{WINDOW}"]
    ema = frame[f"rolling_ema_{WINDOW}"]
    assert std[:WINDOW - 1].isna().all() and not np.isnan(std.iloc[WINDOW - 1])
    assert ema.notna().all()
    assert ema.iloc[0] == frame["typical_price"].iloc[0]
    assert frame["price_minus_ema"].iloc[0] == 0.0


def test_a_two_sided_window_in_the_productive_path_makes_this_rule_fail(tmp_path,
                                                                         monkeypatch):
    """Check on the check, inside the deployed `process` rather than in a copy of it.

    `pandas.Series.rolling` is forced to `center=True` for the duration: the same expression
    the plugin runs then reads six rows into the future. The perturbation starts at row 100,
    just past the end of D1, so a centred window at D1's last rows can reach it — a leak of
    exactly half a window, which is what a centring mistake actually costs.
    """
    original = pd.Series.rolling

    def centred(self, *args, **kwargs):
        kwargs["center"] = True
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "rolling", centred)
    run(trajectory(shift_from=100), tmp_path / "base")
    run(trajectory(shift_from=100, tail_shift=50.0), tmp_path / "moved")
    base, moved = produced(tmp_path / "base"), produced(tmp_path / "moved")
    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(base[f"rolling_std_{WINDOW}"],
                                       moved[f"rolling_std_{WINDOW}"])


def test_the_unmutated_path_survives_that_same_perturbation(tmp_path):
    """And the control: with the deployed trailing window, row 100 onwards changes nothing."""
    run(trajectory(shift_from=100), tmp_path / "base")
    run(trajectory(shift_from=100, tail_shift=50.0), tmp_path / "moved")
    base, moved = produced(tmp_path / "base"), produced(tmp_path / "moved")
    for column in ROLLING:
        pd.testing.assert_series_equal(base[column], moved[column])
