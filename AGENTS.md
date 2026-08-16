# AGENTS.md — preprocessor

Instructions for AI coding agents working in this repository.
Human-facing documentation is in [`README.md`](README.md) and the per-plugin
`README_*.md` files.

## Project overview

preprocessor is a CLI that turns one CSV of financial time-series features into
the six train/validation/test splits used by the rest of this stack, with the
normalization parameters saved to JSON so the same transform can be replayed
elsewhere. The default plugin trims warm-up rows, drops rows around market gaps,
optionally adds cyclic time encodings and rolling features, splits the data into
D1–D6, and z-score normalizes with two separate normalizers (A fit on D1 and
applied to D1–D3; B fit on D4 and applied to D4–D6). Other plugins do
normalization, unbiasing, trimming, cleaning and feature selection on their own.

It does not engineer domain features (feature-eng), does not train models
(feature-extractor, predictor) and does not fetch market data — its only
data-download script (`dataset_analyzer.py`) is a standalone Kaggle utility, not
part of the pipeline.

## Agent quickstart (install → run → show the user results)

Verified end to end on Python 3.12.13 with pandas, numpy, scipy, matplotlib,
seaborn.

### 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scipy matplotlib seaborn
```

That is what the default pipeline actually imports. `setup.py` declares only
`pandas`, `numpy`, `requests`; `requirements.txt` carries a PyScaffold
deprecation banner and lists extras that the default pipeline does not use
(`boruta` and `scikit-learn` are for `plugin_feature_selector_post`;
`kagglehub` and `statsmodels` are for `dataset_analyzer.py`).

`pip install -e .` is **optional**. It installs the `preprocessor` console
script, but plugins are *not* resolved through entry points at runtime:
`app/plugin_loader.py` maps the group `preprocessor.plugins` to the directory
`app/plugins/` and imports `<plugin_name>.py` by file path. Everything below
runs straight from a source checkout.

### 2. Smoke test

The pytest suite does not pass — do not use it as a health signal:

```bash
PYTHONPATH=./ python -m pytest tests -q --continue-on-collection-errors
# observed: 15 failed, 1 passed, 3 errors in ~6s
```

The fastest real proof that the code works is the run in step 3: it is
deterministic and its outputs are committed, so a successful run leaves
`git status` clean.

### 3. Representative run

```bash
PYTHONPATH=./ python app/main.py --load_config examples/config_downsampled/phase_1b.json
```

(Equivalent: `sh ./preprocessor.sh --load_config examples/config_downsampled/phase_1b.json`,
which sets `PYTHONPATH=./` for you.)

Observed: exits 0 in about **2 seconds** over the bundled 22 521-row input
`examples/data/phase_3b_downsampled.csv`, printing a per-file summary table and
writing 15 files into `examples/data_downsampled/phase_1b/`:

| Output | Contents |
|---|---|
| `base_d1.csv` … `base_d6.csv` | The six splits before normalization (6298 / 1584 / 1584 / 6298 / 1584 / 1737 rows, 13 columns) |
| `normalized_d1.csv` … `normalized_d6.csv` | The same splits after z-score normalization |
| `normalization_config_a.json`, `normalization_config_b.json` | Per-column normalization parameters (A fit on D1, B fit on D4) |
| `preprocessor_config.json` | The merged effective config for replay |

The 13 columns come from this config's `"features_included":
["typical_price_only", "cyclic_features", "rolling_features"]` —
`DATE_TIME, typical_price, hod_sin, hod_cos, dow_sin, dow_cos, dom_sin, dom_cos,
moy_sin, moy_cos, rolling_std_24, rolling_ema_24, price_minus_ema` — out of the
49 columns in the input.

Two observed details worth knowing: the run rewrites files that are **committed**
in the repository (byte-identically, so `git status` stays clean — if it does
not, the change is real and worth inspecting), and `output_file`
(`output_phase_1b.csv`) and `debug_file` are named in the config but no such
files are produced; the splits above are the actual output.

The heavier sibling `examples/config/phase_1b.json` processes the full
88 084-row `examples/data/phase_3b.csv` into `examples/data/phase_1b/`.
`examples/scripts/run_all.sh` and `run_all_downsampled.sh` run every config in
their respective directories — minutes, not seconds.

### 4. Analytics

There is no bundled plotting entry point for pipeline outputs. Inspect the
result with a few lines of pandas, e.g.:

```bash
PYTHONPATH=./ python -c "
import pandas as pd
raw = pd.read_csv('examples/data/phase_3b_downsampled.csv', nrows=5)
out = pd.read_csv('examples/data_downsampled/phase_1b/normalized_d1.csv', nrows=5)
print('input columns :', len(raw.columns))
print('output columns:', list(out.columns))
"
```

`dataset_analyzer.py` at the repository root is a **separate** script that
downloads EURUSD datasets from Kaggle via `kagglehub` and writes plots to
`output/`. It needs network access and Kaggle credentials — do not run it as
part of a quickstart.

### 5. Final message to the user

> Preprocessing finished in about two seconds. The results are in
> `examples/data_downsampled/phase_1b/`: `base_d1.csv`–`base_d6.csv` (the six
> splits before normalization), `normalized_d1.csv`–`normalized_d6.csv` (after
> z-score normalization), `normalization_config_a.json` /
> `normalization_config_b.json` (the fitted normalization parameters, A from D1
> and B from D4), and `preprocessor_config.json` (the merged effective config,
> enough to replay the run). There is no UI.
>
> Suggested first analysis: plot the engineered feature `price_minus_ema` from
> `base_d1.csv` on top of the raw `typical_price` series from
> `examples/data/phase_3b_downsampled.csv` over the same first ~500 rows. It
> shows directly what the rolling transform extracted, and whether it is
> centered on zero the way a stationary input should be. Then overlay
> `normalized_d1.csv`'s version of the same column to see the effect of
> normalizer A.

## Build, test and lint commands

```bash
PYTHONPATH=./ python app/main.py --help                 # full CLI reference (exits 0)
sh ./preprocessor.sh --load_config <config.json>        # same, sets PYTHONPATH
PYTHONPATH=./ python -m pytest tests -q --continue-on-collection-errors
pip install -e .                                        # optional: installs the `preprocessor` console script
```

No linter, formatter or CI workflow is configured in this repository. Do not add
one without asking.

## Layout

| Path | Contents |
|---|---|
| `app/main.py` | Entry point: parse args → merge config → load plugin → run pipeline → save config |
| `app/config.py` | `DEFAULT_VALUES` plus `PARAMETER_VALIDATION` rules and `validate_config()` |
| `app/cli.py`, `app/config_merger.py`, `app/config_handler.py` | Flags, merge order, local/remote config load-save |
| `app/data_processor.py` | Pipeline: validation, plugin invocation, output checks, summary table |
| `app/plugin_loader.py`, `app/plugin_manager.py`, `app/plugin_registry.py` | File-path plugin discovery and isolated loading |
| `app/plugins/plugin_default.py` | Default pipeline: trim, market-gap margin, cyclic/rolling features, D1–D6 split, dual z-score normalization |
| `app/plugins/plugin_normalizer.py`, `plugin_unbiaser.py`, `plugin_trimmer.py`, `plugin_cleaner.py` | Single-purpose transforms |
| `app/plugins/plugin_feature_selector_pre.py`, `plugin_feature_selector_post.py` | Feature selection (the `_post` one needs scikit-learn + boruta) |
| `app/plugins/anti_naive_lock.py` | Transforms that keep models from locking onto the naive/persistence baseline |
| `app/validation_service.py`, `app/date_validator.py`, `app/error_handler.py`, `app/logging_service.py` | Support services |
| `examples/config/`, `examples/config_downsampled/` | Phase configs (full-size and downsampled) |
| `examples/data/`, `examples/data_downsampled/` | Committed inputs and committed pipeline outputs |
| `examples/scripts/` | `run_all.sh`, `run_all_downsampled.sh` |
| `README_*.md` | Per-plugin documentation (normalizer, unbiaser, trimmer, feature selectors, CRISP-DM notes) |
| `dataset_analyzer.py` | Standalone Kaggle download/analysis script, not part of the pipeline |
| `tests/` | pytest suite (mostly failing, see Smoke test) |

## Conventions and constraints

- **Config-driven**: defaults in `app/config.py` → `--load_config` JSON → CLI
  flags → unknown `--flags` (merged too, so any config key is settable from the
  CLI). `app/config.py` also declares validation rules; the D1–D6 proportions
  are checked to sum to 1.0 and a mismatch is reported as a warning, not an
  error (the bundled configs sum to 0.992).
- **Plugins are files, not entry points.** `load_plugin('preprocessor.plugins',
  name)` imports `app/plugins/<name>.py` and expects a `Plugin` class with
  `plugin_params`, `set_params` and the pipeline hooks. The
  `preprocessor.plugins` entry points declared in `setup.py` use different names
  (`default_plugin`, `normalizer`, ...) and are not what the runtime uses —
  configs name the *file* (`plugin_default`). Adding a plugin means adding a
  file, not reinstalling.
- **Data contract**: input is a CSV with a `DATE_TIME` column and numeric
  feature columns, `headers: true`. Which columns are written is controlled by
  `features_included`, a list of group names resolved in `plugin_default.py`:
  `base_features`, `typical_price_only`, `cyclic_features`, `rolling_features`,
  `technical_features`, `fundamental_features`, `seasonal_features`,
  `high_frequency_features`. Omitting the key writes every column. The filter
  applies only to what is saved, not to the normalization statistics.
- **The split contract is the point of this repository**: D1/D2/D3 are the
  autoencoder train/validation/test sets and D4/D5/D6 the predictor ones; two
  normalizers keep the second group from leaking statistics of the first. Do
  not "simplify" to a single normalizer.
- **Fit only on training data**: `fit_on_training_only` is true by default —
  parameters come from D1 and D4 only.
- **Reproducibility**: the pipeline is deterministic. The normalization JSONs
  are the replay contract for downstream repositories; changing column naming or
  normalization semantics silently invalidates artifacts elsewhere in the stack.
- Windows/Linux: some defaults elsewhere in this stack use Windows path
  separators. This repository's example configs use forward slashes and run on
  Linux as-is.

## Do not touch

- `examples/data/` and `examples/data_downsampled/` — both the committed inputs
  and the committed pipeline outputs. Running the examples rewrites the outputs;
  that is expected, but only commit the change if you meant to alter behavior.
- `output/` — scratch directory for `dataset_analyzer.py`.
- `preprocessor.egg-info/`, `__pycache__/`, `.pytest_cache/` — generated.
- `app/plugins/plugin_default_backup.py` and `app/plugin_loade_oldr.py` — dead
  copies kept in the tree; do not edit them to "fix" behavior and do not import
  them.
- Sibling repositories (feature-eng, feature-extractor, predictor). The
  `feature_eng_plugin_path` default points at a local absolute path outside this
  repository and is only used when `use_external_feature_eng` is true.
