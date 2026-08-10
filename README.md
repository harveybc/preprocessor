# preprocessor

Dataset preprocessing pipeline for time-series prediction. `preprocessor`
takes a feature CSV (typically the output of
[feature-eng](https://github.com/harveybc/feature-eng)), splits it into six
datasets, fits normalization parameters on the configured training splits,
applies plugin-based transformations (normalize, unbias, trim, clean,
feature-select), and writes the processed datasets plus the fitted parameters
so that downstream trainers and inference services operate in exactly the same
value space.

## Status

**Active component** of the harveybc trading stack (package `preprocessor`
0.1.1). This `master` branch is the packaged default branch; development also
continues on feature branches (e.g. `phase_6`).

## Role and non-responsibilities

`preprocessor` prepares already-engineered datasets for model training:
splitting, normalization, bias removal, trimming, cleaning and feature
selection, with persisted parameters for reproducibility.

It does **not**:

- compute domain features or labels — that is
  [feature-eng](https://github.com/harveybc/feature-eng);
- learn compressed representations — that is
  [feature-extractor](https://github.com/harveybc/feature-extractor);
- train or serve models — that is
  [predictor](https://github.com/harveybc/predictor) and
  [prediction_provider](https://github.com/harveybc/prediction_provider);
- generate synthetic data — that is
  [synthetic-datagen](https://github.com/harveybc/synthetic-datagen).

## Architecture

[`app/main.py`](app/main.py) delegates to the CLI in
[`app/cli.py`](app/cli.py), which drives the preprocessing core under
[`app/core/`](app/core): load input (CSV/JSON/Parquet) → split into six
datasets (`temporal`, `random` or `stratified`) → fit normalization on the
configured training datasets → run configured feature/postprocessing plugins →
write outputs and metadata.

Plugins are registered in the `preprocessor.plugins` entry-point group in
[`setup.py`](setup.py):

| Entry point | Implementation | Purpose | Sub-doc |
|---|---|---|---|
| `default_plugin` | [`app/plugins/plugin_default.py`](app/plugins/plugin_default.py) | Pass-through default | — |
| `normalizer` | [`app/plugins/plugin_normalizer.py`](app/plugins/plugin_normalizer.py) | Normalization (z-score / min-max) with persisted parameters | [`README_normalizer.md`](README_normalizer.md) |
| `unbiaser` | [`app/plugins/plugin_unbiaser.py`](app/plugins/plugin_unbiaser.py) | Bias removal (moving-average / EMA) | [`README_unbiaser.md`](README_unbiaser.md) |
| `trimmer` | [`app/plugins/plugin_trimmer.py`](app/plugins/plugin_trimmer.py) | Row/column trimming | [`README_trimmer.md`](README_trimmer.md) |
| `cleaner` | [`app/plugins/plugin_cleaner.py`](app/plugins/plugin_cleaner.py) | Data cleaning (missing/invalid values) | — |
| `feature_selector` | [`app/plugins/plugin_feature_selector_pre.py`](app/plugins/plugin_feature_selector_pre.py) | Feature selection, **pre-training** variant | [`README_feature_selector_pre.md`](README_feature_selector_pre.md) |

A post-training feature selector
([`app/plugins/plugin_feature_selector_post.py`](app/plugins/plugin_feature_selector_post.py),
documented in
[`README_feature_selector_post.md`](README_feature_selector_post.md)) exists
in the tree but is **not registered** as an entry point; the registered
`feature_selector` name resolves to the pre-training variant only.

Methodology background is in [`README_CrISP_DM.md`](README_CrISP_DM.md), and
design/test records are under [`docs/`](docs).

## Requirements

- Python 3 (no `python_requires` pin in [`setup.py`](setup.py); verified below
  under Python 3.12.13).
- Core dependencies from `setup.py`: `pandas`, `numpy`, `requests`. The fuller
  development set is in [`requirements.txt`](requirements.txt)
  (`scikit-learn`, `scipy`, `statsmodels`, `PyWavelets`, `boruta`, ...); the
  unregistered post-training feature selector additionally requires `boruta`.

## Installation

Unverified (not executed in a clean environment for this README):

```bash
git clone https://github.com/harveybc/preprocessor.git
cd preprocessor
pip install -r requirements.txt
pip install -e .
```

Verified in the maintainer environment (Python 3.12.13, 2026-08-10):

- `python -m app.main --help` → prints the full CLI usage.
- `python -c "import app.plugins.plugin_normalizer, app.plugins.plugin_unbiaser, app.plugins.plugin_trimmer, app.plugins.plugin_cleaner, app.plugins.plugin_default"`
  → `core plugin imports OK`; `plugin_feature_selector_pre` also imports OK.

## Quickstart (verified)

Run the full pipeline in dry-run mode on the repo-owned test dataset:

```bash
python -m app.main tests/data/feature_eng_output.csv --dry-run
```

Observed result (2026-08-10): the pipeline computed normalization parameters
for 54 features from 55,424 samples, applied normalization to all 6 datasets,
and finished with `Processing completed successfully in 0.37 seconds` /
`Dry run completed - no files written`.

For a real run, drop `--dry-run` and choose an output directory:

```bash
python -m app.main tests/data/feature_eng_output.csv -o ./output \
  --split-method temporal --normalization-method z-score --save-metadata
```

Key flags (see `--help` for all): `--config/-c` JSON configuration file;
`--split-ratios` (six comma-separated values summing to 1.0);
`--split-method {temporal,random,stratified}`;
`--normalization-method {z-score,min-max,robust,none}`;
`--training-datasets` (which splits fit the normalizer);
`--feature-plugins` / `--postprocessing-plugins` (plugin chains);
`--output-format {csv,json,parquet}`; `--validate-only`; `--dry-run`.

This is a standalone CLI stage; it has no distributed/DOIN runtime role.

## Tests

```bash
python -m pytest -q --collect-only
```

Observed result (2026-08-10, Python 3.12.13): `243 tests collected in 0.52s`
with no collection errors. Test suites live under [`tests/`](tests)
(acceptance, integration, system, unit).

## Outputs and reproducibility

- Processed datasets are written to `--output-dir` (default `./output`) in the
  chosen format; `--save-metadata` writes processing metadata alongside.
- Normalization/debug parameters are persisted (default `./debug_out.json`)
  so the exact same transform can be re-applied or inverted downstream —
  feature-extractor and predictor consume these persisted parameters.
- `--save-config` persists the effective configuration for replaying a run.
- `config_out.json` and `debug_out.json` at the repository root are committed
  run residue from earlier executions, kept as format examples.

## Safety and security

- No credentials are stored in this repository. Remote config/log endpoints
  take `--remote_username`/`--remote_password` as CLI arguments — do not embed
  secrets in committed config files.
- Bundled test data are historical market-derived datasets for testing only.
  Nothing in this repository is financial advice.

## Limitations

- The `preprocessor.plugins` entry-point group name is also claimed by other
  repositories in the stack
  ([predictor](https://github.com/harveybc/predictor) and
  [gym-fx](https://github.com/harveybc/gym-fx) register the same group for
  their own local plugin packages), so installing them together can shadow
  plugin names across packages.
- The post-training feature selector is present but unregistered (see
  Architecture); only the pre-training variant is reachable through the
  `feature_selector` entry point.
- Several debug/one-off scripts (`quick_test.py`, `debug_*.py`,
  `dataset_analyzer.py`) and generated outputs are committed at the root.
- Development branches (e.g. `phase_6`) carry additional examples and
  experiments not present on `master`.

## Related repositories

- [feature-eng](https://github.com/harveybc/feature-eng) — upstream feature
  and label generation (its output CSV is this repo's canonical input).
- [feature-extractor](https://github.com/harveybc/feature-extractor) —
  downstream autoencoder training on normalized datasets.
- [predictor](https://github.com/harveybc/predictor) — downstream model
  training using the persisted normalization parameters.
- [prediction_provider](https://github.com/harveybc/prediction_provider) —
  serving layer that must replicate the same preprocessing.

## License and authors

[MIT](LICENSE.txt). Authors are listed in [`AUTHORS.rst`](AUTHORS.rst).
