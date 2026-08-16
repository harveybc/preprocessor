# Preprocessor

A command-line application for preprocessing financial time-series CSVs. It
loads one CSV, applies a plugin (trimming, market-gap filtering, cyclic and
rolling feature construction, splitting, normalization, unbiasing, cleaning or
feature selection), and writes the results plus the fitted transform parameters
to disk, so the same transform can be replayed on other datasets.

The default plugin produces the six-way split this stack trains on: D1/D2/D3
(autoencoder train/validation/test) and D4/D5/D6 (predictor
train/validation/test), z-score normalized with two separate normalizers — A fit
on D1 and applied to D1–D3, B fit on D4 and applied to D4–D6.

Scope: this repository transforms datasets. It does not engineer domain features
(see [feature-eng](https://github.com/harveybc/feature-eng)), does not train
models (see [feature-extractor](https://github.com/harveybc/feature-extractor)
and [predictor](https://github.com/harveybc/predictor)), and does not trade.

## Run this with an AI agent

Paste this into Claude Code, Cursor, Codex, GitHub Copilot or any coding agent with shell access:

> Read `AGENTS.md` in this repository and follow the **Agent quickstart** section end to end: set up the environment, run the smoke test, execute the example preprocessing run, then tell me the exact file paths or URL where I can see the results and one analysis I should try first.

`AGENTS.md` is the [agents.md](https://agents.md) convention, read natively by most coding agents.

## Installation

```bash
git clone https://github.com/harveybc/preprocessor.git
cd preprocessor
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas numpy scipy matplotlib seaborn
```

Those are the packages the default pipeline imports. `setup.py` declares only
`pandas`, `numpy` and `requests`; `requirements.txt` additionally lists packages
used by optional parts (`scikit-learn` and `boruta` for the post feature
selector, `kagglehub` and `statsmodels` for `dataset_analyzer.py`).

`pip install .` is optional — it installs the `preprocessor` console script.
Plugins are loaded by file path from `app/plugins/`, not through entry points, so
a source checkout is enough.

## Usage

```bash
PYTHONPATH=./ python app/main.py --load_config examples/config_downsampled/phase_1b.json
```

or, equivalently, using the wrapper that sets `PYTHONPATH` for you:

```bash
sh ./preprocessor.sh --load_config examples/config_downsampled/phase_1b.json   # Linux
preprocessor.bat --load_config examples\config_downsampled\phase_1b.json       # Windows
```

Verified: exits 0 in about 2 seconds over the bundled 22 521-row
`examples/data/phase_3b_downsampled.csv`, writing 15 files into
`examples/data_downsampled/phase_1b/` — `base_d1.csv`–`base_d6.csv`,
`normalized_d1.csv`–`normalized_d6.csv`, `normalization_config_a.json`,
`normalization_config_b.json` and `preprocessor_config.json` (the merged
effective config). Those outputs are committed in this repository, and the run
reproduces them byte-identically.

Running with no arguments fails: the default `input_file` in
[`app/config.py`](app/config.py) is `examples/data/feature_eng_output.csv`,
which is not part of this repository. Always pass `--load_config` or
`--input_file`.

Use `-h` / `--help` for the full flag list. Any configuration key can also be
set as a CLI flag; flags override the loaded config file, which overrides the
defaults.

Larger example: `examples/config/phase_1b.json` processes the full 88 084-row
`examples/data/phase_3b.csv`. `examples/scripts/run_all.sh` and
`run_all_downsampled.sh` run every config in their directory.

## Plugins

Selected with `--plugin <name>`, where `<name>` is a file in `app/plugins/`:

| Plugin | Purpose | Notes |
|---|---|---|
| `plugin_default` | Full pipeline: trim, market-gap margin, cyclic and rolling features, D1–D6 split, dual z-score normalization | Default |
| `plugin_normalizer` | Normalization only, with saved parameters | [README_normalizer.md](README_normalizer.md) |
| `plugin_unbiaser` | Bias removal (moving-average / EMA variants) | [README_unbiaser.md](README_unbiaser.md) |
| `plugin_trimmer` | Row/column trimming | [README_trimmer.md](README_trimmer.md) |
| `plugin_cleaner` | Missing-value and outlier handling | |
| `plugin_feature_selector_pre` | Feature selection before modeling | [README_feature_selector_pre.md](README_feature_selector_pre.md) |
| `plugin_feature_selector_post` | Feature selection using model feedback | Needs `scikit-learn` + `boruta`; [README_feature_selector_post.md](README_feature_selector_post.md) |
| `anti_naive_lock` | Selective transforms that stop a model from copying inputs to outputs | |

[README_CrISP_DM.md](README_CrISP_DM.md) records the CRISP-DM framing of the
data-preparation stage.

## Configuration

Precedence: defaults in [`app/config.py`](app/config.py) → `--load_config` JSON →
CLI flags. Key groups: file paths (`input_file`, `dataset_prefix`,
`target_prefix`, `save_config`, `normalization_config_a`/`_b`), split
proportions (`d1_proportion` … `d6_proportion`), feature construction
(`features_included`, `use_cyclic_encoding`, `use_rolling_features`,
`rolling_window`), and normalization (`normalization_method`,
`normalization_range`, `fit_on_training_only`). `app/config.py` also carries
validation rules; a split-proportion sum other than 1.0 is reported as a warning
(the bundled configs sum to 0.992).

## Tests

```bash
PYTHONPATH=./ python -m pytest tests -q --continue-on-collection-errors
```

Observed: `15 failed, 1 passed, 3 errors`. The suite predates the current
pipeline and needs repair; the deterministic example run above is the working
smoke check. Some tests were originally written against an external
[harveybc/data-logger](https://github.com/harveybc/data-logger) instance for the
remote logging paths.

## Repository layout

```
preprocessor/
├── app/
│   ├── main.py                     # entry point
│   ├── cli.py                      # CLI flags
│   ├── config.py                   # defaults + validation rules
│   ├── config_merger.py            # defaults / file / CLI merge
│   ├── config_handler.py           # local and remote config load-save
│   ├── data_handler.py             # CSV IO
│   ├── data_processor.py           # pipeline orchestration
│   ├── plugin_loader.py            # loads app/plugins/<name>.py by path
│   ├── plugin_manager.py           # plugin lifecycle
│   ├── plugin_registry.py          # plugin discovery/metadata
│   ├── validation_service.py       # config and output validation
│   ├── date_validator.py           # date column checks
│   ├── error_handler.py            # error reporting
│   ├── logging_service.py          # logging setup
│   └── plugins/                    # plugin implementations (see table above)
├── examples/
│   ├── config/                     # phase configs (full-size data)
│   ├── config_downsampled/         # phase configs (downsampled data)
│   ├── data/                       # committed inputs and outputs
│   ├── data_downsampled/           # committed downsampled outputs
│   └── scripts/                    # run_all.sh, run_all_downsampled.sh
├── tests/                          # pytest suite (see Tests)
├── dataset_analyzer.py             # standalone Kaggle download/analysis script
├── AGENTS.md                       # instructions for AI coding agents
├── setup.py
└── requirements.txt
```

## Other tools

`dataset_analyzer.py` downloads EURUSD datasets from Kaggle with `kagglehub`,
analyzes them and writes plots to `output/`. It is standalone: it needs network
access and Kaggle credentials, and it is not part of the preprocessing pipeline.

Generate API documentation with `pdoc --html -o docs app` (requires `pdoc3`).

## License

[MIT](LICENSE.txt). Authors: [AUTHORS.rst](AUTHORS.rst).
