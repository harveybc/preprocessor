
# Preprocessor 

## Description

The Preprocessor project is a flexible and modular application for preprocessing time series data. It supports dynamic loading of plugins for various preprocessing tasks such as normalization, unbiasing, trimming, and feature selection. Each plugin can save and load its parameters for consistent preprocessing across different datasets.

## Plugins

### 1. Default (Normalizer) Plugin

The Normalizer Plugin is used to normalize data using methods such as z-score and min-max normalization.

[Read more about the Normalizer Plugin](https://github.com/harveybc/preprocessor/blob/master/README_normalizer.md)

### 2. Unbiaser Plugin

The Unbiaser Plugin removes bias from time series data using moving average (MA) and exponential moving average (EMA) methods.

[Read more about the Unbiaser Plugin](https://github.com/harveybc/preprocessor/blob/master/README_unbiaser.md)

### 3. Trimmer Plugin

The Trimmer Plugin removes specified columns and rows from the dataset.

[Read more about the Trimmer Plugin](https://github.com/harveybc/preprocessor/blob/master/README_trimmer.md)

### 4. Pre-FeatureExtraction, Feature Selector Plugin

Performs the initial screening for redundant or non-informative data, the Pre-FeatureExtraction Feature Selector Plugin performs feature selection using methods such as ACF, PACF, and Granger Causality Test. 

This selection is meant to be performed before feature extraction or other dimensionality reduction technique.

[Read more about the Pre-Feature Selector Plugin](https://github.com/harveybc/preprocessor/blob/master/README_feature_selector_pre.md)

### 5. Post-FeatureExtraction, Feature Selector Plugin

The Post-FeatureEtraction Feature Selector Plugin performs feature selection after initial preprocessing and feature extraction using methods such as LASSO, Elastic Net, Mutual Information, Cross-Validation with LSTM/CNN, and Boruta Algorithm.

This selection is meant to be performed after feature extraction or other dimensionality reduction technique.

[Read more about the Post-Feature Selector Plugin](https://github.com/harveybc/preprocessor/blob/master/README_feature_selector_post.md)

## Examples of Use

### Command Line

You can use the Preprocessor application from the command line with various plugins. Below are some examples:

#### Using Default Parameters (Normalizer Plugin)

```bash
python app/main.py --config config.json --plugin default_plugin
```

#### Using Unbiaser Plugin with EMA Method

```bash
python app/main.py --config config.json --plugin unbiaser_plugin --method ema --ema_alphas 0.2 --save_params ema_params.json
```

#### Using Trimmer Plugin to Remove Columns and Rows

```bash
python app/main.py --config config.json --plugin trimmer_plugin --columns 0 1 2 --rows 0 1 2
```

#### Using Pre-Feature Selector Plugin with ACF Method

```bash
python app/main.py --config config.json --plugin pre_feature_selector_plugin --method acf --save_params acf_params.json
```

#### Using Post-Feature Selector Plugin with LASSO Method

```bash
python app/main.py --config config.json --plugin post_feature_selector_plugin --method lasso --save_params lasso_params.json
```

### Example Configuration Files

#### Example 1: Normalizer Plugin Configuration

```json
{
    "csv_file": "path/to/input.csv",
    "output_file": "path/to/output.csv",
    "plugins": [
        {
            "name": "default_plugin",
            "params": {
                "method": "z-score",
                "save_params": "normalization_params.json",
                "load_params": "normalization_params.json"
            }
        }
    ],
    "remote_log": "http://remote-log-server/api/logs"
}
```

#### Example 2: Unbiaser Plugin Configuration

```json
{
    "csv_file": "path/to/input.csv",
    "output_file": "path/to/output.csv",
    "plugins": [
        {
            "name": "unbiaser_plugin",
            "params": {
                "method": "ema",
                "ema_alphas": [0.2],
                "save_params": "ema_params.json",
                "load_params": "ema_params.json"
            }
        }
    ],
    "remote_log": "http://remote-log-server/api/logs"
}
```

#### Example 3: Trimmer Plugin Configuration

```json
{
    "csv_file": "path/to/input.csv",
    "output_file": "path/to/output.csv",
    "plugins": [
        {
            "name": "trimmer_plugin",
            "params": {
                "columns": [0, 1, 2],
                "rows": [0, 1, 2],
                "save_params": "trimmer_params.json",
                "load_params": "trimmer_params.json"
            }
        }
    ],
    "remote_log": "http://remote-log-server/api/logs"
}
```

#### Example 4: Pre-Feature Selector Plugin Configuration

```json
{
    "csv_file": "path/to/input.csv",
    "output_file": "path/to/output.csv",
    "plugins": [
        {
            "name": "pre_feature_selector_plugin",
            "params": {
                "method": "acf",
                "max_lag": 5,
                "significance_level": 0.05,
                "save_params": "acf_params.json",
                "load_params": "acf_params.json"
            }
        }
    ],
    "remote_log": "http://remote-log-server/api/logs"
}
```

#### Example 5: Post-Feature Selector Plugin Configuration

```json
{
    "csv_file": "path/to/input.csv",
    "output_file": "path/to/output.csv",
    "plugins": [
        {
            "name": "post_feature_selector_plugin",
            "params": {
                "method": "lasso",
                "alpha": 1.0,
                "l1_ratio": 0.5,
                "model_type": "lstm",
                "timesteps": 1,
                "features": 1,
                "save_params": "feature_selection_params.json",
                "load_params": "feature_selection_params.json"
            }
        }
    ],
    "remote_log": "http://remote-log-server/api/logs"
}
```

### File Structure:

```md
preprocessor/
│
├── app/                           # Main application package
│   ├── __init__.py                    # Initializes the Python package
│   ├── main.py                        # Entry point for the application
│   ├── config.py                      # Configuration settings for the app
│   ├── cli.py                         # Command line interface handling
│   ├── data_handler.py                # Module to handle data loading
│   ├── default_plugin.py              # Default plugin (normalizer)
│   └── plugins/                       # Plugins directory
│       ├── __init__.py                # Makes plugins a Python package
│       ├── plugin_unbiaser.py
│       ├── plugin_trimmer.py
│       ├── plugin_feature_selector_pre.py
│       └── plugin_feature_selector_post.py
│
├── tests/                             # Test modules for your application
│   ├── __init__.py                         # Initializes the Python package for tests
│   ├── test_preprocessor.py                # Tests for preprocessor functionality
│   ├── datasets/                           # Test datasets directory
│   └── configs/                            # Test configurations directory
│
├── setup.py                           # Setup file for the package installation
├── README.md                          # Project description and instructions
├── requirements.txt                   # External packages needed
└── .gitignore                         # Specifies intentionally untracked files to ignore
```




