
# Preprocessor 

The Preprocessor project is a flexible and modular application for preprocessing time series data. It supports dynamic loading of plugins for various preprocessing tasks such as normalization, unbiasing, trimming, and feature selection. Each plugin can save and load its parameters for consistent preprocessing across different datasets.

## Installation Instructions

To install and set up the Preprocessor application, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/harveybc/preprocessor.git
    cd preprocessor
    ```

2. **Create and Activate a Virtual Environment**:
    - **Using `venv` (Python 3.3+)**:
        ```bash
        python -m venv env
        source env/bin/activate  # On Windows use `env\Scripts\activate`
        ```

    - **Using `conda`**:
        ```bash
        conda create --name preprocessor_env python=3.9
        conda activate preprocessor_env
        ```

3. **Install Dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Build the Package**:
    ```bash
    python -m build
    ```

5. **Install the Package**:
    ```bash
    pip install .
    ```

6. **Run the Preprocessor**:
    - On Windows, run the following command to verify installation (it generates an example output file csv_output.csv):
        ```bash
        preprocessor.bat tests\data\EURUSD_5m_2006_2007.csv --plugin feature_selector --method select_single --single 0
        ```

    - On Linux, run:
        ```bash
        sh preprocessor.sh tests\data\EURUSD_5m_2006_2007.csv --plugin feature_selector --method select_single --single 0
        ```

7. **Run Tests (Optional, requires external repo)**:
For pasing remote tests, requires an instance of [harveybc/data-logger](https://github.com/harveybc/data-logger)
    - On Windows, run the following command to run the tests:
        ```bash
        set_env.bat
        pytest
        ```

    - On Linux, run:
        ```bash
        sh ./set_env.sh
        pytest
        ```

8. **Generate Documentation (Optional)**:
    - Run the following command to generate code documentation in HTML format in the docs directory:
        ```bash
        pdoc --html -o docs app
        ```

This command should display the help message for the Preprocessor application, confirming that the installation was successful.

## Examples of Use

You can use the Preprocessor application from the command line with various plugins. Below are some examples:

### Default Plugin (Normalizer Plugin)

The Normalizer Plugin is used to normalize data using methods such as z-score and min-max normalization.

[Read more about the Normalizer Plugin](https://github.com/harveybc/preprocessor/blob/master/README_normalizer.md)

#### Using Z-Score Normalization

```bash
preprocessor.bat  path/to/input.csv --plugin default_plugin --method z-score --save_config save_config.json
```

#### Using Min-Max Normalization

```bash
preprocessor.bat  path/to/input.csv --plugin default_plugin --method min-max --range 0 1 --save_config save_config.json
```
### Cleaner Plugin

The Cleaner Plugin performs data cleaning during preprocessing. It has two main methods: missing_values and outlier.

Parameters
method: The method to use for cleaning (missing_values or outlier).
period: The period in minutes for continuity checking. Used with missing_values method.
outlier_threshold: The threshold for outlier detection. Used with outlier method.
solve_missing: Boolean flag to solve missing values by interpolating the average of previous and next ticks.
delete_outliers: Boolean flag to delete rows with outliers.
interpolate_outliers: Boolean flag to interpolate outlier values.
delete_nan: Boolean flag to delete rows with NaN values.
interpolate_nan: Boolean flag to interpolate NaN values.
headers: Boolean flag to indicate if the input CSV contains headers.
Example Usage
Detect and handle missing values with a 5-minute period, solving missing values:

sh
Copiar código
preprocessor.bat path/to/input.csv --plugin cleaner --method missing_values --period 5 --solve_missing
Detect and handle outliers with a threshold of 3, deleting outliers:

sh
Copiar código
preprocessor.bat path/to/input.csv --plugin cleaner --method outlier --outlier_threshold 3 --delete_outliers
Detect and handle outliers with a threshold of 3, interpolating outliers:

sh
Copiar código
preprocessor.bat path/to/input.csv --plugin cleaner --method outlier --outlier_threshold 3 --interpolate_outliers
Detect missing values with a 5-minute period without solving them:

sh
Copiar código
preprocessor.bat path/to/input.csv --plugin cleaner --method missing_values --period 5
### Unbiaser Plugin

The Unbiaser Plugin removes bias from time series data using moving average (MA) and exponential moving average (EMA) methods.

[Read more about the Unbiaser Plugin](https://github.com/harveybc/preprocessor/blob/master/README_unbiaser.md)

#### Using Moving Average Method

```bash
preprocessor.bat  path/to/input.csv --plugin unbiaser_plugin --method ma --window_size 5 --save_config save_config.json
```

#### Using Exponential Moving Average Method

```bash
preprocessor.bat  path/to/input.csv --plugin unbiaser_plugin --method ema --ema_alphas 0.2 --save_config save_config.json
Trimmer Plugin
```

### Removing Specific Rows and Columns

The Trimmer Plugin removes specified columns and rows from the dataset.

[Read more about the Trimmer Plugin](https://github.com/harveybc/preprocessor/blob/master/README_trimmer.md)

```bash
preprocessor.bat  path/to/input.csv --plugin trimmer_plugin --rows 0 1 2 --columns 0 1 --save_config path/to/save_config.json
```

### Pre-Feature Selector Plugin

Performs the initial screening for redundant or non-informative data, the Pre-FeatureExtraction Feature Selector Plugin performs feature selection using methods such as ACF, PACF, and Granger Causality Test. 

This selection is meant to be performed before feature extraction or other dimensionality reduction technique.

[Read more about the Pre-Feature Selector Plugin](https://github.com/harveybc/preprocessor/blob/master/README_feature_selector_pre.md)

#### Using Autocorrelation Function (ACF) Method

```bash
preprocessor.bat  path/to/input.csv --plugin pre_feature_selector_plugin --method acf --max_lag 5 --significance_level 0.05 --save_config path/to/save_config.json
```

### Post-Feature Selector Plugin

The Post-FeatureEtraction Feature Selector Plugin performs feature selection after initial preprocessing and feature extraction using methods such as LASSO, Elastic Net, Mutual Information, Cross-Validation with LSTM/CNN, and Boruta Algorithm.

This selection is meant to be performed after feature extraction or other dimensionality reduction technique.

[Read more about the Post-Feature Selector Plugin](https://github.com/harveybc/preprocessor/blob/master/README_feature_selector_post.md)

#### Using LASSO Method

```bash
preprocessor.bat  path/to/input.csv --plugin post_feature_selector_plugin --method lasso --alpha 1.0 --save_config path/to/save_config.json
```

#### Using Cross-Validation with LSTM Model

```bash
preprocessor.bat  path/to/input.csv --plugin post_feature_selector_plugin --method cross_val --model_type lstm --timesteps 10 --features 1 --save_config path/to/save_config.json
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




