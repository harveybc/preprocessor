# Pre-Feature Selector Plugin for Preprocessor

## Description

The Pre-Feature Selector Plugin is used to perform initial screening for feature selection on time series data in the Preprocessor application. It supports three methods: Autocorrelation Function (ACF), Partial Autocorrelation Function (PACF), and Granger Causality Test. This plugin can save and load feature selection parameters, making it reusable for consistent preprocessing across different datasets.

## Parameters

The plugin accepts the following parameters:

- **method** (str): The method for feature selection. Options are `'acf'`, `'pacf'`, and `'granger'`. Default is `'granger'`.
- **save_params** (str): The file path to save the feature selection parameters. Default is `'feature_selection_params.json'`.
- **load_params** (str): The file path to load the feature selection parameters. If provided, the parameters will be loaded from this file instead of being specified in the command line arguments.
- **max_lag** (int): Maximum lag for the Granger causality test. Default is `5`.
- **significance_level** (float): Significance level for the statistical tests. Default is `0.05`.

## Methods

### Autocorrelation Function (ACF)

The Autocorrelation Function measures the correlation between a time series and its lagged values. It helps identify repeating patterns and the presence of seasonal effects in the data. Features with high autocorrelation values are often useful for time series forecasting.

### Partial Autocorrelation Function (PACF)

The Partial Autocorrelation Function measures the correlation between a time series and its lagged values while controlling for the values of the time series at all shorter lags. PACF helps identify the direct relationship between a time series and its lagged values, filtering out the indirect effects.

### Granger Causality Test

The Granger Causality Test determines whether one time series can predict another time series. It assesses whether past values of one time series contain information that helps predict future values of another time series. Features that Granger-cause the target variable are useful predictors for time series forecasting.

## Usage

### From Command Line

You can use the Pre-Feature Selector Plugin from the Preprocessor application via command line parameters. Below are some examples:

### Example 1: Using Default Parameters (Granger Causality Test)

```bash
python app/main.py --config config.json --plugin pre_feature_selector_plugin
```
### Example 2: Using ACF Feature Selection and Saving Parameters

bash
python app/main.py --config config.json --plugin pre_feature_selector_plugin --method acf --save_params acf_params.json

### Example 3: Loading Pre-Saved Feature Selection Parameters

bash
python app/main.py --config config.json --plugin pre_feature_selector_plugin --load_params feature_selection_params.json

### Configuration File Example

Here is an example of a JSON configuration file (config.json) that can be used with the Preprocessor application:

json
{
    "csv_file": "path/to/input.csv",
    "output_file": "path/to/output.csv",
    "plugins": [
        {
            "name": "pre_feature_selector_plugin",
            "params": {
                "method": "granger",
                "max_lag": 5,
                "significance_level": 0.05,
                "save_params": "feature_selection_params.json",
                "load_params": "feature_selection_params.json"
            }
        }
    ],
    "remote_log": "http://remote-log-server/api/logs"
}

### Output Configuration File Examples

#### ACF Feature Selection Parameters (acf_params.json)

json
{
    "method": "acf",
    "selected_features": ["feature1", "feature2", "feature3"]
}

#### PACF Feature Selection Parameters (pacf_params.json)

json
{
    "method": "pacf",
    "selected_features": ["feature1", "feature2", "feature3"]
}

#### Granger Causality Test Parameters (feature_selection_params.json)

json
{
    "method": "granger",
    "selected_features": ["feature1", "feature2", "feature3"]
}


