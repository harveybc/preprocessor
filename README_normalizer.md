# Default Plugin for Preprocessor

## Description

The Default Plugin is a normalization plugin for the Preprocessor application. It supports two normalization methods: z-score and min-max normalization. This plugin can save and load normalization parameters, making it reusable for consistent preprocessing across different datasets.

## Parameters

The plugin accepts the following parameters:

- **method** (str): The normalization method to use. Options are `'z-score'` and `'min-max'`. Default is `'z-score'`.
- **save_params** (str): The file path to save the normalization parameters. Default is `'normalization_params.json'`.
- **load_params** (str): The file path to load the normalization parameters. If provided, the parameters will be loaded from this file instead of being calculated from the data.
- **range** (tuple): The desired range for min-max normalization. Options are `(0, 1)` and `(-1, 1)`. Default is `(0, 1)`.

## Methods

### Z-Score Normalization

Z-score normalization, also known as standard score normalization, transforms the data to have a mean of 0 and a standard deviation of 1. This method is useful when the data follows a Gaussian distribution. The formula for z-score normalization is:

$`Z = \frac{(X - \mu)}{\sigma}`$

where:
- $`X`$ is the original data point
- $`\mu`$ is the mean of the data
- $`\sigma`$ is the standard deviation of the data

### Min-Max Normalization

Min-max normalization scales the data to a fixed range, typically between 0 and 1. This method is useful when the data does not follow a Gaussian distribution and you want to preserve the relationships between data points. The formula for min-max normalization is:

$`X' = \frac{(X - X_{\text{min}})}{(X_{\text{max}} - X_{\text{min}})}`$

where:
- $`X`$ is the original data point
- $`X_{\text{min}}`$ is the minimum value in the data
- $`X_{\text{max}}`$ is the maximum value in the data

To scale the data between -1 and 1, the formula is:

$`X' = \frac{(X - X_{\text{min}})}{(X_{\text{max}} - X_{\text{min}})} \times (1 - (-1)) + (-1)`$

## Usage

### From Command Line

You can use the Default Plugin from the Preprocessor application via command line parameters. Below are some examples:

### Example 1: Using Default Parameters (Z-Score Normalization)

```bash
python app/main.py --config config.json --plugin default_plugin
```

### Example 2: Using Min-Max Normalization and Saving Parameters

```bash
python app/main.py --config config.json --plugin default_plugin --method min-max --range 0 1 --save_params min_max_params.json
json
```

### Example 3: Loading Pre-Saved Normalization Parameters

```bash
python app/main.py --config config.json --plugin default_plugin --load_params min_max_params.json
```
### Plugin Configuration File Example

```json
{
    "csv_file": "path/to/input.csv",
    "output_file": "path/to/output.csv",
    "plugins": [
        {
            "name": "default_plugin",
            "params": {
                "method": "z-score",
                "save_params": "normalization_params.json"
            }
        }
    ],
    "remote_log": "http://remote-log-server/api/logs"
}
```

### Min-Max Configuration File Example

```json
{
    "method": "min-max",
    "min": {
        "feature1": 0.0,
        "feature2": -1.0,
        "feature3": 10.0,
        "feature4": 5.0
    },
    "max": {
        "feature1": 1.0,
        "feature2": 1.0,
        "feature3": 50.0,
        "feature4": 15.0
    }
}
```

### Z-Score (standardization) Configuration File Example

```json
{
    "method": "z-score",
    "mean": {
        "feature1": 0.5,
        "feature2": 0.0,
        "feature3": 30.0,
        "feature4": 10.0
    },
    "std": {
        "feature1": 0.1,
        "feature2": 0.5,
        "feature3": 10.0,
        "feature4": 2.0
    }
}
```



