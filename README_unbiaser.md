# Unbiaser Plugin for Preprocessor

## Description

The Unbiaser Plugin is used to remove bias from time series data in the Preprocessor application. It supports two methods: moving average (MA) and exponential moving average (EMA) bias removal. This plugin can save and load bias removal parameters, making it reusable for consistent preprocessing across different datasets.

## Parameters

The plugin accepts the following parameters:

- **method** (str): The bias removal method to use. Options are `'ma'` and `'ema'`. Default is `'ma'`.
- **window_sizes** (list): List of window sizes for the moving average method for each column. Default is `[3]`.
- **ema_alphas** (list): List of alpha parameters for the exponential moving average method for each column. Default is `[0.3]`.
- **save_params** (str): The file path to save the bias removal parameters. Default is `'unbias_params.json'`.
- **load_params** (str): The file path to load the bias removal parameters. If provided, the parameters will be loaded from this file instead of being calculated from the data.

## Usage

### From Command Line

You can use the Unbiaser Plugin from the Preprocessor application via command line parameters. Below are some examples:

### Example 1: Using Default Parameters (Moving Average Bias Removal)

```bash
python app/main.py --config config.json --plugin unbiaser_plugin
```
### Example 2: Using EMA Bias Removal and Saving Parameters
python app/main.py --config config.json --plugin unbiaser_plugin --method ema --ema_alphas 0.2 --save_params ema_params.json

### Example 3: Loading Pre-Saved Bias Removal Parameters
```bash
python app/main.py --config config.json --plugin unbiaser_plugin --load_params unbias_params.json
```

## Configuration File Example

Here is an example of a JSON configuration file (config.json) that can be used with the Preprocessor application:

```json
{
    "csv_file": "path/to/input.csv",
    "output_file": "path/to/output.csv",
    "plugins": [
        {
            "name": "unbiaser_plugin",
            "params": {
                "method": "ma",
                "window_sizes": [3],
                "save_params": "unbias_params.json",
                "load_params": "unbias_params.json"
            }
        }
    ],
    "remote_log": "http://remote-log-server/api/logs"
}
```

## Output Configuration File Examples

### Moving Average (MA) Output Configuration File Example
```json
{
    "method": "ma",
    "window_sizes": [3]
}
```

### Exponential Moving Average (EMA) Output Configuration File Example
```json
{
    "method": "ema",
    "ema_alphas": [0.3]
}
```


