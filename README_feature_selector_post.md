# Post-Feature Selector Plugin for Preprocessor

## Description

The Post-Feature Selector Plugin is used to perform feature selection on the dataset after initial preprocessing in the Preprocessor application. It supports five methods: LASSO, Elastic Net, Mutual Information, Cross-Validation with LSTM/CNN, and Boruta Algorithm. This plugin can save and load feature selection parameters, making it reusable for consistent preprocessing across different datasets.

## Parameters

The plugin accepts the following parameters:

- **method** (str): The method for feature selection. Options are `'lasso'`, `'elastic_net'`, `'mutual_info'`, `'cross_val'`, and `'boruta'`. Default is `'lasso'`.
- **save_params** (str): The file path to save the feature selection parameters. Default is `'feature_selection_params.json'`.
- **load_params** (str): The file path to load the feature selection parameters. If provided, the parameters will be loaded from this file instead of being specified in the command line arguments.
- **alpha** (float): Regularization strength for LASSO and Elastic Net. Default is `1.0`.
- **l1_ratio** (float): The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. Default is `0.5`.
- **model_type** (str): The type of model to use for Cross-Validation (`'lstm'` or `'cnn'`). Default is `'lstm'`.
- **timesteps** (int): The number of timesteps for the LSTM/CNN model. Default is `1`.
- **features** (int): The number of features for the LSTM/CNN model. Default is `1`.

## Methods

### LASSO

LASSO (Least Absolute Shrinkage and Selection Operator) is a linear regression method that uses L1 regularization to enhance the prediction accuracy and interpretability of the model it produces. It effectively selects a subset of the features by shrinking the coefficients of less important features to zero.

### Elastic Net

Elastic Net is a regularized regression method that linearly combines the L1 and L2 penalties of the LASSO and Ridge methods. It is useful when there are multiple correlated features.

### Mutual Information

Mutual Information measures the amount of information obtained about one random variable through another random variable. It helps in identifying the dependency between the input features and the target variable.

### Cross-Validation with LSTM/CNN

Cross-Validation with feature importance from LSTM/CNN involves using deep learning models to determine the importance of each feature. LSTM (Long Short-Term Memory) and CNN (Convolutional Neural Network) models are trained, and feature importance is derived based on the model's performance.

### Boruta Algorithm

Boruta is an all-relevant feature selection method that works by iteratively training a Random Forest classifier and comparing the importance of real features to that of shadow features (randomly permuted copies of the original features). Features that have higher importance than the best shadow feature are selected.

## Usage

### From Command Line

You can use the Post-Feature Selector Plugin from the Preprocessor application via command line parameters. Below are some examples:

### Example 1: Using Default Parameters (LASSO)

```bash
python app/main.py --config config.json --plugin post_feature_selector_plugin
```

### Example 2: Using Elastic Net Feature Selection and Saving Parameters

```bash
python app/main.py --config config.json --plugin post_feature_selector_plugin --method elastic_net --save_params elastic_net_params.json
```

### Example 3: Loading Pre-Saved Feature Selection Parameters

```bash
python app/main.py --config config.json --plugin post_feature_selector_plugin --load_params feature_selection_params.json
```

### Configuration File Example

Here is an example of a JSON configuration file (config.json) that can be used with the Preprocessor application:

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

### Output Configuration File Examples

#### LASSO Feature Selection Parameters (lasso_params.json)

```json
{
    "method": "lasso",
    "selected_features": ["feature1", "feature2", "feature3"]
}
```

#### Elastic Net Feature Selection Parameters (elastic_net_params.json)

```json
{
    "method": "elastic_net",
    "selected_features": ["feature1", "feature2", "feature3"]
}
```

#### Mutual Information Feature Selection Parameters (mutual_info_params.json)

```json
{
    "method": "mutual_info",
    "selected_features": ["feature1", "feature2", "feature3"]
}
```

#### Cross-Validation Feature Selection Parameters (cross_val_params.json)

```json
{
    "method": "cross_val",
    "selected_features": ["feature1", "feature2", "feature3"]
}
```

#### Boruta Algorithm Feature Selection Parameters (boruta_params.json)

```json
{
    "method": "boruta",
    "selected_features": ["feature1", "feature2", "feature3"]
}
```