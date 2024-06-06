import pandas as pd
import json
import os
import numpy as np

class DefaultPlugin:
    """
    Default Plugin to normalize the dataset using various methods.
    """
    # Define the parameters for this plugin and their default values
    plugin_params = {
        'method': 'min-max',
        'range': (-1, 1),
        'save_params': None,  # Change default to None
        'load_params': None
    }

    # Define the debug variables for this plugin
    plugin_debug_vars = ['min_val', 'max_val', 'mean', 'std']

    def __init__(self):
        """
        Initialize the Plugin with default parameters.
        """
        self.params = self.plugin_params.copy()
        self.normalization_params = None

    def set_params(self, **kwargs):
        """
        Set the parameters for the plugin.

        Args:
            **kwargs: Arbitrary keyword arguments for plugin parameters.
        """
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        """
        Get debug information for the plugin.

        Returns:
            dict: Debug information including min_val, max_val, mean, and std.
        """
        debug_info = {var: None for var in self.plugin_debug_vars}
        if self.normalization_params:
            if self.normalization_params['method'] == 'z-score':
                debug_info['mean'] = self.normalization_params['mean']
                debug_info['std'] = self.normalization_params['std']
            elif self.normalization_params['method'] == 'min-max':
                debug_info['min_val'] = self.normalization_params['min']
                debug_info['max_val'] = self.normalization_params['max']
        return debug_info

    def add_debug_info(self, debug_info):
        """
        Add debug information to the given dictionary.

        Args:
            debug_info (dict): The dictionary to add debug information to.
        """
        debug_info.update(self.get_debug_info())

    def process(self, data):
        """
        Normalize the data using the specified parameters or calculate them if not provided.

        Args:
            data (pd.DataFrame): The input data to be processed.

        Returns:
            pd.DataFrame: The normalized data.
        """
        method = self.params.get('method', 'min-max')
        save_params = self.params.get('save_params')
        load_params = self.params.get('load_params')
        range_vals = self.params.get('range', (-1, 1))

        # Retain the date column
        date_column = data.select_dtypes(include=[np.datetime64]).columns
        non_numeric_data = data[date_column]
        
        # Select only numeric columns for processing
        numeric_data = data.select_dtypes(include=[np.number])

        if load_params and os.path.exists(load_params):
            with open(load_params, 'r') as f:
                self.normalization_params = json.load(f)

        if self.normalization_params is None:
            if method == 'z-score':
                mean = numeric_data.mean()
                std = numeric_data.std()
                self.normalization_params = {'method': 'z-score', 'mean': mean.to_dict(), 'std': std.to_dict()}
                normalized_data = (numeric_data - mean) / std
            elif method == 'min-max':
                min_val = numeric_data.min()
                max_val = numeric_data.max()
                self.normalization_params = {'method': 'min-max', 'min': min_val.to_dict(), 'max': max_val.to_dict(), 'range': range_vals}
                normalized_data = (numeric_data - min_val) / (max_val - min_val) * (range_vals[1] - range_vals[0]) + range_vals[0]
            else:
                raise ValueError(f"Unknown normalization method: {method}")

            # Save normalization parameters if save_params path is provided and not None
            if save_params:
                with open(save_params, 'w') as f:
                    json.dump(self.normalization_params, f)
        else:
            if self.normalization_params['method'] == 'z-score':
                mean = pd.Series(self.normalization_params['mean'])
                std = pd.Series(self.normalization_params['std'])
                normalized_data = (numeric_data - mean) / std
            elif self.normalization_params['method'] == 'min-max':
                min_val = pd.Series(self.normalization_params['min'])
                max_val = pd.Series(self.normalization_params['max'])
                range_vals = self.normalization_params.get('range', (-1, 1))
                normalized_data = (numeric_data - min_val) / (max_val - min_val) * (range_vals[1] - range_vals[0]) + range_vals[0]
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_params['method']}")

        # Combine numeric data back with non-numeric data (e.g., date columns)
        result = pd.concat([non_numeric_data, normalized_data], axis=1)

        # Debug information
        for column in data.columns:
            if column in non_numeric_data.columns:
                print(f"Column '{column}' is non-numeric and was not processed.")
            elif column in numeric_data.columns:
                print(f"Column '{column}' was successfully processed.")
            else:
                print(f"Column '{column}' was not found in the processed data.")

        return result
