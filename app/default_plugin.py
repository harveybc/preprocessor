import pandas as pd
import json
import os
import numpy as np

class DefaultPlugin:
    def __init__(self):
        self.normalization_params = None

    def process(self, data, method='z-score', save_params='normalization_params.json', load_params=None, range=(0, 1)):
        """
        Normalize the data using the specified parameters or calculate them if not provided.

        Args:
            data (pd.DataFrame): The input data to be processed.
            method (str): The normalization method to use.
            save_params (str): Path to save the normalization parameters.
            load_params (str): Path to load the normalization parameters.
            range (tuple): The range for min-max normalization.

        Returns:
            pd.DataFrame: The normalized data.
        """
        # Detect and retain the date column
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
                self.normalization_params = {'method': 'min-max', 'min': min_val.to_dict(), 'max': max_val.to_dict(), 'range': range}
                normalized_data = (numeric_data - min_val) / (max_val - min_val) * (range[1] - range[0]) + range[0]
            else:
                raise ValueError(f"Unknown normalization method: {method}")

            # Save normalization parameters if save_params path is provided
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
                range = self.normalization_params.get('range', (0, 1))
                normalized_data = (numeric_data - min_val) / (max_val - min_val) * (range[1] - range[0]) + range[0]
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
