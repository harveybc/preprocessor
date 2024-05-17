import pandas as pd
import json
import os

class DefaultPlugin:
    def __init__(self):
        self.normalization_params = None

    def process(self, data, **params):
        """
        Normalize the data using the specified parameters or calculate them if not provided.

        Args:
            data (pd.DataFrame): The input data to be processed.
            **params: Additional parameters for the normalization process.

        Returns:
            pd.DataFrame: The normalized data.
        """
        method = params.get('method', 'z-score')
        save_params_path = params.get('save_params', 'normalization_params.json')
        load_params_path = params.get('load_params')

        if load_params_path and os.path.exists(load_params_path):
            with open(load_params_path, 'r') as f:
                self.normalization_params = json.load(f)

        if self.normalization_params is None:
            if method == 'z-score':
                mean = data.mean()
                std = data.std()
                self.normalization_params = {'method': 'z-score', 'mean': mean.to_dict(), 'std': std.to_dict()}
                normalized_data = (data - mean) / std
            elif method == 'min-max':
                min_val = data.min()
                max_val = data.max()
                self.normalization_params = {'method': 'min-max', 'min': min_val.to_dict(), 'max': max_val.to_dict()}
                normalized_data = (data - min_val) / (max_val - min_val)
            else:
                raise ValueError(f"Unknown normalization method: {method}")

            # Save normalization parameters if save_params_path is provided
            if save_params_path:
                with open(save_params_path, 'w') as f:
                    json.dump(self.normalization_params, f)
        else:
            if self.normalization_params['method'] == 'z-score':
                mean = pd.Series(self.normalization_params['mean'])
                std = pd.Series(self.normalization_params['std'])
                normalized_data = (data - mean) / std
            elif self.normalization_params['method'] == 'min-max':
                min_val = pd.Series(self.normalization_params['min'])
                max_val = pd.Series(self.normalization_params['max'])
                normalized_data = (data - min_val) / (max_val - min_val)
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_params['method']}")

        return normalized_data
