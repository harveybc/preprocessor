import pandas as pd
import json
import os
import numpy as np

class Plugin:
    """1.60139	1.16481

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
    plugin_debug_vars = ['min_val', 'max_val', 'mean', 'std', 'range', 'method', 'mae_per_pip']

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
                debug_info['method'] = 'z-score'
                first_col = list(debug_info['std'].keys())[0]
                debug_info['mae_per_pip'] = self.calculate_mae_for_pips_z(1, debug_info['mean'][first_col], debug_info['std'][first_col])

            elif self.normalization_params['method'] == 'min-max':
                debug_info['min_val'] = self.normalization_params['min']
                debug_info['max_val'] = self.normalization_params['max']
                debug_info['range'] = self.normalization_params['range']
                debug_info['method'] = 'min-max'
                # Assuming we are interested in the first column for MAE per pip calculation
                first_col = 0
                debug_info['mae_per_pip'] = self.calculate_mae_for_pips(1, debug_info['min_val'][first_col], debug_info['max_val'][first_col], debug_info['range'])
        return debug_info

    def add_debug_info(self, debug_info):
        """
        Add debug information to the given dictionary.

        Args:
            debug_info (dict): The dictionary to add debug information to.
        """
        debug_info.update(self.get_debug_info())

    def calculate_mae_for_pips(self, pips, original_min, original_max, normalized_range=(-1, 1)):
        """
        Calculate the MAE in the normalized range corresponding to the given number of pips.

        Parameters:
        - pips (float): The number of pips.
        - original_min (float): The minimum value of the original data.
        - original_max (float): The maximum value of the original data.
        - normalized_range (tuple): The range of the normalized data. Default is (-1, 1).

        Returns:
        - float: The MAE in the normalized range corresponding to the given number of pips.
        """
        # Step 1: Calculate the original range
        original_range = original_max - original_min
        
        # Step 2: Calculate the span of the normalized range
        normalized_range_span = normalized_range[1] - normalized_range[0]
        
        # Step 3: Calculate the conversion factor from the original range to the normalized range
        conversion_factor = normalized_range_span / original_range
        
        # Step 4: Calculate the value of 1 pip in the normalized range
        pip_value_in_normalized_range = pips * 0.0001 * conversion_factor
        
        return pip_value_in_normalized_range
    
    def calculate_mae_for_pips_z(self, pips, mean, std):
        """
        Calculate the MAE in the normalized range corresponding to the given number of pips using z-score normalization.

        Parameters:
        - pips (float): The number of pips.
        - mean (float): The mean value of the original data.
        - std (float): The standard deviation of the original data.

        Returns:
        - float: The MAE in the normalized range corresponding to the given number of pips.
        """
        # Step 1: Calculate the value of pips in the original scale
        pip_value_in_original_scale = pips * 0.0001

        # Step 2: Normalize the pip value using z-score normalization
        pip_value_in_normalized_scale = pip_value_in_original_scale / std
        
        return pip_value_in_normalized_scale

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
        #date_column = data.select_dtypes(include=[np.datetime64]).columns
        #non_numeric_data = data[date_column]
        
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
        # result = pd.concat([non_numeric_data, normalized_data], axis=1)
        result = normalized_data

        # Debug information
        #for column in data.columns:
        #    if column in non_numeric_data.columns:
        #        print(f"Column '{column}' is non-numeric and was not processed.")
        #    elif column in numeric_data.columns:
        #        print(f"Column '{column}' was successfully processed.")
        #    else:
        #        print(f"Column '{column}' was not found in the processed data.")

        return result

# Example usage
if __name__ == "__main__":
    plugin = DefaultPlugin()
    data = pd.read_csv('path_to_your_csv.csv')
    processed_data = plugin.process(data)
    print(processed_data)
    
    # Calculate MAE for 1 pip with example values
    mae_for_pip = plugin.calculate_mae_for_pips(1, 1.24665, 1.3676, (-1, 1))
    print(f"MAE for 1 pip: {mae_for_pip}")
