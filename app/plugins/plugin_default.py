import pandas as pd
import json
import os
import numpy as np

class Plugin:
    """
    Plugin to preprocess the dataset for feature extraction.
    """
    # Define the parameters for this plugin and their default values
    plugin_params = {
        'input_column_order': ["d", "o", "h", "l", "c", "v"],
        'output_column_order': ["d", "o", "l", "h", "c", "v"],
        'validation_proportion': 0.5,
        'dataset_prefix': "x_",
        'target_prefix': "y_",
        'target_column': 4
    }

    # Define the debug variables for this plugin
    plugin_debug_vars = ['min_val', 'max_val', 'range', 'method', 'mae_per_pip']

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
            dict: Debug information including min_val, max_val, range, method, and mae_per_pip.
        """
        debug_info = {var: None for var in self.plugin_debug_vars}
        if self.normalization_params:
            target_column_name = self.params['output_column_order'][self.params['target_column']]
            debug_info['min_val'] = self.normalization_params['min'][target_column_name]
            debug_info['max_val'] = self.normalization_params['max'][target_column_name]
            debug_info['range'] = self.normalization_params['range']
            debug_info['method'] = 'min-max'
            debug_info['mae_per_pip'] = self.calculate_mae_for_pips(
                1, debug_info['min_val'], debug_info['max_val'], debug_info['range']
            )
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

    def normalize_data(self, data):
        """
        Normalize the data using min-max normalization.

        Args:
            data (pd.DataFrame): The input data to be normalized.

        Returns:
            pd.DataFrame: The normalized data.
        """
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        min_val = data[numeric_columns].min()
        max_val = data[numeric_columns].max()
        range_vals = (-1, 1)
        self.normalization_params = {
            'min': min_val.to_dict(),
            'max': max_val.to_dict(),
            'range': range_vals
        }
        normalized_data = (data[numeric_columns] - min_val) / (max_val - min_val) * (range_vals[1] - range_vals[0]) + range_vals[0]
        data[numeric_columns] = normalized_data
        return data

    def process(self, data):
        """
        Process the data by reordering columns, splitting into training and validation datasets,
        normalizing the training and validation datasets, and saving target columns.

        Args:
            data (pd.DataFrame): The input data to be processed.

        Returns:
            dict: Contains processed training and validation datasets.
        """
        # Reorder columns based on input and output orders
        input_column_order = self.params['input_column_order']
        output_column_order = self.params['output_column_order']

        # Create a mapping from input column order to output column order
        column_mapping = {input_column_order[i]: output_column_order[i] for i in range(len(input_column_order))}

        # Reorder columns based on the mapping
        data.columns = input_column_order  # Set columns to input order first
        data = data[[column_mapping[col] for col in data.columns]]  # Reorder to output order
        print("Columns reordered based on output_column_order")

        # Split into training and validation datasets
        validation_proportion = self.params['validation_proportion']
        validation_size = int(len(data) * validation_proportion)
        training_data = data[:-validation_size]
        validation_data = data[-validation_size:]

        # Normalize the training dataset
        training_data = self.normalize_data(training_data)

        # Normalize the validation dataset using training normalization parameters
        numeric_columns = validation_data.select_dtypes(include=[np.number]).columns
        min_val = pd.Series(self.normalization_params['min'])
        max_val = pd.Series(self.normalization_params['max'])
        range_vals = self.normalization_params['range']
        normalized_validation_data = (validation_data[numeric_columns] - min_val) / (max_val - min_val) * (range_vals[1] - range_vals[0]) + range_vals[0]
        validation_data[numeric_columns] = normalized_validation_data

        # Extract and save the target columns for training and validation datasets
        target_column_index = self.params['target_column']
        target_column_name = output_column_order[target_column_index]
        target_prefix = self.params['target_prefix']

        training_target = training_data[[target_column_name]]
        validation_target = validation_data[[target_column_name]]

        training_target.to_csv(f"{target_prefix}training.csv", index=False)
        validation_target.to_csv(f"{target_prefix}validation.csv", index=False)

        # Save debug information for the target column
        debug_info = self.get_debug_info()
        with open(f"{target_prefix}debug_info.json", 'w') as f:
            json.dump(debug_info, f)

        print("Target columns and debug information saved")

        return {'training_data': training_data, 'validation_data': validation_data}

# Example usage
if __name__ == "__main__":
    plugin = Plugin()
    data = pd.read_csv('path_to_your_csv.csv', header=None)
    processed_data = plugin.process(data)
    print(processed_data)
