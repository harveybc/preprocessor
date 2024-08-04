import pandas as pd
import json
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
        original_range = original_max - original_min
        normalized_range_span = normalized_range[1] - normalized_range[0]
        conversion_factor = normalized_range_span / original_range
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
        data.loc[:, numeric_columns] = normalized_data  # Use .loc to avoid SettingWithCopyWarning
        return data

    def process(self, data):
        """
        Process the data by reordering columns, splitting into training and validation datasets,
        normalizing the training and validation datasets, and saving target columns.

        Args:
            data (pd.DataFrame): The input data to be processed.

        Returns:
            pd.DataFrame: The processed data.
        """
        # Step 1: Reorder columns based on input and output orders
        input_column_order = self.params['input_column_order']
        output_column_order = self.params['output_column_order']

        data.columns = input_column_order  # Set columns to input order first
        data = data[output_column_order]
        print(f"Step 1: Columns reordered based on output_column_order. New order: {list(data.columns)}")

        # Step 2: Split into training and validation datasets
        validation_proportion = self.params['validation_proportion']
        validation_size = int(len(data) * validation_proportion)
        training_data = data.iloc[:-validation_size].copy()
        validation_data = data.iloc[-validation_size:].copy()
        print(f"Step 2: Split data into training and validation datasets.")
        print(f"Training data shape: {training_data.shape}")
        print(f"Validation data shape: {validation_data.shape}")

        # Step 3: Normalize the training dataset
        training_data = self.normalize_data(training_data)
        print(f"Step 3: Normalized the training dataset.")
        print(f"Normalized training data shape: {training_data.shape}")

        # Normalize the validation dataset using training normalization parameters
        numeric_columns = validation_data.select_dtypes(include=[np.number]).columns
        min_val = pd.Series(self.normalization_params['min'])
        max_val = pd.Series(self.normalization_params['max'])
        range_vals = self.normalization_params['range']
        normalized_validation_data = (validation_data[numeric_columns] - min_val) / (max_val - min_val) * (range_vals[1] - range_vals[0]) + range_vals[0]
        validation_data.loc[:, numeric_columns] = normalized_validation_data  # Use .loc to avoid SettingWithCopyWarning
        print(f"Step 4: Normalized the validation dataset.")
        print(f"Normalized validation data shape: {validation_data.shape}")

        # Extract and save the target columns for training and validation datasets
        target_column_index = self.params['target_column']
        target_column_name = output_column_order[target_column_index]
        target_prefix = self.params['target_prefix']

        training_target = training_data[[target_column_name]]
        validation_target = validation_data[[target_column_name]]

        training_target_file = f"{target_prefix}training.csv"
        validation_target_file = f"{target_prefix}validation.csv"

        training_target.to_csv(training_target_file, index=False)
        validation_target.to_csv(validation_target_file, index=False)

        print(f"Step 5: Extracted and saved target columns.")
        print(f"Training target data saved to: {training_target_file}")
        print(f"Validation target data saved to: {validation_target_file}")

        # Save debug information for the target column
        debug_info = self.get_debug_info()
        debug_info_file = f"{target_prefix}debug_info.json"
        with open(debug_info_file, 'w') as f:
            json.dump(debug_info, f)

        print(f"Step 6: Saved debug information.")
        print(f"Debug information saved to: {debug_info_file}")

        return data  # Return the entire data as per the original return type requirement

# Example usage
if __name__ == "__main__":
    plugin = Plugin()
    data = pd.read_csv('tests/data/EURUSD_5m_2010_2015.csv', header=None)
    processed_data = plugin.process(data)
    print(processed_data)
