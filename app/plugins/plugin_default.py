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
        'target_column': 4,
        'pip_value': 0.00001,
        'range': (0, 1),
        'd1_proportion': 0.3,
        'd2_proportion': 0.3
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
        pip_value = self.params['pip_value']
        original_range = original_max - original_min
        normalized_range_span = normalized_range[1] - normalized_range[0]
        conversion_factor = normalized_range_span / original_range
        pip_value_in_normalized_range = pips * pip_value * conversion_factor
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
        range_vals = self.params['range']
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
        Process the data by reordering columns, splitting into three datasets (D1, D2, D3),
        normalizing D2 and D3 based on D1, and saving the target columns.

        Args:
            data (pd.DataFrame): The input data to be processed.

        Returns:
            pd.DataFrame: The summary of processed datasets.
        """
        # Print the dimensions of the loaded data
        print(f"Loaded data shape: {data.shape}")

        # Step 1: Reorder columns based on input and output orders
        input_column_order = self.params['input_column_order']
        output_column_order = self.params['output_column_order']

        data.columns = input_column_order  # Set columns to input order first
        data = data[output_column_order]
        print(f"Step 1: Columns reordered based on output_column_order. New order: {list(data.columns)}")

        # Step 2: Split into three datasets (D1 for training, D2 for validation, D3 for testing)
        total_len = len(data)
        d1_size = int(total_len * self.params['d1_proportion'])
        d2_size = int(total_len * self.params['d2_proportion'])

        d1_data = data.iloc[:d1_size].copy()
        d2_data = data.iloc[d1_size:d1_size + d2_size].copy()
        d3_data = data.iloc[d1_size + d2_size:].copy()

        print(f"Step 2: Split data into D1, D2, and D3 datasets.")
        print(f"D1 data shape: {d1_data.shape}")
        print(f"D2 data shape: {d2_data.shape}")
        print(f"D3 data shape: {d3_data.shape}")

        # Step 3: Save D1 dataset (prior to normalization)
        dataset_prefix = self.params['dataset_prefix']
        d1_data_file = f"{dataset_prefix}d1_training.csv"
        d1_data.to_csv(d1_data_file, header=False, index=False)
        print(f"D1 data saved to: {d1_data_file}")

        # Step 4: Calculate min and max values from D1
        min_vals = d1_data.min()
        max_vals = d1_data.max()
        self.normalization_params = {'min': min_vals, 'max': max_vals}
        print(f"Step 4: Calculated min and max values from D1.")

        # Step 5: Normalize D2 and D3 using D1's min and max values
        def normalize(df, min_vals, max_vals):
            return (df - min_vals) / (max_vals - min_vals)
        
        d2_data = normalize(d2_data, min_vals, max_vals)
        d3_data = normalize(d3_data, min_vals, max_vals)
        print(f"Step 5: Normalized D2 and D3 datasets using D1's normalization parameters.")

        # Save the D2 and D3 datasets
        d2_data_file = f"{dataset_prefix}d2_validation.csv"
        d3_data_file = f"{dataset_prefix}d3_testing.csv"

        d2_data.to_csv(d2_data_file, header=False, index=False)
        d3_data.to_csv(d3_data_file, header=False, index=False)

        print(f"D2 data saved to: {d2_data_file}")
        print(f"D3 data saved to: {d3_data_file}")

        # Step 6: Extract and save the target columns for D1, D2, and D3 datasets
        target_column_index = self.params['target_column']
        target_column_name = output_column_order[target_column_index]
        target_prefix = self.params['target_prefix']

        d1_target = d1_data[[target_column_name]]
        d2_target = d2_data[[target_column_name]]
        d3_target = d3_data[[target_column_name]]

        d1_target_file = f"{target_prefix}d1_target.csv"
        d2_target_file = f"{target_prefix}d2_target.csv"
        d3_target_file = f"{target_prefix}d3_target.csv"

        d1_target.to_csv(d1_target_file, index=False, header=False)
        d2_target.to_csv(d2_target_file, index=False, header=False)
        d3_target.to_csv(d3_target_file, index=False, header=False)

        print(f"Step 6: Extracted and saved target columns for D1, D2, and D3.")
        print(f"D1 target data saved to: {d1_target_file}")
        print(f"D2 target data saved to: {d2_target_file}")
        print(f"D3 target data saved to: {d3_target_file}")

        # Step 7: Save debug information for the target column
        debug_info = self.get_debug_info()
        debug_info_file = f"{target_prefix}debug_info.json"
        with open(debug_info_file, 'w') as f:
            json.dump(debug_info, f)

        print(f"Step 7: Saved debug information.")
        print(f"Debug information saved to: {debug_info_file}")

        # Create a summary DataFrame with the dataset details
        summary_data = {
            'Filename': [d1_data_file, d2_data_file, d3_data_file, d1_target_file, d2_target_file, d3_target_file],
            'Rows': [d1_data.shape[0], d2_data.shape[0], d3_data.shape[0], d1_target.shape[0], d2_target.shape[0], d3_target.shape[0]],
            'Columns': [d1_data.shape[1], d2_data.shape[1], d3_data.shape[1], d1_target.shape[1], d2_target.shape[1], d3_target.shape[1]]
        }
        summary_df = pd.DataFrame(summary_data)

        return summary_df

# Example usage
if __name__ == "__main__":
    plugin = Plugin()
    data = pd.read_csv('tests/data/EURUSD_5m_2010_2015.csv', header=None)
    print(f"Loaded data shape: {data.shape}")
    processed_data = plugin.process(data)
    print(processed_data)
