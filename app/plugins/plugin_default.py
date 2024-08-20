import pandas as pd
import json
import numpy as np

class Plugin:
    """
    Plugin to preprocess the dataset for feature extraction.
    """
    # Define the parameters for this plugin and their default values
    plugin_params = {
        'input_column_order': ["d", "o", "h", "l", "c", "v","co"],
        'output_column_order': ["d", "o", "l", "h", "c", "v","co"],
        'dataset_prefix': "x_",
        'target_prefix': "y_",
        'target_column': 6,
        'pip_value': 0.00001,
        'range': (-1, 1),
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
            debug_info['min_val'] = self.normalization_params['min']
            debug_info['max_val'] = self.normalization_params['max']
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

    def normalize(self, df, min_vals, max_vals, range_vals):
        """
        Normalize the DataFrame using min-max normalization with a specified range.

        Args:
            df (pd.DataFrame): The DataFrame to be normalized.
            min_vals (pd.Series): The minimum values for each column.
            max_vals (pd.Series): The maximum values for each column.
            range_vals (tuple): The range (min, max) for normalization.

        Returns:
            pd.DataFrame: The normalized DataFrame.
        """
        norm_min, norm_max = range_vals
        normalized_df = (df - min_vals) / (max_vals - min_vals) * (norm_max - norm_min) + norm_min
        return normalized_df

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
        print(f"[DEBUG] Loaded data shape: {data.shape}")
        print(f"[DEBUG] First few rows of loaded data:\n{data.head()}")

        # Step 1: Reorder columns based on input and output orders
        input_column_order = self.params['input_column_order']
        output_column_order = self.params['output_column_order']

        data.columns = input_column_order  # Set columns to input order first
        data = data[output_column_order]
        print(f"[DEBUG] Step 1: Columns reordered based on output_column_order. New order: {list(data.columns)}")
        print(f"[DEBUG] First few rows after reordering columns:\n{data.head()}")

        # Step 2: Ensure only numeric columns are converted to numeric
        non_numeric_columns = ['d']
        numeric_columns = data.columns.difference(non_numeric_columns)
        data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
        print("[DEBUG] Step 2: Converted columns to numeric where applicable.")
        print(f"[DEBUG] First few rows after conversion to numeric:\n{data.head()}")

        # Drop any rows with NaN values that resulted from conversion (optional, based on your data handling policy)
        data = data.dropna()
        print(f"[DEBUG] Step 3: Dropped rows with NaN values (if any).")
        print(f"[DEBUG] Data shape after dropping NaN rows: {data.shape}")
        print(f"[DEBUG] First few rows after dropping NaN rows:\n{data.head()}")

        # Step 4: Split into three datasets (D1 for training, D2 for validation, D3 for testing)
        total_len = len(data)
        d1_size = int(total_len * self.params['d1_proportion'])
        d2_size = int(total_len * self.params['d2_proportion'])

        d1_data = data.iloc[:d1_size].copy()
        d2_data = data.iloc[d1_size:d1_size + d2_size].copy()
        d3_data = data.iloc[d1_size + d2_size:].copy()

        print(f"[DEBUG] Step 4: Split data into D1, D2, and D3 datasets.")
        print(f"[DEBUG] D1 data shape: {d1_data.shape}")
        print(f"[DEBUG] D2 data shape: {d2_data.shape}")
        print(f"[DEBUG] D3 data shape: {d3_data.shape}")

        # Save D1, D2, and D3 datasets (prior to normalization)
        dataset_prefix = self.params['dataset_prefix']
        d1_data_file = f"{dataset_prefix}d1_original.csv"
        d2_data_file = f"{dataset_prefix}d2_original.csv"
        d3_data_file = f"{dataset_prefix}d3_original.csv"
        d1_data.to_csv(d1_data_file, header=False, index=False)
        d2_data.to_csv(d2_data_file, header=False, index=False)
        d3_data.to_csv(d3_data_file, header=False, index=False)
        print(f"[DEBUG] D1 data saved to: {d1_data_file}")
        print(f"[DEBUG] D2 data saved to: {d2_data_file}")
        print(f"[DEBUG] D3 data saved to: {d3_data_file}")

        # Step 5: Calculate min and max values from D1
        target_column_name = output_column_order[self.params['target_column']]
        min_vals = d1_data[target_column_name].min()
        max_vals = d1_data[target_column_name].max()
        self.normalization_params = {'min': min_vals, 'max': max_vals, 'range': self.params['range']}
        print(f"[DEBUG] Step 6: Calculated min and max values from D1's target column.")
        print(f"[DEBUG] First few rows from D1 target before normalization:\n{d1_data[target_column_name].head()}")
        print(f"[DEBUG] First few rows from D2 target before normalization:\n{d2_data[target_column_name].head()}")
        print(f"[DEBUG] First few rows from D3 target before normalization:\n{d3_data[target_column_name].head()}")

        # Step 6: Normalize the target column of D1, D2, and D3 using D1's min and max values
        d1_data[target_column_name] = self.normalize(d1_data[target_column_name], min_vals, max_vals, self.params['range'])
        d2_data[target_column_name] = self.normalize(d2_data[target_column_name], min_vals, max_vals, self.params['range'])
        d3_data[target_column_name] = self.normalize(d3_data[target_column_name], min_vals, max_vals, self.params['range'])
        print(f"[DEBUG] Step 7: Normalized D1, D2, and D3 datasets using D1's normalization parameters.")

        # Save the normalized target columns
        target_prefix = self.params['target_prefix']
        d1_target_file = f"{target_prefix}d1_target.csv"
        d2_target_file = f"{target_prefix}d2_target.csv"
        d3_target_file = f"{target_prefix}d3_target.csv"

        d1_data[[target_column_name]].to_csv(d1_target_file, index=False, header=False)
        d2_data[[target_column_name]].to_csv(d2_target_file, index=False, header=False)
        d3_data[[target_column_name]].to_csv(d3_target_file, index=False, header=False)

        print(f"[DEBUG] D1 target data saved to: {d1_target_file}")
        print(f"[DEBUG] D2 target data saved to: {d2_target_file}")
        print(f"[DEBUG] D3 target data saved to: {d3_target_file}")

        # Step 7: Save debug information for the target column
        debug_info = self.get_debug_info()
        debug_info_file = f"{target_prefix}debug_info.json"
        with open(debug_info_file, 'w') as f:
            json.dump(debug_info, f)

        print(f"[DEBUG] Step 8: Saved debug information.")
        print(f"[DEBUG] Debug information saved to: {debug_info_file}")

        # Create a summary DataFrame with the dataset details
        summary_data = {
            'Filename': [d1_data_file, d2_data_file, d3_data_file, d1_target_file, d2_target_file, d3_target_file],
            'Rows': [d1_data.shape[0], d2_data.shape[0], d3_data.shape[0], d1_data.shape[0], d2_data.shape[0], d3_data.shape[0]],
            'Columns': [d1_data.shape[1], d2_data.shape[1], d3_data.shape[1], 1, 1, 1]
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
