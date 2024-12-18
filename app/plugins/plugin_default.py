import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import json

class Plugin:
    """
    Plugin to preprocess the dataset for feature extraction.
    """
    # Define the parameters for this plugin and their default values
    plugin_params = {
        'input_column_order': ["d", "o", "h", "l", "c"],
        'output_column_order': ["d", "o", "l", "h", "c"],
        'dataset_prefix': "base_",
        'target_prefix': "normalized_",
        'target_column': 4,  # Index in output_column_order (zero-based)
        'pip_value': 0.00001,
        'range': (-1, 1),
        'd1_proportion': 0.3,
        'd2_proportion': 0.3,
        'only_low_CV': False  # Parameter to control processing of low CV columns
    }

    # Define the debug variables for this plugin
    plugin_debug_vars = ['column_metrics', 'normalization_params']

    def __init__(self):
        """
        Initialize the Plugin with default parameters.
        """
        self.params = self.plugin_params.copy()
        self.normalization_params = {}  # To store normalization parameters for each column
        self.column_metrics = {}  # To store metrics for each column

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
            dict: Debug information including column metrics and normalization parameters.
        """
        debug_info = {
            'column_metrics': self.column_metrics,
            'normalization_params': self.normalization_params
        }
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
        Process the data by reordering columns, splitting into three datasets (D1, D2, D3),
        normalizing columns based on D1, and saving the datasets.

        Args:
            data (pd.DataFrame): The input data to be processed.

        Returns:
            pd.DataFrame: The summary of processed datasets.
        """
        print(f"[DEBUG] Loaded data shape: {data.shape}")
        print(f"[DEBUG] First few rows of loaded data:\n{data.head()}")

        # Step 1: Reorder columns based on input and output orders without renaming them
        input_column_order = self.params['input_column_order']
        output_column_order = self.params['output_column_order']

        column_indices = list(range(len(input_column_order)))
        reordered_indices = [input_column_order.index(col) for col in output_column_order]
        reordered_data = data.iloc[:, column_indices].copy()
        reordered_data = reordered_data.iloc[:, reordered_indices]

        if data.shape[1] > len(input_column_order):
            additional_columns = data.iloc[:, len(input_column_order):].copy()
            reordered_data = pd.concat([reordered_data, additional_columns], axis=1)

        print(f"[DEBUG] Reordered columns: {list(reordered_data.columns)}")

        # Step 2: Ensure only numeric columns are converted to numeric
        non_numeric_columns = ['d']
        numeric_columns = reordered_data.columns.difference(non_numeric_columns)
        reordered_data[numeric_columns] = reordered_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
        reordered_data = reordered_data.dropna()

        print(f"[DEBUG] Data shape after dropping NaN rows: {reordered_data.shape}")
        print(f"[DEBUG] First few rows after dropping NaN rows:\n{reordered_data.head()}")

        # Step 3: Split into three datasets (D1 for training, D2 for validation, D3 for testing)
        total_len = len(reordered_data)
        d1_size = int(total_len * self.params['d1_proportion'])
        d2_size = int(total_len * self.params['d2_proportion'])

        d1_data = reordered_data.iloc[:d1_size].copy()
        d2_data = reordered_data.iloc[d1_size:d1_size + d2_size].copy()
        d3_data = reordered_data.iloc[d1_size + d2_size:].copy()

        print(f"[DEBUG] D1 data shape: {d1_data.shape}")
        print(f"[DEBUG] D2 data shape: {d2_data.shape}")
        print(f"[DEBUG] D3 data shape: {d3_data.shape}")

        # Step 4: Validate required columns for output_order
        output_order = ['DATE_TIME', 'OPEN', 'LOW', 'HIGH', 'CLOSE']
        if set(output_order) - set(reordered_data.columns):
            print(f"[DEBUG] Adjusting column names for compatibility with output_order.")
            column_mapping = dict(zip(self.params['output_column_order'], output_order))
            reordered_data.rename(columns=column_mapping, inplace=True)

        # Step 5: Normalize the selected columns in D1, D2, and D3
        epsilon = 1e-8
        for column in numeric_columns:
            col_data = d1_data[column]
            if column == 'CLOSE':
                method = 'min-max'
            else:
                method = 'z-score' if abs(skew(col_data)) <= 0.5 and -1.0 <= kurtosis(col_data) <= 6.0 else 'min-max'

            if method == 'z-score':
                mean = col_data.mean()
                std_dev = col_data.std()
                d1_data[column] = (d1_data[column] - mean) / (std_dev + epsilon)
                d2_data[column] = (d2_data[column] - mean) / (std_dev + epsilon)
                d3_data[column] = (d3_data[column] - mean) / (std_dev + epsilon)
            else:  # min-max normalization
                min_val = col_data.min()
                max_val = col_data.max()
                d1_data[column] = (d1_data[column] - min_val) / (max_val - min_val + epsilon)
                d2_data[column] = (d2_data[column] - min_val) / (max_val - min_val + epsilon)
                d3_data[column] = (d3_data[column] - min_val) / (max_val - min_val + epsilon)

            print(f"[DEBUG] Normalized column '{column}' using {method} method.")

        # Step 6: Save reordered and target datasets
        dataset_prefix = self.params['dataset_prefix']
        d1_data_file = f"{dataset_prefix}d1.csv"
        d2_data_file = f"{dataset_prefix}d2.csv"
        d3_data_file = f"{dataset_prefix}d3.csv"

        d1_data.to_csv(d1_data_file, header=False, index=False)
        d2_data.to_csv(d2_data_file, header=False, index=False)
        d3_data.to_csv(d3_data_file, header=False, index=False)

        print(f"[DEBUG] D1 data saved to: {d1_data_file}")
        print(f"[DEBUG] D2 data saved to: {d2_data_file}")
        print(f"[DEBUG] D3 data saved to: {d3_data_file}")

        # Create summary DataFrame
        summary_data = {
            'Filename': [d1_data_file, d2_data_file, d3_data_file],
            'Rows': [d1_data.shape[0], d2_data.shape[0], d3_data.shape[0]],
            'Columns': [d1_data.shape[1], d2_data.shape[1], d3_data.shape[1]]
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
