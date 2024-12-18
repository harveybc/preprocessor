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
        'only_low_CV': True  # Parameter to control processing of low CV columns
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
        print("[DEBUG] Starting process...")
        print(f"[DEBUG] Loaded data shape: {data.shape}")
        print(f"[DEBUG] First few rows of loaded data:\n{data.head()}")

        # Step 1: Reorder columns based on output_column_order
        print("[DEBUG] Step 1: Reordering columns...")
        column_mapping = {'d': 'DATE_TIME', 'o': 'OPEN', 'l': 'LOW', 'h': 'HIGH', 'c': 'CLOSE'}
        output_column_order = [column_mapping[char] for char in self.params['output_column_order']]
        print(f"[DEBUG] Expected output column order: {output_column_order}")

        # Check if all required columns exist
        missing_columns = set(output_column_order) - set(data.columns)
        if missing_columns:
            raise ValueError(f"[ERROR] Missing columns in input data: {missing_columns}")

        reordered_data = data[output_column_order].copy()
        print(f"[DEBUG] Actual columns after reordering: {list(reordered_data.columns)}")
        print(f"[DEBUG] First few rows of reordered data:\n{reordered_data.head()}")

        # Step 2: Split into three datasets (D1, D2, D3)
        print("[DEBUG] Step 2: Splitting data into D1, D2, and D3...")
        total_len = len(reordered_data)
        d1_size = int(total_len * self.params['d1_proportion'])
        d2_size = int(total_len * self.params['d2_proportion'])
        print(f"[DEBUG] Total rows: {total_len}, D1 size: {d1_size}, D2 size: {d2_size}, D3 size: {total_len - d1_size - d2_size}")

        d1_data = reordered_data.iloc[:d1_size].copy()
        d2_data = reordered_data.iloc[d1_size:d1_size + d2_size].copy()
        d3_data = reordered_data.iloc[d1_size + d2_size:].copy()

        print(f"[DEBUG] D1 shape: {d1_data.shape}")
        print(f"[DEBUG] D2 shape: {d2_data.shape}")
        print(f"[DEBUG] D3 shape: {d3_data.shape}")

        # Step 3: Save dataset prefix files (NOT normalized)
        print("[DEBUG] Step 3: Saving dataset prefix files (base_)...")
        dataset_prefix = self.params['dataset_prefix']
        d1_data.to_csv(f"{dataset_prefix}d1.csv", header=False, index=False)
        d2_data.to_csv(f"{dataset_prefix}d2.csv", header=False, index=False)
        d3_data.to_csv(f"{dataset_prefix}d3.csv", header=False, index=False)
        print(f"[DEBUG] Saved base_d1.csv, base_d2.csv, base_d3.csv")

        # Step 4: Normalize all numeric columns using Min-Max Normalization
        print("[DEBUG] Step 4: Normalizing all numeric columns using Min-Max Normalization...")
        numeric_columns = data.columns.difference(['DATE_TIME'])  # Exclude non-numeric DATE_TIME
        print(f"[DEBUG] Numeric columns to normalize: {list(numeric_columns)}")

        epsilon = 1e-8
        normalization_params = {}  # Store min-max params for each column

        # Fit normalization on D1
        for column in numeric_columns:
            min_val = d1_data[column].min()
            max_val = d1_data[column].max()
            normalization_params[column] = (min_val, max_val)

            d1_data[column] = (d1_data[column] - min_val) / (max_val - min_val + epsilon)
            d2_data[column] = (d2_data[column] - min_val) / (max_val - min_val + epsilon)
            d3_data[column] = (d3_data[column] - min_val) / (max_val - min_val + epsilon)

            print(f"[DEBUG] Column '{column}': min={min_val}, max={max_val}")

        # Step 5: Save target prefix files (normalized)
        print("[DEBUG] Step 5: Saving target prefix files (normalized_)...")
        target_prefix = self.params['target_prefix']
        d1_data.to_csv(f"{target_prefix}d1_target.csv", header=False, index=False)
        d2_data.to_csv(f"{target_prefix}d2_target.csv", header=False, index=False)
        d3_data.to_csv(f"{target_prefix}d3_target.csv", header=False, index=False)
        print(f"[DEBUG] Saved normalized_d1_target.csv, normalized_d2_target.csv, normalized_d3_target.csv")

        # Step 6: Summary
        print("[DEBUG] Step 6: Creating summary DataFrame...")
        summary_data = {
            'Filename': [
                f"{dataset_prefix}d1.csv", f"{dataset_prefix}d2.csv", f"{dataset_prefix}d3.csv",
                f"{target_prefix}d1_target.csv", f"{target_prefix}d2_target.csv", f"{target_prefix}d3_target.csv"
            ],
            'Rows': [
                d1_data.shape[0], d2_data.shape[0], d3_data.shape[0],
                d1_data.shape[0], d2_data.shape[0], d3_data.shape[0]
            ],
            'Columns': [
                len(output_column_order), len(output_column_order), len(output_column_order),
                len(numeric_columns), len(numeric_columns), len(numeric_columns)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        print("[DEBUG] Processing complete. Summary of saved files:")
        print(summary_df)
        return summary_df






# Example usage
if __name__ == "__main__":
    plugin = Plugin()
    data = pd.read_csv('tests/data/EURUSD_5m_2010_2015.csv', header=None)
    print(f"Loaded data shape: {data.shape}")
    processed_data = plugin.process(data)
    print(processed_data)
