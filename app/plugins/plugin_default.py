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
        print(f"[DEBUG] Loaded data shape: {data.shape}")
        print(f"[DEBUG] Columns in the data: {list(data.columns)}")

        # Step 1: Ensure DATE_TIME column is included as a regular column
        if isinstance(data.index, pd.DatetimeIndex):
            print("[DEBUG] DATE_TIME is currently the index. Resetting it to a regular column...")
            data.reset_index(inplace=True)
        if 'DATE_TIME' not in data.columns:
            raise ValueError("[ERROR] DATE_TIME column is missing in the input data!")

        # Step 2: Reorder columns based on output order
        output_column_order = ['DATE_TIME', 'OPEN', 'LOW', 'HIGH', 'CLOSE']
        print(f"[DEBUG] Expected output column order: {output_column_order}")
        print(f"[DEBUG] Actual columns before reordering: {list(data.columns)}")

        # Check for missing columns
        missing_columns = set(output_column_order) - set(data.columns)
        if missing_columns:
            raise ValueError(f"[ERROR] Missing columns in input data: {missing_columns}")

        base_data = data[output_column_order]
        print(f"[DEBUG] Reordered data columns: {list(base_data.columns)}")

        # Step 3: Split data into D1, D2, and D3
        total_len = len(base_data)
        d1_size = int(total_len * self.params['d1_proportion'])
        d2_size = int(total_len * self.params['d2_proportion'])

        d1_data = base_data.iloc[:d1_size].copy()
        d2_data = base_data.iloc[d1_size:d1_size + d2_size].copy()
        d3_data = base_data.iloc[d1_size + d2_size:].copy()

        print(f"[DEBUG] Total rows: {total_len}, D1 size: {d1_size}, D2 size: {d2_size}, D3 size: {total_len - d1_size - d2_size}")
        print(f"[DEBUG] D1 shape: {d1_data.shape}, D2 shape: {d2_data.shape}, D3 shape: {d3_data.shape}")

        # Step 4: Save the base datasets (with headers)
        dataset_prefix = self.params['dataset_prefix']
        d1_data.to_csv(f"{dataset_prefix}d1.csv", index=False, header=True)
        d2_data.to_csv(f"{dataset_prefix}d2.csv", index=False, header=True)
        d3_data.to_csv(f"{dataset_prefix}d3.csv", index=False, header=True)
        print(f"[DEBUG] Saved base_d1.csv, base_d2.csv, base_d3.csv with headers and DATE_TIME")

        # Step 5: Normalize all numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"[DEBUG] Numeric columns identified for normalization: {numeric_columns}")

        # Full target datasets: Normalize all numeric columns
        d1_full = data.iloc[:d1_size].copy()
        d2_full = data.iloc[d1_size:d1_size + d2_size].copy()
        d3_full = data.iloc[d1_size + d2_size:].copy()

        # Initialize normalized DataFrames with all columns (including DATE_TIME)
        normalized_d1 = d1_full.copy()
        normalized_d2 = d2_full.copy()
        normalized_d3 = d3_full.copy()

        # Normalize only numeric columns
        for column in numeric_columns:
            min_val = d1_full[column].min()
            max_val = d1_full[column].max()
            print(f"[DEBUG] Normalizing column '{column}': min={min_val}, max={max_val}")

            normalized_d1[column] = (d1_full[column] - min_val) / (max_val - min_val + 1e-8)
            normalized_d2[column] = (d2_full[column] - min_val) / (max_val - min_val + 1e-8)
            normalized_d3[column] = (d3_full[column] - min_val) / (max_val - min_val + 1e-8)

        # Step 6: Save the normalized datasets (with headers and DATE_TIME)
        target_prefix = self.params['target_prefix']
        normalized_d1.to_csv(f"{target_prefix}d1.csv", index=False, header=True)
        normalized_d2.to_csv(f"{target_prefix}d2.csv", index=False, header=True)
        normalized_d3.to_csv(f"{target_prefix}d3.csv", index=False, header=True)
        print(f"[DEBUG] Saved normalized_d1.csv, normalized_d2.csv, normalized_d3.csv with headers and DATE_TIME")

        # Step 7: Return summary
        summary_data = {
            'Filename': [f"{dataset_prefix}d1.csv", f"{dataset_prefix}d2.csv", f"{dataset_prefix}d3.csv",
                        f"{target_prefix}d1.csv", f"{target_prefix}d2.csv", f"{target_prefix}d3.csv"],
            'Rows': [d1_data.shape[0], d2_data.shape[0], d3_data.shape[0],
                    normalized_d1.shape[0], normalized_d2.shape[0], normalized_d3.shape[0]],
            'Columns': [d1_data.shape[1], d2_data.shape[1], d3_data.shape[1],
                        normalized_d1.shape[1], normalized_d2.shape[1], normalized_d3.shape[1]]
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
