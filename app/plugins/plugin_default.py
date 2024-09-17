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
        'dataset_prefix': "x_",
        'target_prefix': "y_",
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

        # Step 4: Calculate CV (Coefficient of Variation) and decide normalization method
        cvs = {}
        high_cv_columns = []
        low_cv_columns = []
        for column in numeric_columns:
            col_data = d1_data[column]
            mean = col_data.mean()
            std_dev = col_data.std()
            cv = std_dev / abs(mean) if abs(mean) > 1e-8 else 0
            cvs[column] = cv

            # Print CV for each column
            print(f"[DEBUG] {column} has CV {cv:.5f}")

            # Classify as high or low volatility based on CV
            if cv > 1:  # Adjust threshold as per your specific needs
                high_cv_columns.append((column, cv))
            else:
                low_cv_columns.append((column, cv))

        # Print high and low CV columns
        print(f"[DEBUG] High CV columns: {high_cv_columns}")
        print(f"[DEBUG] Low CV columns: {low_cv_columns}")

        # Step 5: Normalize the selected columns in D1, D2, and D3
        epsilon = 1e-8
        for column in numeric_columns:
            col_data = d1_data[column]
            # Ensure min-max normalization for 'Close' column in the target files
            if column == 'Close':
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

        # Step 6: Save D1, D2, and D3 datasets in the correct output order without headers (Original values for open, low, high, and close)
        # Ensure that the columns in D1, D2, and D3 are ordered according to 'output_column_order' (["d", "o", "l", "h", "c"])
        output_order = ['Date', 'Open', 'Low', 'High', 'Close']  # Mapping of 'o', 'l', 'h', 'c' to column names

        # Use the unnormalized data for the original columns (Open, Low, High, Close)
        d1_data_reordered = reordered_data[output_order].iloc[:d1_size]
        d2_data_reordered = reordered_data[output_order].iloc[d1_size:d1_size + d2_size]
        d3_data_reordered = reordered_data[output_order].iloc[d1_size + d2_size:]

        # Save the reordered datasets without headers
        dataset_prefix = self.params['dataset_prefix']
        d1_data_file = f"{dataset_prefix}d1.csv"
        d2_data_file = f"{dataset_prefix}d2.csv"
        d3_data_file = f"{dataset_prefix}d3.csv"

        # Save without headers
        d1_data_reordered.to_csv(d1_data_file, header=False, index=False)
        d2_data_reordered.to_csv(d2_data_file, header=False, index=False)
        d3_data_reordered.to_csv(d3_data_file, header=False, index=False)

        print(f"[DEBUG] D1 data saved to: {d1_data_file}")
        print(f"[DEBUG] D2 data saved to: {d2_data_file}")
        print(f"[DEBUG] D3 data saved to: {d3_data_file}")

        # Step 7: Ensure columns_to_process is properly set before creating the target file
        numeric_columns = reordered_data.columns.difference(non_numeric_columns)

        if self.params['only_low_CV']:
            columns_to_process = [col for col, cv in cvs.items() if cv <= 0.3]
        else:
            columns_to_process = list(numeric_columns)

        # Step 8: Exclude 'Low', 'High', and 'Open' columns for the target file and reorder to have 'Close' as the first column
        columns_to_exclude = ['Open', 'Low', 'High']  # Exclude Open, Low, High
        columns_to_include_in_target = [col for col in columns_to_process if col not in columns_to_exclude and col != 'd']

        # Ensure 'Close' is the first column
        if 'Close' in columns_to_include_in_target:
            columns_to_include_in_target.remove('Close')
        columns_to_include_in_target = ['Close'] + columns_to_include_in_target

        print(f"[DEBUG] Columns included in target file: {columns_to_include_in_target}")

        # Create target datasets with 'Close' as the first column
        d1_target = d1_data[columns_to_include_in_target]
        d2_target = d2_data[columns_to_include_in_target]
        d3_target = d3_data[columns_to_include_in_target]

        # Save the target datasets
        target_prefix = self.params['target_prefix']
        d1_target_file = f"{target_prefix}d1_target.csv"
        d2_target_file = f"{target_prefix}d2_target.csv"
        d3_target_file = f"{target_prefix}d3_target.csv"

        d1_target.to_csv(d1_target_file, index=False, header=False)
        d2_target.to_csv(d2_target_file, index=False, header=False)
        d3_target.to_csv(d3_target_file, index=False, header=False)

        print(f"[DEBUG] D1 target data saved to: {d1_target_file}")
        print(f"[DEBUG] D2 target data saved to: {d2_target_file}")
        print(f"[DEBUG] D3 target data saved to: {d3_target_file}")

        # Step 9: Plot distribution of each column in D1 target dataset
        num_columns = len(d1_target.columns)
        num_rows = (num_columns + 3) // 4  # Create a grid of 4 plots per row
        fig, axes = plt.subplots(num_rows, 4, figsize=(20, num_rows * 5))
        axes = axes.flatten()

        for i, column in enumerate(d1_target.columns):
            sns.histplot(d1_target[column], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {column}')

        plt.tight_layout(h_pad=10, pad=3)  # Adjust layout to prevent overlap
        plt.savefig('d1_target_distributions.png')
        plt.show()

        print(f"[DEBUG] Distribution plots saved for D1 target.")

        # Step 10: Save debug information
        debug_info = self.get_debug_info()
        debug_info_file = f"{target_prefix}debug_info.json"
        with open(debug_info_file, 'w') as f:
            json.dump(debug_info, f, indent=4)

        print(f"[DEBUG] Debug information saved to: {debug_info_file}")

        # Step 11: Create a summary DataFrame with the dataset details
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
