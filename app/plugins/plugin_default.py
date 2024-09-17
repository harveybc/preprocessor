import pandas as pd
import json
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

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
        # Print the dimensions of the loaded data
        print(f"[DEBUG] Loaded data shape: {data.shape}")
        print(f"[DEBUG] First few rows of loaded data:\n{data.head()}")

        # Step 1: Reorder columns based on input and output orders without renaming them
        input_column_order = self.params['input_column_order']
        output_column_order = self.params['output_column_order']

        # We assume that the first len(input_column_order) columns correspond to 'input_column_order'
        column_indices = list(range(len(input_column_order)))  # The positions of the first columns

        # Create a mapping of the input positions to output positions
        reordered_indices = [input_column_order.index(col) for col in output_column_order]
        
        # Reorder the columns accordingly (only the first len(input_column_order) columns)
        reordered_data = data.iloc[:, column_indices].copy()  # Select the first N columns based on input order
        reordered_data = reordered_data.iloc[:, reordered_indices]  # Reorder based on output order
        
        # Preserve any additional columns (keep their order intact)
        if data.shape[1] > len(input_column_order):
            additional_columns = data.iloc[:, len(input_column_order):].copy()
            reordered_data = pd.concat([reordered_data, additional_columns], axis=1)
        
        print(f"[DEBUG] Reordered columns. New order: {list(reordered_data.columns)}")
        
        # Step 2: Ensure only numeric columns are converted to numeric
        non_numeric_columns = ['d']  # Assuming 'd' refers to date or non-numeric column (position, not name)
        numeric_columns = reordered_data.columns.difference(non_numeric_columns)
        reordered_data[numeric_columns] = reordered_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
        print("[DEBUG] Step 2: Converted columns to numeric where applicable.")
        print(f"[DEBUG] First few rows after conversion to numeric:\n{reordered_data.head()}")

        # Drop any rows with NaN values that resulted from conversion
        reordered_data = reordered_data.dropna()
        print(f"[DEBUG] Step 3: Dropped rows with NaN values (if any).")
        print(f"[DEBUG] Data shape after dropping NaN rows: {reordered_data.shape}")
        print(f"[DEBUG] First few rows after dropping NaN rows:\n{reordered_data.head()}")

        # Step 4: Split into three datasets (D1 for training, D2 for validation, D3 for testing)
        total_len = len(reordered_data)
        d1_size = int(total_len * self.params['d1_proportion'])
        d2_size = int(total_len * self.params['d2_proportion'])

        d1_data = reordered_data.iloc[:d1_size].copy()
        d2_data = reordered_data.iloc[d1_size:d1_size + d2_size].copy()
        d3_data = reordered_data.iloc[d1_size + d2_size:].copy()

        print(f"[DEBUG] Step 4: Split data into D1, D2, and D3 datasets.")
        print(f"[DEBUG] D1 data shape: {d1_data.shape}")
        print(f"[DEBUG] D2 data shape: {d2_data.shape}")
        print(f"[DEBUG] D3 data shape: {d3_data.shape}")

        # Step 5: Analyze columns in D1 to decide on normalization method and calculate metrics
        # Adjust target_column index if necessary
        target_column_index = self.params['target_column']
        output_column_order = self.params['output_column_order']
        target_column_name = output_column_order[target_column_index]

        # All numeric columns including additional ones
        numeric_columns = d1_data.columns.difference(non_numeric_columns)

        # Calculate CV, skewness, kurtosis for each column (ensure exact CV classification as in feature-engineering)
        cvs = {}
        skewness_dict = {}
        kurtosis_dict = {}
        epsilon = 1e-8  # Small value to prevent division by zero

        for column in numeric_columns:
            col_data = d1_data[column]

            # Calculate mean and std dev
            mean = col_data.mean()
            std_dev = col_data.std()
            adjusted_mean = mean if abs(mean) > epsilon else epsilon
            cv = std_dev / abs(adjusted_mean)
            cvs[column] = cv

            # Calculate skewness and kurtosis
            skewness = skew(col_data)
            kurt = kurtosis(col_data)
            skewness_dict[column] = skewness
            kurtosis_dict[column] = kurt

            # Decide on normalization method
            if abs(skewness) <= 0.5 and -1.0 <= kurt <= 6.0:
                method = 'z-score'
                norm_params = {
                    'mean': mean,
                    'std': std_dev
                }
            else:
                method = 'min-max'
                min_val = col_data.min()
                max_val = col_data.max()
                norm_params = {
                    'min': min_val,
                    'max': max_val
                }

            # Save metrics and normalization parameters
            self.column_metrics[column] = {
                'CV': cv,
                'skewness': skewness,
                'kurtosis': kurt,
                'normality_score': abs(skewness) + abs(kurt),
                'method': method
            }
            self.normalization_params[column] = norm_params

        # ** Use the exact threshold for low and high CV from feature-eng **
        # Assuming feature-eng uses the median CV as the threshold for classifying low/high variability
        median_cv = np.median(list(cvs.values()))
        print(f"[DEBUG] Median CV across columns: {median_cv}")

        # Identify columns to process based on only_low_CV parameter
        if self.params['only_low_CV']:
            columns_to_process = [col for col, cv in cvs.items() if cv <= median_cv]
            print(f"[DEBUG] Processing only low CV columns: {columns_to_process}")
        else:
            columns_to_process = list(numeric_columns)
            print(f"[DEBUG] Processing all columns: {columns_to_process}")

        # Step 6: Normalize the selected columns in D1, D2, and D3 using parameters from D1
        for column in columns_to_process:
            method = self.column_metrics[column]['method']
            norm_params = self.normalization_params[column]

            if method == 'z-score':
                mean = norm_params['mean']
                std_dev = norm_params['std']
                # Normalize
                d1_data[column] = (d1_data[column] - mean) / std_dev
                d2_data[column] = (d2_data[column] - mean) / std_dev
                d3_data[column] = (d3_data[column] - mean) / std_dev
            else:  # min-max
                min_val = norm_params['min']
                max_val = norm_params['max']
                range_min, range_max = self.params['range']
                # Normalize
                d1_data[column] = (d1_data[column] - min_val) / (max_val - min_val + epsilon) * (range_max - range_min) + range_min
                d2_data[column] = (d2_data[column] - min_val) / (max_val - min_val + epsilon) * (range_max - range_min) + range_min
                d3_data[column] = (d3_data[column] - min_val) / (max_val - min_val + epsilon) * (range_max - range_min) + range_min

            print(f"[DEBUG] Normalized column '{column}' using method '{method}'.")

        # Step 7: Save D1, D2, and D3 datasets (all numeric columns including low CV)
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

        # Step 8: Save the target datasets (excluding date and the first 5 columns)

        # Identify columns to include in target files
        columns_to_exclude = output_column_order  # ['d', 'o', 'l', 'h', 'c']
        columns_to_include_in_target = [col for col in columns_to_process if col not in columns_to_exclude]

        print(f"[DEBUG] Columns to include in target files: {columns_to_include_in_target}")

        # Create target datasets
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

        # Step 9: Plot all the columns (except date and the first 5 columns) of the post-processed D1 target file

        # Plotting each column in d1_target
        for column in d1_target.columns:
            plt.figure()
            plt.plot(d1_target[column].reset_index(drop=True))
            plt.title(f'D1 Target - {column}')
            plt.xlabel('Index')
            plt.ylabel(column)
            plt.savefig(f'd1_target_{column}.png')
            plt.close()

        print(f"[DEBUG] Plots of D1 target columns saved.")

        # Step 10: Save debug information
        debug_info = self.get_debug_info()
        debug_info_file = f"{target_prefix}debug_info.json"
        with open(debug_info_file, 'w') as f:
            json.dump(debug_info, f, indent=4)

        print(f"[DEBUG] Step 10: Saved debug information.")
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
