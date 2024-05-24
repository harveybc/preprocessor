import pandas as pd
import numpy as np
import json
import os

class Plugin:
    def __init__(self):
        self.params = None

    def process(self, data, method='continuity', frequency='5T', outlier_threshold=3.0, save_params=None, load_params=None):
        """
        Clean the data using the specified method (continuity or outlier).
        Only numeric columns after the date column are processed.

        Args:
            data (pd.DataFrame): The input data to be processed.
            method (str): The method to use for cleaning ('continuity' or 'outlier').
            frequency (str): The expected frequency of the datetime index.
            outlier_threshold (float): The threshold for outlier detection.
            save_params (str): Path to save the parameters.
            load_params (str): Path to load the parameters.

        Returns:
            pd.DataFrame: The cleaned data.
        """
        print("Starting the process method.")
        print(f"Method: {method}, Frequency: {frequency}, Outlier threshold: {outlier_threshold}")

        # Identify numeric columns excluding the date column which should be at index 0
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Numeric columns identified for processing: {numeric_columns}")

        # Load or save parameters as needed
        if load_params and os.path.exists(load_params):
            with open(load_params, 'r') as f:
                self.params = json.load(f)
            print("Loaded parameters:", self.params)

        if self.params is None:
            self.params = {'method': method, 'frequency': frequency, 'outlier_threshold': outlier_threshold}
            if save_params:
                with open(save_params, 'w') as f:
                    json.dump(self.params, f)
            print("Saved parameters:", self.params)

        # Apply the selected cleaning method
        if method == 'continuity':
            print("Verifying continuity of data.")
            processed_data = self._verify_continuity(data, frequency)
        elif method == 'outlier':
            print("Removing or correcting outliers.")
            processed_data = self._remove_outliers(data[numeric_columns], outlier_threshold)
        else:
            raise ValueError(f"Unknown method: {method}")

        print("Processing complete. Returning processed data.")
        return processed_data

    def _verify_continuity(self, data, frequency):
        """
        Verify the continuity of the data and generate missing values if needed.

        Args:
            data (pd.DataFrame): The input data to be processed.
            frequency (str): The expected frequency of the datetime index.

        Returns:
            pd.DataFrame: The data with continuity verified.
        """
        print(f"Verifying data continuity with frequency: {frequency}")
        data.set_index(data.columns[0], inplace=True)  # Set the date column as index
        data.index = pd.to_datetime(data.index)

        all_times = pd.date_range(start=data.index.min(), end=data.index.max(), freq=frequency)
        missing_times = all_times.difference(data.index)
        print(f"Missing times identified: {missing_times}")

        for missing_time in missing_times:
            data.loc[missing_time] = np.nan

        data.sort_index(inplace=True)
        data.interpolate(method='time', inplace=True)
        print("Data after continuity check (first 5 rows):\n", data.head())
        return data.reset_index()

    def _remove_outliers(self, data, threshold):
        """
        Remove or correct outliers in the data.

        Args:
            data (pd.DataFrame): The input data to be processed.
            threshold (float): The threshold for outlier detection.

        Returns:
            pd.DataFrame: The data with outliers removed or corrected.
        """
        print(f"Removing outliers with threshold: {threshold}")
        for col in data.columns:
            print(f"Processing column: {col}")
            rolling_mean = data[col].rolling(window=5, center=True).mean()
            rolling_std = data[col].rolling(window=5, center=True).std()
            outliers = (data[col] - rolling_mean).abs() > threshold * rolling_std
            print(f"Outliers detected: {outliers.sum()}")

            data.loc[outliers, col] = np.nan
            data[col].interpolate(method='linear', inplace=True)

        print("Data after outlier removal (first 5 rows):\n", data.head())
        return data

# Remember to ensure this plugin's code also checks whether parameters like 'frequency' and 'outlier_threshold' are set appropriately before use.
