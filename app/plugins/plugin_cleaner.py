import pandas as pd
import numpy as np
import json
import os

class Plugin:
    def __init__(self):
        self.params = None

    def process(self, data, method='missing_values', period=5, outlier_threshold=None, solve_missing=False, delete_outliers=False, interpolate_outliers=False, delete_nan=False, interpolate_nan=False, save_params=None, load_params=None):
        """
        Clean the data using the specified method.
        Args:
            data (pd.DataFrame): The input data to be processed.
            method (str): The method to use for cleaning ('missing_values' or 'outlier').
            period (int): The period in minutes for continuity checking.
            outlier_threshold (float): The threshold for outlier detection.
            solve_missing (bool): Whether to solve missing values.
            delete_outliers (bool): Whether to delete outliers.
            interpolate_outliers (bool): Whether to interpolate outliers.
            delete_nan (bool): Whether to delete rows with NaN values.
            interpolate_nan (bool): Whether to interpolate NaN values.
            save_params (str): Path to save the parameters.
            load_params (str): Path to load the parameters.
        Returns:
            pd.DataFrame: The cleaned data.
        """
        print("Starting the process method.")
        print(f"Method: {method}, Period: {period} minutes, Outlier threshold: {outlier_threshold}")
        print(f"Solve missing: {solve_missing}, Delete outliers: {delete_outliers}, Interpolate outliers: {interpolate_outliers}")
        print(f"Delete NaN: {delete_nan}, Interpolate NaN: {interpolate_nan}")

        # Load or save parameters as needed
        if load_params and os.path.exists(load_params):
            with open(load_params, 'r') as f:
                self.params = json.load(f)
            print("Loaded parameters:", self.params)

        if self.params is None:
            self.params = {
                'method': method,
                'period': period,
                'outlier_threshold': outlier_threshold,
                'solve_missing': solve_missing,
                'delete_outliers': delete_outliers,
                'interpolate_outliers': interpolate_outliers,
                'delete_nan': delete_nan,
                'interpolate_nan': interpolate_nan
            }
            if save_params:
                with open(save_params, 'w') as f:
                    json.dump(self.params, f)
            print("Saved parameters:", self.params)

        if method == 'missing_values':
            cleaned_data = self._handle_missing_values(data, period, solve_missing)
        elif method == 'outlier':
            cleaned_data = self._handle_outliers(data, outlier_threshold, delete_outliers, interpolate_outliers, delete_nan, interpolate_nan)
        else:
            raise ValueError(f"Unknown method: {method}")

        print("Processing complete. Returning cleaned data.")
        return cleaned_data

    def _handle_missing_values(self, data, period, solve_missing):
        """
        Handle missing values in the data.
        Args:
            data (pd.DataFrame): The input data to be processed.
            period (int): The period in minutes for continuity checking.
            solve_missing (bool): Whether to solve missing values.
        Returns:
            pd.DataFrame: The cleaned data.
        """
        print(f"Handling missing values with period: {period} minutes, Solve missing: {solve_missing}")

        data['date'] = pd.to_datetime(data.index, format='%m/%d/%Y %H:%M')

        # Create a complete date range
        complete_date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq=f'{period}T')

        # Reindex the data to this complete date range
        data = data.reindex(complete_date_range)

        if solve_missing:
            data = data.interpolate(method='time')

        print("Missing values handled (first 5 rows):\n", data.head())
        return data

    def _handle_outliers(self, data, threshold, delete_outliers, interpolate_outliers, delete_nan, interpolate_nan):
        """
        Handle outliers in the data.
        Args:
            data (pd.DataFrame): The input data to be processed.
            threshold (float): The threshold for outlier detection.
            delete_outliers (bool): Whether to delete outliers.
            interpolate_outliers (bool): Whether to interpolate outliers.
            delete_nan (bool): Whether to delete rows with NaN values.
            interpolate_nan (bool): Whether to interpolate NaN values.
        Returns:
            pd.DataFrame: The cleaned data.
        """
        print(f"Handling outliers with threshold: {threshold}, Delete outliers: {delete_outliers}, Interpolate outliers: {interpolate_outliers}")
        print(f"Delete NaN: {delete_nan}, Interpolate NaN: {interpolate_nan}")

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if delete_outliers:
            data = data[(np.abs(data[numeric_cols] - data[numeric_cols].mean()) <= (threshold * data[numeric_cols].std()))]

        if interpolate_outliers:
            data[numeric_cols] = data[numeric_cols].apply(lambda x: x.mask((x - x.mean()).abs() > (threshold * x.std())).interpolate())

        if delete_nan:
            data = data.dropna()

        if interpolate_nan:
            data = data.interpolate(method='linear')

        print("Outliers handled (first 5 rows):\n", data.head())
        return data
