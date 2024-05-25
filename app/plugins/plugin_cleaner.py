import pandas as pd
import numpy as np
import json
import os
from datetime import timedelta

class Plugin:
    def __init__(self):
        self.params = None

    def process(self, data, method='missing_values', period=5, outlier_threshold=3.0, solve_missing=False, delete_outliers=False, interpolate_outliers=False, delete_nan=False, interpolate_nan=False, save_params=None, load_params=None):
        """
        Clean the data using the specified method (missing values or outlier).

        Args:
            data (pd.DataFrame): The input data to be processed.
            method (str): The method to use for cleaning ('missing_values' or 'outlier').
            period (int): The expected period in minutes for continuity checking.
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

        # Load parameters if specified
        if load_params and os.path.exists(load_params):
            with open(load_params, 'r') as f:
                self.params = json.load(f)
            print("Loaded parameters:", self.params)

        # Save parameters if specified
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

        # Apply the selected cleaning method
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
        Handle missing values in the data based on the specified period.

        Args:
            data (pd.DataFrame): The input data to be processed.
            period (int): The expected period in minutes for continuity checking.
            solve_missing (bool): Whether to solve missing values by interpolation.

        Returns:
            pd.DataFrame: The data with handled missing values.
        """
        print(f"Handling missing values with period: {period} minutes, Solve missing: {solve_missing}")
        data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y %H:%M')
        data.set_index('date', inplace=True)

        start_date = data.index.min()
        end_date = data.index.max()
        full_range = pd.date_range(start=start_date, end=end_date, freq=f'{period}T')

        missing_dates = full_range.difference(data.index)
        print(f"Missing dates: {missing_dates}")

        if not missing_dates.empty:
            if solve_missing:
                for missing_date in missing_dates:
                    prev_value = data.loc[:missing_date].iloc[-1]
                    next_value = data.loc[missing_date:].iloc[1]
                    avg_value = (prev_value + next_value) / 2
                    data.loc[missing_date] = avg_value
                data.sort_index(inplace=True)
                print("Missing dates solved by interpolation.")
            else:
                print("Missing dates found but not solved.")

        data.reset_index(inplace=True)
        return data

    def _handle_outliers(self, data, threshold, delete_outliers, interpolate_outliers, delete_nan, interpolate_nan):
        """
        Handle outliers and NaN values in the data.

        Args:
            data (pd.DataFrame): The input data to be processed.
            threshold (float): The threshold for outlier detection.
            delete_outliers (bool): Whether to delete outliers.
            interpolate_outliers (bool): Whether to interpolate outliers.
            delete_nan (bool): Whether to delete rows with NaN values.
            interpolate_nan (bool): Whether to interpolate NaN values.

        Returns:
            pd.DataFrame: The data with handled outliers and NaN values.
        """
        print(f"Handling outliers with threshold: {threshold}, Delete outliers: {delete_outliers}, Interpolate outliers: {interpolate_outliers}")
        print(f"Delete NaN: {delete_nan}, Interpolate NaN: {interpolate_nan}")

        numeric_data = data.select_dtypes(include=[np.number])
        for col in numeric_data.columns:
            print(f"Processing column: {col}")
            mean = numeric_data[col].mean()
            std = numeric_data[col].std()
            outliers = (numeric_data[col] - mean).abs() > threshold * std
            print(f"Outliers detected: {numeric_data[outliers]}")

            if delete_outliers:
                data = data[~outliers]
                print("Outliers deleted.")
            elif interpolate_outliers:
                numeric_data[col][outliers] = np.nan
                numeric_data[col].interpolate(method='linear', inplace=True)
                print("Outliers interpolated.")

        if delete_nan:
            data = data.dropna()
            print("Rows with NaN values deleted.")
        elif interpolate_nan:
            numeric_data.interpolate(method='linear', inplace=True)
            data[numeric_data.columns] = numeric_data
            print("NaN values interpolated.")

        return data
