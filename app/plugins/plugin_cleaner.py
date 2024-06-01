import pandas as pd
import numpy as np
import json
import os

class Plugin:
    def __init__(self):
        self.params = None

    def process(self, data, method='missing_values', period=5, outlier_threshold=None, solve_missing=False, delete_outliers=False, interpolate_outliers=False, delete_nan=False, interpolate_nan=False, save_params=None, load_params=None):
        """
        Clean the data using the specified method (handling missing values or outliers).

        Args:
            data (pd.DataFrame): The input data to be processed.
            method (str): The method to use for cleaning ('missing_values' or 'outlier').
            period (int): The expected period in minutes for continuity checking.
            outlier_threshold (float): The threshold for outlier detection.
            solve_missing (bool): Whether to solve missing values by interpolation.
            delete_outliers (bool): Whether to delete outlier rows.
            interpolate_outliers (bool): Whether to interpolate outlier values.
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

        # Load parameters if provided
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

        # Apply the selected cleaning method
        if method == 'missing_values':
            cleaned_data = self._handle_missing_values(data, period, solve_missing)
        elif method == 'outlier':
            cleaned_data = self._handle_outliers(data, outlier_threshold, delete_outliers, interpolate_outliers, delete_nan, interpolate_nan)
        else:
            raise ValueError(f"Unknown method: {method}")

        print("Cleaning complete. Returning cleaned data.")
        return cleaned_data

    def _handle_missing_values(self, data, period, solve_missing):
        """
        Handle missing values in the data.

        Args:
            data (pd.DataFrame): The input data to be processed.
            period (int): The expected period in minutes for continuity checking.
            solve_missing (bool): Whether to solve missing values by interpolation.

        Returns:
            pd.DataFrame: The data with handled missing values.
        """
        print(f"Handling missing values with period: {period} minutes, Solve missing: {solve_missing}")

        # Ensure the index is a datetime type
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            print("Converting index to datetime.")
            data.index = pd.to_datetime(data.index, errors='coerce')
        
        if data.index.isnull().any():
            raise ValueError("The date conversion resulted in NaT values. Please check the date format in the input data.")

        # Generate a full date range based on the period
        full_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq=f'{period}min')

        # Identify and report missing values
        missing = full_range.difference(data.index)
        if not missing.empty:
            print(f"Missing timestamps found: {missing}")
            if solve_missing:
                print("Solving missing values by interpolation.")
                missing_df = pd.DataFrame(index=missing, columns=data.columns).infer_objects()
                data = pd.concat([data, missing_df]).sort_index()
                data.interpolate(method='linear', inplace=True)

        return data

    def _handle_outliers(self, data, outlier_threshold, delete_outliers, interpolate_outliers, delete_nan, interpolate_nan):
        """
        Handle outliers in the data.

        Args:
            data (pd.DataFrame): The input data to be processed.
            outlier_threshold (float): The threshold for outlier detection.
            delete_outliers (bool): Whether to delete outlier rows.
            interpolate_outliers (bool): Whether to interpolate outlier values.
            delete_nan (bool): Whether to delete rows with NaN values.
            interpolate_nan (bool): Whether to interpolate NaN values.

        Returns:
            pd.DataFrame: The data with handled outliers.
        """
        print(f"Handling outliers with threshold: {outlier_threshold}, Delete outliers: {delete_outliers}, Interpolate outliers: {interpolate_outliers}")
        print(f"Delete NaN: {delete_nan}, Interpolate NaN: {interpolate_nan}")

        for col in data.columns:
            if outlier_threshold is not None:
                # Identify outliers
                mean = data[col].mean()
                std = data[col].std()
                outliers = (data[col] - mean).abs() > outlier_threshold * std

                if outliers.any():
                    print(f"Outliers detected in column {col}: {data[col][outliers]}")
                    if delete_outliers:
                        data = data[~outliers]
                    elif interpolate_outliers:
                        data[col][outliers] = np.nan

            if delete_nan:
                data.dropna(inplace=True)
            elif interpolate_nan:
                data.interpolate(method='linear', inplace=True)

        return data
