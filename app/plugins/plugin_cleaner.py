import pandas as pd
import numpy as np
import json
import os

class Plugin:
    def __init__(self):
        self.params = None

    def process(self, data, method='missing_values', period=5, outlier_threshold=None, solve_missing=False, delete_outliers=False, interpolate_outliers=False, delete_nan=False, interpolate_nan=False, save_params=None, load_params=None):
        """
        Clean the data using the specified method (missing values or outlier detection).

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
            print(f"Handling missing values with period: {period} minutes, Solve missing: {solve_missing}")
            cleaned_data = self._handle_missing_values(data, period, solve_missing)
        elif method == 'outlier':
            print(f"Handling outliers with threshold: {outlier_threshold}, Delete outliers: {delete_outliers}, Interpolate outliers: {interpolate_outliers}")
            cleaned_data = self._handle_outliers(data, outlier_threshold, delete_outliers, interpolate_outliers, delete_nan, interpolate_nan)
        else:
            raise ValueError(f"Unknown method: {method}")

        print("Processing complete. Returning cleaned data.")
        return cleaned_data

    def _handle_missing_values(self, data, period, solve_missing):
        """
        Handle missing values based on the specified period.

        Args:
            data (pd.DataFrame): The input data to be processed.
            period (int): The period in minutes for continuity checking.
            solve_missing (bool): Whether to solve missing values.

        Returns:
            pd.DataFrame: The data with missing values handled.
        """
        data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y %H:%M')
        data = data.set_index('date').sort_index()
        full_index = pd.date_range(start=data.index[0], end=data.index[-1], freq=f'{period}T')
        missing_ticks = full_index.difference(data.index)

        if len(missing_ticks) > 0:
            print(f"Missing ticks found: {len(missing_ticks)}")
            if not self.params['quiet_mode']:
                print("Missing ticks:")
                for missing_tick in missing_ticks:
                    print(missing_tick)
        
        if solve_missing:
            print("Solving missing ticks...")
            for missing_tick in missing_ticks:
                prev_tick = data[:missing_tick].iloc[-1]
                next_tick = data[missing_tick:].iloc[0]
                new_row = (prev_tick + next_tick) / 2
                new_row.name = missing_tick
                data = data.append(new_row).sort_index()

        return data

    def _handle_outliers(self, data, threshold, delete_outliers, interpolate_outliers, delete_nan, interpolate_nan):
        """
        Handle outliers based on the specified threshold.

        Args:
            data (pd.DataFrame): The input data to be processed.
            threshold (float): The threshold for outlier detection.
            delete_outliers (bool): Whether to delete outliers.
            interpolate_outliers (bool): Whether to interpolate outliers.
            delete_nan (bool): Whether to delete rows with NaN values.
            interpolate_nan (bool): Whether to interpolate NaN values.

        Returns:
            pd.DataFrame: The data with outliers handled.
        """
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_columns:
            col_mean = data[col].mean()
            col_std = data[col].std()
            outliers = (data[col] - col_mean).abs() > (threshold * col_std)
            
            if outliers.any():
                print(f"Outliers detected in column {col}: {outliers.sum()}")
                if delete_outliers:
                    data = data[~outliers]
                elif interpolate_outliers:
                    data[col] = data[col].mask(outliers).interpolate()

        if delete_nan:
            data = data.dropna()

        if interpolate_nan:
            data = data.interpolate()

        return data
