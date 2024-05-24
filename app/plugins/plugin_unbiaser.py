import pandas as pd
import numpy as np
import json
import os

class Plugin:
    def __init__(self):
        self.params = None

    def process(self, data, method='ma', window_size=5, ema_alpha=0.1, save_params=None, load_params=None):
        """
        Unbias the data using the specified method (moving average or EMA).

        Args:
            data (pd.DataFrame): The input data to be processed.
            method (str): The method to use for unbiasing ('ma' or 'ema').
            window_size (int): The window size for the moving average.
            ema_alpha (float): The alpha value for the exponential moving average.
            save_params (str): Path to save the parameters.
            load_params (str): Path to load the parameters.

        Returns:
            pd.DataFrame: The unbiassed data.
        """
        print("Starting the process method.")
        print(f"Method: {method}, Window size: {window_size}, EMA alpha: {ema_alpha}")
        print(f"Save params path: {save_params}, Load params path: {load_params}")

        if load_params and os.path.exists(load_params):
            with open(load_params, 'r') as f:
                self.params = json.load(f)
            print("Loaded parameters:", self.params)

        if self.params == None:
            self.params = {
                'method': method,
                'window_size': window_size,
                'ema_alpha': ema_alpha
            }
            if save_params:
                with open(save_params, 'w') as f:
                    json.dump(self.params, f)
            print("Saved parameters:", self.params)

        if self.params['method'] == 'ma':
            print("Applying moving average unbiasing.")
            processed_data = self._moving_average_unbias(data.iloc[:, 0:], self.params['window_size'])
        elif self.params['method'] == 'ema':
            print("Applying exponential moving average unbiasing.")
            processed_data = self._ema_unbias(data.iloc[:, 0:], self.params['ema_alpha'])
        else:
            raise ValueError(f"Unknown method: {self.params['method']}")

        # Concatenate the date column back with the processed data
        processed_data = pd.concat([data.iloc[:, 1], processed_data], axis=1)
        print("Processing complete. Returning processed data.")
        return processed_data

    def _moving_average_unbias(self, data, window_size):
        """
        Apply moving average unbiasing to the data.

        Args:
            data (pd.DataFrame): The input data to be processed.
            window_size (int): The window size for the moving average.

        Returns:
            pd.DataFrame: The unbiassed data.
        """
        print(f"Applying moving average with window size: {window_size}")
        unbiassed_data = data.astype(float).copy()  # Ensure all data is float

        for col in data.columns:
            print(f"Processing column: {col}")
            for i in range(len(data)):
                if i == 0:
                    # First row, subtracting the value itself
                    unbiassed_data.at[data.index[i], col] = 0.0
                elif i < window_size:
                    # For initial rows where the window is not fully populated
                    unbiassed_data.at[data.index[i], col] = data.at[data.index[i], col] - data[col][:i+1].mean()
                else:
                    # For rows where the window is fully populated
                    unbiassed_data.at[data.index[i], col] = data.at[data.index[i], col] - data[col][i-window_size+1:i+1].mean()

        print("Unbiassed data (first 5 rows):\n", unbiassed_data.head())
        return unbiassed_data

    def _ema_unbias(self, data, alpha):
        """
        Apply exponential moving average unbiasing to the data.

        Args:
            data (pd.DataFrame): The input data to be processed.
            alpha (float): The alpha value for the exponential moving average.

        Returns:
            pd.DataFrame: The unbiassed data.
        """
        print(f"Applying exponential moving average with alpha: {alpha}")
        ema = data.ewm(alpha=alpha).mean()
        print("Exponential moving average values:\n", ema.head())
        unbiassed_data = data - ema
        print("Unbiassed data (first 5 rows):\n", unbiassed_data.head())
        return unbiassed_data
