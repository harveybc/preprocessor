import pandas as pd
import numpy as np
import json
import os

class Plugin:
    def __init__(self):
        self.params = None

    def process(self, data, method='ma', window_size=5, ema_alpha=0.1, save_params=None, load_params=None):
        print("Starting the process method.")
        print(f"Method: {method}, Window size: {window_size}, EMA alpha: {ema_alpha}")

        # Load or initialize parameters
        if load_params and os.path.exists(load_params):
            with open(load_params, 'r') as f:
                self.params = json.load(f)
            print("Loaded parameters:", self.params)
        else:
            self.params = {'method': method, 'window_size': window_size, 'ema_alpha': ema_alpha}
            if save_params:
                with open(save_params, 'w') as f:
                    json.dump(self.params, f)
            print("Saved parameters:", self.params)

        # Select only numeric columns excluding the first column presumed to be date
        numeric_data = data.iloc[:, 1:].select_dtypes(include=[np.number])
        if self.params['method'] == 'ma':
            print("Applying moving average unbiasing.")
            processed_data = self._moving_average_unbias(numeric_data, self.params['window_size'])
        elif self.params['method'] == 'ema':
            print("Applying exponential moving average unbiasing.")
            processed_data = self._ema_unbias(numeric_data, self.params['ema_alpha'])
        else:
            raise ValueError(f"Unknown method: {self.params['method']}")

        # Concatenate the date column back with the processed data
        processed_data = pd.concat([data.iloc[:, 0], processed_data], axis=1)
        print("Processing complete. Returning processed data.")
        return processed_data

    def _moving_average_unbias(self, data, window_size):
        print(f"Applying moving average with window size: {window_size}")
        unbiassed_data = data.copy()  # Make a copy of the data to avoid changing original DataFrame
        for col in unbiassed_data.columns:
            print(f"Processing column: {col}")
            unbiassed_data[col] = data[col].rolling(window=window_size, min_periods=1).mean().apply(lambda x: data[col] - x)
        print("Unbiassed data (first 5 rows):\n", unbiassed_data.head())
        return unbiassed_data

    def _ema_unbias(self, data, alpha):
        print(f"Applying exponential moving average with alpha: {alpha}")
        ema = data.ewm(alpha=alpha).mean()
        unbiassed_data = data - ema
        print("Unbiassed data (first 5 rows):\n", unbiassed_data.head())
        return unbiassed_data

