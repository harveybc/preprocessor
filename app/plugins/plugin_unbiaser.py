import pandas as pd
import numpy as np

class Plugin:
    plugin_params = ['method', 'window_size', 'ema_alpha']

    def __init__(self):
        self.method = 'ma'
        self.window_size = 5
        self.ema_alpha = 0.1

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def process(self, data):
        print("Starting the process method.")
        print(f"Method: {self.method}, Window size: {self.window_size}, EMA alpha: {self.ema_alpha}")

        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Numeric columns identified for processing: {numeric_columns}")

        if self.method == 'ma':
            print("Applying moving average unbiasing.")
            processed_data = self._moving_average_unbias(data[numeric_columns], self.window_size)
        elif self.method == 'ema':
            print("Applying exponential moving average unbiasing.")
            processed_data = self._ema_unbias(data[numeric_columns], self.ema_alpha)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        print("Processing complete. Returning processed data.")
        return processed_data

    def _moving_average_unbias(self, data, window_size):
        print(f"Applying moving average with window size: {window_size}")
        unbiassed_data = data.astype(float).copy()

        for col in data.columns:
            print(f"Processing column: {col}")
            for i in range(len(data)):
                window = data[col][max(0, i-window_size+1):i+1].mean()
                unbiassed_data.at[data.index[i], col] = data.at[data.index[i], col] - window

        print("Unbiassed data (first 5 rows):\n", unbiassed_data.head())
        return unbiassed_data

    def _ema_unbias(self, data, alpha):
        print(f"Applying exponential moving average with alpha: {alpha}")
        ema = data.ewm(alpha=alpha).mean()
        unbiassed_data = data - ema

        print("Exponential moving average values:\n", ema.head())
        print("Unbiassed data (first 5 rows):\n", unbiassed_data.head())
        return unbiassed_data
