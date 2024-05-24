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
        if load_params and os.path.exists(load_params):
            with open(load_params, 'r') as f:
                self.params = json.load(f)

        if self.params is None:
            self.params = {
                'method': method,
                'window_size': window_size,
                'ema_alpha': ema_alpha
            }
            if save_params:
                with open(save_params, 'w') as f:
                    json.dump(self.params, f)

        if self.params['method'] == 'ma':
            return self._moving_average_unbias(data, self.params['window_size'])
        elif self.params['method'] == 'ema':
            return self._ema_unbias(data, self.params['ema_alpha'])
        else:
            raise ValueError(f"Unknown method: {self.params['method']}")

    def _moving_average_unbias(self, data, window_size):
        return data.rolling(window=window_size, min_periods=1).mean()

    def _ema_unbias(self, data, alpha):
        return data.ewm(alpha=alpha).mean()
