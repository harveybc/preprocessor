import pandas as pd
import json
import os

class Plugin:
    def __init__(self):
        # Initialize the unbias parameters to None
        self.unbias_params = None

    def process(self, data, method='ma', window_sizes=None, save_params=None, load_params=None, ema_alphas=None):
        """
        Remove bias from the data using the specified method for each column.

        Args:
            data (pd.DataFrame): The input data to be processed.
            method (str): The method for bias removal ('ma' or 'ema').
            window_sizes (list): List of window sizes for the moving average method for each column.
            save_params (str): Path to save the unbias parameters.
            load_params (str): Path to load the unbias parameters.
            ema_alphas (list): List of alpha parameters for the exponential moving average method for each column.

        Returns:
            pd.DataFrame: The unbiased data.
        """
        # Load parameters if load_params path is provided
        if load_params and os.path.exists(load_params):
            with open(load_params, 'r') as f:
                self.unbias_params = json.load(f)

        # Initialize lists if they are not provided
        if window_sizes is None:
            window_sizes = [3] * data.shape[1]
        if ema_alphas is None:
            ema_alphas = [0.3] * data.shape[1]

        if method == 'ma':
            if self.unbias_params is None:
                # Save the provided parameters if they are not already loaded
                self.unbias_params = {'method': 'ma', 'window_sizes': window_sizes}
                unbiased_data = pd.DataFrame({col: self._ma_unbias(data[col], window_sizes[idx]) 
                                              for idx, col in enumerate(data.columns)})
                # Save the parameters to a file if save_params path is provided
                if save_params:
                    with open(save_params, 'w') as f:
                        json.dump(self.unbias_params, f)
            else:
                # Load window sizes from the loaded parameters
                window_sizes = self.unbias_params['window_sizes']
                unbiased_data = pd.DataFrame({col: self._ma_unbias(data[col], window_sizes[idx]) 
                                              for idx, col in enumerate(data.columns)})

        elif method == 'ema':
            if self.unbias_params is None:
                # Save the provided parameters if they are not already loaded
                self.unbias_params = {'method': 'ema', 'ema_alphas': ema_alphas}
                unbiased_data = pd.DataFrame({col: self._ema_unbias(data[col], ema_alphas[idx]) 
                                              for idx, col in enumerate(data.columns)})
                # Save the parameters to a file if save_params path is provided
                if save_params:
                    with open(save_params, 'w') as f:
                        json.dump(self.unbias_params, f)
            else:
                # Load EMA alphas from the loaded parameters
                ema_alphas = self.unbias_params['ema_alphas']
                unbiased_data = pd.DataFrame({col: self._ema_unbias(data[col], ema_alphas[idx]) 
                                              for idx, col in enumerate(data.columns)})

        else:
            raise ValueError(f"Unknown unbias method: {method}")

        return unbiased_data

    def _ma_unbias(self, series, window_size):
        """
        Apply moving average bias removal to a series.

        Args:
            series (pd.Series): The input data series.
            window_size (int): The window size for the moving average.

        Returns:
            pd.Series: The unbiased data series.
        """
        # Initialize the unbiased series with the same values as the input series
        unbiased_series = series.copy()
        for i in range(len(series)):
            if i < window_size:
                # For elements in positions less than window_size, use the average of the previous ticks
                avg = series[:i+1].mean()
            else:
                # For other elements, use the average of the last window_size elements
                avg = series[i-window_size+1:i+1].mean()
            # Subtract the average from the current value to remove bias
            unbiased_series[i] = series[i] - avg
        return unbiased_series

    def _ema_unbias(self, series, alpha):
        """
        Apply exponential moving average bias removal to a series.

        Args:
            series (pd.Series): The input data series.
            alpha (float): The smoothing factor for EMA.

        Returns:
            pd.Series: The unbiased data series.
        """
        # Calculate the exponential moving average
        ema = series.ewm(alpha=alpha).mean()
        # Subtract the EMA from the original series to remove bias
        return series - ema
