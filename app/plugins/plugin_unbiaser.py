import pandas as pd
import numpy as np

class Plugin:
    """
    Unbiaser Plugin to apply unbiasing methods to the dataset.
    """

    # Define the parameters for this plugin and their default values
    plugin_params = {
        'method': 'ma',
        'window_size': 5,
        'ema_alpha': 0.1
    }

    # Define the debug variables for this plugin
    plugin_debug_vars = ['method', 'window_size', 'ema_alpha']

    def __init__(self):
        """
        Initialize the Plugin with default parameters.
        """
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        """
        Set the parameters for the plugin.

        Args:
            **kwargs: Arbitrary keyword arguments for plugin parameters.
        """
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        """
        Get debug information for the plugin.

        Returns:
            dict: Debug information including method, window_size, and ema_alpha.
        """
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Add plugin-specific debug information to the existing debug info.

        Args:
            debug_info (dict): The existing debug information dictionary.
        """
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def process(self, data):
        """
        Perform unbiasing on the dataset using the specified method.

        Args:
            data (pd.DataFrame): The input data to be processed.

        Returns:
            pd.DataFrame: The dataset with unbiasing applied.
        """
        method = self.params.get('method', 'ma')
        window_size = int(self.params.get('window_size', 5))  # Ensure window_size is an integer
        ema_alpha = float(self.params.get('ema_alpha', 0.1))  # Ensure ema_alpha is a float

        print("Starting the process method.")
        print(f"Method: {method}, Window size: {window_size}, EMA alpha: {ema_alpha}")

        # Identify numeric columns excluding the date column which should be at index 0
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Numeric columns identified for processing: {numeric_columns}")

        # Apply the selected unbiasing method
        if method == 'ma':
            print("Applying moving average unbiasing.")
            processed_data = self._moving_average_unbias(data[numeric_columns], window_size)
        elif method == 'ema':
            print("Applying exponential moving average unbiasing.")
            processed_data = self._ema_unbias(data[numeric_columns], ema_alpha)
        else:
            raise ValueError(f"Unknown method: {method}")

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
                window = data[col][max(0, i-window_size+1):i+1].mean()
                unbiassed_data.at[data.index[i], col] = data.at[data.index[i], col] - window

        print("Unbiassed data (first 5 rows):\n", unbiassed_data.head())
        return unbiassed_data

    def _ema_unbias(self, data, alpha):
        """
        Apply exponential moving average unbiasing to the data.

        Args:
            data (pd.DataFrame): The input data to be processed.
            alpha (float): The alpha parameter for the exponential moving average.

        Returns:
            pd.DataFrame: The unbiassed data.
        """
        print(f"Applying exponential moving average with alpha: {alpha}")
        ema = data.ewm(alpha=alpha).mean()
        unbiassed_data = data - ema

        print("Exponential moving average values:\n", ema.head())
        print("Unbiassed data (first 5 rows):\n", unbiassed_data.head())
        return unbiassed_data
