import pandas as pd
import numpy as np
import json
import os
from statsmodels.tsa.stattools import grangercausalitytests

class Plugin:
    def __init__(self):
        self.feature_selection_params = None

    def process(self, data, method='granger', save_params=None, load_params=None, max_lag=5, significance_level=0.05):
        """
        Perform feature selection on the dataset using the specified method.

        Args:
            data (pd.DataFrame): The input data to be processed.
            method (str): The method for feature selection ('acf', 'pacf', 'granger').
            save_params (str): Path to save the feature selection parameters.
            load_params (str): Path to load the feature selection parameters.
            max_lag (int): Maximum lag for the Granger causality test.
            significance_level (float): Significance level for the statistical tests.

        Returns:
            pd.DataFrame: The dataset with only the selected features.
        """
        if load_params and os.path.exists(load_params):
            with open(load_params, 'r') as f:
                self.feature_selection_params = json.load(f)
            selected_features = self.feature_selection_params.get('selected_features', data.columns.tolist())
        else:
            selected_features = self._select_features(data, method, max_lag, significance_level)
            self.feature_selection_params = {'method': method, 'selected_features': selected_features}
            if save_params:
                with open(save_params, 'w') as f:
                    json.dump(self.feature_selection_params, f)

        return data[selected_features]

    def _select_features(self, data, method, max_lag, significance_level):
        """
        Select features based on the specified method.

        Args:
            data (pd.DataFrame): The input data to be processed.
            method (str): The method for feature selection.
            max_lag (int): Maximum lag for the Granger causality test.
            significance_level (float): Significance level for the statistical tests.

        Returns:
            list: List of selected features.
        """
        if method == 'acf':
            return self._acf_feature_selection(data, significance_level)
        elif method == 'pacf':
            return self._pacf_feature_selection(data, significance_level)
        elif method == 'granger':
            return self._granger_causality_feature_selection(data, max_lag, significance_level)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")

    def _acf_feature_selection(self, data, significance_level):
        """
        Select features based on Autocorrelation Function (ACF).

        Args:
            data (pd.DataFrame): The input data to be processed.
            significance_level (float): Significance level for ACF.

        Returns:
            list: List of selected features.
        """
        selected_features = []
        for column in data.columns:
            acf_values = [abs(val) for val in np.correlate(data[column], data[column], mode='full')]
            if any(val > significance_level for val in acf_values):
                selected_features.append(column)
        return selected_features

    def _pacf_feature_selection(self, data, significance_level):
        """
        Select features based on Partial Autocorrelation Function (PACF).

        Args:
            data (pd.DataFrame): The input data to be processed.
            significance_level (float): Significance level for PACF.

        Returns:
            list: List of selected features.
        """
        selected_features = []
        for column in data.columns:
            pacf_values = [abs(val) for val in np.correlate(data[column], data[column], mode='full')]
            if any(val > significance_level for val in pacf_values):
                selected_features.append(column)
        return selected_features

    def _granger_causality_feature_selection(self, data, max_lag, significance_level):
        """
        Select features based on Granger Causality Test.

        Args:
            data (pd.DataFrame): The input data to be processed.
            max_lag (int): Maximum lag for the Granger causality test.
            significance_level (float): Significance level for the Granger causality test.

        Returns:
            list: List of selected features.
        """
        selected_features = []
        target_column = data.columns[0]  # Assuming the first column is the target column
        for column in data.columns:
            if column != target_column:
                test_result = grangercausalitytests(data[[target_column, column]], max_lag, verbose=False)
                p_values = [test_result[i+1][0]['ssr_chi2test'][1] for i in range(max_lag)]
                if any(p_val < significance_level for p_val in p_values):
                    selected_features.append(column)
        return selected_features

    def select_single(self, data, column_index):
        """
        Select a single column from the dataset.

        Args:
            data (pd.DataFrame): The input data to be processed.
            column_index (int): The index of the column to select.

        Returns:
            pd.DataFrame: The dataset with only the selected column.
        """
        if column_index < 0 or column_index >= len(data.columns):
            raise ValueError(f"Column index {column_index} is out of bounds.")
        selected_column = [data.columns[column_index]]
        return data[selected_column]

    def select_multi(self, data, column_indices):
        """
        Select multiple columns from the dataset.

        Args:
            data (pd.DataFrame): The input data to be processed.
            column_indices (list): The indices of the columns to select.

        Returns:
            pd.DataFrame: The dataset with only the selected columns.
        """
        selected_columns = []
        for index in column_indices:
            if index < 0 or index >= len(data.columns):
                raise ValueError(f"Column index {index} is out of bounds.")
            selected_columns.append(data.columns[index])
        return data[selected_columns]
