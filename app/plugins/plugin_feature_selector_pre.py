import pandas as pd
import numpy as np
import json
import os
from statsmodels.tsa.stattools import grangercausalitytests

class Plugin:
    def __init__(self):
        # Initialize the feature selection parameters to None
        self.feature_selection_params = None

    def process(self, data, method='granger', save_params=None, load_params=None, max_lag=5, significance_level=0.05, column=None, columns=None):
        """
        Perform feature selection on the dataset using the specified method.

        Args:
            data (pd.DataFrame): The input data to be processed.
            method (str): The method for feature selection ('acf', 'pacf', 'granger', 'select_single', 'select_multi').
            save_params (str): Path to save the feature selection parameters.
            load_params (str): Path to load the feature selection parameters.
            max_lag (int): Maximum lag for the Granger causality test.
            significance_level (float): Significance level for the statistical tests.
            column (int): Index of the single column to select.
            columns (list): List of column indices to select.

        Returns:
            pd.DataFrame: The dataset with only the selected features.
        """
        # Load parameters if load_params path is provided
        if load_params and os.path.exists(load_params):
            with open(load_params, 'r') as f:
                self.feature_selection_params = json.load(f)
            # Load the selected features from the loaded parameters
            selected_features = self.feature_selection_params.get('selected_features', data.columns.tolist())

        if self.feature_selection_params is None:
            if method == 'acf':
                selected_features = self._acf_feature_selection(data, significance_level)
            elif method == 'pacf':
                selected_features = self._pacf_feature_selection(data, significance_level)
            elif method == 'granger':
                selected_features = self._granger_causality_feature_selection(data, max_lag, significance_level)
            elif method == 'select_single':
                selected_features = self._select_single_column(data, column)
            elif method == 'select_multi':
                selected_features = self._select_multiple_columns(data, columns)
            else:
                raise ValueError(f"Unknown feature selection method: {method}")

            # Save the selected features if save_params path is provided
            self.feature_selection_params = {'method': method, 'selected_features': selected_features}
            if save_params:
                with open(save_params, 'w') as f:
                    json.dump(self.feature_selection_params, f)
        else:
            # Load the selected features from the parameters
            selected_features = self.feature_selection_params['selected_features']

        # Return the dataset with only the selected features
        return data[selected_features]

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

    def _select_single_column(self, data, column):
        """
        Select a single column based on the given index.

        Args:
            data (pd.DataFrame): The input data to be processed.
            column (int): Index of the column to select.

        Returns:
            list: List containing the selected column.
        """
        return [data.columns[column]]

    def _select_multiple_columns(self, data, columns):
        """
        Select multiple columns based on the given list of indices.

        Args:
            data (pd.DataFrame): The input data to be processed.
            columns (list): List of column indices to select.

        Returns:
            list: List of selected columns.
        """
        return [data.columns[col] for col in columns]
