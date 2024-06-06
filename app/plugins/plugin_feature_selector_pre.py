import pandas as pd
import numpy as np
import json
import os
from statsmodels.tsa.stattools import grangercausalitytests

class Plugin:
    # Define the parameters for this plugin and their default values
    plugin_params = {
        'method': 'select_single',
        'max_lag': 5,
        'significance_level': 0.05,
        'single': 1,
        'multi': None
    }

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.feature_selection_params = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        # Provide plugin-specific debug information
        return {
            'method': self.params.get('method'),
            'max_lag': self.params.get('max_lag'),
            'significance_level': self.params.get('significance_level'),
            'single': self.params.get('single'),
            'multi': self.params.get('multi')
        }

    def process(self, data):
        method = self.params.get('method', 'select_single')
        max_lag = self.params.get('max_lag', 5)
        significance_level = self.params.get('significance_level', 0.05)
        single = self.params.get('single', 1)
        multi = self.params.get('multi')

        print("Starting the process method.")
        print(f"Method: {method}, Max Lag: {max_lag}, Significance Level: {significance_level}, Single: {single}, Multi: {multi}")

        if method == 'select_single':
            selected_features = [data.columns[single]]
        elif method == 'select_multi':
            if multi is None:
                multi = [0]
            selected_features = [data.columns[i] for i in multi]
        else:
            if self.feature_selection_params is None:
                if method == 'acf':
                    selected_features = self._acf_feature_selection(data, significance_level)
                elif method == 'pacf':
                    selected_features = self._pacf_feature_selection(data, significance_level)
                elif method == 'granger':
                    selected_features = self._granger_causality_feature_selection(data, max_lag, significance_level)
                else:
                    raise ValueError(f"Unknown feature selection method: {method}")

                self.feature_selection_params = {'method': method, 'selected_features': selected_features}
            else:
                selected_features = self.feature_selection_params['selected_features']

        return data[selected_features]

    def _acf_feature_selection(self, data, significance_level):
        selected_features = []
        for column in data.columns:
            acf_values = [abs(val) for val in np.correlate(data[column], data[column], mode='full')]
            if any(val > significance_level for val in acf_values):
                selected_features.append(column)
        return selected_features

    def _pacf_feature_selection(self, data, significance_level):
        selected_features = []
        for column in data.columns:
            pacf_values = [abs(val) for val in np.correlate(data[column], data[column], mode='full')]
            if any(val > significance_level for val in pacf_values):
                selected_features.append(column)
        return selected_features

    def _granger_causality_feature_selection(self, data, max_lag, significance_level):
        selected_features = []
        target_column = 'eur_usd_rate'
        for column in data.columns:
            if column != target_column:
                test_result = grangercausalitytests(data[[target_column, column]], max_lag, verbose=False)
                p_values = [test_result[i+1][0]['ssr_chi2test'][1] for i in range(max_lag)]
                if any(p_val < significance_level for p_val in p_values):
                    selected_features.append(column)
        return selected_features
