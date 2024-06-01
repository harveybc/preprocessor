import pandas as pd
import numpy as np
import json
import os
from statsmodels.tsa.stattools import grangercausalitytests

class Plugin:
    def __init__(self):
        self.feature_selection_params = None

    def process(self, data, method='granger', save_params=None, load_params=None, max_lag=5, significance_level=0.05, single=None, multi=None):
        """
        Perform feature selection on the dataset using the specified method.
        """
        if load_params and os.path.exists(load_params):
            with open(load_params, 'r') as f:
                self.feature_selection_params = json.load(f)
            selected_features = self.feature_selection_params.get('selected_features', data.columns.tolist())
        else:
            selected_features = data.columns.tolist()

        if method == 'select_single':
            selected_features = [single]
        elif method == 'select_multi':
            selected_features = multi
        elif method == 'acf':
            selected_features = self._acf_feature_selection(data, significance_level)
        elif method == 'pacf':
            selected_features = self._pacf_feature_selection(data, significance_level)
        elif method == 'granger':
            selected_features = self._granger_causality_feature_selection(data, max_lag, significance_level)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        self.feature_selection_params = {'method': method, 'selected_features': selected_features}
        if save_params:
            with open(save_params, 'w') as f:
                json.dump(self.feature_selection_params, f)

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
