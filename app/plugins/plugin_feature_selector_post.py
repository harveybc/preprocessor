import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
class Plugin:
    def __init__(self):
        # Initialize the feature selection parameters to None
        self.feature_selection_params = None

    def process(self, data, target, method='lasso', save_params=None, load_params=None, **kwargs):
        """
        Perform feature selection on the dataset using the specified method.

        Args:
            data (pd.DataFrame): The input data to be processed.
            target (pd.Series): The target variable.
            method (str): The method for feature selection ('lasso', 'elastic_net', 'mutual_info', 'cross_val', 'boruta').
            save_params (str): Path to save the feature selection parameters.
            load_params (str): Path to load the feature selection parameters.

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
            if method == 'lasso':
                selected_features = self._lasso_feature_selection(data, target, kwargs.get('alpha', 1.0))
            elif method == 'elastic_net':
                selected_features = self._elastic_net_feature_selection(data, target, kwargs.get('alpha', 1.0), kwargs.get('l1_ratio', 0.5))
            elif method == 'mutual_info':
                selected_features = self._mutual_info_feature_selection(data, target)
            elif method == 'boruta':
                selected_features = self._boruta_feature_selection(data, target)
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

    def _lasso_feature_selection(self, data, target, alpha):
        """
        Select features using LASSO regularization.

        Args:
            data (pd.DataFrame): The input data to be processed.
            target (pd.Series): The target variable.
            alpha (float): Regularization strength for LASSO.

        Returns:
            list: List of selected features.
        """
        lasso = LassoCV(alphas=[alpha], cv=5).fit(data, target)
        importance = np.abs(lasso.coef_)
        selected_features = data.columns[importance > 0].tolist()
        return selected_features

    def _elastic_net_feature_selection(self, data, target, alpha, l1_ratio):
        """
        Select features using Elastic Net regularization.

        Args:
            data (pd.DataFrame): The input data to be processed.
            target (pd.Series): The target variable.
            alpha (float): Regularization strength for Elastic Net.
            l1_ratio (float): The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.

        Returns:
            list: List of selected features.
        """
        elastic_net = ElasticNetCV(alphas=[alpha], l1_ratio=l1_ratio, cv=5).fit(data, target)
        importance = np.abs(elastic_net.coef_)
        selected_features = data.columns[importance > 0].tolist()
        return selected_features

    def _mutual_info_feature_selection(self, data, target):
        """
        Select features using Mutual Information.

        Args:
            data (pd.DataFrame): The input data to be processed.
            target (pd.Series): The target variable.

        Returns:
            list: List of selected features.
        """
        mutual_info = mutual_info_regression(data, target)
        selected_features = data.columns[np.argsort(mutual_info)[-10:]].tolist()  # Select top 10 features
        return selected_features

    
    def _boruta_feature_selection(self, data, target):
        """
        Select features using the Boruta algorithm.

        Args:
            data (pd.DataFrame): The input data to be processed.
            target (pd.Series): The target variable.

        Returns:
            list: List of selected features.
        """
        rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
        boruta = BorutaPy(rf, n_estimators='auto', random_state=1)
        boruta.fit(data.values, target.values)

        selected_features = data.columns[boruta.support_].tolist()
        return selected_features
