# Inside plugin.py

class Plugin:
    def __init__(self):
        self.feature_selection_params = None

    def process(self, data, method='granger', save_params=None, load_params=None, max_lag=5, significance_level=0.05, single=None, multi=None, force_date=True):
        # Ensure that the force_date parameter is passed to the relevant methods

        if method == 'select_single':
            if single >= len(data.columns):
                raise ValueError(f"Column index '{single}' is out of range.")
            selected_features = [data.columns[single]]
            if not force_date:
                selected_features = [col for col in selected_features if col != 'date']  # Exclude date column if force_date is False

        elif method == 'select_multi':
            selected_features = [str(col) for col in multi]
            if not force_date:
                selected_features = [col for col in selected_features if col != 'date']  # Exclude date column if force_date is False

        else:
            if load_params and os.path.exists(load_params):
                with open(load_params, 'r') as f:
                    self.feature_selection_params = json.load(f)
                selected_features = self.feature_selection_params.get('selected_features', data.columns.tolist())

            if self.feature_selection_params is None:
                if method == 'acf':
                    selected_features = self._acf_feature_selection(data, significance_level, force_date)
                elif method == 'pacf':
                    selected_features = self._pacf_feature_selection(data, significance_level, force_date)
                elif method == 'granger':
                    selected_features = self._granger_causality_feature_selection(data, max_lag, significance_level, force_date)
                else:
                    raise ValueError(f"Unknown feature selection method: {method}")

                self.feature_selection_params = {'method': method, 'selected_features': selected_features}
                if save_params:
                    with open(save_params, 'w') as f:
                        json.dump(self.feature_selection_params, f)
            else:
                selected_features = self.feature_selection_params['selected_features']

        return data[selected_features]
