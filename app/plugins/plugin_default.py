import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import json
import sys
import importlib.util
from pathlib import Path

class Plugin:
    """
    Plugin to preprocess the dataset for feature extraction with external feature engineering support.
    
    This plugin performs:
    1. Optional external feature engineering (technical indicators, decomposition)
    2. Trimming of initial rows to remove values affected by decomposition windows
    3. Dataset splitting into 6 sets (D1-D6) for autoencoder and predictor training
    4. Z-score normalization with separate normalizers:
       - Normalizer A: fit on D1, apply to D1, D2, D3 → save to normalization_config_a.json
       - Normalizer B: fit on D4, apply to D4, D5, D6 → save to normalization_config_b.json
    """
    
    # Define the parameters for this plugin and their default values
    plugin_params = {
        'input_column_order': ["d", "o", "h", "l", "c"],
        'output_column_order': ["d", "o", "l", "h", "c"],
        'dataset_prefix': "base_",
        'target_prefix': "normalized_",
        'trim_start_rows': 168,  # Trim start rows to remove initial values affected by largest decomposition window
        'target_column': 4,  # Index in output_column_order (zero-based)
        'pip_value': 0.00001,
        'range': (0, 1),
        'd1_proportion': 0.33,  # Autoencoder training
        'd2_proportion': 0.083, # Autoencoder validation  
        'd3_proportion': 0.083, # Autoencoder test
        'd4_proportion': 0.33,  # Predictor training
        'd5_proportion': 0.083, # Predictor validation
        'd6_proportion': 0.083, # Predictor test
        'only_low_CV': True,
        
        # Market margin filtering
        'market_close_margin_ticks': 2,  # Number of ticks to remove before/after market gaps
        
        # Cyclic sinusoidal encoding (computed from DATE_TIME)
        'use_cyclic_encoding': True,
        
        # Rolling features
        'use_rolling_features': False,
        'rolling_window': 24,
        'rolling_price_column': 'typical_price',
        
        # External feature engineering
        'use_external_feature_eng': False,
        'feature_eng_plugin_path': '/home/harveybc/Documents/GitHub/feature-eng/app/plugins',
        
        # Technical indicators
        'technical_indicators': False,
        'short_window': 14,
        'medium_window': 50,
        'long_window': 200,
        'indicators': ['rsi', 'macd', 'ema', 'sma', 'bollinger_bands'],
        
        # Decomposition
        'decomposition_enabled': False,
        'decomp_features': [],
        'decomp_methods': {},
        'stl_period': 12,
        'stl_robust': True,
        'wavelet_name': 'db4',
        'wavelet_levels': 3,
        'mtm_bandwidth': 2.5,
        'mtm_n_tapers': 4,
        
        # Normalization
        'normalization_method': 'z_score',
        'fit_on_training_only': True
    }

    # Define the debug variables for this plugin
    plugin_debug_vars = ['column_metrics', 'normalization_params']

    def __init__(self):
        """Initialize the Plugin with default parameters."""
        self.params = self.plugin_params.copy()
        self.normalization_params = {}  # To store normalization parameters for each column
        self.column_metrics = {}  # To store metrics for each column

    def set_params(self, **kwargs):
        """Set the parameters for the plugin."""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        """Get debug information for the plugin."""
        debug_info = {
            'column_metrics': self.column_metrics,
            'normalization_params': self.normalization_params
        }
        return debug_info

    def add_debug_info(self, debug_info):
        """Add debug information to the given dictionary."""
        debug_info.update(self.get_debug_info())

    def _load_external_plugin(self, plugin_path: str, plugin_name: str):
        """Load an external feature engineering plugin."""
        plugin_file = Path(plugin_path) / f"{plugin_name}.py"
        
        if not plugin_file.exists():
            print(f"[WARNING] Plugin file not found: {plugin_file}")
            return None
            
        try:
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
            if spec is None or spec.loader is None:
                print(f"[WARNING] Cannot load external plugin: {plugin_file}")
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get Plugin class and instantiate
            if hasattr(module, 'Plugin'):
                plugin_class = getattr(module, 'Plugin')
                return plugin_class()
            else:
                print(f"[WARNING] No Plugin class found in {plugin_file}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Failed to load external plugin {plugin_name}: {e}")
            return None

    def _apply_feature_engineering(self, data: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Apply external feature engineering plugins if enabled."""
        processed_data = data.copy()
        
        if not self.params.get('use_external_feature_eng', False):
            return processed_data
            
        print("[DEBUG] Applying external feature engineering...")
        
        # Load technical indicators plugin
        if self.params.get('technical_indicators', False):
            tech_plugin = self._load_external_plugin(
                self.params['feature_eng_plugin_path'], 
                'tech_indicator'
            )
            
            if tech_plugin:
                try:
                    # Configure technical indicators
                    tech_config = {
                        'short_window': self.params.get('short_window', 14),
                        'medium_window': self.params.get('medium_window', 50),
                        'long_window': self.params.get('long_window', 200),
                        'indicators': self.params.get('indicators', ['rsi', 'macd'])
                    }
                    
                    tech_plugin.set_params(**tech_config)
                    processed_data = tech_plugin.process(processed_data, tech_config)
                    print(f"[DEBUG] Technical indicators applied. New shape: {processed_data.shape}")
                    
                except Exception as e:
                    print(f"[ERROR] Technical indicators failed: {e}")
        
        # Load decomposition post-processor plugin
        if self.params.get('decomposition_enabled', False) and self.params.get('decomp_features'):
            decomp_plugin = self._load_external_plugin(
                self.params['feature_eng_plugin_path'] + '/post_processors',
                'decomposition_post_processor'
            )
            
            if decomp_plugin:
                try:
                    # Configure decomposition
                    decomp_config = {
                        'decomp_features': self.params.get('decomp_features', []),
                        'decomp_methods': self.params.get('decomp_methods', {}),
                        'stl_period': self.params.get('stl_period', 12),
                        'stl_robust': self.params.get('stl_robust', True),
                        'wavelet_name': self.params.get('wavelet_name', 'db4'),
                        'wavelet_levels': self.params.get('wavelet_levels', 3),
                        'mtm_bandwidth': self.params.get('mtm_bandwidth', 2.5),
                        'mtm_n_tapers': self.params.get('mtm_n_tapers', 4)
                    }
                    
                    decomp_plugin.set_params(**decomp_config)
                    processed_data = decomp_plugin.process(processed_data, decomp_config)
                    print(f"[DEBUG] Decomposition applied. New shape: {processed_data.shape}")
                    
                except Exception as e:
                    print(f"[ERROR] Decomposition failed: {e}")
        
        return processed_data

    def process(self, data, config):
        """
        Process the data with optional feature engineering, proper dataset splitting,
        and correct normalization using training sets (D1, D4) to fit normalizers
        and applying them to validation/test sets (D2, D3, D5, D6).
        
        Args:
            data (pd.DataFrame): The input data to be processed.
            config (dict): Configuration parameters.
        
        Returns:
            pd.DataFrame: The summary of processed datasets.
        """
        # 1.0: Debug: Display loaded data details.
        print(f"[DEBUG] Loaded data shape: {data.shape}")
        print(f"[DEBUG] Columns in the data: {list(data.columns)}")

        # 1.1: Ensure DATE_TIME column is included as a regular column.
        if isinstance(data.index, pd.DatetimeIndex):
            print("[DEBUG] DATE_TIME is currently the index. Resetting it to a regular column...")
            data.reset_index(inplace=True)
        if 'DATE_TIME' not in data.columns:
            raise ValueError("[ERROR] DATE_TIME column is missing in the input data!")

        # 2.0: Apply external feature engineering if enabled
        processed_data = self._apply_feature_engineering(data, config)
        print(f"[DEBUG] After feature engineering shape: {processed_data.shape}")

        # 2.0.5: Generate derived features (cyclic sinusoidal encoding + rolling features)
        # Cyclic sinusoidal encoding computed directly from DATE_TIME timestamps
        if self.params.get('use_cyclic_encoding', False):
            dt_enc = pd.to_datetime(processed_data['DATE_TIME'])
            processed_data['hod_sin'] = np.sin(2 * np.pi * dt_enc.dt.hour / 24)
            processed_data['hod_cos'] = np.cos(2 * np.pi * dt_enc.dt.hour / 24)
            processed_data['dow_sin'] = np.sin(2 * np.pi * dt_enc.dt.dayofweek / 7)
            processed_data['dow_cos'] = np.cos(2 * np.pi * dt_enc.dt.dayofweek / 7)
            processed_data['dom_sin'] = np.sin(2 * np.pi * (dt_enc.dt.day - 1) / 31)
            processed_data['dom_cos'] = np.cos(2 * np.pi * (dt_enc.dt.day - 1) / 31)
            processed_data['moy_sin'] = np.sin(2 * np.pi * (dt_enc.dt.month - 1) / 12)
            processed_data['moy_cos'] = np.cos(2 * np.pi * (dt_enc.dt.month - 1) / 12)
            print(f"[DEBUG] Generated cyclic sinusoidal features from DATE_TIME. Columns: hod_sin/cos, dow_sin/cos, dom_sin/cos, moy_sin/cos")

        # Rolling features from price column (computed before trim so NaN rows get trimmed)
        if self.params.get('use_rolling_features', False):
            price_col = self.params.get('rolling_price_column', 'typical_price')
            window = self.params.get('rolling_window', 24)
            if price_col in processed_data.columns:
                processed_data[f'rolling_std_{window}'] = processed_data[price_col].rolling(window=window).std()
                processed_data[f'rolling_ema_{window}'] = processed_data[price_col].ewm(span=window, adjust=False).mean()
                processed_data['price_minus_ema'] = processed_data[price_col] - processed_data[f'rolling_ema_{window}']
                print(f"[DEBUG] Generated rolling features (window={window}) from '{price_col}': rolling_std_{window}, rolling_ema_{window}, price_minus_ema")
            else:
                print(f"[WARNING] Rolling price column '{price_col}' not found in data. Skipping rolling features.")

        # 2.1: Trim starting rows to remove initial values affected by decomposition windows
        trim_rows = self.params.get('trim_start_rows', 0)
        if trim_rows > 0:
            if len(processed_data) > trim_rows:
                processed_data = processed_data.iloc[trim_rows:].copy()
                print(f"[DEBUG] Trimmed {trim_rows} starting rows. New shape: {processed_data.shape}")
            else:
                print(f"[WARNING] Cannot trim {trim_rows} rows from dataset with only {len(processed_data)} rows")

        # 2.2: Remove ticks near market gaps (weekends, holidays, etc.) to avoid volatility from discontinuities
        market_margin = self.params.get('market_close_margin_ticks', 0)
        if market_margin > 0:
            dt_col = pd.to_datetime(processed_data['DATE_TIME'])
            rows_to_drop = set()
            
            # Detect gaps by looking at consecutive date differences
            # A "gap" is any place where the time jump between consecutive ticks
            # is larger than the normal tick interval (detected as the mode of diffs)
            time_diffs = dt_col.diff()
            normal_interval = time_diffs.mode()[0]
            
            # Find indices where a gap occurs (time diff > normal interval)
            gap_mask = time_diffs > normal_interval
            gap_indices = processed_data.index[gap_mask].tolist()
            
            print(f"[DEBUG] Detected {len(gap_indices)} date gaps (normal interval: {normal_interval})")
            
            for gap_idx in gap_indices:
                # gap_idx is the first tick AFTER the gap — remove first N ticks after gap
                pos = processed_data.index.get_loc(gap_idx)
                after_indices = processed_data.index[pos:pos + market_margin]
                rows_to_drop.update(after_indices)
                
                # Remove last N ticks BEFORE the gap
                before_start = max(0, pos - market_margin)
                before_indices = processed_data.index[before_start:pos]
                rows_to_drop.update(before_indices)
            
            if rows_to_drop:
                processed_data = processed_data.drop(index=rows_to_drop).reset_index(drop=True)
                print(f"[DEBUG] Removed {len(rows_to_drop)} ticks near market gaps (margin={market_margin}). New shape: {processed_data.shape}")
            else:
                print(f"[DEBUG] Market margin filtering: no ticks to remove")

        # 2.3: Reorder columns based on output order.
        output_column_order = ['DATE_TIME', 'OPEN', 'LOW', 'HIGH', 'CLOSE']
        
        # Update column order to include any new features from feature engineering
        available_columns = list(processed_data.columns)
        base_columns = [col for col in output_column_order if col in available_columns]
        feature_columns = [col for col in available_columns if col not in output_column_order]
        final_column_order = base_columns + feature_columns
        
        print(f"[DEBUG] Final column order: {final_column_order}")
        base_data = processed_data[final_column_order]
        print(f"[DEBUG] Final data shape: {base_data.shape}")

        # 3.0: Split data into D1, D2, D3, D4, D5, and D6.
        total_len = len(base_data)
        d1_size = int(total_len * self.params['d1_proportion'])
        d2_size = int(total_len * self.params['d2_proportion'])
        d3_size = int(total_len * self.params['d3_proportion'])
        d4_size = int(total_len * self.params['d4_proportion'])
        d5_size = int(total_len * self.params['d5_proportion'])
        d6_size = total_len - (d1_size + d2_size + d3_size + d4_size + d5_size)

        # Split the datasets
        d1_data = base_data.iloc[:d1_size].copy()
        d2_data = base_data.iloc[d1_size:d1_size + d2_size].copy()
        d3_data = base_data.iloc[d1_size + d2_size:d1_size + d2_size + d3_size].copy()
        d4_data = base_data.iloc[d1_size + d2_size + d3_size:d1_size + d2_size + d3_size + d4_size].copy()
        d5_data = base_data.iloc[d1_size + d2_size + d3_size + d4_size:d1_size + d2_size + d3_size + d4_size + d5_size].copy()
        d6_data = base_data.iloc[d1_size + d2_size + d3_size + d4_size + d5_size:].copy()

        print(f"[DEBUG] Dataset splits - D1: {d1_size}, D2: {d2_size}, D3: {d3_size}, D4: {d4_size}, D5: {d5_size}, D6: {d6_size}")

        # -- NEW: Feature filtering exactly before saving outputs --
        # Build selected feature list from config without altering existing processing flow.
        # This affects only what is written to disk, not normalization logic or metrics.
        def _build_selected_columns(cfg: dict, available_cols: list) -> list:
            # Accept both 'features_included' and legacy 'features included'
            groups = cfg.get('features_included')
            if groups is None:
                groups = cfg.get('features included')
            if not groups:
                return None  # No filtering requested

            base_feats = [
                'OPEN','HIGH','LOW','CLOSE','BC-BO','BH-BL','BH-BO','BO-BL',
                'typical_price'
            ]
            if cfg.get('use_typical_sd', False):
                base_feats.append('typical_sd')

            feature_groups = {
                'base_features': base_feats,
                'typical_price_only': ['typical_price'],
                'cyclic_features': [
                    'hod_sin','hod_cos','dow_sin','dow_cos',
                    'dom_sin','dom_cos','moy_sin','moy_cos'
                ],
                'rolling_features': [c for c in available_cols
                                     if c.startswith('rolling_') or c == 'price_minus_ema'],
                'technical_features': [
                    'RSI','MACD','MACD_Signal','MACD_Histogram','EMA','Stochastic_%K','Stochastic_%D',
                    'ADX','DI+','DI-','ATR','CCI','BB_MID_20_2','BB_UP_20_2','BB_LOW_20_2','BB_WIDTH_20_2',
                    'WilliamsR','Momentum','ROC'
                ],
                'fundamental_features': [
                    'S&P500_Close','vix_close'
                ],
                'seasonal_features': [
                    'day_of_week','day_of_month','hour_of_day','dow_sin','dow_cos','dom_sin','dom_cos','hod_sin','hod_cos',
                    'moy_sin','moy_cos'
                ],
                'high_frequency_features': [
                    'CLOSE_15m_tick_1','CLOSE_15m_tick_2','CLOSE_15m_tick_3','CLOSE_15m_tick_4',
                    'CLOSE_15m_tick_5','CLOSE_15m_tick_6','CLOSE_15m_tick_7','CLOSE_15m_tick_8',
                    'CLOSE_30m_tick_1','CLOSE_30m_tick_2','CLOSE_30m_tick_3','CLOSE_30m_tick_4',
                    'CLOSE_30m_tick_5','CLOSE_30m_tick_6','CLOSE_30m_tick_7','CLOSE_30m_tick_8'
                ],
            }

            ordered = []
            # Always ensure DATE_TIME is first if present
            if 'DATE_TIME' in available_cols:
                ordered.append('DATE_TIME')

            # Add groups in the provided order; dedupe while preserving order
            seen = set(ordered)
            for grp in groups:
                cols = feature_groups.get(grp, [])
                for c in cols:
                    if c in available_cols and c not in seen:
                        ordered.append(c)
                        seen.add(c)

            # If nothing matched besides DATE_TIME, fall back to no filtering
            return ordered if len(ordered) > (1 if 'DATE_TIME' in available_cols else 0) else None

        # Compute selected columns list against the current columns of each dataset
        # We base it on base_data columns to keep consistency across splits.
        selected_columns = _build_selected_columns(config, list(base_data.columns))

        # Prepare filtered views for saving base datasets without mutating originals
        if selected_columns is not None:
            def _filtered(df: pd.DataFrame) -> pd.DataFrame:
                present = [c for c in selected_columns if c in df.columns]
                return df.loc[:, present]

            d1_save = _filtered(d1_data)
            d2_save = _filtered(d2_data)
            d3_save = _filtered(d3_data)
            d4_save = _filtered(d4_data)
            d5_save = _filtered(d5_data)
            d6_save = _filtered(d6_data)
        else:
            # No filtering requested; use datasets as-is for saving
            d1_save, d2_save, d3_save, d4_save, d5_save, d6_save = d1_data, d2_data, d3_data, d4_data, d5_data, d6_data

        # 4.0: Save the base datasets (with headers).
        dataset_prefix = self.params['dataset_prefix']
        d1_save.to_csv(f"{dataset_prefix}d1.csv", index=False, header=True)
        d2_save.to_csv(f"{dataset_prefix}d2.csv", index=False, header=True)
        d3_save.to_csv(f"{dataset_prefix}d3.csv", index=False, header=True)
        d4_save.to_csv(f"{dataset_prefix}d4.csv", index=False, header=True)
        d5_save.to_csv(f"{dataset_prefix}d5.csv", index=False, header=True)
        d6_save.to_csv(f"{dataset_prefix}d6.csv", index=False, header=True)
        print(f"[DEBUG] Saved base datasets with headers")

        # 5.0: Z-SCORE NORMALIZATION WITH SEPARATE NORMALIZERS FOR A AND B GROUPS
        # Identify numeric columns for normalization
        numeric_columns = base_data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"[DEBUG] Numeric columns for normalization: {numeric_columns}")

        # 5.1: FIT NORMALIZER A on D1 and apply to D2, D3
        print("[DEBUG] Fitting normalizer A on D1...")
        normalization_params_a = {}
        for column in numeric_columns:
            mean_val = d1_data[column].mean()
            std_val = d1_data[column].std()
            
            # Convert numpy scalars to native Python types
            if hasattr(mean_val, "item"):
                mean_val = mean_val.item()
            if hasattr(std_val, "item"):
                std_val = std_val.item()
                
            print(f"[DEBUG] Normalizer A params for '{column}': mean={mean_val}, std={std_val}")
            normalization_params_a[column] = {"mean": mean_val, "std": std_val}

        # 5.2: FIT NORMALIZER B on D4 and apply to D5, D6
        print("[DEBUG] Fitting normalizer B on D4...")
        normalization_params_b = {}
        for column in numeric_columns:
            mean_val = d4_data[column].mean()
            std_val = d4_data[column].std()
            
            # Convert numpy scalars to native Python types
            if hasattr(mean_val, "item"):
                mean_val = mean_val.item()
            if hasattr(std_val, "item"):
                std_val = std_val.item()
                
            print(f"[DEBUG] Normalizer B params for '{column}': mean={mean_val}, std={std_val}")
            normalization_params_b[column] = {"mean": mean_val, "std": std_val}

        # Store normalization params for debug
        self.normalization_params = {
            'normalizer_a': normalization_params_a,
            'normalizer_b': normalization_params_b
        }

        # 5.3: Apply z-score normalization to datasets
        datasets = {
            'd1': d1_data.copy(),
            'd2': d2_data.copy(), 
            'd3': d3_data.copy(),
            'd4': d4_data.copy(),
            'd5': d5_data.copy(),
            'd6': d6_data.copy()
        }
        
        normalized_datasets = {}
        
        # Group A: D1 (fit), D2, D3 (apply) using normalizer A
        for dataset_name in ['d1', 'd2', 'd3']:
            dataset = datasets[dataset_name]
            normalized_dataset = dataset.copy()
            
            print(f"[DEBUG] Applying normalizer A to {dataset_name}")
            for column in numeric_columns:
                mean_val = normalization_params_a[column]["mean"]
                std_val = normalization_params_a[column]["std"]
                
                # Avoid division by zero
                if std_val == 0:
                    print(f"[WARNING] Zero std for column '{column}' in normalizer A, setting to 0")
                    normalized_dataset[column] = 0.0
                else:
                    normalized_dataset[column] = (dataset[column] - mean_val) / std_val
            
            normalized_datasets[dataset_name] = normalized_dataset

        # Group B: D4 (fit), D5, D6 (apply) using normalizer B  
        for dataset_name in ['d4', 'd5', 'd6']:
            dataset = datasets[dataset_name]
            normalized_dataset = dataset.copy()
            
            print(f"[DEBUG] Applying normalizer B to {dataset_name}")
            for column in numeric_columns:
                mean_val = normalization_params_b[column]["mean"]
                std_val = normalization_params_b[column]["std"]
                
                # Avoid division by zero
                if std_val == 0:
                    print(f"[WARNING] Zero std for column '{column}' in normalizer B, setting to 0")
                    normalized_dataset[column] = 0.0
                else:
                    normalized_dataset[column] = (dataset[column] - mean_val) / std_val
            
            normalized_datasets[dataset_name] = normalized_dataset

        # 5.4: Save normalization parameters in separate JSON files
        try:
            # Save normalizer A parameters
            config_file_a = config.get('normalization_config_a', 'normalization_config_a.json')
            
            # Create directory if it doesn't exist
            import os
            if os.path.dirname(config_file_a):
                os.makedirs(os.path.dirname(config_file_a), exist_ok=True)
            
            with open(config_file_a, 'w') as f:
                json.dump(normalization_params_a, f, indent=4)
            print(f"[DEBUG] Normalizer A parameters saved to {config_file_a}")
            
            # Save normalizer B parameters
            config_file_b = config.get('normalization_config_b', 'normalization_config_b.json')
            
            # Create directory if it doesn't exist
            if os.path.dirname(config_file_b):
                os.makedirs(os.path.dirname(config_file_b), exist_ok=True)
            
            with open(config_file_b, 'w') as f:
                json.dump(normalization_params_b, f, indent=4)
            print(f"[DEBUG] Normalizer B parameters saved to {config_file_b}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save normalization parameters to JSON: {e}")
            raise

        # 6.0: Save the normalized datasets (with headers and DATE_TIME).
        target_prefix = self.params['target_prefix']
        for dataset_name, normalized_dataset in normalized_datasets.items():
            filename = f"{target_prefix}{dataset_name}.csv"
            if selected_columns is not None:
                present = [c for c in selected_columns if c in normalized_dataset.columns]
                normalized_dataset.loc[:, present].to_csv(filename, index=False, header=True)
            else:
                normalized_dataset.to_csv(filename, index=False, header=True)
            print(f"[DEBUG] Saved {filename}")

        # 7.0: Return summary of processed files.
        summary_data = {
            'Filename': [
                f"{dataset_prefix}d1.csv", f"{dataset_prefix}d2.csv", f"{dataset_prefix}d3.csv", 
                f"{dataset_prefix}d4.csv", f"{dataset_prefix}d5.csv", f"{dataset_prefix}d6.csv",
                f"{target_prefix}d1.csv", f"{target_prefix}d2.csv", f"{target_prefix}d3.csv", 
                f"{target_prefix}d4.csv", f"{target_prefix}d5.csv", f"{target_prefix}d6.csv"
            ],
            'Rows': [
                d1_data.shape[0], d2_data.shape[0], d3_data.shape[0], 
                d4_data.shape[0], d5_data.shape[0], d6_data.shape[0],
                normalized_datasets['d1'].shape[0], normalized_datasets['d2'].shape[0], normalized_datasets['d3'].shape[0],
                normalized_datasets['d4'].shape[0], normalized_datasets['d5'].shape[0], normalized_datasets['d6'].shape[0]
            ],
            'Columns': [
                d1_data.shape[1], d2_data.shape[1], d3_data.shape[1], 
                d4_data.shape[1], d5_data.shape[1], d6_data.shape[1],
                normalized_datasets['d1'].shape[1], normalized_datasets['d2'].shape[1], normalized_datasets['d3'].shape[1],
                normalized_datasets['d4'].shape[1], normalized_datasets['d5'].shape[1], normalized_datasets['d6'].shape[1]
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        print("[DEBUG] Processing complete. Summary of saved files:")
        print(summary_df)
        
        return summary_df


# Example usage
if __name__ == "__main__":
    plugin = Plugin()
    data = pd.read_csv('tests/data/EURUSD_5m_2010_2015.csv', header=None)
    print(f"Loaded data shape: {data.shape}")
    config = {'debug_file': 'debug_out.json'}
    processed_data = plugin.process(data, config)
    print(processed_data)
