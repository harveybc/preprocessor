import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import json
import sys
import os
import importlib.util
from pathlib import Path

class Plugin:
    """
    Default preprocessing plugin for dataset normalization and splitting.
    
    This plugin:
    1. Optionally applies feature engineering from external plugins
    2. Splits data into 6 datasets (D1-D6) for training autoencoder and predictor
    3. Normalizes data using training sets (D1, D4) parameters
    4. Applies fitted normalizers to validation/test sets (D2, D3, D5, D6)
    """
    # Define the parameters for this plugin and their default values
    plugin_params = {
        'input_column_order': ["d", "o", "h", "l", "c"],
        'output_column_order': ["d", "o", "l", "h", "c"],
        'dataset_prefix': "base_",
        'target_prefix': "normalized_",
        'target_column': 4,  # Index in output_column_order (zero-based)
        'pip_value': 0.00001,
        'range': (0, 1),
        
        # Dataset split proportions
        'd1_proportion': 0.33,   # Training set for autoencoder
        'd2_proportion': 0.083,  # Validation set for autoencoder 
        'd3_proportion': 0.083,  # Test set for autoencoder
        'd4_proportion': 0.33,   # Training set for predictor
        'd5_proportion': 0.083,  # Validation set for predictor
        'd6_proportion': 0.083,  # Test set for predictor
        
        # External feature engineering options
        'use_external_feature_eng': False,
        'feature_eng_plugin_path': '/home/harveybc/Documents/GitHub/feature-eng/app/plugins',
        'technical_indicators': True,
        'decomposition_enabled': False,
        'decomp_features': [],
        'decomp_methods': {},
        
        # Technical indicator configuration
        'short_window': 14,
        'medium_window': 50, 
        'long_window': 200,
        'indicators': ['rsi', 'macd', 'ema', 'sma', 'bollinger_bands'],
        
        # Decomposition configuration
        'stl_period': 12,
        'stl_robust': True,
        'wavelet_name': 'db4',
        'wavelet_levels': 3,
        
        'only_low_CV': True  # Parameter to control processing of low CV columns
    }

    # Define the debug variables for this plugin
    plugin_debug_vars = ['column_metrics', 'normalization_params']

    def __init__(self):
        """
        Initialize the Plugin with default parameters.
        """
        self.params = self.plugin_params.copy()
        self.normalization_params = {}  # To store normalization parameters for each column
        self.column_metrics = {}  # To store metrics for each column

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
            dict: Debug information including column metrics and normalization parameters.
        """
        debug_info = {
            'column_metrics': self.column_metrics,
            'normalization_params': self.normalization_params
        }
        return debug_info

    def add_debug_info(self, debug_info):
        """
        Add debug information to the given dictionary.

        Args:
            debug_info (dict): The dictionary to add debug information to.
        """
        debug_info.update(self.get_debug_info())

    def _load_external_plugin(self, plugin_path: str, plugin_name: str):
        """Load external plugin from specified path."""
        try:
            plugin_file = Path(plugin_path) / f"{plugin_name}.py"
            if not plugin_file.exists():
                print(f"[WARNING] External plugin not found: {plugin_file}")
                return None
                
            # Add plugin path to sys.path temporarily
            if str(plugin_path) not in sys.path:
                sys.path.insert(0, str(plugin_path))
            
            # Import plugin module
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
                        'wavelet_levels': self.params.get('wavelet_levels', 3)
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

        # 2.1: Reorder columns based on output order.
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

        # 4.0: Save the base datasets (with headers).
        dataset_prefix = self.params['dataset_prefix']
        d1_data.to_csv(f"{dataset_prefix}d1.csv", index=False, header=True)
        d2_data.to_csv(f"{dataset_prefix}d2.csv", index=False, header=True)
        d3_data.to_csv(f"{dataset_prefix}d3.csv", index=False, header=True)
        d4_data.to_csv(f"{dataset_prefix}d4.csv", index=False, header=True)
        d5_data.to_csv(f"{dataset_prefix}d5.csv", index=False, header=True)
        d6_data.to_csv(f"{dataset_prefix}d6.csv", index=False, header=True)
        print(f"[DEBUG] Saved base datasets with headers")

        # 5.0: CORRECTED NORMALIZATION LOGIC
        # Identify numeric columns for normalization
        numeric_columns = base_data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"[DEBUG] Numeric columns for normalization: {numeric_columns}")

        # 5.1: Calculate normalization parameters from TRAINING sets (D1 and D4)
        # Combine D1 and D4 to calculate global normalization parameters
        training_data = pd.concat([d1_data[numeric_columns], d4_data[numeric_columns]], ignore_index=True)
        
        normalization_params = {}
        for column in numeric_columns:
            min_val = training_data[column].min()
            max_val = training_data[column].max()
            
            # Convert numpy scalars to native Python types
            if hasattr(min_val, "item"):
                min_val = min_val.item()
            if hasattr(max_val, "item"):
                max_val = max_val.item()
                
            print(f"[DEBUG] Normalization params for '{column}': min={min_val}, max={max_val}")
            normalization_params[column] = {"min": min_val, "max": max_val}

        # 5.2: Apply normalization to ALL datasets using the training-derived parameters
        datasets = {
            'd1': d1_data.copy(),
            'd2': d2_data.copy(), 
            'd3': d3_data.copy(),
            'd4': d4_data.copy(),
            'd5': d5_data.copy(),
            'd6': d6_data.copy()
        }
        
        normalized_datasets = {}
        for dataset_name, dataset in datasets.items():
            normalized_dataset = dataset.copy()
            
            # Apply normalization to numeric columns only
            for column in numeric_columns:
                min_val = normalization_params[column]["min"]
                max_val = normalization_params[column]["max"]
                
                # Avoid division by zero
                range_val = max_val - min_val
                if range_val == 0:
                    print(f"[WARNING] Zero range for column '{column}', setting to 0.5")
                    normalized_dataset[column] = 0.5
                else:
                    normalized_dataset[column] = (dataset[column] - min_val) / range_val
            
            normalized_datasets[dataset_name] = normalized_dataset

        # 5.3: Save normalization parameters
        try:
            debug_file = config.get("debug_file", "debug_out.json")
            with open(debug_file, 'w') as f:
                json.dump(normalization_params, f, indent=4)
            print(f"[DEBUG] Normalization parameters saved to {debug_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save normalization parameters: {e}")
            raise

        # 6.0: Save the normalized datasets
        target_prefix = self.params['target_prefix']
        for dataset_name, normalized_dataset in normalized_datasets.items():
            filename = f"{target_prefix}{dataset_name}.csv"
            normalized_dataset.to_csv(filename, index=False, header=True)
        print(f"[DEBUG] Saved all normalized datasets with prefix '{target_prefix}'")

        # 7.0: Return summary of processed files
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
        print("[DEBUG] Processing complete. Summary:")
        print(summary_df)
        
        # Store normalization parameters for debug info
        self.normalization_params = normalization_params
        
        return summary_df






# Example usage
if __name__ == "__main__":
    plugin = Plugin()
    data = pd.read_csv('tests/data/EURUSD_5m_2010_2015.csv', header=None)
    print(f"Loaded data shape: {data.shape}")
    processed_data = plugin.process(data)
    print(processed_data)
