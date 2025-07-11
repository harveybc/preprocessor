"""
Data Processor Component

This module orchestrates the preprocessing pipeline, including external feature engineering
integration, dataset splitting, and normalization.

Key Features:
- External feature engineering plugin integration
- Correct normalization logic (fit on training sets D1, D4 only)
- Dataset splitting for autoencoder and predictor training
- Comprehensive debugging and error handling
- BDD-compliant architecture

Author: Preprocessor System
Date: 2025-07-11
"""

import numpy as np
import pandas as pd
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
from typing import Dict, Any, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message="p-value may not be accurate for N > 5000.")


def run_preprocessor_pipeline(config: Dict[str, Any], plugin) -> pd.DataFrame:
    """
    Execute the complete preprocessing pipeline with external feature engineering support.
    
    This function orchestrates:
    1. Data loading and validation
    2. External feature engineering (if enabled)
    3. Dataset splitting into 6 sets (D1-D6)
    4. Correct normalization (fit on D1, D4; apply to all)
    5. Data saving and validation
    
    Args:
        config (Dict[str, Any]): Configuration parameters
        plugin: The preprocessing plugin instance
    
    Returns:
        pd.DataFrame: Summary of processed datasets
        
    Raises:
        ValueError: If data validation fails
        FileNotFoundError: If input file doesn't exist
        Exception: For other processing errors
    """
    
    try:
        # 1. Load and validate input data
        print(f"[INFO] Loading data from {config['input_file']}...")
        data = load_csv(config['input_file'])
        
        if data is None or data.empty:
            raise ValueError("Input data is empty or could not be loaded")
            
        print(f"[DEBUG] Loaded data shape: {data.shape}")
        print(f"[DEBUG] Columns: {list(data.columns)}")
        
        # 2. Execute preprocessing with plugin
        print("[INFO] Starting preprocessing pipeline...")
        processed_data = plugin.process(data, config)
        
        if processed_data is None or processed_data.empty:
            raise ValueError("Plugin processing returned empty data")
            
        print(f"[DEBUG] Processed data shape: {processed_data.shape}")
        
        # 3. Save debug information if enabled
        if config.get('save_log'):
            debug_info = plugin.get_debug_info()
            save_debug_info(debug_info, config['save_log'])
            print(f"[INFO] Debug information saved to {config['save_log']}")
        
        # 4. Remote logging if configured
        if config.get('remote_log'):
            try:
                debug_info = plugin.get_debug_info()
                remote_log(debug_info, config['remote_log'], 
                          config.get('remote_username'), 
                          config.get('remote_password'))
                print("[INFO] Debug information logged remotely")
            except Exception as e:
                print(f"[WARNING] Remote logging failed: {e}")
        
        # 5. Validate output consistency
        if isinstance(processed_data, pd.DataFrame):
            _validate_output_consistency(processed_data, config)
        
        print("[INFO] Preprocessing pipeline completed successfully")
        return processed_data
        
    except Exception as e:
        print(f"[ERROR] Preprocessing pipeline failed: {e}")
        raise


def _validate_output_consistency(processed_data: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Validate the consistency of processed output data.
    
    Args:
        processed_data (pd.DataFrame): The processed data summary
        config (Dict[str, Any]): Configuration parameters
        
    Raises:
        ValueError: If validation fails
    """
    
    if processed_data.empty:
        raise ValueError("Processed data is empty")
    
    # Check for required columns in summary
    required_columns = ['Filename', 'Rows', 'Columns']
    missing_columns = [col for col in required_columns if col not in processed_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in summary: {missing_columns}")
    
    # Validate split proportions sum to 1.0
    proportions = [
        config.get('d1_proportion', 0.33),
        config.get('d2_proportion', 0.083),
        config.get('d3_proportion', 0.083),
        config.get('d4_proportion', 0.33),
        config.get('d5_proportion', 0.083),
        config.get('d6_proportion', 0.083)
    ]
    
    total_proportion = sum(proportions)
    if abs(total_proportion - 1.0) > 0.001:  # Allow small floating point errors
        print(f"[WARNING] Dataset proportions sum to {total_proportion:.6f}, not 1.0")
    
    print("[DEBUG] Output validation passed")


def validate_external_plugin_integration(config: Dict[str, Any]) -> bool:
    """
    Validate external plugin integration configuration.
    
    Args:
        config (Dict[str, Any]): Configuration parameters
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    
    if not config.get('use_external_feature_eng', False):
        return True
    
    # Check feature-eng plugin path
    feature_eng_path = config.get('feature_eng_plugin_path')
    if not feature_eng_path:
        print("[WARNING] External feature engineering enabled but no plugin path specified")
        return False
    
    # Validate technical indicators configuration
    if config.get('technical_indicators', False):
        required_params = ['short_window', 'medium_window', 'long_window']
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            print(f"[WARNING] Technical indicators enabled but missing parameters: {missing_params}")
            return False
    
    # Validate decomposition configuration
    if config.get('decomposition_enabled', False):
        if not config.get('decomp_features'):
            print("[WARNING] Decomposition enabled but no features specified")
            return False
    
    print("[DEBUG] External plugin integration validation passed")
    return True


def get_pipeline_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate metadata about the preprocessing pipeline configuration.
    
    Args:
        config (Dict[str, Any]): Configuration parameters
        
    Returns:
        Dict[str, Any]: Pipeline metadata
    """
    
    metadata = {
        'pipeline_type': 'preprocessor',
        'external_feature_eng': config.get('use_external_feature_eng', False),
        'technical_indicators': config.get('technical_indicators', False),
        'decomposition_enabled': config.get('decomposition_enabled', False),
        'normalization_method': config.get('normalization_method', 'min_max'),
        'fit_on_training_only': config.get('fit_on_training_only', True),
        'dataset_splits': {
            'd1_proportion': config.get('d1_proportion', 0.33),
            'd2_proportion': config.get('d2_proportion', 0.083),
            'd3_proportion': config.get('d3_proportion', 0.083),
            'd4_proportion': config.get('d4_proportion', 0.33),
            'd5_proportion': config.get('d5_proportion', 0.083),
            'd6_proportion': config.get('d6_proportion', 0.083)
        }
    }
    
    return metadata
