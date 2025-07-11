"""
Configuration Management for Preprocessor System

This module provides comprehensive configuration handling with
external plugin integration and feature engineering parameters.
"""

import os

# Default configuration values
DEFAULT_VALUES = {
    # Core file paths
    'input_file': 'examples/data/phase_3.csv',
    'output_file': 'output_phase_3.csv',
    'debug_file': 'examples/data/phase_3/phase_3_debug_out.json',
    
    # Configuration management
    'load_config': None,
    'save_config': 'output_config.json',
    'remote_load_config': None,
    'remote_save_config': None,
    'remote_log': None,
    'remote_username': None,
    'remote_password': None,
    
    # Plugin configuration
    'plugin': 'plugin_default',
    'headers': True,
    'force_date': False,
    'quiet_mode': False,
    'only_low_CV': True,
    
    # Dataset generation
    'dataset_prefix': "examples/data/phase_3/base_",
    'target_prefix': "examples/data/phase_3/normalized_",
    
    # Dataset split proportions (for autoencoder and predictor training)
    'd1_proportion': 0.33,   # Training set for autoencoder
    'd2_proportion': 0.083,  # Validation set for autoencoder 
    'd3_proportion': 0.083,  # Test set for autoencoder
    'd4_proportion': 0.33,   # Training set for predictor
    'd5_proportion': 0.083,  # Validation set for predictor
    'd6_proportion': 0.083,  # Test set for predictor
    
    # External feature engineering integration
    'use_external_feature_eng': False,
    'feature_eng_plugin_path': '/home/harveybc/Documents/GitHub/feature-eng/app/plugins',
    'external_plugin_paths': [],
    
    # Technical indicators configuration
    'technical_indicators': False,
    'short_window': 14,
    'medium_window': 50,
    'long_window': 200,
    'indicators': ['rsi', 'macd', 'ema', 'sma', 'bollinger_bands'],
    
    # Decomposition configuration  
    'decomposition_enabled': False,
    'decomp_features': [],
    'decomp_methods': {},
    'stl_period': 12,
    'stl_robust': True,
    'wavelet_name': 'db4',
    'wavelet_levels': 3,
    'mtm_bandwidth': 2.5,
    'mtm_n_tapers': 4,
    
    # Normalization configuration
    'normalization_method': 'min_max',
    'normalization_range': (0, 1),
    'fit_on_training_only': True,  # Use only training sets (D1, D4) to fit normalizer
    
    # Validation and quality control
    'validate_splits': True,
    'min_split_size': 10,
    'data_quality_checks': True,
    
    # Performance and logging
    'performance_monitoring': False,
    'log_level': 'INFO',
    'log_to_file': True,
    'log_to_console': True
}

# Parameter validation rules
PARAMETER_VALIDATION = {
    'input_file': {'type': str, 'required': True},
    'output_file': {'type': str, 'required': False},
    'plugin': {'type': str, 'required': True},
    'd1_proportion': {'type': float, 'min': 0.0, 'max': 1.0},
    'd2_proportion': {'type': float, 'min': 0.0, 'max': 1.0},
    'd3_proportion': {'type': float, 'min': 0.0, 'max': 1.0},
    'd4_proportion': {'type': float, 'min': 0.0, 'max': 1.0},
    'd5_proportion': {'type': float, 'min': 0.0, 'max': 1.0},
    'd6_proportion': {'type': float, 'min': 0.0, 'max': 1.0},
    'short_window': {'type': int, 'min': 1, 'max': 1000},
    'medium_window': {'type': int, 'min': 1, 'max': 1000},
    'long_window': {'type': int, 'min': 1, 'max': 1000},
    'stl_period': {'type': int, 'min': 2, 'max': 365},
    'wavelet_levels': {'type': int, 'min': 1, 'max': 10},
    'min_split_size': {'type': int, 'min': 1, 'max': 10000}
}

# Feature engineering plugin integration settings
EXTERNAL_PLUGIN_CONFIG = {
    'supported_repositories': [
        'feature-eng',
        'prediction_provider'
    ],
    'plugin_interface_version': '1.0',
    'isolation_level': 'strict',
    'replicability_validation': True
}

def validate_config(config):
    """
    Validate configuration parameters against defined rules.
    
    Args:
        config (dict): Configuration to validate
        
    Returns:
        dict: Validation result with errors and warnings
    """
    errors = []
    warnings = []
    
    # Check split proportions sum to 1.0
    split_props = [
        config.get('d1_proportion', 0),
        config.get('d2_proportion', 0),
        config.get('d3_proportion', 0),
        config.get('d4_proportion', 0),
        config.get('d5_proportion', 0),
        config.get('d6_proportion', 0)
    ]
    
    total_proportion = sum(split_props)
    if abs(total_proportion - 1.0) > 0.001:
        errors.append(f"Dataset split proportions sum to {total_proportion}, should be 1.0")
    
    # Validate individual parameters
    for param, rules in PARAMETER_VALIDATION.items():
        if param in config:
            value = config[param]
            
            # Type validation
            if 'type' in rules and not isinstance(value, rules['type']):
                errors.append(f"Parameter '{param}' should be {rules['type'].__name__}, got {type(value).__name__}")
            
            # Range validation for numeric types
            if isinstance(value, (int, float)):
                if 'min' in rules and value < rules['min']:
                    errors.append(f"Parameter '{param}' value {value} below minimum {rules['min']}")
                if 'max' in rules and value > rules['max']:
                    errors.append(f"Parameter '{param}' value {value} above maximum {rules['max']}")
        
        elif rules.get('required', False):
            errors.append(f"Required parameter '{param}' is missing")
    
    # Validate external plugin paths if external feature engineering is enabled
    if config.get('use_external_feature_eng', False):
        plugin_path = config.get('feature_eng_plugin_path')
        if not plugin_path:
            errors.append("feature_eng_plugin_path required when use_external_feature_eng is True")
        elif not os.path.exists(plugin_path):
            warnings.append(f"External plugin path does not exist: {plugin_path}")
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def get_effective_config(config_dict):
    """
    Get effective configuration by merging with defaults.
    
    Args:
        config_dict (dict): User-provided configuration
        
    Returns:
        dict: Effective configuration with defaults applied
    """
    effective_config = DEFAULT_VALUES.copy()
    effective_config.update(config_dict)
    return effective_config

