"""
Command Line Interface for Preprocessor System

This module provides comprehensive CLI argument parsing with support for
external feature engineering integration and advanced preprocessing options.
"""

import argparse
from typing import Tuple, List, Any

def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    """
    Parse command line arguments for the preprocessor system.
    
    Returns:
        Tuple of (parsed_args, unknown_args)
    """
    parser = argparse.ArgumentParser(
        description='Preprocessor: Advanced data preprocessing with external plugin integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing
  preprocessor --input_file data.csv --plugin plugin_default
  
  # With external feature engineering
  preprocessor --input_file data.csv --use_external_feature_eng --technical_indicators
  
  # With decomposition
  preprocessor --input_file data.csv --decomposition_enabled --decomp_features Close Volume
  
  # Custom split proportions
  preprocessor --input_file data.csv --d1_proportion 0.4 --d4_proportion 0.4
        """
    )
    
    # Core file arguments
    parser.add_argument('--input_file', 
                       help='Path to the CSV file to be processed')
    parser.add_argument('--output_file', 
                       help='Path to save the output data')
    parser.add_argument('--debug_file', 
                       help='Path to save debug information')
    
    # Configuration management
    parser.add_argument('--load_config', 
                       help='Path to the configuration file to load')
    parser.add_argument('--save_config', 
                       help='Path to save the configuration file')
    parser.add_argument('--remote_load_config', 
                       help='URL to load remote configuration')
    parser.add_argument('--remote_save_config', 
                       help='URL to save remote configuration')
    parser.add_argument('--remote_log', 
                       help='URL for remote logging')
    parser.add_argument('--remote_username', 
                       help='Username for remote logging')
    parser.add_argument('--remote_password', 
                       help='Password for remote logging')
    
    # Plugin configuration
    parser.add_argument('--plugin', 
                       default='plugin_default',
                       help='Preprocessor plugin to use (default: plugin_default)')
    
    # Data handling options
    parser.add_argument('--headers', 
                       action='store_true', 
                       help='Indicate if the CSV file has headers')
    parser.add_argument('--force_date', 
                       action='store_true', 
                       help='Force inclusion of the date column in the output')
    parser.add_argument('--quiet_mode', 
                       action='store_true', 
                       help='Suppress all output except for errors')
    parser.add_argument('--only_low_CV', 
                       action='store_true', 
                       help='Process only low coefficient of variation columns')
    
    # Dataset generation
    parser.add_argument('--dataset_prefix', 
                       help='Prefix for base dataset files')
    parser.add_argument('--target_prefix', 
                       help='Prefix for normalized dataset files')
    
    # Dataset split proportions
    parser.add_argument('--d1_proportion', 
                       type=float, 
                       help='Proportion for D1 dataset (autoencoder training)')
    parser.add_argument('--d2_proportion', 
                       type=float, 
                       help='Proportion for D2 dataset (autoencoder validation)')
    parser.add_argument('--d3_proportion', 
                       type=float, 
                       help='Proportion for D3 dataset (autoencoder test)')
    parser.add_argument('--d4_proportion', 
                       type=float, 
                       help='Proportion for D4 dataset (predictor training)')
    parser.add_argument('--d5_proportion', 
                       type=float, 
                       help='Proportion for D5 dataset (predictor validation)')
    parser.add_argument('--d6_proportion', 
                       type=float, 
                       help='Proportion for D6 dataset (predictor test)')
    
    # External feature engineering integration
    parser.add_argument('--use_external_feature_eng', 
                       action='store_true',
                       help='Enable external feature engineering plugins')
    parser.add_argument('--feature_eng_plugin_path', 
                       help='Path to external feature engineering plugins')
    parser.add_argument('--external_plugin_paths', 
                       nargs='*',
                       help='Additional external plugin paths')
    
    # Technical indicators
    parser.add_argument('--technical_indicators', 
                       action='store_true',
                       help='Enable technical indicator generation')
    parser.add_argument('--short_window', 
                       type=int, 
                       help='Short-term window for technical indicators')
    parser.add_argument('--medium_window', 
                       type=int, 
                       help='Medium-term window for technical indicators')
    parser.add_argument('--long_window', 
                       type=int, 
                       help='Long-term window for technical indicators')
    parser.add_argument('--indicators', 
                       nargs='*',
                       help='List of technical indicators to compute')
    
    # Decomposition configuration
    parser.add_argument('--decomposition_enabled', 
                       action='store_true',
                       help='Enable feature decomposition')
    parser.add_argument('--decomp_features', 
                       nargs='*',
                       help='List of features to decompose')
    parser.add_argument('--decomp_methods', 
                       help='JSON string or file path specifying decomposition methods for each feature')
    parser.add_argument('--stl_period', 
                       type=int, 
                       help='Period for STL decomposition')
    parser.add_argument('--stl_robust', 
                       action='store_true',
                       help='Use robust STL decomposition')
    parser.add_argument('--wavelet_name', 
                       help='Wavelet name for wavelet decomposition')
    parser.add_argument('--wavelet_levels', 
                       type=int, 
                       help='Number of wavelet decomposition levels')
    parser.add_argument('--mtm_bandwidth', 
                       type=float, 
                       help='Bandwidth for Multi-taper method')
    parser.add_argument('--mtm_n_tapers', 
                       type=int, 
                       help='Number of tapers for Multi-taper method')
    
    # Normalization configuration
    parser.add_argument('--normalization_method', 
                       choices=['min_max', 'z_score', 'robust'],
                       help='Normalization method to use')
    parser.add_argument('--normalization_range', 
                       nargs=2, 
                       type=float,
                       help='Target range for min-max normalization (min max)')
    parser.add_argument('--fit_on_training_only', 
                       action='store_true',
                       help='Fit normalizer only on training sets (D1, D4)')
    
    # Validation and quality control
    parser.add_argument('--validate_splits', 
                       action='store_true',
                       help='Validate dataset splits before processing')
    parser.add_argument('--min_split_size', 
                       type=int, 
                       help='Minimum size for each dataset split')
    parser.add_argument('--data_quality_checks', 
                       action='store_true',
                       help='Perform data quality validation')
    
    # Performance and logging
    parser.add_argument('--performance_monitoring', 
                       action='store_true',
                       help='Enable performance monitoring')
    parser.add_argument('--log_level', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    parser.add_argument('--log_to_file', 
                       action='store_true',
                       help='Enable file logging')
    parser.add_argument('--log_to_console', 
                       action='store_true',
                       help='Enable console logging')
    
    args, unknown = parser.parse_known_args()
    return args, unknown

def validate_args(args: argparse.Namespace) -> dict:
    """
    Validate parsed command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dictionary with validation results
    """
    errors = []
    warnings = []
    
    # Check required arguments
    if not args.input_file:
        errors.append("input_file is required")
    
    # Validate split proportions if provided
    split_args = [
        args.d1_proportion, args.d2_proportion, args.d3_proportion,
        args.d4_proportion, args.d5_proportion, args.d6_proportion
    ]
    
    provided_splits = [x for x in split_args if x is not None]
    if provided_splits:
        if len(provided_splits) != 6:
            warnings.append("Some split proportions provided but not all six. Using defaults for missing values.")
        else:
            total = sum(provided_splits)
            if abs(total - 1.0) > 0.001:
                errors.append(f"Split proportions sum to {total}, should sum to 1.0")
    
    # Validate external feature engineering configuration
    if args.use_external_feature_eng:
        if not args.feature_eng_plugin_path:
            warnings.append("External feature engineering enabled but no plugin path specified. Using default.")
    
    # Validate decomposition configuration
    if args.decomposition_enabled:
        if not args.decomp_features:
            warnings.append("Decomposition enabled but no features specified for decomposition.")
    
    # Validate technical indicators
    if args.technical_indicators:
        if args.short_window and args.medium_window and args.short_window >= args.medium_window:
            errors.append("Short window must be less than medium window")
        if args.medium_window and args.long_window and args.medium_window >= args.long_window:
            errors.append("Medium window must be less than long window")
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def generate_help_text() -> str:
    """
    Generate comprehensive help text for the CLI.
    
    Returns:
        Formatted help text string
    """
    help_sections = [
        "Preprocessor System - Advanced Data Preprocessing",
        "=" * 50,
        "",
        "BASIC USAGE:",
        "  preprocessor --input_file data.csv",
        "",
        "EXTERNAL FEATURE ENGINEERING:",
        "  # Enable technical indicators from feature-eng repo",
        "  preprocessor --input_file data.csv --use_external_feature_eng --technical_indicators",
        "",
        "  # Enable decomposition with specific features",
        "  preprocessor --input_file data.csv --decomposition_enabled --decomp_features Close Volume",
        "",
        "DATASET SPLITTING:",
        "  # Custom proportions for autoencoder/predictor training",
        "  preprocessor --input_file data.csv --d1_proportion 0.4 --d4_proportion 0.4",
        "",
        "PLUGIN INTEGRATION:",
        "  # Use plugins from external repositories",
        "  preprocessor --input_file data.csv --feature_eng_plugin_path /path/to/feature-eng/plugins",
        "",
        "For detailed parameter descriptions, use --help"
    ]
    
    return "\n".join(help_sections)

def parse_decomp_methods(decomp_methods_arg: str) -> dict:
    """
    Parse decomposition methods argument (JSON string or file path).
    
    Args:
        decomp_methods_arg: JSON string or file path
        
    Returns:
        Dictionary of decomposition methods
    """
    import json
    import os
    
    if not decomp_methods_arg:
        return {}
    
    # Check if it's a file path
    if os.path.exists(decomp_methods_arg):
        try:
            with open(decomp_methods_arg, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in decomposition methods file: {e}")
    
    # Try to parse as JSON string
    try:
        return json.loads(decomp_methods_arg)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in decomposition methods argument: {e}")
