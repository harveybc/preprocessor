"""
Main Entry Point for Preprocessor System

This module provides the main entry point for the preprocessing system with 
external feature engineering integration and BDD-compliant architecture.

Key Features:
- Configuration management with multi-source merging
- External plugin integration validation
- Comprehensive error handling and logging
- Remote configuration and logging support
- Plugin parameter validation and override

Author: Preprocessor System  
Date: 2025-07-11
"""

import sys
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from app.config_handler import load_config, save_config, remote_load_config, remote_save_config, remote_log
from app.cli import parse_args
from app.data_processor import run_preprocessor_pipeline, validate_external_plugin_integration
from app.data_handler import load_csv
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from config_merger import merge_config, process_unknown_args


def main():
    """
    Main entry point for the preprocessor system.
    
    Orchestrates:
    1. CLI argument parsing
    2. Configuration loading and merging
    3. External plugin validation
    4. Plugin loading and parameter override
    5. Pipeline execution
    6. Configuration and debug data saving
    """
    
    try:
        print("[INFO] Starting Preprocessor System...")
        
        # 1. Parse CLI arguments
        print("[DEBUG] Parsing initial arguments...")
        args, unknown_args = parse_args()
        cli_args = vars(args)

        # 2. Load and merge configuration
        print("[DEBUG] Loading default configuration...")
        config = DEFAULT_VALUES.copy()

        file_config = {}
        
        # Load remote configuration if specified
        if args.remote_load_config:
            print(f"[INFO] Loading remote configuration from {args.remote_load_config}...")
            file_config = remote_load_config(
                args.remote_load_config, 
                args.remote_username, 
                args.remote_password
            )
            print(f"[DEBUG] Loaded remote config: {len(file_config)} parameters")

        # Load local configuration if specified
        if args.load_config:
            print(f"[INFO] Loading local configuration from {args.load_config}...")
            file_config = load_config(args.load_config)
            print(f"[DEBUG] Loaded local config: {len(file_config)} parameters")

        # Merge all configuration sources
        print("[DEBUG] Merging configuration with CLI arguments and unknown args...")
        unknown_args_dict = process_unknown_args(unknown_args)
        config = merge_config(config, {}, file_config, cli_args, unknown_args_dict)

        # 3. Validate external plugin integration
        if not validate_external_plugin_integration(config):
            print("[WARNING] External plugin integration validation failed")

        # 4. Load and configure plugin
        plugin_name = config['plugin']
        print(f"[INFO] Loading plugin: {plugin_name}")
        
        try:
            plugin_class, _ = load_plugin('preprocessor.plugins', plugin_name)
            plugin = plugin_class()
        except Exception as e:
            print(f"[ERROR] Failed to load plugin {plugin_name}: {e}")
            raise

        # Override plugin parameters with configuration
        print("[DEBUG] Applying configuration to plugin parameters...")
        plugin.set_params(**config)
        
        # Get plugin-specific parameters for final configuration merge
        plugin_params = getattr(plugin, 'plugin_params', {})
        print(f"[DEBUG] Plugin has {len(plugin_params)} default parameters")
        
        # Final configuration merge with plugin-specific parameters
        print("[DEBUG] Final configuration merge with plugin parameters...")
        config = merge_config(config, plugin_params, file_config, cli_args, unknown_args_dict)

        # 5. Execute preprocessing pipeline
        print("[INFO] Running preprocessing pipeline...")
        processed_data = run_preprocessor_pipeline(config, plugin)
        
        if processed_data is not None:
            print("[INFO] Preprocessing completed successfully")
            print(f"[DEBUG] Summary:\n{processed_data}")
        else:
            print("[ERROR] Preprocessing returned no data")

        # 6. Save configurations and debug data
        _save_outputs(config, args)
        
        print("[INFO] Preprocessor System completed successfully")
        
    except KeyboardInterrupt:
        print("\n[INFO] Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        sys.exit(1)


def _save_outputs(config: Dict[str, Any], args) -> None:
    """
    Save configuration and debug outputs.
    
    Args:
        config (Dict[str, Any]): Final configuration
        args: Parsed CLI arguments
    """
    
    # Save local configuration if specified
    if config.get('save_config'):
        try:
            save_config(config, config['save_config'])
            print(f"[INFO] Configuration saved to {config['save_config']}")
        except Exception as e:
            print(f"[WARNING] Failed to save configuration: {e}")

    # Save configuration remotely if specified
    if config.get('remote_save_config'):
        try:
            print(f"[INFO] Saving remote configuration to {config['remote_save_config']}...")
            remote_save_config(
                config, 
                config['remote_save_config'], 
                config.get('remote_username'), 
                config.get('remote_password')
            )
            print("[INFO] Remote configuration saved successfully")
        except Exception as e:
            print(f"[WARNING] Failed to save remote configuration: {e}")

    # Log data remotely if specified
    if config.get('remote_log'):
        try:
            print(f"[INFO] Logging data remotely to {config['remote_log']}...")
            # Create debug data for remote logging
            debug_data = {
                'config': config,
                'timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }
            remote_log(
                debug_data, 
                config['remote_log'], 
                config.get('remote_username'), 
                config.get('remote_password')
            )
            print("[INFO] Data logged remotely successfully")
        except Exception as e:
            print(f"[WARNING] Failed to log data remotely: {e}")


def validate_configuration(config: Dict[str, Any]) -> bool:
    """
    Validate the final configuration before processing.
    
    Args:
        config (Dict[str, Any]): Configuration to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    
    # Check required parameters
    required_params = ['input_file', 'plugin']
    missing_params = [param for param in required_params if param not in config or config[param] is None]
    
    if missing_params:
        print(f"[ERROR] Missing required parameters: {missing_params}")
        return False
    
    # Validate file paths
    import os
    if not os.path.exists(config['input_file']):
        print(f"[ERROR] Input file not found: {config['input_file']}")
        return False
    
    # Validate dataset proportions
    proportions = [
        config.get('d1_proportion', 0.33),
        config.get('d2_proportion', 0.083),
        config.get('d3_proportion', 0.083),
        config.get('d4_proportion', 0.33),
        config.get('d5_proportion', 0.083),
        config.get('d6_proportion', 0.083)
    ]
    
    total = sum(proportions)
    if abs(total - 1.0) > 0.01:  # Allow small floating point errors
        print(f"[ERROR] Dataset proportions sum to {total}, should be 1.0")
        return False
    
    print("[DEBUG] Configuration validation passed")
    return True


if __name__ == "__main__":
    main()
