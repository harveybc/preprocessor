# app/cli.py

import argparse
from plugin_loader import get_plugin_params

def parse_args():
    print("Parsing initial arguments...")
    parser = argparse.ArgumentParser(description='Preprocessor CLI')
    
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('--save_config', type=str, help='Path to save the configuration')
    parser.add_argument('--load_config', type=str, help='Path to load the configuration')
    parser.add_argument('--plugin', type=str, help='Name of the plugin to use')
    parser.add_argument('--output_file', type=str, help='Path to the output CSV file')
    parser.add_argument('--remote_log', type=str, help='URL for remote logging')
    parser.add_argument('--remote_save_config', type=str, help='URL to save the configuration remotely')
    parser.add_argument('--remote_load_config', type=str, help='URL to load the configuration remotely')
    parser.add_argument('--remote_username', type=str, help='Username for remote operations')
    parser.add_argument('--remote_password', type=str, help='Password for remote operations')
    parser.add_argument('--quiet_mode', action='store_true', help='Run in quiet mode')
    parser.add_argument('--force_date', action='store_true', help='Force date inclusion')
    parser.add_argument('--headers', action='store_true', help='Indicate if CSV has headers')
    parser.add_argument('--debug_file', type=str, help='Path to save debug information')
    parser.add_argument('--method', type=str, help='Method to use for the plugin')  # Global method parameter

    # First pass parse to get the plugin name
    args, unknown = parser.parse_known_args()
    print(f"Initial args: {args}")
    print(f"Unknown args: {unknown}")

    # Dynamically add plugin parameters if a plugin is specified
    if args.plugin:
        print(f"Getting plugin parameters for: {args.plugin}")
        plugin_params = get_plugin_params(args.plugin)
        print(f"Retrieved plugin params: {plugin_params}")
        for param, default in plugin_params.items():
            parser.add_argument(f'--{param}', type=type(default), default=default, help=f'{param} for the plugin {args.plugin}')
    
    # Parse again to include dynamically added plugin parameters
    args = parser.parse_args()
    print(f"Final args: {args}")

    return args
