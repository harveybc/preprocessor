# app/cli.py

import argparse
from plugin_loader import get_plugin_params

def parse_args():
    print("Parsing initial arguments...")
    parser = argparse.ArgumentParser(description='Preprocessor CLI')

    # Standard arguments
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

    # Parse known and unknown args separately
    args, unknown = parser.parse_known_args()
    print(f"Initial args: {args}")
    print(f"Unknown args: {unknown}")

    return args, unknown
