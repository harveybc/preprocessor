import sys
import os
import json
import requests
import pkg_resources
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from app.cli import parse_args
from app.config import (
    CSV_INPUT_PATH,
    CSV_OUTPUT_PATH,
    CONFIG_SAVE_PATH,
    CONFIG_LOAD_PATH,
    DEFAULT_PLUGIN,
    REMOTE_LOG_URL,
    REMOTE_CONFIG_URL,
    PLUGIN_DIRECTORY,
    DEFAULT_NORMALIZATION_METHOD,
    DEFAULT_NORMALIZATION_RANGE,
    DEFAULT_QUIET_MODE
)
from app.data_handler import load_csv, write_csv
from app.default_plugin import DefaultPlugin
from app.config_handler import load_config, save_config, save_debug_info, load_remote_config, default_values

def load_plugin(plugin_name):
    try:
        entry_point = next(pkg_resources.iter_entry_points('preprocessor.plugins', plugin_name))
        return entry_point.load()
    except StopIteration:
        print(f"Plugin {plugin_name} not found.", file=sys.stderr)
        return None

def save_remote_config(config, url, username, password):
    try:
        response = requests.post(
            url,
            auth=(username, password),
            data={'json_config': config}
        )
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Failed to save remote configuration: {e}", file=sys.stderr)
        return False

def log_remote_info(config, debug_info, url, username, password):
    try:
        data = {
            'json_config': config,
            'json_result': json.dumps(debug_info)
        }
        response = requests.post(
            url,
            auth=(username, password),
            data=data
        )
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Failed to log remote information: {e}", file=sys.stderr)
        return False

def merge_config(config, args):
    cli_args = vars(args)
    for key, value in cli_args.items():
        if value is not None:
            config[key] = value
    return config

def main():
    args = parse_args()

    debug_info = {
        "execution_time": "",
        "input_rows": 0,
        "output_rows": 0,
        "input_columns": 0,
        "output_columns": 0
    }

    start_time = time.time()

    config = load_config(args)
    config = merge_config(config, args)

    if not config.get('csv_file'):
        print("Error: No CSV file specified.", file=sys.stderr)
        return

    data = load_csv(config['csv_file'], headers=config['headers'])
    debug_info["input_rows"] = len(data)
    debug_info["input_columns"] = len(data.columns)

    plugin_class = load_plugin(config['plugin_name'])
    if plugin_class is None:
        print(f"Error: The plugin {config['plugin_name']} could not be loaded.")
        return

    plugin = plugin_class()
    processed_data = plugin.process(
        data,
        method=config['method'],
        save_params=config['save_config'],
        load_params=config['load_config'],
        window_size=config['window_size'],
        ema_alpha=config['ema_alpha'],
        single=config['single'],
        multi=config['multi'],
        max_lag=config['max_lag'],
        significance_level=config['significance_level'],
        clean_method=config['clean_method'],
        period=config['period'],
        outlier_threshold=config['outlier_threshold'],
        solve_missing=config['solve_missing'],
        delete_outliers=config['delete_outliers'],
        interpolate_outliers=config['interpolate_outliers'],
        delete_nan=config['delete_nan'],
        interpolate_nan=config['interpolate_nan']
    )

    debug_info["output_rows"] = len(processed_data)
    debug_info["output_columns"] = len(processed_data.columns)

    include_date = config['force_date'] or not (config['method'] in ['select_single', 'select_multi'])

    if not config['quiet_mode']:
        print("Processing complete. Writing output...")

    write_csv(config['output_file'], processed_data, include_date=include_date, headers=config['headers'])

    if not config['quiet_mode']:
        print(f"Output written to {config['output_file']}")
        print(f"Configuration saved to {os.path.basename(config['save_config'])}")

    config_str, config_filename = save_config(config)

    execution_time = time.time() - start_time
    debug_info["execution_time"] = execution_time

    save_debug_info(debug_info, args.debug_file)

    if not config['quiet_mode']:
        print(f"Debug info saved to {args.debug_file}")
        print(f"Execution time: {execution_time} seconds")

    if args.remote_save_config:
        if save_remote_config(config_str, args.remote_save_config, args.remote_username, args.remote_password):
            print(f"Configuration successfully saved to remote URL {args.remote_save_config}")
        else:
            print(f"Failed to save configuration to remote URL {args.remote_save_config}")

    if args.remote_log:
        if log_remote_info(config_str, debug_info, args.remote_log, args.remote_username, args.remote_password):
            print(f"Debug information successfully logged to remote URL {args.remote_log}")
        else:
            print(f"Failed to log debug information to remote URL {args.remote_log}")

if __name__ == '__main__':
    main()
