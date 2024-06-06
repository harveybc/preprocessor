import json
import requests
import sys

from app.config import (
    CSV_INPUT_PATH,
    CSV_OUTPUT_PATH,
    CONFIG_SAVE_PATH,
    CONFIG_LOAD_PATH,
    DEFAULT_PLUGIN,
    DEFAULT_NORMALIZATION_METHOD,
    DEFAULT_NORMALIZATION_RANGE,
    DEFAULT_QUIET_MODE
)

default_values = {
    'csv_file': CSV_INPUT_PATH,
    'output_file': CSV_OUTPUT_PATH,
    'plugin_name': DEFAULT_PLUGIN,
    'norm_method': DEFAULT_NORMALIZATION_METHOD,
    'range': DEFAULT_NORMALIZATION_RANGE,
    'save_config': CONFIG_SAVE_PATH,
    'load_config': CONFIG_LOAD_PATH,
    'quiet_mode': DEFAULT_QUIET_MODE,
    'window_size': None,
    'ema_alpha': None,
    'remove_rows': None,
    'remove_columns': None,
    'max_lag': None,
    'significance_level': None,
    'alpha': None,
    'l1_ratio': None,
    'model_type': None,
    'timesteps': None,
    'features': None,
    'clean_method': None,
    'period': None,
    'outlier_threshold': None,
    'solve_missing': False,
    'delete_outliers': False,
    'interpolate_outliers': False,
    'delete_nan': False,
    'interpolate_nan': False,
    'method': None,
    'single': 0,
    'multi': [0],
    'force_date': False,
    'headers': False,
    'remote_log': None,
    'remote_save_config': None,
    'remote_load_config': None,
    'remote_username': 'test',
    'remote_password': 'pass'
}

# Modify load_config to ensure 'plugin' argument is prioritized
def load_config(args):
    config = default_values.copy()

    # Load configuration from file if specified
    if args.load_config:
        try:
            with open(args.load_config, 'r') as f:
                config.update(json.load(f))
        except FileNotFoundError:
            print(f"Error: The file {args.load_config} does not exist.")
            raise

    # Ensure the CLI 'plugin' argument is prioritized
    if args.plugin:
        config['plugin_name'] = args.plugin

    # Merge remaining CLI arguments
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    return config

def merge_config(config, args):
    cli_args = vars(args)
    for key, value in cli_args.items():
        if value is not None:
            config[key] = value
    return config

def save_config(config):
    plugin_specific_params = {
        'normalizer': ['method', 'norm_method', 'range'],
        'unbiaser': ['method', 'window_size', 'ema_alpha'],
        'trimmer': ['method', 'remove_rows', 'remove_columns'],
        'feature_selector': ['method', 'single', 'multi', 'max_lag', 'significance_level'],
        'cleaner': ['method', 'clean_method', 'period', 'outlier_threshold', 'solve_missing', 'delete_outliers', 'interpolate_outliers', 'delete_nan', 'interpolate_nan'],
    }

    plugin_name = config['plugin_name'] if config['plugin_name'] != 'default_plugin' else 'normalizer'
    general_params = ['csv_file', 'plugin_name', 'output_file', 'save_config', 'quiet_mode', 'headers', 'force_date', 'remote_log', 'remote_save_config', 'remote_load_config']
    selected_plugin_params = plugin_specific_params.get(plugin_name, [])
    filtered_params = {k: v for k, v in config.items() if (k in general_params or k in selected_plugin_params) and v is not None and v != default_values.get(k)}

    config_filename = config['save_config'] if config['save_config'] else 'config_output.json'
    with open(config_filename, 'w') as f:
        config_str = json.dumps(filtered_params, indent=4)
        f.write(config_str)
    return config_str, config_filename

def load_remote_config(remote_config_url, username, password):
    try:
        response = requests.get(remote_config_url, auth=(username, password))
        response.raise_for_status()
        remote_config = response.json()
        if 'json_config' in remote_config:
            return json.loads(remote_config['json_config']), True
        else:
            print("Error: 'json_config' not found in the response.", file=sys.stderr)
            return None, False
    except requests.RequestException as e:
        print(f"Failed to load remote configuration: {e}", file=sys.stderr)
        return None, False

def save_debug_info(debug_info, debug_file):
    with open(debug_file, 'w') as f:
        json.dump(debug_info, f, indent=4)
