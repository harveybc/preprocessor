import sys
import os
import json
import requests
import pkg_resources

# Ensure the parent directory is in the PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Print PYTHONPATH after modification
print("Modified Python path:", sys.path)

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

def load_plugin(plugin_name):
    """
    Load a plugin based on the name specified.
    """
    try:
        entry_point = next(pkg_resources.iter_entry_points('preprocessor.plugins', plugin_name))
        return entry_point.load()
    except StopIteration:
        print(f"Plugin {plugin_name} not found.", file=sys.stderr)
        return None

def load_remote_config(remote_config_url):
    """
    Load configuration from a remote URL.
    """
    try:
        response = requests.get(remote_config_url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Failed to load remote configuration: {e}", file=sys.stderr)
        return None

def main():
    # Parse command line arguments
    args = parse_args()

    # Initialize config with CLI arguments
    config = {
        'csv_file': args.csv_file,
        'output_file': args.output_file if args.output_file else CSV_OUTPUT_PATH,
        'plugin_name': args.plugin if args.plugin else DEFAULT_PLUGIN,
        'method': args.method if args.method else DEFAULT_NORMALIZATION_METHOD,
        'save_config': args.save_config if args.save_config else CONFIG_SAVE_PATH,
        'load_config': args.load_config if args.load_config else CONFIG_LOAD_PATH,
        'quiet_mode': args.quiet_mode if args.quiet_mode else DEFAULT_QUIET_MODE,
        'window_size': args.window_size,
        'ema_alpha': args.ema_alpha,
        'remove_rows': args.remove_rows,
        'remove_columns': args.remove_columns,
        'max_lag': args.max_lag,
        'significance_level': args.significance_level,
        'alpha': args.alpha,
        'l1_ratio': args.l1_ratio,
        'model_type': args.model_type,
        'timesteps': args.timesteps,
        'features': args.features,
        'period': args.period,
        'outlier_threshold': args.outlier_threshold,
        'solve_missing': args.solve_missing,
        'delete_outliers': args.delete_outliers,
        'interpolate_outliers': args.interpolate_outliers,
        'delete_nan': args.delete_nan,
        'interpolate_nan': args.interpolate_nan,
        'headers': args.headers
    }

    # Load remote configuration if provided
    if args.remote_config:
        remote_config = load_remote_config(args.remote_config)
        if remote_config:
            config.update(remote_config)

    # Load local configuration if provided
    if args.load_config:
        try:
            with open(args.load_config, 'r') as f:
                local_config = json.load(f)
            config.update(local_config)
        except FileNotFoundError:
            print(f"Error: The file {args.load_config} does not exist.")
            raise

    # Load the CSV data
    data = load_csv(config['csv_file'], headers=config['headers'])

    # Load and apply the plugin
    plugin_class = load_plugin(config['plugin_name'])
    if plugin_class is None:
        print(f"Plugin {config['plugin_name']} could not be loaded. Exiting.")
        return

    plugin = plugin_class()

    # Determine the plugin-specific parameters
    if config['plugin_name'] == 'unbiaser':
        processed_data = plugin.process(data, method=config['method'], window_size=config['window_size'], ema_alpha=config['ema_alpha'], save_params=config['save_config'], load_params=config['load_config'])
    elif config['plugin_name'] == 'trimmer':
        processed_data = plugin.process(data, remove_rows=config['remove_rows'], remove_columns=config['remove_columns'], save_params=config['save_config'], load_params=config['load_config'])
    elif config['plugin_name'] == 'feature_selector_pre':
        if config['method'] == 'select_single':
            processed_data = plugin.select_single(data, config['select_single'])
        elif config['method'] == 'select_multi':
            processed_data = plugin.select_multi(data, config['select_multi'])
        else:
            processed_data = plugin.process(data, method=config['method'], max_lag=config['max_lag'], significance_level=config['significance_level'], save_params=config['save_config'], load_params=config['load_config'])
    elif config['plugin_name'] == 'feature_selector_post':
        processed_data = plugin.process(data, alpha=config['alpha'], l1_ratio=config['l1_ratio'], model_type=config['model_type'], timesteps=config['timesteps'], features=config['features'], save_params=config['save_config'], load_params=config['load_config'])
    elif config['plugin_name'] == 'cleaner':
        processed_data = plugin.process(data, method=config['method'], period=config['period'], outlier_threshold=config['outlier_threshold'], solve_missing=config['solve_missing'], delete_outliers=config['delete_outliers'], interpolate_outliers=config['interpolate_outliers'], delete_nan=config['delete_nan'], interpolate_nan=config['interpolate_nan'], save_params=config['save_config'], load_params=config['load_config'])
    else:
        processed_data = plugin.process(data, method=config['method'], save_params=config['save_config'], load_params=config['load_config'])

    # Save the processed data to output CSV
    write_csv(config['output_file'], processed_data, headers=config['headers'])

    # Save configuration if save_config path is provided
    if config['save_config']:
        with open(config['save_config'], 'w') as f:
            json.dump(config, f)

    # Log processing completion if remote logging is configured
    if 'remote_log' in config and config['remote_log']:
        try:
            response = requests.post(config['remote_log'], json={'message': 'Processing complete', 'output_file': config['output_file']})
            if not config['quiet_mode']:
                print(f"Remote log response: {response.text}")
        except requests.RequestException as e:
            if not config['quiet_mode']:
                print(f"Failed to send remote log: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
