import sys
import os
import json
import requests
import pkg_resources

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

def load_plugin(plugin_name):
    try:
        entry_point = next(pkg_resources.iter_entry_points('preprocessor.plugins', plugin_name))
        return entry_point.load()
    except StopIteration:
        print(f"Plugin {plugin_name} not found.", file=sys.stderr)
        return None

def load_remote_config(remote_config_url):
    try:
        response = requests.get(remote_config_url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Failed to load remote configuration: {e}", file=sys.stderr)
        return None

def main():
    args = parse_args()

    config = {
        'csv_file': args.csv_file,
        'output_file': args.output_file if args.output_file else CSV_OUTPUT_PATH,
        'plugin_name': args.plugin if args.plugin else DEFAULT_PLUGIN,
        'norm_method': args.norm_method if args.norm_method else DEFAULT_NORMALIZATION_METHOD,
        'range': tuple(args.range) if args.range else DEFAULT_NORMALIZATION_RANGE,
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
        'clean_method': args.clean_method,
        'period': args.period,
        'outlier_threshold': args.outlier_threshold,
        'solve_missing': args.solve_missing,
        'delete_outliers': args.delete_outliers,
        'interpolate_outliers': args.interpolate_outliers,
        'delete_nan': args.delete_nan,
        'interpolate_nan': args.interpolate_nan,
        'method': args.method,
        'single': args.single,
        'multi': args.multi
    }

    if args.remote_config:
        remote_config = load_remote_config(args.remote_config)
        if remote_config:
            config.update(remote_config)

    if args.load_config:
        try:
            with open(args.load_config, 'r') as f:
                local_config = json.load(f)
            config.update(local_config)
        except FileNotFoundError:
            print(f"Error: The file {args.load_config} does not exist.")
            raise

    data = load_csv(config['csv_file'])

    plugin_class = load_plugin(config['plugin_name'])
    if plugin_class is None:
        print(f"Error: The plugin {config['plugin_name']} could not be loaded.")
        return

    plugin = plugin_class()
    processed_data = plugin.process(data, method=config['method'], save_params=config['save_config'], load_params=config['load_config'])

    if not config['quiet_mode']:
        print("Processing complete. Writing output...")

    write_csv(config['output_file'], processed_data)

    if not config['quiet_mode']:
        print(f"Output written to {config['output_file']}")

if __name__ == '__main__':
    main()
