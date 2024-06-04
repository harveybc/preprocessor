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
    
    # Debugging: Print parsed arguments
    print("Parsed arguments:", args)

    config = {}

    if args.load_config:
        try:
            with open(args.load_config, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Error: The file {args.load_config} does not exist.")
            raise

    # Override config values with command line arguments if provided
    config['csv_file'] = args.csv_file if args.csv_file else config.get('csv_file', CSV_INPUT_PATH)
    config['output_file'] = args.output_file if args.output_file else config.get('output_file', CSV_OUTPUT_PATH)
    config['plugin_name'] = args.plugin if args.plugin else config.get('plugin_name', DEFAULT_PLUGIN)
    config['norm_method'] = args.norm_method if args.norm_method else config.get('norm_method', DEFAULT_NORMALIZATION_METHOD)
    config['range'] = tuple(args.range) if args.range else config.get('range', DEFAULT_NORMALIZATION_RANGE)
    config['save_config'] = args.save_config if args.save_config else config.get('save_config', CONFIG_SAVE_PATH)
    config['load_config'] = args.load_config if args.load_config else config.get('load_config', CONFIG_LOAD_PATH)
    config['quiet_mode'] = args.quiet_mode if args.quiet_mode else config.get('quiet_mode', DEFAULT_QUIET_MODE)
    config['window_size'] = args.window_size if args.window_size else config.get('window_size')
    config['ema_alpha'] = args.ema_alpha if args.ema_alpha else config.get('ema_alpha')
    config['remove_rows'] = args.remove_rows if args.remove_rows else config.get('remove_rows')
    config['remove_columns'] = args.remove_columns if args.remove_columns else config.get('remove_columns')
    config['max_lag'] = args.max_lag if args.max_lag else config.get('max_lag')
    config['significance_level'] = args.significance_level if args.significance_level else config.get('significance_level')
    config['alpha'] = args.alpha if args.alpha else config.get('alpha')
    config['l1_ratio'] = args.l1_ratio if args.l1_ratio else config.get('l1_ratio')
    config['model_type'] = args.model_type if args.model_type else config.get('model_type')
    config['timesteps'] = args.timesteps if args.timesteps else config.get('timesteps')
    config['features'] = args.features if args.features else config.get('features')
    config['clean_method'] = args.clean_method if args.clean_method else config.get('clean_method')
    config['period'] = args.period if args.period else config.get('period')
    config['outlier_threshold'] = args.outlier_threshold if args.outlier_threshold else config.get('outlier_threshold')
    config['solve_missing'] = args.solve_missing if args.solve_missing else config.get('solve_missing', False)
    config['delete_outliers'] = args.delete_outliers if args.delete_outliers else config.get('delete_outliers', False)
    config['interpolate_outliers'] = args.interpolate_outliers if args.interpolate_outliers else config.get('interpolate_outliers', False)
    config['delete_nan'] = args.delete_nan if args.delete_nan else config.get('delete_nan', False)
    config['interpolate_nan'] = args.interpolate_nan if args.interpolate_nan else config.get('interpolate_nan', False)
    config['method'] = args.method if args.method else config.get('method')
    config['single'] = args.single if args.single else config.get('single')
    config['multi'] = args.multi if args.multi else config.get('multi')
    config['force_date'] = args.force_date if args.force_date else config.get('force_date', False)
    config['headers'] = args.headers if args.headers else config.get('headers', False)

    # Debugging: Print configuration
    print("Configuration:", config)

    if args.remote_config:
        remote_config = load_remote_config(args.remote_config)
        if remote_config:
            config.update(remote_config)

    # Ensure CSV file is specified
    if not config.get('csv_file'):
        print("Error: No CSV file specified.", file=sys.stderr)
        return

    data = load_csv(config['csv_file'], headers=config['headers'])

    # Debugging: Print loaded data
    print("Loaded data:\n", data.head())

    plugin_class = load_plugin(config['plugin_name'])
    if plugin_class is None:
        print(f"Error: The plugin {config['plugin_name']} could not be loaded.")
        return

    plugin = plugin_class()
    processed_data = plugin.process(data, method=config['method'], save_params=config['save_config'], load_params=config['load_config'], single=config['single'], multi=config['multi'])

    # Debugging: Print processed data
    print("Processed data:\n", processed_data.head())

    # Determine if date column should be included in the output
    include_date = config['force_date'] or not (config['method'] in ['select_single', 'select_multi'])

    if not config['quiet_mode']:
        print("Processing complete. Writing output...")

    write_csv(config['output_file'], processed_data, include_date=include_date, headers=config['headers'])

    # Save configuration to file
    config_filename = config['save_config'] if config['save_config'] else 'config_output.json'
    with open(config_filename, 'w') as f:
    json.dump({k: v for k, v in config.items() if v is not None}, f, indent=4)

if not config['quiet_mode']:
    print(f"Output written to {config['output_file']}")
    print(f"Configuration saved to {os.path.basename(config_filename)}")


if __name__ == '__main__':
    main()
