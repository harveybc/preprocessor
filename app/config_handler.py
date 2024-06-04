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
    'headers': False
}

def load_config(args):
    config = {}
    if args.load_config:
        try:
            with open(args.load_config, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Error: The file {args.load_config} does not exist.")
            raise

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
    config['single'] = args.single if args.single else config.get('single', 0)
    config['multi'] = args.multi if args.multi else config.get('multi', [0])
    config['force_date'] = args.force_date if args.force_date else config.get('force_date', False)
    config['headers'] = args.headers if args.headers else config.get('headers', False)

    return config

def save_config(config):
    plugin_specific_params = {
        'normalizer': ['norm_method', 'range'],
        'unbiaser': ['window_size', 'ema_alpha'],
        'trimmer': ['remove_rows', 'remove_columns'],
        'feature_selector': ['method', 'single', 'multi', 'max_lag', 'significance_level'],
        'cleaner': ['clean_method', 'period', 'outlier_threshold', 'solve_missing', 'delete_outliers', 'interpolate_outliers', 'delete_nan', 'interpolate_nan'],
    }

    plugin_name = config['plugin_name'] if config['plugin_name'] != 'default_plugin' else 'normalizer'
    general_params = ['csv_file', 'plugin_name', 'output_file', 'save_config', 'quiet_mode', 'headers', 'force_date']
    selected_plugin_params = plugin_specific_params.get(plugin_name, [])
    filtered_params = {k: v for k, v in config.items() if (k in general_params or k in selected_plugin_params) and v is not None and v != default_values.get(k)}

    config_filename = config['save_config'] if config['save_config'] else 'config_output.json'
    with open(config_filename, 'w') as f:
        json.dump(filtered_params, f, indent=4)
    return config_filename

def load_remote_config(remote_config_url):
    try:
        response = requests.get(remote_config_url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Failed to load remote configuration: {e}", file=sys.stderr)
        return None
