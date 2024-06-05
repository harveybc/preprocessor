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
    'force_date': False,
    'headers': False,
    'remote_log': None,
    'remote_save_config': None,
    'remote_load_config': None,
    'remote_username': 'test',
    'remote_password': 'pass'
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

    if hasattr(args, 'remote_load_config') and args.remote_load_config:
        remote_config, success = load_remote_config(args.remote_load_config, args.remote_username, args.remote_password)
        if success:
            config.update(remote_config)
            print(f"Downloaded configuration from {args.remote_load_config}")

    for key in vars(args):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)

    return config

def save_config(config):
    config_filename = config.get('save_config', 'config_output.json')
    with open(config_filename, 'w') as f:
        config_str = json.dumps(config, indent=4)
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
