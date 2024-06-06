# app/config_handler.py

import json

def load_config(args):
    if args.load_config:
        with open(args.load_config, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    # Ensure the plugin key is present
    if 'plugin' not in config:
        config['plugin'] = 'default_plugin'

    return config

def save_config(config, path='config_out.json'):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    return config, path

def merge_config(config, cli_args):
    # Ensure mandatory keys are present and set default values if not
    default_values = {
        'plugin': 'default_plugin',
        'output_file': 'output.csv',
        'remote_log': '',
        'remote_save_config': '',
        'remote_load_config': '',
        'remote_username': '',
        'remote_password': '',
        'quiet_mode': False,
        'force_date': False,
        'headers': False,
        'debug_file': 'debug_out.json'
    }

    # Set default values
    for key, value in default_values.items():
        config.setdefault(key, value)

    # Merge CLI arguments, overriding config file values
    for key, value in cli_args.items():
        if value is not None:
            config[key] = value

    return config

def save_debug_info(debug_info, path='debug_out.json'):
    with open(path, 'w') as f:
        json.dump(debug_info, f, indent=4)
