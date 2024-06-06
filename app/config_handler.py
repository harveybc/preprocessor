import json

# Define default values for CLI parameters
DEFAULT_VALUES = {
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
    # Remove default, false, null, or None values
    config = {k: v for k, v in config.items() if v not in [None, False, '', [], {}, 'None']}
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    return config, path

def merge_config(config, cli_args):
    # Set CLI arguments, overriding config file values
    for key, value in cli_args.items():
        if value is not None:
            config[key] = value

    # Set default values for missing keys (for internal use)
    for key, value in DEFAULT_VALUES.items():
        config.setdefault(key, value)

    return config

def save_debug_info(debug_info, path='debug_out.json'):
    # Remove default, false, null, or None values
    debug_info = {k: v for k, v in debug_info.items() if v not in [None, False, '', [], {}, 'None']}
    with open(path, 'w') as f:
        json.dump(debug_info, f, indent=4)
