import json

def load_config(args):
    if args.load_config:
        with open(args.load_config, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    if args.plugin:
        config['plugin'] = args.plugin

    return config

def save_config(config, path='config_out.json'):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    return config, path

def merge_config(config, cli_args):
    # Ensure mandatory keys are present and set default values if not
    mandatory_keys = ['plugin', 'csv_file']
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

    for key in mandatory_keys:
        if key not in config:
            if key in cli_args:
                config[key] = cli_args[key]
            else:
                config[key] = default_values[key]

    # Merge CLI arguments, overriding config file values
    for key, value in cli_args.items():
        if value is not None:
            config[key] = value

    return config

def save_debug_info(debug_info, path='debug_out.json'):
    with open(path, 'w') as f:
        json.dump(debug_info, f, indent=4)
