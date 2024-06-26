import sys
import os
import json
import requests
import time
from app.plugin_loader import load_plugin
from app.cli import parse_args
from app.config_handler import load_config, save_config, save_debug_info, merge_config, DEFAULT_VALUES
from app.data_handler import load_csv, write_csv

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

def main():
    print("Parsing initial arguments...")
    args, unknown_args = parse_args()
    print(f"Initial args: {args}")
    print(f"Unknown args: {unknown_args}")

    cli_args = vars(args)
    print(f"CLI arguments: {cli_args}")

    # Convert unknown args to a dictionary
    unknown_args_dict = {}
    current_key = None
    for arg in unknown_args:
        if arg.startswith('--'):
            if current_key:
                unknown_args_dict[current_key] = True  # Flags without values are treated as True
            current_key = arg[2:]
        else:
            if current_key:
                unknown_args_dict[current_key] = arg
                current_key = None

    if current_key:
        unknown_args_dict[current_key] = True

    print(f"Unknown args as dict: {unknown_args_dict}")

    # Specific handling for --range argument
    if 'range' in unknown_args_dict:
        range_str = unknown_args_dict['range']
        try:
            unknown_args_dict['range'] = tuple(map(int, range_str.strip("()").split(',')))
        except ValueError:
            print(f"Error: Invalid format for --range argument: {range_str}", file=sys.stderr)
            return

    print("Loading configuration...")
    config = load_config(args)
    print(f"Initial loaded config: {config}")

    print("Merging configuration with CLI arguments...")
    config = merge_config(config, cli_args)
    print(f"Config after merging with CLI args: {config}")

    # Merge plugin-specific arguments
    config.update(unknown_args_dict)
    print(f"Final merged config: {config}")

    debug_info = {
        "execution_time": "",
        "input_rows": 0,
        "output_rows": 0,
        "input_columns": 0,
        "output_columns": 0
    }

    start_time = time.time()

    if not config.get('csv_file'):
        print("Error: No CSV file specified.", file=sys.stderr)
        return

    data = load_csv(config['csv_file'], headers=config['headers'])
    debug_info["input_rows"] = len(data)
    debug_info["input_columns"] = len(data.columns)

    plugin_class, required_params = load_plugin(config['plugin'])
    if plugin_class is None:
        print(f"Error: The plugin {config['plugin']} could not be loaded.")
        return

    plugin = plugin_class()
    plugin_params = {param: config[param] for param in required_params if param in config}
    print(f"Setting plugin parameters: {plugin_params}")
    plugin.set_params(**plugin_params)

    processed_data = plugin.process(data)

    debug_info["output_rows"] = len(processed_data)
    debug_info["output_columns"] = len(processed_data.columns)

    include_date = config['force_date'] or not (config.get('method') in ['select_single', 'select_multi'])

    print("Processing complete. Writing output...")
    write_csv(config['output_file'], processed_data, include_date=include_date, headers=config['headers'])
    print(f"Output written to {config['output_file']}")

    config_str, config_filename = save_config(config)
    print(f"Configuration saved to {config_filename}")

    execution_time = time.time() - start_time
    debug_info["execution_time"] = execution_time

    if 'debug_file' not in config or not config['debug_file']:
        config['debug_file'] = DEFAULT_VALUES['debug_file']

    plugin.add_debug_info(debug_info)
    save_debug_info(debug_info, config['debug_file'])

    print(f"Debug info saved to {config['debug_file']}")
    print(f"Execution time: {execution_time} seconds")

    if config['remote_save_config']:
        if save_remote_config(config_str, config['remote_save_config'], config['remote_username'], config['remote_password']):
            print(f"Configuration successfully saved to remote URL {config['remote_save_config']}")
        else:
            print(f"Failed to save configuration to remote URL {config['remote_save_config']}")

    if config['remote_log']:
        if log_remote_info(config_str, debug_info, config['remote_log'], config['remote_username'], config['remote_password']):
            print(f"Debug information successfully logged to remote URL {config['remote_log']}")
        else:
            print(f"Failed to log debug information to remote URL {config['remote_log']}")

if __name__ == '__main__':
    main()
