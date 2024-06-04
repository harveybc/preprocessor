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
from app.config_handler import load_config, save_config, save_debug_info

def load_plugin(plugin_name):
    try:
        entry_point = next(pkg_resources.iter_entry_points('preprocessor.plugins', plugin_name))
        return entry_point.load()
    except StopIteration:
        print(f"Plugin {plugin_name} not found.", file=sys.stderr)
        return None

def save_remote_config(config, url, username, password):
    try:
        response = requests.post(
            url,
            auth=(username, password),
            data={'json_config': json.dumps(config)}
        )
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Failed to save remote configuration: {e}", file=sys.stderr)
        return False

def main():
    args = parse_args()

    debug_info = {
        "parsed_arguments": str(args),
        "configuration": "",
        "loaded_data": "",
        "processed_data": ""
    }

    config = load_config(args)
    debug_info["configuration"] = str(config)

    if not config.get('csv_file'):
        print("Error: No CSV file specified.", file=sys.stderr)
        return

    data = load_csv(config['csv_file'], headers=config['headers'])
    debug_info["loaded_data"] = str(data.head())

    plugin_class = load_plugin(config['plugin_name'])
    if plugin_class is None:
        print(f"Error: The plugin {config['plugin_name']} could not be loaded.")
        return

    plugin = plugin_class()
    processed_data = plugin.process(
        data,
        method=config['method'],
        save_params=config['save_config'],
        load_params=config['load_config'],
        single=config['single'],
        multi=config['multi']
    )
    debug_info["processed_data"] = str(processed_data.head())

    include_date = config['force_date'] or not (config['method'] in ['select_single', 'select_multi'])

    if not config['quiet_mode']:
        print("Processing complete. Writing output...")

    write_csv(config['output_file'], processed_data, include_date=include_date, headers=config['headers'])

    if not config['quiet_mode']:
        print(f"Output written to {config['output_file']}")
        print(f"Configuration saved to {os.path.basename(config['save_config'])}")

    config_filename = save_config(config)
    save_debug_info(debug_info, args.debug_file)

    if not config['quiet_mode']:
        print(f"Debug info saved to {args.debug_file}")

    if args.remote_save_config:
        filtered_config = {k: v for k, v in config.items() if v is not None and v != default_values.get(k)}
        if save_remote_config(filtered_config, args.remote_save_config, args.remote_username, args.remote_password):
            print(f"Configuration successfully saved to remote URL {args.remote_save_config}")
        else:
            print(f"Failed to save configuration to remote URL {args.remote_save_config}")

if __name__ == '__main__':
    main()
