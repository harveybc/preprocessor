import sys
import os
import json
import time
from app.cli import parse_args, setup_arg_parser
from app.config_handler import load_config, save_config, save_debug_info, merge_config
from app.data_handler import load_csv, write_csv
from plugin_loader import load_plugin, add_plugin_params

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
    # Setup argument parser and parse arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Add plugin-specific arguments to parser
    if args.plugin:
        add_plugin_params(parser, args.plugin)
        args = parser.parse_args()  # Reparse arguments to include plugin-specific ones

    # Load and merge configuration
    config = load_config(args)
    config = merge_config(config, args)

    print(f"Initial loaded config: {config}")

    if not config.get('csv_file'):
        print("Error: No CSV file specified.", file=sys.stderr)
        return

    data = load_csv(config['csv_file'], headers=config['headers'])

    plugin, required_params = load_plugin(config['plugin_name'])
    if plugin is None:
        print(f"Error: The plugin {config['plugin_name']} could not be loaded.")
        return

    plugin_params = {param: config[param] for param in required_params if param in config}
    print(f"Setting plugin parameters: {plugin_params}")
    plugin.set_params(**plugin_params)

    debug_info = {
        "execution_time": "",
        "input_rows": len(data),
        "output_rows": 0,
        "input_columns": len(data.columns),
        "output_columns": 0
    }
    
    start_time = time.time()
    processed_data = plugin.process(data)
    debug_info["output_rows"] = len(processed_data)
    debug_info["output_columns"] = len(processed_data.columns)
    execution_time = time.time() - start_time
    debug_info["execution_time"] = execution_time
    debug_info.update(plugin.get_debug_info())

    include_date = config['force_date'] or not (config.get('method') in ['select_single', 'select_multi'])

    if not config['quiet_mode']:
        print("Processing complete. Writing output...")

    write_csv(config['output_file'], processed_data, include_date=include_date, headers=config['headers'])

    if not config['quiet_mode']:
        print(f"Output written to {config['output_file']}")
        print(f"Configuration saved to {os.path.basename(config['save_config'])}")

    config_str, config_filename = save_config(config)
    save_debug_info(debug_info, args.debug_file)

    if not config['quiet_mode']:
        print(f"Debug info saved to {args.debug_file}")
        print(f"Execution time: {execution_time} seconds")

    if args.remote_save_config:
        if save_remote_config(config_str, args.remote_save_config, args.remote_username, args.remote_password):
            print(f"Configuration successfully saved to remote URL {args.remote_save_config}")
        else:
            print(f"Failed to save configuration to remote URL {args.remote_save_config}")

    if args.remote_log:
        if log_remote_info(config_str, debug_info, args.remote_log, args.remote_username, args.remote_password):
            print(f"Debug information successfully logged to remote URL {args.remote_log}")
        else:
            print(f"Failed to log debug information to remote URL {args.remote_log}")

if __name__ == '__main__':
    main()
