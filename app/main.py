import sys
import time
import json
from plugin_loader import load_plugin
from app.cli import parse_args
from app.config_handler import load_config, save_config, save_debug_info
from app.data_handler import load_csv, write_csv
from app.plugin_handler import set_plugin_params

def main():
    args = parse_args()

    debug_info = {
        "execution_time": "",
        "input_rows": 0,
        "output_rows": 0,
        "input_columns": 0,
        "output_columns": 0
    }

    start_time = time.time()

    config = load_config(args)

    if not config.get('csv_file'):
        print("Error: No CSV file specified.", file=sys.stderr)
        return

    data = load_csv(config['csv_file'], headers=config['headers'])
    debug_info["input_rows"] = len(data)
    debug_info["input_columns"] = len(data.columns)

    plugin, required_params = load_plugin(config['plugin_name'])
    if plugin is None:
        print(f"Error: The plugin {config['plugin_name']} could not be loaded.")
        return

    set_plugin_params(plugin, config, required_params)

    processed_data = plugin.process(data)

    debug_info["output_rows"] = len(processed_data)
    debug_info["output_columns"] = len(processed_data.columns)

    include_date = config['force_date'] or not (config.get('method') in ['select_single', 'select_multi'])

    if not config['quiet_mode']:
        print("Processing complete. Writing output...")

    write_csv(config['output_file'], processed_data, include_date=include_date, headers=config['headers'])

    if not config['quiet_mode']:
        print(f"Output written to {config['output_file']}")
        print(f"Configuration saved to {config['save_config']}")

    config_str, config_filename = save_config(config)

    execution_time = time.time() - start_time
    debug_info["execution_time"] = execution_time

    save_debug_info(debug_info, args.debug_file)

    if not config['quiet_mode']:
        print(f"Debug info saved to {args.debug_file}")
        print(f"Execution time: {execution_time} seconds")

    if config.get('remote_save_config'):
        if save_remote_config(config_str, config['remote_save_config'], config['remote_username'], config['remote_password']):
            print(f"Configuration successfully saved to remote URL {config['remote_save_config']}")
        else:
            print(f"Failed to save configuration to remote URL {config['remote_save_config']}")

    if config.get('remote_log'):
        if log_remote_info(config_str, debug_info, config['remote_log'], config['remote_username'], config['remote_password']):
            print(f"Debug information successfully logged to remote URL {config['remote_log']}")
        else:
            print(f"Failed to log debug information to remote URL {config['remote_log']}")

if __name__ == '__main__':
    main()
