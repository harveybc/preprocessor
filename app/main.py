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
from app.config_handler import load_config, save_config, load_remote_config
from app.plugin_loader import load_plugin
from app.data_processor import process_data

def main():
    args = parse_args()
    
    # Debugging: Print parsed arguments
    print("Parsed arguments:", args)

    config = load_config(args)

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

    plugin_class = load_plugin(config['plugin_name'])
    if plugin_class is None:
        print(f"Error: The plugin {config['plugin_name']} could not be loaded.")
        return

    plugin = plugin_class()
    processed_data = process_data(config, plugin)

    # Save configuration to file
    config_filename = save_config(config)
    
    # Debugging: Print saved configuration
    if not config['quiet_mode']:
        print(f"Configuration saved to {os.path.basename(config_filename)}")

if __name__ == '__main__':
    main()
