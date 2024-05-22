import sys
import os
import json
import requests
import pkg_resources

print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

# Ensure the current directory is in the PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

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

def load_plugin(plugin_name):
    """
    Load a plugin based on the name specified.
    """
    try:
        entry_point = next(pkg_resources.iter_entry_points('preprocessor.plugins', plugin_name))
        return entry_point.load()
    except StopIteration:
        print(f"Plugin {plugin_name} not found.", file=sys.stderr)
        return DefaultPlugin()

def load_remote_config(remote_config_url):
    """
    Load configuration from a remote URL.
    """
    try:
        response = requests.get(remote_config_url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Failed to load remote configuration: {e}", file=sys.stderr)
        return None

def main():
    # Parse command line arguments
    args = parse_args()

    # Attempt to load remote configuration if provided
    remote_config = load_remote_config(args.remote_config) if args.remote_config else None

    # Use remote configuration if available, otherwise fallback to CLI arguments
    config = remote_config if remote_config else {
        'csv_file': args.csv_file,
        'output_file': args.output_file if args.output_file else CSV_OUTPUT_PATH,
        'plugin_name': args.plugin if args.plugin else DEFAULT_PLUGIN,
        'remote_log': args.remote_log if args.remote_log else REMOTE_LOG_URL,
        'method': args.method if args.method else DEFAULT_NORMALIZATION_METHOD,
        'range': tuple(args.range) if args.range else DEFAULT_NORMALIZATION_RANGE,
        'save_config': args.save_config if args.save_config else CONFIG_SAVE_PATH,
        'load_config': args.load_config if args.load_config else CONFIG_LOAD_PATH,
        'quiet_mode': args.quiet_mode if args.quiet_mode else DEFAULT_QUIET_MODE
    }

    # Load the CSV data
    data = load_csv(config['csv_file'])

    # Load and apply the plugin
    plugin = load_plugin(config['plugin_name'])
    processed_data = plugin.process(data, method=config.get('method'), range=config.get('range'))

    # Save the processed data to output CSV
    write_csv(config['output_file'], processed_data)

    # Save configuration if save_config path is provided
    if config['save_config']:
        with open(config['save_config'], 'w') as f:
            json.dump(config, f)

    # Log processing completion
    if config['remote_log']:
        try:
            response = requests.post(config['remote_log'], json={'message': 'Processing complete', 'output_file': config['output_file']})
            if not config['quiet_mode']:
                print(f"Remote log response: {response.text}")
        except requests.RequestException as e:
            if not config['quiet_mode']:
                print(f"Failed to send remote log: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
