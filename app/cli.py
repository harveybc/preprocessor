import argparse
from plugin_loader import load_plugin

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessor for CSV data.")
    parser.add_argument('csv_file', type=str, nargs='?', help='Path to the input CSV file.')
    parser.add_argument('-sc', '--save_config', type=str, help='Path to save the configuration file.')
    parser.add_argument('-lc', '--load_config', type=str, help='Path to load the configuration file.')
    parser.add_argument('-p', '--plugin', type=str, help='Plugin to use for processing the data.')
    parser.add_argument('-o', '--output_file', type=str, help='Path to the output CSV file.')
    parser.add_argument('-rl', '--remote_log', type=str, help='Remote logging URL.')
    parser.add_argument('-rc', '--remote_save_config', type=str, help='Remote save configuration URL.')
    parser.add_argument('--remote_username', type=str, default='test', help='Remote username for authentication.')
    parser.add_argument('--remote_password', type=str, default='pass', help='Remote password for authentication.')
    parser.add_argument('-qm', '--quiet_mode', action='store_true', help='Enable quiet mode (no console output).')
    parser.add_argument('--force_date', action='store_true', help='Force inclusion of the date column in the output.')
    parser.add_argument('--headers', action='store_true', help='Indicate if the input CSV has headers.')

    args, unknown = parser.parse_known_args()
    
    if args.plugin:
        plugin, required_params = load_plugin(args.plugin)
        for param in required_params:
            parser.add_argument(f'--{param}', type=str, help=f'{param} for the plugin {args.plugin}')

    return parser.parse_args()
