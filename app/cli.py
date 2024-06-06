import argparse
from plugin_loader import get_plugin_params

def setup_arg_parser():
    """
    Set up the argument parser with common parameters and dynamically add plugin-specific parameters.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description='Preprocessor CLI')
    
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('--save_config', type=str, help='Path to save the configuration')
    parser.add_argument('--load_config', type=str, help='Path to load the configuration')
    parser.add_argument('--plugin', type=str, help='Name of the plugin to use')
    parser.add_argument('--output_file', type=str, help='Path to the output CSV file')
    parser.add_argument('--remote_log', type=str, help='URL for remote logging')
    parser.add_argument('--remote_save_config', type=str, help='URL to save the configuration remotely')
    parser.add_argument('--remote_load_config', type=str, help='URL to load the configuration remotely')
    parser.add_argument('--remote_username', type=str, help='Username for remote operations')
    parser.add_argument('--remote_password', type=str, help='Password for remote operations')
    parser.add_argument('--quiet_mode', action='store_true', help='Run in quiet mode')
    parser.add_argument('--force_date', action='store_true', help='Force date inclusion')
    parser.add_argument('--headers', action='store_true', help='Indicate if CSV has headers')
    parser.add_argument('--debug_file', type=str, help='Path to save debug information')
    
    return parser

def parse_args():
    """
    Parse the command line arguments, including dynamically added plugin-specific parameters.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = setup_arg_parser()
    
    # First pass to get the plugin name
    args, unknown = parser.parse_known_args()

    # Dynamically add plugin-specific parameters if the plugin is specified
    if args.plugin:
        plugin_params = get_plugin_params(args.plugin)
        for param, default in plugin_params.items():
            parser.add_argument(f'--{param}', type=type(default), default=default, help=f'{param} for the plugin {args.plugin}')
    
    # Final parse to include dynamically added arguments
    return parser.parse_args()
