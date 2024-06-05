import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessor CLI')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file to process')
    parser.add_argument('--save_config', type=str, help='Path to save the configuration file', default=None)
    parser.add_argument('--load_config', type=str, help='Path to load the configuration file', default=None)
    parser.add_argument('--plugin', type=str, help='Name of the plugin to use', default='default_plugin')
    parser.add_argument('--output_file', type=str, help='Path to save the output CSV file', default=None)
    parser.add_argument('--remote_log', type=str, help='URL to log remote information', default=None)
    parser.add_argument('--remote_save_config', type=str, help='URL to save the remote configuration', default=None)
    parser.add_argument('--remote_load_config', type=str, help='URL to load the remote configuration', default=None)
    parser.add_argument('--remote_username', type=str, help='Username for remote configuration', default='test')
    parser.add_argument('--remote_password', type=str, help='Password for remote configuration', default='pass')
    parser.add_argument('--quiet_mode', action='store_true', help='Enable quiet mode')
    parser.add_argument('--force_date', action='store_true', help='Force date inclusion in output')
    parser.add_argument('--headers', action='store_true', help='CSV file includes headers')
    parser.add_argument('--debug_file', type=str, help='Path to save the debug information', default='debug_out.json')
    # Add other plugin-specific parameters as needed
    parser.add_argument('--method', type=str, help='Method for the plugin', default=None)
    parser.add_argument('--window_size', type=int, help='Window size for the plugin', default=None)
    parser.add_argument('--ema_alpha', type=float, help='EMA alpha for the plugin', default=None)
    args = parser.parse_args()
    return args
