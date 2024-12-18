import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Feature-extractor: A tool for encoding and decoding CSV data with support for dynamic plugins.')
    parser.add_argument('--input_file', help='Path to the CSV file to be processed')
    parser.add_argument('--load_config', help='Path to the configuration file to load')
    parser.add_argument('--save_config', help='Path to save the configuration file')
    parser.add_argument('--remote_load_config', help='URL to load remote configuration')
    parser.add_argument('--remote_save_config', help='URL to save remote configuration')
    parser.add_argument('--remote_log', help='URL for remote logging')
    parser.add_argument('--remote_username', help='Username for remote logging')
    parser.add_argument('--remote_password', help='Password for remote logging')
    parser.add_argument('--plugin', help='Encoder plugin to use')
    parser.add_argument('--output_file', help='Path to save the output data')
    parser.add_argument('--headers', action='store_true', help='Indicate if the CSV file has headers')
    parser.add_argument('--force_date', action='store_true', help='Force inclusion of the date column in the output')
    parser.add_argument('--debug_file', help='Path to save debug information')
    # quiet mode
    parser.add_argument('--quiet_mode', action='store_true', help='Suppress all output except for errors')
    parser.add_argument('--only_low_CV', action='store_true', help='Suppress all output except for errors')

    
    args, unknown = parser.parse_known_args()
    return args, unknown
