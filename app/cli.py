import argparse

def parse_args():
    """
    Parses command line arguments provided to the preprocessor application.

    Returns:
        argparse.Namespace: The namespace containing the arguments and their values.
    """
    parser = argparse.ArgumentParser(description="Preprocessor: A tool for preprocessing CSV data with support for dynamic plugins.")

    # Required positional argument for the input CSV file
    parser.add_argument('csv_file', type=str, help='Path to the CSV file to process.')

    # Optional arguments for saving and loading preprocessing configurations
    parser.add_argument('-sc', '--save_config', type=str, help='Filename to save the preprocessing configuration.')
    parser.add_argument('-lc', '--load_config', type=str, help='Filename to load preprocessing configuration from.')

    # Optional arguments for selecting plugins
    parser.add_argument('-p', '--plugin', type=str, default='default_plugin', help='Name of the preprocessing plugin to use.')
    parser.add_argument('--method', type=str, help='Method to use in the plugin (e.g., z-score, min-max).')
    parser.add_argument('--range', type=float, nargs=2, help='Range for min-max normalization (e.g., 0 1 or -1 1).')

    # Optional argument for remote logging, monitoring, and storage of results
    parser.add_argument('-rl', '--remote_log', type=str, help='URL of a remote data-logger API endpoint.')

    # Optional argument for downloading and executing a remote JSON configuration
    parser.add_argument('-rc', '--remote_config', type=str, help='URL of a remote JSON configuration file to download and execute.')

    # Optional argument for quiet mode
    parser.add_argument('-qm', '--quiet_mode', action='store_true', help='Run in quiet mode without printing to console.')

    return parser.parse_args()

if __name__ == '__main__':
    # Parse the command line arguments
    args = parse_args()

    print(args)
