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
    parser.add_argument('-p', '--plugin', type=str, help='Name of the plugin to use for processing.')
    parser.add_argument('--method', type=str, help='Method to use for the plugin.')

    # Optional arguments for normalization
    parser.add_argument('--norm_method', type=str, help='Normalization method to apply.')
    parser.add_argument('--range', type=float, nargs=2, metavar=('MIN', 'MAX'), help='Range for normalization.')

    # Optional arguments for output file
    parser.add_argument('-o', '--output_file', type=str, help='Path to save the output CSV file.')

    # Optional arguments for remote logging and configuration
    parser.add_argument('-rl', '--remote_log', type=str, help='URL for remote logging.')
    parser.add_argument('-rc', '--remote_config', type=str, help='URL to fetch remote configuration.')

    # Optional argument for quiet mode
    parser.add_argument('-qm', '--quiet_mode', action='store_true', help='Run in quiet mode without printing to the console.')

    # Plugin-specific arguments
    parser.add_argument('--window_size', type=int, help='Window size for moving average.')
    parser.add_argument('--ema_alpha', type=float, help='Alpha for exponential moving average.')
    parser.add_argument('--remove_rows', nargs='+', help='Rows to remove from the dataset.')
    parser.add_argument('--remove_columns', nargs='+', help='Columns to remove from the dataset.')
    parser.add_argument('--max_lag', type=int, help='Maximum lag for time series analysis.')
    parser.add_argument('--significance_level', type=float, help='Significance level for statistical tests.')
    parser.add_argument('--alpha', type=float, help='Alpha value for statistical tests.')
    parser.add_argument('--l1_ratio', type=float, help='L1 ratio for regularization.')
    parser.add_argument('--model_type', type=str, help='Type of model to use for prediction.')
    parser.add_argument('--timesteps', type=int, help='Number of timesteps for time series prediction.')
    parser.add_argument('--features', type=str, nargs='+', help='List of features to include in the dataset.')

    # Cleaner plugin arguments
    parser.add_argument('--clean_method', type=str, help='Method to use for cleaning (continuity or outlier).')
    parser.add_argument('--period', type=int, help='Period in minutes for the continuity method.')
    parser.add_argument('--outlier_threshold', type=float, help='Threshold for detecting outliers.')
    parser.add_argument('--solve_missing', action='store_true', help='Solve missing values in the data.')
    parser.add_argument('--delete_outliers', action='store_true', help='Delete outliers from the data.')
    parser.add_argument('--interpolate_outliers', action='store_true', help='Interpolate outliers in the data.')
    parser.add_argument('--delete_nan', action='store_true', help='Delete rows with NaN values.')
    parser.add_argument('--interpolate_nan', action='store_true', help='Interpolate NaN values in the data.')

    # Feature selector plugin arguments
    parser.add_argument('--single', type=int, help='Single column to select.')
    parser.add_argument('--multi', type=int, nargs='+', help='Multiple columns to select.')

    # Argument for handling headers in CSV files
    parser.add_argument('--headers', action='store_true', help='Treat the first row of the CSV file as headers.')

    # Argument for forcing the date column in the output
    parser.add_argument('--force_date', action='store_true', help='Force inclusion of the date column in the output CSV file.')

    return parser.parse_args()
