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

    # Optional arguments for the unbiaser plugin
    parser.add_argument('--window_size', type=int, help='Window size for the unbiaser plugin.')
    parser.add_argument('--ema_alpha', type=float, help='Alpha value for EMA in the unbiaser plugin.')

    # Optional arguments for the trimmer plugin
    parser.add_argument('--remove_rows', type=int, nargs='+', help='Rows to remove in the trimmer plugin.')
    parser.add_argument('--remove_columns', type=int, nargs='+', help='Columns to remove in the trimmer plugin.')

    # Optional arguments for the feature_selector_pre plugin
    parser.add_argument('--method', type=str, default='granger', help='Method for feature selection (acf, pacf, granger).')
    parser.add_argument('--max_lag', type=int, help='Max lag for Granger causality in the feature_selector_pre plugin.')
    parser.add_argument('--significance_level', type=float, help='Significance level for statistical tests in the feature_selector_pre plugin.')
    parser.add_argument('--select_single', type=int, help='Index of the single column to select in the feature_selector_pre plugin.')
    parser.add_argument('--select_multi', type=int, nargs='+', help='Indices of multiple columns to select in the feature_selector_pre plugin.')

    # Optional arguments for the feature_selector_post plugin
    parser.add_argument('--alpha', type=float, help='Alpha value for Lasso and Elastic Net in the feature_selector_post plugin.')
    parser.add_argument('--l1_ratio', type=float, help='L1 ratio for Elastic Net in the feature_selector_post plugin.')
    parser.add_argument('--model_type', type=str, help='Model type for cross-validation feature selection (lstm or cnn) in the feature_selector_post plugin.')
    parser.add_argument('--timesteps', type=int, help='Timesteps for LSTM/CNN in the feature_selector_post plugin.')
    parser.add_argument('--features', type=int, help='Number of features for LSTM/CNN in the feature_selector_post plugin.')

    # Optional arguments for cleaner plugin
    parser.add_argument('--period', type=int, help='Expected period for continuity checking in minutes.')
    parser.add_argument('--outlier_threshold', type=float, help='Threshold for outlier detection.')
    parser.add_argument('--solve_missing', action='store_true', help='Solve missing values by interpolation.')
    parser.add_argument('--delete_outliers', action='store_true', help='Delete outliers from the data.')
    parser.add_argument('--interpolate_outliers', action='store_true', help='Interpolate outliers in the data.')
    parser.add_argument('--delete_nan', action='store_true', help='Delete rows with NaN values.')
    parser.add_argument('--interpolate_nan', action='store_true', help='Interpolate NaN values.')

    # Optional argument for the output file
    parser.add_argument('-o', '--output_file', type=str, help='Output CSV file path.')

    # Optional argument for remote logging, monitoring, and storage of results
    parser.add_argument('-rl', '--remote_log', type=str, help='URL of a remote data-logger API endpoint.')

    # Optional argument for downloading and executing a remote JSON configuration
    parser.add_argument('-rc', '--remote_config', type=str, help='URL of a remote JSON configuration file to download and execute.')

    # Optional argument for quiet mode
    parser.add_argument('-qm', '--quiet_mode', action='store_true', help='Run in quiet mode without printing to console.')

    # Optional argument for headers
    parser.add_argument('--headers', action='store_true', help='Indicate if the CSV file has headers.')

    return parser.parse_args()

if __name__ == '__main__':
    # Parse the command line arguments
    args = parse_args()

    print(args)
