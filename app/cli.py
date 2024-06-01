import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessor: A tool for preprocessing CSV data with support for dynamic plugins.")
    
    parser.add_argument('csv_file', type=str, help='Path to the CSV file to process.')

    parser.add_argument('-sc', '--save_config', type=str, help='Filename to save the preprocessing configuration.')
    parser.add_argument('-lc', '--load_config', type=str, help='Filename to load preprocessing configuration from.')
    
    parser.add_argument('-p', '--plugin', type=str, default='default_plugin', help='Name of the preprocessing plugin to use.')
    
    parser.add_argument('--norm_method', type=str, default='z-score', help='Normalization method to use (z-score or min-max).')
    parser.add_argument('--range', type=float, nargs=2, help='Normalization range (min max).')
    
    parser.add_argument('-o', '--output_file', type=str, help='Output CSV file path.')
    parser.add_argument('-rl', '--remote_log', type=str, help='URL of a remote data-logger API endpoint.')
    parser.add_argument('-rc', '--remote_config', type=str, help='URL of a remote JSON configuration file to download and execute.')
    parser.add_argument('-qm', '--quiet_mode', action='store_true', help='Run in quiet mode without printing to console.')

    parser.add_argument('--window_size', type=int, help='Window size for the unbiaser plugin.')
    parser.add_argument('--ema_alpha', type=float, help='Alpha value for EMA in the unbiaser plugin.')
    
    parser.add_argument('--remove_rows', type=int, nargs='+', help='Rows to remove in the trimmer plugin.')
    parser.add_argument('--remove_columns', type=int, nargs='+', help='Columns to remove in the trimmer plugin.')
    
    parser.add_argument('--max_lag', type=int, help='Max lag for Granger causality in the feature_selector_pre plugin.')
    parser.add_argument('--significance_level', type=float, help='Significance level for statistical tests in the feature_selector_pre plugin.')
    
    parser.add_argument('--alpha', type=float, help='Alpha value for Lasso and Elastic Net in the feature_selector_post plugin.')
    parser.add_argument('--l1_ratio', type=float, help='L1 ratio for Elastic Net in the feature_selector_post plugin.')
    parser.add_argument('--model_type', type=str, help='Model type for cross-validation feature selection (lstm or cnn) in the feature_selector_post plugin.')
    parser.add_argument('--timesteps', type=int, help='Timesteps for LSTM/CNN in the feature_selector_post plugin.')
    parser.add_argument('--features', type=int, help='Number of features for LSTM/CNN in the feature_selector_post plugin.')

    parser.add_argument('--clean_method', type=str, help='Method to use for cleaning (missing_values or outlier).')
    parser.add_argument('--period', type=str, help='Expected period for continuity checking.')
    parser.add_argument('--outlier_threshold', type=float, help='Threshold for outlier detection.')
    parser.add_argument('--solve_missing', action='store_true', help='Solve missing values by interpolation.')
    parser.add_argument('--delete_outliers', action='store_true', help='Delete rows with outliers.')
    parser.add_argument('--interpolate_outliers', action='store_true', help='Interpolate outliers with neighboring values.')
    parser.add_argument('--delete_nan', action='store_true', help='Delete rows with NaN values.')
    parser.add_argument('--interpolate_nan', action='store_true', help='Interpolate NaN values with neighboring values.')

    parser.add_argument('--method', type=str, help='Method to use for feature selection (select_single or select_multi).')
    parser.add_argument('--single', type=int, help='Select a single column.')
    parser.add_argument('--multi', type=int, nargs='+', help='Select multiple columns.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)
