# config.py

DEFAULT_VALUES = {
    'input_file': 'tests/data/indicators_output.csv',  # Default path for the CSV file
    'output_file': './output.csv',  # Default output file for processed data
    'load_config': None,  # Path to load configuration file (if provided)
    'save_config': './output_config.json',  # Path to save the configuration file
    'remote_load_config': None,  # URL for remote configuration loading
    'remote_save_config': None,  # URL for remote configuration saving
    'remote_log': None,  # URL for remote logging
    'remote_username': None,  # Username for remote logging/authentication
    'remote_password': None,  # Password for remote logging/authentication
    'plugin': 'default_plugin',  # Default plugin to use for feature extraction
    'headers': True,  # Whether the CSV file has headers (True by default)
    'force_date': False,  # Force inclusion of date column in the output
    'debug_file': './debug_out.json',  # Path to save debug information
    'quiet_mode': False,  # Suppress all output except for errors
    'only_low_CV': False  # Process only low CV columns (False by default)
}

