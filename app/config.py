# Configuration file for the preprocessor application

# Path configurations
CSV_INPUT_PATH = './csv_input.csv'  # Default path for CSV input if not specified
CSV_OUTPUT_PATH = './csv_output.csv'  # Default path for CSV output if not specified
CONFIG_SAVE_PATH = './config_out.json'  # Default file to save preprocessing configurations
CONFIG_LOAD_PATH = './config_in.json'  # Default file to load configurations from

# Default plugin configuration
DEFAULT_PLUGIN = 'default_plugin'  # Default preprocessing plugin name

# Remote logging and configuration
REMOTE_LOG_URL = 'http://remote-log-server/api/logs'  # Default URL for remote logging
REMOTE_CONFIG_URL = 'http://remote-config-server/api/config'  # Default URL for remote configuration

# Plugin configurations
PLUGIN_DIRECTORY = 'app/plugins/'  # Directory containing all plugins

# Quiet mode
DEFAULT_QUIET_MODE = False  # Default setting for quiet mode
