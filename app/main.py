"""Main entry point for the Preprocessor System

This module provides the main entry point for command-line execution
of the preprocessor system.
"""

import sys
from app.cli import main as cli_main

def main():
    """Main entry point"""
    return cli_main()

if __name__ == '__main__':
    sys.exit(main())

    file_config = {}
    # remote config file load
    if args.remote_load_config:
        file_config = remote_load_config(args.remote_load_config, args.username, args.password)
        print(f"Loaded remote config: {file_config}")

    # local config file load
    if args.load_config:
        file_config = load_config(args.load_config)
        print(f"Loaded local config: {file_config}")

    print("Merging configuration with CLI arguments and unknown args...")
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(config, {}, file_config, cli_args, unknown_args_dict)

    # Load data using data_handler
    print(f"Loading data from {config['input_file']}...")
    data = load_csv(config['input_file'])

    # Plugin loading and processing
    plugin_name = config['plugin']
    print(f"Loading plugin: {plugin_name}")
    plugin_class, _ = load_plugin('preprocessor.plugins', plugin_name)
    plugin = plugin_class()
    # override plugin parames with already configured params
    plugin.set_params(**config)
    plugin_params = getattr(plugin, 'plugin_params', {})
    
    print("Merging configuration with plugin_specific arguments...")
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(config, plugin_params, file_config, cli_args, unknown_args_dict)

    print("Running the feature engineering pipeline...")
    run_preprocessor_pipeline(config, plugin)

    # Save local configuration if specified
    if 'save_config' in config and config['save_config']:
        save_config(config, config['save_config'])
        print(f"Configuration saved to {config['save_config']}.")

    # Save configuration remotely if specified
    if 'remote_save_config' in config and config['remote_save_config']:
        print(f"Remote saving configuration to {config['remote_save_config']}")
        remote_save_config(config, config['remote_save_config'], config['username'], config['password'])
        print(f"Remote configuration saved.")

    # Log data remotely if specified
    if 'remote_log' in config and config['remote_log']:
        print(f"Logging data remotely to {config['remote_log']}")
        remote_log(config, config['remote_log'], config['username'], config['password'])
        print(f"Data logged remotely.")

if __name__ == "__main__":
    main()
