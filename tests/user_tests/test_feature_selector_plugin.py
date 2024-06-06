import subprocess
import json

def test_feature_selector_plugin_single():
    # Define the command to use the feature selector plugin with single selection
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--plugin', 'feature_selector',
        '--single', '3'
    ]

    # Run the command
    subprocess.run(command, check=True)

    # Load the generated config and debug files
    with open('config_out.json', 'r') as f:
        config = json.load(f)
    with open('debug_out.json', 'r') as f:
        debug_info = json.load(f)

    # Assertions for the config file
    assert config['csv_file'] == 'tests/data/EURUSD_5m_2006_2007.csv'
    assert config['plugin'] == 'feature_selector'
    assert int(config['single']) == 3  # Ensure single is treated as integer
    expected_config_keys = {"csv_file", "plugin", "single"}
    assert set(config.keys()) == expected_config_keys, f"Unexpected keys in config: {set(config.keys()) - expected_config_keys}"

    # Assertions for the debug file
    expected_debug_keys = {"execution_time", "input_rows", "output_rows", "input_columns", "output_columns", "method", "max_lag", "significance_level", "single", "multi"}
    assert set(debug_info.keys()) == expected_debug_keys, f"Unexpected keys in debug info: {set(debug_info.keys()) - expected_debug_keys}"

if __name__ == '__main__':
    test_feature_selector_plugin_single()
