import subprocess
import json

def test_normalization_plugin_z_score():
    # Define the initial command to normalize with the "z-score" method
    initial_command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--method', 'z-score'
    ]

    # Run the initial command
    subprocess.run(initial_command, check=True)

    # Load the generated config and debug files
    with open('config_out.json', 'r') as f:
        config = json.load(f)
    with open('debug_out.json', 'r') as f:
        debug_info = json.load(f)

    # Assertions for the config file
    expected_config_keys = {"csv_file", "output_file", "plugin", "method"}
    assert set(config.keys()) == expected_config_keys, f"Unexpected keys in config: {set(config.keys()) - expected_config_keys}"
    assert config['csv_file'] == 'tests/data/EURUSD_5m_2006_2007.csv'
    assert config['output_file'] == 'output.csv'
    assert config['plugin'] == 'default_plugin'
    assert config['method'] == 'z-score'

    # Assertions for the debug file
    expected_debug_keys = {"execution_time", "input_rows", "output_rows", "input_columns", "output_columns", "mean", "std"}
    assert set(debug_info.keys()) == expected_debug_keys
