import subprocess
import json

def test_normalization_plugin():
    # Define the command
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv'
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
    assert config['output_file'] == 'output.csv'
    assert config['plugin'] == 'default_plugin'
    assert 'method' not in config  # method is not set, so it should not be in the config

    # Assertions for the debug file
    expected_debug_keys = {"execution_time", "input_rows", "output_rows", "input_columns", "output_columns", "min_val", "max_val"}
    assert set(debug_info.keys()) == expected_debug_keys
