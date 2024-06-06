import os
import subprocess
import json

def test_normalization_plugin():
    # Define the command to run the normalization plugin with default parameters
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--output_file', 'tests/data/normalized_output.csv',
        '--save_config', 'tests/data/config_out.json',
        '--debug_file', 'tests/data/debug_out.json'
    ]

    # Run the command
    subprocess.run(command, check=True)

    # Load the generated config file
    with open('tests/data/config_out.json', 'r') as f:
        config = json.load(f)

    # Expected config file content
    expected_config = {
        "csv_file": "tests/data/EURUSD_5m_2006_2007.csv"
    }

    # Assert that the config file matches the expected content
    assert config == expected_config, "Config file content does not match the expected content."

    # Load the generated debug file
    with open('tests/data/debug_out.json', 'r') as f:
        debug_info = json.load(f)

    # Expected debug information
    expected_debug_info = {
        "input_rows": 73841,
        "output_rows": 73841,
        "input_columns": 1,
        "output_columns": 1,
        "min_val": -1,  # Replace with the actual min_val from the dataset
        "max_val": 100  # Replace with the actual max_val from the dataset
    }

    # Assert that the debug file matches the expected content
    for key, value in expected_debug_info.items():
        assert debug_info.get(key) == value, f"Debug info '{key}' does not match the expected value."

if __name__ == '__main__':
    test_normalization_plugin()
