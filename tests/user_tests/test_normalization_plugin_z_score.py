# tests/user_tests/test_normalization_plugin_z_score.py

import os
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
    assert config['csv_file'] == 'tests/data/EURUSD_5m_2006_2007.csv'
    assert config['output_file'] == 'output.csv'
    assert config['plugin'] == 'default_plugin'
    assert config['method'] == 'z-score'

    # Assertions for the debug file
    assert debug_info['input_rows'] == 73841
    assert debug_info['output_rows'] == 73841
    assert debug_info['input_columns'] == 1
    assert debug_info['output_columns'] == 1
    assert 'execution_time' in debug_info
    assert debug_info['mean'] is not None
    assert debug_info['std'] is not None

if __name__ == '__main__':
    test_normalization_plugin_z_score()
